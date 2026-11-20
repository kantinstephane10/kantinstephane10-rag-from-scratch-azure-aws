"""Project-scoped artifact ledger and mandatory verified agent-to-agent handoffs.

The ledger records evidence; it does not claim that AI-generated text is executed code.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from store import StoreError
from roles import ROLES

MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
ALLOWED_SUFFIXES = {'.txt', '.md', '.py', '.json', '.csv', '.pdf', '.ipynb', '.yaml', '.yml', '.toml', '.html', '.js', '.css'}


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def init_handoffs(store):
    with store.db() as db:
        db.executescript('''
            CREATE TABLE IF NOT EXISTS artifacts (
              id TEXT PRIMARY KEY, project_id TEXT NOT NULL, name TEXT NOT NULL,
              sha256 TEXT NOT NULL, size INTEGER NOT NULL, creator TEXT NOT NULL,
              created_at INTEGER NOT NULL);
            CREATE INDEX IF NOT EXISTS ix_artifacts_project ON artifacts(project_id,created_at);
            CREATE TABLE IF NOT EXISTS handoffs (
              id TEXT PRIMARY KEY, project_id TEXT NOT NULL, source_message_id TEXT NOT NULL,
              sender TEXT NOT NULL, recipient TEXT NOT NULL, manifest TEXT NOT NULL,
              digest TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'ready',
              created_at INTEGER NOT NULL, acknowledged_at INTEGER);
            CREATE INDEX IF NOT EXISTS ix_handoffs_project ON handoffs(project_id,created_at);
        ''')
        cols={r['name'] for r in db.execute('PRAGMA table_info(messages)')}
        if 'handoff_id' not in cols:
            db.execute('ALTER TABLE messages ADD COLUMN handoff_id TEXT')


def _artifact_path(store, project_id, aid, name):
    # All file path components are server-generated or validated basenames.
    return Path(store.filename).parent/'artifacts'/project_id/aid/name


def save_artifact(store, pid, name, raw, creator='human'):
    store.project(pid)
    if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', name) or name in {'.', '..'}:
        raise StoreError('Use a simple filename (letters, numbers, dot, underscore or dash).')
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise StoreError('Unsupported document type.')
    if creator != 'human' and creator not in ROLES:
        raise StoreError('Unknown artifact author.')
    if not isinstance(raw, bytes) or not 0 < len(raw) <= MAX_ARTIFACT_BYTES:
        raise StoreError('Artifact must be between 1 byte and 8 MiB.')
    aid=uuid4().hex; path=_artifact_path(store,pid,aid,name)
    path.parent.mkdir(parents=True,exist_ok=False)
    try:
        with open(path,'xb') as f:
            f.write(raw)
        with store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            state=db.execute('SELECT status FROM projects WHERE id=?',(pid,)).fetchone()
            if not state or state['status'] in ('approved','deleting'):
                raise StoreError('Project is no longer accepting uploaded documents.')
            db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)',
                       (aid,pid,name,_digest(raw),len(raw),creator,int(time.time())))
            store._event(db,pid,'artifact_uploaded',creator,{'id':aid,'name':name,'sha256':_digest(raw)})
    except Exception:
        path.unlink(missing_ok=True)
        try: path.parent.rmdir()
        except OSError: pass
        raise
    return {'id':aid,'name':name,'sha256':_digest(raw),'size':len(raw),'creator':creator}


def artifacts(store,pid):
    store.project(pid)
    with store.db() as db:
        return [dict(r) for r in db.execute('SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at,id',(pid,))]


def read_artifact(store,pid,aid):
    store.project(pid)
    with store.db() as db:
        row=db.execute('SELECT * FROM artifacts WHERE id=? AND project_id=?',(aid,pid)).fetchone()
    if not row: raise StoreError('Unknown project artifact.')
    path=_artifact_path(store,pid,aid,row['name'])
    try: raw=path.read_bytes()
    except (OSError,ValueError) as exc: raise StoreError('Artifact was removed or cannot be read.') from exc
    if len(raw)!=row['size'] or _digest(raw)!=row['sha256']:
        raise StoreError('Artifact integrity verification failed; handoff blocked.')
    return dict(row),raw


def _manifest(db,store,pid,source,sender,recipient):
    files=[]
    # Cap context metadata; large project collections must be retrieved by artifact ID.
    rows=db.execute('SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at DESC,id DESC LIMIT 40',(pid,)).fetchall()
    for row in rows:
        files.append({'id':row['id'],'name':row['name'],'sha256':row['sha256'],'size':row['size']})
    return {
        'from':sender,'to':recipient,'source_message_id':source['id'],
        'summary':source['reply'] or '',
        'files':files,
        'tests':[],  # No tests executed by the current planning/messaging engine.
        'test_status':'not_executed',
        'decisions':[],
        'open_issues':[],
        'limits':'Proposals are not code changes; no tests or deployment have been executed by this worker.',
    }


def record_handoff(db,store,pid,source,sender,recipient):
    if source['status']!='done' or source['recipient']!=sender:
        raise StoreError('A handoff requires a completed sender task.')
    m=_manifest(db,store,pid,source,sender,recipient)
    hid=uuid4().hex
    db.execute('''INSERT INTO handoffs(id,project_id,source_message_id,sender,recipient,manifest,digest,created_at)
                  VALUES(?,?,?,?,?,?,?,?)''',(hid,pid,source['id'],sender,recipient,
                         _canonical(m).decode('utf-8'),_digest(_canonical(m)),int(time.time())))
    store._event(db,pid,'handoff_created',sender,{'id':hid,'recipient':recipient,'files':len(m['files'])})
    return hid


def _verified(db,store,hid,pid=None):
    row=db.execute('SELECT * FROM handoffs WHERE id=?',(hid,)).fetchone()
    if not row or (pid is not None and row['project_id']!=pid):
        raise StoreError('Required handoff missing or belongs to another project.')
    raw=row['manifest'].encode('utf-8')
    if _digest(raw)!=row['digest']:
        raise StoreError('Handoff manifest integrity mismatch.')
    manifest=json.loads(raw)
    for f in manifest['files']:
        ar=db.execute('SELECT * FROM artifacts WHERE id=? AND project_id=?',(f['id'],row['project_id'])).fetchone()
        if not ar or ar['sha256']!=f['sha256'] or ar['size']!=f['size']:
            raise StoreError('Handoff artifact metadata changed.')
        path=_artifact_path(store,row['project_id'],ar['id'],ar['name'])
        try: data=path.read_bytes()
        except OSError as exc: raise StoreError('Handoff artifact is missing.') from exc
        if len(data)!=ar['size'] or _digest(data)!=ar['sha256']:
            raise StoreError('Handoff artifact was modified; downstream task blocked.')
    return dict(row),manifest


def inspect_handoff(store,hid,pid=None):
    with store.db() as db:
        row,m=_verified(db,store,hid,pid)
    row['manifest']=m
    return row


def list_handoffs(store,pid):
    store.project(pid)
    with store.db() as db:
        rows=[dict(r) for r in db.execute('SELECT * FROM handoffs WHERE project_id=? ORDER BY created_at,id',(pid,))]
    for row in rows:
        row['manifest']=json.loads(row['manifest'])
        try: inspect_handoff(store,row['id'],pid); row['integrity']='verified'
        except StoreError: row['integrity']='invalid'
    return rows


def acknowledge_handoff(store,hid,pid):
    with store.db() as db:
        row,_=_verified(db,store,hid,pid)
        db.execute("UPDATE handoffs SET status='acknowledged', acknowledged_at=COALESCE(acknowledged_at,?) WHERE id=?",
                   (int(time.time()),hid))
        store._event(db,pid,'handoff_acknowledged',row['recipient'],{'id':hid})
