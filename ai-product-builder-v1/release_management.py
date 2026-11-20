"""Approval-gated, atomic local release staging; no unapproved external publishing."""
import hashlib
import json
import os
import shutil
import threading
import time
from pathlib import Path
from uuid import uuid4

from store import StoreError
from website_builder import website_status, website_file, SITE_FILES

_DEPLOY_LOCK=threading.RLock()


def init_releases(store):
    with store.db() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS project_releases (
          id TEXT PRIMARY KEY,project_id TEXT NOT NULL,status TEXT NOT NULL,
          source_generation TEXT NOT NULL,manifest TEXT NOT NULL,
          created_at INTEGER NOT NULL,approved_by TEXT,approved_at INTEGER,
          staged_at INTEGER);
        CREATE INDEX IF NOT EXISTS ix_releases_project ON project_releases(project_id,created_at);
        ''')


def releases(store,pid):
    store.project(pid)
    with store.db() as db:
        return [dict(r) for r in db.execute('SELECT * FROM project_releases WHERE project_id=? ORDER BY created_at DESC,id DESC',(pid,))]


def prepare(store,pid):
    if store.project(pid)['intake']['product_type']!='website':
        raise StoreError('Local website release requires a website project.')
    data=website_status(store,pid)
    if not data['built']:raise StoreError('Build the website first.')
    manifest={name:hashlib.sha256(website_file(store,pid,name)).hexdigest() for name in SITE_FILES}
    now=int(time.time());rid=uuid4().hex[:16]
    with store.db() as db:
        db.execute('INSERT INTO project_releases(id,project_id,status,source_generation,manifest,created_at) VALUES(?,?,?,?,?,?)',
                   (rid,pid,'pending_approval',data['generation'],json.dumps(manifest),now))
        store._event(db,pid,'release_prepared','human',{'release_id':rid,'generation':data['generation']})
    return {'id':rid,'status':'pending_approval','generation':data['generation'],'manifest':manifest}


def approve(store,pid,rid,actor,confirmation):
    project=store.project(pid)
    if confirmation!=project['name']:raise StoreError('Exact project name required to approve release.')
    with store.db() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute('SELECT status FROM project_releases WHERE id=? AND project_id=?',(rid,pid)).fetchone()
        if not row or row['status']!='pending_approval':raise StoreError('Release not awaiting approval.')
        db.execute("UPDATE project_releases SET status='approved',approved_by=?,approved_at=? WHERE id=?",
                   (actor,int(time.time()),rid))
        store._event(db,pid,'release_approved',actor,{'release_id':rid})
    return {'id':rid,'status':'approved','approved_by':actor}


def stage(store,pid,rid):
    """Promote checked website assets into local staging; no network or remote deployment."""
    root=Path(store.filename).parent/'staged_releases'/pid
    with _DEPLOY_LOCK:
        with store.db() as db:
            row=db.execute('SELECT * FROM project_releases WHERE id=? AND project_id=?',(rid,pid)).fetchone()
        if not row or row['status']!='approved':raise StoreError('Human-approved release required.')
        current=website_status(store,pid)
        if current.get('generation')!=row['source_generation']:
            raise StoreError('Source generation changed; prepare and approve a new release.')
        manifest=json.loads(row['manifest'])
        data={name:website_file(store,pid,name) for name in SITE_FILES}
        if any(hashlib.sha256(data[name]).hexdigest()!=digest for name,digest in manifest.items()):
            raise StoreError('Website assets changed since release approval.')
        stage_dir=root/'revisions'/rid
        stage_dir.mkdir(parents=True,exist_ok=False)
        try:
            for name,raw in data.items():
                with open(stage_dir/name,'xb') as f:f.write(raw)
            pointer=root/('.next-'+rid)
            pointer.write_text(rid,encoding='utf-8')
            os.replace(pointer,root/'active')
        except Exception:
            shutil.rmtree(stage_dir,ignore_errors=True)
            raise
        with store.db() as db:
            db.execute("UPDATE project_releases SET status='staged',staged_at=? WHERE id=?",(int(time.time()),rid))
            store._event(db,pid,'release_staged','human',{'release_id':rid})
    return {'id':rid,'status':'staged','path':str(stage_dir),
            'note':'Staged on local disk only; external deployment requires a separate operator action.'}


def remove_releases(store,pid):
    root=Path(store.filename).parent/'staged_releases'/pid
    if root.is_symlink():root.unlink()
    elif root.exists():shutil.rmtree(root)
    with store.db() as db:
        db.execute('DELETE FROM project_releases WHERE project_id=?',(pid,))
