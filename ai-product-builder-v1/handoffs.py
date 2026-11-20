"""Project-scoped artifact ledger and mandatory verified agent-to-agent handoffs.

The ledger records evidence; it does not claim that AI-generated text is executed code.
"""
import hashlib
from functools import lru_cache
import json
import os
import re
import time
from pathlib import Path
from uuid import uuid4

from store import StoreError
from roles import ROLES

MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024  # 1 GiB per artifact; streamed, never buffered
ALLOWED_SUFFIXES = {'.txt', '.md', '.py', '.json', '.csv', '.pdf', '.ipynb', '.yaml', '.yml', '.toml', '.html', '.js', '.css'}
CHUNK_BYTES = 1024 * 1024
EXCERPT_MAX = 12000


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
        db.execute('''CREATE TABLE IF NOT EXISTS meeting_attachments (
            id TEXT PRIMARY KEY, meeting_id TEXT NOT NULL, project_id TEXT,
            artifact_id TEXT, name TEXT NOT NULL, sha256 TEXT NOT NULL,
            size INTEGER NOT NULL, created_at INTEGER NOT NULL)''')
        cols={r['name'] for r in db.execute('PRAGMA table_info(messages)')}
        if 'handoff_id' not in cols:
            db.execute('ALTER TABLE messages ADD COLUMN handoff_id TEXT')


def _artifact_path(store, project_id, aid, name):
    # All file path components are server-generated or validated basenames.
    return Path(store.filename).parent/'artifacts'/project_id/aid/name


def _validate_name(name):
    if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.() -]{1,160}',name)
        or name in {'.','..'} or name.endswith((' ','.')) or name.startswith(' ')):
        raise StoreError('Filename must be a safe basename up to 160 characters.')
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        raise StoreError('Unsupported file type.')
    return name


def _stream_to_file(stream, size, path):
    if isinstance(size,bool) or not isinstance(size,int) or not 1 <= size <= MAX_ARTIFACT_BYTES:
        raise StoreError('File must be between 1 byte and 1 GiB.')
    digest=hashlib.sha256()
    remaining=size
    with path.open('xb') as out:
        while remaining:
            piece=stream.read(min(CHUNK_BYTES,remaining))
            if not piece:
                raise StoreError('Upload ended before the declared file size was received.')
            if len(piece)>remaining:
                raise StoreError('Upload size exceeded the declared length.')
            digest.update(piece)
            out.write(piece)
            remaining-=len(piece)
    return digest.hexdigest()


def save_artifact_stream(store, pid, name, stream, size, creator='human'):
    """Bounded-RAM upload with per-project isolation, SHA-256 and atomic visibility."""
    store.project(pid)
    name=_validate_name(name)
    if creator!='human' and creator not in ROLES:
        raise StoreError('Unknown artifact author.')
    if isinstance(size,bool) or not isinstance(size,int) or not 1<=size<=MAX_ARTIFACT_BYTES:
        raise StoreError('File must be between 1 byte and 1 GiB.')
    aid=uuid4().hex
    target=_artifact_path(store,pid,aid,name)
    directory=target.parent
    directory.mkdir(parents=True,exist_ok=False)
    staging=directory/'.partial-upload'
    try:
        checksum=_stream_to_file(stream,size,staging)
        with store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            state=db.execute('SELECT status FROM projects WHERE id=?',(pid,)).fetchone()
            if not state or state['status'] in ('approved','deleting'):
                raise StoreError('Project is no longer accepting uploaded files.')
            # Publish bytes before metadata while holding the SQLite write lock.
            staging.replace(target)
            db.execute('INSERT INTO artifacts VALUES(?,?,?,?,?,?,?)',
                       (aid,pid,name,checksum,size,creator,int(time.time())))
            store._event(db,pid,'artifact_uploaded',creator,{'id':aid,'name':name,'sha256':checksum})
        return {'id':aid,'name':name,'sha256':checksum,'size':size,'creator':creator}
    except Exception:
        staging.unlink(missing_ok=True)
        target.unlink(missing_ok=True)
        try:directory.rmdir()
        except OSError:pass
        raise


def save_artifact(store, pid, name, raw, creator='human'):
    """Compatibility helper for small in-memory test artifacts."""
    from io import BytesIO
    if not isinstance(raw, bytes):
        raise StoreError('Artifact content must be bytes.')
    return save_artifact_stream(store,pid,name,BytesIO(raw),len(raw),creator=creator)


def _meeting_root(store,mid):
    return Path(store.filename).parent/'meeting-artifacts'/mid


def save_meeting_artifact_stream(store,mid,name,stream,size):
    """Project rooms share uploaded files with every project worker.

    Standalone rooms keep their files isolated to their own meeting context.
    """
    name=_validate_name(name)
    if isinstance(size,bool) or not isinstance(size,int) or not 1<=size<=MAX_ARTIFACT_BYTES:
        raise StoreError('File must be between 1 byte and 1 GiB.')
    with store.db() as db:
        room=db.execute('SELECT id,project_id,status FROM meetings WHERE id=?',(mid,)).fetchone()
    if not room or room['status'] in ('ended','cancelled'):
        raise StoreError('Meeting is missing or closed.')
    pid=room['project_id']
    if pid:
        receipt=save_artifact_stream(store,pid,name,stream,size)
        mid_attachment=uuid4().hex
        with store.db() as db:
            db.execute('INSERT INTO meeting_attachments VALUES(?,?,?,?,?,?,?,?)',
                       (mid_attachment,mid,pid,receipt['id'],name,receipt['sha256'],size,int(time.time())))
        receipt['meeting_attachment_id']=mid_attachment
        return receipt
    aid=uuid4().hex
    directory=_meeting_root(store,mid)/aid
    directory.mkdir(parents=True,exist_ok=False)
    stage=directory/'.partial-upload'
    final=directory/name
    try:
        checksum=_stream_to_file(stream,size,stage)
        with store.db() as db:
            db.execute('BEGIN IMMEDIATE')
            state=db.execute('SELECT status FROM meetings WHERE id=?',(mid,)).fetchone()
            if not state or state['status'] in ('ended','cancelled'):
                raise StoreError('Meeting closed during upload.')
            stage.replace(final)
            db.execute('INSERT INTO meeting_attachments VALUES(?,?,?,?,?,?,?,?)',
                       (aid,mid,None,None,name,checksum,size,int(time.time())))
        return {'id':aid,'name':name,'sha256':checksum,'size':size,'meeting_attachment_id':aid}
    except Exception:
        stage.unlink(missing_ok=True);final.unlink(missing_ok=True)
        try:directory.rmdir()
        except OSError:pass
        raise


def meeting_artifacts(store,mid):
    with store.db() as db:
        if not db.execute('SELECT 1 FROM meetings WHERE id=?',(mid,)).fetchone():
            raise StoreError('Unknown meeting.')
        return [dict(row) for row in db.execute(
            'SELECT * FROM meeting_attachments WHERE meeting_id=? ORDER BY created_at,id',(mid,))]


@lru_cache(maxsize=512)
def _verified_digest(canonical_path, size, mtime_ns, ctime_ns, inode):
    """Rehash on file changes; cache stable immutable uploads across workers."""
    checksum=hashlib.sha256()
    consumed=0
    with open(canonical_path,'rb') as source:
        while True:
            chunk=source.read(CHUNK_BYTES)
            if not chunk:break
            consumed+=len(chunk)
            if consumed>size:break
            checksum.update(chunk)
    return consumed,checksum.hexdigest()


def _verify_disk(path,expected_size,expected_sha):
    try:
        stat=path.stat()
        if stat.st_size!=expected_size:
            raise StoreError('Uploaded file integrity verification failed.')
        consumed,digest=_verified_digest(str(path),expected_size,stat.st_mtime_ns,
                                         stat.st_ctime_ns,getattr(stat,'st_ino',0))
    except OSError as exc:
        raise StoreError('Uploaded file is missing or unreadable.') from exc
    if consumed!=expected_size or digest!=expected_sha:
        raise StoreError('Uploaded file integrity verification failed.')


def artifact_file(store,pid,aid,verify=True):
    store.project(pid)
    with store.db() as db:
        row=db.execute('SELECT * FROM artifacts WHERE id=? AND project_id=?',(aid,pid)).fetchone()
    if not row:raise StoreError('Unknown file for this project.')
    meta=dict(row)
    path=_artifact_path(store,pid,aid,row['name'])
    if verify:_verify_disk(path,row['size'],row['sha256'])
    return meta,path


def _read_excerpt_from_path(path,meta,page=None,offset=0,query='',max_chars=EXCERPT_MAX):
    if isinstance(offset,bool) or not isinstance(offset,int) or offset<0:
        raise StoreError('Offset must be a nonnegative integer.')
    if not isinstance(query,str) or len(query)>160:
        raise StoreError('Search query must be at most 160 characters.')
    max_chars=min(EXCERPT_MAX,max(1,int(max_chars)))
    extension=Path(meta['name']).suffix.lower()
    if extension=='.pdf':
        try:import pymupdf
        except ImportError as exc:raise StoreError('PDF extraction requires PyMuPDF; install requirements-rag.txt.') from exc
        try:
            with pymupdf.open(str(path)) as doc:
                if doc.needs_pass:raise StoreError('PDF is password-protected.')
                selected=page if page is not None else 1
                if isinstance(selected,bool) or not isinstance(selected,int) or not 1<=selected<=doc.page_count:
                    raise StoreError(f'PDF page must be between 1 and {doc.page_count}.')
                text=doc[selected-1].get_text('text',sort=True)
                return {'page':selected,'pages':doc.page_count,'text':text[offset:offset+max_chars],
                        'next_offset':offset+max_chars if offset+max_chars<len(text) else None,
                        'readable':bool(text.strip())}
        except StoreError:raise
        except (ValueError,RuntimeError,OSError) as exc:raise StoreError('Unable to extract PDF page.') from exc
    if extension not in {'.txt','.md','.csv','.py','.json','.ipynb','.yaml','.yml','.toml','.html','.js','.css'}:
        raise StoreError('No readable text parser for this file type.')
    # UTF-8 offset is a BYTE position. Seek in the file; never load a 1 GB file in RAM.
    if query.strip():
        needle=query.strip().encode('utf-8').lower()
        with path.open('rb') as source:
            overlap=b'';start=0;match=None
            while True:
                block=source.read(CHUNK_BYTES)
                if not block:break
                combined=overlap+block
                pos=combined.lower().find(needle)
                if pos>=0:
                    match=max(0,start-len(overlap)+pos-2048)
                    break
                overlap=combined[-max(len(needle),256):]
                start+=len(block)
        if match is not None:offset=match
    with path.open('rb') as source:
        source.seek(offset)
        data=source.read(max_chars)
    text=data.decode('utf-8',errors='replace')[:max_chars]
    next_offset=offset+len(data) if offset+len(data)<meta['size'] else None
    return {'page':None,'pages':None,'text':text,'next_offset':next_offset,
            'readable':bool(text.strip())}


def read_artifact_excerpt(store,pid,aid,*,page=None,offset=0,query='',max_chars=EXCERPT_MAX):
    meta,path=artifact_file(store,pid,aid)
    excerpt=_read_excerpt_from_path(path,meta,page,offset,query,max_chars)
    return {'artifact_id':aid,'filename':meta['name'],'sha256':meta['sha256'],
            'size':meta['size'],**excerpt}


def read_meeting_artifact_excerpt(store,mid,aid,**kwargs):
    with store.db() as db:
        row=db.execute('SELECT * FROM meeting_attachments WHERE id=? AND meeting_id=?',(aid,mid)).fetchone()
    if not row:raise StoreError('Meeting attachment not found.')
    if row['project_id']:
        return read_artifact_excerpt(store,row['project_id'],row['artifact_id'],**kwargs)
    path=_meeting_root(store,mid)/aid/row['name']
    _verify_disk(path,row['size'],row['sha256'])
    meta={'name':row['name'],'size':row['size']}
    return {'artifact_id':aid,'filename':row['name'],'sha256':row['sha256'],
            'size':row['size'],**_read_excerpt_from_path(path,meta,**kwargs)}


def agent_file_context(store,pid=None,mid=None,max_previews=2):
    """Give every authorized worker a file inventory and bounded real text excerpts."""
    if pid:
        inventory=artifacts(store,pid)
    elif mid:
        inventory=meeting_artifacts(store,mid)
    else:return {'files':[],'excerpts':[]}
    items=[{'artifact_id':x.get('artifact_id') or x['id'], 'name':x['name'],
            'size':x['size'],'sha256':x['sha256']} for x in inventory]
    excerpts=[]
    for asset in inventory[:max_previews]:
        try:
            if pid:entry=read_artifact_excerpt(store,pid,asset['id'],max_chars=4500)
            else:entry=read_meeting_artifact_excerpt(store,mid,asset['id'],max_chars=4500)
            excerpts.append(entry)
        except StoreError as exc:
            excerpts.append({'artifact_id':asset['id'],'error':str(exc)})
    return {'files':items,'excerpts':excerpts,'note':
            'Files belong to this project/meeting. Text excerpts are verified actual file content. '
            'Request additional pages or byte offsets using file_requests; do not infer unseen content.'}

def artifacts(store,pid):
    store.project(pid)
    with store.db() as db:
        return [dict(r) for r in db.execute('SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at,id',(pid,))]


def read_artifact(store,pid,aid):
    """Small-blob compatibility path; large documents require streaming readers."""
    meta,path=artifact_file(store,pid,aid)
    if meta['size']>8*1024*1024:
        raise StoreError('File is too large for in-memory read. Use the streaming reader instead.')
    return meta,path.read_bytes()


def _manifest(db,store,pid,source,sender,recipient):
    files=[]
    # Cap context metadata; large project collections must be retrieved by artifact ID.
    rows=db.execute('SELECT * FROM artifacts WHERE project_id=? ORDER BY created_at DESC,id DESC LIMIT 40',(pid,)).fetchall()
    for row in rows:
        files.append({'id':row['id'],'name':row['name'],'sha256':row['sha256'],'size':row['size']})
    from execution_workspace import list_files
    workspace_files=list_files(store,pid)[:40]
    return {
        'workspace_files':workspace_files,
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
    from execution_workspace import read_version
    for f in manifest.get('workspace_files',[]):
        pinned=read_version(store,row['project_id'],f['sha256'])
        if len(pinned)!=f['size']:
            raise StoreError('Verified handoff source-version length mismatch.')
    for f in manifest['files']:
        ar=db.execute('SELECT * FROM artifacts WHERE id=? AND project_id=?',(f['id'],row['project_id'])).fetchone()
        if not ar or ar['sha256']!=f['sha256'] or ar['size']!=f['size']:
            raise StoreError('Handoff artifact metadata changed.')
        path=_artifact_path(store,row['project_id'],ar['id'],ar['name'])
        _verify_disk(path,ar['size'],ar['sha256'])
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
