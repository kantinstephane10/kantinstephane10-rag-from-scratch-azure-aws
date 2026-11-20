"""Project-private code workspace with unattended Windows Python execution.

Windows mode executes agent-generated Python with the server user's full host
permissions. It is not a sandbox. Docker is available for optional checks.
"""
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

from store import StoreError

ALLOWED_EXTENSIONS = {'.py','.js','.css','.html','.json','.md','.txt','.toml','.yaml','.yml','.csv','.ipynb'}
MAX_SOURCE_BYTES = 256 * 1024
MAX_FILE_COUNT = 400
DEFAULT_IMAGE = 'python:3.12-slim'
_ALLOWED_COMMANDS = {
    'unittest': ['python','-m','unittest','discover','-s','tests','-v'],
    'compile': ['python','-m','compileall','-q','.'],
    'pytest': ['python','-m','pytest','-q'],
}
_LOCKS = {}
_GUARD = threading.Lock()


def _lock(pid):
    with _GUARD:
        return _LOCKS.setdefault(pid,threading.RLock())


def init_workspace(store):
    with store.db() as db:
        db.executescript('''
          CREATE TABLE IF NOT EXISTS workspace_runs (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, command TEXT NOT NULL,
            actor TEXT NOT NULL, status TEXT NOT NULL, exit_code INTEGER,
            output TEXT, created_at INTEGER NOT NULL, ended_at INTEGER);
          CREATE INDEX IF NOT EXISTS ix_workspace_runs ON workspace_runs(project_id,created_at);
          CREATE TABLE IF NOT EXISTS workspace_auto_settings (
            project_id TEXT PRIMARY KEY, enabled INTEGER NOT NULL DEFAULT 1);
          CREATE TABLE IF NOT EXISTS workspace_auto_jobs (
            id TEXT PRIMARY KEY, project_id TEXT NOT NULL, command TEXT NOT NULL,
            source_sha256 TEXT NOT NULL, actor TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued', run_id TEXT, error TEXT,
            created_at INTEGER NOT NULL, started_at INTEGER, ended_at INTEGER,
            cancel_requested INTEGER NOT NULL DEFAULT 0,
            UNIQUE(project_id,command,source_sha256));
          CREATE INDEX IF NOT EXISTS ix_workspace_auto_jobs ON workspace_auto_jobs(status,created_at);
        ''')


def workspace_root(store,pid):
    if not isinstance(pid,str) or not re.fullmatch(r'[a-f0-9]{12}',pid):
        raise StoreError('Invalid project identifier.')
    store.project(pid)
    base=Path(store.filename).parent/'workspaces'
    base.mkdir(mode=0o700,parents=True,exist_ok=True)
    root=base/pid
    if root.is_symlink(): raise StoreError('Unsafe workspace symlink.')
    root.mkdir(mode=0o700,exist_ok=True)
    return root


def _safe_target(root,path):
    if not isinstance(path,str) or not 1<=len(path)<=160 or '\\' in path or '\x00' in path:
        raise StoreError('Invalid workspace filename.')
    p=PurePosixPath(path)
    if p.is_absolute() or '..' in p.parts or any(x in ('','.','..') for x in path.split('/')) or len(p.parts)>5:
        raise StoreError('Workspace path traversal rejected.')
    if p.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise StoreError('Unsupported source file extension.')
    result=root
    for segment in p.parts[:-1]:
        result=result/segment
        if result.is_symlink():raise StoreError('Workspace directory symlink rejected.')
        result.mkdir(mode=0o700,exist_ok=True)
    target=result/p.parts[-1]
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise StoreError('Workspace destination is unsafe.')
    return target


def _version_path(store,pid,digest):
    if not re.fullmatch(r'[a-f0-9]{64}',digest):raise StoreError('Invalid source version identifier.')
    base=Path(store.filename).parent/'workspace_versions'/pid
    if base.is_symlink() or (hasattr(base,'is_junction') and base.is_junction()):
        raise StoreError('Unsafe source-version directory.')
    base.mkdir(mode=0o700,parents=True,exist_ok=True)
    return base/digest


def read_version(store,pid,digest):
    store.project(pid)
    path=_version_path(store,pid,digest)
    if path.is_symlink():raise StoreError('Unsafe version link.')
    try:data=path.read_bytes()
    except OSError as exc:raise StoreError('Immutable source snapshot missing.') from exc
    if hashlib.sha256(data).hexdigest()!=digest:
        raise StoreError('Immutable source snapshot hash mismatch.')
    return data


def _save_version(store,pid,digest,data):
    target=_version_path(store,pid,digest)
    if target.is_symlink():raise StoreError('Unsafe version file link.')
    if target.exists():
        if target.read_bytes()!=data:raise StoreError('Immutable source snapshot changed unexpectedly.')
        return
    pending=target.parent/('.pending-'+secrets.token_hex(12))
    try:
        with open(pending,'xb') as f:
            f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(pending,target)
    finally:pending.unlink(missing_ok=True)


def write_file(store,pid,path,content,actor='human',expected_sha256=None):
    if not isinstance(content,str):raise StoreError('Source must be UTF-8 text.')
    data=content.encode('utf-8')
    if not 0<len(data)<=MAX_SOURCE_BYTES:raise StoreError('Source exceeds 256 KiB limit.')
    root=workspace_root(store,pid)
    with _lock(pid):
        count=sum(1 for f in root.rglob('*') if f.is_file() and
            not any(part in ('__pycache__','.pytest_cache','.python-runner-temp') for part in f.relative_to(root).parts))
        target=_safe_target(root,path)
        if not target.exists() and count>=MAX_FILE_COUNT:raise StoreError('Workspace file-count limit reached.')
        if target.exists():
            prior=hashlib.sha256(target.read_bytes()).hexdigest()
            if expected_sha256 != prior:
                raise StoreError('File already exists; provide its current SHA-256 for a safe update.')
        elif expected_sha256 is not None:
            raise StoreError('Expected existing source file was not found.')
        digest=hashlib.sha256(data).hexdigest()
        _save_version(store,pid,digest,data)
        temp=target.parent/('.pending-'+secrets.token_hex(12))
        try:
            with open(temp,'xb') as f:
                f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(temp,target)
        finally:
            temp.unlink(missing_ok=True)
        with store.db() as db:
            store._event(db,pid,'workspace_file_written',actor,
                         {'path':path,'sha256':digest,'bytes':len(data)})
    return {'path':path,'sha256':digest,'size':len(data)}


def list_files(store,pid):
    root=workspace_root(store,pid)
    with _lock(pid):
        files=[]
        for item in sorted(root.rglob('*')):
            if any(part in ('__pycache__','.pytest_cache','.python-runner-temp') for part in item.relative_to(root).parts):continue
            if item.suffix.lower() in ('.pyc','.pyo'):continue
            if item.is_symlink():raise StoreError('Unsafe symlink encountered in workspace.')
            if not item.is_file():continue
            if len(files)>=MAX_FILE_COUNT:raise StoreError('Workspace file count exceeded.')
            raw=item.read_bytes()
            files.append({'path':item.relative_to(root).as_posix(),
                          'sha256':hashlib.sha256(raw).hexdigest(),'size':len(raw)})
        return files


def read_file(store,pid,path):
    root=workspace_root(store,pid)
    target=_safe_target(root,path)
    try:
        data=target.read_bytes()
    except OSError as exc:raise StoreError('Workspace file unavailable.') from exc
    if len(data)>MAX_SOURCE_BYTES:raise StoreError('Source file too large to display.')
    return {'path':path,'content':data.decode('utf-8'),
            'sha256':hashlib.sha256(data).hexdigest()}


def runs(store,pid):
    store.project(pid)
    with store.db() as db:
        return [dict(r) for r in db.execute('SELECT * FROM workspace_runs WHERE project_id=? ORDER BY created_at DESC,id DESC LIMIT 35',(pid,))]


def run_check(store,pid,command,*,actor='human',image=DEFAULT_IMAGE,docker_bin='docker',transport=subprocess.run):
    """Run a fixed test command inside a no-network, non-root Docker container.

    No shell, sockets, host credentials, external mounts or host programs are run.
    Docker daemon security is a separate deployment prerequisite, not guaranteed here.
    """
    if command not in _ALLOWED_COMMANDS:raise StoreError('Only approved test commands are available.')
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9._:/@-]{1,150}',image):
        raise StoreError('Invalid Docker image reference.')
    root=workspace_root(store,pid)
    with _lock(pid):
        if not list_files(store,pid):raise StoreError('Write source files before running tests.')
        run_id=secrets.token_hex(12);name='builder-'+run_id
        start=int(time.time())
        with store.db() as db:
            db.execute('INSERT INTO workspace_runs(id,project_id,command,actor,status,created_at) VALUES(?,?,?,?,?,?)',
                       (run_id,pid,command,actor,'running',start))
        cmd=[docker_bin,'run','--rm','--name',name,'--pull=never',
             '--network','none','--read-only','--cap-drop','ALL',
             '--security-opt','no-new-privileges','--pids-limit','64',
             '--memory','512m','--cpus','1', '--user','65534:65534',
             '--tmpfs','/tmp:rw,noexec,nosuid,nodev,size=64m',
             '--mount',f'type=bind,src={root.resolve()},dst=/workspace,readonly',
             '--workdir','/workspace','-e','PYTHONDONTWRITEBYTECODE=1',image,
             *_ALLOWED_COMMANDS[command]]
        status='failed'; code=None; output=''
        try:
            result=transport(cmd,capture_output=True,text=True,timeout=90,check=False)
            code=result.returncode
            output=(result.stdout or '')+'\n'+(result.stderr or '')
            status='passed' if code==0 else 'failed'
        except subprocess.TimeoutExpired:
            status='timeout';output='Sandbox command exceeded 90-second deadline.'
            try:subprocess.run([docker_bin,'rm','-f',name],capture_output=True,timeout=6,check=False)
            except (OSError,subprocess.SubprocessError):pass
        except (OSError,subprocess.SubprocessError) as exc:
            status='unavailable';output=f'Isolated Docker runner unavailable: {type(exc).__name__}'
        # Never publish secrets, whole core dumps or unbounded log output.
        output=output[:12000]
        with store.db() as db:
            db.execute('UPDATE workspace_runs SET status=?,exit_code=?,output=?,ended_at=? WHERE id=?',
                       (status,code,output,int(time.time()),run_id))
            store._event(db,pid,'workspace_test_finished',actor,
                         {'run_id':run_id,'command':command,'status':status,'exit_code':code})
        return {'id':run_id,'status':status,'exit_code':code,'output':output,'command':command}



def auto_enabled(store,pid):
    store.project(pid)
    with store.db() as db:
        row=db.execute('SELECT enabled FROM workspace_auto_settings WHERE project_id=?',(pid,)).fetchone()
    return bool(row['enabled']) if row else True


def set_auto_enabled(store,pid,enabled,actor='human'):
    store.project(pid)
    if not isinstance(enabled,bool):raise StoreError('Automatic execution must be true or false.')
    with store.db() as db:
        db.execute('INSERT INTO workspace_auto_settings(project_id,enabled) VALUES(?,?) '
                   'ON CONFLICT(project_id) DO UPDATE SET enabled=excluded.enabled',(pid,int(enabled)))
        store._event(db,pid,'windows_auto_execution_changed',actor,{'enabled':enabled})
    return {'enabled':enabled}


def auto_jobs(store,pid):
    store.project(pid)
    with store.db() as db:
        return [dict(r) for r in db.execute("""SELECT * FROM workspace_auto_jobs
              WHERE project_id=? ORDER BY created_at DESC,rowid DESC LIMIT 75""",(pid,))]


def _auto_candidates(store,pid,saved_files,requested=()):
    """All Python is compiled; run entrypoints, test suites and model-requested scripts.

    Library modules are never mistaken for app entrypoints. Explicit .py requests
    can run any project-local module without per-script approval.
    """
    filenames=[row['path'] for row in saved_files if row['path'].endswith('.py')]
    commands=[]
    if filenames:
        commands.append('compile')
        for name in filenames:
            code=read_file(store,pid,name)['content']
            basename=Path(name).name.lower()
            if basename in ('app.py','main.py','run.py','server.py','manage.py') or (
                '__name__' in code and '__main__' in code):
                commands.append(name)
        if any(x.startswith('tests/') or Path(x).name.startswith('test_') for x in filenames):
            commands.append('unittest')
    for item in requested:
        if not isinstance(item,str):continue
        if item in _ALLOWED_COMMANDS:
            commands.append(item)
        elif item.endswith('.py') and not item.startswith('/'):
            # Validated by the same project-scoped path function used by manual runs.
            try:target=_safe_target(workspace_root(store,pid),item)
            except StoreError:continue
            if target.exists() and target.is_file():commands.append(item)
    return list(dict.fromkeys(commands))[:40]


def queue_auto_runs(store,pid,saved_files=(),requested=(),actor='agent'):
    """Persist a batch before executing it. No approval or confirmation gate."""
    if not isinstance(requested,(list,tuple)):requested=[]
    commands=_auto_candidates(store,pid,list(saved_files),requested)
    if not commands:return []
    inventory={x['path']:x['sha256'] for x in list_files(store,pid)}
    # Batch digest ensures a rebuilt source revision gets a new test run.
    fingerprint=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
    created=[]
    with store.db() as db:
        for command in commands:
            source_sha=hashlib.sha256((command+'\n'+fingerprint).encode()).hexdigest()
            jid=secrets.token_hex(12)
            cur=db.execute("""INSERT OR IGNORE INTO workspace_auto_jobs
                  (id,project_id,command,source_sha256,actor,status,created_at)
                  VALUES(?,?,?,?,?,'queued',?)""",(jid,pid,command,source_sha,actor,int(time.time())))
            if cur.rowcount:created.append(jid)
        if created:
            store._event(db,pid,'windows_auto_jobs_queued',actor,
                         {'count':len(created),'commands':commands})
    return created


def recover_auto_jobs(store):
    with store.db() as db:
        # A service restart can safely retry unfinished project-local jobs.
        db.execute("UPDATE workspace_auto_jobs SET status='queued',cancel_requested=0,started_at=NULL "
                   "WHERE status='running'")


def claim_auto_job(store):
    with store.db() as db:
        row=db.execute("""SELECT j.* FROM workspace_auto_jobs j
            LEFT JOIN workspace_auto_settings s ON s.project_id=j.project_id
            JOIN projects p ON p.id=j.project_id
            WHERE j.status='queued' AND COALESCE(s.enabled,1)=1
              AND p.status NOT IN ('cancelled')
            ORDER BY j.created_at,j.rowid LIMIT 1""").fetchone()
        if row is None:return None
        updated=db.execute("UPDATE workspace_auto_jobs SET status='running',started_at=? "
                           "WHERE id=? AND status='queued'",(int(time.time()),row['id']))
        return dict(row) if updated.rowcount else None


def cancel_auto_jobs(store,pid,actor='human'):
    store.project(pid)
    with store.db() as db:
        queued=db.execute("UPDATE workspace_auto_jobs SET status='cancelled',ended_at=? "
                          "WHERE project_id=? AND status='queued'",(int(time.time()),pid)).rowcount
        running=db.execute("UPDATE workspace_auto_jobs SET cancel_requested=1 "
                           "WHERE project_id=? AND status='running'",(pid,)).rowcount
        store._event(db,pid,'windows_auto_jobs_cancelled',actor,
                     {'queued_cancelled':queued,'running_stop_requested':running})
    return {'queued_cancelled':queued,'running_stop_requested':running}


def process_auto_job(store,stop=None):
    job=claim_auto_job(store)
    if not job:return False
    pid=job['project_id']
    def should_cancel():
        if stop is not None and stop.is_set():return True
        with store.db() as db:
            r=db.execute('SELECT cancel_requested FROM workspace_auto_jobs WHERE id=?',(job['id'],)).fetchone()
            return not r or bool(r['cancel_requested'])
    try:
        # A queued job must not silently execute a later, different source version.
        inventory={x['path']:x['sha256'] for x in list_files(store,pid)}
        current_hash=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest()
        expected=hashlib.sha256((job['command']+'\n'+current_hash).encode()).hexdigest()
        if expected != job['source_sha256']:
            status='superseded';error='Project source changed while this run was queued.';rid=None
        elif should_cancel():
            status='cancelled';error='Execution cancellation requested.';rid=None
        else:
            result=run_windows_python(store,pid,job['command'],actor=job['actor'],
                                      timeout=180,should_cancel=should_cancel)
            status=result['status'];error=None if status=='passed' else (result['output'] or '')[:600]
            rid=result['id']
    except Exception as exc:
        status='failed';error=f'{type(exc).__name__}: {exc}'[:600];rid=None
    with store.db() as db:
        db.execute("""UPDATE workspace_auto_jobs SET status=?,error=?,run_id=?,ended_at=? WHERE id=?""",
                   (status,error,rid,int(time.time()),job['id']))
        store._event(db,pid,'windows_auto_job_finished',job['actor'],
                     {'job_id':job['id'],'command':job['command'],'status':status,'run_id':rid})
    return True


def delete_workspace(store,pid):
    # A running host job holds this lock; deletion waits for its cancellation.
    with _lock(pid):
        base=Path(store.filename).parent/'workspaces'
        target=base/pid
        if target.is_symlink():target.unlink()
        elif target.exists():shutil.rmtree(target)
        versions=Path(store.filename).parent/'workspace_versions'/pid
        if versions.is_symlink():versions.unlink()
        elif versions.exists():shutil.rmtree(versions)
        with store.db() as db:
            db.execute('DELETE FROM workspace_runs WHERE project_id=?',(pid,))
            db.execute('DELETE FROM workspace_auto_jobs WHERE project_id=?',(pid,))
            db.execute('DELETE FROM workspace_auto_settings WHERE project_id=?',(pid,))



def _run_host_bounded(cmd, cwd, env, seconds, should_cancel=None):
    """Bound stdout memory and try to kill a whole local subprocess tree on timeout.

    This is operational containment only, NOT a Windows sandbox or privilege barrier.
    """
    import signal
    from types import SimpleNamespace
    creationflags=getattr(subprocess,'CREATE_NEW_PROCESS_GROUP',0) if os.name=='nt' else 0
    proc=subprocess.Popen(cmd,cwd=cwd,env=env,stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                          creationflags=creationflags,start_new_session=os.name!='nt')
    capture=bytearray();exceeded=threading.Event()
    def stop_tree():
        if proc.poll() is not None:return
        if os.name=='nt':
            try:subprocess.run(['taskkill','/F','/T','/PID',str(proc.pid)],
                 stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=7)
            except (OSError,subprocess.SubprocessError):proc.kill()
        else:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except (OSError,ProcessLookupError):proc.kill()
    def reader():
        while True:
            part=proc.stdout.read(4096)
            if not part:break
            if len(capture)+len(part)>1048576:
                exceeded.set();stop_tree();break
            capture.extend(part)
    thread=threading.Thread(target=reader,daemon=True)
    thread.start()
    deadline=time.monotonic()+seconds
    cancelled=False
    try:
        while proc.poll() is None:
            if should_cancel is not None and should_cancel():
                cancelled=True;stop_tree();break
            if time.monotonic()>=deadline:
                stop_tree();raise subprocess.TimeoutExpired(cmd,seconds)
            try:proc.wait(timeout=min(.25,max(0.01,deadline-time.monotonic())))
            except subprocess.TimeoutExpired:pass
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        stop_tree();proc.wait(timeout=10)
        thread.join(timeout=3)
        proc.stdout.close()
        raise
    thread.join(timeout=3)
    proc.stdout.close()
    msg=capture.decode('utf-8',errors='replace')
    if cancelled:return SimpleNamespace(returncode=130,stdout=msg+'\n[Windows run cancelled]',stderr='',cancelled=True)
    if exceeded.is_set():return SimpleNamespace(returncode=1,stdout=msg+'\n[Output limit 1 MiB exceeded; run terminated.]',stderr='')
    return SimpleNamespace(returncode=proc.returncode,stdout=msg,stderr='')

# Windows-native runs deliberately execute with the privileges of the server process.
# They are NOT a sandbox: malicious Python can read host files and contact networks.
# The API requires loopback and an authenticated administrator in multiuser mode.
# Agent-generated Python may execute unattended using the Windows account.
def run_windows_python(store, pid, filename, *, confirmation=None, actor='human',
                       python_executable=None, transport=subprocess.run, timeout=90,
                       should_cancel=None):
    """Run a single approved project-local .py script using the server's Python.

    No shells, arguments, package installers, or arbitrary executables are accepted.
    Secrets from the server process are not passed in the child environment. This
    is a best-effort secret boundary, NOT host filesystem or network isolation.
    """
    import sys
    store.project(pid)
    # Backward-compatible confirmation argument is intentionally ignored.
    standard_command = filename if filename in _ALLOWED_COMMANDS else None
    if standard_command is None and (not isinstance(filename,str) or not filename.lower().endswith('.py')):
        raise StoreError('Choose a project Python file (.py), unittest, pytest or compile.')
    if not isinstance(timeout,int) or not 1 <= timeout <= 1800:
        raise StoreError('Windows execution timeout must be 1–1800 seconds.')
    root=workspace_root(store,pid)
    with _lock(pid):
        target=None if standard_command else _safe_target(root,filename)
        if not list_files(store,pid):
            raise StoreError('Add source files before running Python.')
        if target is not None:
            if not target.exists() or not target.is_file():
                raise StoreError('Python script not found in project workspace.')
            # Check symlinks/junctions in every parent, including on Windows 3.12+.
            if not target.resolve().is_relative_to(root.resolve()):
                raise StoreError('Python script points outside the project workspace.')
            for n in range(len(target.relative_to(root).parts)):
                item = root / Path(*target.relative_to(root).parts[:n])
                if item.is_symlink() or (hasattr(item,'is_junction') and item.is_junction()):
                    raise StoreError('Workspace links are not allowed for host runs.')
        interpreter=str(Path(python_executable or sys.executable).resolve())
        if not Path(interpreter).exists():
            raise StoreError('Configured Python interpreter does not exist.')
        if not Path(interpreter).name.lower().startswith(('python','pypy')):
            raise StoreError('Only a Python interpreter may execute project code.')
        # Minimal environment: prevents accidental inheritance of OPENAI_API_KEY,
        # cloud tokens, authenticated proxies, Python paths and other app secrets.
        keep=('SystemRoot','WINDIR','PATH','PATHEXT','COMSPEC','LANG','LC_ALL')
        env={key:os.environ[key] for key in keep if key in os.environ}
        env.update({'PYTHONDONTWRITEBYTECODE':'1','PYTHONNOUSERSITE':'1',
                    'PYTHONIOENCODING':'utf-8'})
        temp=root/'.python-runner-temp'
        temp.mkdir(exist_ok=True)
        if temp.is_symlink() or (hasattr(temp,'is_junction') and temp.is_junction()):
            raise StoreError('Unsafe temporary directory.')
        env.update({'TEMP':str(temp),'TMP':str(temp),'TMPDIR':str(temp),'HOME':str(temp)})
        run_id=secrets.token_hex(12);started=int(time.time())
        with store.db() as db:
            db.execute('INSERT INTO workspace_runs(id,project_id,command,actor,status,created_at) VALUES(?,?,?,?,?,?)',
                       (run_id,pid,'windows:'+filename,actor,'running',started))
            store._event(db,pid,'windows_python_started',actor,{'run_id':run_id,'path':filename})
        cmd=([interpreter,'-B',*(_ALLOWED_COMMANDS[standard_command][1:])]
             if standard_command else [interpreter,'-B',str(target)])
        status='unavailable';code=None; output=''
        try:
            if transport is subprocess.run:
                result=_run_host_bounded(cmd,str(root),env,timeout,should_cancel)
            else:
                result=transport(cmd,cwd=str(root),env=env,stdin=subprocess.DEVNULL,
                                 capture_output=True,text=True,timeout=timeout,check=False,errors='replace')
            code=result.returncode
            output=(result.stdout or '')+'\n'+(result.stderr or '')
            status=('cancelled' if getattr(result,'cancelled',False) else
                    'passed' if code==0 else 'failed')
        except subprocess.TimeoutExpired:
            status='timeout';output=f'Windows Python execution exceeded {timeout}s.'
        except (OSError,subprocess.SubprocessError) as exc:
            status='unavailable';output=f'Windows Python could not start: {type(exc).__name__}'
        # Console output is untrusted; it may disclose secrets deliberately read by
        # malicious code, so users must review scripts before authorizing execution.
        output=output[:12000]
        with store.db() as db:
            db.execute('UPDATE workspace_runs SET status=?,exit_code=?,output=?,ended_at=? WHERE id=?',
                       (status,code,output,int(time.time()),run_id))
            store._event(db,pid,'workspace_test_finished',actor,
                         {'run_id':run_id,'command':'windows:'+filename,'status':status,'exit_code':code})
        return {'id':run_id,'status':status,'exit_code':code,'output':output,
                'command':'windows:'+filename,'interpreter':interpreter,
                'security':'HOST EXECUTION — not sandboxed; Python can access Windows user resources.'}
