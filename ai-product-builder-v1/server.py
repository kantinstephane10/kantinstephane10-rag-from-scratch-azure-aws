"""Local AI Product Builder dashboard and JSON API (Python standard library only).

Start with `python server.py --demo` for synthetic offline verification, or set
OPENAI_API_KEY and launch `python server.py` for real GPT API calls.
"""
import argparse
import base64
import binascii
import hmac
import json
import logging
import os
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from http.cookies import SimpleCookie
from time import monotonic
from urllib.parse import urlparse, parse_qs

from architecture import blueprint
from config import settings
from runtime_settings import RuntimeSettings
from engine import Engine
from meetings import (create_meeting, meeting, list_meetings, start_meeting,
                      take_floor, yield_floor, end_meeting, cancel_meeting)
from roles import ROLES
from store import Store, StoreError
from handoffs import (save_artifact, save_artifact_stream, save_meeting_artifact_stream,
    artifacts, meeting_artifacts, list_handoffs, read_artifact_excerpt, read_meeting_artifact_excerpt,
    MAX_ARTIFACT_BYTES)
from rag_pipeline import init_rag
from large_rag import (init_large_rag, rag_status_large, start_large_index_job, answer_large_question, cancel_large_index)
from hf_embeddings import DEFAULT_MODEL, MODELS as HF_MODELS
from website_builder import build_website, website_status, website_file, website_zip
from system_readiness import readiness
from execution_workspace import (init_workspace, write_file, list_files, read_file,
    runs as workspace_runs, run_windows_python, run_check as docker_check)
from release_management import (init_releases, releases as project_releases, prepare as prepare_release,
    approve as approve_release, stage as stage_release)
from auth_security import UserAuth, AccessDenied
from metrics_service import Metrics, capacity

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
MAX_POST = 11_500_000  # Legacy JSON payloads; up to 1 GiB uses streaming PUT instead


def make_server(store,engine,host='127.0.0.1',port=8766,access_token=None,runtime=None,enable_multiuser=False):
    is_loopback = host in ('127.0.0.1','localhost','::1')
    if not is_loopback and not access_token:
        raise ValueError('Non-loopback binding requires BUILDER_ACCESS_TOKEN and TLS reverse proxy.')
    init_rag(store)
    init_large_rag(store)
    init_workspace(store)
    init_releases(store)
    metrics=Metrics()
    if enable_multiuser and not is_loopback:
        raise ValueError('Multiuser mode is localhost-only; put an authenticated TLS proxy in front.')
    user_auth=UserAuth(store) if enable_multiuser else None

    def rag_access(pid):
        project=store.project(pid)
        if not is_loopback:
            raise StoreError('RAG execution is restricted to the localhost dashboard.')
        if engine.demo or runtime is None or runtime.data['mode']!='openai':
            raise StoreError('Select Live OpenAI mode under Settings to run document indexing and RAG questions.')
        if not runtime.key:raise StoreError('Save an OpenAI API key in Settings first.')
        if not project['intake']['provider_consent']:
            raise StoreError('Confirm OpenAI document processing consent for this project.')
        return project

    def auto_index(pid):
        # Local Hugging Face embeddings: no paid embedding API and no OpenAI key required.
        project=store.project(pid)
        if project['intake']['product_type']!='rag' or engine.demo:return
        if not any(Path(a['name']).suffix.lower() in {'.pdf','.md','.txt','.csv','.png','.jpg','.jpeg','.webp'}
                   for a in artifacts(store,pid)):return
        try:
            start_large_index_job(store,pid,runtime.key if runtime else None,
                embedding_model=runtime.data['embedding_model'] if runtime else DEFAULT_MODEL,
                ocr_mode=runtime.data['ocr_mode'] if runtime else 'auto',
                ocr_model=runtime.model_for('rag-engineer') if runtime else None)
        except StoreError:
            LOG.exception('Automatic local RAG index scheduling failed for %s',pid)


    class Handler(BaseHTTPRequestHandler):
        server_version='AIProductBuilder/1.6'
        _response_status=500
        _principal=None
        def send_response(self,code,message=None):
            self._response_status=code
            return super().send_response(code,message)

        def _cookie_token(self):
            try:
                cookies=SimpleCookie(self.headers.get('Cookie',''))
                return cookies['builder_session'].value if 'builder_session' in cookies else ''
            except Exception:
                return ''

        def _require_scope(self,path,write=False):
            if user_auth is None:return
            person=self._principal
            if not person:raise AccessDenied('Sign in required.')
            if path.startswith('/api/system/') or (path.startswith('/api/settings') and write) or path.startswith('/api/providers/') or path in ('/api/metrics','/metrics'):
                if not person['admin']:raise AccessDenied('Administrator access required.')
            parts=path.split('/')
            if len(parts)>3 and parts[1:3]==['api','projects'] and re.fullmatch(r'[a-f0-9]{12}',parts[3]):
                level='editor' if write else 'viewer'
                if write and ('delete' in parts or 'releases' in parts):level='owner'
                if write and len(parts)>=5 and parts[4]=='workspace' and len(parts)>=6 and parts[5] in ('run','docker-check'):
                    if not person['admin']:raise AccessDenied('Windows host execution requires an administrator.')
                user_auth.require(person,parts[3],level)
            elif len(parts)>3 and parts[1:3]==['api','meetings'] and re.fullmatch(r'[a-f0-9]{12}',parts[3]):
                user_auth.require_meeting(person,parts[3],'editor' if write else 'viewer')
            elif len(parts)>3 and parts[1:3]==['api','tasks']:
                user_auth.require_task(person,parts[3],'editor' if write else 'viewer')

        def _metric_class(self,path):
            if path.startswith('/api/health'):return 'health'
            if path.startswith('/api/auth/'):return 'auth'
            for segment in ('workspace','releases','rag','website'):
                if '/'+segment+'/' in path or path.endswith('/'+segment):return segment
            if path.startswith('/api/settings') or path.startswith('/api/providers'):return 'settings'
            if path.startswith('/api/projects'):return 'projects'
            if path.startswith('/api/meetings'):return 'meetings'
            if path.startswith('/api/system') or path.startswith('/api/metrics'):return 'other'
            return 'static' if not path.startswith('/api/') else 'other'

        def log_message(self,format,*args):
            LOG.info('%s - %s',self.address_string(),format%args)

        def _authorized(self):
            url=urlparse(self.path);path=url.path
            # DNS rebinding hardening in local mode. Never allow cross-host access.
            if is_loopback:
                hostname=self.headers.get('Host','').split(':')[0].lower()
                if hostname not in ('localhost','127.0.0.1','[::1]'):
                    self._json(403,{'error':'Localhost-only service.'})
                    return False
            elif not hmac.compare_digest(self.headers.get('X-Builder-Token',''),access_token):
                self._json(401,{'error':'Access token required.'})
                return False
            if user_auth and (path.startswith('/api/') or path=='/metrics') and path not in ('/api/auth/login','/api/health'):
                token=self._cookie_token()
                write=self.command in ('POST','PUT','DELETE','PATCH')
                csrf=self.headers.get('X-CSRF-Token') if write else None
                user=user_auth.session(token,csrf,write=write)
                if user is None:
                    self._json(401,{'error':'Sign in required or CSRF token missing.'})
                    return False
                self._principal=user
                try:self._require_scope(path,write=write)
                except AccessDenied as exc:
                    self._json(403,{'error':str(exc)})
                    return False
            return True

        def _json(self,status,value):
            encoded=json.dumps(value,ensure_ascii=False).encode('utf-8')
            self.send_response(status)
            self.send_header('Content-Type','application/json; charset=utf-8')
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Content-Length',str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def _binary(self, status, raw, mime, filename=None, preview=False):
            self.send_response(status)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if filename:
                self.send_header('Content-Disposition', 'attachment; filename="' + filename + '"')
            if preview:
                self.send_header('Content-Security-Policy',
                    "default-src 'none'; style-src 'self'; img-src 'self' data:; "
                    "script-src 'none'; form-action 'none'; object-src 'none'; "
                    "base-uri 'none'; frame-ancestors 'self'; sandbox allow-same-origin")
            self.end_headers()
            self.wfile.write(raw)

        def _file(self,filename,mime):
            data=(ROOT/'static'/filename).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type',mime)
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
                             "style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
            self.send_header('X-Frame-Options','DENY')
            self.send_header('Content-Length',str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _body(self):
            if self.headers.get('Content-Type','').split(';')[0] != 'application/json':
                raise StoreError('Expected Content-Type: application/json.')
            try: size=int(self.headers.get('Content-Length','0'))
            except ValueError: raise StoreError('Invalid content length.')
            if size<=0 or size>MAX_POST: raise StoreError('Invalid request size.')
            try: data=json.loads(self.rfile.read(size))
            except (ValueError,UnicodeDecodeError): raise StoreError('Invalid JSON request.')
            if not isinstance(data,dict): raise StoreError('Expected JSON object.')
            return data

        def _handle_error(self,exc):
            if isinstance(exc,StoreError): self._json(400,{'error':str(exc)})
            elif isinstance(exc,(TypeError,ValueError,KeyError)):
                self._json(400,{'error':f'Invalid request: {str(exc)[:250]}'})
            else:
                LOG.exception('API failure')
                self._json(500,{'error':'Internal server error; see server logs.'})

        def _origin_ok(self):
            origin=self.headers.get('Origin')
            if origin:
                parsed=urlparse(origin)
                expected=self.headers.get('Host','')
                if parsed.scheme not in ('http','https') or parsed.netloc.lower()!=expected.lower():
                    self._json(403,{'error':'Cross-origin writes are not permitted.'})
                    return False
            return True

        def do_PUT(self):
            if not self._authorized() or not self._origin_ok():return
            # A 1 GiB upload must have a known length. Never accept indefinite streams.
            if self.headers.get('Transfer-Encoding'):
                self._json(400,{'error':'Transfer encoding is not supported for file uploads.'})
                return
            try:
                length=int(self.headers.get('Content-Length',''))
            except (TypeError,ValueError):
                self._json(411,{'error':'File upload requires Content-Length.'})
                return
            if not 1<=length<=MAX_ARTIFACT_BYTES:
                self._json(413,{'error':'File must be between 1 byte and 1 GiB.'})
                self.close_connection=True
                return
            parsed=urlparse(self.path);path=parsed.path
            params=parse_qs(parsed.query,keep_blank_values=True)
            filename=params.get('name',[''])[0]
            try:
                self.connection.settimeout(600)
                if re.fullmatch(r'/api/projects/[a-f0-9]{12}/artifacts/upload',path):
                    pid=path.split('/')[3]
                    result=save_artifact_stream(store,pid,filename,self.rfile,length)
                elif re.fullmatch(r'/api/meetings/[a-f0-9]{12}/artifacts/upload',path):
                    mid=path.split('/')[3]
                    result=save_meeting_artifact_stream(store,mid,filename,self.rfile,length)
                else:
                    self._json(404,{'error':'Upload endpoint not found.'})
                    self.close_connection=True
                    return
                self._json(201,result)
            except (TimeoutError,ConnectionError,OSError) as exc:
                self.close_connection=True
                self._handle_error(StoreError('File upload was interrupted; retry.'))
            except Exception as exc:
                self.close_connection=True
                self._handle_error(exc)

        def do_GET(self):
            if not self._authorized(): return
            url=urlparse(self.path); path=url.path
            try:
                if path=='/': self._file('index.html','text/html; charset=utf-8')
                elif path=='/app.js': self._file('app.js','application/javascript; charset=utf-8')
                elif path=='/app.css': self._file('app.css','text/css; charset=utf-8')
                elif path=='/icons.svg': self._file('icons.svg','image/svg+xml')
                elif path=='/api/auth/me':
                    self._json(200,{'username':self._principal['username'], 'admin':self._principal['admin']}
                               if self._principal else {'username':'Local operator','admin':True,'single_user':True})
                elif path=='/api/metrics':
                    self._json(200,metrics.snapshot())
                elif path=='/api/system/capacity':
                    self._json(200,capacity(store))
                elif path=='/metrics':
                    raw=metrics.prometheus().encode('utf-8')
                    self._binary(200,raw,'text/plain; version=0.0.4')
                elif path=='/api/health':
                    self._json(200,{'status':'ok','llm':'OpenAI GPT','offline_demo':engine.demo,
                                    'api_key_configured':bool(engine.client.key) if hasattr(engine.client,'key') else False,
                                    'multiuser_required':bool(user_auth)})
                elif path=='/api/system/readiness':
                    if not is_loopback: raise StoreError('Readiness inspection requires localhost.')
                    self._json(200,readiness(Path(store.filename).parent))
                elif path=='/api/architecture': self._json(200,blueprint())
                elif path=='/api/settings':
                    self._json(200,runtime.snapshot() if runtime else settings())
                elif path=='/api/providers/openai/models':
                    if not is_loopback: raise StoreError('OpenAI configuration requires local access.')
                    if runtime is None: raise StoreError('Runtime settings not configured.')
                    self._json(200,{'models':runtime.available_models()})
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/website/status',path):
                    self._json(200,website_status(store,path.split('/')[3]))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/website/preview/(index\.html|about\.html|contact\.html|styles\.css)',path):
                    parts=path.split('/')
                    name=parts[-1]
                    raw=website_file(store,parts[3],name)
                    mime='text/css; charset=utf-8' if name.endswith('.css') else 'text/html; charset=utf-8'
                    self._binary(200,raw,mime,preview=True)
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/website/download',path):
                    pid=path.split('/')[3]
                    self._binary(200,website_zip(store,pid),'application/zip',filename='website-'+pid+'.zip')
                elif path=='/api/projects':
                    rows=store.projects()
                    if user_auth:rows=user_auth.visible_projects(self._principal,rows)
                    self._json(200,{'projects':rows})
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/workspace',path):
                    pid=path.split('/')[3]
                    self._json(200,{'files':list_files(store,pid),'runs':workspace_runs(store,pid),
                                   'host_python':sys.executable,'host_risk':'Not sandboxed; administrator approval required per run.'})
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/workspace/file',path):
                    pid=path.split('/')[3]
                    file_path=parse_qs(url.query).get('path',[''])[0]
                    self._json(200,read_file(store,pid,file_path))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/releases',path):
                    self._json(200,{'releases':project_releases(store,path.split('/')[3])})
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/rag/status',path):
                    self._json(200,rag_status_large(store,path.split('/')[3]))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/artifacts/[a-f0-9]{32}/text',path):
                    if not is_loopback: raise StoreError('File text access requires the local UI.')
                    parts=path.split('/');pid,aid=parts[3],parts[5]
                    q=parse_qs(urlparse(self.path).query)
                    page=int(q['page'][0]) if q.get('page') else None
                    offset=int(q.get('offset',['0'])[0])
                    self._json(200,read_artifact_excerpt(store,pid,aid,page=page,offset=offset,
                        query=q.get('query',[''])[0]))
                elif re.fullmatch(r'/api/meetings/[a-f0-9]{12}/artifacts/[a-f0-9]{32}/text',path):
                    if not is_loopback: raise StoreError('File text access requires the local UI.')
                    parts=path.split('/');mid,aid=parts[3],parts[5]
                    q=parse_qs(urlparse(self.path).query)
                    page=int(q['page'][0]) if q.get('page') else None
                    offset=int(q.get('offset',['0'])[0])
                    self._json(200,read_meeting_artifact_excerpt(store,mid,aid,page=page,offset=offset,
                        query=q.get('query',[''])[0]))
                elif re.fullmatch(r'/api/meetings/[a-f0-9]{12}/artifacts',path):
                    mid=path.split('/')[3]
                    self._json(200,{'artifacts':meeting_artifacts(store,mid)})
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/artifacts',path):
                    pid=path.split('/')[3]
                    self._json(200,{'artifacts':artifacts(store,pid)})
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/handoffs',path):
                    pid=path.split('/')[3]
                    self._json(200,{'handoffs':list_handoffs(store,pid)})
                elif path=='/api/meetings':
                    params=parse_qs(url.query)
                    project_id=params.get('project_id',[None])[0]
                    if user_auth and project_id:user_auth.require(self._principal,project_id,'viewer')
                    rooms=list_meetings(store,project_id)
                    if user_auth:
                        visible=[]
                        for room in rooms:
                            if self._principal['admin'] or (room.get('project_id') and user_auth.role(self._principal,room['project_id'])):
                                visible.append(room)
                        rooms=visible
                    self._json(200,{'meetings':rooms})
                elif re.fullmatch(r'/api/meetings/[a-zA-Z0-9]+',path):
                    self._json(200,meeting(store,path.rsplit('/',1)[-1]))
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/usage',path):
                    pid=path.split('/')[3];store.project(pid)
                    self._json(200,store.llm_usage(pid))
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/activity',path):
                    pid=path.split('/')[3]
                    self._json(200,store.activity(pid))
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+',path):
                    pid=path.rsplit('/',1)[-1]
                    self._json(200,{'project':store.project(pid),'messages':store.messages(pid),
                                    'events':store.events(pid), 'meetings':list_meetings(store,pid)})
                else: self._json(404,{'error':'Endpoint not found.'})
            except Exception as exc: self._handle_error(exc)

        def do_POST(self):
            if not self._authorized(): return
            if not self._origin_ok():return
            path=urlparse(self.path).path
            try:
                data=self._body()
                if path=='/api/auth/login':
                    if not user_auth:raise StoreError('Multiuser login is disabled.')
                    session=user_auth.login(data.get('username'),data.get('password'))
                    encoded=json.dumps({'csrf':session['csrf'],'username':session['username'],
                                        'admin':session['admin'],'expires_at':session['expires_at']}).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type','application/json')
                    self.send_header('Cache-Control','no-store')
                    self.send_header('Set-Cookie','builder_session='+session['token']+'; HttpOnly; SameSite=Strict; Path=/; Max-Age=28800')
                    self.send_header('Content-Length',str(len(encoded)))
                    self.end_headers();self.wfile.write(encoded)
                elif path=='/api/auth/logout':
                    if user_auth:user_auth.logout(self._cookie_token())
                    self.send_response(200)
                    self.send_header('Set-Cookie','builder_session=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0')
                    body=b'{"logged_out":true}'
                    self.send_header('Content-Type','application/json')
                    self.send_header('Content-Length',str(len(body)))
                    self.end_headers();self.wfile.write(body)
                elif path=='/api/settings':
                    if runtime is None or not is_loopback: raise StoreError('Settings can only be changed on localhost.')
                    response=runtime.update(data);engine.apply_runtime();self._json(200,response)
                elif path=='/api/settings/key':
                    if runtime is None or not is_loopback: raise StoreError('Key updates require localhost.')
                    response=runtime.save_key(data.get('api_key'));engine.apply_runtime();self._json(200,response)
                elif path=='/api/settings/key/clear':
                    if runtime is None or not is_loopback: raise StoreError('Key removal requires localhost.')
                    response=runtime.clear_key();engine.apply_runtime();self._json(200,response)
                elif path=='/api/projects':
                    if data.get('kickoff',False) and not engine.demo and data.get('intake',{}).get('provider_consent') is not True:
                        raise StoreError('Confirm provider consent before starting GPT workers.')
                    project=store.create_project(data.get('name',''),data.get('brief',''),intake=data.get('intake'))
                    if user_auth:user_auth.own(self._principal,project['id'])
                    if project['intake']['product_type']=='website':
                        # The scoped static website can be built without a GPT key or code runner.
                        build_website(store,project['id'],data.get('website',{}))
                    if data.get('kickoff',False):
                        engine.kickoff(project['id'])
                        auto_index(project['id'])
                    self._json(201,project)
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/workspace/file',path):
                    pid=path.split('/')[3]
                    result=write_file(store,pid,data.get('path'),data.get('content'),
                                      actor=self._principal['username'] if self._principal else 'human',
                                      expected_sha256=data.get('expected_sha256'))
                    self._json(201,result)
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/workspace/run',path):
                    if not is_loopback:raise StoreError('Windows execution is localhost-only.')
                    if user_auth and not self._principal['admin']:
                        raise AccessDenied('Only an administrator can authorize host Python runs.')
                    pid=path.split('/')[3]
                    self._json(200,run_windows_python(store,pid,data.get('filename'),
                        confirmation=data.get('confirmation'),timeout=data.get('timeout',90),
                        actor=self._principal['username'] if self._principal else 'human'))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/workspace/docker-check',path):
                    if not is_loopback:raise StoreError('Docker testing is localhost-only.')
                    pid=path.split('/')[3]
                    if data.get('confirmation')!=store.project(pid)['name']:
                        raise StoreError('Confirm by typing the exact project name.')
                    self._json(200,docker_check(store,pid,data.get('command'),actor='human'))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/releases/prepare',path):
                    self._json(201,prepare_release(store,path.split('/')[3]))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/releases/[a-f0-9]{16}/approve',path):
                    parts=path.split('/')
                    self._json(200,approve_release(store,parts[3],parts[5],
                        self._principal['username'] if self._principal else 'human',data.get('confirmation')))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/releases/[a-f0-9]{16}/stage',path):
                    parts=path.split('/')
                    self._json(200,stage_release(store,parts[3],parts[5]))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/website/build',path):
                    if not is_loopback: raise StoreError('Website generation requires localhost.')
                    self._json(201,build_website(store,path.split('/')[3],data))
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/rag/index',path):
                    pid=path.split('/')[3]
                    if not is_loopback:raise StoreError('Local indexing requires localhost access.')
                    self._json(202,start_large_index_job(store,pid,runtime.key if runtime else None,
                        embedding_model=runtime.data['embedding_model'] if runtime else DEFAULT_MODEL,
                        ocr_mode=runtime.data['ocr_mode'] if runtime else 'auto',
                        ocr_model=runtime.model_for('rag-engineer') if runtime else None,
                        resume=bool(data.get('resume',False))))
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/rag/cancel',path):
                    if not is_loopback:raise StoreError('Local indexing requires localhost access.')
                    self._json(200,cancel_large_index(store,path.split('/')[3]))
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/rag/ask',path):
                    pid=path.split('/')[3]
                    rag_access(pid)
                    idx=rag_status_large(store,pid)
                    if idx.get('indexed') and idx['embedding_model']!=runtime.data['embedding_model']:
                        raise StoreError('Local embedding model changed; rebuild the index before asking.')
                    self._json(200,answer_large_question(store,pid,data.get('question'),runtime.key,
                                      runtime.model_for('rag-engineer')))
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/delete',path):
                    pid=path.split('/')[3]
                    # Store's tombstone-based deletion also purges execution files,
                    # immutable versions, staged releases and project membership rows.
                    self._json(200,store.delete_project(pid,data.get('confirmation')))
                elif re.fullmatch(r'/api/projects/[a-f0-9]{12}/artifacts/[a-f0-9]{32}/text',path):
                    if not is_loopback: raise StoreError('File text access requires the local UI.')
                    parts=path.split('/');pid,aid=parts[3],parts[5]
                    q=parse_qs(urlparse(self.path).query)
                    page=int(q['page'][0]) if q.get('page') else None
                    offset=int(q.get('offset',['0'])[0])
                    self._json(200,read_artifact_excerpt(store,pid,aid,page=page,offset=offset,
                        query=q.get('query',[''])[0]))
                elif re.fullmatch(r'/api/meetings/[a-f0-9]{12}/artifacts/[a-f0-9]{32}/text',path):
                    if not is_loopback: raise StoreError('File text access requires the local UI.')
                    parts=path.split('/');mid,aid=parts[3],parts[5]
                    q=parse_qs(urlparse(self.path).query)
                    page=int(q['page'][0]) if q.get('page') else None
                    offset=int(q.get('offset',['0'])[0])
                    self._json(200,read_meeting_artifact_excerpt(store,mid,aid,page=page,offset=offset,
                        query=q.get('query',[''])[0]))
                elif re.fullmatch(r'/api/meetings/[a-f0-9]{12}/artifacts',path):
                    mid=path.split('/')[3]
                    self._json(200,{'artifacts':meeting_artifacts(store,mid)})
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/artifacts',path):
                    pid=path.split('/')[3]
                    try: raw=base64.b64decode(data.get('base64',''),validate=True)
                    except (ValueError,binascii.Error): raise StoreError('Invalid base64 artifact content.')
                    self._json(201,save_artifact(store,pid,data.get('name',''),raw))
                elif path=='/api/meetings':
                    if user_auth and data.get('project_id'):
                        user_auth.require(self._principal,data['project_id'],'editor')
                    elif user_auth and not self._principal['admin']:
                        raise AccessDenied('Standalone meetings require an administrator.')
                    start=int(data['starts_at']) if data.get('starts_at') is not None else None
                    self._json(201,create_meeting(store,topic=data.get('topic',''),participants=data.get('participants',[]),
                                                 requested_by='human',project_id=data.get('project_id') or None,
                                                 starts_at=start,start_now=start is None))
                elif re.fullmatch(r'/api/meetings/[a-zA-Z0-9]+/(start|interrupt|speak|yield|end|cancel)',path):
                    parts=path.split('/'); mid=parts[3]; action=parts[4]
                    if action=='start': start_meeting(store,mid); result=meeting(store,mid)
                    elif action=='interrupt': result=take_floor(store,mid)
                    elif action=='speak': result=take_floor(store,mid,human_message=data.get('message'),
                                                           target=data.get('target','all'))
                    elif action=='yield': result=yield_floor(store,mid)
                    elif action=='end': result=end_meeting(store,mid)
                    else: result=cancel_meeting(store,mid,data.get('reason'))
                    self._json(200,result)
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/(message|kickoff|approve|resume|retry)',path):
                    parts=path.split('/'); pid=parts[3]; action=parts[4]
                    if action=='message':
                        result=store.send(pid,'human',data.get('recipient',''),data.get('message',''))
                    elif action=='kickoff':
                        result=engine.kickoff(pid)
                        auto_index(pid)
                    elif action=='approve':
                        store.approve(pid,data.get('confirmation'),data.get('operator',''));result=store.project(pid)
                    elif action=='resume': store.resume(pid);result=store.project(pid)
                    else: store.retry(data.get('message_id',''),project_id=pid);result={'retried':True}
                    self._json(200,result)

                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/cancel-tasks',path):
                    pid=path.split('/')[3]
                    self._json(200,store.cancel_tasks(pid,data.get('task_ids'),data.get('reason')))
                elif re.fullmatch(r'/api/tasks/[a-f0-9]+/cancel',path):
                    mid=path.split('/')[3]
                    self._json(200,store.cancel(mid,data.get('reason')))
                else: self._json(404,{'error':'Endpoint not found.'})
            except Exception as exc: self._handle_error(exc)

    # Class methods keep the application compatible with the existing test harness.
    for method_name in ('do_GET','do_POST','do_PUT'):
        original=getattr(Handler,method_name)
        def wrapped(self,*args,_original=original,_method=method_name,**kwargs):
            started=metrics.enter()
            try:return _original(self,*args,**kwargs)
            finally:
                metrics.leave(_method[3:],self._metric_class(urlparse(self.path).path),
                              self._response_status,started)
        setattr(Handler,method_name,wrapped)
    return ThreadingHTTPServer((host,port),Handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description='OpenAI GPT multi-agent product builder')
    parser.add_argument('--demo',action='store_true',help='Offline synthetic responses, visibly labeled; no GPT calls')
    parser.add_argument('--port',type=int,default=int(os.getenv('BUILDER_PORT','8766')))
    parser.add_argument('--host',default=os.getenv('BUILDER_HOST','127.0.0.1'))
    parser.add_argument('--db',default=os.getenv('BUILDER_DB',str(ROOT/'data'/'builder.sqlite3')))
    parser.add_argument('--multiuser',action='store_true',help='Require sign-in and project ACLs (localhost only)')
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    store=Store(args.db)
    runtime=RuntimeSettings(Path(args.db).parent)
    if args.demo and runtime.data['mode'] != 'demo':
        runtime.update({'mode':'demo'})
    engine=Engine(store,demo=True,runtime=runtime)
    server=make_server(store,engine,args.host,args.port,os.getenv('BUILDER_ACCESS_TOKEN'),runtime,
                       enable_multiuser=args.multiuser)
    engine.start()
    print(f'AI Product Builder: http://{args.host}:{server.server_address[1]}/  | '
          f'{"OFFLINE DEMO (not GPT)" if engine.demo else "OPENAI GPT"}',flush=True)
    try: server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt: pass
    finally:
        server.shutdown(); server.server_close(); engine.stop()
    return 0


if __name__=='__main__': sys.exit(main())
