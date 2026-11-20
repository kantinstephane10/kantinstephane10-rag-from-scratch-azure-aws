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
from urllib.parse import urlparse, parse_qs

from architecture import blueprint
from config import settings
from engine import Engine
from meetings import (create_meeting, meeting, list_meetings, start_meeting,
                      take_floor, yield_floor, end_meeting, cancel_meeting)
from roles import ROLES
from store import Store, StoreError
from handoffs import save_artifact, artifacts, list_handoffs, read_artifact

ROOT = Path(__file__).resolve().parent
LOG = logging.getLogger(__name__)
MAX_POST = 1_400_000  # 1 MB artifact after base64 transport


def make_server(store,engine,host='127.0.0.1',port=8766,access_token=None):
    is_loopback = host in ('127.0.0.1','localhost','::1')
    if not is_loopback and not access_token:
        raise ValueError('Non-loopback binding requires BUILDER_ACCESS_TOKEN and TLS reverse proxy.')

    class Handler(BaseHTTPRequestHandler):
        server_version='AIProductBuilder/0.4'

        def log_message(self,format,*args):
            LOG.info('%s - %s',self.address_string(),format%args)

        def _authorized(self):
            # DNS rebinding hardening in local mode. Never allow cross-host access.
            if is_loopback:
                hostname=self.headers.get('Host','').split(':')[0].lower()
                if hostname not in ('localhost','127.0.0.1','[::1]'):
                    self._json(403,{'error':'Localhost-only service.'})
                    return False
            elif not hmac.compare_digest(self.headers.get('X-Builder-Token',''),access_token):
                self._json(401,{'error':'Access token required.'})
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

        def _file(self,filename,mime):
            data=(ROOT/'static'/filename).read_bytes()
            self.send_response(200)
            self.send_header('Content-Type',mime)
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Cache-Control','no-store')
            self.send_header('Content-Security-Policy',
                             "default-src 'self'; connect-src 'self'; img-src 'self' data:; "
                             "style-src 'self'; script-src 'self'; object-src 'none'; base-uri 'none'")
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

        def do_GET(self):
            if not self._authorized(): return
            url=urlparse(self.path); path=url.path
            try:
                if path=='/': self._file('index.html','text/html; charset=utf-8')
                elif path=='/app.js': self._file('app.js','application/javascript; charset=utf-8')
                elif path=='/app.css': self._file('app.css','text/css; charset=utf-8')
                elif path=='/icons.svg': self._file('icons.svg','image/svg+xml')
                elif path=='/api/health':
                    self._json(200,{'status':'ok','llm':'OpenAI GPT','offline_demo':engine.demo,
                                    'api_key_configured':bool(engine.client.key) if hasattr(engine.client,'key') else False})
                elif path=='/api/architecture': self._json(200,blueprint())
                elif path=='/api/settings': self._json(200,settings())
                elif path=='/api/projects': self._json(200,{'projects':store.projects()})
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/artifacts',path):
                    pid=path.split('/')[3]
                    self._json(200,{'artifacts':artifacts(store,pid)})
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/handoffs',path):
                    pid=path.split('/')[3]
                    self._json(200,{'handoffs':list_handoffs(store,pid)})
                elif path=='/api/meetings':
                    params=parse_qs(url.query)
                    self._json(200,{'meetings':list_meetings(store,params.get('project_id',[None])[0])})
                elif re.fullmatch(r'/api/meetings/[a-zA-Z0-9]+',path):
                    self._json(200,meeting(store,path.rsplit('/',1)[-1]))
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
            origin=self.headers.get('Origin')
            if origin:
                parsed=urlparse(origin)
                # Prevent a malicious web page from issuing state-changing requests to localhost.
                expected=self.headers.get('Host','')
                if parsed.scheme not in ('http','https') or parsed.netloc.lower()!=expected.lower():
                    self._json(403,{'error':'Cross-origin writes are not permitted.'})
                    return
            path=urlparse(self.path).path
            try:
                data=self._body()
                if path=='/api/projects':
                    if data.get('kickoff',False) and not engine.demo and data.get('intake',{}).get('provider_consent') is not True:
                        raise StoreError('Confirm provider consent before starting GPT workers.')
                    project=store.create_project(data.get('name',''),data.get('brief',''),intake=data.get('intake'))
                    if data.get('kickoff',False): engine.kickoff(project['id'])
                    self._json(201,project)
                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/artifacts',path):
                    pid=path.split('/')[3]
                    try: raw=base64.b64decode(data.get('base64',''),validate=True)
                    except (ValueError,binascii.Error): raise StoreError('Invalid base64 artifact content.')
                    self._json(201,save_artifact(store,pid,data.get('name',''),raw))
                elif path=='/api/meetings':
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
                    elif action=='kickoff': result=engine.kickoff(pid)
                    elif action=='approve':
                        store.approve(pid,data.get('confirmation'),data.get('operator',''));result=store.project(pid)
                    elif action=='resume': store.resume(pid);result=store.project(pid)
                    else: store.retry(data.get('message_id',''));result={'retried':True}
                    self._json(200,result)

                elif re.fullmatch(r'/api/projects/[a-zA-Z0-9]+/cancel-tasks',path):
                    pid=path.split('/')[3]
                    self._json(200,store.cancel_tasks(pid,data.get('task_ids'),data.get('reason')))
                elif re.fullmatch(r'/api/tasks/[a-f0-9]+/cancel',path):
                    mid=path.split('/')[3]
                    self._json(200,store.cancel(mid,data.get('reason')))
                else: self._json(404,{'error':'Endpoint not found.'})
            except Exception as exc: self._handle_error(exc)

    return ThreadingHTTPServer((host,port),Handler)


def main(argv=None):
    parser=argparse.ArgumentParser(description='OpenAI GPT multi-agent product builder')
    parser.add_argument('--demo',action='store_true',help='Offline synthetic responses, visibly labeled; no GPT calls')
    parser.add_argument('--port',type=int,default=int(os.getenv('BUILDER_PORT','8766')))
    parser.add_argument('--host',default=os.getenv('BUILDER_HOST','127.0.0.1'))
    parser.add_argument('--db',default=os.getenv('BUILDER_DB',str(ROOT/'data'/'builder.sqlite3')))
    args=parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    store=Store(args.db); engine=Engine(store,demo=args.demo)
    server=make_server(store,engine,args.host,args.port,os.getenv('BUILDER_ACCESS_TOKEN'))
    engine.start()
    print(f'AI Product Builder: http://{args.host}:{server.server_address[1]}/  | '
          f'{"OFFLINE DEMO (not GPT)" if args.demo else "OPENAI GPT"}',flush=True)
    try: server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt: pass
    finally:
        server.shutdown(); server.server_close(); engine.stop()
    return 0


if __name__=='__main__': sys.exit(main())
