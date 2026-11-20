"""Regression checks for opt-in Windows-host execution, sessions and releases."""
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from auth_security import UserAuth
from engine import Engine
from execution_workspace import (write_file, list_files, run_windows_python, workspace_root, runs)
from release_management import prepare, approve, stage
from server import make_server
from store import Store, StoreError
from website_builder import build_website


class HostRunnerTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.store=Store(str(Path(self.t.name)/'builder.sqlite3'))
        self.project=self.store.create_project('Windows RAG','RAG with tests')
        self.pid=self.project['id']
        write_file(self.store,self.pid,'app.py',"import os\nprint('HOST_OK')\nprint('secret inherited', 'OPENAI_API_KEY' in os.environ)\n")

    def test_host_python_runs_same_interpreter_and_scrubs_api_secrets(self):
        with patch.dict(os.environ,{'OPENAI_API_KEY':'THIS_IS_A_SENTINEL_SECRET'}):
            result=run_windows_python(self.store,self.pid,'app.py',confirmation='Windows RAG')
        self.assertEqual(result['status'],'passed',result['output'])
        self.assertIn('HOST_OK',result['output'])
        self.assertIn('secret inherited False',result['output'])
        self.assertNotIn('THIS_IS_A_SENTINEL_SECRET',result['output'])
        self.assertEqual(result['interpreter'],str(Path(sys.executable).resolve()))
        self.assertTrue(runs(self.store,self.pid))

    def test_unittest_command_uses_host_python_and_is_recorded(self):
        result=run_windows_python(self.store,self.pid,'compile',confirmation='Windows RAG')
        self.assertEqual(result['status'],'passed',result['output'])
        self.assertEqual(result['command'],'windows:compile')

    def test_script_execution_needs_no_approval(self):
        result=run_windows_python(self.store,self.pid,'app.py')
        self.assertEqual(result['status'],'passed')
        self.assertTrue(runs(self.store,self.pid))

    def test_arbitrary_exec_or_path_traversal_rejected(self):
        for name in ('../evil.py','C:/Windows/system32/evil.py','evil.ps1','python -c print(1)', 'tests/../evil.py'):
            with self.subTest(name=name),self.assertRaises(StoreError):
                run_windows_python(self.store,self.pid,name,confirmation='Windows RAG')
        self.assertFalse(runs(self.store,self.pid))

    def test_host_runner_never_falls_back_to_unsafe_executable(self):
        with self.assertRaisesRegex(StoreError,'Python interpreter'):
            run_windows_python(self.store,self.pid,'app.py',confirmation='Windows RAG',python_executable=__file__)


class HostHttpTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.store=Store(str(Path(self.t.name)/'builder.sqlite3'))
        self.project=self.store.create_project('Host API','Host API tests')
        self.pid=self.project['id']
        self.engine=Engine(self.store,client=object(),demo=True)
        self.srv=make_server(self.store,self.engine,port=0)
        self.thread=threading.Thread(target=self.srv.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self._stop)
        self.base='http://127.0.0.1:'+str(self.srv.server_address[1])

    def _stop(self):
        self.srv.shutdown();self.srv.server_close();self.thread.join(3)

    def post(self,path,data):
        req=Request(self.base+path,data=json.dumps(data).encode(),method='POST',headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=10) as resp:return resp.status,json.load(resp)

    def test_http_workspace_save_run_and_metrics(self):
        path=f'/api/projects/{self.pid}/workspace'
        status,out=self.post(path+'/file',{'path':'main.py','content':'print("run through browser")'})
        self.assertEqual(status,201)
        status,run=self.post(path+'/run',{'filename':'main.py','confirmation':'Host API'})
        self.assertEqual(status,200)
        self.assertIn('run through browser',run['output'])
        with urlopen(self.base+path,timeout=10) as response: ledger=json.load(response)
        self.assertEqual(len(ledger['files']),1)
        self.assertEqual(ledger['runs'][0]['status'],'passed')
        with urlopen(self.base+'/api/metrics') as response:m=json.load(response)
        self.assertGreater(m['requests'][0]['count'],0)
        with urlopen(self.base+'/api/system/capacity') as response:capacity=json.load(response)
        self.assertGreater(capacity['disk_free_bytes'],0)

    def test_http_wrong_confirmation_ignored(self):
        path=f'/api/projects/{self.pid}/workspace'
        self.post(path+'/file',{'path':'main.py','content':'print("no approval")'})
        status,out=self.post(path+'/run',{'filename':'main.py','confirmation':'another name'})
        self.assertEqual(status,200)
        self.assertIn('no approval',out['output'])

    def test_delete_cleans_workspace_and_releases(self):
        path=f'/api/projects/{self.pid}/workspace'
        self.post(path+'/file',{'path':'main.py','content':'print(1)'})
        self.post(f'/api/projects/{self.pid}/delete',{'confirmation':'Host API'})
        self.assertFalse((Path(self.t.name)/'workspaces'/self.pid).exists())


class MultiuserHttpTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.store=Store(str(Path(self.t.name)/'builder.sqlite3'))
        self.auth=UserAuth(self.store)
        self.auth.add_user('rootoperator','verylongsecretpass1',admin=True)
        self.auth.add_user('alice','verylongsecretpass2')
        self.auth.add_user('bob','verylongsecretpass3')
        self.project=self.store.create_project('Project Alpha','Private file')
        self.pid=self.project['id']
        self.auth.grant(self.pid,'alice','editor')
        self.auth.grant(self.pid,'bob','viewer')
        self.secret=self.store.create_project('Project Secret','Hidden project')
        self.engine=Engine(self.store,client=object(),demo=True)
        self.srv=make_server(self.store,self.engine,port=0,enable_multiuser=True)
        self.thread=threading.Thread(target=self.srv.serve_forever,daemon=True);self.thread.start()
        self.addCleanup(self._stop)
        self.base='http://127.0.0.1:'+str(self.srv.server_address[1])

    def _stop(self):
        self.srv.shutdown();self.srv.server_close();self.thread.join(3)

    def login(self,username,password):
        req=Request(self.base+'/api/auth/login',data=json.dumps({'username':username,'password':password}).encode(),
                    method='POST',headers={'Content-Type':'application/json'})
        with urlopen(req,timeout=10) as resp:
            data=json.load(resp)
            return {'Cookie':resp.headers['Set-Cookie'].split(';',1)[0], 'X-CSRF-Token':data['csrf']}

    def call(self,path,headers=None,body=None):
        req=Request(self.base+path,method='POST' if body is not None else 'GET',
                    data=json.dumps(body).encode() if body is not None else None,
                    headers=({**(headers or {}),'Content-Type':'application/json'} if body is not None else headers or {}))
        with urlopen(req,timeout=10) as resp:return json.load(resp)

    def test_no_session_or_csrf_is_denied(self):
        with self.assertRaises(HTTPError) as ctx:self.call('/api/projects')
        self.assertEqual(ctx.exception.code,401)
        alice=self.login('alice','verylongsecretpass2')
        with self.assertRaises(HTTPError) as ctx:self.call(f'/api/projects/{self.pid}/workspace/file',{'Cookie':alice['Cookie']},
                    {'path':'app.py','content':'print(1)'})
        self.assertEqual(ctx.exception.code,401)

    def test_project_visibility_permissions_and_admin_only_host_runner(self):
        alice=self.login('alice','verylongsecretpass2')
        projects=self.call('/api/projects',alice)['projects']
        self.assertEqual([x['id'] for x in projects],[self.pid])
        with self.assertRaises(HTTPError) as denied:self.call(f'/api/projects/{self.pid}/workspace/file',alice,{'path':'app.py','content':'print(1)'})
        self.assertEqual(denied.exception.code,403)
        admin=self.login('rootoperator','verylongsecretpass1')
        self.call(f'/api/projects/{self.pid}/workspace/auto',admin,{'enabled':False})
        self.call(f'/api/projects/{self.pid}/workspace/file',alice,{'path':'app.py','content':'print(1)'})
        self.call(f'/api/projects/{self.pid}/workspace/auto',admin,{'enabled':True})
        with self.assertRaises(HTTPError) as denied:self.call(f'/api/projects/{self.pid}/workspace/run',alice,
                                         {'filename':'app.py','confirmation':'Project Alpha'})
        self.assertEqual(denied.exception.code,403)
        bob=self.login('bob','verylongsecretpass3')
        with self.assertRaises(HTTPError) as denied:self.call(f'/api/projects/{self.pid}/workspace/file',bob,
                                         {'path':'a.py','content':'print(1)'})
        self.assertEqual(denied.exception.code,403)
        with self.assertRaises(HTTPError) as denied:self.call(f'/api/projects/{self.secret["id"]}',alice)
        self.assertEqual(denied.exception.code,403)
        admin=self.login('rootoperator','verylongsecretpass1')
        result=self.call(f'/api/projects/{self.pid}/workspace/run',admin,
                         {'filename':'app.py','confirmation':'Project Alpha'})
        self.assertEqual(result['status'],'passed')


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.store=Store(str(Path(self.t.name)/'builder.sqlite3'))
        self.project=self.store.create_project('Site Release','Static website',intake={'product_type':'website'})
        self.pid=self.project['id'];build_website(self.store,self.pid,{})

    def test_release_approval_required_and_staging_is_local_only(self):
        candidate=prepare(self.store,self.pid)
        self.assertEqual(candidate['status'],'pending_approval')
        with self.assertRaises(StoreError):stage(self.store,self.pid,candidate['id'])
        with self.assertRaises(StoreError):approve(self.store,self.pid,candidate['id'],'operator','invalid')
        approve(self.store,self.pid,candidate['id'],'operator','Site Release')
        result=stage(self.store,self.pid,candidate['id'])
        self.assertEqual(result['status'],'staged')
        self.assertTrue(Path(result['path']).joinpath('index.html').is_file())

if __name__=='__main__':unittest.main()

class AgentSourceHandoffTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.store=Store(str(Path(self.t.name)/'builder.sqlite3'))
        self.pid=self.store.create_project('Agent Code','Make a Python program')['id']

    def test_agent_creates_source_and_next_agent_gets_pinned_handoff(self):
        class SourceClient:
            def generate(self,role,model,instructions,context):
                if role=='dev-1':
                    return {'reply':'Propose a file for the test runner.',
                      'workspace_files':[{'path':'main.py','content':'print("generated code")\n','expected_sha256':None}],
                      'outgoing':[{'recipient':'tester-1','message':'Review main.py'}],
                      'needs_human':False,'meeting_requests':[],'file_requests':[]}
                return {'reply':'Received verified source.', 'workspace_files':[], 'outgoing':[],
                        'needs_human':False,'meeting_requests':[],'file_requests':[]}
        task=self.store.send(self.pid,'human','dev-1','Write a simple Python file')
        engine=Engine(self.store,client=SourceClient(),demo=True)
        self.assertTrue(engine.process_once())
        self.assertEqual(self.store.message(task['id'])['status'],'done')
        from execution_workspace import read_file, read_version
        original=read_file(self.store,self.pid,'main.py')
        self.assertIn('generated code',original['content'])
        self.assertEqual(read_version(self.store,self.pid,original['sha256']),original['content'].encode())
        from handoffs import inspect_handoff
        next_task=[x for x in self.store.messages(self.pid) if x['recipient']=='tester-1'][0]
        manifest=inspect_handoff(self.store,next_task['handoff_id'],self.pid)['manifest']
        self.assertEqual(manifest['workspace_files'][0]['sha256'],original['sha256'])
        self.assertIn('main.py',self.store.message(task['id'])['reply'])
        self.assertIn('automatic runs are queued',self.store.message(task['id'])['reply'])
        # The next change creates a new immutable snapshot without invalidating
        # the handoff pinned to the old revision.
        write_file(self.store,self.pid,'main.py','print("new revision")\n',expected_sha256=original['sha256'])
        old_manifest=inspect_handoff(self.store,next_task['handoff_id'],self.pid)['manifest']
        self.assertEqual(old_manifest['workspace_files'][0]['sha256'],original['sha256'])
        self.assertTrue(engine.process_once())
        self.assertEqual(self.store.message(next_task['id'])['status'],'done')

    def test_rejects_workspace_source_traversal_without_failing_response(self):
        class BadClient:
            def generate(self,*_):
                return {'reply':'Cannot write malicious file.',
                        'workspace_files':[{'path':'../../evil.py','content':'print(1)','expected_sha256':None}],
                        'outgoing':[],'meeting_requests':[],'needs_human':False,'file_requests':[]}
        task=self.store.send(self.pid,'human','dev-1','Write code')
        self.assertTrue(Engine(self.store,client=BadClient(),demo=True).process_once())
        self.assertEqual(self.store.message(task['id'])['status'],'done')
        self.assertIn('not saved',self.store.message(task['id'])['reply'])
        self.assertFalse(list_files(self.store,self.pid))
