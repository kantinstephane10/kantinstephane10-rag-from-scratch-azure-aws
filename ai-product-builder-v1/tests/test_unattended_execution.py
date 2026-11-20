"""Unattended host execution regressions. Runs only harmless fixture scripts."""
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from store import Store
from engine import Engine
from execution_workspace import (write_file, auto_enabled, auto_jobs,
    set_auto_enabled, queue_auto_runs, process_auto_job, cancel_auto_jobs,
    recover_auto_jobs, runs, list_files, init_workspace)
from gpt_client import RESPONSE_SCHEMA, parse_response

class UnattendedHostTests(unittest.TestCase):
    def setUp(self):
        self.t=tempfile.TemporaryDirectory();self.addCleanup(self.t.cleanup)
        self.store=Store(str(Path(self.t.name)/'builder.sqlite3'))
        self.pid=self.store.create_project('Unattended project','Write a working script')['id']
        init_workspace(self.store)

    def test_default_enabled_and_executes_agent_entrypoint(self):
        code="print('AUTOMATED_HOST_EXECUTION')\n"
        saved=write_file(self.store,self.pid,'main.py',code,actor='dev-1')
        self.assertTrue(auto_enabled(self.store,self.pid))
        ids=queue_auto_runs(self.store,self.pid,[saved],actor='dev-1')
        self.assertEqual(len(ids),2)
        self.assertTrue(process_auto_job(self.store))
        self.assertTrue(process_auto_job(self.store))
        self.assertFalse(process_auto_job(self.store))
        states={j['command']:j['status'] for j in auto_jobs(self.store,self.pid)}
        self.assertEqual(states,{'compile':'passed','main.py':'passed'})
        self.assertIn('AUTOMATED_HOST_EXECUTION',next(r['output'] for r in runs(self.store,self.pid) if r['command']=='windows:main.py'))

    def test_pause_resume_and_stop_queue(self):
        saved=write_file(self.store,self.pid,'main.py',"print('no user permission')\n",actor='dev-1')
        self.assertEqual(len(queue_auto_runs(self.store,self.pid,[saved])),2)
        set_auto_enabled(self.store,self.pid,False)
        self.assertFalse(auto_enabled(self.store,self.pid))
        self.assertFalse(process_auto_job(self.store))
        set_auto_enabled(self.store,self.pid,True)
        self.assertTrue(process_auto_job(self.store))
        result=cancel_auto_jobs(self.store,self.pid)
        self.assertEqual(result['queued_cancelled'],1)
        self.assertFalse(process_auto_job(self.store))

    def test_duplicate_revision_is_not_reexecuted(self):
        row=write_file(self.store,self.pid,'main.py','print(1)\n')
        first=queue_auto_runs(self.store,self.pid,[row])
        self.assertEqual(len(first),2)
        self.assertEqual(queue_auto_runs(self.store,self.pid,[row]),[])
        self.assertEqual(len(auto_jobs(self.store,self.pid)),2)

    def test_queued_old_revision_is_superseded(self):
        old=write_file(self.store,self.pid,'main.py','print("v1")\n')
        queue_auto_runs(self.store,self.pid,[old])
        write_file(self.store,self.pid,'main.py','print("v2")\n',expected_sha256=old['sha256'])
        self.assertTrue(process_auto_job(self.store))
        self.assertIn(auto_jobs(self.store,self.pid)[0]['status'],('queued','superseded'))
        self.assertEqual(sum(x['status']=='superseded' for x in auto_jobs(self.store,self.pid)),1)
        self.assertFalse(runs(self.store,self.pid))

    def test_non_entrypoint_compiles_but_does_not_run_module_import_side_effect(self):
        row=write_file(self.store,self.pid,'libmath.py',"raise RuntimeError('do not execute module directly')\n")
        queue_auto_runs(self.store,self.pid,[row])
        self.assertEqual([j['command'] for j in auto_jobs(self.store,self.pid)],['compile'])
        self.assertTrue(process_auto_job(self.store))
        self.assertEqual(auto_jobs(self.store,self.pid)[0]['status'],'passed')

    def test_explicit_agent_request_runs_python_module_without_approval(self):
        row=write_file(self.store,self.pid,'libmath.py',"print('requested module')\n")
        queue_auto_runs(self.store,self.pid,[row],requested=['libmath.py','../bad.py','powershell.exe'],actor='dev-1')
        self.assertEqual(len(auto_jobs(self.store,self.pid)),2)
        self.assertTrue(process_auto_job(self.store))
        self.assertTrue(process_auto_job(self.store))
        self.assertIn('requested module',runs(self.store,self.pid)[0]['output'])

    def test_restart_recovers_claimed_job(self):
        row=write_file(self.store,self.pid,'app.py','print("restart")\n')
        queue_auto_runs(self.store,self.pid,[row])
        from execution_workspace import claim_auto_job
        job=claim_auto_job(self.store)
        self.assertEqual(job['status'],'queued')
        recover_auto_jobs(self.store)
        self.assertTrue(process_auto_job(self.store))

    def test_failed_python_is_not_reported_as_passed(self):
        row=write_file(self.store,self.pid,'main.py','raise RuntimeError("fixture failure")\n')
        queue_auto_runs(self.store,self.pid,[row])
        process_auto_job(self.store)
        process_auto_job(self.store)
        self.assertEqual({j['command']:j['status'] for j in auto_jobs(self.store,self.pid)},
                         {'compile':'passed','main.py':'failed'})

    def test_running_job_can_be_stopped_without_host_approval(self):
        import threading
        row=write_file(self.store,self.pid,'main.py',
             'import time\ntime.sleep(20)\nprint("SHOULD_NOT_COMPLETE")\n')
        queue_auto_runs(self.store,self.pid,[row])
        self.assertTrue(process_auto_job(self.store))  # compilation
        done=threading.Event()
        def execute():
            process_auto_job(self.store)
            done.set()
        t=threading.Thread(target=execute,daemon=True);t.start()
        deadline=time.monotonic()+4
        while time.monotonic()<deadline:
            if any(j['status']=='running' for j in auto_jobs(self.store,self.pid)):
                break
            time.sleep(.05)
        cancel_auto_jobs(self.store,self.pid)
        t.join(timeout=6)
        self.assertTrue(done.is_set(),'Running subprocess tree did not stop')
        self.assertIn(auto_jobs(self.store,self.pid)[0]['status'],('cancelled','superseded'))

    def test_model_json_schema_requires_workspace_runs_list(self):
        self.assertIn('workspace_runs',RESPONSE_SCHEMA['required'])
        self.assertEqual(RESPONSE_SCHEMA['properties']['workspace_runs']['type'],'array')

    def test_agent_writes_code_and_queues_execution_without_approval(self):
        class CodingAgent:
            def generate(self,role,model,instructions,context):
                return {'reply':'I wrote the script.',
                        'workspace_files':[{'path':'app.py','content':'print("AGENT_RAN")\n','expected_sha256':None}],
                        'workspace_runs':['app.py'],'file_requests':[], 'needs_human':False,
                        'meeting_requests':[],'outgoing':[]}
        task=self.store.send(self.pid,'human','dev-1','Build a script')
        engine=Engine(self.store,client=CodingAgent(),demo=True)
        self.assertTrue(engine.process_once())
        self.assertEqual(self.store.message(task['id'])['status'],'done')
        self.assertEqual(len(auto_jobs(self.store,self.pid)),2)
        self.assertTrue(process_auto_job(self.store))
        self.assertTrue(process_auto_job(self.store))
        self.assertIn('AGENT_RAN',runs(self.store,self.pid)[0]['output'])

if __name__=='__main__':unittest.main()
