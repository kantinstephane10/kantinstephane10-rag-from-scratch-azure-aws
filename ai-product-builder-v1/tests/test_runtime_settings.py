"""Offline tests for UI-managed OpenAI settings, model routing and token budgets."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from engine import Engine
from runtime_settings import RuntimeSettings, family_for
from server import make_server
from store import Store, StoreError


class RuntimeConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # No actual API key is used to invoke OpenAI in these tests.
        self.env = patch.dict(os.environ, {'OPENAI_API_KEY':''})
        self.env.start()
        self.store = Store(str(Path(self.tmp.name)/'state.sqlite'))
        self.settings = RuntimeSettings(Path(self.tmp.name))

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def test_safe_demo_default(self):
        self.assertEqual(self.settings.snapshot()['mode'],'demo')
        self.assertFalse(self.settings.snapshot()['api_key_configured'])

    def test_api_key_never_in_snapshot_or_plaintext_settings(self):
        fake='sk-'+'f'*40
        data=self.settings.save_key(fake)
        self.assertNotIn(fake,json.dumps(data))
        self.settings.update({'max_output_tokens':800})
        self.assertNotIn(fake,self.settings.path.read_text())
        if os.name!='nt':
            self.assertFalse(self.settings.key_path.exists())

    def test_live_mode_requires_key(self):
        with self.assertRaises(StoreError):
            self.settings.update({'mode':'openai'})
        self.assertEqual(self.settings.snapshot()['mode'],'demo')

    def test_model_groups_and_all_agent_overrides(self):
        self.settings.update({'models':{'architecture':'gpt-4.1','engineering':'gpt-4.1-mini',
                                        'business':'gpt-4.1-mini','routine':'gpt-4.1-mini'},
                              'role_models':{'rag-engineer':'gpt-4.1'}})
        self.assertEqual(self.settings.model_for('rag-engineer'),'gpt-4.1')
        self.assertEqual(self.settings.model_for('dev-1'),'gpt-4.1-mini')
        self.assertEqual(len(self.settings.snapshot()['roles']),49)
        reopened=RuntimeSettings(Path(self.tmp.name))
        self.assertEqual(reopened.model_for('rag-engineer'),'gpt-4.1')

    def test_invalid_role_and_model_rejected(self):
        with self.assertRaises(StoreError):self.settings.update({'role_models':{'ghost':'gpt-4.1'}})
        with self.assertRaises(StoreError):self.settings.update({'role_models':{'dev-1':'bad model id!'}})

    def test_budgets_reject_invalid_values(self):
        for bad in ('100',0,False,4001):
            with self.subTest(bad=bad), self.assertRaises(StoreError):
                self.settings.update({'max_output_tokens':bad})

    def test_budget_reservations_are_atomic_and_persisted(self):
        self.assertEqual(self.store.reserve_llm_call('project',2,100)['call_count'],1)
        self.store.record_llm_usage('project',90)
        self.assertEqual(self.store.reserve_llm_call('project',2,100)['call_count'],2)
        with self.assertRaises(StoreError):self.store.reserve_llm_call('project',2,100)
        self.assertEqual(self.store.llm_usage('project')['total_tokens'],90)
        self.assertEqual(self.store.llm_usage('project')['call_count'],2)

    def test_hard_stop_after_reported_token_threshold(self):
        self.store.reserve_llm_call('other',10,500)
        self.store.record_llm_usage('other',500)
        with self.assertRaises(StoreError):self.store.reserve_llm_call('other',10,500)

    def test_switching_mode_updates_engine_without_network(self):
        key='sk-'+'a'*40
        self.settings.save_key(key)
        engine=Engine(self.store,runtime=self.settings)
        self.assertTrue(engine.demo)
        self.settings.update({'mode':'openai','max_output_tokens':600})
        engine.apply_runtime()
        self.assertFalse(engine.demo)
        self.assertEqual(engine.client.max_output_tokens,600)
        self.assertEqual(engine.client.key,key)
        self.settings.clear_key()
        engine.apply_runtime()
        self.assertTrue(engine.demo)


class RuntimeConfigurationHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ, {'OPENAI_API_KEY':''})
        self.env.start()
        self.store=Store(str(Path(self.tmp.name)/'state.sqlite'))
        self.settings=RuntimeSettings(Path(self.tmp.name))
        self.engine=Engine(self.store,runtime=self.settings)
        self.server=make_server(self.store,self.engine,port=0,runtime=self.settings)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base=f'http://127.0.0.1:{self.server.server_address[1]}'
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(timeout=3)
        self.env.stop();self.tmp.cleanup()

    def post(self,path,body):
        req=Request(self.base+path,json.dumps(body).encode(),headers={'Content-Type':'application/json'},method='POST')
        with urlopen(req,timeout=3) as r:return json.load(r)

    def test_key_api_save_and_clear_and_no_echo(self):
        key='sk-'+'x'*40
        out=self.post('/api/settings/key',{'api_key':key})
        self.assertTrue(out['api_key_configured'])
        self.assertNotIn(key,json.dumps(out))
        self.post('/api/settings',{'mode':'openai'})
        with urlopen(self.base+'/api/health') as r:self.assertFalse(json.load(r)['offline_demo'])
        cleared=self.post('/api/settings/key/clear',{})
        self.assertEqual(cleared['mode'],'demo')
        self.assertFalse(cleared['api_key_configured'])

    def test_refuses_live_switch_without_key(self):
        with self.assertRaises(HTTPError) as e:self.post('/api/settings',{'mode':'openai'})
        self.assertEqual(e.exception.code,400)

    def test_settings_serve_all_49_roles_and_budget(self):
        with urlopen(self.base+'/api/settings') as r:cfg=json.load(r)
        self.assertEqual(len(cfg['roles']),49)
        self.assertIn('max_total_tokens_per_project',cfg)
        with urlopen(self.base+'/api/projects') as r:self.assertIn('projects',json.load(r))

    def test_settings_api_rejects_cross_origin_write(self):
        req=Request(self.base+'/api/settings',json.dumps({'mode':'demo'}).encode(),
            headers={'Content-Type':'application/json','Origin':'https://evil.invalid'},method='POST')
        with self.assertRaises(HTTPError) as e:urlopen(req,timeout=3)
        self.assertEqual(e.exception.code,403)


if __name__=='__main__':unittest.main()
