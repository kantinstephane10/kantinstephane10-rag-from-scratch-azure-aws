"""Deterministic intake, activity and consent tests; no OpenAI calls."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from config import model_for, model_for_project
from engine import Engine
from intake import validate_intake
from server import make_server
from store import Store, StoreError


class WorkspaceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.filename = str(Path(self.tmp.name) / 'workspace.sqlite')
        self.store = Store(self.filename)
    def tearDown(self):
        self.tmp.cleanup()

    def test_budget_is_explicit_planning_math(self):
        v = validate_intake({'product_type': 'agentic', 'estimated_hours': 10,
                             'hourly_rate': 100, 'one_time_expenses': 75,
                             'contingency_pct': 10, 'monthly_cost': 35})
        self.assertEqual(v['estimated_total'], 1182.50)
        self.assertEqual(v['monthly_cost'], 35)

    def test_bad_numbers_do_not_enter_db(self):
        for v in ('nan', -1, 'inf', 'not a number'):
            with self.assertRaises(StoreError):
                self.store.create_project('x', 'brief', intake={'hourly_rate': v})
        self.assertEqual(len(self.store.projects()), 0)

    def test_consent_requires_boolean(self):
        with self.assertRaises(StoreError):
            self.store.create_project('x', 'brief', intake={'provider_consent': 'true'})

    def test_stored_intake_survives_restart(self):
        p = self.store.create_project('A', 'B', intake={'product_type': 'agent',
            'model_profile': 'deep-review', 'provider_consent': True, 'estimated_hours': 4})
        reopened = Store(self.filename).project(p['id'])
        self.assertEqual(reopened['intake']['product_type'], 'agent')
        self.assertTrue(reopened['intake']['provider_consent'])
        self.assertEqual(reopened['intake']['estimated_hours'], 4)

    def test_worker_activity_uses_actual_message_states(self):
        p = self.store.create_project('x', 'brief')
        self.store.send(p['id'], 'human', 'dev-1', 'Please plan task')
        activity = self.store.activity(p['id'])
        self.assertEqual(activity['counts']['queued'], 1)
        task = self.store.claim()
        self.assertEqual(self.store.activity(p['id'])['counts']['processing'], 1)
        self.store.complete(task['id'], 'test-model', 'proposal; no code executed')
        self.assertEqual(self.store.activity(p['id'])['counts']['done'], 1)
        self.assertEqual(self.store.activity(p['id'])['messages'][0]['reply'], 'proposal; no code executed')


    def test_task_can_be_cancelled_before_completion(self):
        p = self.store.create_project('x', 'brief')
        msg = self.store.send(p['id'], 'human', 'dev-1', 'Please plan task')
        cancelled = self.store.cancel(msg['id'], 'Operator changed priorities')
        self.assertEqual(cancelled['status'], 'cancelled')
        self.assertEqual(self.store.activity(p['id'])['counts']['cancelled'], 1)
        self.assertIsNone(self.store.claim())

    def test_cancelled_processing_task_drops_late_completion(self):
        p = self.store.create_project('x', 'brief')
        self.store.send(p['id'], 'human', 'dev-1', 'Please plan task')
        task = self.store.claim()
        self.store.cancel(task['id'], 'Operator changed priorities')
        self.assertFalse(self.store.complete(task['id'], 'test-model', 'late reply'))
        self.assertEqual(self.store.message(task['id'])['status'], 'cancelled')

    def test_project_model_profile_can_be_overridden(self):
        old = os.environ.get('GPT_MODEL_DEEP_REVIEW')
        try:
            os.environ['GPT_MODEL_DEEP_REVIEW'] = 'test-gpt-model'
            self.assertEqual(model_for_project('dev-1', 'deep-review'), 'test-gpt-model')
            self.assertEqual(model_for_project('dev-1', 'standard'), model_for('dev-1'))
        finally:
            if old is None: os.environ.pop('GPT_MODEL_DEEP_REVIEW', None)
            else: os.environ['GPT_MODEL_DEEP_REVIEW'] = old

    def test_live_gpt_kickoff_requires_consent(self):
        p = self.store.create_project('x', 'brief')
        engine = Engine(self.store, client=object(), demo=False)
        with self.assertRaises(StoreError):
            engine.kickoff(p['id'])
        self.assertEqual(self.store.activity(p['id'])['counts']['total'], 0)


class WorkspaceHttpTests(WorkspaceTests):
    def setUp(self):
        super().setUp()
        # Engine is intentionally never started. No GPT network calls occur.
        self.engine = Engine(self.store, client=object(), demo=False)
        self.server = make_server(self.store, self.engine, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f'http://127.0.0.1:{self.server.server_address[1]}'
    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)
        super().tearDown()

    def _post(self, data):
        req = Request(self.base+'/api/projects', json.dumps(data).encode(),
                      headers={'Content-Type':'application/json'}, method='POST')
        with urlopen(req, timeout=3) as response:
            return response.status, json.load(response)

    def test_http_create_and_activity(self):
        status, p = self._post({'name':'Research Agent', 'brief':'Build research agent',
             'kickoff':True, 'intake':{'product_type':'agent','provider_consent':True,
                                     'estimated_hours':8,'hourly_rate':50}})
        self.assertEqual(status, 201)
        self.assertEqual(p['intake']['estimated_total'], 400.0)
        with urlopen(self.base+'/api/projects/'+p['id']+'/activity', timeout=3) as response:
            activity=json.load(response)
        self.assertEqual(activity['counts']['queued'],3)
        self.assertEqual([m['recipient'] for m in activity['messages']],
                         ['business-analyst-1','business-analyst-2','product-manager'])


    def test_http_cancel_task(self):
        status, p = self._post({'name':'Research Agent', 'brief':'Build research agent',
             'kickoff':True, 'intake':{'product_type':'agent','provider_consent':True}})
        self.assertEqual(status, 201)
        message_id = self.store.activity(p['id'])['messages'][0]['id']
        req = Request(self.base+f'/api/tasks/{message_id}/cancel', json.dumps({'reason':'Operator changed priorities'}).encode(),
                      headers={'Content-Type':'application/json'}, method='POST')
        with urlopen(req, timeout=3) as response:
            payload = json.load(response)
        self.assertEqual(payload['status'], 'cancelled')

    def test_http_rejects_nonconsensual_gpt_kickoff_before_create(self):
        with self.assertRaises(HTTPError) as ctx:
            self._post({'name':'Sensitive', 'brief':'Private info', 'kickoff':True})
        self.assertEqual(ctx.exception.code,400)
        self.assertEqual(len(self.store.projects()),0)

    def test_ui_project_form_and_live_activity_exposed(self):
        with urlopen(self.base+'/',timeout=3) as response:
            html=response.read()
        self.assertIn(b'project-type',html)
        self.assertIn(b'provider-consent',html)
        self.assertIn(b'activity-sequence',html)
        with urlopen(self.base+'/app.js',timeout=3) as response:
            js=response.read()
        self.assertIn(b'refreshActivity',js)


if __name__=='__main__': unittest.main()
