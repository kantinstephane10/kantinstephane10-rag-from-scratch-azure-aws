"""Regression tests for model replies at relay boundaries and historical failure recovery.

No paid OpenAI API calls are made.
"""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine import Engine
from server import make_server
from store import Store, StoreError


class DelegatingFakeClient:
    def generate(self, role, model, instructions, context):
        recipient = 'dev-1' if role == 'product-manager' else 'product-manager'
        return {
            'reply': f'{role} completed review with evidence still pending.',
            'outgoing': [{'recipient': recipient, 'message': 'Review the next step.'}],
            'needs_human': False,
            'meeting_requests': [],
        }


class RelayRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / 'builder.sqlite3'))
        self.pid = self.store.create_project('RAG workflow', 'Build and verify RAG')['id']

    def tearDown(self):
        self.tmp.cleanup()

    def chain_at_depth(self, depth):
        """Create valid, verified handoffs right up to the requested queued task."""
        parent = None
        sender = 'human'
        recipient = 'product-manager'
        for level in range(depth + 1):
            task = self.store.send(self.pid, sender, recipient, f'Review phase {level}',
                parent_id=parent['id'] if parent else None,
                chain_id=parent['chain_id'] if parent else None,
                depth=level)
            if level == depth:
                return task
            claimed = self.store.claim()
            self.assertEqual(claimed['id'], task['id'])
            self.store.complete(task['id'], 'test-model', f'Phase {level} analysis completed')
            parent = task
            sender = recipient
            recipient = 'dev-1' if recipient == 'product-manager' else 'product-manager'

    def test_four_hops_no_longer_is_a_hard_failure(self):
        task = self.chain_at_depth(4)
        engine = Engine(self.store, client=DelegatingFakeClient(), demo=True)
        self.assertTrue(engine.process_once())
        completed = self.store.message(task['id'])
        self.assertEqual(completed['status'], 'done')
        self.assertIn('completed review', completed['reply'])
        later = [m for m in self.store.messages(self.pid) if m['depth'] == 5]
        self.assertEqual(len(later), 1, 'The next specialist should be scheduled')
        self.assertIsNotNone(later[0]['handoff_id'])

    def test_long_delegation_chain_is_supported(self):
        final_task=self.chain_at_depth(12)
        engine=Engine(self.store,client=DelegatingFakeClient(),demo=True)
        self.assertTrue(engine.process_once())
        result=self.store.message(final_task['id'])
        self.assertEqual(result['status'],'done')
        later=[m for m in self.store.messages(self.pid) if m['depth']==13]
        self.assertEqual(len(later),1)
        self.assertEqual(later[0]['status'],'queued')

    def test_invalid_followups_do_not_discard_answer(self):
        task = self.store.send(self.pid, 'human', 'product-manager', 'Make a plan')
        class InvalidClient:
            def generate(self, *_):
                return {'reply': 'The plan is ready.', 'outgoing': [
                    {'recipient': 'product-manager', 'message': 'Ask myself'},
                    {'recipient': 'not-a-worker', 'message': 'Invalid role'},
                    {'recipient': 'dev-1', 'message': 'Valid review'},
                ], 'meeting_requests': [
                    {'topic': '', 'attendees': ['dev-1']},
                    {'topic': 'Valid discussion', 'attendees': ['dev-1']},
                ], 'needs_human': False}
        engine = Engine(self.store, client=InvalidClient(), demo=True)
        self.assertTrue(engine.process_once())
        completed = self.store.message(task['id'])
        self.assertEqual(completed['status'], 'done')
        self.assertIn('The plan is ready', completed['reply'])
        self.assertIn('Workflow notice', completed['reply'])
        self.assertEqual(len(self.store.messages(self.pid)), 2)

    def test_existing_legacy_failure_auto_requeues_once(self):
        task = self.store.send(self.pid, 'human', 'product-manager', 'Continue planning')
        other = self.store.send(self.pid, 'human', 'dev-1', 'Unrelated issue')
        first = self.store.claim()
        self.assertEqual(first['id'], task['id'])
        self.store.failure(task['id'], 'Agent relay limit reached; no further peer messages allowed.', retryable=False)
        second = self.store.claim()
        self.assertEqual(second['id'], other['id'])
        self.store.failure(other['id'], 'Invalid model API key', retryable=False)
        self.assertEqual(self.store.recover_processing(), 1)
        self.assertEqual(self.store.message(task['id'])['status'], 'queued')
        self.assertEqual(self.store.message(other['id'])['status'], 'failed')
        self.assertEqual(self.store.recover_processing(), 0)
        engine = Engine(self.store, client=DelegatingFakeClient(), demo=True)
        self.assertTrue(engine.process_once())
        self.assertEqual(self.store.message(task['id'])['status'], 'done')
        self.assertEqual(self.store.message(other['id'])['status'], 'failed')

    def test_retry_is_project_scoped(self):
        task = self.store.send(self.pid, 'human', 'product-manager', 'Plan')
        claimed = self.store.claim()
        self.store.failure(task['id'], 'Unexpected provider error', retryable=False)
        another = self.store.create_project('Separate', 'Different information')['id']
        with self.assertRaises(StoreError):
            self.store.retry(task['id'], project_id=another)
        self.assertEqual(self.store.message(task['id'])['status'], 'failed')
        self.store.retry(task['id'], project_id=self.pid)
        self.assertEqual(self.store.message(task['id'])['status'], 'queued')


class RelayRetryHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(str(Path(self.tmp.name) / 'db.sqlite3'))
        self.engine = Engine(self.store, client=object(), demo=True)
        self.server = make_server(self.store, self.engine, port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = 'http://127.0.0.1:' + str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(3)
        self.tmp.cleanup()

    def test_retry_task_endpoint_rejects_cross_project(self):
        one = self.store.create_project('One', 'Private')['id']
        two = self.store.create_project('Two', 'Another')['id']
        task = self.store.send(one, 'human', 'product-manager', 'Plan')
        self.store.claim()
        self.store.failure(task['id'], 'Failure', retryable=False)
        def request(pid):
            req = Request(f'{self.base}/api/projects/{pid}/retry',
                          json.dumps({'message_id': task['id']}).encode(),
                          headers={'Content-Type': 'application/json'}, method='POST')
            with urlopen(req, timeout=3) as result:
                return json.load(result)
        with self.assertRaises(HTTPError) as ctx:
            request(two)
        self.assertEqual(ctx.exception.code, 400)
        self.assertEqual(self.store.message(task['id'])['status'], 'failed')
        self.assertTrue(request(one)['retried'])
        self.assertEqual(self.store.message(task['id'])['status'], 'queued')


if __name__ == '__main__':
    unittest.main()
