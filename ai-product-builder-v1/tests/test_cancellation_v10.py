"""Cancellation persistence, safety, exact targeting and recoverable project flow.
All tests use a local SQLite database and fake API client; no OpenAI calls.
"""
import base64
import json
import os
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from engine import Engine
from meetings import create_meeting, meeting, cancel_meeting, claim_turn, finish_turn
from server import make_server
from store import Store, StoreError

REASON = 'Scope changed after review'


class CancellationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = str(Path(self.temp.name) / 'builder.sqlite3')
        self.store = Store(self.db)
        self.p = self.store.create_project('RAG assistant', 'Create an AI assistant')

    def tearDown(self):
        self.temp.cleanup()

    def task(self, project_id=None):
        return self.store.send(project_id or self.p['id'], 'human', 'dev-1', 'Create a plan')

    def test_single_requires_reason_and_stores_metadata(self):
        t = self.task()
        for bad in (None, '', '   ', 'short'[:3], 32, 'a'*501):
            with self.assertRaises(StoreError):
                self.store.cancel(t['id'], bad)
        self.assertEqual(self.store.message(t['id'])['status'], 'queued')
        result = self.store.cancel(t['id'], REASON)
        self.assertEqual(result['status'], 'cancelled')
        self.assertEqual(result['cancel_reason'], REASON)
        self.assertEqual(result['cancelled_by'], 'human')
        self.assertIsInstance(result['cancelled_at'], int)
        event = self.store.events(self.p['id'])[0]
        self.assertEqual(event['kind'], 'task_cancelled')
        self.assertEqual(json.loads(event['payload'])['reason'], REASON)
        with self.assertRaises(StoreError):
            self.store.retry(t['id'])

    def test_bulk_selects_only_given_tasks(self):
        one, two, untouched = [self.task() for _ in range(3)]
        result = self.store.cancel_tasks(self.p['id'], [one['id'], two['id']], REASON)
        self.assertEqual(result['cancelled'], 2)
        for t in (one, two):
            self.assertEqual(self.store.message(t['id'])['status'], 'cancelled')
        self.assertEqual(self.store.message(untouched['id'])['status'], 'queued')
        self.assertEqual(self.store.activity(self.p['id'])['counts']['cancelled'], 2)

    def test_bulk_is_all_or_nothing_for_completed_cross_project_duplicate_and_unknown(self):
        a,b,c=[self.task() for _ in range(3)]
        other=self.store.create_project('other','other')
        outside=self.task(other['id'])
        self.store.cancel(c['id'], REASON)
        for selection in ([a['id'], c['id']], [a['id'],outside['id']],
                          [a['id'], a['id']], [a['id'], 'deadbeef'*4], []):
            with self.assertRaises(StoreError):
                self.store.cancel_tasks(self.p['id'],selection,REASON)
            self.assertEqual(self.store.message(a['id'])['status'],'queued')
            self.assertEqual(self.store.message(b['id'])['status'],'queued')
        with self.assertRaises(StoreError):
            self.store.cancel_tasks(self.p['id'],[a['id']],None)

    def test_processing_task_cannot_publish_after_cancellation(self):
        task=self.task(); claimed=self.store.claim()
        self.assertEqual(task['id'], claimed['id'])
        self.store.cancel(task['id'], REASON)
        self.assertFalse(self.store.complete(task['id'],'gpt-test','Late answer'))
        self.assertFalse(self.store.failure(task['id'],'HTTP 503',retryable=True))
        self.assertEqual(self.store.message(task['id'])['status'],'cancelled')
        self.assertIsNone(self.store.claim())
        reopened=Store(self.db)
        reopened.recover_processing()
        self.assertEqual(reopened.message(task['id'])['status'],'cancelled')

    def test_cancelled_work_blocks_human_approval(self):
        task=self.task();self.store.cancel(task['id'],REASON)
        with self.assertRaises(StoreError):
            self.store.approve(self.p['id'],'APPROVE','operator')

    def test_meeting_reason_persists_and_drops_in_flight_replies(self):
        room=create_meeting(self.store,topic='Review AI architecture',participants=['rag-engineer'])
        turn=claim_turn(self.store)
        changed=cancel_meeting(self.store,room['id'],REASON)
        self.assertEqual(changed['status'],'cancelled')
        self.assertEqual(changed['cancel_reason'],REASON)
        self.assertFalse(finish_turn(self.store,turn['id'],'STALE','gpt-test'))
        self.assertNotIn('STALE',[msg['body'] for msg in meeting(self.store,room['id'])['messages']])
        self.assertIsNone(claim_turn(self.store))
        self.assertEqual(meeting(Store(self.db),room['id'])['cancel_reason'],REASON)

    def test_scheduled_room_cannot_activate_after_cancel(self):
        import time
        room=create_meeting(self.store,topic='Planned meeting',participants=['dev-2'],
                            starts_at=int(time.time())+100,start_now=False)
        cancel_meeting(self.store,room['id'],REASON)
        self.assertIsNone(claim_turn(self.store))

    def test_schema_migrations_work_with_legacy_db(self):
        with self.store.db() as db:
            db.execute('ALTER TABLE messages RENAME TO messages_renamed')
            db.execute('''CREATE TABLE messages (
                id TEXT PRIMARY KEY, project_id TEXT NOT NULL, sender TEXT NOT NULL,
                recipient TEXT NOT NULL, body TEXT NOT NULL, chain_id TEXT NOT NULL,
                parent_id TEXT, depth INTEGER DEFAULT 0,status TEXT DEFAULT 'queued',
                attempts INTEGER DEFAULT 0,next_attempt_at INTEGER DEFAULT 0,
                model TEXT, reply TEXT, error TEXT,created_at INTEGER,updated_at INTEGER)''')
            db.execute('DROP TABLE messages_renamed')
            db.execute('ALTER TABLE meetings RENAME TO meetings_renamed')
            db.execute('''CREATE TABLE meetings (
                id TEXT PRIMARY KEY,project_id TEXT,topic TEXT NOT NULL,
                requested_by TEXT NOT NULL,participants TEXT NOT NULL,
                chair TEXT DEFAULT 'product-manager', status TEXT DEFAULT 'scheduled',
                floor TEXT DEFAULT 'agents',epoch INTEGER DEFAULT 0,
                starts_at INTEGER NOT NULL,created_at INTEGER NOT NULL,ended_at INTEGER)''')
            db.execute('DROP TABLE meetings_renamed')
        # This is intentionally a minimal legacy schema recreation to verify additive upgrades.
        updated=Store(self.db)
        with updated.db() as db:
            cols={r['name'] for r in db.execute('PRAGMA table_info(messages)')}
            meeting_cols={r['name'] for r in db.execute('PRAGMA table_info(meetings)')}
        self.assertTrue({'cancel_reason','cancelled_by','cancelled_at','handoff_id'}<=cols)
        self.assertTrue({'cancel_reason','cancelled_by'}<=meeting_cols)

    def test_cancel_during_model_request_prevents_outgoing(self):
        task=self.task()
        store=self.store
        class InterruptClient:
            def generate(self,role,model,instructions,context):
                store.cancel(task['id'],REASON)
                return {'reply':'Late response','outgoing':[{'recipient':'code-review-1','message':'Review'}],
                        'meeting_requests':[],'needs_human':False}
        engine=Engine(store,client=InterruptClient(),demo=True)
        self.assertTrue(engine.process_once())
        self.assertEqual(len(store.messages(self.p['id'])),1)
        self.assertEqual(store.message(task['id'])['status'],'cancelled')


class CancellationHTTPTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.temp.name)/'test.sqlite3'))
        self.engine=Engine(self.store,client=object(),demo=True)
        self.server=make_server(self.store,self.engine,port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.root='http://127.0.0.1:'+str(self.server.server_address[1])

    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(3);self.temp.cleanup()

    def post(self,path,value):
        request=Request(self.root+path,json.dumps(value).encode('utf8'),
                        {'Content-Type':'application/json'},method='POST')
        with urlopen(request,timeout=4) as resp:return resp.status,json.load(resp)

    def test_project_start_and_document_upload(self):
        _,p=self.post('/api/projects',{'name':'Research RAG','brief':'Explain this corpus',
                                      'intake':{'product_type':'rag'},'kickoff':False})
        self.assertEqual(p['status'],'draft')
        _,artifact=self.post(f"/api/projects/{p['id']}/artifacts",
                            {'name':'readme.md','base64':base64.b64encode(b'RAG reference').decode()})
        self.assertEqual(artifact['name'],'readme.md')
        self.post(f"/api/projects/{p['id']}/kickoff",{})
        with urlopen(f"{self.root}/api/projects/{p['id']}/activity",timeout=4) as resp:
            data=json.load(resp)
        self.assertEqual(data['counts']['queued'],3)
        self.assertEqual([m['recipient'] for m in data['messages']],
                         ['business-analyst-1','business-analyst-2','product-manager'])

    def test_http_single_bulk_meeting_require_reasons(self):
        _,p=self.post('/api/projects',{'name':'AI Builder','brief':'Design a RAG', 'kickoff':False})
        ids=[self.store.send(p['id'],'human','dev-1',f'Task {i}')['id'] for i in range(3)]
        for endpoint,payload in ((f'/api/tasks/{ids[0]}/cancel',{}),
                                 (f"/api/projects/{p['id']}/cancel-tasks",{'task_ids':ids[:2]})):
            with self.assertRaises(HTTPError) as err:self.post(endpoint,payload)
            self.assertEqual(err.exception.code,400)
        _,result=self.post(f"/api/projects/{p['id']}/cancel-tasks",{'task_ids':ids[:2],'reason':REASON})
        self.assertEqual(result['cancelled'],2)
        self.assertEqual(self.store.message(ids[2])['status'],'queued')
        _,room=self.post('/api/meetings',{'topic':'Risk review','participants':['rag-engineer']})
        with self.assertRaises(HTTPError) as err:self.post(f"/api/meetings/{room['id']}/cancel",{})
        self.assertEqual(err.exception.code,400)
        _,result=self.post(f"/api/meetings/{room['id']}/cancel",{'reason':REASON})
        self.assertEqual(result['cancel_reason'],REASON)
        self.assertEqual(result['status'],'cancelled')


if __name__=='__main__':
    unittest.main()
