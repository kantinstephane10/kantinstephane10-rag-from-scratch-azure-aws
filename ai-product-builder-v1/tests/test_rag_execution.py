"""Real local PDF extraction and SQLite RAG execution tests; no paid API calls."""
import hashlib
import json
import tempfile
import time
import unittest
from pathlib import Path

from engine import Engine
from handoffs import save_artifact
from meetings import claim_turn, create_meeting, meeting
from rag_pipeline import init_rag, build_index, rag_status, answer_question, _extract, start_index_job
from store import Store, StoreError


class Splitter:
    def split_text(self, text):
        return [x.strip() for x in text.split('\n') if x.strip()]


def embeddings(texts):
    return [[2.0, 0.0] if 'alpha' in x.lower() else [0.0, 2.0] for x in texts]


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.tmp.name)/'store.sqlite3'))
        init_rag(self.store)
        self.project=self.store.create_project('Test RAG', 'Answer uploaded documents', intake={
            'product_type':'rag','provider_consent':True})
        self.pid=self.project['id']

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_real_pdf_and_index_with_cited_answer(self):
        import pymupdf
        doc=pymupdf.open()
        page=doc.new_page()
        page.insert_text((72,72),'Alpha controls require review.\nBeta controls require tests.')
        raw=doc.tobytes()
        meta=save_artifact(self.store,self.pid,'policy.pdf',raw)
        self.assertEqual(_extract('policy.pdf',raw)[0][0],1)
        receipt=build_index(self.store,self.pid,'',embedder=embeddings,splitter=Splitter())
        self.assertGreaterEqual(receipt['chunks'],1)
        self.assertEqual(receipt['documents'][0]['sha256'],hashlib.sha256(raw).hexdigest())
        self.assertFalse(rag_status(self.store,self.pid)['stale'])
        result=answer_question(self.store,self.pid,'alpha','', 'mock',embedder=embeddings,
            responder=lambda q, passages:'Alpha controls require review [S1].')
        self.assertIn('Alpha controls',result['answer'])
        self.assertEqual(result['sources'][0]['file'],'policy.pdf')
        self.assertEqual(result['sources'][0]['page'],1)
        self.assertEqual(result['sources'][0]['ref'],'S1')

    def test_new_uploaded_document_marks_index_stale(self):
        save_artifact(self.store,self.pid,'source.txt',b'Alpha policy')
        build_index(self.store,self.pid,'',embedder=embeddings,splitter=Splitter())
        save_artifact(self.store,self.pid,'new.txt',b'Beta policy')
        self.assertTrue(rag_status(self.store,self.pid)['stale'])
        with self.assertRaisesRegex(StoreError,'Documents changed'):
            answer_question(self.store,self.pid,'alpha','', 'mock',embedder=embeddings,
                responder=lambda *args:'Answer')

    def test_bad_embedding_does_not_overwrite_old_index(self):
        save_artifact(self.store,self.pid,'source.txt',b'Alpha policy')
        build_index(self.store,self.pid,'',embedder=embeddings,splitter=Splitter())
        before=rag_status(self.store,self.pid)['receipt']
        with self.assertRaises(StoreError):
            build_index(self.store,self.pid,'',embedder=lambda texts:[[0.0,0.0] for _ in texts],splitter=Splitter())
        self.assertEqual(rag_status(self.store,self.pid)['receipt'],before)

    def test_delete_project_removes_files_index_meetings_and_other_project_survives(self):
        other=self.store.create_project('Other', 'Keep me', intake={'product_type':'rag','provider_consent':True})
        save_artifact(self.store,other['id'],'keep.txt',b'Beta text')
        save_artifact(self.store,self.pid,'source.txt',b'Alpha policy')
        build_index(self.store,self.pid,'',embedder=embeddings,splitter=Splitter())
        room=create_meeting(self.store,topic='RAG planning',participants=['rag-engineer'],project_id=self.pid)
        with self.assertRaisesRegex(StoreError,'exact project name'):
            self.store.delete_project(self.pid,'wrong')
        response=self.store.delete_project(self.pid,'Test RAG')
        self.assertTrue(response['deleted'])
        with self.assertRaises(StoreError):self.store.project(self.pid)
        self.assertEqual(self.store.project(other['id'])['name'],'Other')
        with self.store.db() as db:
            self.assertEqual(db.execute('SELECT COUNT(*) FROM rag_chunks WHERE project_id=?',(self.pid,)).fetchone()[0],0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM meetings WHERE project_id=?',(self.pid,)).fetchone()[0],0)
        self.assertFalse((Path(self.store.filename).parent/'artifacts'/self.pid).exists())

    def test_cannot_index_without_consent(self):
        p=self.store.create_project('No consent','Private',intake={'product_type':'rag'})
        save_artifact(self.store,p['id'],'source.txt',b'Confidential')
        with self.assertRaisesRegex(StoreError,'consent'):
            build_index(self.store,p['id'],'',embedder=embeddings,splitter=Splitter())

    def test_project_cannot_be_deleted_by_other_name(self):
        with self.assertRaises(StoreError):self.store.delete_project(self.pid,'Other')
        self.assertEqual(self.store.project(self.pid)['name'],'Test RAG')


class MeetingResilienceTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.tmp.name)/'state.sqlite3'))

    def tearDown(self):self.tmp.cleanup()

    def test_bad_agent_question_does_not_discard_the_answer(self):
        room=create_meeting(self.store,topic='Testing',participants=['rag-engineer'])
        class FakeClient:
            def generate(self,role,model,instructions,context):
                return {'reply':'The requirements are not production approved.',
                    'outgoing':[{'recipient':'product-manager','message':'Self-question?'}],
                    'meeting_requests':[],'needs_human':False}
        engine=Engine(self.store,client=FakeClient(),demo=True)
        self.assertTrue(engine.process_once())
        after=meeting(self.store,room['id'])
        self.assertTrue(any(m['actor']=='product-manager' and 'not production approved' in m['body'] for m in after['messages']))
        self.assertFalse(any(m['kind']=='error' for m in after['messages']))
        self.assertTrue(any(t['agent']=='product-manager' and t['status']=='done' for t in after['turns']))


if __name__=='__main__': unittest.main()

class DeleteProjectHttpTests(unittest.TestCase):
    def setUp(self):
        import threading
        from server import make_server
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.tmp.name)/'store.sqlite3'))
        self.engine=Engine(self.store,client=object(),demo=True)
        self.server=make_server(self.store,self.engine,port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_address[1])
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(2); self.tmp.cleanup()
    def test_http_delete_requires_name_and_removes_project(self):
        from urllib.request import Request,urlopen
        from urllib.error import HTTPError
        project=self.store.create_project('RemoveMe','No longer needed')
        url=self.base+'/api/projects/'+project['id']+'/delete'
        def call(confirmation):
            request=Request(url,json.dumps({'confirmation':confirmation}).encode(),
                            headers={'Content-Type':'application/json'},method='POST')
            with urlopen(request,timeout=3) as r:return r.status,json.load(r)
        with self.assertRaises(HTTPError) as ctx:call('wrong')
        self.assertEqual(ctx.exception.code,400)
        status,body=call('RemoveMe')
        self.assertEqual(status,200)
        self.assertTrue(body['deleted'])
        with urlopen(self.base+'/api/projects',timeout=3) as r:projects=json.load(r)['projects']
        self.assertFalse(any(p['id']==project['id'] for p in projects))
