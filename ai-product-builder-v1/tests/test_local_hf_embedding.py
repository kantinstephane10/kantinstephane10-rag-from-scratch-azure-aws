"""Offline tests for free local embeddings and incremental indexing.

No model downloads, OpenAI calls, or external network required.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hf_embeddings import DEFAULT_MODEL, MODELS, _load, embed_texts, validate_model
from handoffs import save_artifact
from large_rag import (init_large_rag, start_large_index_job, rag_status_large,
                       answer_large_question)
from runtime_settings import RuntimeSettings
from store import Store, StoreError


def fake_embeddings(texts):
    """Deterministic, stable, normalized vectors: injectable offline fixture."""
    return [[float(len(t) % 7+1),float('azure' in t.lower())+1,float('aws' in t.lower())+1]
            for t in texts]


class FakeSentenceTransformer:
    def encode(self, batch, **kwargs):
        return fake_embeddings(batch)


class HuggingFaceTests(unittest.TestCase):
    def test_embedding_model_allowlist_and_local_default(self):
        self.assertEqual(DEFAULT_MODEL,'sentence-transformers/all-MiniLM-L6-v2')
        self.assertIn('BAAI/bge-small-en-v1.5', MODELS)
        with self.assertRaises(StoreError): validate_model('untrusted/model')

    def test_automatic_local_ocr_never_calls_openai_vision(self):
        from large_rag import _ocr_image
        with patch('large_rag._local_ocr',return_value=('', 'none')), patch('large_rag._vision_ocr') as vision:
            with self.assertRaises(StoreError):
                _ocr_image(b'fake', 'auto', 'fake-key', 'gpt-vision')
            vision.assert_not_called()

    def test_local_inference_without_openai(self):
        with patch('hf_embeddings._load', return_value=FakeSentenceTransformer()) as load:
            out=embed_texts(['Azure VNet','AWS VPC'],DEFAULT_MODEL)
            self.assertEqual(len(out),2)
            self.assertEqual(len(out[0]),3)
            self.assertEqual(load.call_count,1)

    def test_bge_query_instruction_not_in_document_vectors(self):
        with patch('hf_embeddings._load', return_value=FakeSentenceTransformer()):
            doc=embed_texts(['AWS IAM'],'BAAI/bge-small-en-v1.5')
            query=embed_texts(['AWS IAM'],'BAAI/bge-small-en-v1.5',query=True)
            self.assertNotEqual(doc,query)


class DiskBackedRagTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.tmp.name)/'state.sqlite'))
        init_large_rag(self.store)
        self.project=self.store.create_project('RAG','Build test',intake={'product_type':'rag','provider_consent':True})
        self.pid=self.project['id']
        save_artifact(self.store,self.pid,'cloud.txt',b'Page 1 - Azure\nAzure VNet is private networking.\nPage 2 - AWS\nAWS VPC provides networking.')

    def tearDown(self): self.tmp.cleanup()

    def test_build_and_retrieve_without_openai_embedding_api(self):
        result=start_large_index_job(self.store,self.pid,None,embedder=fake_embeddings,background=False)
        self.assertEqual(result['status'],'complete')
        st=rag_status_large(self.store,self.pid)
        self.assertTrue(st['indexed'])
        self.assertEqual(st['embedding_model'],DEFAULT_MODEL)
        self.assertGreater(st['chunk_count'],0)
        answer=answer_large_question(self.store,self.pid,'Azure VNet',None,'test-chat',
                  embedder=fake_embeddings,responder=lambda q,docs:'Azure VNet is private networking.')
        self.assertIn('VNet',answer['answer'])
        self.assertTrue(answer['sources'])
        self.assertEqual(answer['sources'][0]['file'],'cloud.txt')

    def test_local_index_even_without_provider_consent(self):
        p=self.store.create_project('offline','local content',intake={'product_type':'rag'})
        save_artifact(self.store,p['id'],'sample.txt',b'Azure RBAC controls access.')
        result=start_large_index_job(self.store,p['id'],None,embedder=fake_embeddings,background=False)
        self.assertEqual(result['status'],'complete')
        with self.assertRaisesRegex(StoreError,'consent'):
            answer_large_question(self.store,p['id'],'Azure RBAC',None,'gpt-test',
                      embedder=fake_embeddings,responder=lambda q,docs:'should never run')

    def test_model_selection_is_saved_and_project_isolation(self):
        cfg=RuntimeSettings(Path(self.tmp.name))
        cfg.update({'embedding_model':'BAAI/bge-small-en-v1.5','ocr_mode':'rapidocr'})
        reopened=RuntimeSettings(Path(self.tmp.name))
        self.assertEqual(reopened.snapshot()['embedding_model'],'BAAI/bge-small-en-v1.5')
        self.assertEqual(reopened.snapshot()['ocr_mode'],'rapidocr')
        with self.assertRaises(StoreError):cfg.update({'embedding_model':'invalid/model'})
        start_large_index_job(self.store,self.pid,None,embedding_model='BAAI/bge-small-en-v1.5',
                     embedder=fake_embeddings,background=False)
        self.assertEqual(rag_status_large(self.store,self.pid)['embedding_model'],'BAAI/bge-small-en-v1.5')

    def test_deleting_project_purges_new_index_tables(self):
        start_large_index_job(self.store,self.pid,None,embedder=fake_embeddings,background=False)
        deleted=self.store.delete_project(self.pid,'RAG')
        self.assertTrue(deleted['deleted'])
        with self.store.db() as db:
            for name in ('rag_v2_chunks','rag_v2_indexes','rag_v2_jobs'):
                self.assertEqual(db.execute(f'SELECT COUNT(*) FROM {name} WHERE project_id=?',(self.pid,)).fetchone()[0],0)
            self.assertEqual(db.execute('SELECT COUNT(*) FROM rag_v2_fts').fetchone()[0],0)
