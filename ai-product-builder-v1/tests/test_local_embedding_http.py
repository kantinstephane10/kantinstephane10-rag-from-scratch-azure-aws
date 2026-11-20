"""Offline HTTP smoke test for the dashboard's local RAG indexing route."""
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request,urlopen

from engine import Engine
from handoffs import save_artifact
from large_rag import start_large_index_job
from server import make_server
from store import Store


def fake_embeddings(texts):
    return [[float(len(x) % 10+1),float('azure' in x.lower())+1,float('network' in x.lower())+1] for x in texts]

class LocalEmbeddingHttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.tmp.name)/'state.sqlite'))
        self.engine=Engine(self.store,client=object(),demo=True)
        self.server=make_server(self.store,self.engine,port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_address[1])
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(3);self.tmp.cleanup()

    def test_index_is_available_without_an_openai_key_in_demo_mode(self):
        project=self.store.create_project('Local HF','Offline',intake={'product_type':'rag'})
        pid=project['id']
        save_artifact(self.store,pid,'document.txt',b'Azure network connectivity uses VNets and subnets.')
        def deterministic_start(store,pid,key=None,**kwargs):
            return start_large_index_job(store,pid,key,embedder=fake_embeddings,background=False,**kwargs)
        req=Request(self.base+f'/api/projects/{pid}/rag/index',data=b'{}',
                    headers={'Content-Type':'application/json'},method='POST')
        with patch('server.start_large_index_job',side_effect=deterministic_start) as called:
            with urlopen(req,timeout=8) as response:
                self.assertEqual(response.status,202)
                response_body=json.load(response)
        self.assertEqual(response_body['status'],'complete')
        self.assertIsNone(called.call_args.args[2])
        with urlopen(self.base+f'/api/projects/{pid}/rag/status',timeout=5) as response:
            status=json.load(response)
        self.assertTrue(status['indexed'])
        self.assertTrue(status['embedding_model'].startswith('sentence-transformers/'))
        self.assertGreater(status['chunk_count'],0)
        with urlopen(self.base+'/') as response:
            markup=response.read()
        self.assertIn(b'hf-embedding-model',markup)
        self.assertIn(b'rag-resume',markup)
