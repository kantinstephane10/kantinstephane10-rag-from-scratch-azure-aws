"""Regression coverage for 1 GiB streaming uploads and scoped agent-visible content."""
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError

from engine import Engine
from gpt_client import parse_response
from handoffs import (MAX_ARTIFACT_BYTES, save_artifact_stream, read_artifact_excerpt,
    agent_file_context, save_meeting_artifact_stream, read_meeting_artifact_excerpt,
    meeting_artifacts, artifact_file)
from meetings import create_meeting
from server import make_server
from store import Store, StoreError


class FakeReaderClient:
    key='test-key'
    def __init__(self):
        self.contexts=[]
    def generate(self,role,model,instructions,context):
        self.contexts.append(context)
        files=context.get('authorized_files',{}).get('files',[])
        if files and 'file_read_results' not in context:
            return {'reply':'Need source section','outgoing':[], 'meeting_requests':[],
                    'needs_human':False,'file_requests':[
                        {'artifact_id':files[0]['artifact_id'],'page':None,
                         'offset':0,'query':'distinctive fact'}]}
        return {'reply':'Verified citation from uploaded file: '+str(context.get('file_read_results'))[:200],
                'outgoing':[],'meeting_requests':[],'needs_human':False,'file_requests':[]}


class StreamingFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.tmp.name)/'builder.sqlite'))
        self.pid=self.store.create_project('Azure AWS','Plan RAG with attached document',
                    intake={'product_type':'rag','provider_consent':True})['id']
    def tearDown(self):self.tmp.cleanup()

    def test_max_upload_is_one_gib(self):
        self.assertEqual(MAX_ARTIFACT_BYTES,1024**3)
        with self.assertRaises(StoreError):
            save_artifact_stream(self.store,self.pid,'too-big.txt',io.BytesIO(b'x'),1024**3+1)
        self.assertFalse((Path(self.tmp.name)/'artifacts'/self.pid).exists())

    def test_truncated_stream_does_not_publish_metadata(self):
        with self.assertRaises(StoreError):
            save_artifact_stream(self.store,self.pid,'short.txt',io.BytesIO(b'ab'),900)
        self.assertEqual(agent_file_context(self.store,self.pid)['files'],[])

    def test_stream_larger_than_old_eight_megabyte_limit(self):
        raw=b'distinctive fact: the Azure policy is configured.\n' + b'x'*(9*1024*1024)
        receipt=save_artifact_stream(self.store,self.pid,'large.txt',io.BytesIO(raw),len(raw))
        self.assertEqual(receipt['size'],len(raw))
        text=read_artifact_excerpt(self.store,self.pid,receipt['id'],query='distinctive fact')['text']
        self.assertIn('distinctive fact',text)
        with self.assertRaises(StoreError):
            from handoffs import read_artifact
            read_artifact(self.store,self.pid,receipt['id'])

    def test_cross_project_access_denied_and_tampering_detected(self):
        receipt=save_artifact_stream(self.store,self.pid,'facts.txt',io.BytesIO(b'distinctive fact'),16)
        other=self.store.create_project('Other','Confidential project')['id']
        with self.assertRaises(StoreError):
            read_artifact_excerpt(self.store,other,receipt['id'])
        _,path=artifact_file(self.store,self.pid,receipt['id'])
        path.write_bytes(b'confidential edit')
        with self.assertRaises(StoreError):
            read_artifact_excerpt(self.store,self.pid,receipt['id'])

    def test_linked_meeting_upload_accessible_to_project_agents(self):
        room=create_meeting(self.store,topic='Review data',participants=['rag-engineer'],project_id=self.pid)
        receipt=save_meeting_artifact_stream(self.store,room['id'],'minutes.txt',
                              io.BytesIO(b'distinctive fact from meeting'),29)
        self.assertEqual(len(meeting_artifacts(self.store,room['id'])),1)
        self.assertIn('distinctive fact',read_meeting_artifact_excerpt(self.store,room['id'],
                    receipt['meeting_attachment_id'])['text'])
        self.assertTrue(any(x['name']=='minutes.txt' for x in agent_file_context(self.store,self.pid)['files']))

    def test_standalone_meeting_is_isolated(self):
        room=create_meeting(self.store,topic='Standalone',participants=['rag-engineer'])
        upload=b'distinctive fact from standalone'
        receipt=save_meeting_artifact_stream(self.store,room['id'],'meeting.txt',io.BytesIO(upload),len(upload))
        self.assertIn('distinctive fact',read_meeting_artifact_excerpt(self.store,room['id'],receipt['id'])['text'])
        self.assertEqual(agent_file_context(self.store,pid=self.pid)['files'],[])

    def test_every_registered_agent_role_can_request_project_file(self):
        from roles import ROLES
        raw=b'distinctive fact: cloud governance is required.'
        save_artifact_stream(self.store,self.pid,'governance.txt',io.BytesIO(raw),len(raw))
        for role in ROLES:
            client=FakeReaderClient()
            engine=Engine(self.store,client=client,demo=True)
            result=engine._generate_with_files(role,'demo','Use authorized evidence',
                          {'authorized_files':agent_file_context(self.store,pid=self.pid)},pid=self.pid)
            self.assertIn('governance',result['reply'],role)
            self.assertEqual(len(client.contexts),2,role)

    def test_reading_multiple_text_offsets_does_not_skip_data(self):
        raw=b'abcdef' * 6000
        receipt=save_artifact_stream(self.store,self.pid,'long.txt',io.BytesIO(raw),len(raw))
        one=read_artifact_excerpt(self.store,self.pid,receipt['id'],max_chars=1000)
        two=read_artifact_excerpt(self.store,self.pid,receipt['id'],offset=one['next_offset'],max_chars=1000)
        self.assertEqual((one['text']+two['text']).encode(),raw[:2000])

    def test_real_worker_reads_uploaded_content(self):
        upload=b'distinctive fact: the approved service is Azure Firewall'
        save_artifact_stream(self.store,self.pid,'cloud.txt',io.BytesIO(upload),len(upload))
        task=self.store.send(self.pid,'human','security-agent','Check distinctive fact from the uploaded document')
        client=FakeReaderClient()
        engine=Engine(self.store,client=client,demo=True)
        self.assertTrue(engine.process_once())
        self.assertEqual(self.store.message(task['id'])['status'],'done')
        self.assertEqual(len(client.contexts),2)
        self.assertIn('Azure Firewall',client.contexts[1]['file_read_results'][0]['text'])

    def test_meeting_agent_reads_standalone_file(self):
        room=create_meeting(self.store,topic='Check uploaded file',participants=['security-agent'])
        upload=b'distinctive fact: meeting file is readable'
        save_meeting_artifact_stream(self.store,room['id'],'room.txt',io.BytesIO(upload),len(upload))
        client=FakeReaderClient();engine=Engine(self.store,client=client,demo=True)
        self.assertTrue(engine.process_once())
        self.assertEqual(len(client.contexts),2)
        self.assertIn('meeting file',client.contexts[1]['file_read_results'][0]['text'])

    def test_scoped_pdf_attachment_receipt(self):
        # Receipt storage is independent of availability of the PDF parser.
        room=create_meeting(self.store,topic='PDF review',participants=['rag-engineer'],project_id=self.pid)
        receipt=save_meeting_artifact_stream(self.store,room['id'],'review.pdf',
                                  io.BytesIO(b'%PDF-1.4\nplaceholder'),20)
        self.assertEqual(receipt['name'],'review.pdf')
        self.assertEqual(len(agent_file_context(self.store,pid=self.pid)['files']),1)


class HttpFileTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=Store(str(Path(self.tmp.name)/'builder.sqlite'))
        self.pid=self.store.create_project('Storage','Attached TXT reference')['id']
        self.engine=Engine(self.store,client=FakeReaderClient(),demo=True)
        self.server=make_server(self.store,self.engine,port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True)
        self.thread.start();self.base=f'http://127.0.0.1:{self.server.server_address[1]}'
    def tearDown(self):
        self.server.shutdown();self.server.server_close();self.thread.join(2);self.tmp.cleanup()
    def test_http_stream_then_read_text(self):
        payload=b'distinctive fact is 2026-11-20'
        req=Request(f'{self.base}/api/projects/{self.pid}/artifacts/upload?name=source.txt',
                    data=payload,method='PUT',headers={'Content-Type':'application/octet-stream'})
        with urlopen(req,timeout=6) as response:self.assertEqual(response.status,201);receipt=json.load(response)
        with urlopen(f'{self.base}/api/projects/{self.pid}/artifacts/{receipt["id"]}/text',timeout=6) as response:
            excerpt=json.load(response)
        self.assertEqual(excerpt['text'],payload.decode())
    def test_http_invalid_oversize_rejected_before_body(self):
        req=Request(f'{self.base}/api/projects/{self.pid}/artifacts/upload?name=source.txt',
                data=b'x',method='PUT',headers={'Content-Length':str(MAX_ARTIFACT_BYTES+1)})
        # urllib can normalize Content-Length. Direct streaming validator is also covered above.
        try:
            with urlopen(req,timeout=6) as resp:self.assertNotEqual(resp.status,201)
        except HTTPError as exc:self.assertIn(exc.code,(400,413))
    def test_cross_origin_write_rejected(self):
        req=Request(f'{self.base}/api/projects/{self.pid}/artifacts/upload?name=source.txt',
                data=b'hello',method='PUT',headers={'Origin':'https://malicious.example'})
        with self.assertRaises(HTTPError) as error:urlopen(req,timeout=6)
        self.assertEqual(error.exception.code,403)
    def test_meeting_http_upload_list_and_read(self):
        room=create_meeting(self.store,topic='Document review',participants=['rag-engineer'],project_id=self.pid)
        raw=b'distinctive fact, available in meeting'
        req=Request(f'{self.base}/api/meetings/{room["id"]}/artifacts/upload?name=review.txt',
            data=raw,method='PUT')
        with urlopen(req,timeout=6) as response:uploaded=json.load(response)
        with urlopen(f'{self.base}/api/meetings/{room["id"]}/artifacts',timeout=6) as response:
            files=json.load(response)['artifacts']
        self.assertEqual(len(files),1)
        with urlopen(f'{self.base}/api/meetings/{room["id"]}/artifacts/{uploaded["meeting_attachment_id"]}/text',timeout=6) as response:
            excerpt=json.load(response)
        self.assertIn('distinctive fact',excerpt['text'])
    def test_ui_shows_upload_and_resume_buttons(self):
        with urlopen(self.base,timeout=6) as response:html=response.read().decode()
        self.assertIn('meeting-upload',html)
        self.assertIn('project-upload-progress',html)
        with urlopen(self.base+'/app.js',timeout=6) as response:script=response.read().decode()
        self.assertIn('Start agents',script)
        self.assertIn('XMLHttpRequest',script)

if __name__=='__main__':unittest.main()
