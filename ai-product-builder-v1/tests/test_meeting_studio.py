import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from architecture import blueprint
from engine import Engine
from meetings import (create_meeting, meeting, list_meetings, claim_turn,
                      finish_turn, fail_turn, take_floor, yield_floor, end_meeting, cancel_meeting,
                      start_meeting, activate_due, recover_meetings)
from roles import ROLES
from server import make_server
from store import Store,StoreError


class FakeGPT:
    key='fake-api-key'
    def __init__(self):self.calls=[]
    def generate(self, role, model, instructions, context):
        self.calls.append((role,context))
        if context.get('phase')=='opening':
            return {'reply':'The PM opens the meeting.','outgoing':[{'recipient':'security-agent','message':'What security risks should we prioritize?'}],
                    'meeting_requests':[],'needs_human':False}
        return {'reply':f'{role} answers: {context.get("specific_question", "agenda")}',
                'outgoing':[],'meeting_requests':[],'needs_human':False}


class MeetingTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.db=str(Path(self.directory.name)/'state.sqlite')
        self.store=Store(self.db)
    def tearDown(self): self.directory.cleanup()
    def room(self,participants=None):
        return create_meeting(self.store,topic='Review our RAG design',participants=participants or ['rag-engineer'])

    def test_architecture_has_49_agents_and_nine_layers(self):
        b=blueprint();self.assertEqual(len(b['roles']),49);self.assertEqual(len(b['layers']),9)
        self.assertTrue(b['human_approval_gate'])
    def test_all_role_ids_are_unique(self):self.assertEqual(len(set(ROLES)),49)
    def test_pm_is_always_chair(self):
        r=self.room();self.assertEqual(r['chair'],'product-manager');self.assertEqual(r['participants'][0],'product-manager')
    def test_human_meeting_can_start_without_project(self):self.assertEqual(self.room()['status'],'live')
    def test_scheduled_meeting_is_not_claimable(self):
        import time
        r=create_meeting(self.store,topic='Later',participants=['rag-engineer'],starts_at=int(time.time())+3600,start_now=False)
        self.assertEqual(r['status'],'scheduled');self.assertIsNone(claim_turn(self.store))
    def test_unknown_participant_rejected(self):
        with self.assertRaises(StoreError):create_meeting(self.store,topic='x',participants=['ghost'])
    def test_agent_request_requires_project(self):
        with self.assertRaises(StoreError):create_meeting(self.store,topic='x',participants=['rag-engineer'],requested_by='dev-1')
    def test_human_can_address_any_agent_even_not_invited(self):
        r=self.room();take_floor(self.store,r['id'],human_message='Can we prevent prompt injection?',target='security-agent')
        current=meeting(self.store,r['id'])
        self.assertIn('security-agent',current['participants'])
        self.assertEqual(current['messages'][-1]['target'],'security-agent')
        self.assertEqual(current['messages'][-1]['kind'],'question')
        self.assertEqual(current['floor'],'human')
        self.assertIsNone(claim_turn(self.store),'No agent can speak while human has floor')
        yield_floor(self.store,r['id'])
        answer=claim_turn(self.store)
        self.assertEqual((answer['agent'],answer['phase']),('security-agent','response'))
        self.assertEqual(answer['reply_to'],'human')
        finish_turn(self.store,answer['id'],'We use least privilege.','test-model')
        current=meeting(self.store,r['id'])
        self.assertEqual(current['messages'][-1]['target'],'human')
        self.assertEqual(current['messages'][-1]['body'],'We use least privilege.')
    def test_agent_direct_question_gets_immediate_priority(self):
        r=self.room();t=claim_turn(self.store)
        ok=finish_turn(self.store,t['id'],'I will request security review.','test-model',
            questions=[{'recipient':'security-agent','message':'Please assess retrieval permissions.'}])
        self.assertTrue(ok)
        current=meeting(self.store,r['id'])
        self.assertIn('security-agent',current['participants'])
        q=current['messages'][-1]
        self.assertEqual((q['actor'],q['target'],q['kind']),('product-manager','security-agent','question'))
        answer=claim_turn(self.store)
        self.assertEqual((answer['agent'],answer['phase']),('security-agent','answer'))
        self.assertEqual(answer['prompt'],'Please assess retrieval permissions.')
        self.assertEqual(answer['question_message_id'],q['id'])
        finish_turn(self.store,answer['id'],'Check document-level ACLs.','test-model')
        self.assertEqual(meeting(self.store,r['id'])['messages'][-1]['target'],'product-manager')
    def test_question_raised_to_any_known_agent_without_roster_membership(self):
        r=self.room();t=claim_turn(self.store)
        finish_turn(self.store,t['id'],'Review please','x',[{'recipient':'dev-8','message':'What is your code status?'}])
        self.assertIn('dev-8',meeting(self.store,r['id'])['participants'])
        self.assertEqual(claim_turn(self.store)['agent'],'dev-8')
    def test_respondent_can_request_one_followup(self):
        r=self.room();t=claim_turn(self.store)
        finish_turn(self.store,t['id'],'Security review needed','x',[{'recipient':'security-agent','message':'Any risks?'}])
        a=claim_turn(self.store)
        finish_turn(self.store,a['id'],'Need help from data team','x',[{'recipient':'data-engineer','message':'What sources exist?'}])
        self.assertEqual(claim_turn(self.store)['agent'],'data-engineer')
    def test_chain_cannot_continue_indefinitely(self):
        r=self.room();t=claim_turn(self.store)
        finish_turn(self.store,t['id'],'ask','x',[{'recipient':'security-agent','message':'Ask security?'}])
        a=claim_turn(self.store)
        finish_turn(self.store,a['id'],'ask','x',[{'recipient':'data-engineer','message':'Ask data?'}])
        b=claim_turn(self.store)
        with self.assertRaises(StoreError):finish_turn(self.store,b['id'],'ask','x',[{'recipient':'dev-1','message':'Keep asking?'}])
        self.assertEqual(meeting(self.store,r['id'])['messages'][-1]['target'],'data-engineer')
    def test_bad_targets_or_self_questions_rejected_without_partial_publication(self):
        r=self.room();t=claim_turn(self.store);before=len(meeting(self.store,r['id'])['messages'])
        for recipient in ('ghost','product-manager'):
            with self.assertRaises(StoreError):finish_turn(self.store,t['id'],'should not publish','x',[{'recipient':recipient,'message':'Hello?'}])
            self.assertEqual(len(meeting(self.store,r['id'])['messages']),before)
    def test_no_more_than_one_directed_question_per_turn(self):
        r=self.room();t=claim_turn(self.store)
        with self.assertRaises(StoreError):finish_turn(self.store,t['id'],'text','x',[
            {'recipient':'dev-1','message':'one'}, {'recipient':'dev-2','message':'two'}])
    def test_human_floor_cancels_inflight_publication(self):
        r=self.room();t=claim_turn(self.store);take_floor(self.store,r['id'])
        self.assertFalse(finish_turn(self.store,t['id'],'STALE RESPONSE','gpt'))
        self.assertNotIn('STALE RESPONSE',[x['body'] for x in meeting(self.store,r['id'])['messages']])
        self.assertIsNone(claim_turn(self.store))
        yield_floor(self.store,r['id'])
        self.assertIsNotNone(claim_turn(self.store))

    def test_cancel_meeting_stops_pending_turns(self):
        r=self.room();take_floor(self.store,r['id'],human_message='Question',target='dev-1')
        cancel_meeting(self.store,r['id'], 'Operator changed priorities')
        current=meeting(self.store,r['id'])
        self.assertEqual(current['status'],'cancelled')
        self.assertIsNone(claim_turn(self.store))

    def test_end_meeting_drops_pending_questions(self):
        r=self.room();take_floor(self.store,r['id'],human_message='Question',target='dev-1')
        end_meeting(self.store,r['id']);self.assertIsNone(claim_turn(self.store))
        self.assertEqual(meeting(self.store,r['id'])['status'],'ended')
    def test_persisted_transcript_survives_restart(self):
        r=self.room();take_floor(self.store,r['id'],human_message='Is this durable?',target='dev-1')
        reopened=Store(self.db)
        self.assertEqual(meeting(reopened,r['id'])['messages'][-1]['body'],'Is this durable?')
    def test_processing_recovered_after_restart(self):
        r=self.room();claim_turn(self.store)
        self.assertEqual(recover_meetings(self.store),1)
        self.assertEqual(claim_turn(self.store)['agent'],'product-manager')
    def test_automatic_orchestration_from_gpt_question(self):
        r=self.room();gpt=FakeGPT();eng=Engine(self.store,client=gpt,demo=True)
        eng.process_once()  # PM asks security agent
        eng.process_once()  # security responds before RAG
        self.assertEqual([c[0] for c in gpt.calls[:2]],['product-manager','security-agent'])
        self.assertEqual(gpt.calls[1][1]['specific_question'],'What security risks should we prioritize?')
        self.assertIn('security-agent',meeting(self.store,r['id'])['participants'])
    def test_demo_directed_question_is_visible_but_not_misrepresented(self):
        from gpt_client import DemoClient
        answer=DemoClient().generate('security-agent','gpt-6-luna','',{
            'phase':'response','specific_question':'How do we secure RAG?',
            'question_asked_by':'human'})['reply']
        self.assertIn('How do we secure RAG?',answer)
        self.assertIn('OFFLINE DEMO',answer)
        self.assertIn('human',answer)

    def test_no_local_model_transport(self):
        from gpt_client import GPTClient
        self.assertIn('api.openai.com',GPTClient(api_key='x').endpoint)


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.store=Store(str(Path(self.tmp.name)/'db.sqlite'))
        self.engine=Engine(self.store,client=FakeGPT(),demo=True)
        self.server=make_server(self.store,self.engine,port=0)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
        self.base='http://127.0.0.1:'+str(self.server.server_address[1])
    def tearDown(self):self.server.shutdown();self.server.server_close();self.thread.join(2);self.tmp.cleanup()
    def get(self,path):
        with urlopen(self.base+path,timeout=3) as response:return response.status,response.read()
    def send(self,path,data={},origin=None):
        hdr={'Content-Type':'application/json'}
        if origin:hdr['Origin']=origin
        request=Request(self.base+path,json.dumps(data).encode('utf8'),hdr,method='POST')
        with urlopen(request,timeout=3) as response:return response.status,json.load(response)
    def test_dashboard_files_and_api(self):
        for p in ('/','/app.js','/app.css','/api/architecture','/api/health'):
            self.assertEqual(self.get(p)[0],200,p)
        self.assertIn(b'Meeting studio',self.get('/')[1])
    def test_live_meeting_http_ask_and_yield(self):
        _,r=self.send('/api/meetings',{'topic':'Review','participants':['rag-engineer']});mid=r['id']
        self.send('/api/meetings/'+mid+'/speak',{'message':'Please answer','target':'dev-8'})
        self.assertEqual(json.loads(self.get('/api/meetings/'+mid)[1])['floor'],'human')
        self.send('/api/meetings/'+mid+'/yield')
        t=claim_turn(self.store)
        self.assertEqual(t['agent'],'dev-8')

    def test_http_cancel_meeting(self):
        _,r=self.send('/api/meetings',{'topic':'Review','participants':['rag-engineer']});mid=r['id']
        status,payload=self.send('/api/meetings/'+mid+'/cancel',{'reason':'Operator changed priorities'})
        self.assertEqual(status,200)
        self.assertEqual(payload['status'],'cancelled')

    def test_cross_origin_writes_rejected(self):
        with self.assertRaises(HTTPError) as ctx:
            self.send('/api/meetings',{'topic':'x','participants':['dev-1']},origin='https://attacker.example')
        self.assertEqual(ctx.exception.code,403)
    def test_not_configured_does_not_pretend_live_gpt(self):
        from gpt_client import GPTClient,ModelError
        with self.assertRaises(ModelError):GPTClient(api_key='').generate('dev-1','gpt-6-luna','test',{})

if __name__=='__main__':unittest.main()
