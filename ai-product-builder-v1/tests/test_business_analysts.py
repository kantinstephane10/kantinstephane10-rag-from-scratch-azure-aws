"""Mandatory business-analyst workflow; no external GPT network calls."""
import tempfile
import unittest
from pathlib import Path
from engine import Engine
from store import Store, StoreError


class EchoSpecialists:
    def __init__(self): self.calls=[]
    def generate(self,role,model,instructions,context):
        self.calls.append({'role':role,'context':context})
        return {'reply':role+' reviewed requirements and found approval is not yet evidenced.',
                'outgoing':[],'meeting_requests':[],'needs_human':False}


class BusinessAnalystWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.db=str(Path(self.temp.name)/'state.sqlite3')
        self.store=Store(self.db)
        self.pid=self.store.create_project('Atlas RAG','Build a document RAG')['id']
        self.client=EchoSpecialists()
        self.engine=Engine(self.store,client=self.client,demo=True)

    def tearDown(self): self.temp.cleanup()

    def test_kickoff_queues_two_independent_business_analysts_before_pm(self):
        self.engine.kickoff(self.pid)
        tasks=self.store.activity(self.pid)['messages']
        self.assertEqual([t['recipient'] for t in tasks],
                         ['business-analyst-1','business-analyst-2','product-manager'])
        self.assertEqual([t['workflow_stage'] for t in tasks],
                         ['mandatory_business_analysis','mandatory_business_analysis','pm_kickoff'])
        with self.assertRaises(StoreError):self.engine.kickoff(self.pid)
        self.assertEqual(len(self.store.activity(self.pid)['messages']),3)

    def test_pm_does_not_start_until_both_analysts_finish(self):
        self.engine.kickoff(self.pid)
        self.assertTrue(self.engine.process_once())
        self.assertEqual([c['role'] for c in self.client.calls],['business-analyst-1'])
        self.assertTrue(self.engine.process_once())
        self.assertEqual([c['role'] for c in self.client.calls],
                         ['business-analyst-1','business-analyst-2'])
        self.assertTrue(self.engine.process_once())
        self.assertEqual(self.client.calls[2]['role'],'product-manager')
        reviews=self.client.calls[2]['context']['mandatory_business_analysis']
        self.assertEqual(len(reviews),2)
        self.assertTrue(all(x['status']=='done' and 'reviewed requirements' in x['findings']
                            for x in reviews))
        self.assertEqual(self.store.activity(self.pid)['counts']['done'],3)

    def test_pm_claim_rejected_while_analysis_is_processing(self):
        self.engine.kickoff(self.pid)
        first=self.store.claim()
        self.assertEqual(first['recipient'],'business-analyst-1')
        second=self.store.claim()
        self.assertEqual(second['recipient'],'business-analyst-2')
        self.assertIsNone(self.store.claim())
        self.store.complete(first['id'],'test','first analysis')
        self.assertIsNone(self.store.claim())
        self.store.complete(second['id'],'test','second analysis')
        self.assertEqual(self.store.claim()['recipient'],'product-manager')

    def test_failed_analyst_is_reported_instead_of_perpetual_queue(self):
        self.engine.kickoff(self.pid)
        first=self.store.claim()
        self.store.failure(first['id'],'Unavailable provider',retryable=False)
        self.engine.process_once()  # second analyst
        self.engine.process_once()  # PM sees failure
        review=self.client.calls[-1]['context']['mandatory_business_analysis']
        self.assertEqual(review[0]['status'],'failed')
        self.assertIn('Unavailable provider',review[0]['error'])
        self.assertEqual(review[1]['status'],'done')

    def test_kickoff_is_atomic_for_excessive_briefs(self):
        pid=self.store.create_project('Long project','X'*20000)['id']
        with self.assertRaises(StoreError):self.engine.kickoff(pid)
        self.assertEqual(self.store.activity(pid)['counts']['total'],0)
        self.assertEqual(self.store.project(pid)['status'],'draft')

    def test_schema_survives_restart(self):
        self.engine.kickoff(self.pid)
        reopened=Store(self.db)
        stages=[t['workflow_stage'] for t in reopened.activity(self.pid)['messages']]
        self.assertEqual(stages,['mandatory_business_analysis','mandatory_business_analysis','pm_kickoff'])


if __name__=='__main__': unittest.main()
