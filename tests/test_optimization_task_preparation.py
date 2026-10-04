import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from scripts.self_improvement_task_inventory import _canonical

from scripts.build_kernel_optimization_tasks import prepare
from scripts.self_improvement_task_inventory import inventory

class TaskPreparationTests(unittest.TestCase):
    def test_source_adapter_is_ast_only_and_groups_constant_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'LICENSE').write_text('MIT License\nfixture attribution\n')
            for level, constant in [('level1', 1), ('level2', 9)]:
                folder = root / 'KernelBench' / level
                folder.mkdir(parents=True)
                (folder / 'a.py').write_text(f'raise RuntimeError("never execute upstream")\nclass Model:\n def forward(self,x):\n  return x+{constant}\ndef get_inputs(): return [1]\ndef get_init_inputs(): return []\n')
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            subprocess.run(['git', '-C', str(root), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture'], check=True)
            revision, _, tasks, held = prepare(root, ['level1', 'level2'])
            self.assertEqual(len(tasks), 2)
            self.assertEqual(held, [])
            self.assertEqual(tasks[0]['group'], tasks[1]['group'])
            self.assertEqual(tasks[0]['split'], tasks[1]['split'])
            self.assertEqual(tasks[0]['source']['revision'], revision)
            (root / 'LICENSE').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'clean'):
                prepare(root, ['level1'])

    def test_inventory_requires_matching_audit_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = '{"task":1}\n'
            digest = hashlib.sha256(data.encode()).hexdigest()
            (root / 'tasks').write_text(data)
            (root / 'manifest').write_text(json.dumps({'tasks': 1, 'sha256': digest}))
            (root / 'audit').write_text(json.dumps({'input_sha256': digest, 'errors': []}))
            registry = {'corpora': [{'id':'test','tasks':'tasks','manifest':'manifest','audit':'audit','state':'prepared','sha256':digest}], 'backlog': []}
            self.assertEqual(inventory(root, registry)['errors'], [])
            (root / 'audit').write_text('{"errors":[]}')
            self.assertTrue(inventory(root, registry)['errors'])
            (root / 'audit').write_text(json.dumps({'input_sha256': 'other', 'errors': []}))
            self.assertTrue(inventory(root, registry)['errors'])

    def test_inventory_separates_backing_sources_and_held_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            body = (json.dumps({'support':{'cases':[{}]*7}})+'\n\n').encode()
            digest = hashlib.sha256(body).hexdigest()
            (root / 'tasks').write_bytes(body)
            corpora = []
            for name, state, manifest in [
                ('episodes', 'prepared_requires_collection', {'episodes':1,'candidate_rows':7,'outputs':{'tasks':{'sha256':digest}}}),
                ('backing', 'source_backing_episodes', {'tasks':1,'sha256':digest}),
                ('superseded', 'superseded_full_source_candidate', {'tasks':1,'sha256':digest}),
                ('translation', 'held_semantic_evaluator', {'candidate_counts':{'task_count':1},'artifacts':{'tasks':{'sha256':digest}}})
            ]:
                (root / name).write_text(json.dumps(manifest))
                corpora.append({'id':name,'state':state,'manifest':name,'tasks':'tasks','sha256':digest,'audit':None})
            report=inventory(root,{'corpora':corpora,'backlog':[]})
            self.assertEqual(report['errors'],[])
            self.assertEqual(report['totals']['active_prepared_episodes'],1)
            self.assertEqual(report['totals']['executor_pending_source_tasks'],7)
            self.assertEqual(report['held'],[{'id':'translation','state':'held_semantic_evaluator','episodes':0,'cases':1}])
            (root / 'translation').write_text(json.dumps({'task_count':2,'artifacts':{'tasks':{'sha256':digest}}}))
            self.assertTrue(inventory(root,{'corpora':corpora,'backlog':[]})['errors'])

    def _screened_collection_fixture(self, root):
        def write(path, content):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content if isinstance(content, bytes) else content.encode())
            return hashlib.sha256(path.read_bytes()).hexdigest()
        episodes = [
            {'id':'episode-a','family':'research:test','split':'train','support':{'cases':[{}]},'query':{'cases':[{}]},'provenance':{}},
            {'id':'episode-b','family':'research:test','split':'train','support':{'cases':[{}]},'query':{'cases':[{}]},'provenance':{}},
        ]
        input_text=''.join(_canonical(row)+'\n' for row in episodes)
        input_path=root/'input/episodes.jsonl'; input_sha=write(input_path,input_text)
        runtime=root/'runtime'; runtime_sha=write(runtime/'frozen-runtime.json','{"schema":"runtime"}\n')
        batch_ids=['episode-a','episode-b']; ids_path=root/'plan'/'batch-01.json'
        ids_sha=write(ids_path,json.dumps(batch_ids,separators=(',',':'))+'\n')
        plan={'schema':'natlang.headroom-batch-plan/1','input_sha256':input_sha,
            'collection_runtime_sha256':runtime_sha,'episodes':2,'batches':[{'id':'batch-01','ids':'batch-01.json','sha256':ids_sha,'episodes':2}]}
        plan_path=root/'plan'/'manifest.json'; plan_sha=write(plan_path,json.dumps(plan,separators=(',',':'))+'\n')
        screens=[
            {'episode':'episode-a','input_sha256':input_sha,'executor':'http://127.0.0.1:1:model','schema':'natlang.episode-headroom/2','screen':'fixed-screen','support_quality':0.5,'cases':1},
            {'episode':'episode-b','input_sha256':input_sha,'executor':'http://127.0.0.1:1:model','schema':'natlang.episode-headroom/2','screen':'fixed-screen','support_quality':1.0,'cases':1},
        ]
        selected={**episodes[0],'provenance':{'headroom':{'executor':screens[0]['executor'],'screen':'fixed-screen','support_quality':0.5,'band':[0,0.95]}}}
        kept_path=root/'kept/batch-01/kept.jsonl'; kept_sha=write(kept_path,_canonical(selected)+'\n')
        handoff={'schema':'natlang.screened-skill-handoff/1','input_sha256':input_sha,'runtime_manifest_sha256':runtime_sha,
            'screened_batch_ids_sha256':ids_sha,'screened':2,'screen_rows':screens,
            'screen_sha256':hashlib.sha256(json.dumps(screens,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'kept_sha256':kept_sha,'selected':1,'band':[0,0.95],'screen_identity':'fixed-screen'}
        handoff_path=root/'queue/batch-01/screen-handoff.json'; handoff_sha=write(handoff_path,json.dumps(handoff,separators=(',',':'))+'\n')
        screen_path=root/'screens/batch-01.jsonl'; screen_sha=write(screen_path,''.join(json.dumps(item,separators=(',',':'))+'\n' for item in screens))
        queue={'schema':'natlang.skill-authoring-queue/1','input_sha256':kept_sha,'runtime_manifest_sha256':runtime_sha,'episode_ids':['episode-a']}
        queue_path=root/'queue/batch-01/queue.json'; write(queue_path,json.dumps(queue,separators=(',',':'))+'\n')
        task_id=hashlib.sha256(b'episode-a').hexdigest()[:20]
        task=root/'queue/batch-01/tasks'/task_id
        write(task/'episode.jsonl',_canonical(selected)+'\n')
        result={'episode':'episode-a','disposition':'candidate_improved','positive':True}
        result_path=task/'attempt-001'/hashlib.sha256(b'episode-a').hexdigest()[:20]/'result.json'
        result_sha=write(result_path,json.dumps(result,separators=(',',':'))+'\n')
        state={'episode':'episode-a','terminal':True,'disposition':'candidate_improved','positive':True,
            'attempts':[{'attempt':1,'path':str(result_path),'sha256':result_sha,'disposition':'candidate_improved','positive':True}]}
        state_path=task/'state.json'; write(state_path,json.dumps(state,separators=(',',':'))+'\n')
        registry={'corpora':[],'backlog':[],'screened_collection_plans':[{'id':'screened-test','state':'candidate_collection',
            'plan_manifest':'plan/manifest.json','plan_manifest_sha256':plan_sha,'input_path':'input/episodes.jsonl','input_sha256':input_sha,
            'runtime_path':'runtime','runtime_manifest_sha256':runtime_sha,'batch_plan_dir':'plan','screen_identity':'fixed-screen',
            'batches':[{'id':'batch-01','kept_path':'kept/batch-01/kept.jsonl','kept_sha256':kept_sha,
                'queue_path':'queue/batch-01','screen_path':'screens/batch-01.jsonl','screen_sha256':screen_sha,
                'handoff_path':'queue/batch-01/screen-handoff.json','handoff_sha256':handoff_sha}]}]}
        return registry, state_path, kept_path

    def test_screened_plan_distinguishes_selected_terminal_positive_and_admission(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); registry,state_path,kept_path=self._screened_collection_fixture(root)
            result=inventory(root,registry)
            self.assertEqual(result['screened_collection_plans'][0]['selected'],1)
            self.assertEqual(result['screened_collection_plans'][0]['raw_screen_rows'],2)
            self.assertEqual(result['screened_collection_plans'][0]['handoff_screened'],2)
            self.assertEqual(result['screened_collection_plans'][0]['terminal'],1)
            self.assertEqual(result['screened_collection_plans'][0]['positive_candidates'],1)
            self.assertIsNone(result['screened_collection_plans'][0]['admitted_training_rows'])
            state_path.unlink()
            partial=inventory(root,registry)['screened_collection_plans'][0]
            self.assertEqual(partial['selected'],1)
            self.assertEqual(partial['terminal'],0)
            self.assertEqual(partial['positive_candidates'],0)
            self.assertEqual(partial['waiting_for_terminal'],1)
            self.assertEqual(partial['observed_state'],'collection_incomplete_or_waiting')
            kept_path.write_text(kept_path.read_text()+'{}\n')
            invalid=inventory(root,registry)['screened_collection_plans'][0]
            self.assertEqual(invalid['observed_state'],'unavailable_or_invalid')
            self.assertEqual(invalid['unavailable'],1)

    def test_supplemental_runtime_receipt_is_hash_and_identity_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); registry,_,_=self._screened_collection_fixture(root)
            row=registry['screened_collection_plans'][0]
            receipt={'schema':'natlang.collection-runtime-receipt/1','input_sha256':row['input_sha256'],
                     'runtime_manifest_sha256':row['runtime_manifest_sha256']}
            path=root/'runtime-receipt.json'; path.write_text(json.dumps(receipt))
            row['runtime_receipt_path']='runtime-receipt.json'
            row['runtime_receipt_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertNotEqual(inventory(root,registry)['screened_collection_plans'][0]['observed_state'],'unavailable_or_invalid')
            receipt['runtime_manifest_sha256']='wrong';path.write_text(json.dumps(receipt))
            row['runtime_receipt_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertIn('binding differs',inventory(root,registry)['screened_collection_plans'][0]['error'])

    def test_screened_plan_reports_missing_handoff_as_waiting_and_rejects_plan_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); registry,state_path,_=self._screened_collection_fixture(root)
            handoff=root/'queue/batch-01/screen-handoff.json'; handoff.unlink()
            registry['screened_collection_plans'][0]['batches'][0].pop('handoff_sha256')
            report=inventory(root,registry)['screened_collection_plans'][0]
            self.assertEqual(report['observed_state'],'waiting_for_screen_handoff')
            self.assertEqual(report['waiting_for_screen'],0)
            self.assertEqual(report['waiting_for_screen_handoff'],2)
            plan=root/'plan/manifest.json'; plan.write_text(plan.read_text()+' ')
            report=inventory(root,registry)['screened_collection_plans'][0]
            self.assertEqual(report['observed_state'],'unavailable_or_invalid')
            self.assertEqual(report['unavailable'],1)

    def test_inventory_accepts_audit_array_shape_in_corpus_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); data='{"id":"episode"}\n'; digest=hashlib.sha256(data.encode()).hexdigest()
            (root/'tasks.jsonl').write_text(data)
            (root/'manifest.json').write_text(json.dumps({'audit': [], 'episodes': 1, 'sha256': digest}))
            report=inventory(root,{'corpora':[{'id':'research-v3','tasks':'tasks.jsonl','manifest':'manifest.json',
                'sha256':digest,'state':'prepared_candidate'}],'backlog':[]})
            self.assertEqual(report['errors'],[])
            self.assertEqual(report['corpora'][0]['episodes'],1)

    def test_screened_handoff_without_registry_hash_pins_is_observed_unreviewed(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); registry,_,_=self._screened_collection_fixture(root)
            batch=registry['screened_collection_plans'][0]['batches'][0]
            batch.pop('handoff_sha256'); batch.pop('kept_sha256')
            report=inventory(root,registry)['screened_collection_plans'][0]
            self.assertEqual(report['unavailable'],0)
            self.assertEqual(report['unreviewed_handoffs'],1)
            self.assertEqual(report['batches'][0]['receipt_integrity'],'self_consistent_unpinned')
            self.assertEqual(report['batches'][0]['review_state'],'handoff_observed_unreviewed')
            self.assertIsNone(report['admitted_training_rows'])

    def test_inventory_checks_manifest_and_selection_ledger_hashes_without_claiming_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); tasks='{"task":"candidate"}\n'; ledger='{"source_id":"a","selected":true}\n{"source_id":"b","selected":false}\n'
            task_sha=hashlib.sha256(tasks.encode()).hexdigest()
            manifest={'tasks':1,'sha256':task_sha}; manifest_text=json.dumps(manifest)+'\n'
            (root/'tasks.jsonl').write_text(tasks); (root/'manifest.json').write_text(manifest_text); (root/'ledger.jsonl').write_text(ledger)
            registry={'corpora':[{'id':'held','tasks':'tasks.jsonl','manifest':'manifest.json','manifest_sha256':hashlib.sha256(manifest_text.encode()).hexdigest(),
                'sha256':task_sha,'audit':'ledger.jsonl','selection_ledger_sha256':hashlib.sha256(ledger.encode()).hexdigest(),
                'state':'held_review'}],'backlog':[]}
            result=inventory(root,registry)
            self.assertEqual(result['errors'],[])
            self.assertIn('semantic admission not inferred',result['corpora'][0]['audit'])
            self.assertEqual(result['totals']['active_prepared_episodes'],0)
            (root/'ledger.jsonl').write_text(ledger+'\n')
            self.assertTrue(inventory(root,registry)['errors'])

if __name__ == '__main__':
    unittest.main()
