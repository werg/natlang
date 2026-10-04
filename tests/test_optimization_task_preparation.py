import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

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
            body = (json.dumps({'support':{'cases':[{}]*7}})+'\n').encode()
            digest = hashlib.sha256(body).hexdigest()
            (root / 'tasks').write_bytes(body)
            corpora = []
            for name, state, manifest in [
                ('episodes', 'prepared_requires_collection', {'episodes':1,'candidate_rows':7,'outputs':{'tasks':{'sha256':digest}}}),
                ('backing', 'source_backing_episodes', {'tasks':1,'sha256':digest}),
                ('translation', 'held_semantic_evaluator', {'task_count':1,'artifacts':{'tasks':{'sha256':digest}}})
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

if __name__ == '__main__':
    unittest.main()
