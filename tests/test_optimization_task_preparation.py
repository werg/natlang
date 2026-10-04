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

if __name__ == '__main__':
    unittest.main()
