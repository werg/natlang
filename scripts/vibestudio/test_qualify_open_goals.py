import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from qualify_open_goals import main


class QualificationBatchesTest(unittest.TestCase):
    def test_recycles_instances_and_continues_after_a_failed_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / 'source.jsonl'
            rows = [{
                'scenario': f'open-goal-case-{index}',
                'goalSha256': f'{index:064x}',
                'interaction': 'open-user-goal',
                'family': 'workspace-open-user-goal-demo',
                'split': 'train',
                'workspace': 'examples',
                'sourceGroups': [f'demo-{index}'],
                **({'workspaceBaseline': 'a' * 40} if index == 13 else {}),
            } for index in range(14)]
            catalog.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            out = root / 'out'
            argv = ['qualify_open_goals', '--host', str(root), '--catalog', str(catalog),
                    '--out', str(out)]
            with patch.object(sys, 'argv', argv), patch('qualify_open_goals.qualify_group',
                                                         side_effect=[False, True, True]) as run:
                self.assertEqual(main(), 1)

            calls = run.call_args_list
            self.assertEqual([len(call.args[3]) for call in calls], [12, 1, 1])
            self.assertEqual([call.kwargs['batch'] for call in calls], [1, 2, None])
            self.assertEqual([call.args[2] for call in calls], ['', '', 'a' * 40])
            self.assertEqual([case for call in calls for case in call.args[3]],
                             [f'open-goal-oracle-case-{index}' for index in range(14)])
            self.assertEqual((out / 'catalog.jsonl').read_bytes(), catalog.read_bytes())


if __name__ == '__main__':
    unittest.main()
