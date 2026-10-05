"""Selected owner-side restoration cannot claim full-snapshot availability."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('sync_corpora', Path(__file__).parents[1] / 'scripts/sync_training_corpora.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RestoreTests(unittest.TestCase):
    def test_selected_owner_restore_pulls_and_uses_subset_receipt(self):
        entry = {'id': 'sample', 'path': 'runs/sample', 'owner': 'pop'}
        manifest = dict(entry, files=[{'path': 'weights.pt', 'bytes': 3, 'sha256': 'abc'},
                                     {'path': 'notes.json', 'bytes': 2, 'sha256': 'def'}], bytes=5)
        args = SimpleNamespace(action='restore', machine='pop', files=['weights.pt'],
                               remote_repo='/home/werg/natlang', host='dgx', reserve_gib=0)
        with patch.object(module.subprocess, 'run') as run, patch.object(module, 'verify') as verify:
            module.sync(Path('/local/repo'), entry, manifest, args)
        source_check = json.loads(run.call_args_list[0].kwargs['input'])
        self.assertEqual([f['path'] for f in source_check['files']], ['weights.pt'])
        self.assertEqual(run.call_args_list[2].args[0][-2], 'dgx:/home/werg/natlang/runs/sample/')
        subset = verify.call_args.args[1]
        self.assertEqual(subset['bytes'], 3)
        self.assertEqual(verify.call_args.kwargs['receipt_group'], 'corpus-restores')

    def test_missing_or_unsafe_restore_path_rejected_before_network(self):
        args = SimpleNamespace(action='restore', machine='pop', files=['../secret'], remote_repo='/repo')
        with patch.object(module.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                module.sync(Path('/local'), {'path': 'runs/a', 'owner': 'pop'}, {'files': []}, args)
            run.assert_not_called()


if __name__ == '__main__':
    unittest.main()
