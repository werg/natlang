"""Selected owner-side restoration cannot claim full-snapshot availability."""
import importlib.util
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
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
        self.assertIn('-rzH', run.call_args_list[2].args[0])
        self.assertIn('--ignore-existing', run.call_args_list[2].args[0])
        subset = verify.call_args.args[1]
        self.assertEqual(subset['bytes'], 3)
        self.assertEqual(verify.call_args.kwargs['receipt_group'], 'corpus-restores')

    def test_missing_or_unsafe_restore_path_rejected_before_network(self):
        args = SimpleNamespace(action='restore', machine='pop', files=['../secret'], remote_repo='/repo')
        with patch.object(module.subprocess, 'run') as run:
            with self.assertRaises(ValueError):
                module.sync(Path('/local'), {'path': 'runs/a', 'owner': 'pop'}, {'files': []}, args)
            run.assert_not_called()


class OffloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name)
        self.root = self.repo / 'runs' / 'closed'
        self.root.mkdir(parents=True)
        self.body = b'checkpoint bytes'
        (self.root / 'checkpoint.pt').write_bytes(self.body)
        self.manifest = {
            'schema': 'natlang.corpus-snapshot/1', 'id': 'closed-checkpoint', 'owner': 'pop',
            'path': 'runs/closed', 'files': [
                {'path': 'checkpoint.pt', 'bytes': len(self.body), 'sha256': hashlib.sha256(self.body).hexdigest()},
                {'path': 'notes.json', 'bytes': 10, 'sha256': 'f' * 64},
            ], 'bytes': len(self.body) + 10,
        }
        self.entry = {'id': 'closed-checkpoint', 'owner': 'pop', 'path': 'runs/closed'}
        self.args = SimpleNamespace(machine='pop', files=['checkpoint.pt'], host='dgx', remote_repo='/remote/repo',
                                    reserve_gib=0, execute=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_preflight_is_exact_manifest_bound_and_never_unlinks(self):
        output = io.StringIO()
        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 2}), \
             patch.object(module, 'sync') as sync, contextlib.redirect_stdout(output):
            module.offload(self.repo, self.entry, self.manifest, self.args)
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt['status'], 'preflight_only')
        self.assertEqual([item['path'] for item in receipt['files']], ['checkpoint.pt'])
        self.assertTrue(receipt['local_sha256_verified'])
        self.assertIn('--execute', receipt['execute_command'])
        self.assertTrue((self.root / 'checkpoint.pt').exists())
        sync.assert_not_called()

    def test_execute_transfers_and_verifies_subset_before_unlink_and_writes_restore_receipt(self):
        selected = module.select_manifest_files(self.manifest, ['checkpoint.pt'])
        remote = {'id': selected['id'], 'status': 'verified', 'verification_scope': 'ssh-remote',
                  'manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                  'verified_hostname': 'dgx-host', 'verified_repo': '/remote/repo'}
        def transfer(_repo, _entry, _manifest, transfer_args):
            self.assertEqual(transfer_args.action, 'sync')
            self.assertEqual(transfer_args.machine, 'pop')
            self.assertEqual(transfer_args.files, ['checkpoint.pt'])
            self.assertEqual(transfer_args.remote_receipt_group, 'artifact-offload-verifications')
            self.assertTrue((self.root / 'checkpoint.pt').exists())
            return remote
        self.args.execute = True
        output = io.StringIO()
        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 0}), \
             patch.object(module.socket, 'gethostname', return_value='pop-host'), \
             patch.object(module, 'sync', side_effect=transfer), contextlib.redirect_stdout(output):
            module.offload(self.repo, self.entry, self.manifest, self.args)
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'unlinked')
        self.assertFalse((self.root / 'checkpoint.pt').exists())
        receipt = json.loads(Path(result['receipt']).read_text())
        self.assertEqual(receipt['status'], 'unlinked')
        self.assertEqual(receipt['remote_verification'], remote)
        self.assertEqual(receipt['local_hostname'], 'pop-host')
        self.assertIn('--file checkpoint.pt', receipt['restore'])

    def test_execute_refuses_to_unlink_when_ssh_verification_resolves_to_local_host(self):
        selected = module.select_manifest_files(self.manifest, ['checkpoint.pt'])
        remote = {'id': selected['id'], 'status': 'verified', 'verification_scope': 'ssh-remote',
                  'manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                  'verified_hostname': 'pop-host'}
        self.args.execute = True
        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 0}), \
             patch.object(module.socket, 'gethostname', return_value='pop-host'), \
             patch.object(module, 'sync', return_value=remote):
            with self.assertRaisesRegex(ValueError, 'resolved to this local host'):
                module.offload(self.repo, self.entry, self.manifest, self.args)
        self.assertTrue((self.root / 'checkpoint.pt').exists())

    def test_offload_rejects_unmanifested_paths_and_live_references_before_transfer(self):
        with patch.object(module, 'local_references', return_value={
                'open_fds': [{'pid': 7, 'fd': '3', 'path': '/tmp/checkpoint.pt'}],
                'live_job_references': [], 'inaccessible_fd_directories': 0}), patch.object(module, 'sync') as sync:
            with self.assertRaisesRegex(ValueError, 'unsafe relative path'):
                module.select_manifest_files(self.manifest, ['../checkpoint.pt'])
            with self.assertRaisesRegex(ValueError, 'still referenced'):
                module.offload(self.repo, self.entry, self.manifest, self.args)
        sync.assert_not_called()

    def test_local_reference_scan_reports_open_fds_and_does_not_fail_for_unrelated_inaccessible_processes(self):
        target = self.root / 'checkpoint.pt'
        with tempfile.TemporaryDirectory() as proc_temp:
            proc = Path(proc_temp)
            readable = proc / '12345'
            (readable / 'fd').mkdir(parents=True)
            (readable / 'fd' / '8').symlink_to(target)
            (readable / 'cmdline').write_bytes(b'worker\0')
            (readable / 'environ').write_bytes(b'')
            (readable / 'cwd').symlink_to(self.repo)
            unreadable = proc / '23456'
            unreadable.mkdir()
            refs = module.local_references([target], proc_root=proc)
        self.assertEqual(refs['open_fds'][0]['pid'], 12345)
        self.assertEqual(refs['inaccessible_fd_directories'], 1)

    def test_remote_verification_receipt_is_bound_to_selected_manifest_and_not_full_receipt(self):
        selected = module.select_manifest_files(self.manifest, ['checkpoint.pt'])
        expected_hash = hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()
        remote = {'id': selected['id'], 'status': 'verified', 'verification_scope': 'ssh-remote',
                  'manifest_sha256': expected_hash, 'verification_id': 'closed-checkpoint--a', 'checked': 1}
        with patch.object(module.subprocess, 'run', return_value=SimpleNamespace(stdout=json.dumps(remote), returncode=0)) as run:
            receipt = module.verify_remote(selected, SimpleNamespace(host='dgx', remote_repo='/remote/repo'),
                                           receipt_group='artifact-offload-verifications', receipt_key='closed-checkpoint--a')
        self.assertEqual(receipt, remote)
        payload = json.loads(run.call_args.kwargs['input'])
        self.assertEqual(payload['receipt_group'], 'artifact-offload-verifications')
        self.assertEqual(payload['receipt_key'], 'closed-checkpoint--a')
        self.assertEqual(payload['manifest']['files'], selected['files'])

    def test_remote_verification_failure_keeps_the_actual_remote_error(self):
        error = module.subprocess.CalledProcessError(1, ['ssh', 'dgx'])
        error.stderr = 'manifest file missing on DGX'
        with patch.object(module.subprocess, 'run', side_effect=error):
            with self.assertRaisesRegex(RuntimeError, 'manifest file missing on DGX'):
                module.verify_remote(self.manifest, SimpleNamespace(host='dgx', remote_repo='/remote/repo'))

    def test_execution_requires_registry_and_manifest_bytes_committed_at_head(self):
        registry = self.repo / 'training' / 'neuralese_corpora.json'
        manifest_path = self.repo / 'training' / 'corpus-manifests' / 'closed-checkpoint.json'
        registry.parent.mkdir(parents=True)
        manifest_path.parent.mkdir(parents=True)
        registry.write_text('{"corpora":[]}\n')
        manifest_path.write_text(json.dumps(self.manifest) + '\n')
        snapshots = [registry.read_bytes(), manifest_path.read_bytes()]
        results = [SimpleNamespace(returncode=0, stdout=data, stderr=b'') for data in snapshots]
        with patch.object(module.subprocess, 'run', side_effect=results):
            module.require_committed_offload_snapshot(self.repo, 'closed-checkpoint')
        manifest_path.write_text('{}\n')
        results = [SimpleNamespace(returncode=0, stdout=snapshots[0], stderr=b''),
                   SimpleNamespace(returncode=0, stdout=snapshots[1], stderr=b'')]
        with patch.object(module.subprocess, 'run', side_effect=results):
            with self.assertRaisesRegex(ValueError, 'committed, unchanged'):
                module.require_committed_offload_snapshot(self.repo, 'closed-checkpoint')

    def test_push_subset_uses_rsync_then_persists_a_separate_remote_subset_receipt(self):
        selected = module.select_manifest_files(self.manifest, ['checkpoint.pt'])
        receipt_key = 'closed-checkpoint--selection'
        remote = {'id': selected['id'], 'status': 'verified', 'verification_scope': 'ssh-remote',
                  'manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                  'verification_id': receipt_key, 'checked': 1}
        calls = [SimpleNamespace(stdout='', returncode=0), SimpleNamespace(stdout='', returncode=0),
                 SimpleNamespace(stdout=json.dumps(remote), returncode=0)]
        args = SimpleNamespace(action='sync', machine='pop', files=['checkpoint.pt'], host='dgx',
                               remote_repo='/remote/repo', reserve_gib=0,
                               remote_receipt_group='artifact-offload-verifications', remote_receipt_key=receipt_key)
        with patch.object(module.subprocess, 'run', side_effect=calls) as run:
            verified = module.sync(self.repo, self.entry, self.manifest, args)
        self.assertEqual(verified, remote)
        rsync = run.call_args_list[1].args[0]
        self.assertTrue(any(arg.startswith('--files-from=') for arg in rsync))
        self.assertEqual(rsync[-2], str(self.root) + '/')
        self.assertEqual(rsync[-1], 'dgx:/remote/repo/runs/closed/')
        payload = json.loads(run.call_args_list[2].kwargs['input'])
        self.assertEqual(payload['receipt_group'], 'artifact-offload-verifications')
        self.assertEqual(payload['receipt_key'], receipt_key)
        self.assertEqual(payload['manifest']['files'], selected['files'])


if __name__ == '__main__':
    unittest.main()
