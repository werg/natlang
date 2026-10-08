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
        permission = receipt['unlink_permissions'][str(self.root / 'checkpoint.pt')]
        self.assertEqual(permission['parent'], str(self.root.resolve()))
        self.assertTrue(permission['write_and_search_access'])
        self.assertIn('--execute', receipt['execute_command'])
        self.assertTrue((self.root / 'checkpoint.pt').exists())
        sync.assert_not_called()

    def test_offload_uses_local_manifest_owner_and_allows_an_explicit_other_host(self):
        entry = {**self.entry, 'owner': 'dgx'}
        args = SimpleNamespace(machine='dgx', files=['checkpoint.pt'], host='pop-host',
                               remote_repo='/remote/repo', reserve_gib=0, execute=False)
        output = io.StringIO()
        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 0}), \
             contextlib.redirect_stdout(output):
            module.offload(self.repo, entry, self.manifest, args)
        preflight = json.loads(output.getvalue())
        self.assertEqual(preflight['machine'], 'dgx')
        self.assertIn('--machine dgx --host pop-host', preflight['execute_command'])
        self.assertIn('--machine dgx --host pop-host', preflight['restore'])

    def test_execute_transfers_and_verifies_subset_before_unlink_and_writes_restore_receipt(self):
        selected = module.select_manifest_files(self.manifest, ['checkpoint.pt'])
        remote = {'id': selected['id'], 'status': 'verified', 'verification_scope': 'ssh-remote',
                  'manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                  'verified_hostname': 'pop-host', 'machine_boot_id': 'remote-boot',
                  'verified_repo': '/remote/repo'}
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
             patch.object(module, 'machine_boot_id', return_value='local-boot'), \
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
        history = self.repo / '.coordination' / 'artifact-evictions' / 'history' / (receipt['verification_id'] + '.jsonl')
        states = [json.loads(line)['status'] for line in history.read_text().splitlines()]
        self.assertEqual(states, ['verified_not_unlinked', 'unlinked'])

    def test_execute_unlinks_selected_hardlinks_after_refreshing_only_the_remaining_inode_stats(self):
        checkpoint = self.root / 'checkpoint.pt'
        hardlink = self.root / 'checkpoint-hardlink.pt'
        hardlink.hardlink_to(checkpoint)
        files = [{'path': path.name, 'bytes': len(self.body),
                  'sha256': hashlib.sha256(self.body).hexdigest()} for path in (checkpoint, hardlink)]
        manifest = {**self.manifest, 'files': files, 'bytes': 2 * len(self.body)}
        self.args.files = [item['path'] for item in files]
        selected = module.select_manifest_files(manifest, self.args.files)
        remote = {'id': selected['id'], 'status': 'verified', 'verification_scope': 'ssh-remote',
                  'manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                  'verified_hostname': 'dgx-host', 'machine_boot_id': 'remote-boot',
                  'verified_repo': '/remote/repo'}
        self.args.execute = True
        output = io.StringIO()
        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 0}), \
             patch.object(module, 'machine_boot_id', return_value='local-boot'), \
             patch.object(module, 'sync', return_value=remote), contextlib.redirect_stdout(output):
            module.offload(self.repo, self.entry, manifest, self.args)
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'unlinked')
        self.assertFalse(checkpoint.exists())
        self.assertFalse(hardlink.exists())
        receipt = json.loads(Path(result['receipt']).read_text())
        self.assertEqual(receipt['removed_files'], ['checkpoint.pt', 'checkpoint-hardlink.pt'])
        self.assertEqual(receipt['status'], 'unlinked')

    def test_execute_rejects_real_content_change_after_remote_verification(self):
        selected = module.select_manifest_files(self.manifest, ['checkpoint.pt'])
        remote = {'id': selected['id'], 'status': 'verified', 'verification_scope': 'ssh-remote',
                  'manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                  'verified_hostname': 'dgx-host', 'machine_boot_id': 'remote-boot',
                  'verified_repo': '/remote/repo'}
        self.args.execute = True

        def transfer(*_args):
            (self.root / 'checkpoint.pt').write_bytes(b'external content change')
            return remote

        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 0}), \
             patch.object(module, 'machine_boot_id', return_value='local-boot'), \
             patch.object(module, 'sync', side_effect=transfer):
            with self.assertRaisesRegex(ValueError, 'local files or references changed after remote verification'):
                module.offload(self.repo, self.entry, self.manifest, self.args)
        self.assertEqual((self.root / 'checkpoint.pt').read_bytes(), b'external content change')

    def test_execute_refuses_unwritable_resolved_parent_before_transfer(self):
        self.args.execute = True
        real_access = module.os.access
        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 0}), \
             patch.object(module.os, 'access', side_effect=lambda path, mode: False
                          if mode & module.os.W_OK else real_access(path, mode)), \
             patch.object(module, 'sync') as sync:
            with self.assertRaisesRegex(ValueError, 'parent directories are not writable/searchable'):
                module.offload(self.repo, self.entry, self.manifest, self.args)
        sync.assert_not_called()
        self.assertTrue((self.root / 'checkpoint.pt').exists())

    def test_offload_receipt_retry_appends_prior_attempt_without_losing_latest_path(self):
        path = self.repo / '.coordination' / 'artifact-evictions' / 'same.json'
        first = {'schema': 'natlang.artifact-offload/1', 'verification_id': 'same', 'attempt_id': 'a', 'status': 'partial_unlink'}
        second = {'schema': 'natlang.artifact-offload/1', 'verification_id': 'same', 'attempt_id': 'b', 'status': 'unlinked'}
        module.save_offload_receipt(self.repo, path, first)
        module.save_offload_receipt(self.repo, path, second)
        self.assertEqual(json.loads(path.read_text()), second)
        history = self.repo / '.coordination' / 'artifact-evictions' / 'history' / 'same.jsonl'
        self.assertEqual([json.loads(line) for line in history.read_text().splitlines()], [first, second])

    def test_execute_refuses_to_unlink_when_ssh_verification_runs_on_same_kernel_boot(self):
        selected = module.select_manifest_files(self.manifest, ['checkpoint.pt'])
        remote = {'id': selected['id'], 'status': 'verified', 'verification_scope': 'ssh-remote',
                  'manifest_sha256': hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest(),
                  'verified_hostname': 'dgx-host', 'machine_boot_id': 'same-boot'}
        self.args.execute = True
        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 0}), \
             patch.object(module, 'machine_boot_id', return_value='same-boot'), \
             patch.object(module, 'sync', return_value=remote):
            with self.assertRaisesRegex(ValueError, 'same kernel boot'):
                module.offload(self.repo, self.entry, self.manifest, self.args)
        self.assertTrue((self.root / 'checkpoint.pt').exists())

    def test_ordinary_verification_keeps_historical_symlink_alias_support_but_offload_rejects_it(self):
        target = self.root / 'target.pt'
        target.write_bytes(self.body)
        alias = self.root / 'checkpoint.pt'
        alias.unlink()
        alias.symlink_to(target.name)
        errors, checked, _ = module.verify_files(self.root, [self.manifest['files'][0]])
        self.assertEqual(errors, [])
        self.assertEqual(checked, ['checkpoint.pt'])
        errors, _, _ = module.verify_files(self.root, [self.manifest['files'][0]], reject_symlinks=True)
        self.assertEqual(errors[0]['reason'], 'symlink')
        with patch.object(module, 'local_references', return_value={
                'open_fds': [], 'live_job_references': [], 'inaccessible_fd_directories': 0}):
            with self.assertRaisesRegex(ValueError, 'do not match'):
                module.offload(self.repo, self.entry, self.manifest, self.args)
        self.assertTrue(alias.is_symlink())

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

    def test_file_path_arguments_and_environment_protect_only_the_canonical_file(self):
        checkpoint = self.root / 'checkpoint.pt'
        best = self.root / 'best-checkpoint.pt'
        best.write_bytes(b'closed sibling')
        alias = self.repo / 'checkpoint-alias.pt'
        alias.symlink_to(checkpoint)
        with tempfile.TemporaryDirectory() as proc_temp:
            proc = Path(proc_temp)
            process = proc / '12345'
            (process / 'fd').mkdir(parents=True)
            (process / 'cmdline').write_bytes(f'worker\0--continue-from={alias}\0'.encode())
            (process / 'environ').write_bytes(f'CHECKPOINT={checkpoint}\0'.encode())
            (process / 'cwd').symlink_to(self.repo)
            refs = module.local_references([checkpoint, best], proc_root=proc)
        self.assertEqual(refs['open_fds'], [])
        self.assertEqual(refs['live_job_references'], [
            {'pid': 12345, 'path': str(checkpoint.resolve()), 'source': 'process-arguments-or-environment'}])

    def test_explicit_directory_path_operands_protect_selected_descendants(self):
        checkpoint = self.root / 'checkpoint.pt'
        best = self.root / 'best-checkpoint.pt'
        best.write_bytes(b'closed sibling')
        alias = self.repo / 'artifact-directory'
        alias.symlink_to(self.root, target_is_directory=True)
        with tempfile.TemporaryDirectory() as proc_temp:
            proc = Path(proc_temp)
            process = proc / '12345'
            (process / 'fd').mkdir(parents=True)
            (process / 'cmdline').write_bytes(f'worker\0--artifact-dir={alias}\0'.encode())
            (process / 'environ').write_bytes(f'RUN_DIRECTORY={self.root}\0'.encode())
            (process / 'cwd').symlink_to(self.repo)
            refs = module.local_references([checkpoint, best], proc_root=proc)
        self.assertEqual(refs['open_fds'], [])
        self.assertEqual(sorted(refs['live_job_references'], key=lambda row: row['path']), sorted([
            {'pid': 12345, 'path': str(checkpoint.resolve()), 'source': 'artifact-directory-reference'},
            {'pid': 12345, 'path': str(best.resolve()), 'source': 'artifact-directory-reference'}], key=lambda row: row['path']))

    def test_current_working_directory_keeps_the_existing_immediate_parent_rule(self):
        target = self.root / 'checkpoint.pt'
        with tempfile.TemporaryDirectory() as proc_temp:
            proc = Path(proc_temp)
            process = proc / '12345'
            (process / 'fd').mkdir(parents=True)
            (process / 'cmdline').write_bytes(b'worker\0')
            (process / 'environ').write_bytes(b'')
            (process / 'cwd').symlink_to(self.root)
            refs = module.local_references([target], proc_root=proc)
        self.assertEqual(refs['live_job_references'], [
            {'pid': 12345, 'path': str(target.resolve()), 'source': 'artifact-directory-reference'}])

    def test_local_reference_scan_ignores_long_nonpath_environment_data_without_echoing_it(self):
        target = self.root / 'checkpoint.pt'
        secret = 'PRIVATE_ENV_SENTINEL_' + ('x' * 10000)
        with tempfile.TemporaryDirectory() as proc_temp:
            proc = Path(proc_temp)
            process = proc / '12345'
            (process / 'fd').mkdir(parents=True)
            (process / 'cmdline').write_bytes(b'worker\0')
            (process / 'environ').write_bytes(('LS_COLORS=' + secret + '\0').encode())
            (process / 'cwd').symlink_to(self.repo)
            refs = module.local_references([target], proc_root=proc)
        self.assertEqual(refs['open_fds'], [])
        self.assertEqual(refs['live_job_references'], [])
        self.assertNotIn('PRIVATE_ENV_SENTINEL', repr(refs))

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
        registry.write_text(json.dumps({'corpora': [self.entry]}) + '\n')
        manifest_path.write_text(json.dumps(self.manifest) + '\n')
        committed_registry = json.dumps({'corpora': [self.entry]}, indent=2).encode() + b'\n'
        snapshots = [committed_registry, manifest_path.read_bytes()]
        results = [SimpleNamespace(returncode=0, stdout=data, stderr=b'') for data in snapshots]
        with patch.object(module.subprocess, 'run', side_effect=results):
            module.require_committed_offload_snapshot(self.repo, 'closed-checkpoint')
        registry.write_text(json.dumps({'corpora': [self.entry, {'id': 'unrelated', 'owner': 'dgx'}]}) + '\n')
        results = [SimpleNamespace(returncode=0, stdout=snapshots[0], stderr=b''),
                   SimpleNamespace(returncode=0, stdout=snapshots[1], stderr=b'')]
        with patch.object(module.subprocess, 'run', side_effect=results):
            module.require_committed_offload_snapshot(self.repo, 'closed-checkpoint')
        manifest_path.write_text('{}\n')
        results = [SimpleNamespace(returncode=0, stdout=snapshots[0], stderr=b''),
                   SimpleNamespace(returncode=0, stdout=snapshots[1], stderr=b'')]
        with patch.object(module.subprocess, 'run', side_effect=results):
            with self.assertRaisesRegex(ValueError, 'manifest unchanged at HEAD'):
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
