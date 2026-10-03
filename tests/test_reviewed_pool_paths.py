import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import sys

from scripts.reviewed_pool_paths import ReviewedPoolPaths
from scripts import reviewed_single_pool
from scripts.reviewed_single_pool import validate_collector_capabilities


class ReviewedPoolPathsTests(unittest.TestCase):
    def test_campaign_paths_match_remote_sync_names(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = ReviewedPoolPaths.for_campaign(temp, control_name='control-v41')
            self.assertEqual(paths.jobs, Path(temp) / 'jobs')
            self.assertEqual(paths.output, Path(temp) / 'results.jsonl')
            self.assertEqual(paths.supervisor_journal.name, 'journal-pool.jsonl')
            self.assertEqual(paths.case_events.name, 'journal-cases.jsonl')
            self.assertEqual(paths.authorization, Path(temp) / 'bundle/root-pool-authorization.json')

    def test_runner_args_reject_stale_versioned_journal_names(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                journal=str(root / 'journal-pool-v40.jsonl'),
                jobs=str(root / 'jobs'), output=str(root / 'results.jsonl'),
                control_dir=str(root / 'control-v41'), status_file=str(root / 'worker-status.json'),
                case_events_file=str(root / 'journal-cases-v40.jsonl'),
                hard_abort_file=str(root / 'hard-abort.request'),
                authorization=str(root / 'bundle/auth.json'))
            with self.assertRaisesRegex(ValueError, 'journal-pool.jsonl'):
                ReviewedPoolPaths.from_runner_args(args)

    def test_runner_args_reject_misplaced_sync_artifacts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args = SimpleNamespace(
                journal=str(root / 'journal-pool.jsonl'), jobs=str(root / 'jobs'),
                output=str(root / 'results.jsonl'), control_dir=str(root / 'control'),
                status_file=str(root / 'control/worker-status.json'),
                case_events_file=str(root / 'journal-cases.jsonl'),
                hard_abort_file=str(root / 'hard-abort.request'),
                authorization=str(root / 'bundle/auth.json'))
            with self.assertRaisesRegex(ValueError, 'direct child'):
                ReviewedPoolPaths.from_runner_args(args)


class CollectorCapabilityTests(unittest.TestCase):
    def test_current_dist_implements_runner_contract(self):
        self.assertEqual(validate_collector_capabilities('ts-host')['unsupported_flags'], [])

    def test_old_runtime_missing_case_events_and_drain_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            teacher = Path(temp) / 'dist/teacher'
            teacher.mkdir(parents=True)
            (teacher / 'cli.js').write_text("const config = { contextTokens: 1, maxModelRequests: 1, transportRetries: 1, retryDelayMs: 1, executionPlans: true };\n")
            (teacher / 'collector.js').write_text("if (config.maxModelRequests) {} if (config.transportRetries) {} if (config.retryDelayMs) {} if (config.executionPlans) {}\n")
            with self.assertRaisesRegex(ValueError, '--case-events-file.*--final-export-only.*--drain-file'):
                validate_collector_capabilities(temp)


class PreflightOnlyTests(unittest.TestCase):
    def test_preflight_exception_does_not_publish_setup_failure(self):
        argv = [
            'reviewed_single_pool.py', '--ir', '/tmp/input.ir.jsonl', '--ir-sha256', '1' * 64,
            '--runtime', '/tmp/runtime', '--runtime-manifest', '/tmp/runtime/frozen-runtime.json',
            '--runtime-manifest-sha256', '2' * 64, '--authorization', '/tmp/campaign/bundle/auth.json',
            '--native-review', '/tmp/campaign/native.json', '--jobs', '/tmp/campaign/jobs',
            '--output', '/tmp/campaign/results.jsonl', '--control-dir', '/tmp/campaign/control',
            '--chat-request-config', '/tmp/campaign/chat.json', '--chat-request-config-sha256', '3' * 64,
            '--root-seed', '5', '--status-file', '/tmp/campaign/worker-status.json',
            '--journal', '/tmp/campaign/journal-pool.jsonl', '--case-events-file', '/tmp/campaign/journal-cases.jsonl',
            '--preflight-only']
        with mock.patch.object(sys, 'argv', argv), \
             mock.patch.object(reviewed_single_pool, 'run', side_effect=ValueError('expected preflight failure')) as run, \
             mock.patch.object(reviewed_single_pool, 'record_setup_failure') as publish:
            with self.assertRaisesRegex(ValueError, 'expected preflight failure'):
                reviewed_single_pool.main()
        run.assert_called_once()
        publish.assert_not_called()


if __name__ == '__main__':
    unittest.main()
