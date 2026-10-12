import hashlib
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
import start_reviewed_luna_campaign as campaign


class CampaignStopTest(unittest.TestCase):
    def test_worker_command_honors_reviewed_execution_plan_setting(self):
        worker = {'queue': '/q/queue.jsonl', 'journal': '/q/journal.jsonl'}
        common = {'supervisor': '/repo/scripts/run_bonsai_queue.py',
                  'reasoning_effort': 'medium'}
        no_planner = campaign.worker_command({**common, 'execution_plans': False}, worker, '/runtime')
        self.assertNotIn('--execution-plans', no_planner)
        self.assertIn('--reasoning-effort', no_planner)
        planner = campaign.worker_command({**common, 'execution_plans': True}, worker, '/runtime')
        self.assertIn('--execution-plans', planner)
        legacy_default = campaign.worker_command(common, worker, '/runtime')
        self.assertIn('--execution-plans', legacy_default)

    def test_worker_command_rejects_non_boolean_execution_plan_setting(self):
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            campaign.worker_command({'supervisor': 'run.py', 'execution_plans': 'false'},
                                     {'queue': 'q', 'journal': 'j'}, '/runtime')

    def test_operator_interrupt_updates_receipt_and_waits_for_worker_cleanup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'cases.jsonl'
            source.write_text('{}\n')
            queue = root / 'queue.jsonl'
            queue.write_text(json.dumps({'key': 'one', 'source': str(source), 'index': 0}) + '\n')
            runtime = root / 'runtime'
            runtime.mkdir()
            (runtime / 'frozen-runtime.json').write_text('{"files":{}}')
            record = root / 'launch.json'
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'root_approved': True, 'cwd': str(root),
                'runtime': str(runtime), 'supervisor': 'unused.py', 'launch_record': str(record),
                'artifact_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (source, queue)},
                'workers': [{'queue': str(queue), 'journal': str(root / 'journal.jsonl'),
                             'log': str(root / 'supervisor.log')}]}))
            digest = hashlib.sha256(plan.read_bytes()).hexdigest()
            worker = Mock(pid=123)
            worker.poll.side_effect = KeyboardInterrupt()
            worker.wait.return_value = 130
            def kill_group(pid, signum):
                if signum == 0:
                    raise ProcessLookupError()
            previous = signal.getsignal(signal.SIGTERM)
            try:
                with patch.object(sys, 'argv', ['campaign', str(plan), '--sha256', digest]), \
                     patch.object(campaign, 'tree_identity', return_value={}), \
                     patch.object(campaign.subprocess, 'Popen', return_value=worker), \
                     patch.object(campaign.os, 'killpg', side_effect=kill_group) as killpg:
                    self.assertEqual(campaign.main(), 130)
            finally:
                signal.signal(signal.SIGTERM, previous)
            killpg.assert_any_call(123, signal.SIGTERM)
            killpg.assert_any_call(123, 0)
            worker.wait.assert_called_once_with()
            receipt = json.loads(record.read_text())
            self.assertEqual(receipt['status'], 'operator_stopped')
            self.assertIn('unscored', receipt['disposition'])

    def test_late_pin_failure_blocks_only_that_lane_and_preserves_started_sibling(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'cases.jsonl'
            source.write_text('{}\n')
            runtime = root / 'runtime'
            runtime.mkdir()
            (runtime / 'frozen-runtime.json').write_text('{"files":{}}')
            workers = []
            pinned = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
            for lane in (1, 2):
                queue = root / f'worker-{lane:02d}' / 'queue.jsonl'
                queue.parent.mkdir()
                queue.write_text(json.dumps({'key': f'case-{lane}', 'source': str(source),
                                             'index': lane - 1}) + '\n')
                pinned[str(queue)] = hashlib.sha256(queue.read_bytes()).hexdigest()
                workers.append({'number': lane, 'queue': str(queue),
                                'journal': str(queue.parent / 'journal.jsonl'),
                                'log': str(queue.parent / 'worker.log')})
            record = root / 'campaign-launch.json'
            plan = root / 'plan.json'
            plan.write_text(json.dumps({'root_approved': True, 'cwd': str(root),
                'runtime': str(runtime), 'supervisor': 'run_bonsai_queue.py',
                'launch_record': str(record), 'artifact_hashes': pinned,
                'workers': workers}))
            plan_sha = hashlib.sha256(plan.read_bytes()).hexdigest()
            process = Mock(pid=321)
            process.poll.return_value = 0
            previous = signal.getsignal(signal.SIGTERM)
            try:
                with patch.object(sys, 'argv', ['campaign', str(plan), '--sha256', plan_sha]), \
                     patch.object(campaign, 'tree_identity', return_value={}), \
                     patch.object(campaign, 'verify_execution_pins',
                                  side_effect=[None, ValueError('reviewed starter hash changed')]), \
                     patch.object(campaign.subprocess, 'Popen', return_value=process) as popen, \
                     patch.object(campaign.os, 'killpg') as killpg:
                    self.assertEqual(campaign.main(), 1)
            finally:
                signal.signal(signal.SIGTERM, previous)
            popen.assert_called_once()
            killpg.assert_not_called()
            receipt = json.loads(record.read_text())
            self.assertEqual(receipt['status'], 'finished_with_blocked_approval_pins')
            self.assertEqual(receipt['blocked_approval_pin_lanes'], [2])
            self.assertEqual(receipt['lanes']['1']['status'], 'finished')
            self.assertEqual(receipt['lanes']['2']['status'], 'blocked_invalid_approval_pins')
            self.assertIn('starter hash changed', receipt['lanes']['2']['reason'])


if __name__ == '__main__':
    unittest.main()
