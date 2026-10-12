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


if __name__ == '__main__':
    unittest.main()
