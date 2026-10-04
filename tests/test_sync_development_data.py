import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.sync_development_data import classify_rsync_output, run_rsync, RsyncFailure


class RsyncVanishHandlingTests(unittest.TestCase):
    def test_accepts_only_worker_job_partial_paths_on_exit_24(self):
        first = '/home/werg/natlang/runs/active/worker-1/jobs/000001-abc.partial.json'
        second = '/home/werg/natlang/runs/active/worker-2-v3/jobs/000002-def.partial.json'
        output = '\n'.join([
            f'file has vanished: "{first}"',
            f'file has vanished: "{second}"',
            'rsync warning: some files vanished before they could be transferred (code 24) at main.c(1356) [sender=3.2.7]',
        ])
        warning = classify_rsync_output(24, output)
        self.assertEqual(warning['vanished_paths_count'], 2)
        self.assertEqual(warning['vanished_paths_sample'], [first, second])

    def test_exit_23_remains_a_failure(self):
        with self.assertRaises(subprocess.CalledProcessError) as raised:
            classify_rsync_output(23, 'rsync error: some files/attrs were not transferred')
        self.assertEqual(raised.exception.returncode, 23)

    def test_exit_24_does_not_hide_nonpartial_or_unexplained_errors(self):
        prefix = 'rsync warning: some files vanished before they could be transferred (code 24) at main.c(1356) [sender=3.2.7]'
        cases = [
            f'file has vanished: "/tmp/ordinary.json"\n{prefix}',
            f'file has vanished: "/home/werg/runs/active/worker-1/jobs/000001.partial.json"\n{prefix}\nrsync: [sender] Permission denied (13)',
            prefix,
            f'file has vanished: "/home/werg/runs/active/worker-1/jobs/000001.partial.json"\n{prefix}\nrsync warning: unrelated warning',
            prefix,
        ]
        for output in cases:
            with self.subTest(output=output), self.assertRaises(subprocess.CalledProcessError) as raised:
                classify_rsync_output(24, output)
            self.assertEqual(raised.exception.returncode, 24)

    def test_success_exit_cannot_claim_success_after_vanish_output(self):
        with self.assertRaises(ValueError):
            classify_rsync_output(0, 'file has vanished: "/home/werg/runs/active/worker-1/jobs/000001.partial.json"')

    def test_run_rsync_receipt_points_to_durable_log_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'push.log'
            log.write_text('prior log line\n')
            output = ('file has vanished: "/home/werg/runs/active/worker-1/jobs/000001.partial.json"\n'
                      'rsync warning: some files vanished before they could be transferred (code 24) at main.c(1356) [sender=3.2.7]\n')
            with patch('scripts.sync_development_data.subprocess.run', return_value=subprocess.CompletedProcess(['rsync'], 24)) as run:
                # The fake process writes output the same way rsync does.
                def write_log(command, stdout, stderr, check):
                    stdout.write(output.encode())
                    stdout.flush()
                    return subprocess.CompletedProcess(command, 24)
                run.side_effect = write_log
                warning = run_rsync(['rsync'], log, 'home_to_dgx', ['runs'])
            self.assertEqual(warning['direction'], 'home_to_dgx')
            self.assertEqual(warning['vanished_paths_count'], 1)
            self.assertEqual(warning['rsync_log'], str(log))
            with log.open('rb') as handle:
                handle.seek(warning['log_start_byte'])
                logged = handle.read(warning['log_end_byte'] - warning['log_start_byte']).decode()
            self.assertIn('000001.partial.json', logged)

    def test_nonzero_error_keeps_exit_code_and_log_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / 'pull.log'
            with patch('scripts.sync_development_data.subprocess.run', return_value=subprocess.CompletedProcess(['rsync'], 23)):
                with self.assertRaises(RsyncFailure) as raised:
                    run_rsync(['rsync'], log, 'dgx_to_home', 'runs/dgx-development-generated')
            self.assertEqual(raised.exception.returncode, 23)
            self.assertEqual(raised.exception.log_path, str(log))


if __name__ == '__main__':
    unittest.main()
