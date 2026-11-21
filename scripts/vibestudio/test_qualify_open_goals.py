import gzip
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audit_open_goal_references import fingerprint
from qualify_open_goals import main, qualify_group, terminal_startup_failure


class QualificationBatchesTest(unittest.TestCase):
    def test_reuses_only_accepted_reference_for_current_goal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            references = root / 'references'
            references.mkdir()
            catalog = root / 'source.jsonl'
            rows = []
            for label in ('verified', 'changed'):
                goal = f'Check the current {label} record'
                rows.append({
                    'scenario': f'open-goal-{label}', 'goalSha256': fingerprint(goal),
                    'interaction': 'open-user-goal', 'family': 'workspace-open-user-goal-demo',
                    'split': 'train', 'workspace': 'examples', 'sourceGroups': [label],
                })
                name = f'open-goal-oracle-{label}'
                checks = {'readback': True}
                trace = {
                    'test': {'name': name},
                    'result': {'passed': True, 'details': checks},
                    'execution': {
                        'provenance': {'contextId': 'owned'},
                        'diagnostics': {'openGoalVerification': {
                            'caseId': label, 'contextId': 'owned', 'checks': checks, 'passed': True,
                            'goal': goal if label == 'verified' else 'Original goal before the edit',
                        }},
                    },
                }
                with gzip.open(references / f'trajectory-{name}-full.json.gz', 'wt') as stream:
                    json.dump(trace, stream)
            catalog.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            argv = ['qualify_open_goals', '--host', str(root), '--catalog', str(catalog),
                    '--out', str(root / 'out'), '--reuse-references', str(references)]
            with patch.object(sys, 'argv', argv), patch('qualify_open_goals.qualify_group',
                    return_value=(True, False)) as run:
                self.assertEqual(main(), 0)
            self.assertEqual(run.call_args.args[3], ['open-goal-oracle-changed'])

            out = root / 'out'
            references.rename(out / 'completed-previous-batch')
            with patch.object(sys, 'argv', [*argv[:7], '--resume']), patch(
                    'qualify_open_goals.qualify_group', return_value=(True, False)) as run:
                self.assertEqual(main(), 0)
            self.assertEqual(run.call_args.args[3], ['open-goal-oracle-changed'])
            self.assertEqual(run.call_args.args[5].parent, out)
            self.assertTrue((out / 'completed-previous-batch').is_dir())

            catalog.write_text(catalog.read_text().rstrip('\n'))
            with patch.object(sys, 'argv', [*argv[:7], '--resume']), patch(
                    'qualify_open_goals.qualify_group') as run:
                with self.assertRaises(SystemExit) as error:
                    main()
            self.assertEqual(error.exception.code, 2)
            run.assert_not_called()

    def test_does_not_reexport_automatically_retained_failure_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = ['open-goal-oracle-failed', 'open-goal-oracle-passed']
            exported = []

            def invoke(command, **_kwargs):
                action = command[7]
                if action == 'run':
                    folder = root / 'examples' / 'results' / 'st_run' / 'failure-evidence'
                    folder.mkdir(parents=True)
                    (folder / f'trajectory-{cases[0]}-full.json.gz').touch()
                if action == 'trajectory':
                    case = command[9]
                    exported.append(case)
                    (root / 'examples' / 'results' / 'st_run' / f'trajectory-{case}-full.json.gz').touch()
                return subprocess.CompletedProcess(command, int(action == 'run'))

            with patch('qualify_open_goals.subprocess.run', side_effect=invoke):
                self.assertEqual(qualify_group(root, 'examples', '', cases, 'model', root, None),
                                 (False, False))
            self.assertEqual(exported, [cases[1]])

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
                **({'devices': ['desktop']} if index == 14 else {}),
            } for index in range(15)]
            catalog.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            out = root / 'out'
            argv = ['qualify_open_goals', '--host', str(root), '--catalog', str(catalog),
                    '--out', str(out)]
            with patch.object(sys, 'argv', argv), patch('qualify_open_goals.qualify_group',
                    side_effect=[(False, False), (True, False), (True, False), (True, False)]) as run:
                self.assertEqual(main(), 1)

            calls = run.call_args_list
            self.assertEqual([len(call.args[3]) for call in calls], [12, 1, 1, 1])
            self.assertEqual([call.kwargs['batch'] for call in calls], [1, 2, None, None])
            self.assertEqual([call.args[2] for call in calls], ['', '', 'a' * 40, ''])
            self.assertEqual([call.kwargs['devices'] for call in calls], [(), (), (), ('desktop',)])
            self.assertEqual([case for call in calls for case in call.args[3]],
                             [f'open-goal-oracle-case-{index}' for index in range(15)])
            self.assertEqual((out / 'catalog.jsonl').read_bytes(), catalog.read_bytes())

    def test_doctor_distinguishes_transient_and_terminal_setup_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / 'doctor.log'
            log.write_text('[system-test] Created workspace\n'
                           '{"error":"Missing immutable dependency", "recoverable":false}\n')
            self.assertTrue(terminal_startup_failure(log))
            log.write_text('{"error":"Timeout", "recoverable":true}\n')
            self.assertFalse(terminal_startup_failure(log))
            log.write_text('Interrupted before readiness response.\n')
            self.assertFalse(terminal_startup_failure(log))

    def test_attaches_the_catalog_driver_to_the_managed_commands(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case = 'open-goal-oracle-desktop'

            def invoke(command, **_kwargs):
                self.assertEqual(command[7:9], ['--device', 'desktop'])
                if command[9] == 'run':
                    folder = root / 'examples@desktop' / 'results' / 'st_run'
                    folder.mkdir(parents=True)
                    (folder / f'trajectory-{case}-full.json.gz').touch()
                return subprocess.CompletedProcess(command, 0)

            with patch('qualify_open_goals.subprocess.run', side_effect=invoke) as run:
                self.assertEqual(qualify_group(root, 'examples', '', [case], 'model', root, None,
                                               devices=('desktop',)), (True, False))
            self.assertEqual([call.args[0][9] for call in run.call_args_list], ['doctor', 'run', 'stop'])

    def test_drains_between_managed_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalog = root / 'source.jsonl'
            stop = root / 'drain'
            catalog.write_text(''.join(json.dumps({
                'scenario': f'open-goal-case-{index}', 'goalSha256': f'{index:064x}',
                'interaction': 'open-user-goal', 'family': 'workspace-open-user-goal-demo',
                'split': 'train', 'workspace': 'examples', 'sourceGroups': [f'demo-{index}'],
            }) + '\n' for index in range(3)))
            out = root / 'out'
            argv = ['qualify_open_goals', '--host', str(root), '--catalog', str(catalog),
                    '--out', str(out), '--resume', '--drain-file', str(stop), '--cases-per-instance', '2']

            def completed(_host, _workspace, _baseline, cases, _model, destination, _timeout, *, batch, devices):
                self.assertEqual(cases, ['open-goal-oracle-case-0', 'open-goal-oracle-case-1'])
                self.assertEqual(batch, 1)
                self.assertEqual(devices, ())
                (destination / 'complete').touch()
                stop.touch()
                return True, False

            with patch.object(sys, 'argv', argv), patch('qualify_open_goals.qualify_group',
                    side_effect=completed) as run:
                self.assertEqual(main(), 0)
                self.assertEqual(run.call_count, 1)
                self.assertEqual(main(), 0)
                self.assertEqual(run.call_count, 1)
            self.assertEqual(len(list(out.glob('attempt-*/complete'))), 1)
            self.assertEqual((out / 'catalog.jsonl').read_bytes(), catalog.read_bytes())


if __name__ == '__main__':
    unittest.main()
