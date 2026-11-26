import importlib.util
from pathlib import Path
from types import SimpleNamespace
import json
import tempfile
import unittest
import sys

MODULE = Path(__file__).resolve().parents[2] / 'scripts/neuralese_data/vibestudio_android_tail.py'
sys.path.insert(0, str(MODULE.parents[2] / 'training/neuralese'))
spec = importlib.util.spec_from_file_location('android_tail', MODULE)
tail = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tail)


class DeviceRouteTests(unittest.TestCase):
    def test_launch_port_matches_owned_serial_instead_of_a_different_emulator(self):
        self.assertEqual(tail.owned_emulator_port('emulator-5582'), '5582')
        self.assertEqual(tail.owned_emulator_port('emulator-5580'), '5580')
        self.assertEqual(tail.owned_emulator_port('emulator-5554'), '5554')

    def test_physical_devices_and_invalid_console_ports_are_refused(self):
        for serial in ['physical-device', 'emulator-5581', 'emulator-0', 'emulator-65534', 'emulator-5582-extra']:
            with self.subTest(serial=serial), self.assertRaises(ValueError):
                tail.owned_emulator_port(serial)


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.expected = self.root / 'expected'
        self.actual = self.root / 'instance/workspaces/system-owned/source/clients/mobile-browser'
        self.expected.mkdir()
        self.actual.mkdir(parents=True)
        for i in range(180):
            for directory in (self.expected, self.actual):
                (directory / f'{i}.ts').write_text(f'export const value = {i};')
        self.run = self.root / 'run.json'
        self.run.write_text(json.dumps({'stateDir': str(self.root / 'instance')}))
        self.record = self.root / 'approval.json'
        self.actions = []
        self.xml = '''<?xml version="1.0"?><hierarchy>
          <node text="Review code before starting" />
          <node text="https://github.com/panticonic/vibestudio-system.git" />
          <node text="run 1 program on this computer" />
          <node text="Start workspace" clickable="true" enabled="true"
            package="app.vibestudio.mobile.internal" bounds="[100,100][200,200]" />
        </hierarchy>'''
        def adb(*args):
            self.actions.append(args)
            return SimpleNamespace(stdout=self.xml if args[0] == 'exec-out' else '')
        self.device = SimpleNamespace(adb=adb, lease_path=self.root/'lease.json', lease={'deviceBootId': 'owned-boot'})

    def test_unknown_application_is_never_approved(self):
        (self.actual / '7.ts').write_text('unknown code')
        with self.assertRaises(AssertionError):
            tail.approve_owned_workspace(self.device, self.run, self.expected, self.record)
        self.assertEqual(self.actions, [])
        self.assertFalse(self.record.exists())

    def test_read_only_screen_never_causes_approval(self):
        self.xml = '<?xml version="1.0"?><hierarchy><node text="Welcome" /></hierarchy>'
        tail.approve_owned_workspace(self.device, self.run, self.expected, self.record)
        self.assertFalse(any('tap' in args for args in self.actions))
        self.assertFalse(self.record.exists())

    def test_wrong_package_button_is_never_tapped(self):
        self.xml = self.xml.replace('app.vibestudio.mobile.internal', 'another.application')
        with self.assertRaises(AssertionError):
            tail.approve_owned_workspace(self.device, self.run, self.expected, self.record)
        self.assertFalse(any('tap' in args for args in self.actions))

    def test_actual_known_workspace_uses_real_owner_button(self):
        tail.approve_owned_workspace(self.device, self.run, self.expected, self.record)
        self.assertIn(('shell', 'input', 'tap', '150', '150'), self.actions)
        record = tail.read(self.record)
        self.assertEqual(record['role'], 'fixture-owner-approval')
        self.assertEqual(record['modelCalls'], 0)
        self.assertEqual(record['matchedApplicationSourceFiles'], 180)
        self.assertEqual(record['deviceBootId'], 'owned-boot')


class ResumeTests(unittest.TestCase):
    def test_admitted_wave_is_reused_with_its_matching_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root/'waves/jobs'
            jobs.mkdir(parents=True)
            admitted = jobs/'0007.json'
            original = json.dumps({'cases': ['second'], 'attempt_epoch': 'epoch', 'provenance': {'pin': 'exact'}})
            admitted.write_text(original)
            unfinished = lambda *args: admitted
            goals = [{'case': {'scenario': name}} for name in ['first', 'second']]
            ordered = tail.ordered_tail_goals(goals, root, 'epoch', {'pin': 'exact'}, {}, unfinished)
            self.assertEqual(ordered[0]['case']['scenario'], 'second')
            self.assertEqual(tail.reserve_tail_wave(root, 'second', 'epoch', {'pin': 'exact'}, {}, unfinished), '0007')
            self.assertEqual(admitted.read_text(), original)
            self.assertEqual(len(list(jobs.iterdir())), 1)
            with self.assertRaises(AssertionError):
                tail.reserve_tail_wave(root, 'first', 'epoch', {'pin': 'exact'}, {}, unfinished)

    def test_foreign_pending_fixture_is_not_silently_launched(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = root/'waves/jobs'
            jobs.mkdir(parents=True)
            admitted = jobs/'0007.json'
            admitted.write_text(json.dumps({'cases': ['ordinary']}))
            with self.assertRaises(AssertionError):
                tail.ordered_tail_goals([{'case': {'scenario': 'android'}}], root, 'epoch', {}, {}, lambda *args: admitted)


if __name__ == '__main__':
    unittest.main()


class OrdinaryBoundaryTests(unittest.TestCase):
    def test_pending_failed_goals_do_not_invalidate_settled_attempt_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            proof = {'schema': 'vibestudio.ordinary-phase/1', 'unit': 'bulk.service',
                     'invocation': 'owned-invocation', 'attempt_epoch': 'epoch',
                     'provenance': {'pins': 'exact'}, 'reason': 'wave-limit', 'completed_waves': 1}
            tail.save(root/'ordinary-phase.json', proof)
            tail.save(root/'status.json', {'cases': {'genuine-failure': {'status': 'pending'}}})
            check = lambda: tail.ordinary_phase_settled(root, 'bulk.service', 'epoch', {'pins': 'exact'}, 'owned-invocation')
            self.assertTrue(check())
            for field, value in [('unit', 'foreign.service'), ('invocation', 'previous-invocation'),
                                 ('attempt_epoch', 'previous-epoch'), ('provenance', {'pins': 'other'}),
                                 ('reason', 'checkpoint'), ('completed_waves', 0)]:
                tail.save(root/'ordinary-phase.json', {**proof, field: value})
                self.assertFalse(check(), field)
            tail.save(root/'ordinary-phase.json', proof)
            (root/'stop-after-wave').touch()
            self.assertFalse(check())

    def test_unadmitted_cancel_and_resumable_gap_do_not_launch_devices(self):
        from unittest.mock import patch
        from scripts import memory_ledger
        with patch.object(tail, 'active', return_value=False), patch.object(tail.subprocess, 'run', return_value=SimpleNamespace(returncode=1, stdout='')), \
             patch.object(memory_ledger, 'unit_owned', return_value=False):
            self.assertFalse(tail.wait_unit('cancelled.service'))
        ownership = iter([True, False])
        sleeps = []
        with patch.object(tail, 'active', return_value=False), patch.object(tail.subprocess, 'run', return_value=SimpleNamespace(returncode=0, stdout='success')), \
             patch.object(memory_ledger, 'unit_owned', side_effect=lambda unit: next(ownership)), \
             patch.object(tail.time, 'sleep', side_effect=lambda seconds: sleeps.append(seconds)):
            self.assertTrue(tail.wait_unit('requeued.service'))
        self.assertEqual(sleeps, [15])
