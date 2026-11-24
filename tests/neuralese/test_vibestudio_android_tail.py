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
