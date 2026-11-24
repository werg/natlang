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


if __name__ == '__main__':
    unittest.main()
