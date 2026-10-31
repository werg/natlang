from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.rl_tasks.application_controller_frontend import validate_controller_frontend
from scripts.rl_tasks.build_reviewed_luna_plan import _verify_intake_controller_frontend
from scripts.rl_tasks.intake_natlang_codebase import _project


class ApplicationControllerFrontendTests(unittest.TestCase):
    def setUp(self):
        self.files = {
            'natlang.json': '{}',
            'entry.nl': '---\nargs: {}\nreturns: number\nuses: [workflow]\n---\nRun the controller.\n',
            'entry/advance.nl': '---\nargs: {}\nreturns: number\n---\nPerform one operation.\n',
            'workflow.ts': (
                'export async function run(operations: { advance(): Promise<number> }): Promise<number> {\n'
                '  return await operations.advance();\n'
                '}\n'),
        }
        self.controller = {'schema': 'natlang.application-controller/1', 'module': 'workflow', 'export': 'run',
                           'operations': 'advance', 'input_arguments': []}

    def test_preflight_uses_actual_loader_and_attaches_controller_and_operation(self):
        receipt = validate_controller_frontend(self.files, 'entry.nl', self.controller)
        self.assertEqual(receipt['loaded_by'], 'natlang.runtime.loadVirtualNatlang')
        self.assertTrue(receipt['accepted'])

    def test_preflight_rejects_unsupported_typescript_subset_syntax_before_collection(self):
        self.files['workflow.ts'] = (
            'export async function run(operations: { advance(): Promise<number> }): Promise<number> {\n'
            '  while (true) { return await operations.advance(); }\n'
            '}\n')
        with self.assertRaisesRegex(ValueError, 'while.*loops are not allowed'):
            validate_controller_frontend(self.files, 'entry.nl', self.controller)

    def test_plan_preflight_revalidates_against_frozen_runtime_and_rejects_stale_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / 'application'
            package = root / 'package'
            package.mkdir(parents=True)
            for name, text in self.files.items():
                path = package / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding='utf-8')
            (root / 'metadata.json').write_text(json.dumps({'collector_controller': self.controller}), encoding='utf-8')
            project = _project(package)
            frontend = validate_controller_frontend(project, 'entry.nl', self.controller)
            package_sha = hashlib.sha256(json.dumps(
                {'files': project, 'services': {}}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
            receipt = {'package_path': str(package), 'package_sha256': package_sha,
                       'controller_frontend_preflight': [frontend]}
            row = {'semantics': {'root': 'entry.nl', 'application_controller': self.controller}}
            runtime = Path(temporary) / 'runtime'
            runtime.mkdir()
            (runtime / 'dist').symlink_to(Path(__file__).resolve().parents[1] / 'ts-host' / 'dist')
            checked = _verify_intake_controller_frontend([row], receipt, runtime)
            self.assertEqual(checked[0]['frontend_sha256'], frontend['frontend_sha256'])
            stale = json.loads(json.dumps(receipt))
            stale['controller_frontend_preflight'][0]['frontend_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'preflight differs'):
                _verify_intake_controller_frontend([row], stale, runtime)


if __name__ == '__main__':
    unittest.main()
