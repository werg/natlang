from __future__ import annotations

import unittest

from scripts.rl_tasks.application_controller_frontend import validate_controller_frontend


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


if __name__ == '__main__':
    unittest.main()
