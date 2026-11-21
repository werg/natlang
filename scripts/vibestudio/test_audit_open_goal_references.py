import json
import tempfile
import unittest
from pathlib import Path

from audit_open_goal_references import inventory_coverage, managed_prerequisites, write_train_cases


class CapabilityCoverageTest(unittest.TestCase):
    def test_requires_each_agentic_origin_and_reports_auxiliary_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            baseline = root / 'baseline.jsonl'
            crosswalk = root / 'coverage.json'
            baseline.write_text(''.join(json.dumps({
                'kind': 'system-test', 'origin': origin, 'capability': capability,
            }) + '\n' for origin, capability in (
                ('managed-teacher', 'managed'),
                ('direct-agent', 'direct'),
                ('agent-orchestration', 'orchestrated'),
                ('harness-review', 'review'),
                ('open-user-goal', 'new-goal'),
                ('testkit', 'testkit'),
            )))
            crosswalk.write_text(json.dumps({
                'schema': 'vibestudio.open-goal-coverage/2',
                'families': {'scope': {'capabilities': ['direct'], 'skills': []}},
                'cases': {'open-goal-scope': {'capabilities': {'direct': ['independentReadback']}}},
            }))
            report = inventory_coverage({
                'families': {'workspace-open-user-goal-scope': {}},
                'cases': {'open-goal-scope': {'family': 'workspace-open-user-goal-scope'}},
                'proofs': {'open-goal-scope': {'checks': ['independentReadback']}},
            }, baseline, crosswalk)['capabilities']
            self.assertEqual(report['agentic_baseline'], ['direct', 'managed', 'orchestrated'])
            self.assertEqual(report['agentic_without_verified_reference_links'], ['managed', 'orchestrated'])
            self.assertEqual(report['without_verified_reference_links'],
                             ['managed', 'new-goal', 'orchestrated', 'review', 'testkit'])


class ManagedPrerequisitesTest(unittest.TestCase):
    def test_distinguishes_assigned_public_fetch_from_dynamic_search(self):
        for family, required in (
            ('assigned-public-brief', ['public-web-fetch']),
            ('research', ['public-web-search-provider']),
            ('research-and-edit', ['public-web-search-provider']),
            ('calendar', []),
        ):
            with self.subTest(family=family):
                row = {'family': f'workspace-open-user-goal-{family}'}
                self.assertEqual(managed_prerequisites(row), required)


class TrainSelectorTest(unittest.TestCase):
    def test_uses_only_accepted_training_goals_and_preserves_an_existing_plan(self):
        result = {
            'cases': {
                'open-goal-confirmed': {'split': 'train'},
                'open-goal-unverified': {'split': 'train'},
                'open-goal-holdout': {'split': 'test'},
            },
            'proofs': {'open-goal-confirmed': {}, 'open-goal-holdout': {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'capture-cases.txt'
            write_train_cases(destination, result)
            self.assertEqual(destination.read_text(), 'open-goal-confirmed\n')
            with self.assertRaises(FileExistsError):
                write_train_cases(destination, result)

            empty_destination = Path(tmp) / 'empty.txt'
            result['proofs'].pop('open-goal-confirmed')
            with self.assertRaises(ValueError):
                write_train_cases(empty_destination, result)
            self.assertFalse(empty_destination.exists())

    def test_local_plan_excludes_accepted_training_cases_with_prerequisites(self):
        result = {
            'cases': {
                'open-goal-local': {'split': 'train', 'prerequisites': []},
                'open-goal-mobile': {'split': 'train', 'prerequisites': ['android-emulator']},
                'open-goal-held': {'split': 'test', 'prerequisites': []},
            },
            'proofs': {'open-goal-local': {}, 'open-goal-mobile': {}, 'open-goal-held': {}},
        }
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / 'capture-cases.txt'
            write_train_cases(destination, result, local_only=True)
            self.assertEqual(destination.read_text(), 'open-goal-local\n')


if __name__ == '__main__':
    unittest.main()
