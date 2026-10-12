import argparse
import importlib.util
import json
from pathlib import Path
import hashlib
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'prepare_decision_provider_source.py'
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location('prepare_decision_provider_source', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def write_cases(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')


class PriorProviderReservationTests(unittest.TestCase):
    def test_discovers_nested_lane_source_packets_and_nested_references(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_cases(root / 'provider-old' / 'lane-03' / 'source-selection' / 'cases.jsonl', [
                {'id': 'raw-decision-a', 'group': 'raw-group-a'},
                {'id': 'raw-decision-b', 'group': 'raw-group-b'},
            ])
            write_cases(root / 'provider-next' / 'source-selection' / 'cases.jsonl', [
                {'id': 'program-record', 'group': 'program-group',
                 'source_ids': ['raw-decision-a', 'raw-decision-c'],
                 'source_groups': ['raw-group-a', 'raw-group-c']}
            ])

            _, program_ids, program_groups, source_ids, source_groups, manifests = MODULE.read_prior_cases(root)

            self.assertEqual(program_ids, {'raw-decision-a', 'raw-decision-b', 'program-record'})
            self.assertEqual(program_groups, {'raw-group-a', 'raw-group-b', 'program-group'})
            self.assertEqual(source_ids, {'raw-decision-a', 'raw-decision-c'})
            self.assertEqual(source_groups, {'raw-group-a', 'raw-group-c'})
            self.assertEqual(len(manifests), 2)

    def test_selector_excludes_raw_decision_and_program_nested_reservations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prior_root = root / 'prior'
            write_cases(prior_root / 'provider-old' / 'lane-03' / 'source-selection' / 'cases.jsonl', [
                {'id': 'raw-reserved', 'group': 'raw-reserved-group'}
            ])
            write_cases(prior_root / 'provider-old' / 'programs' / 'cases.jsonl', [
                {'id': 'derived-program', 'group': 'derived-program-group',
                 'source_ids': ['nested-reserved'], 'source_groups': ['nested-reserved-group']}
            ])
            source = root / 'source' / 'cases.jsonl'
            rows = [
                {'id': 'raw-reserved', 'group': 'raw-reserved-group', 'source': 'fixture',
                 'family': 'decision:clinc-intent', 'role': 'train', 'kind': 'choice',
                 'state': 'raw collision', 'question': 'Choose one.', 'options': ['a', 'b'], 'answer': 'a'},
                {'id': 'nested-reserved', 'group': 'nested-reserved-group', 'source': 'fixture',
                 'family': 'decision:clinc-intent', 'role': 'train', 'kind': 'choice',
                 'state': 'nested collision', 'question': 'Choose one.', 'options': ['a', 'b'], 'answer': 'b'},
                {'id': 'fresh', 'group': 'fresh-group', 'source': 'fixture',
                 'family': 'decision:clinc-intent', 'role': 'train', 'kind': 'choice',
                 'state': 'new item', 'question': 'Choose one.', 'options': ['a', 'b'], 'answer': 'a'},
            ]
            write_cases(source, rows)
            source_bytes = source.read_bytes()
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            (source.parent / 'source-manifest.json').write_text(json.dumps({
                'cases': {'rows': len(rows), 'sha256': source_hash},
            }), encoding='utf-8')
            destination = root / 'prepared'
            args = argparse.Namespace(
                families=['decision:clinc-intent'], source=str(source), prior_root=str(prior_root),
                source_quality_holds=None, exclude_cases=[], max_state_chars=3000,
                max_options=None, out=str(destination), count=1,
            )

            manifest = MODULE.build(args)

            selected = [json.loads(line) for line in (destination / 'cases.jsonl').read_text().splitlines()]
            self.assertEqual([row['id'] for row in selected], ['fresh'])
            reasons = [json.loads(line)['reasons'] for line in (destination / 'omissions.jsonl').read_text().splitlines()]
            codes = [{reason['code'] for reason in group} for group in reasons]
            self.assertIn('prior_provider_program_id_overlap', codes[0])
            self.assertIn('prior_provider_source_id_overlap', codes[1])
            self.assertEqual(manifest['prior_provider_cases']['unique_reserved_id_union'], 3)

    def test_uses_nearest_lane_disposition_for_nested_reservations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = root / 'provider-old'
            lane = campaign / 'lane-03'
            path = lane / 'source-selection' / 'cases.jsonl'
            row = {'id': 'unused-program', 'group': 'unused-program-group',
                   'source_ids': ['unused-source'], 'source_groups': ['unused-source-group']}
            write_cases(path, [row])
            relative = path.relative_to(lane).as_posix()
            disposition = {
                'schema': 'natlang.decision-source-reservation-disposition/1',
                'decision': 'cancelled-unused-preparation',
                'root_approved': True,
                'provider_calls': 0,
                'reason': 'The prepared lane was never launched.',
                'artifacts': {relative: MODULE.sha256_file(path)},
            }
            (lane / 'reservation-disposition.json').write_text(json.dumps(disposition), encoding='utf-8')

            _, program_ids, program_groups, source_ids, source_groups, manifests = MODULE.read_prior_cases(root)

            self.assertEqual(program_ids, set())
            self.assertEqual(program_groups, set())
            self.assertEqual(source_ids, set())
            self.assertEqual(source_groups, set())
            self.assertEqual(manifests[0]['reservation_status'], 'cancelled-unused-preparation')


if __name__ == '__main__':
    unittest.main()
