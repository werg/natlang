import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from run_screened_skill_queue import validate_handoff


class ScreenedQueueTests(unittest.TestCase):
    def fixture(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        runtime = root / 'runtime'
        runtime.mkdir()
        (runtime / 'frozen-runtime.json').write_text('{}')
        packet = [{'id': str(i), 'split': 'train', 'support': {'cases': [{'id': 'a'}]},
                   'query': {'cases': [{'id': 'sealed-query'}]}, 'provenance': {}} for i in range(3)]
        body = ''.join(json.dumps(row) + '\n' for row in packet).encode()
        source = root / 'source'
        source.write_bytes(body)
        input_hash = hashlib.sha256(body).hexdigest()
        screens = [{'schema': 'natlang.episode-headroom/2', 'episode': str(i),
                    'input_sha256': input_hash, 'executor': 'fixture:model', 'screen': 'pinned',
                    'support_quality': quality, 'cases': 1} for i, quality in enumerate([0, .5, 1])]
        screen_out = root / 'screen'
        screen_out.write_text(''.join(json.dumps(row) + '\n' for row in screens))
        kept = root / 'kept'
        kept.write_text(''.join(json.dumps({**row, 'provenance': {'headroom': {
            'executor': 'fixture:model', 'screen': 'pinned', 'support_quality': screens[i]['support_quality'],
            'band': [0, .95]}}}) + '\n' for i, row in enumerate(packet[:2])))
        return argparse.Namespace(screen_input=source, screen_out=screen_out, kept=kept, runtime=runtime,
            input_sha256=input_hash, runtime_sha256=hashlib.sha256(b'{}').hexdigest(),
            endpoint='fixture', model='model', low=0, high=.95)

    def test_zero_is_retained_and_ceiling_is_excluded_using_support_only(self):
        report = validate_handoff(self.fixture())
        self.assertEqual(report['selected'], 2)
        self.assertEqual(report['held'][0]['episode'], '2')

    def test_incomplete_screen_and_changed_inputs_fail_closed(self):
        args = self.fixture()
        args.screen_out.write_text(args.screen_out.read_text().splitlines()[0] + '\n')
        with self.assertRaisesRegex(ValueError, 'every train episode'):
            validate_handoff(args)
        args = self.fixture()
        args.screen_input.write_bytes(args.screen_input.read_bytes() + b' ')
        with self.assertRaisesRegex(ValueError, 'input changed'):
            validate_handoff(args)

    def test_edited_query_or_runtime_and_mixed_executor_are_rejected(self):
        args = self.fixture()
        packet = [json.loads(line) for line in args.kept.read_text().splitlines()]
        packet[0]['query']['cases'][0]['id'] = 'changed-query'
        args.kept.write_text(''.join(json.dumps(row) + '\n' for row in packet))
        with self.assertRaisesRegex(ValueError, 'independently selected'):
            validate_handoff(args)
        args = self.fixture()
        (args.runtime / 'frozen-runtime.json').write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError, 'runtime identity'):
            validate_handoff(args)
        args = self.fixture()
        args.screen_out.write_text(args.screen_out.read_text().replace('fixture:model', 'other:model'))
        with self.assertRaisesRegex(ValueError, 'executor identity'):
            validate_handoff(args)

    def test_predeclared_batch_collects_before_unrelated_screens_finish(self):
        args=self.fixture()
        args.episode_ids=args.kept.parent/'batch'
        args.episode_ids.write_text('["0"]')
        args.episode_ids_sha256=hashlib.sha256(args.episode_ids.read_bytes()).hexdigest()
        args.kept.unlink()
        lines=args.screen_out.read_text().splitlines()
        args.screen_out.write_text(lines[0]+'\n')
        first=validate_handoff(args)
        self.assertEqual(first['selected'],1)
        self.assertEqual(first['screened'],1)
        original=args.kept.read_bytes()
        args.screen_out.write_text('\n'.join(lines)+'\n')
        self.assertEqual(validate_handoff(args),first)
        self.assertEqual(args.kept.read_bytes(),original)
        args.episode_ids.write_text('["1"]')
        with self.assertRaisesRegex(ValueError,'batch changed'):
            validate_handoff(args)


if __name__ == '__main__':
    unittest.main()
