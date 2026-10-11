"""Held-out split aliases (validation/dev) are rendered as test, counted and recorded, never silently dropped."""
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_renderer():
    path = ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py'
    spec = importlib.util.spec_from_file_location('render_native_turn_corpus_under_test', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Args(Namespace):
    """Selection options not named here are off (the renderer grows options; None means unset)."""

    def __getattr__(self, name):
        return None


def row(rid, split, group):
    return {'version': 'natlang.teacher_training_turn.native/1', 'id': rid, 'split': split,
            'source_groups': [group], 'program_id': rid, 'teacher_trajectory_id': rid,
            'training_admission': {'approved': True}, 'messages': [], 'family': 'f'}


def test_validation_rows_become_test_with_source_split(tmp_path):
    render = load_renderer()
    source = tmp_path / 'turns.jsonl'
    rows = [row('a', 'train', 'g1'), row('b', 'validation', 'g2'), row('c', 'dev', 'g3'), row('d', 'weird', 'g4')]
    source.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    shards = tmp_path / 'shards'
    shards.mkdir()
    args = Args(source=[source], limit=None, train_only=False, registry_admits_reason=None, exclude_programs=[],
                     reserved=None, protected=None, shards=1, full_sample=1.0)
    omissions, counts, _blocked, _cache, test_groups, _reserved, _sample = render.select(args, shards)
    written = [json.loads(line) for line in (shards / '0000.jsonl').read_text().splitlines()]
    by_id = {r['id']: r for r in written}
    assert by_id['b']['split'] == 'test' and by_id['b']['source_split'] == 'validation'
    assert by_id['c']['split'] == 'test' and by_id['c']['source_split'] == 'dev'
    assert by_id['a']['split'] == 'train' and 'source_split' not in by_id['a']
    assert counts['split_mapped:validation->test'] == 1 and counts['split_mapped:dev->test'] == 1
    assert {'g2', 'g3'} <= test_groups  # held groups: train documents sharing them are decontaminated at merge
    # An unknown split is not mapped; it reaches the renderer, which records it as an omission with its reason.
    assert by_id['d']['split'] == 'weird'
    assert not omissions
