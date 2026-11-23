"""Tests for the shared four-arm flagship benchmark harness (scripts/semantic_fp/bench.py, WS6 A0)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts' / 'semantic_fp'))
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))

import bench  # noqa: E402
from natlang_neuralese.rl.held_eval import reservation_share_held  # noqa: E402


def test_label_fixtures_matches_held_v4_share_rule():
    rows = [{'id': f'case-{i}', 'group': f'g-{i}'} for i in range(40)]
    labeled = bench.label_fixtures('pr-shepherd', rows)
    assert len(labeled) == 40
    for row in labeled:
        assert row['split_group'].startswith('flagship:pr-shepherd:')
        expected = 'held' if reservation_share_held(bench.FLAGSHIP_SHARE_RULE, row['split_group']) else 'train'
        assert row['split'] == expected
    held_count = sum(1 for r in labeled if r['split'] == 'held')
    assert 3 <= held_count <= 15


def test_run_benchmark_and_paired_compare_detect_significant_win():
    fixtures = [{'id': f'item-{i}', 'group': f'grp-{i}'} for i in range(60)]

    def fake_exec(arm: str, item: dict) -> dict:
        if arm == 'A':
            return {
                'passed': int(item['id'].split('-')[1]) % 3 == 0,
                'tokens': 5000, 'dollars': 0.12, 'latency_ms': 2400,
                'units': 10, 'crisp_units': 0, 'local_units': 0, 'escalated_units': 10,
                'safety': {'unverified_commits': 1, 'injection_escapes': 0, 'schema_violations': 0},
            }
        return {
            'passed': True,
            'tokens': 400, 'dollars': 0.004, 'latency_ms': 180,
            'units': 10, 'crisp_units': 6, 'local_units': 3, 'escalated_units': 1,
            'safety': {'unverified_commits': 0, 'injection_escapes': 0, 'schema_violations': 0},
        }

    report = bench.run_benchmark('pr-shepherd', fixtures, arms=('A', 'D'), executor=fake_exec, split='held')
    assert report['schema'] == bench.SCHEMA
    assert report['evaluated_items'] > 0
    assert report['arms']['D']['pass_rate'] == 1.0
    assert report['arms']['D']['crisp_share'] == 0.6
    assert report['arms']['D']['safety']['unverified_commits'] == 0
    cmp = report['comparisons']['D_vs_A']
    assert cmp['beats'] is True
    assert cmp['p_value'] < 0.05
    assert cmp['cost_ratio'] < 0.05


def test_cli_split_writes_json(tmp_path):
    src = tmp_path / 'fixtures.json'
    out = tmp_path / 'labeled.json'
    src.write_text(json.dumps([{'id': 'a', 'group': 'g1'}, {'id': 'b', 'group': 'g2'}]), encoding='utf-8')
    assert bench.main(['split', '--app', 'semantic-etl', '--fixtures', str(src), '--out', str(out)]) == 0
    data = json.loads(out.read_text(encoding='utf-8'))
    assert data['total'] == 2
    assert all(item['split_group'].startswith('flagship:semantic-etl:') for item in data['items'])
