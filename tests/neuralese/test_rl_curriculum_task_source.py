import json
from collections import Counter
from pathlib import Path

from natlang_neuralese.rl.curriculum import metadata
from natlang_neuralese.rl.task_source import resolve_task_source


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / 'runs/decision-rich-train-20261016-v1'
V2 = ROOT / 'runs/decision-rich-train-20261016-v2-curriculum'


def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def test_curriculum_successor_preserves_every_task_and_only_adds_schedule_metadata():
    resolved = resolve_task_source(V2 / 'task-source-manifest.json')
    old_rows = read_rows(V1 / 'cases.jsonl')
    rows = read_rows(resolved['cases'])
    assert len(old_rows) == len(rows) == 128
    assert [row['id'] for row in rows] == [row['id'] for row in old_rows]
    assert len({row['id'] for row in rows}) == 128
    assert all(row['split'] == 'train' for row in rows)

    counts = Counter()
    for old, new in zip(old_rows, rows):
        level = metadata(new)['level']
        counts[level] += 1
        original = json.loads(json.dumps(new))
        original['curriculum'].pop('rl')
        assert canonical(original) == canonical(old)
        assert new['source_groups'] == old['source_groups']
        assert new['semantics'] == old['semantics']
    assert counts == {0: 1, 1: 3, 2: 8, 3: 116}

    audit = json.loads(resolved['audit'].read_text())
    schedule = audit['curriculum_schedule']
    assert schedule['all_128_original_rows_retained'] is True
    assert schedule['non_scheduling_payloads_exactly_preserved'] is True
    assert schedule['current_protection_intersections'] == {
        'decision_ids': [], 'held_groups': [], 'quality_groups': [],
    }
