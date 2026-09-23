import json
from pathlib import Path

import pytest

from scripts.prepare_training_stages import prepare


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return path


def output_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_curriculum_unions_splits_across_lanes_and_aligns_teacher(tmp_path):
    code = write_rows(tmp_path / 'code.jsonl', [
        {'id': 'code-a', 'program_id': 'program-a', 'split': 'train',
         'completion': 'return 1;', 'source_groups': ['shared-source']},
    ])
    native = write_rows(tmp_path / 'native.jsonl', [
        {'id': 'native-a', 'program_id': 'program-b', 'split': 'test',
         'completion': 'if (x) return 1;', 'source_groups': ['shared-source']},
    ])
    teacher = write_rows(tmp_path / 'teacher.jsonl', [
        {'id': 'teacher-a', 'program_id': 'teacher-run', 'completion': 'await finish();',
         'source_groups': ['shared-source']},
    ])

    result = prepare(tmp_path / 'curriculum', [code], [native], [teacher], seed=13)

    assert result['counts']['general']['splits'] == {'test': 1}
    assert result['counts']['coding']['splits'] == {'test': 1}
    assert result['counts']['teacher']['splits'] == {'test': 1}
    rows = [row for name in ('general', 'coding', 'teacher')
            for row in output_rows(tmp_path / 'curriculum' / f'{name}.jsonl')]
    assert len({row['split'] for row in rows}) == 1
    assert all('shared-source' in row['source_groups'] for row in rows)


def test_new_links_cannot_conflict_with_frozen_registry_labels(tmp_path):
    code = write_rows(tmp_path / 'code.jsonl', [
        {'id': 'a', 'program_id': 'frozen-train', 'completion': 'return 1;'},
        {'id': 'b', 'program_id': 'frozen-test', 'split': 'test', 'completion': 'return 2;',
         'source_groups': ['frozen-train']},
    ])
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'groups': {'frozen-train': 'train', 'frozen-test': 'test'}}))

    with pytest.raises(ValueError, match='conflicting frozen splits'):
        prepare(tmp_path / 'curriculum', [code], registry=registry)


def test_explicit_holdout_cannot_move_frozen_train_group(tmp_path):
    code = write_rows(tmp_path / 'code.jsonl', [
        {'id': 'new-test', 'program_id': 'old-train', 'split': 'test', 'completion': 'return 1;'},
    ])
    registry = tmp_path / 'registry.json'
    registry.write_text(json.dumps({'groups': {'old-train': 'train'}}))

    with pytest.raises(ValueError, match='holdout overlaps previously frozen training group'):
        prepare(tmp_path / 'curriculum', [code], registry=registry)


def test_curriculum_source_rows_have_deterministic_easy_to_hard_order(tmp_path):
    code = write_rows(tmp_path / 'code.jsonl', [
        {'id': 'hard', 'program_id': 'hard', 'completion': 'await fetch(url);'},
        {'id': 'mid', 'program_id': 'mid', 'completion': 'if (x) return x.map(f);'},
        {'id': 'easy', 'program_id': 'easy', 'completion': 'return 1;'},
        {'id': 'mid-2', 'program_id': 'mid-2', 'completion': 'for (const x of xs) use(x);'},
    ])

    prepare(tmp_path / 'first', [code], seed=7)
    prepare(tmp_path / 'second', [code], seed=7)

    first = (tmp_path / 'first' / 'general.jsonl').read_bytes()
    second = (tmp_path / 'second' / 'general.jsonl').read_bytes()
    assert first == second
    assert [row['difficulty'] for row in output_rows(tmp_path / 'first' / 'general.jsonl')] == [0, 1, 1, 3]
