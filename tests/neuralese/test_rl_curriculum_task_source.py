import json
import hashlib
from collections import Counter
from pathlib import Path
import shutil

import pytest

from natlang_neuralese.rl.curriculum import metadata
from natlang_neuralese.rl.task_source import resolve_task_source


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / 'runs/decision-rich-train-20261016-v1'
V2 = ROOT / 'runs/decision-rich-train-20261016-v2-curriculum'
V3 = ROOT / 'runs/decision-rich-train-20261017-v3-scaffold'


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


def test_qualified_scaffold_continuation_preserves_128_and_binds_12_additions():
    resolved = resolve_task_source(V3 / 'task-source-manifest.json')
    prior = read_rows(V2 / 'cases.jsonl')
    rows = read_rows(resolved['cases'])
    qualified_source = read_rows(V3 / 'additional-scaffold-cases.jsonl')
    assert len(prior) == 128
    assert len(qualified_source) == 12
    assert len(rows) == resolved['rows'] == 140
    assert [row['id'] for row in rows[:128]] == [row['id'] for row in prior]
    assert rows[:128] == prior
    assert [row['id'] for row in rows[128:]] == [row['id'] for row in qualified_source]
    assert not ({row['id'] for row in prior} & {row['id'] for row in qualified_source})
    for source_row, admitted_row in zip(qualified_source, rows[128:]):
        assert admitted_row == source_row
    assert len(resolved['additional_sources']) == 1
    evidence = resolved['additional_sources'][0]
    assert evidence['accepted_teacher_rows'] == 48
    assert evidence['qualified_task_rows'] == 12
    assert evidence['current_protection_intersections'] == {
        'decision_ids': [], 'held_groups': [], 'quality_groups': [],
    }


def test_additional_rl_source_rejects_changed_case_bytes(tmp_path):
    copied = tmp_path / 'v3'
    shutil.copytree(V3, copied)
    additional = copied / 'additional-scaffold-cases.jsonl'
    additional.write_text(additional.read_text().replace('"split":"train"', '"split":"held"', 1))
    with pytest.raises(ValueError, match='additional source bytes differ'):
        resolve_task_source(copied / 'task-source-manifest.json')


@pytest.mark.parametrize('mutate', [
    lambda proof: proof.update(source_id='different-source'),
    lambda proof: proof.update(all_source_bound=False),
    lambda proof: proof['union_task_row_sha256'].pop(next(iter(proof['union_task_row_sha256']))),
])
def test_additional_rl_source_rejects_incomplete_qualification_proof(tmp_path, mutate):
    copied = tmp_path / 'v3'
    shutil.copytree(V3, copied)
    proof_path = copied / 'additional-scaffold-qualification.json'
    proof = json.loads(proof_path.read_text())
    mutate(proof)
    proof_bytes = (json.dumps(proof, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n').encode()
    proof_path.write_bytes(proof_bytes)
    manifest_path = copied / 'task-source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest['additional_admitted_train_sources'][0]['qualification_proof_sha256'] = hashlib.sha256(proof_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    with pytest.raises(ValueError):
        resolve_task_source(manifest_path)
