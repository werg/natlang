import json
import hashlib
from collections import Counter
from pathlib import Path
import shutil

import pytest

from natlang_neuralese.rl.curriculum import metadata
from natlang_neuralese.rl.task_source import resolve_task_source
from scripts.neuralese_data.build_rl_task_source_correction import build as build_correction
from natlang_neuralese.rl.task_source import _ill_posed_port_record_continuation


ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / 'runs/decision-rich-train-20261016-v1'
V2 = ROOT / 'runs/decision-rich-train-20261016-v2-curriculum'
def read_rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def canonical(row):
    return json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def make_v3_fixture(tmp_path):
    """Build a compact, self-contained additional-source fixture.

    The production V3 packet lives in an ignored data directory and is not
    present in isolated test worktrees. The resolver contract only needs the
    exact bindings below, so synthesize the smallest valid successor from the
    checked-in V2 fixture instead of depending on that machine-local packet.
    """
    copied = tmp_path / 'v3'
    shutil.copytree(V2, copied)
    manifest_path = copied / 'task-source-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    prior_rows = read_rows(copied / 'cases.jsonl')
    additions = [
        {'id': f'test-qualified-scaffold-{index:02d}', 'split': 'train',
         'family': 'fixture', 'source_ids': [f'source-{index:02d}'],
         'source_groups': [f'group-{index:02d}'], 'semantics': {'expected': index}}
        for index in range(12)
    ]
    additions_path = copied / 'additional-scaffold-cases.jsonl'
    additions_path.write_text(''.join(json.dumps(row, separators=(',', ':')) + '\n' for row in additions))
    additions_sha = hashlib.sha256(additions_path.read_bytes()).hexdigest()
    qualification = {
        'schema': 'natlang.rl-additional-task-source-qualification/1',
        'source_id': 'fixture-qualified-source-v1',
        'source_cases_sha256': additions_sha,
        'accepted_teacher_rows': 48,
        'qualified_task_rows': len(additions),
        'all_train': True,
        'all_source_bound': True,
        'current_protection_intersections': {
            'decision_ids': [], 'held_groups': [], 'quality_groups': [],
        },
        'task_ids': [row['id'] for row in additions],
        'union_task_row_sha256': {
            row['id']: hashlib.sha256(canonical(row).encode()).hexdigest()
            for row in additions
        },
    }
    qualification_path = copied / 'additional-scaffold-qualification.json'
    qualification_path.write_text(json.dumps(qualification, sort_keys=True, separators=(',', ':')) + '\n')
    qualification_sha = hashlib.sha256(qualification_path.read_bytes()).hexdigest()
    cases_path = copied / 'cases.jsonl'
    cases_path.write_text(''.join(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n'
                                   for row in prior_rows + additions))
    cases_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()

    manifest.update({
        'id': 'fixture-v3', 'rows': len(prior_rows) + len(additions),
        'cases_sha256': cases_sha,
        'additional_admitted_train_sources': [{
            'id': qualification['source_id'],
            'cases': additions_path.name, 'cases_sha256': additions_sha,
            'rows': len(additions),
            'qualification_proof': qualification_path.name,
            'qualification_proof_sha256': qualification_sha,
        }],
    })
    audit_path = copied / 'union-audit.json'
    audit = {
        'schema': 'natlang.rl-task-breadth-union-audit/1',
        'conflicting_ids': [],
        'union': {'sha256': cases_sha, 'rows': len(prior_rows) + len(additions),
                  'exact_same_id_duplicates': 0, 'same_id_different_payload': 0},
        'base': manifest['preserved_base'],
        'broad': {'selection_proof_sha256': manifest['broad_selection_proof_sha256'],
                  'sha256': json.loads((copied / manifest['broad_selection_proof']).read_text())['cases_sha256'],
                  'rows': manifest['added_admitted_train_rows']},
        'source_registry_revalidation': {
            'all_sources_still_registered_admitted': True,
            'all_source_files_sha_match_selection_proof': True,
        },
        'current_protection': {
            'union_rows_intersecting_current_decision_holds': 0,
            'union_source_groups_intersecting_held': 0,
            'union_source_groups_intersecting_quality': 0,
        },
        'selection_protection': {
            'cases': manifest['added_admitted_train_rows'],
            'cases_sha256': json.loads((copied / manifest['broad_selection_proof']).read_text())['cases_sha256'],
            'omissions': {'nontrain': manifest['broad_nontrain_omissions']},
        },
    }
    audit_path.write_text(json.dumps(audit, sort_keys=True, separators=(',', ':')) + '\n')
    manifest['source_audit_sha256'] = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return copied


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


def test_qualified_scaffold_continuation_preserves_128_and_binds_12_additions(tmp_path):
    fixture = make_v3_fixture(tmp_path)
    resolved = resolve_task_source(fixture / 'task-source-manifest.json')
    prior = read_rows(V2 / 'cases.jsonl')
    rows = read_rows(resolved['cases'])
    qualified_source = read_rows(fixture / 'additional-scaffold-cases.jsonl')
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
    copied = make_v3_fixture(tmp_path)
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
    copied = make_v3_fixture(tmp_path)
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


def _port_record_row(requests):
    files = {f'items/item-{index:02d}.json': json.dumps({'request': request})
             for index, request in enumerate(requests, 1)}
    return {'family': 'curriculum_port_record_calls', 'semantics': {
        'folder_files': files,
        'expected': {f'item-{index:02d}': 'observed' for index in range(1, len(requests) + 1)},
    }}


def test_port_record_quarantine_predicate_is_content_based_and_complete():
    prompt = 'Continue this session with its next assistant message.\n\nuser: request'
    assert _ill_posed_port_record_continuation(_port_record_row([prompt, prompt]))
    assert not _ill_posed_port_record_continuation({'family': 'curriculum_port_record_calls',
        'semantics': {'expected': {'item-01': 'x'}, 'folder_files': {}}})
    assert not _ill_posed_port_record_continuation(_port_record_row([prompt, 'Use the supplied record.']))
    assert not _ill_posed_port_record_continuation({**_port_record_row([prompt]), 'family': 'other'})


def test_corrected_task_source_is_exact_parent_order_complement_and_preserves_proofs(tmp_path):
    parent = tmp_path / 'parent'
    shutil.copytree(V2, parent)
    output = tmp_path / 'corrected'
    result = build_correction(parent / 'task-source-manifest.json', output)
    resolved = resolve_task_source(output / 'task-source-manifest.json')
    parent_rows = read_rows(parent / 'cases.jsonl')
    corrected_rows = read_rows(resolved['cases'])
    assert result['rows'] == len(corrected_rows) == 76
    expected = [row for row in parent_rows if not _ill_posed_port_record_continuation(row)]
    assert corrected_rows == expected
    assert (output / 'parent-source' / 'task-source-manifest.json').read_bytes() == \
        (parent / 'task-source-manifest.json').read_bytes()
    for source_file in parent.iterdir():
        if source_file.is_file():
            assert (output / 'parent-source' / source_file.name).read_bytes() == source_file.read_bytes()
    assert resolved['source_correction_excluded_rows'] == 52


def test_corrected_task_source_rejects_arbitrary_row_omission(tmp_path):
    parent = tmp_path / 'parent'
    shutil.copytree(V2, parent)
    output = tmp_path / 'corrected'
    build_correction(parent / 'task-source-manifest.json', output)
    cases = output / 'cases.jsonl'
    cases.write_bytes(b''.join(cases.read_bytes().splitlines(keepends=True)[1:]))
    with pytest.raises(ValueError, match='exact predicate complement|binding'):
        resolve_task_source(output / 'task-source-manifest.json')
