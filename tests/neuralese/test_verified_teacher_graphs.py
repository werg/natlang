import copy
import json
from pathlib import Path

import pytest

from scripts.neuralese_data import build_verified_teacher_trajectory_graphs as converter


def _row():
    return {'id': 'execution-a', 'version': 'natlang.teacher_trajectory.native/1',
        'task': {'program_ir': {'version': 'natlang.program/2', 'split': 'train',
                              'id': 'program-a', 'source_groups': ['train-g']}},
        'outcome': {'status': 'done', 'accepted': True, 'oracle': {'accepted': True},
                    'checks': {'exact_result': True}},
        'provenance': {'model': 'pennyroyal'},
        'execution_identity': {'execution_run_id': 'run-a'},
        'trajectory': [{'phase': 'action', 'assistant': {'content': 'correct'}}]}


@pytest.mark.parametrize('scope', ['held', 'quality', 'pending'])
def test_current_protection_excludes_even_previously_selected_success(monkeypatch, scope):
    monkeypatch.setattr(converter.source_review, 'pending_family_hold_reason', lambda _: None)
    binding = {'selected_ids': {'program-a'}, 'train_groups': {'train-g'}, 'held_groups': set(),
        'source_files': {'execution-a': {}},
        'current': {key: {'train-g'} if key == scope else set() for key in ('held', 'quality', 'pending')}}
    with pytest.raises(ValueError, match='current held/quality/source-review'):
        converter.validate_teacher_row(_row(), binding)


def test_original_response_binding_rejects_changed_combined_action(tmp_path):
    row = _row()
    path = tmp_path / 'response.jsonl'
    path.write_text(json.dumps(row) + '\n')
    binding = {'path': str(path), 'sha256': converter.sha256_file(path)}
    assert converter.verify_original_row(row, binding) == path
    changed = copy.deepcopy(row)
    changed['trajectory'][0]['assistant']['content'] = 'different but claimed success'
    with pytest.raises(ValueError, match='differs from its original response'):
        converter.verify_original_row(changed, binding)
    path.write_text(json.dumps(changed) + '\n')
    with pytest.raises(ValueError, match='artifact changed'):
        converter.verify_original_row(row, binding)


def test_current_projection_verifies_content_and_live_source_review(tmp_path, monkeypatch):
    live = tmp_path / 'live'
    live.mkdir()
    (live / 'holds.jsonl').write_text('')
    monkeypatch.setattr(converter.source_review, 'REGISTRY', live)
    monkeypatch.setattr(converter.source_review, 'pending_unpinned_source_groups', lambda: {'pending-g'})
    files = []
    for name, group in [('held-union/full/groups.jsonl', 'held-g'), ('quality-groups.jsonl', 'quality-g')]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'source_groups': [group]}) + '\n')
        files.append({'path': name, 'bytes': path.stat().st_size, 'sha256': converter.sha256_file(path)})
    manifest = {'schema': 'natlang.training-protection-projection/1', 'files': files,
        'held_groups': 1, 'quality_groups': 1,
        'source_review_sha256': converter.sha256_file(live / 'holds.jsonl')}
    path = tmp_path / 'projection.json'
    path.write_text(json.dumps(manifest))
    result = converter.current_protection(path)
    assert result['held'] == {'held-g'} and result['quality'] == {'quality-g'}
    assert result['pending'] == {'pending-g'}
    (live / 'holds.jsonl').write_text('changed review\n')
    with pytest.raises(ValueError, match='current source reviews'):
        converter.current_protection(path)
    (live / 'holds.jsonl').write_text('')
    (tmp_path / 'quality-groups.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='protection artifact differs'):
        converter.current_protection(path)


def test_historical_batches_are_bound_to_their_own_packet_not_future_queue(tmp_path, monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(converter, 'current_protection', lambda _: {'proof': {}})
    def put(name, value):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + '\n')
        return path
    source = put('native-trajectories.jsonl', {'id': 'execution-a'})
    cases = put('selected-cases.jsonl', {'id': 'program-a'})
    selection = {'schema': 'natlang.native-teacher-task-selection/1', 'status': 'passed',
        'output_sha256': converter.sha256_file(cases), 'selected_ids': ['program-a'],
        'train_groups': ['train-g'], 'held_groups': []}
    put('selection-proof.json', selection)
    teacher = {'teacher_role': 'independent stronger open teacher; not either student/onpolicy'}
    put('penny-server-binding.json', teacher)
    put('evidence-source-v2.json', {'cases_sha256': 'f' * 64, 'scope': 'new future queue'})
    put('global-protection-join.json', {'schema': 'natlang.native-teacher-global-protection-join/1',
        'status': 'passed', 'selected_packet_sha256': selection['output_sha256'],
        'held_group_intersections': [], 'quality_group_intersections': []})
    collection = {'schema': 'natlang.sandbox-strong-teacher-collection/1',
        'source_cases_sha256': selection['output_sha256'], 'protected_selection': selection,
        'teacher': teacher}
    path = put('batch-bindings/batch-0000/teacher-source.json', collection)
    item = {'path': str(path), 'source': '/work/generation/batch-0000/teacher-source.json',
            'bytes': path.stat().st_size, 'sha256': converter.sha256_file(path)}
    transfer = {'schema': 'natlang.verified-teacher-trajectory-transfer/1', 'training_admission': False,
        'combined': {'path': str(source), 'bytes': source.stat().st_size, 'sha256': converter.sha256_file(source)},
        'files': [item]}
    put('transfer-receipt.json', transfer)
    args = SimpleNamespace(source=source, protection_manifest=tmp_path / 'current.json')
    assert converter.source_binding(args)['collections'][item['source']]['sha256'] == item['sha256']
    collection['source_cases_sha256'] = 'e' * 64
    put('batch-bindings/batch-0000/teacher-source.json', collection)
    item.update(bytes=path.stat().st_size, sha256=converter.sha256_file(path))
    put('transfer-receipt.json', transfer)
    with pytest.raises(ValueError, match='collection receipt differs'):
        converter.source_binding(args)
