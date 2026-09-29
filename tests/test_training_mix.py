import json
import pytest
from scripts.audit_training_mix import audit, modality


def test_mix_counts_only_usable_train_decisions_and_reports_expansion(tmp_path):
    path = tmp_path / 'rows.jsonl'
    rows = [{'id': str(i), 'split': 'train', 'task_modality': 'tree-edit' if i == 0 else 'function'} for i in range(12)]
    rows += [{'id': 'held', 'split': 'test', 'task_modality': 'directory-reducer'},
             {'id': 'rejected', 'training_admission': {'approved': False}, 'task_modality': 'directory-reducer'}]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    report = audit([path])
    assert report['train_decisions'] == 12
    assert report['reducer_decisions'] == 0
    assert report['by_modality']['tree-edit'] == 1
    assert report['additional_reducer_decisions_needed'] == 4
    assert not report['target_met']
    rows.extend({'id': 'r' + str(i), 'task_modality': 'directory-reducer'} for i in range(4))
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    assert audit([path])['target_met']
    with pytest.raises(ValueError, match='duplicate decision'):
        audit([path, path])


def test_legacy_families_are_recognized_but_tree_edits_stay_separate():
    assert modality({'program_id': 'inline-curriculum:source_tatqa:123:v1'}) == 'directory-reducer'
    assert modality({'program_id': 'inline-curriculum:source_treedst:123:v1'}) == 'tree-edit'
    assert modality({'task_modality': 'function', 'program_id': 'inline-curriculum:source_tatqa:123:v1'}) == 'function'
