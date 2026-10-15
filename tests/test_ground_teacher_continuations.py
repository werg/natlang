import copy
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('ground_teacher_continuations', Path(__file__).parents[1] / 'scripts/neuralese_data/ground_teacher_continuations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def source():
    item = json.dumps({'source_record_id': 'source:1', 'request': 'Continue this search.',
                       'evidence': 'The lamp is on sidetable 2.', 'answer_format': 'Next message.'})
    return {'id': 'program:1', 'family': 'curriculum_port_record_calls', 'split': 'train',
        'source_groups': ['session:1'], 'source_ids': ['source:1'],
        'semantics': {'expected': {'item-01': 'Action: go to shelf 4'},
            'folder_files': {'task.json': json.dumps({'items': ['items/item-01.json'], 'output_contract': module.OUTPUT_CONTRACT}), 'items/item-01.json': item},
            'expected_files': {'items/item-01.json': item}},
        'curriculum': {'variant': 'iterate-records/1', 'reference': {'children': [
            {'source_binding': {'path': 'items/item-01.json', 'sha256': 'old'},
             'calls': [['eval', {'code': 'item.evidence'}], ['return_result', {'status': 'success', 'value': 'Action: go to shelf 4'}]]}]}}}


def test_arbitrary_future_search_becomes_explicit_grounded_extraction_without_relabeling_source():
    original = source(); unchanged = copy.deepcopy(original)
    derived, changed = module.ground(original)
    assert changed and original == unchanged
    item_text = derived['semantics']['folder_files']['items/item-01.json']
    item = json.loads(item_text)
    assert item['source_request'] == 'Continue this search.'
    assert 'recorded_continuation.assistant_message' in item['request']
    assert json.loads(item['evidence'].split('\n\n')[-1])['recorded_continuation']['assistant_message'] == 'Action: go to shelf 4'
    assert derived['semantics']['expected'] == original['semantics']['expected']
    assert derived['semantics']['files_oracle'] == {'compare': 'json-string-record', 'threshold': 1}
    assert derived['semantics']['expected_files']['items/item-01.json'] == item_text
    assert derived['curriculum']['reference']['children'][0]['source_binding']['sha256'] == module.sha256_hex(item_text.encode())
    assert derived['source_ids'] == original['source_ids'] and derived['source_groups'] == original['source_groups']
    assert derived['teacher_derivation']['training_admission'] is False
    assert derived['teacher_derivation']['source_program_sha256'] == module.canonical_json_sha256_hex(original)


def test_reference_target_mismatch_cannot_be_silently_rewritten():
    row = source()
    row['curriculum']['reference']['children'][0]['calls'][-1][1]['value'] = 'different'
    with pytest.raises(ValueError, match='target differs'):
        module.ground(row)


def test_held_and_unprotected_rows_are_rejected():
    row = source(); row['split'] = 'test'
    with pytest.raises(ValueError, match='train-only'):
        module.ground(row)
    row = source(); row['source_groups'] = []
    with pytest.raises(ValueError, match='source groups'):
        module.ground(row)


def test_unrelated_valid_family_is_preserved_and_explicitly_counted():
    row = source(); row['family'] = 'curriculum_decision_skill_catalog'
    result, changed = module.ground(row)
    assert not changed and result == row and result is not row


def test_streamed_campaign_writes_exact_derivative_and_refuses_overwrite(tmp_path):
    path = tmp_path / 'source.jsonl'; path.write_text(json.dumps(source()) + '\n')
    out = tmp_path / 'derived'
    receipt = module.prepare(path, out)
    row = json.loads((out / 'cases.jsonl').read_text())
    assert receipt['rewritten'] == 1 and row['id'].endswith(':recorded-extraction-v2')
    assert receipt['cases']['sha256'] == module.sha256_file_hex(out / 'cases.jsonl')
    with pytest.raises(ValueError, match='overwrite'):
        module.prepare(path, out)


def test_unrelated_byte_contract_is_not_silently_relaxed():
    row = source()
    task = json.loads(row['semantics']['folder_files']['task.json'])
    task['output_contract'] = 'Write these exact bytes.'
    row['semantics']['folder_files']['task.json'] = json.dumps(task)
    with pytest.raises(ValueError, match='declared JSON string-map'):
        module.ground(row)


def test_unrelated_file_verifier_is_not_replaced():
    row = source(); row['semantics']['files_oracle'] = {'compare': 'csv'}
    with pytest.raises(ValueError, match='unrelated file oracle'):
        module.ground(row)
