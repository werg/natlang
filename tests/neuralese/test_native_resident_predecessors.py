"""Legacy resident checkpoints retain their exact witnessed append predecessor."""
import copy
import json

import pytest

from test_native_resident import setup
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.native_state_sft import continuation_predecessors
from natlang_neuralese.common.jsonio import write_json_durable as write_json


def adopted(tmp_path, monkeypatch):
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path/'cache'))
    paths, directory, manager, recipe = setup(tmp_path)
    resident.prepare(directory, recipe, 'sft', tokenizer_fingerprint='tokenizer-sha', role_ids=())
    train, held = manager.poll(7)
    state = resident.current(directory)
    training = {'source_receipts': [train.receipt(), held.receipt()],
        'options': {'source_predecessors': {'continued_native_graphs': 'earlier-static-prefix'},
                    'resident_adoption': {'generation': state['generation'], 'history': state['history']}}}
    return paths, recipe, training


def test_actual_resident_legacy_checkpoint_accepts_only_exact_bound_recipe_predecessor(tmp_path, monkeypatch):
    paths, recipe, training = adopted(tmp_path, monkeypatch)
    before = copy.deepcopy(training)
    declarations = {'continued_native_graphs': paths['old'], 'continued_held_native_graphs': paths['held']}
    result = continuation_predecessors(training, declarations, stage_id='sft')
    assert result == {'continued_native_graphs': resident.sha(paths['old']),
                      'continued_held_native_graphs': resident.sha(paths['held'])}
    assert training == before, 'verification cannot mutate the saved checkpoint'
    # A newly saved explicit predecessor needs no historical recipe to resume.
    training['options']['source_predecessors'] = result
    recipe.unlink()
    assert continuation_predecessors(training, declarations, stage_id='sft') == result


@pytest.mark.parametrize('change', ['recipe-bytes', 'saved-source', 'unrelated-predecessor', 'wrong-stage', 'unwitnessed'])
def test_legacy_predecessor_rejects_unbound_or_unrelated_evidence(tmp_path, monkeypatch, change):
    paths, recipe, training = adopted(tmp_path, monkeypatch)
    declared = paths['old']
    stage_id = 'sft'
    if change == 'recipe-bytes':
        value = json.loads(recipe.read_text()); value['description'] = 'changed after commit'; write_json(recipe, value)
    elif change == 'saved-source': training['source_receipts'][0]['sha256'] = 'different-source'
    elif change == 'unrelated-predecessor': declared = paths['held']
    elif change == 'wrong-stage': stage_id = 'other'
    elif change == 'unwitnessed': training['options'].pop('resident_adoption')
    with pytest.raises(ValueError):
        continuation_predecessors(training, {'continued_native_graphs': declared}, stage_id=stage_id)


def test_actual_current_source_is_a_valid_new_append_predecessor_without_history(tmp_path, monkeypatch):
    paths, recipe, training = adopted(tmp_path, monkeypatch)
    training['options'] = {}
    assert continuation_predecessors(training, {'continued_native_graphs': paths['new']}, stage_id='sft') == {
        'continued_native_graphs': resident.sha(paths['new'])}
