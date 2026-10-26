import copy
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.train.native_foundation_recovery import SCHEMA, restore_declaration


class Source:
    def __init__(self, path):
        self.path = self.source_identity_path = path
        self.file_sha256 = sha256_file_hex(path)
        self.kind = 'native-graphs'
        self.bindings = {'producer-consumer': 'binding-original'}
        self.entries = [{'id': key, 'binding': binding, 'communication': True,
                         'producer_consumer': True, 'groups': ['train-world'], 'supervised_tokens': 4}
                        for key, binding in self.bindings.items()]

    def __len__(self):
        return len(self.entries)

    def ordered_ids(self):
        return list(self.bindings)


@pytest.fixture
def case(tmp_path):
    def write(name, value):
        path = tmp_path / name
        path.write_text(json.dumps(value))
        return {'path': str(path), 'sha256': sha256_file_hex(path)}
    config = {'architecture': 'lfm2', 'base_inventory_sha256': 'b'*64,
              'tokenizer_fingerprint': 't'*64, 'marker_ids': [1, 2, 3, 4]}
    original_path = '/original/foundation-sources.json'
    source = Source(Path(write('foundation.json', {'sources': ['fixture']})['path']))
    checkpoint = {'sha256': 'c'*64, 'bytes': 1234}
    recipe = {'input_bindings': {'foundation': {'path': original_path, 'sha256': source.file_sha256}},
              'stages': [{'id': 'full', 'kind': 'native_state_sft',
                          'inputs': {'foundation_native_graphs': 'foundation'},
                          'parameters': {'communication_foundation': True, 'epochs': 1.}}]}
    metadata = {'checkpoint_identity': checkpoint, 'model_config': config,
                'training': {'global_step': 4, 'options': {'stage_id': 'full', 'epochs': 1.,
                    'communication_foundation': True, 'foundation_native_graphs': original_path}}}
    verification = {'passed': True, 'checkpoint': checkpoint, 'architecture': 'lfm2',
                    'optimizer_present': True, 'rng_present': True, 'stage_id': 'full', 'global_step': 4}
    proof = {'schema': SCHEMA, 'stage_id': 'full', 'reason': 'Recovered exact historical declaration.',
             'historical_recipe': write('recipe.json', recipe),
             'historical_training_metadata': write('metadata.json', metadata),
             'historical_checkpoint_verification': write('verification.json', verification),
             'foundation_sources': {'path': str(source.path), 'sha256': source.file_sha256}}
    proof_path = tmp_path / 'recovery.json'
    proof_path.write_text(json.dumps(proof))
    frame = {'global_step': 9, 'source_row_bindings': dict(source.bindings),
             'exposure_counts': {'producer-consumer': 0, 'ordinary': 7},
             'required_visits': {'producer-consumer': 1, 'ordinary': 9},
             'pending_source_rows': {'pending': {'binding': 'pending-original', 'required_visits': 1}},
             'options': {'communication_foundation': False}, 'completed_foundations': [],
             'optimizer': {'unchanged': [1, 2]}, 'rng': {'unchanged': [3, 4]}}
    kwargs = dict(model_config=config, stage_id='full', training_frame=frame,
                  snapshot_ancestors=['c'*64], foundation=source,
                  active_bindings=dict(source.bindings), epochs=1.)
    return proof_path, proof, kwargs


def test_recovery_restores_only_declaration_without_mutating_any_earned_state(case):
    path, _, kwargs = case
    prior = copy.deepcopy(kwargs['training_frame'])
    result = restore_declaration(path, **kwargs)
    assert kwargs['training_frame'] == prior
    assert result['new_exposure'] == 0 and result['qualification_granted'] is False
    assert result['historical_checkpoint']['sha256'] == 'c'*64
    assert result['cohort_sha256']


@pytest.mark.parametrize('document,field,value', [
    ('historical_checkpoint_verification', 'passed', False),
    ('historical_checkpoint_verification', 'rng_present', False),
    ('historical_checkpoint_verification', 'optimizer_present', False),
    ('historical_checkpoint_verification', 'stage_id', 'other'),
    ('historical_checkpoint_verification', 'global_step', 8),
    ('historical_training_metadata', 'model_config.tokenizer_fingerprint', 'other'),
    ('historical_training_metadata', 'model_config.architecture', 'mellum'),
    ('historical_training_metadata', 'model_config.marker_ids', [9, 10]),
    ('historical_training_metadata', 'training.options.stage_id', 'other'),
    ('historical_training_metadata', 'training.options.communication_foundation', False),
    ('historical_training_metadata', 'training.options.epochs', 2.),
    ('historical_training_metadata', 'training.options.foundation_native_graphs', '/different/header.json'),
    ('historical_training_metadata', 'checkpoint_identity.sha256', 'd'*64),
    ('historical_training_metadata', 'training.global_step', 20),
    ('historical_recipe', 'input_bindings.foundation.sha256', 'd'*64),
])
def test_recovery_rejects_incompatible_historical_declaration(case, document, field, value):
    path, proof, kwargs = case
    reference = proof[document]
    target = Path(reference['path'])
    contents = json.loads(target.read_text())
    cursor = contents
    parts = field.split('.')
    for part in parts[:-1]:
        cursor = cursor[part]
    cursor[parts[-1]] = value
    target.write_text(json.dumps(contents))
    reference['sha256'] = sha256_file_hex(target)
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match='foundation recovery'):
        restore_declaration(path, **kwargs)


@pytest.mark.parametrize('change', ['unrelated-ancestor', 'active-row', 'saved-row', 'ordinary-row', 'wrong-stage', 'edited-bytes'])
def test_recovery_rejects_wrong_current_state_and_changed_references(case, change):
    path, proof, kwargs = case
    if change == 'unrelated-ancestor':
        kwargs['snapshot_ancestors'] = ['d'*64]
    elif change == 'active-row':
        kwargs['active_bindings']['producer-consumer'] = 'changed'
    elif change == 'saved-row':
        kwargs['training_frame']['source_row_bindings'].clear()
    elif change == 'ordinary-row':
        kwargs['foundation'].entries[0]['producer_consumer'] = False
    elif change == 'wrong-stage':
        kwargs['stage_id'] = 'other'
    else:
        Path(proof['historical_recipe']['path']).write_text('{}')
    with pytest.raises(ValueError, match='foundation recovery'):
        restore_declaration(path, **kwargs)


def test_recovery_role_reaches_recipe_cli_and_seed_fallback(case):
    path, _, _ = case
    from natlang_neuralese.train.recipe import stage_input_args
    from natlang_neuralese.rl.native_supervised import bind_seed_sources
    reference = {'path': str(path), 'sha256': sha256_file_hex(path)}
    assert stage_input_args({'foundation_declaration_recovery': reference}, 'native_state_sft') == [
        '--foundation-declaration-recovery', str(path)]
    config = {'seed_command': ['python', '-m', 'natlang_neuralese.train.native_state_sft']}
    bind_seed_sources(config, {'foundation_declaration_recovery': reference})
    assert config['foundation_declaration_recovery'] == str(path)
    assert '--foundation-declaration-recovery' in config['seed_command']
