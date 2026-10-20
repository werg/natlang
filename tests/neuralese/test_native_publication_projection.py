from pathlib import Path
import copy
import pytest
from natlang_neuralese.model.native_checkpoint_control import publication_selection, publication_owner_configuration
from natlang_neuralese.rl.native_rounds import configure_provider

@pytest.fixture
def declaration(tmp_path):
    runtime = Path(__file__).resolve().parents[2]
    return dict(model='mellum', native_checkpoint_publication=publication_selection(namespace=str(tmp_path)),
        native_checkpoint_publication_role='mellum',
        native_checkpoint_publication_owner_pointer=str(tmp_path/'mellum/current-owner.json'),
        native_checkpoint_publication_runtime_root=str(runtime),
        native_checkpoint_publication_reserve_bytes=20*1024**3)

@pytest.mark.parametrize('stage', ['native_full_sft', 'd24_offline_preference', 'native_rl'])
def test_common_provider_factory_preserves_full_owner_contract_for_every_stage(tmp_path, declaration, stage):
    original = dict(model='mellum', native_actor_mode='raw-state', runtime='/typed/task/runtime',
        trainer_parameters={'stage_id':stage, 'learning_rate':3e-6})
    config = copy.deepcopy(original)
    configure_provider(config, declaration, {}, {}, tmp_path/'config.json', write=False)
    for key,value in publication_owner_configuration(declaration).items():assert config[key] == value
    assert config['runtime'] == original['runtime']
    assert config['trainer_parameters']['stage_id'] == stage
    assert config['trainer_parameters']['learning_rate'] == 3e-6
    assert config['trainer_parameters']['checkpoint_publication'] is True

@pytest.mark.parametrize('field', ['native_checkpoint_publication', 'native_checkpoint_publication_role',
    'native_checkpoint_publication_owner_pointer', 'native_checkpoint_publication_runtime_root'])
def test_partial_owner_contract_is_rejected(declaration, field):
    declaration.pop(field)
    with pytest.raises(ValueError, match='complete declared'):
        publication_owner_configuration(declaration)

@pytest.mark.parametrize('field,value,pattern', [
    ('native_checkpoint_publication_role','lfm','declared model'),
    ('native_checkpoint_publication_owner_pointer','relative/path','absolute'),
    ('native_checkpoint_publication_reserve_bytes',True,'nonnegative'),
    ('native_checkpoint_publication_reserve_bytes',-1,'nonnegative'),
])
def test_conflicting_or_malformed_owner_fields_rejected(declaration, field, value, pattern):
    declaration[field] = value
    with pytest.raises(ValueError, match=pattern):publication_owner_configuration(declaration)

def test_wrong_python_runtime_does_not_use_task_runtime_or_another_package(tmp_path,declaration):
    declaration['native_checkpoint_publication_runtime_root'] = str(tmp_path)
    with pytest.raises(ValueError,match='sealed producer'):publication_owner_configuration(declaration)

def test_configured_model_cannot_receive_another_roles_pointer(tmp_path,declaration):
    with pytest.raises(ValueError,match='declared model'):
        configure_provider({'model':'lfm'},declaration,{}, {}, tmp_path/'config.json',write=False)

def test_selection_only_legacy_configuration_remains_valid(declaration):
    assert publication_owner_configuration({'native_checkpoint_publication':declaration['native_checkpoint_publication']}) == {}
