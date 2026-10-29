import pytest
from natlang_neuralese.train.recipe import HANDLERS,validate_stage_parameters,stage_parameter_args,effective_stage_parameters
from natlang_neuralese.train.native_state_sft import build_parser


@pytest.mark.parametrize('kind',['native_state_sft','native_state_rl','native_offline_preference'])
def test_muon_rounding_is_declared_inherited_and_forwarded(kind):
    assert 'muon_rounding' in HANDLERS[kind]['parameters']
    recipe={'stage_parameter_defaults':{kind:{'muon_rounding':'stochastic'}},'stages':[{'id':'run','kind':kind,'parameters':{}}]}
    params=effective_stage_parameters(recipe['stages'][0],{},recipe)
    assert params['muon_rounding']=='stochastic'
    args=stage_parameter_args(params);assert args==['--muon-rounding','stochastic']
    assert build_parser().parse_args(args+['--native-graphs','train','--held-native-graphs','held','--out','out']).muon_rounding=='stochastic'
    validate_stage_parameters(kind,params)
    with pytest.raises(ValueError):validate_stage_parameters(kind,{'muon_rounding':'bogus'})
