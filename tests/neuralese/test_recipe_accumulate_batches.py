import pytest
from natlang_neuralese.train.recipe import HANDLERS, validate_stage_parameters, stage_parameter_args, effective_stage_parameters, native_stage_parameter_handoffs
from natlang_neuralese.train.native_state_sft import build_parser

def test_accumulation_is_declared_validated_and_forwarded():
    assert 'accumulate_batches' in HANDLERS['native_state_sft']['parameters']
    params={'accumulate_batches':16}
    validate_stage_parameters('native_state_sft',params)
    args=stage_parameter_args(params)
    assert args==['--accumulate-batches','16']
    assert build_parser().parse_args(args+['--native-graphs','train.json','--held-native-graphs','held.json','--out','tmp']).accumulate_batches==16
    for invalid in (0,-1,True,1.5):
        with pytest.raises(ValueError):validate_stage_parameters('native_state_sft',{'accumulate_batches':invalid})


def test_rl_inherits_and_validates_accumulation(tmp_path):
    recipe={'stage_parameter_defaults':{'native_state_rl':{'accumulate_batches':7}},'stages':[{'id':'rl','kind':'native_state_rl','parameters':{}}]}
    stage=recipe['stages'][0]
    assert effective_stage_parameters(stage,{},recipe)['accumulate_batches']==7
    successor={'stage_parameter_defaults':{'native_state_rl':{'accumulate_batches':9}},'stages':recipe['stages']}
    changes,_,_=native_stage_parameter_handoffs(recipe,successor,tmp_path)
    assert changes['rl']['accumulate_batches']==[7,9]
    validate_stage_parameters('native_state_rl',{'accumulate_batches':7})
    with pytest.raises(ValueError):validate_stage_parameters('native_state_rl',{'accumulate_batches':0})
