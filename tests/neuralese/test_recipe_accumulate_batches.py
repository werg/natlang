import pytest
from natlang_neuralese.train.recipe import HANDLERS, validate_stage_parameters, stage_parameter_args, effective_stage_parameters
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
