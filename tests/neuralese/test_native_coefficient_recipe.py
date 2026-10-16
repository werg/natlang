"""Coefficient recipes stay separate from full backbone state/semantic updates."""
import copy
import json
from pathlib import Path

import pytest


def test_coefficient_handler_uses_real_admission_and_separate_result_contract():
    from natlang_neuralese.train.recipe import HANDLERS,NATIVE_TRAINING_KINDS,HEADLESS_NATIVE_KINDS,require_gate,validate_stage_parameters
    handler=HANDLERS['native_coefficient_rl']
    assert handler['module']=='natlang_neuralese.rl.native_stage'
    assert 'coefficient_policy' in handler['required_inputs']
    assert 'native_coefficient_rl' in HEADLESS_NATIVE_KINDS and 'native_coefficient_rl' not in NATIVE_TRAINING_KINDS
    validate_stage_parameters('native_coefficient_rl',{'epochs':2,'coefficient_optimizer':{'name':'adam','learning_rate':.003}})
    good={'schema':'natlang.native-coefficient-training-report/1','completed':True,
          'qualification_complete':True,'base_checkpoint_unchanged':True}
    require_gate(good,'native_coefficient_rl')
    for key in good:
        wrong=dict(good);wrong.pop(key)
        with pytest.raises(ValueError,match='native coefficient'):require_gate(wrong,'native_coefficient_rl')
    for parameters in ({'epochs':0},{'coefficient_optimizer':{'name':'fake'}},{'fp8_training':'pretend'}):
        with pytest.raises(ValueError):validate_stage_parameters('native_coefficient_rl',parameters)


def test_parallel_coefficient_cycles_resume_without_fabricating_semantic_or_backbone_updates(tmp_path,monkeypatch):
    from natlang_neuralese.rl import native_rounds
    from natlang_neuralese.rl import native_refresh
    parent=tmp_path/'actual-parent.pt';parent.write_bytes(b'coordinator fixture parent')
    adapter=tmp_path/'adapter.json';adapter.write_text(json.dumps({'semantic_obligations':{'complete':True}}))
    paths=[]
    for name in ('lfm','mellum'):
        policy=tmp_path/(name+'-policy.json');policy.write_text(json.dumps({'fixture_actor':name}))
        config={'native_state':True,'native_participant':name,'checkpoint':str(parent),
            'semantic_adapter':str(adapter),'semantic_refresh_config':str(tmp_path/'curriculum.json'),
            'out':str(tmp_path/name),'coefficient_policy':str(policy)}
        path=tmp_path/(name+'-config.json');path.write_text(json.dumps(config));paths.append(path)
    calls=[]
    def learner(path,**kwargs):
        config=json.loads(path.read_text());calls.append((config['native_participant'],kwargs['rounds']))
        out=Path(config['out']);out.mkdir(exist_ok=True)
        slot=out/'state.json';prior=json.loads(slot.read_text()) if slot.exists() else {'trained_rounds':0}
        state={'phase':'collect','trained_rounds':prior['trained_rounds']+1,'checkpoint':str(parent),
            'coefficient_policy':config['coefficient_policy'],
            'last_learner_progress':{'schema':'natlang.native-coefficient-training-report/1','qualification_complete':True}}
        slot.write_text(json.dumps(state));return state
    monkeypatch.setattr(native_rounds,'run',learner)
    monkeypatch.setattr(native_refresh,'refresh',lambda **kwargs:pytest.fail('unchanged base cannot be passed off as a semantic refresh'))
    first=native_rounds.run_rounds(paths,tmp_path/'joint',rounds=2,device='cpu')
    assert first['deployment'] is None and set(first['coefficient_policies'])=={'lfm','mellum'}
    receipt=json.loads((tmp_path/'joint/coefficient-latest.json').read_text())
    assert receipt['next_round']==2 and receipt['base_checkpoint_updated'] is False and receipt['semantic_refresh_executed'] is False
    native_rounds.run_rounds(paths,tmp_path/'joint',rounds=1,device='cpu')
    receipt=json.loads((tmp_path/'joint/coefficient-latest.json').read_text())
    assert receipt['next_round']==3 and all(value['trained_rounds']==3 for value in receipt['policies'].values())
    assert not (tmp_path/'joint/latest.json').exists() and len(calls)==6
