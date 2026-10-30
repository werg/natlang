"""Coefficient recipes stay separate from full backbone state updates."""
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


def test_direct_coefficient_learner_cannot_share_an_owned_update_slot(tmp_path):
    import fcntl
    from natlang_neuralese.rl.native_coefficient_stage import train
    with (tmp_path/'learner.lock').open('w') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):train(None,out=tmp_path)
        assert not (tmp_path/'checkpoint.pt').exists()


def test_native_admission_lease_rejects_before_any_source_mutation(tmp_path,monkeypatch):
    import fcntl
    from natlang_neuralese.rl import native_stage
    for name in ('rl-native-graphs.jsonl','rl-native-objective.json','rl-data-receipt.json'):
        (tmp_path/name).write_bytes(b'exact currently owned source')
    monkeypatch.setattr(native_stage,'prepare',lambda *args,**kwargs:pytest.fail('competing admission reached source preparation'))
    argv=[]
    for option in ('rollouts','rl-tokenizer','coefficient-policy','continue-from','held-native-graphs','out'):
        argv.extend(['--'+option,str(tmp_path)])
    with (tmp_path/'learner.lock').open('a+') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):native_stage.main(argv)
    assert all((tmp_path/name).read_bytes()==b'exact currently owned source'
               for name in ('rl-native-graphs.jsonl','rl-native-objective.json','rl-data-receipt.json'))


def test_native_output_lease_reuses_one_nested_owner_and_rejects_competing_thread(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from natlang_neuralese.rl.native_output_lease import coefficient_output_lease
    def compete():
        with coefficient_output_lease(tmp_path):pytest.fail('second caller entered another caller output lease')
    with coefficient_output_lease(tmp_path):
        with coefficient_output_lease(tmp_path):
            with ThreadPoolExecutor(max_workers=1) as pool:
                with pytest.raises(BlockingIOError):pool.submit(compete).result()
    with coefficient_output_lease(tmp_path):pass
