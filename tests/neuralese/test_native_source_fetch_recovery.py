"""Only a pre-update source failure can save completed optimizer work."""
import copy
import contextlib
import json
import random

import pytest
import torch

from natlang_neuralese.train import native_state_sft as trainer
from test_native_state_sft import graph


def source(base, family):
    base.mkdir()
    if family == 'lfm':
        from test_native_serve import _tiny_source
        _tiny_source(base)
    else:
        from test_mellum_port import tiny_mellum
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast
        tiny_mellum(base)
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel(
            {f't{i}':i for i in range(128)},unk_token='t0')),unk_token='t0',
            bos_token='t1',eos_token='t2',pad_token='t3')
        tokenizer.backend_tokenizer.pre_tokenizer=WhitespaceSplit()
        tokenizer.chat_template="{% for message in messages %}{{ message.get('reasoning_content', '') }}{{ message['content'] }}{% endfor %}assistant:"
        tokenizer.save_pretrained(base)
    _,_,_,config=trainer.load_native_model(base,dtype=torch.bfloat16,rank=4)
    paths={}
    for split in ('train','test'):
        paths[split]=base.parent/(split+'.jsonl')
        paths[split].write_text(json.dumps(dict(graph(split),split=split,
            source_groups=[split],tokenizer_fingerprint=config['tokenizer_fingerprint']))+'\n')
    return paths


def equal(a,b):
    if torch.is_tensor(a):torch.testing.assert_close(a,b,atol=0,rtol=0)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for left,right in zip(a,b):equal(left,right)
    else:assert a==b


def run(base,paths,out,*,refresh=None,resume=None):
    args=['--base',str(base),'--dtype','bfloat16','--rank','4','--device','cpu',
        '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
        '--epochs','4','--eval-every','1','--checkpoint-every','2',
        '--checkpoint-seconds','0','--seed','23','--out',str(out)]
    if json.loads((base/'config.json').read_text())['model_type']=='mellum':
        args+=['--fp8-training','triton-experts']
    if refresh is not None:args+=['--refresh-after-updates',str(refresh)]
    if resume is not None:args+=['--continue-from',str(resume)]
    return trainer.main(args)


@pytest.mark.parametrize('family',['lfm','mellum'])
@pytest.mark.parametrize('save_fails',[False,True])
def test_fetch_failure_checkpoint_matches_completed_reference_and_resumes(tmp_path,monkeypatch,family,save_fails):
    if family=='mellum':
        # Exercise the genuine tiny Mellum backbone/optimizer on CPU. Only the
        # GPU quantization backend is replaced; checkpoint its state explicitly.
        from natlang_neuralese.train import fp8_backend
        class CPUBackend:
            def reset_row_counts(self):pass
            def autocast(self):return contextlib.nullcontext()
            def native_modules(self):return contextlib.nullcontext()
            def checkpoint(self,function,*args,**kwargs):
                from torch.utils.checkpoint import checkpoint
                return checkpoint(function,*args,**kwargs)
            def extra_state(self):return {'cpu_test_backend':True}
            def restore_extra_state(self,value):assert value==self.extra_state()
            def inventory(self):return {'cpu_test_backend':True}
        monkeypatch.setattr(fp8_backend,'make_backend',lambda *a,**k:CPUBackend())
    base=tmp_path/'base';paths=source(base,family);reference=tmp_path/'reference'
    original_save=trainer._save;prefix={}
    def capture_save(policy,*args,**kwargs):
        result=original_save(policy,*args,**kwargs)
        if kwargs['step']==2:prefix['checkpoint']=torch.load(policy.slot,weights_only=False,map_location='cpu')
        return result
    with monkeypatch.context()as patch:
        patch.setattr(trainer,'_save',capture_save)
        assert run(base,paths,reference,refresh=3)==0
    reference_three=torch.load(reference/'checkpoint.pt',weights_only=False,map_location='cpu')
    original_stream=trainer.stream_pending_batches;failure=ValueError('injected source CE failure before update four');boundary={}
    def broken_stream(*args,**kwargs):
        stream=original_stream(*args,**kwargs)
        try:
            for count in range(5):
                if count==3:
                    boundary['rng']=trainer.capture_training_rng_state('cpu')
                    random.random();torch.rand(13)
                    raise failure
                yield next(stream)
        finally:stream.close()
    save_calls=[]
    def save(policy,*args,**kwargs):
        save_calls.append(dict(kwargs.get('metrics',{})))
        if save_fails and 'source_fetch_failure'in kwargs['metrics']:
            random.random();torch.rand(7)
            raise OSError('injected recovery disk failure')
        return original_save(policy,*args,**kwargs)
    interrupted=tmp_path/'interrupted'
    with monkeypatch.context()as patch:
        patch.setattr(trainer,'stream_pending_batches',broken_stream)
        patch.setattr(trainer,'_save',save)
        with pytest.raises(ValueError)as caught:run(base,paths,interrupted)
    assert caught.value is failure
    assert len(save_calls)==2  # Step-two cadence plus one specialized step-three recovery attempt.
    assert sum('source_fetch_failure' in metrics for metrics in save_calls)==1
    equal(trainer.capture_training_rng_state('cpu'),boundary['rng'])
    actual=torch.load(interrupted/'checkpoint.pt',weights_only=False,map_location='cpu')
    expected=prefix['checkpoint']if save_fails else reference_three
    for key in ('optimizer','rng','exposure_counts','global_step','step','epoch','cursor','source_row_bindings','held_row_bindings'):
        equal(actual['training'][key],expected['training'][key])
    for key in actual.keys()-{'training'}:equal(actual[key],expected[key])
    assert actual['training']['exposure_counts']=={'train':2 if save_fails else 3}
    if save_fails:
        assert any('injected recovery disk failure'in note for note in failure.__notes__)
        assert 'source_fetch_failure'not in actual['training']['metrics']
    else:
        assert actual['training']['metrics']['source_fetch_failure']=={'type':'ValueError','message':str(failure)}
    assert not(interrupted/'native-training-receipt.json').exists()
    uninterrupted=tmp_path/'uninterrupted';assert run(base,paths,uninterrupted)==0
    assert run(base,paths,interrupted,resume=interrupted/'checkpoint.pt')==0
    left=torch.load(interrupted/'checkpoint.pt',weights_only=False,map_location='cpu')
    right=torch.load(uninterrupted/'checkpoint.pt',weights_only=False,map_location='cpu')
    for key in ('optimizer','rng','exposure_counts','global_step','step','cursor'):
        equal(left['training'][key],right['training'][key])
    for key in left.keys()-{'training'}:equal(left[key],right[key])


def test_initial_fetch_failure_has_no_update_credit_or_checkpoint(tmp_path,monkeypatch):
    base=tmp_path/'base';paths=source(base,'lfm');failure=ValueError('invalid first source batch');boundary={}
    def fail(*args,**kwargs):
        boundary['rng']=trainer.capture_training_rng_state('cpu');random.random();torch.rand(9)
        raise failure
        yield
    monkeypatch.setattr(trainer,'stream_pending_batches',fail)
    monkeypatch.setattr(trainer,'native_training_update',lambda *a,**k:pytest.fail('bad batch reached update'))
    monkeypatch.setattr(trainer,'_save',lambda *a,**k:pytest.fail('zero updates earned a recovery checkpoint'))
    out=tmp_path/'out'
    with pytest.raises(ValueError)as caught:run(base,paths,out)
    assert caught.value is failure
    equal(trainer.capture_training_rng_state('cpu'),boundary['rng'])
    assert not(out/'checkpoint.pt').exists()and not(out/'native-training-receipt.json').exists()


def test_partial_update_exception_does_not_enter_source_recovery(tmp_path,monkeypatch):
    base=tmp_path/'base';paths=source(base,'lfm');update=trainer.native_training_update;calls=0;failure=RuntimeError('partial optimizer failure')
    def fail(model,optimizer,*args,**kwargs):
        nonlocal calls
        calls+=1
        if calls==3:
            with torch.no_grad():next(model.parameters()).add_(.125)
            raise failure
        return update(model,optimizer,*args,**kwargs)
    save=trainer._save
    def guarded_save(*args,**kwargs):
        assert 'source_fetch_failure'not in kwargs['metrics']
        return save(*args,**kwargs)
    monkeypatch.setattr(trainer,'native_training_update',fail);monkeypatch.setattr(trainer,'_save',guarded_save)
    out=tmp_path/'out'
    with pytest.raises(RuntimeError)as caught:run(base,paths,out)
    assert caught.value is failure
    checkpoint=torch.load(out/'checkpoint.pt',weights_only=False,map_location='cpu')
    assert checkpoint['training']['step']==2 and checkpoint['training']['exposure_counts']=={'train':2}
    assert not(out/'native-training-receipt.json').exists()
