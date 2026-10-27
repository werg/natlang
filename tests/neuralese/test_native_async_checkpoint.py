import copy
import errno
import random
import threading
import time

import pytest
import torch

from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.trajectory_state import immutable_cpu_snapshot


def _full_state(model, optimizer, step, exposure):
    return {'step':step, 'model':model.state_dict(), 'training':{
        'optimizer':optimizer.state_dict(), 'rng':{'python':random.getstate(),
        'torch':torch.get_rng_state().clone()}, 'exposure_counts':dict(exposure)}}


def _assert_tree_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor) and torch.equal(actual,expected)
    elif isinstance(expected, dict):
        assert actual.keys()==expected.keys()
        for key,value in expected.items():_assert_tree_equal(actual[key],value)
    elif isinstance(expected, (tuple,list)):
        assert type(actual) is type(expected) and len(actual)==len(expected)
        for left,right in zip(actual,expected):_assert_tree_equal(left,right)
    else:
        assert actual==expected


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_native_optimizer_update_proceeds_while_exact_full_snapshot_is_written(tmp_path,monkeypatch,family):
    from test_shared_state import tiny
    from natlang_neuralese.data.native_graph import SCHEMA,parse_graph
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.train.native_state_sft import make_native_optimizer,native_training_update
    from natlang_neuralese.train import trajectory_state

    torch.manual_seed(1907)
    port=tiny(family)
    if family=='mellum':port.hf.natlang_precision='bf16'
    model=SharedStateModel(port,(60,61,62,63),rank=4)
    model.train()
    optimizer,_=make_native_optimizer(model,lr=1e-4,weight_decay=0.)
    example=parse_graph({'schema':SCHEMA,'id':'async-'+family,'split':'train','source_groups':['fixture'],
        'tokenizer_fingerprint':'fixture','nodes':[{'name':'answer','token_ids':[4,5,6,7,8],
        'target_weights':[0,1,1,1,1]}]},tokenizer_fingerprint='fixture')
    native_training_update(model,optimizer,[example],checkpoint_layers=False,chunk_tokens=8)
    original_exposure={'async-'+family:1}
    state=_full_state(model,optimizer,1,original_exposure)
    state['training']['options']={'nested':{'values':[17]}}
    parameter=next(model.parameters()).detach()
    state['alias_probe']=parameter
    state['alias_probe_view']=parameter.reshape(-1)[::2]
    expected=immutable_cpu_snapshot(state)
    entered,release=threading.Event(),threading.Event()
    real_write=trajectory_state.atomic_checkpoint
    def held_write(path,value):
        entered.set()
        if not release.wait(10):raise TimeoutError('checkpoint test did not release writer')
        return real_write(path,value)
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',held_write)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1<<50)
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    event=policy.save(state,async_allowed=True,async_optimizer=optimizer)
    assert event['async_pending'] and entered.wait(5)
    state['training']['exposure_counts']['async-'+family]=99
    state['training']['options']['nested']['values'][0]=99
    before={name:value.detach().clone() for name,value in model.named_parameters()}
    native_training_update(model,optimizer,[example],checkpoint_layers=False,chunk_tokens=8)
    assert any(not torch.equal(before[name],value) for name,value in model.named_parameters())
    original_exposure['async-'+family]=2
    release.set()
    policy.drain_async()
    saved=torch.load(policy.slot,map_location='cpu',weights_only=False)
    assert saved['step']==1
    assert saved['training']['exposure_counts']=={'async-'+family:1}
    assert saved['training']['options']['async_checkpoint_admission']['mode']=='background'
    assert saved['training']['options']['nested']['values']==[17]
    _assert_tree_equal(saved['training']['rng'],expected['training']['rng'])
    for name,value in expected['model'].items():
        assert torch.equal(saved['model'][name],value),name
    assert (saved['alias_probe'].untyped_storage().data_ptr()
            == saved['alias_probe_view'].untyped_storage().data_ptr())
    _assert_tree_equal(saved['training']['optimizer'],expected['training']['optimizer'])
    assert event['async_pending'] is False and event['committed'] is True


def test_insufficient_memory_falls_back_to_synchronous_full_save(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1)
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    state={'step':4,'model':{'weight':torch.ones(8)},'training':{'optimizer':{},'rng':{},'exposure_counts':{'a':4}}}
    event=policy.save(state,async_allowed=True,async_optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))]))
    assert event['async_fallback']['eligible'] is False
    assert not policy.async_pending and event['committed']
    saved=torch.load(policy.slot,weights_only=False)
    assert saved['training']['exposure_counts']=={'a':4}
    assert saved['training']['options']['async_checkpoint_admission']['mode']=='synchronous-fallback'


def test_async_publication_hook_runs_after_durable_exact_snapshot(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1<<50)
    observed=[]
    def publish(path,state,event):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['step']==state['step']
        observed.append((saved['step'],event['path']))
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True,on_saved=publish)
    optimizer=torch.optim.AdamW([torch.nn.Parameter(torch.ones(2))])
    event=policy.save({'step':8,'training':{'optimizer':optimizer.state_dict(),'rng':{},
        'exposure_counts':{'a':8},'options':{}}},async_allowed=True,async_optimizer=optimizer)
    policy.drain_async()
    assert observed==[(8,str(policy.slot))]
    assert event['committed'] is True


def test_async_admission_preserves_previous_slot_when_atomic_disk_headroom_is_short(tmp_path,monkeypatch):
    from collections import namedtuple
    from natlang_neuralese.train import trajectory_state
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1<<50)
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    previous={'step':1,'model':{'weight':torch.ones(4)},'training':{'optimizer':{},'rng':{},'exposure_counts':{}}}
    policy.save(previous)
    Usage=namedtuple('Usage','total used free')
    monkeypatch.setattr('natlang_neuralese.train.checkpoint_policy.shutil.disk_usage',
        lambda _path:Usage(10,10,1))
    parameter=torch.nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([parameter])
    with pytest.raises(OSError) as refused:
        policy.save({'step':2,'model':{'weight':torch.zeros(4)},'training':{
            'optimizer':optimizer.state_dict(),'rng':{},'exposure_counts':{}}},
            async_allowed=True,async_optimizer=optimizer)
    assert refused.value.errno==errno.ENOSPC
    assert torch.load(policy.slot,weights_only=False)['step']==1
    assert not policy.async_pending


def test_recipe_declares_async_checkpoint_as_operational_native_sft_control():
    from natlang_neuralese.train.recipe import HANDLERS,OPERATIONAL_STAGE_PARAMETERS
    assert 'async_checkpoint' in HANDLERS['native_state_sft']['parameters']
    assert 'async_checkpoint' in OPERATIONAL_STAGE_PARAMETERS


@pytest.mark.parametrize('explicit_refresh_updates', [None, 3])
def test_native_trainer_drains_on_update_and_schedule_close_errors_without_saving_partial_update(tmp_path,monkeypatch,explicit_refresh_updates):
    import json
    from test_native_serve import _tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train import trajectory_state

    base=tmp_path/'base'
    _tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    paths={}
    for split in ('train','test'):
        row={'schema':'natlang.neuralese-native-graph/1','id':split,'split':split,
            'source_groups':[split],'tokenizer_fingerprint':config['tokenizer_fingerprint'],
            'nodes':[{'name':'answer','token_ids':[1,2,3], 'target_weights':[0,1,1]}]}
        paths[split]=tmp_path/f'{split}.jsonl'
        paths[split].write_text(json.dumps(row)+'\n')
    entered,release=threading.Event(),threading.Event()
    writes=[]
    real_write=trajectory_state.atomic_checkpoint
    def held_write(path,state):
        writes.append(state['step'])
        entered.set()
        if not release.wait(10):raise TimeoutError('trainer exit test did not release async writer')
        return real_write(path,state)
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',held_write)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1<<50)
    original_update=trainer.native_training_update
    update_calls=[]
    def fail_inside_second_update(*args,**kwargs):
        update_calls.append(True)
        if len(update_calls)==1:
            return original_update(*args,**kwargs)
        assert entered.wait(5)
        release.set()
        raise RuntimeError('deliberate partial optimizer-update failure')
    monkeypatch.setattr(trainer,'native_training_update',fail_inside_second_update)
    original_schedule=trainer.stream_pending_batches
    class CloseFailure:
        def __init__(self,*args,**kwargs):self.iterator=original_schedule(*args,**kwargs)
        def __iter__(self):return self
        def __next__(self):return next(self.iterator)
        def close(self):
            self.iterator.close()
            raise OSError('deliberate schedule close failure')
    monkeypatch.setattr(trainer,'stream_pending_batches',lambda *args,**kwargs:CloseFailure(*args,**kwargs))
    drains=[]
    original_drain=trainer.CheckpointPolicy.drain_async
    def observed_drain(policy):
        drains.append(True)
        return original_drain(policy)
    monkeypatch.setattr(trainer.CheckpointPolicy,'drain_async',observed_drain)
    argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
        '--epochs','3','--eval-every','1','--checkpoint-every','1','--task-eval-every','0',
        '--async-checkpoint','--out',str(tmp_path/'run')]
    if explicit_refresh_updates is not None:
        argv += ['--refresh-at-checkpoint', '--refresh-after-updates', str(explicit_refresh_updates)]
    with pytest.raises(RuntimeError,match='deliberate partial optimizer-update failure') as failure:
        trainer.main(argv)
    assert entered.is_set() and release.is_set() and drains
    assert writes==[1]  # no recovery serialization of possibly-partial second update
    assert any('schedule.close() also failed' in note for note in failure.value.__notes__)
    assert any('potentially partial optimizer state was not checkpointed' in note
               for note in failure.value.__notes__)
    saved=torch.load(tmp_path/'run'/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert saved['step']==saved['training']['global_step']==1
    assert saved['training']['exposure_counts']=={'train':1}


def test_native_trainer_generic_safe_exception_still_saves_completed_update(tmp_path,monkeypatch):
    import json
    from test_native_serve import _tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer

    base=tmp_path/'base'
    _tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    paths={}
    for split in ('train','test'):
        row={'schema':'natlang.neuralese-native-graph/1','id':split,'split':split,
            'source_groups':[split],'tokenizer_fingerprint':config['tokenizer_fingerprint'],
            'nodes':[{'name':'answer','token_ids':[1,2,3], 'target_weights':[0,1,1]}]}
        paths[split]=tmp_path/f'{split}.jsonl'
        paths[split].write_text(json.dumps(row)+'\n')

    original_write=trainer.write_json
    failure=OSError('unexpected progress telemetry failure')
    writes=[]
    def fail_progress(path,value):
        if path.name=='progress.json':
            writes.append(path)
            raise failure
        return original_write(path,value)
    monkeypatch.setattr(trainer,'write_json',fail_progress)
    out=tmp_path/'run'
    argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
        '--epochs','3','--eval-every','1','--checkpoint-every','3','--task-eval-every','0',
        '--async-checkpoint','--out',str(out)]
    with pytest.raises(OSError,match='unexpected progress telemetry failure') as caught:
        trainer.main(argv)
    assert caught.value is failure and len(writes)==1
    saved=torch.load(out/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert saved['step']==saved['training']['global_step']==1
    assert saved['training']['exposure_counts']=={'train':1}


def test_failed_async_write_recovers_latest_state_before_returning(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    real_write=trajectory_state.atomic_checkpoint
    calls=[]
    def fail_first(path,state):
        calls.append(state['step'])
        if len(calls)==1:raise OSError('simulated checkpoint device failure')
        return real_write(path,state)
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',fail_first)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1<<50)
    parameter=torch.nn.Parameter(torch.ones(2))
    optimizer=torch.optim.AdamW([parameter])
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    first={'step':1,'model':{'p':parameter.detach().clone()},'training':{'optimizer':optimizer.state_dict(),
        'rng':{'torch':torch.get_rng_state()},'exposure_counts':{'a':1},'options':{}}}
    policy.save(first,async_allowed=True,async_optimizer=optimizer)
    assert parameter.device.type=='cpu'
    current=copy.deepcopy(first)
    current['step']=2;current['training']['exposure_counts']['a']=2
    event=policy.save(current)  # detects writer error, then synchronously commits current state
    assert event['async_writer_recovery']['recovered_synchronously'] is True
    saved=torch.load(policy.slot,weights_only=False)
    assert saved['step']==2 and saved['training']['exposure_counts']=={'a':2}
    assert current['training']['options']['async_checkpoint_recovery_history']


def test_forced_sync_save_and_finalize_drain_writer_before_consumption(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    entered,release=threading.Event(),threading.Event()
    real_write=trajectory_state.atomic_checkpoint
    def held(path,state):
        entered.set()
        if not release.wait(10):raise TimeoutError('checkpoint test did not release writer')
        return real_write(path,state)
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',held)
    monkeypatch.setattr(trajectory_state,'available_system_memory_bytes',lambda:1<<50)
    parameter=torch.nn.Parameter(torch.ones(3))
    optimizer=torch.optim.AdamW([parameter])
    policy=CheckpointPolicy(tmp_path,async_checkpoint=True)
    policy.save({'step':1,'training':{'optimizer':optimizer.state_dict(),'rng':{},'exposure_counts':{}}},
        async_allowed=True,async_optimizer=optimizer)
    assert entered.wait(5)
    released=[]
    def delayed_release():
        time.sleep(.1)
        release.set()
        released.append(True)
    threading.Thread(target=delayed_release,daemon=True).start()
    event=policy.save({'step':2,'training':{'optimizer':optimizer.state_dict(),'rng':{},'exposure_counts':{}}})
    assert event['committed'] and released
    policy.finalize({'weights':torch.ones(1)})
    assert torch.load(policy.slot,weights_only=False)['step']==2
