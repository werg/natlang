"""Production scheduling plumbing; receipt fixtures do not claim task quality."""
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

from natlang_neuralese.rl import native_supervised as supervised
from natlang_neuralese.common.hashing import sha256_file_hex


def test_mandatory_worker_uses_existing_seed_command_and_retains_partial_pending_update(tmp_path,monkeypatch):
    from natlang_neuralese.rl import production
    parent=tmp_path/'parent';parent.write_bytes(b'actual exact parent binding')
    config={'native_state':True,'mandatory_sft_only':True,'out':str(tmp_path/'job'),
        'checkpoint':str(parent),'seed_command':['same-shared-trainer','{trainer_args}'],
        'trainer_parameters':{'refresh_at_checkpoint':True,'checkpoint_every':7}}
    config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
    calls=[]
    def trainer(command,values):
        calls.append((command,values))
        assert '--refresh-at-checkpoint' in values['trainer_args']
        out=Path(values['learner_out']);out.mkdir(parents=True)
        (out/'checkpoint.pt').write_bytes(b'actual updated weights fixture')
        (out/'report.json').write_text(json.dumps({'completed':False,'numerics_ok':True,
            'continuation_boundary':{'kind':'semantic-refresh','reached':True,'new_optimizer_updates':7},
            'mandatory_sft':{'complete':False,'remaining_graph_visits':10000}}))
    monkeypatch.setattr(production,'run_command',trainer)
    result=production.run(config_path,rounds=1,defer_native_refresh=True)
    assert len(calls)==1 and result['phase']=='awaiting_semantic_refresh'
    assert result['checkpoint']==str(parent) and result['pending_update_kind']=='mandatory_sft'
    assert result['mandatory_sft']['remaining_graph_visits']==10000
    # Restart before refresh never repeats completed optimizer work.
    again=production.run(config_path,rounds=1,defer_native_refresh=True)
    assert again==result and len(calls)==1


def test_pending_source_comparison_checks_real_source_epochs_and_completed_credits():
    sources={'native_graphs':{'sha256':'train'},'held_native_graphs':{'sha256':'held'}}
    frame={'complete':True,'train_source_sha256':'train','held_source_sha256':'held','epochs':1}
    adapter={'models':{'student':{'mandatory_sft':{'stages':{'full':frame}}}}}
    assert not supervised.stage_pending(adapter,'student','full',sources,1)
    assert supervised.stage_pending(adapter,'student','full',sources,2)
    assert supervised.stage_pending(adapter,'student','full',{**sources,'native_graphs':{'sha256':'append'}},1)
    frame['complete']=False
    assert supervised.stage_pending(adapter,'student','full',sources,1)


def test_interleave_shares_parent_merges_both_partial_updates_and_does_not_repeat_cycle(tmp_path,monkeypatch):
    import threading
    from natlang_neuralese.rl import native_rounds
    parent=tmp_path/'parent.pt';parent.write_bytes(b'joint parent')
    adapter=tmp_path/'adapter.json'
    adapter.write_text(json.dumps({'checkpoint':{'path':str(parent),'sha256':sha256_file_hex(parent)},
        'models':{name:{} for name in ('a','b')}}))
    train=tmp_path/'train.jsonl';train.write_text('{}\n')
    held=tmp_path/'held.jsonl';held.write_text('{}\n')
    recipe=tmp_path/'recipe.json'
    recipe.write_text(json.dumps({'schema':'natlang.neuralese-training-recipe/1','id':'mechanism-fixture',
        'input_bindings':{'train':{'path':str(train),'sha256':sha256_file_hex(train)},
                          'held':{'path':str(held),'sha256':sha256_file_hex(held)}},
        'stages':[{'id':'full','kind':'native_state_sft','requires':[],
            'inputs':{'native_graphs':'train','held_native_graphs':'held'},
            'parameters':{'epochs':1,'checkpoint_every':7,'eval_every':7,'communication_foundation':True}}]}))
    configs=[{'native_state':True,'native_participant':name,'checkpoint':str(parent),
        'semantic_adapter':str(adapter),'production_recipe':str(recipe),'production_stage':'full',
        'semantic_refresh_config':str(tmp_path/'semantic.json'),'out':str(tmp_path/name),
        'seed_command':['existing-seed']} for name in ('a','b')]
    barrier=threading.Barrier(2);calls=[]
    def job(path,config,directory,stop):
        assert config['mandatory_sft_only'] and config['checkpoint']==str(parent)
        assert config['seed_native_graphs']==str(train)
        assert config['trainer_parameters']['refresh_at_checkpoint']
        assert 'communication_foundation' not in config['trainer_parameters']
        barrier.wait(timeout=10)
        child=Path(config['out'])/'child.pt';child.write_bytes(config['native_participant'].encode())
        calls.append(config['native_participant'])
        return {'pending_native_checkpoint':str(child)}
    monkeypatch.setattr(native_rounds,'round_job',job)
    refreshed=tmp_path/'joint.pt';refreshed.write_bytes(b'updated joint parent')
    next_adapter=tmp_path/'next-adapter.json'
    next_adapter.write_text(json.dumps({'checkpoint':{'path':str(refreshed),'sha256':sha256_file_hex(refreshed)},
        'models':{name:{'mandatory_sft':{'complete':False}} for name in ('a','b')}}))
    refreshes=[]
    def refresh(configs,source,updates,*args):
        assert source==str(adapter) and set(updates)=={'a','b'}
        refreshes.append(updates)
        return {'checkpoint':{'path':str(refreshed),'sha256':sha256_file_hex(refreshed)},
            'semantic_adapter':{'path':str(next_adapter),'sha256':sha256_file_hex(next_adapter)},
            'updated_participants':{name:{'sha256':sha256_file_hex(path)} for name,path in updates.items()}}
    monkeypatch.setattr(native_rounds,'refresh_job',refresh)
    latest=tmp_path/'latest.json';stop=SimpleNamespace(requested=False)
    result=supervised.interleave([],configs,str(adapter),tmp_path/'work',cycle=0,latest=latest,device='cpu',stop=stop)
    assert result==str(next_adapter) and sorted(calls)==['a','b'] and len(refreshes)==1
    assert json.loads(latest.read_text())['supervised_cycle']==0
    for config in configs:
        state=json.loads((Path(config['out'])/'state.json').read_text())
        assert state['checkpoint']==str(refreshed) and state['semantic_adapter']==str(next_adapter)
    assert supervised.interleave([],configs,result,tmp_path/'work',cycle=0,latest=latest,device='cpu',stop=stop)==result
    assert len(calls)==2  # resume does not redo either completed quantum


def test_pending_metadata_admission_is_not_claimed_as_optimizer_work_and_is_resumable(tmp_path,monkeypatch):
    from natlang_neuralese.rl import production
    parent=tmp_path/'parent.pt';parent.write_bytes(b'actual parent numeric fixture')
    declared=tmp_path/'pending.json';declared.write_text('{}')
    out=tmp_path/'run';config={'schema':'natlang.native-rl-production/1','out':str(out),
        'mandatory_sft_only':True,'checkpoint':str(parent),'seed_command':['shared-trainer'],
        'pending_native_sources':str(declared),'trainer_parameters':{},'native_state':True}
    path=tmp_path/'config.json';path.write_text(json.dumps(config));calls=[]
    def trainer(command,values):
        calls.append(True);directory=Path(values['learner_out']);directory.mkdir(parents=True)
        child=directory/'checkpoint.pt';child.write_bytes(b'same numeric state plus mandatory source metadata')
        digest=sha256_file_hex(declared)
        report={'completed':False,'numerics_ok':True,'continuation_boundary':{'reached':False,'new_optimizer_updates':0},
            'mandatory_sft':{'complete':False,'stages':{'full':{'pending_source_declarations':{digest:{}}}}},
            'pending_source_admission':{'schema':'natlang.native-pending-source-admission/1',
                'parent_checkpoint_sha256':sha256_file_hex(parent),'checkpoint_sha256':sha256_file_hex(child),
                'new_optimizer_updates':0,'declarations':[digest]}}
        (directory/'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(production,'run_command',trainer)
    state=production.run(path,rounds=1,defer_native_refresh=True)
    assert state['phase']=='awaiting_semantic_refresh' and not state['mandatory_sft']['complete']
    assert state['last_learner_progress']['pending_source_admission']['new_optimizer_updates']==0
    assert production.run(path,rounds=1,defer_native_refresh=True)==state and len(calls)==1
