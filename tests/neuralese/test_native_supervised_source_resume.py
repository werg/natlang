"""Already adopted partial checkpoints retain exact source and visit identities."""
import copy,json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.rl import native_supervised as supervised
from natlang_neuralese.train.native_resident import binding
from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA


def fixture(tmp_path,architecture='lfm2'):
    directory=tmp_path/'learner';directory.mkdir()
    targets=[]
    for name in ('train','held','stale-train','stale-held'):
        path=tmp_path/(name+'.jsonl');path.write_text(name+'\n');targets.append(path)
    training={'schema':TRAINING_SCHEMA,'optimizer':{'actual':'moments'},'rng':{'actual':'rng'},
        'options':{'stage_id':'full','source_predecessors':{'continued_native_graphs':'real predecessor'}},
        'identity':{'objective':None},'source_receipts':[binding(p) for p in targets[:2]],
        'exposure_counts':{'row':17}}
    state={'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':architecture},'training':training}
    torch.save(state,directory/'checkpoint.pt')
    config={'native_actor_mode':'raw-state','production_stage':'full','trainer_parameters':{'stage_id':'full'},
        'seed_native_graphs':str(targets[0]),'held_native_graphs':str(targets[1]),
        'seed_command':['python','--continued-native-graphs','{continued_native_graphs}',
                        '--continued-held-native-graphs={continued_held_native_graphs}','{seed_trainer_args}']}
    values={'trainer_args':['--learning-rate','0.1','--continued-native-graphs',str(targets[2])],
            'seed_trainer_args':['--continued-held-native-graphs',str(targets[3]),'--epochs','3']}
    return config,values,directory,state,targets


@pytest.mark.parametrize('architecture',['lfm2','mellum'])
def test_exact_current_sources_omit_only_obsolete_flags_without_touching_config_or_checkpoint(tmp_path,architecture):
    config,values,directory,state,_=fixture(tmp_path,architecture)
    before=copy.deepcopy(config);raw=(directory/'checkpoint.pt').read_bytes()
    command,reuse=supervised._resume_source_arguments(config,directory,values)
    assert command==['python','{seed_trainer_args}']
    assert values=={'trainer_args':['--learning-rate','0.1'],'seed_trainer_args':['--epochs','3']}
    assert reuse['checkpoint']==binding(directory/'checkpoint.pt') and reuse['stage_id']=='full'
    assert set(reuse['already_current_sources'])=={'continued_native_graphs','continued_held_native_graphs'}
    assert config==before and (directory/'checkpoint.pt').read_bytes()==raw
    assert torch.load(directory/'checkpoint.pt',weights_only=False)['training']==state['training']


def test_resident_effective_source_wins_over_old_declared_source_only_when_exact(tmp_path,monkeypatch):
    config,values,directory,_,targets=fixture(tmp_path)
    config.update(seed_native_graphs=str(targets[2]),resident_data_directory=str(tmp_path/'resident'))
    from natlang_neuralese.train import native_resident
    monkeypatch.setattr(native_resident,'current',lambda root:{'payload':{
        'stage_id':'full','train':binding(targets[0]),'held':binding(targets[1])}})
    command,reuse=supervised._resume_source_arguments(config,directory,values)
    assert '--continued-native-graphs' not in command
    assert reuse['already_current_sources']['continued_native_graphs']==binding(targets[0])


@pytest.mark.parametrize('mutation',['new-source','other-stage','other-objective','missing-partial','nonraw'])
def test_genuinely_new_sources_and_other_stage_objectives_keep_strict_predecessors(tmp_path,mutation):
    config,values,directory,state,targets=fixture(tmp_path)
    if mutation=='new-source':config['seed_native_graphs']=str(targets[2])
    elif mutation=='other-stage':config['trainer_parameters']['stage_id']='other'
    elif mutation=='other-objective':
        state['training']['identity']['objective']={'sha256':'objective'};torch.save(state,directory/'checkpoint.pt')
    elif mutation=='missing-partial':(directory/'checkpoint.pt').unlink()
    else:config['native_actor_mode']='semantic'
    command,reuse=supervised._resume_source_arguments(config,directory,values)
    assert '--continued-native-graphs' in command
    if mutation=='new-source':assert set(reuse['already_current_sources'])=={'continued_held_native_graphs'}
    else:assert reuse is None


def test_partial_checkpoint_change_during_source_verification_rejects(tmp_path,monkeypatch):
    config,values,directory,_,_=fixture(tmp_path)
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    original=CheckpointPolicy.load
    def change(path):
        state=original(path)
        with Path(path).open('ab') as stream:stream.write(b'changed')
        return state
    monkeypatch.setattr(CheckpointPolicy,'load',change)
    with pytest.raises(ValueError,match='checkpoint changed'):supervised._resume_source_arguments(config,directory,values)


def test_actual_partial_trainer_retry_updates_once_and_retains_exact_source_credits(tmp_path,monkeypatch):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.data.native_graph import SCHEMA
    from natlang_neuralese.train.native_state_sft import main,continuation_predecessors
    from natlang_neuralese.rl import production
    base=tmp_path/'base';tiny_source(base)
    _,_,_,model_config=load_native_model(base,dtype=torch.float32,rank=4)
    paths={}
    for split in ('train','test'):
        path=tmp_path/(split+'.jsonl');paths[split]=path
        path.write_text(json.dumps({'schema':SCHEMA,'id':split,'split':split,'source_groups':[split],
            'tokenizer_fingerprint':model_config['tokenizer_fingerprint'],
            'nodes':[{'name':'answer','token_ids':[1,2,3],'target_weights':[0,1,1]}]})+'\n')
    out=tmp_path/'work';directory=out/'learner'
    parameters={'base':str(base),'dtype':'float32','rank':4,'device':'cpu','epochs':3,
        'checkpoint_every':1,'eval_every':1,'refresh_at_checkpoint':True,'stage_id':'full',
        'forward_atol':.001,'gradient_atol':.01}
    from natlang_neuralese.train.recipe import stage_parameter_args
    argv=['--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),
          '--out',str(directory),*stage_parameter_args(parameters)]
    assert main(argv)==0
    before=torch.load(directory/'checkpoint.pt',weights_only=False)
    assert before['training']['step']==1 and before['training']['exposure_counts']=={'train':1}
    stale=tmp_path/'older.jsonl';stale.write_text('obsolete historical predecessor\n')
    with pytest.raises(ValueError,match='predecessor differs'):
        continuation_predecessors(before['training'],{'continued_native_graphs':stale},stage_id='full')
    config={'out':str(out),'native_actor_mode':'raw-state','production_stage':'full','trainer_parameters':parameters,
        'seed_native_graphs':str(paths['train']),'held_native_graphs':str(paths['test']),
        'continued_native_graphs':str(stale),'seed_command':['python','-m','natlang_neuralese.train.native_state_sft',
            '--native-graphs','{seed_native_graphs}','--held-native-graphs','{held_native_graphs}',
            '--out','{learner_out}','--continued-native-graphs','{continued_native_graphs}','{seed_trainer_args}']}
    calls=[]
    def run(command,values):
        assert '--continued-native-graphs' not in command
        calls.append(command)
        assert main(['--native-graphs',values['seed_native_graphs'],'--held-native-graphs',values['held_native_graphs'],
            '--out',values['learner_out'],*values['seed_trainer_args']])==0
    monkeypatch.setattr(production,'run_command',run)
    state={'checkpoint':str(directory/'checkpoint.pt'),'round':0};state_path=out/'state.json'
    result=supervised.execute(config,state,state_path,None)
    after=torch.load(directory/'checkpoint.pt',weights_only=False)
    assert after['training']['step']==2 and after['training']['exposure_counts']=={'train':2}
    assert after['training']['source_receipts']==before['training']['source_receipts']
    assert after['training']['optimizer'] and after['training']['rng']
    assert result['phase']=='awaiting_semantic_refresh' and result['resumed_current_source_inputs']['checkpoint']['sha256']!=binding(directory/'checkpoint.pt')['sha256']
    assert supervised.execute(config,result,state_path,None)==result and len(calls)==1
