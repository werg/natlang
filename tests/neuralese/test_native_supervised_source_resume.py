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


@pytest.mark.parametrize('architecture',['lfm2'])
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


@pytest.mark.parametrize('entry',['mandatory','rl-seed'])
def test_actual_partial_trainer_retry_updates_once_and_retains_exact_source_credits(tmp_path,monkeypatch,entry):
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
    out=tmp_path/'work'
    directory=(out if entry=='mandatory' else out/'round-000000')/'learner'
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
        if command==['publish']:return
        assert '--continued-native-graphs' not in command
        calls.append(command)
        assert main(['--native-graphs',values['seed_native_graphs'],'--held-native-graphs',values['held_native_graphs'],
            '--out',values['learner_out'],*values['seed_trainer_args']])==0
    monkeypatch.setattr(production,'run_command',run)
    state={'checkpoint':str(directory/'checkpoint.pt'),'round':0};state_path=out/'state.json'
    if entry=='mandatory':
        result=supervised.execute(config,state,state_path,None)
    else:
        # A real partial checkpoint with no qualified publication boundary must
        # resume through production's zero-success fallback, not mandatory SFT.
        report_path=directory/'report.json';report=json.loads(report_path.read_text())
        report['continuation_boundary']={};report_path.write_text(json.dumps(report))
        cases=tmp_path/'cases.jsonl';cases.write_text(json.dumps({'id':'task','split':'train'})+'\n')
        config.update(native_state=True,checkpoint=str(directory/'checkpoint.pt'),base=str(base),
            cases=str(cases),endpoints=['unused'],after_update=['publish'],
            seed_trainer_parameters=parameters,trainer_parameters={'stage_id':'policy-stage'})
        config_path=tmp_path/'config.json';config_path.write_text(json.dumps(config))
        from natlang_neuralese.rl import native_raw_policy
        monkeypatch.setattr(native_raw_policy,'prepare_raw_policy',lambda *a,**kw:{'tokenizer':'bound-parent'})
        def collect(**kwargs):
            (kwargs['out']/'rollouts.jsonl').write_text(json.dumps({'task_id':'task','split':'train',
                'reward':{'quality':0},'turns':[]})+'\n')
            return {'scored':1,'groups_with_spread':0,'successes':0}
        monkeypatch.setattr(production,'collect',collect)
        result=production.run(config_path,rounds=1)
    after=torch.load(directory/'checkpoint.pt',weights_only=False)
    assert after['training']['step']==2 and after['training']['exposure_counts']=={'train':2}
    assert after['training']['source_receipts']==before['training']['source_receipts']
    assert after['training']['optimizer'] and after['training']['rng']
    assert result['resumed_current_source_inputs']['checkpoint']['sha256']!=binding(directory/'checkpoint.pt')['sha256']
    assert len(calls)==1
    if entry=='mandatory':
        assert result['phase']=='awaiting_semantic_refresh'
        assert supervised.execute(config,result,state_path,None)==result and len(calls)==1
    else:
        assert result['round']==1 and result['trained_task_ids']==[]
        audit=result['last_completed_update']
        assert audit['mode']=='seed'
        assert audit['operation_receipt']['scope']=='seed-supervised-exposure'
        assert not result['production_complete']
        receipt=json.loads((directory/'native-training-receipt.json').read_text())
        report=json.loads((directory/'report.json').read_text())
        assert receipt['completed'] is False and report['numerics_ok'] is False
        assert report['progress_numerics']['passed'] is True


def test_seed_refresh_rejects_tampered_deferred_runtime_certificate(tmp_path):
    from natlang_neuralese.rl.production import qualified_seed_boundary
    checkpoint=tmp_path/'checkpoint.pt';checkpoint.write_bytes(b'actual checkpoint')
    sha=binding(checkpoint)['sha256']
    certificate=tmp_path/'runtime.json'
    certificate.write_text(json.dumps({'passed':False,'deferred':True,
        'reason':'semantic refresh; qualification deferred until resume','checkpoint_sha256':sha}))
    report={'schema':'natlang.neuralese-native-sft-report/1','numerics_ok':False,
        'runtime_certificate':{**binding(certificate),'passed':False},
        'progress_numerics':{'schema':'natlang.native-finite-training-state/1','passed':True,
            'checkpoint_sha256':sha,'runtime_qualification_granted':False,'nonfinite':[],
            'floating_tensors':{'model':1,'optimizer':1}},
        'continuation_boundary':{'kind':'semantic-refresh','reached':True,
            'new_optimizer_updates':1,'checkpoint_sha256':sha}}
    assert qualified_seed_boundary({'native_state':True},'seed',report,checkpoint)
    certificate.write_text('{}')
    with pytest.raises(ValueError):
        qualified_seed_boundary({'native_state':True},'seed',report,checkpoint)

# Written for one optimizer update per packed batch; accumulation is tested in test_native_update_accumulation.py.
pytestmark = pytest.mark.usefixtures('per_batch_native_updates')


@pytest.mark.parametrize('new_source',[False,True])
def test_rl_resume_uses_exact_carried_sft_receipts_without_credit_or_parent_changes(tmp_path,new_source):
    config,values,directory,state,targets=fixture(tmp_path)
    carried=copy.deepcopy(state['training'])
    state['training'].update(options={'stage_id':'outcome-learning'},identity={'objective':{'kind':'native-policy'}},
        source_receipts=[binding(p) for p in targets[2:]],sft_progress={'full':carried})
    torch.save(state,directory/'checkpoint.pt');raw=(directory/'checkpoint.pt').read_bytes()
    if new_source:config['seed_native_graphs']=str(targets[2])
    command,reuse=supervised._resume_source_arguments(config,directory,values)
    assert reuse['source_frame']=='carried-sft-progress'
    assert set(reuse['already_current_sources'])==({'continued_held_native_graphs'} if new_source else
        {'continued_native_graphs','continued_held_native_graphs'})
    assert ('--continued-native-graphs' in command) is new_source
    assert (directory/'checkpoint.pt').read_bytes()==raw
    assert carried['exposure_counts']=={'row':17}


@pytest.mark.parametrize('committed',[False,True])
def test_current_correction_and_predecessor_retire_together_only_after_exact_replay(tmp_path,committed):
    config,values,directory,state,targets=fixture(tmp_path)
    ledger=tmp_path/'rows.jsonl';ledger.write_text('')
    proof=tmp_path/'correction.json';proof.write_text(json.dumps({'schema':'natlang.native-source-correction/1',
        'model_id':'lfm2:'+('a'*64),'stage_id':'full','prior_source':binding(targets[2]),
        'current_source':binding(targets[0]),'row_bindings_path':ledger.name,'row_bindings_sha256':binding(ledger)['sha256']}))
    ref=binding(proof);frame=state['training'];frame.update(source_row_bindings={'row':'binding'},source_row_order=['row'])
    if committed:
        frame['options'].update(source_correction_binding=ref,source_correction_history=[{
            'proof_sha256':ref['sha256'],'prior_source':binding(targets[2]),'current_source':binding(targets[0])}])
    carried=copy.deepcopy(frame);state['training'].update(options={'stage_id':'rl'},identity={'objective':{'kind':'native-policy'}},sft_progress={'full':carried})
    state['model_config']['base_inventory_sha256']='a'*64
    torch.save(state,directory/'checkpoint.pt');raw=(directory/'checkpoint.pt').read_bytes()
    config['source_correction']=str(proof);config['seed_command']+=['--source-correction','{source_correction}']
    command,reuse=supervised._resume_source_arguments(config,directory,values)
    assert ('--source-correction' in command) is not committed
    assert ('--continued-native-graphs' in command) is not committed
    if committed:assert reuse['retired_committed_source_correction']==ref
    assert (directory/'checkpoint.pt').read_bytes()==raw


@pytest.mark.parametrize('case',['append','wrong-predecessor','no-history'])
def test_append_retires_only_replayed_parent_correction_and_keeps_append_predecessor(tmp_path,case):
    config,values,directory,state,targets=fixture(tmp_path)
    ledger=tmp_path/'rows.jsonl';ledger.write_text('')
    proof=tmp_path/'correction.json';proof.write_text(json.dumps({'schema':'natlang.native-source-correction/1',
        'model_id':'lfm2:'+('a'*64),'stage_id':'full','prior_source':binding(targets[3]),
        'current_source':binding(targets[0]),'row_bindings_path':ledger.name,'row_bindings_sha256':binding(ledger)['sha256']}))
    ref=binding(proof);frame=state['training'];frame.update(source_row_bindings={'row':'binding'},source_row_order=['row'])
    if case!='no-history':
        frame['options'].update(source_correction_binding=ref,source_correction_history=[{
            'proof_sha256':ref['sha256'],'prior_source':binding(targets[3]),'current_source':binding(targets[0])}])
    carried=copy.deepcopy(frame)
    state['training'].update(options={'stage_id':'rl'},identity={'objective':{'kind':'native-policy'}},sft_progress={'full':carried})
    state['model_config']['base_inventory_sha256']='a'*64
    parent=tmp_path/'selected-parent.pt';torch.save(state,parent)
    # The output slot contains a different, uncorrected frame: only the
    # selected parent can prove this handoff.
    config.update(seed_native_graphs=str(targets[2]),source_correction=str(proof),
        continued_native_graphs=str(targets[3] if case=='wrong-predecessor' else targets[0]),
        seed_command=['python','--continued-native-graphs={continued_native_graphs}',
                      '--source-correction','{source_correction}'])
    values.update(checkpoint=str(parent),trainer_args=[],seed_trainer_args=[])
    command,reuse=supervised._resume_source_arguments(config,directory,values)
    assert '--continued-native-graphs={continued_native_graphs}' in command
    if case=='append':
        assert '--source-correction' not in command
        assert reuse['checkpoint']==binding(parent) and reuse['source_frame']=='carried-sft-progress'
        assert reuse['already_current_sources']=={} and reuse['retired_committed_source_correction']==ref
    else:
        assert '--source-correction' in command and reuse is None


def test_rolling_resume_seals_carried_stage_obligations_before_slot_replacement(tmp_path,monkeypatch):
    import shutil
    config,_,directory,state,targets=fixture(tmp_path)
    frame=copy.deepcopy(state['training'])
    frame.update(source_row_bindings={'row':'old-binding'},source_row_order=['row'],
        required_visits={'row':3,'pending':2},exposure_counts={'row':1,'pending':0},
        pending_source_rows={'pending':{'binding':'pending-binding','required_visits':2}},
        pending_source_declarations={'source':{'sha256':'a'*64}})
    state['training'].update(options={'stage_id':'rl'},identity={'objective':{'kind':'native-policy'}},global_step=72,
                             sft_progress={'full':frame})
    slot=directory/'checkpoint.pt';torch.save(state,slot)
    previous=targets[2];slot_ref=binding(slot);previous_ref=binding(previous)
    config['rolling_sft_resume']={'schema':'natlang.rolling-sft-resume/1',
        'checkpoint_path':slot_ref['path'],'checkpoint_sha256':slot_ref['sha256'],
        'previous_checkpoint_path':previous_ref['path'],'previous_checkpoint_sha256':previous_ref['sha256']}
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    selected,record=supervised.rolling_resume_parent(config,{'checkpoint':str(previous)},directory)
    assert selected==str(slot.resolve())
    proof=json.loads(Path(record['custody_snapshot']['path']).read_text())
    assert proof['checkpoint']['sha256']==slot_ref['sha256']
    assert proof['sft_frames']['full']['required_visits']=={'row':3,'pending':2}
    assert proof['sft_frames']['full']['exposure_counts']=={'row':1,'pending':0}
    assert proof['sft_frames']['full']['pending_source_rows']['pending']['binding']=='pending-binding'
    slot.write_text('newer checkpoint bytes')
    assert binding(record['custody_snapshot']['path'])==record['custody_snapshot']


def test_rolling_custody_snapshot_rejects_parent_changed_during_metadata_read(tmp_path,monkeypatch):
    import shutil
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    config,_,directory,state,targets=fixture(tmp_path);slot=directory/'checkpoint.pt'
    state['training'].update(source_row_bindings={'row':'binding'},source_row_order=['row'],global_step=72)
    torch.save(state,slot)
    slot_ref=binding(slot);previous=binding(targets[2])
    config['rolling_sft_resume']={'schema':'natlang.rolling-sft-resume/1',
        'checkpoint_path':slot_ref['path'],'checkpoint_sha256':slot_ref['sha256'],
        'previous_checkpoint_path':previous['path'],'previous_checkpoint_sha256':previous['sha256']}
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    load=CheckpointPolicy.load
    def raced(path):
        result=load(path)
        with Path(path).open('ab') as stream:stream.write(b'changed')
        return result
    monkeypatch.setattr(CheckpointPolicy,'load',raced)
    with pytest.raises(ValueError,match='changed during custody snapshot'):
        supervised.rolling_resume_parent(config,{'checkpoint':previous['path']},directory)


def test_same_earned_checkpoint_keeps_each_snapshotter_evidence(tmp_path,monkeypatch):
    import shutil
    from natlang_neuralese.train import native_resident
    _,_,directory,state,_=fixture(tmp_path)
    state['training'].update(source_row_bindings={'row':'binding'},source_row_order=['row'],global_step=72)
    slot=directory/'checkpoint.pt';torch.save(state,slot);sha=binding(slot)['sha256']
    monkeypatch.setattr(shutil,'disk_usage',lambda path:SimpleNamespace(free=100*1024**3))
    original=native_resident.binding;version=['a'*64]
    def versioned(path):
        value=original(path)
        return {**value,'sha256':version[0]} if Path(path)==Path(supervised.__file__) else value
    monkeypatch.setattr(native_resident,'binding',versioned)
    first=supervised._rolling_parent_custody_snapshot(slot,sha,directory)
    original_bytes=Path(first['path']).read_bytes()
    version[0]='b'*64
    second=supervised._rolling_parent_custody_snapshot(slot,sha,directory)
    assert first['path']!=second['path']
    assert Path(first['path']).read_bytes()==original_bytes
    left=json.loads(original_bytes);right=json.loads(Path(second['path']).read_text())
    assert left['sft_frames']==right['sft_frames']
    assert left['snapshotter']['sha256']=='a'*64 and right['snapshotter']['sha256']=='b'*64


def test_actual_resume_keeps_inactive_historical_visits_outside_active_quota(tmp_path):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.data.native_graph import SCHEMA
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    paths=[]
    for split in ('train','test'):
        path=tmp_path/(split+'.jsonl');path.write_text(json.dumps({'schema':SCHEMA,'id':split,
            'split':split,'source_groups':[split],'tokenizer_fingerprint':config['tokenizer_fingerprint'],
            'nodes':[{'name':'answer','token_ids':[4,5,6],'target_weights':[0,1,1]}]})+'\n');paths.append(path)
    args=['--base',str(base),'--device','cpu','--dtype','float32','--rank','4',
          '--native-graphs',str(paths[0]),'--held-native-graphs',str(paths[1]),'--epochs','2',
          '--eval-every','100','--checkpoint-every','100','--checkpoint-seconds','0','--stage-id','full',
          '--refresh-after-updates','1']
    first=tmp_path/'first';assert trainer.main([*args,'--out',str(first)])==0
    saved=torch.load(first/'checkpoint.pt',weights_only=False)
    # Explicit historical checkpoint metadata fixture, not observed task credit.
    saved['training']['exposure_counts']['previously-removed-row']=3
    saved['training']['sft_progress']['full']['exposure_counts']['previously-removed-row']=3
    parent=tmp_path/'historical-parent.pt';torch.save(saved,parent)
    second=tmp_path/'second';assert trainer.main([*args,'--continue-from',str(parent),'--out',str(second)])==0
    result=torch.load(second/'checkpoint.pt',weights_only=False)['training']
    assert result['exposure_counts']=={'train':2,'previously-removed-row':3}
    assert result['sft_progress']['full']['exposure_counts']==result['exposure_counts']
    assert result['global_step']==saved['training']['global_step']+1
    assert 'previously-removed-row' not in result['source_row_bindings']
