"""Mandatory supervised credits survive objective switches, not just metadata registration."""
import copy
import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train.native_continuation import (collect_sft_progress,restore_sft_progress,
    progress_obligations,communication_foundation,verify_initial_foundation_cohort)
from natlang_neuralese.data.native_sources import NativeGraphSource
from test_native_state_sft import graph, authenticated_pair, source_manifest


def progress():
    return {'schema':'natlang.neuralese-native-sft-training/1','step':3,'global_step':3,'epoch':0,'cursor':1,
        'identity':{'objective':None},'source_row_bindings':{'old':'binding0','new':'binding1'},
        'source_row_order':['old','new'],'exposure_counts':{'old':1,'new':0},
        'options':{'stage_id':'full','epochs':1},'optimizer':{'actual':'current moments'},
        'rng':{'actual':'current stream'},'fp8':{'actual':'current scale'}}


def test_borrowed_metadata_matches_owned_pending_merge_without_mutating_checkpoint():
    old=progress()
    old['source_row_bindings']={'old':'binding0'}
    old['source_row_order']=['old']
    old['options']['_pending_native_source_declarations']=[{
        'binding':{'path':'pending.json','sha256':'a'*64},
        'rows':{'new':{'binding':'binding1','required_visits':2}}}]
    prior=collect_sft_progress(old)
    current=progress()
    current['sft_progress']=prior
    current['exposure_counts']={'old':1}
    current['options']['_objective_required_visits']={'new':3}
    before=copy.deepcopy(current)
    borrowed=collect_sft_progress(current,borrow_frames=True)
    owned=collect_sft_progress(current)
    assert borrowed==owned
    assert borrowed['full']['required_visits']['new']==3
    assert borrowed['full']['exposure_counts']['new']==0
    assert current==before
    assert borrowed['full']['source_row_bindings'] is current['source_row_bindings']
    assert owned['full']['source_row_bindings'] is not current['source_row_bindings']
    # Historical objective frames are only read, never duplicated by discovery.
    policy={'sft_progress':prior,'identity':{'objective':'policy'}}
    assert collect_sft_progress(policy,borrow_frames=True)['full'] is prior['full']


@pytest.mark.parametrize('prefix_stage',[False,True])
def test_current_frame_replaces_old_snapshot_without_copying_discarded_payload(prefix_stage):
    class DiscardedSnapshot:
        def __deepcopy__(self, memo):
            raise AssertionError('superseded active snapshot must not be copied')

    old=progress()
    old['source_row_bindings']={'old':'binding0'}
    old['source_row_order']=['old']
    old['options']['_pending_native_source_declarations']=[{
        'binding':{'path':'pending.json','sha256':'a'*64},
        'rows':{'new':{'binding':'binding1','required_visits':2}}}]
    history=collect_sft_progress(old)
    history['full']['completed_foundations']=[{'cohort':'earned'}]
    other=copy.deepcopy(history['full'])
    history['other']=other
    if prefix_stage:history={'before':other,**history}
    current=progress()
    current['sft_progress']=history
    current['exposure_counts']={'old':1}
    expected=collect_sft_progress(current,borrow_frames=True)
    history['full']['obsolete_snapshot_payload']=DiscardedSnapshot()
    result=collect_sft_progress(current)
    assert result==expected
    assert list(result)==list(expected)
    if prefix_stage:assert result['before'] is result['other']
    assert result['full']['required_visits']=={'old':1,'new':2}
    assert result['full']['exposure_counts']=={'old':1,'new':0}
    result['full']['completed_foundations'][0]['cohort']='changed'
    result['other']['completed_foundations'][0]['cohort']='also changed'
    result['full']['exposure_counts']['old']=99
    assert history['full']['completed_foundations']==[{'cohort':'earned'}]
    assert history['other']['completed_foundations']==[{'cohort':'earned'}]
    assert current['exposure_counts']=={'old':1}


@pytest.mark.parametrize('epochs',[0.5,1,1.5,2])
@pytest.mark.parametrize('counts',[{}, {'old':1}, {'old':9,'new':2}])
def test_metadata_completion_matches_full_exposure_report(epochs,counts):
    from natlang_neuralese.train.native_continuation import sft_frame_complete
    current=progress();current['options']['epochs']=epochs;current['exposure_counts']=counts
    frame=collect_sft_progress(current)['full']
    assert sft_frame_complete(frame)==progress_obligations({'full':frame})['stages']['full']['complete']
    frame.pop('required_visits')
    assert sft_frame_complete(frame)==progress_obligations({'full':frame})['stages']['full']['complete']


def test_metadata_completion_still_checks_bad_counters_after_a_deficit():
    from natlang_neuralese.train.native_continuation import sft_frame_complete
    with pytest.raises(ValueError):
        sft_frame_complete({'required_visits':{'missing':1,'bad':1},'exposure_counts':{'bad':'invalid'}})


def test_metadata_completion_rejects_missing_historical_objective_like_full_report():
    from natlang_neuralese.train.native_continuation import sft_frame_complete
    frame={'required_visits':{},'identity':{'objective_kind':'native-offline-pairwise'}}
    with pytest.raises(KeyError,match='objective'):
        sft_frame_complete(frame)
    with pytest.raises(KeyError,match='objective'):
        progress_obligations({'complete-historical':frame})


def test_stage_frames_keep_deficits_and_current_optimizer_rng_after_objective_switch():
    old=progress();frames=collect_sft_progress(old)
    assert not {'optimizer','rng','fp8','sft_progress'} & frames['full'].keys()
    assert progress_obligations(frames)['remaining_graph_visits']==1
    current={'sft_progress':frames,'global_step':17,'optimizer':{'new':'joint moments'},'rng':{'new':'joint RNG'}}
    restored=restore_sft_progress(current,'full')
    assert restored['exposure_counts']==old['exposure_counts'] and restored['global_step']==17
    assert restored['optimizer'] is current['optimizer'] and restored['rng'] is current['rng']
    assert restored['step']==3
    # A separately declared CE stage cannot overwrite the previous obligation.
    next_stage=progress();next_stage['options']['stage_id']='next';next_stage['sft_progress']=frames
    assert set(collect_sft_progress(next_stage))=={'full','next'}


@pytest.mark.parametrize('stage',['full','not-yet-started'])
def test_restore_reuses_owned_history_without_recollecting_or_mutating_it(monkeypatch,stage):
    import natlang_neuralese.train.native_continuation as continuation
    current=progress()
    current.update(global_step=17,optimizer={'current':'moments'},rng={'current':'RNG'},fp8={'current':'scale'})
    frames=collect_sft_progress(current)
    before=copy.deepcopy(frames)
    expected=restore_sft_progress(current,stage)
    def unexpected(*args,**kwargs):
        raise AssertionError('already collected history was copied again')
    monkeypatch.setattr(continuation,'collect_sft_progress',unexpected)
    restored=restore_sft_progress(current,stage,collected_progress=frames)
    assert restored==expected and frames==before
    if restored is not None:
        assert restored['sft_progress']is frames
        for key in ('optimizer','rng','fp8'):assert restored[key]is current[key]
        restored['exposure_counts'].clear()
        assert frames==before and current['exposure_counts']==before['full']['exposure_counts']


def test_foundation_uses_entire_current_admitted_cohort_but_ordinary_append_does_not_delay_it(tmp_path):
    ordinary=[dict(graph('ordinary'),tokenizer_fingerprint='1'*64)]
    path=source_manifest(tmp_path,ordinary,[authenticated_pair('a'),authenticated_pair('b')])
    source=NativeGraphSource(path,'1'*64,required_split='train',role_ids=(58,59,60,61))
    incomplete=communication_foundation(source,{'lfm:a':1},epochs=1,trained_bindings=source.bindings)
    assert not incomplete['complete'] and incomplete['rows']['lfm:b']['graph_visits']==0
    full=communication_foundation(source,{'lfm:a':1,'lfm:b':1},epochs=1,trained_bindings=source.bindings)
    assert full['complete'] and full['graphs']==2
    old=dict(source.bindings)
    ordinary.append(dict(graph('extra'),tokenizer_fingerprint='1'*64))
    path=source_manifest(tmp_path,ordinary,[authenticated_pair('a'),authenticated_pair('b')])
    added=NativeGraphSource(path,'1'*64,required_split='train',role_ids=(58,59,60,61))
    same=communication_foundation(added,{'lfm:a':1,'lfm:b':1},epochs=1,trained_bindings=old)
    assert same['complete'] and same['cohort_sha256']==full['cohort_sha256']
    # Missing immutable bindings cannot gain credit merely by reusing row IDs.
    bad=communication_foundation(added,{'lfm:a':10,'lfm:b':10},epochs=1,trained_bindings={'lfm:a':'wrong'})
    assert not bad['complete'] and bad['graph_visits']==0


def test_initial_foundation_accepts_pair_only_complete_cohort_but_rejects_weaker_subset(tmp_path):
    from types import SimpleNamespace
    ordinary=[dict(graph('ordinary'),tokenizer_fingerprint='1'*64)]
    path=source_manifest(tmp_path,ordinary,[authenticated_pair('a'),authenticated_pair('b')])
    train=NativeGraphSource(path,'1'*64,required_split='train',role_ids=(58,59,60,61))
    pairs=SimpleNamespace(entries=[dict(row) for row in train.entries if row['communication']])
    verify_initial_foundation_cohort(train,pairs)
    weaker=SimpleNamespace(entries=pairs.entries[:1])
    with pytest.raises(ValueError,match='entire currently declared communication cohort'):
        verify_initial_foundation_cohort(train,weaker)
    changed=SimpleNamespace(entries=[{**row,'binding':'changed'} for row in pairs.entries])
    with pytest.raises(ValueError,match='entire currently declared communication cohort'):
        verify_initial_foundation_cohort(train,changed)
    # Subsequent admitted communication is a mandatory deficit, not a moving
    # first foundation. The recipe separately retains its exact source binding.
    verify_initial_foundation_cohort(train,weaker,already_declared=True)


def test_actual_sft_foundation_stops_before_ordinary_and_shared_resume_consumes_only_deficits(tmp_path):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model,SHARED_CHECKPOINT_SCHEMA
    from natlang_neuralese.train.native_state_sft import main
    from natlang_neuralese.train.loop import capture_training_rng_state
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint']
    ordinary=[dict(graph(name),tokenizer_fingerprint=fp) for name in ('a','b')]
    manifest=source_manifest(tmp_path,ordinary,[authenticated_pair('real',roles=roles.ids,fingerprint=fp)])
    held=tmp_path/'held.jsonl'
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    out=tmp_path/'sft'
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(manifest),'--foundation-native-graphs',str(manifest),
        '--communication-foundation','--held-native-graphs',str(held),'--epochs','1',
        '--stage-id','full','--eval-every','100','--out',str(out),
        '--forward-atol','1e-4','--gradient-atol','1e-5']
    assert main(args)==0
    first=torch.load(out/'checkpoint.pt',weights_only=False)
    frame=first['training']
    assert frame['exposure_counts']=={'a':0,'b':0,'lfm:real':1}
    report=json.loads((out/'report.json').read_text())
    assert not report['completed'] and report['communication_foundation']['complete']
    assert not report['foundation_ready']  # no task-execution proof was supplied
    assert report['mandatory_sft']['remaining_graph_visits']==2
    # Exact shared checkpoint bridge fixture. Actual joint-stage preservation is
    # separately exercised against real VAE/joint/distill optimizer updates.
    frames=collect_sft_progress(frame)
    frames['full']['completed_foundations']=[report['communication_foundation']['cohort_sha256']]
    current_rng=capture_training_rng_state('cpu')
    shared=tmp_path/'shared.pt'
    torch.save({'schema':SHARED_CHECKPOINT_SCHEMA,'participants':{'student':{
        'model_config':first['model_config'],'model_state':{k:first[k] for k in ('backbone_state','interface_state')},
        'optimizer':frame['optimizer'],'fp8':None,'sft_progress':frames,'global_step':10}},
        'training':{'step':4,'rng':current_rng}},shared)
    resumed=tmp_path/'resumed'
    next_args=[*args]
    next_args[next_args.index('--out')+1]=str(resumed)
    next_args += ['--continue-from',str(shared),'--native-participant','student','--objective-continuation']
    next_args[next_args.index('--eval-every')+1]='1'
    next_args += ['--refresh-at-checkpoint','--checkpoint-every','1']
    assert main(next_args)==0
    quantum=torch.load(resumed/'checkpoint.pt',weights_only=False)['training']
    assert quantum['exposure_counts']=={'a':1,'b':0,'lfm:real':1}
    partial=json.loads((resumed/'report.json').read_text())
    assert not partial['completed'] and partial['continuation_boundary']['reached']
    assert partial['continuation_boundary']['new_optimizer_updates']==1
    assert partial['mandatory_sft']['remaining_graph_visits']==1
    assert main(next_args)==0
    final=torch.load(resumed/'checkpoint.pt',weights_only=False)['training']
    assert final['exposure_counts']=={'a':1,'b':1,'lfm:real':1}
    assert final['step']==3 and final['global_step']==12
    assert progress_obligations(final['sft_progress'])['complete']
    assert json.loads((resumed/'report.json').read_text())['completed']


def test_actual_joint_curriculum_preserves_ce_deficits_and_monotonic_model_steps(tmp_path,monkeypatch):
    from test_native_semantic_stage import _fixture
    from natlang_neuralese.train import native_semantic_stage as stage
    path,manifest,_,_=_fixture(tmp_path,monkeypatch)
    load=stage.load_participant
    frames=collect_sft_progress(progress())
    def with_progress(*args,**kwargs):
        participant=load(*args,**kwargs)
        participant.sft_progress=copy.deepcopy(frames)
        participant.global_step=10
        return participant
    monkeypatch.setattr(stage,'load_participant',with_progress)
    report=stage.run(path,manifest,tmp_path/'joint',max_updates=1,device='cpu')
    assert not report['completed']
    checkpoint=tmp_path/'joint/checkpoint.pt'
    stage.run(path,manifest,tmp_path/'joint',resume=checkpoint,device='cpu')
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    saved=CheckpointPolicy.load(checkpoint)
    for entry in saved['participants'].values():
        assert entry['sft_progress']==frames
        assert progress_obligations(entry['sft_progress'])['remaining_graph_visits']==1
        assert entry['global_step']==12  # VAE freezes model, joint+distill update it


def test_readiness_foundation_is_explicit_and_checks_full_admitted_cohort(tmp_path):
    from test_native_readiness import ready
    from natlang_neuralese.train.native_readiness import validate_ready
    from natlang_neuralese.common.hashing import sha256_file_hex
    path,receipt,requirement=ready(tmp_path,'lfm')
    source_dir=tmp_path/'sources';source_dir.mkdir()
    fp=receipt['native_identity']['tokenizer_fingerprint']
    ordinary=[dict(graph('ordinary'),tokenizer_fingerprint=fp)]
    manifest=source_manifest(source_dir,ordinary,[authenticated_pair('a',fingerprint=fp)])
    source=NativeGraphSource(manifest,fp,required_split='train',role_ids=(58,59,60,61))
    foundation=communication_foundation(source,{'lfm:a':1},epochs=1,trained_bindings=source.bindings)
    obligations=progress_obligations(collect_sft_progress(progress()))
    report_path=Path(receipt['training_receipt']['path'])
    report=json.loads(report_path.read_text())
    report.update(completed=False,foundation_ready=True,communication_foundation=foundation,mandatory_sft=obligations,
                  model_config={'marker_ids':[58,59,60,61]})
    report_path.write_text(json.dumps(report))
    receipt.update(completed=False,foundation_ready=True,communication_foundation=foundation,mandatory_sft=obligations)
    receipt['training_receipt']['sha256']=sha256_file_hex(report_path)
    path.write_text(json.dumps(receipt))
    assert validate_ready(path,requirement) is None  # never forge full-stage completion
    requirement.update(readiness_boundary='communication-foundation',foundation_epochs=1,
                       foundation_sources={'path':str(manifest),'sha256':sha256_file_hex(manifest)})
    selected=validate_ready(path,requirement)
    assert selected['readiness_boundary']=='communication-foundation'
    assert selected['mandatory_sft']['remaining_graph_visits']==1
    # Later admissions outside the explicitly declared foundation remain
    # mandatory, but cannot move this already-completed cohort's finish line.
    pending={**report['communication_exposure'],'complete':False,'required_graph_visits':8,'graph_visits':4}
    report['communication_exposure']=pending;report_path.write_text(json.dumps(report))
    receipt['communication_exposure']=pending;receipt['training_receipt']['sha256']=sha256_file_hex(report_path)
    path.write_text(json.dumps(receipt))
    assert validate_ready(path,requirement)['readiness_boundary']=='communication-foundation'
    # A larger admitted cohort cannot be qualified by the earlier tiny shard.
    bigger=tmp_path/'bigger';bigger.mkdir()
    full=source_manifest(bigger,ordinary,[authenticated_pair('a',fingerprint=fp),authenticated_pair('b',fingerprint=fp)])
    requirement['foundation_sources']={'path':str(full),'sha256':sha256_file_hex(full)}
    with pytest.raises(ValueError,match='entire declared admitted'):
        validate_ready(path,requirement)


def test_new_raw_cohort_deficits_keep_previous_semantic_membership_without_losing_pending_rows(tmp_path):
    from natlang_neuralese.train.native_continuation import qualified_refresh_config
    from natlang_neuralese.common.hashing import canonical_json_sha256_hex,sha256_file_hex
    old=tmp_path/'old.jsonl';new=tmp_path/'new.jsonl'
    a,b=authenticated_pair('a'),authenticated_pair('b')
    old.write_text(json.dumps(a)+'\n');new.write_text(json.dumps(a)+'\n'+json.dumps(b)+'\n')
    previous={'pairs':{'path':str(old),'sha256':sha256_file_hex(old)},'held_pairs':{'path':'prior-held','sha256':'held'},'vae_lr':.01}
    proposed={**previous,'pairs':{'path':str(new),'sha256':sha256_file_hex(new)},'vae_lr':.02}
    config=tmp_path/'config.json';config.write_text(json.dumps(proposed))
    models={}
    indexes={}
    for name,fp in [('lfm','1'*64),('mellum','2'*64)]:
        index=NativeGraphSource(new,fp,required_split='train',pair_model=name,role_ids=(58,59,60,61))
        indexes[name]=index
        first=index.entries[0]
        models[name]={'native_identity':{'tokenizer_fingerprint':fp},'marker_ids':[58,59,60,61],
            'raw_communication_rows':{first['id']:{'binding':first['binding'],'qualified':True}}}
    source={'qualified_curriculum':{'config':previous,'sha256':canonical_json_sha256_hex(previous)}}
    selected,pending=qualified_refresh_config(source,config,models)
    assert selected['pairs']==previous['pairs'] and selected['vae_lr']==.02
    assert {entry['graph_id'] for entry in pending}=={'lfm:b','mellum:b'}
    with pytest.raises(ValueError,match='no exact previous'):
        qualified_refresh_config({},config,models)
    for name,index in indexes.items():
        models[name]['raw_communication_rows'].update({entry['id']:{'binding':entry['binding'],'qualified':True} for entry in index.entries})
    selected,pending=qualified_refresh_config(source,config,models)
    assert selected==proposed and pending==[]


@pytest.mark.parametrize('objective_continuation', [False, True])
def test_actual_resume_drops_setup_metadata_before_next_update(tmp_path, monkeypatch, objective_continuation):
    import weakref
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer

    base=tmp_path/'base';tiny_source(base)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint']
    source=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    source.write_text(''.join(json.dumps(dict(graph(name),tokenizer_fingerprint=fp))+'\n' for name in ('a','b')))
    held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    out=tmp_path/'first'
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(source),'--held-native-graphs',str(held),'--epochs','1',
        '--stage-id','full','--eval-every','1','--out',str(out),
        '--refresh-at-checkpoint','--checkpoint-every','1',
        '--forward-atol','1e-4','--gradient-atol','1e-5']
    assert trainer.main(args)==0
    first=torch.load(out/'checkpoint.pt',weights_only=False)
    assert sum(first['training']['exposure_counts'].values())==1
    references=[]

    class TrackedDict(dict):
        def __deepcopy__(self,memo):
            result={};memo[id(self)]=result
            result.update((copy.deepcopy(k,memo),copy.deepcopy(v,memo)) for k,v in self.items())
            return result

    class TrackedList(list):
        def __deepcopy__(self,memo):
            result=[];memo[id(self)]=result
            result.extend(copy.deepcopy(v,memo) for v in self)
            return result

    def track(frame,label):
        frame=TrackedDict(frame);references.append((label,weakref.ref(frame)))
        for key in ('source_row_bindings','held_row_bindings','exposure_counts','options','source_row_order','source_receipts'):
            value=frame.get(key)
            if isinstance(value,(dict,list)):
                value=(TrackedDict if isinstance(value,dict)else TrackedList)(value)
                frame[key]=value;references.append((label+'.'+key,weakref.ref(value)))
        return frame

    load=trainer.load_native_checkpoint
    def tracked_load(*positional,**keywords):
        model,tokenizer,roles,config,state=load(*positional,**keywords)
        state['training']=track(state['training'],'decoded')
        state['training']['sft_progress']={key:track(value,'historical.'+key)
            for key,value in state['training'].get('sft_progress',{}).items()}
        return model,tokenizer,roles,config,state
    monkeypatch.setattr(trainer,'load_native_checkpoint',tracked_load)
    restore=trainer.restore_sft_progress
    def tracked_restore(*positional,**keywords):
        frame=restore(*positional,**keywords)
        return track(frame,'restored') if frame is not None else None
    monkeypatch.setattr(trainer,'restore_sft_progress',tracked_restore)
    update=trainer.native_training_update;observations=[]
    def checked_update(*positional,**keywords):
        assert references
        alive=[name for name,ref in references if ref()is not None]
        assert not alive, 'setup-only checkpoint metadata retained: '+', '.join(alive)
        observations.append(True)
        return update(*positional,**keywords)
    monkeypatch.setattr(trainer,'native_training_update',checked_update)
    resumed=out
    if objective_continuation:
        resumed=tmp_path/'continued';args[args.index('--out')+1]=str(resumed)
        args+=['--continue-from',str(out/'checkpoint.pt'),'--objective-continuation']
    assert trainer.main(args)==0
    assert observations==[True]
    final=torch.load(resumed/'checkpoint.pt',weights_only=False)
    assert final['training']['global_step']==2
    assert final['training']['exposure_counts']=={'a':1,'b':1}
    assert final['training']['rng'] and final['training']['optimizer']
