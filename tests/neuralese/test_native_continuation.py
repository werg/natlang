"""Mandatory supervised credits survive objective switches, not just metadata registration."""
import copy
import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.train.native_continuation import (collect_sft_progress,restore_sft_progress,
    progress_obligations,communication_foundation)
from natlang_neuralese.data.native_sources import NativeGraphSource
from test_native_state_sft import graph, authenticated_pair, source_manifest


def progress():
    return {'schema':'natlang.neuralese-native-sft-training/1','step':3,'global_step':3,'epoch':0,'cursor':1,
        'identity':{'objective':None},'source_row_bindings':{'old':'binding0','new':'binding1'},
        'source_row_order':['old','new'],'exposure_counts':{'old':1,'new':0},
        'options':{'stage_id':'full','epochs':1},'optimizer':{'actual':'current moments'},
        'rng':{'actual':'current stream'},'fp8':{'actual':'current scale'}}


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
    saved=torch.load(checkpoint,weights_only=False)
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
