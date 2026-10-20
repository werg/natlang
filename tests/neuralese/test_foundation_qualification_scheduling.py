"""Deferred exposure is not readiness; exact requests reuse the actual trainer."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train import native_state_sft as trainer
from natlang_neuralese.train.native_continuation import communication_foundation, collect_sft_progress, progress_obligations
from natlang_neuralese.train.native_readiness import (validate_foundation_candidate,
    validate_foundation_candidate_document, validate_foundation_qualification_request, validate_ready)
from natlang_neuralese.common.hashing import sha256_file_hex
from test_native_state_sft import graph, authenticated_pair, source_manifest


def candidate_fixture(root, name):
    from test_native_readiness import ready
    path,receipt,requirement=ready(root,name)
    sources=root/(name+'-sources');sources.mkdir()
    fp=receipt['native_identity']['tokenizer_fingerprint']
    manifest=source_manifest(sources,[dict(graph('ordinary'),tokenizer_fingerprint=fp)],
        [authenticated_pair('real',fingerprint=fp)])
    source=trainer.NativeGraphSource(manifest,fp,required_split='train',role_ids=(58,59,60,61))
    foundation=communication_foundation(source,{'lfm:real':1},epochs=1,trained_bindings=source.bindings)
    from test_native_continuation import progress
    obligations=progress_obligations(collect_sft_progress(progress()))
    report_path=Path(receipt['training_receipt']['path']);report=json.loads(report_path.read_text())
    changes=dict(completed=False,foundation_ready=False,communication_foundation=foundation,mandatory_sft=obligations)
    report.update(**changes,model_config={'marker_ids':[58,59,60,61]});report_path.write_text(json.dumps(report))
    receipt.update(changes);receipt['training_receipt']['sha256']=sha256_file_hex(report_path)
    path.write_text(json.dumps(receipt))
    requirement.update(readiness_boundary='communication-foundation',foundation_epochs=1,
        foundation_sources={'path':str(manifest),'sha256':sha256_file_hex(manifest)})
    return path,receipt,requirement


def test_unqualified_candidates_require_every_admitted_row_and_never_release(tmp_path):
    path,receipt,requirement=candidate_fixture(tmp_path,'lfm')
    assert validate_ready(path,requirement) is None
    candidate=validate_foundation_candidate(path,requirement)
    assert candidate['qualified'] is False
    assert validate_foundation_candidate_document(candidate,requirement) is candidate
    forged=copy.deepcopy(candidate)
    next(iter(forged['communication_foundation']['rows'].values()))['graph_visits']=0
    with pytest.raises(ValueError):validate_foundation_candidate_document(forged,requirement)
    forged=copy.deepcopy(candidate);forged['native_identity'].pop('checkpoint_sha256');forged['checkpoint'].pop('sha256')
    with pytest.raises(ValueError,match='identities'):validate_foundation_candidate_document(forged,requirement)


def test_shared_request_requires_two_exact_unqualified_cohorts(tmp_path):
    candidates={};requirements={}
    for name in ('lfm','mellum'):
        path,receipt,requirement=candidate_fixture(tmp_path,name)
        candidates[name]=validate_foundation_candidate(path,requirement);requirements[name]=requirement
    path=tmp_path/'request.json'
    request={'schema':'natlang.native-foundation-qualification-request/1','readiness_granted':False,
        'contract':{'models':requirements},'candidates':candidates}
    path.write_text(json.dumps(request))
    ref=validate_foundation_qualification_request(path,'lfm',candidates['lfm']['communication_foundation'])
    assert ref['sha256']==sha256_file_hex(path)
    request['candidates'].pop('mellum');path.write_text(json.dumps(request))
    with pytest.raises(ValueError,match='two actual'):validate_foundation_qualification_request(path,'lfm',candidates['lfm']['communication_foundation'])


def test_request_boundaries_remain_explicit_even_for_inherited_cohort():
    foundation={'complete':True,'cohort_sha256':'earned'}
    options=dict(completed=False,new_optimizer_updates=1,policy_operation=False,signaled=False,
        refresh_requested=True,readiness_policy='on-shared-request')
    assert not trainer._foundation_readiness_boundary(foundation,[],**options)
    assert trainer._foundation_readiness_boundary(foundation,['earned'],**options,qualification_requested=True)
    assert not trainer._foundation_readiness_boundary(foundation,[],**{**options,'signaled':True},qualification_requested=True)


def test_actual_deferred_foundation_continues_and_exactly_resumes_optimizer(tmp_path,monkeypatch):
    from test_native_model import tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.eval import task_cadence
    base=tmp_path/'base';tiny_source(base)
    _,_,roles,config=load_native_model(base,dtype=torch.float32,rank=4)
    fp=config['tokenizer_fingerprint']
    manifest=source_manifest(tmp_path,[dict(graph(n),tokenizer_fingerprint=fp) for n in ('a','b','c','d','e')],
        [authenticated_pair('real',roles=roles.ids,fingerprint=fp)])
    held=tmp_path/'held.jsonl';held.write_text(json.dumps(dict(graph('held'),split='test',tokenizer_fingerprint=fp))+'\n')
    calls=[]
    monkeypatch.setattr(task_cadence,'TaskEval',lambda spec,out,engine,**kwargs:
        SimpleNamespace(engine=engine,cadence=lambda *a,**k:calls.append((a,k))))
    args=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(manifest),'--foundation-native-graphs',str(manifest),
        '--communication-foundation','--held-native-graphs',str(held),'--epochs','1',
        '--stage-id','full','--eval-every','1','--checkpoint-every','1','--task-eval-every','1',
        '--task-eval',json.dumps({'packet':{'path':'unused','sha256':'s'},'cases':[0]}),
        '--foundation-readiness-policy','on-shared-request']
    # Both paths run two genuine optimizer updates. The split path crosses an
    # inherited complete foundation after update one without inventing an ACK.
    reference=tmp_path/'reference';split=tmp_path/'split'
    assert trainer.main([*args,'--out',str(reference),'--refresh-after-updates','2'])==0
    for _ in range(2):assert trainer.main([*args,'--out',str(split),'--refresh-after-updates','1'])==0
    assert not calls  # no 16-case synchronous work despite cadence=1
    def equal(a,b):
        if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
        elif isinstance(a,dict):
            assert a.keys()==b.keys()
            for k in a:equal(a[k],b[k])
        elif isinstance(a,(tuple,list)):
            assert len(a)==len(b)
            for x,y in zip(a,b,strict=True):equal(x,y)
        elif hasattr(a,'shape'):assert (a==b).all()
        else:assert a==b
    actual=torch.load(split/'checkpoint.pt',weights_only=False);expected=torch.load(reference/'checkpoint.pt',weights_only=False)
    for key in ('optimizer','rng','exposure_counts','global_step','cursor'):equal(actual['training'][key],expected['training'][key])
    for key in ('backbone_state','interface_state'):equal(actual[key],expected[key])
    receipt=json.loads((split/'native-training-receipt.json').read_text())
    assert receipt['foundation_qualification_schedule']['candidate'] is True
    assert not receipt['foundation_ready'] and not receipt['completed']
    assert receipt['mandatory_sft']['remaining_graph_visits']==4
    assert not any(frame.get('completed_foundations') for frame in collect_sft_progress(actual['training']).values())
    runtime=json.loads((split/'native-runtime-certificate.json').read_text())
    assert runtime['deferred'] and runtime['reason']=='shared qualification request pending'
    # Issue a two-candidate scheduling request, then remove it on the next
    # ordinary continuation. Peer metadata is a transport fixture, not proof
    # that a second CUDA model was qualified in this test.
    own=receipt['model_id']
    candidate={'schema':'natlang.native-foundation-candidate/1','qualified':False,
        **{key:receipt[key] for key in ('model_id','stage_id','checkpoint','native_identity',
            'communication_foundation','mandatory_sft')}}
    peer=copy.deepcopy(candidate);peer['model_id']='peer-protocol-fixture'
    requirement={'model_id':own,'stage_ids':['full'],'readiness_boundary':'communication-foundation',
        'foundation_epochs':1,'foundation_sources_sha256':sha256_file_hex(manifest)}
    request=tmp_path/'qualified-request.json'
    request.write_text(json.dumps({'schema':'natlang.native-foundation-qualification-request/1',
        'readiness_granted':False,'contract':{'models':{'own':requirement,'peer':{
            **requirement,'model_id':peer['model_id']}}},'candidates':{own:candidate,peer['model_id']:peer}}))
    from natlang_neuralese.eval import native_backend
    original=native_backend.qualify_complete_runtime;checks=[]
    def qualify(*a,**kw):
        checks.append(kw['checkpoint_sha256'])
        return original(*a,**kw)
    monkeypatch.setattr(native_backend,'qualify_complete_runtime',qualify)
    assert trainer.main([*args,'--out',str(split),'--refresh-after-updates','1',
        '--foundation-qualification-request',str(request),'--forward-atol','1e-4','--gradient-atol','1e-5'])==0
    assert len(calls)==len(checks)==1
    checked=torch.load(split/'checkpoint.pt',weights_only=False)
    binding=checked['training']['options']['foundation_qualification_request_binding']
    assert binding['sha256']==sha256_file_hex(request)
    assert trainer.main([*args,'--out',str(split),'--refresh-after-updates','1'])==0
    after=torch.load(split/'checkpoint.pt',weights_only=False)
    assert after['training']['step']==4
    assert after['training']['options']['foundation_qualification_history']==[binding]
    assert 'foundation_qualification_request_binding' not in after['training']['options']
    assert len(calls)==len(checks)==1
    assert trainer.main([*args,'--out',str(reference),'--refresh-after-updates','2'])==0
    expected=torch.load(reference/'checkpoint.pt',weights_only=False)
    for key in ('optimizer','rng','exposure_counts','global_step','cursor'):
        equal(after['training'][key],expected['training'][key])
    for key in ('backbone_state','interface_state'):equal(after[key],expected[key])


