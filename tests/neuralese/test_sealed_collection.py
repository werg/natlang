import copy
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.rl.sealed_collection import resume_sealed_collection
from natlang_neuralese.rl.production import validate_actor_receipts, collect_with_actor


def fixture(tmp_path):
    out = tmp_path/'round';out.mkdir()
    policy = 'a'*64
    cases = out/'selected-cases.jsonl';cases.write_text('{}\n')
    output = out/'result.jsonl';output.write_text('{}\n')
    binding = dict(schema='natlang.rl-collection-binding/1', policy_id=policy,
        cases_sha256=hashlib.sha256(cases.read_bytes()).hexdigest(), rows=1,offset=0,group=1,
        root_mode='student',seed_base=700000,temperature=1.,guided=False,case_seconds=600,
        max_model_requests=128,max_turns=160,endpoints=['http://old'],neuralese_server=False,
        runtime={'dist/old.js':'b'*64},collector_sha256='c'*64)
    report = dict(schema='natlang.s7-task-rollout-score/1',collection_complete=True,root_mode='student',rollouts=1,
        emitted_rollouts=str(out/'rollouts.jsonl'),
        actor_receipts={'http://old':{'checkpoint_sha256':policy,'communication':'text'}})
    values={'collection-binding.json':binding,'score.json':report,'packet.jsonl':{},
            'queue.jsonl':{'output':str(output)},'journal.jsonl':{},'preparation.json':{},
            'sampling-config.json':{},'rollouts.jsonl':{'schema':'natlang.rollout/1',
                'policy':{'checkpoint_sha256':policy},'turns':[]},'result.jsonl':{}}
    for name,value in values.items():(out/name).write_text(json.dumps(value)+'\n')
    def bind(path):return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    proof=out/'continuation.json';proof.write_text(json.dumps({'schema':'natlang.sealed-collection-continuation/1',
        'out':str(out),'files':{name:bind(out/name)for name in values}}))
    request=dict(out=out,cases=cases,policy_id=policy,rows=1,group=1,endpoints=['http://old'],
                 neuralese_server=False,runtime=tmp_path/'new-runtime')
    return out,bind(proof),request,bind


def test_completed_round_uses_original_provenance_after_future_runtime_change(tmp_path,monkeypatch):
    out,ref,request,_=fixture(tmp_path)
    from natlang_neuralese.rl import production
    monkeypatch.setattr(production,'collect',lambda **kw:pytest.fail('sealed round must not recollect'))
    config={'sealed_collection_continuations':{str(out):ref}}
    before=(out/'collection-binding.json').read_bytes()
    report=collect_with_actor(config,{},**request)
    assert report['collection_complete']
    assert (out/'collection-binding.json').read_bytes()==before


@pytest.mark.parametrize('field,value',[('policy_id','d'*64),('rows',2),('group',2),('seed_base',9)])
def test_completed_round_rejects_changed_sampling_or_policy(tmp_path,field,value):
    _,ref,request,_=fixture(tmp_path);request[field]=value
    with pytest.raises(ValueError,match='policy/cases/sampling'):
        resume_sealed_collection(ref,request,validate_actors=validate_actor_receipts)


def test_completed_round_rejects_changed_captured_bytes(tmp_path):
    out,ref,request,_=fixture(tmp_path);(out/'rollouts.jsonl').write_text('{}\n')
    with pytest.raises(ValueError):resume_sealed_collection(ref,request,validate_actors=validate_actor_receipts)


def test_completed_round_requires_explicit_pinned_result_sources(tmp_path):
    out,ref,request,bind=fixture(tmp_path);p=Path(ref['path']);proof=json.loads(p.read_text())
    proof['files'].pop('result.jsonl');p.write_text(json.dumps(proof));ref=bind(p)
    with pytest.raises(ValueError,match='completed result'):
        resume_sealed_collection(ref,request,validate_actors=validate_actor_receipts)


def test_incomplete_round_cannot_use_completed_continuation(tmp_path):
    out,ref,request,bind=fixture(tmp_path);score=out/'score.json';d=json.loads(score.read_text());d['collection_complete']=False;score.write_text(json.dumps(d))
    p=Path(ref['path']);proof=json.loads(p.read_text());proof['files']['score.json']=bind(score);p.write_text(json.dumps(proof))
    with pytest.raises(ValueError,match='completed collection'):
        resume_sealed_collection(bind(p),request,validate_actors=validate_actor_receipts)
