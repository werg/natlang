import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.neuralese_data import provider_campaign_source as source


def ref(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'bytes':path.stat().st_size}


def fixture(tmp_path, monkeypatch, mutation=None):
    def put(name,value):
        p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value)+'\n');return p
    program={'id':'task','version':'natlang.program/2','split':'train','source_groups':['world-a','world-b']}
    cases=put('cases.jsonl',program)
    row={'id':'execution','version':'natlang.teacher_trajectory.native/1',
         'task':{'program_ir':program},
         'provenance':{'provider':'openai-codex','model':'gpt-6-luna',
                       'program_ir_sha256':source.canonical_json_sha256_hex(program)},
         'execution_identity':{'execution_run_id':'actual-episode'},
         'trajectory':[{'phase':'action','assistant':{'content':'return 1;'}}],
         'outcome':{'status':'done','accepted':True,'oracle':{'accepted':True},'checks':{'exact':True}}}
    if mutation=='actor':row['provenance']['model']='a-student'
    if mutation=='provider':row['provenance']['provider']='another-provider'
    if mutation=='oracle':row['outcome']['oracle']['accepted']=False
    if mutation=='trajectory':row['trajectory']=[]
    result=put('result.jsonl',row);combined=tmp_path/'combined.jsonl';combined.write_bytes(result.read_bytes())
    manifest=put('result.jsonl.manifest.json',{'version':'natlang.teacher_batch.native/1',
        'source_sha256':ref(cases)['sha256'],'range':{'start':0,'count':1},'completed':1,
        'missing':[],'output_sha256':ref(result)['sha256'],'model':'gpt-6-luna'})
    queue=put('queue.jsonl',{'index':0,'output':str(result)})
    plan_path=put('plan.json',{'fixture':'validated separately by dispatcher tests'})
    plan={'source':str(cases),'source_sha256':ref(cases)['sha256'],
          'model_id':'gpt-6-luna','provider':'openai-codex','runtime_receipt_sha256':'frozen-runtime'}
    identity={'campaign_id':'campaign','source':str(cases),'runtime':str(tmp_path/'runtime'),
              'campaign_root':str(tmp_path),'slots':6,'cases':[{'index':0,'seed':7}]}
    identity['identity_sha256']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    monkeypatch.setattr(source,'validated_dispatch_plan',lambda *args:(plan,identity))
    claim={'event':'claim','claim_id':'campaign-case-000000','index':0,'slot':1,
           'source':str(cases),'runtime':identity['runtime'],'queue':str(queue),
           'queue_sha256':ref(queue)['sha256'],'identity_sha256':identity['identity_sha256']}
    terminal={**claim,'event':'terminal','exit_code':0,
              'output_accounting':{'complete':True,'output_sha256':ref(result)['sha256']}}
    if mutation=='unclosed':terminal['exit_code']=1
    events=[{'event':'campaign_open','identity':identity},claim,terminal]
    ledger=tmp_path/'ledger.jsonl';ledger.write_text(''.join(json.dumps(v)+'\n'for v in events))
    declaration={'schema':source.SCHEMA,'training_admission':False,'source':ref(combined),
        'plan':ref(plan_path),'claim_ledger':ref(ledger),'results':[{'claim_id':claim['claim_id'],
        'queue':ref(queue),'result':ref(result),'manifest':ref(manifest)}]}
    witness=put('provider-source.json',declaration)
    if mutation=='changed-output':result.write_text('{}\n')
    protection={'held':{'world-b'}if mutation=='protected-related-world'else set(),
                'quality':set(),'pending':set(),'proof':{'sha256':'protected-projection'}}
    args=SimpleNamespace(source=combined,provider_source_manifest=witness,
                         protection_manifest=tmp_path/'protected.json')
    return args,protection


def test_closed_provider_source_keeps_every_world_and_actual_actor(tmp_path,monkeypatch):
    args,protection=fixture(tmp_path,monkeypatch)
    result=source.source_binding(args,protection_reader=lambda _:protection)
    assert result['train_groups']=={'world-a','world-b'}
    assert result['teacher']['teacher']=='gpt-6-luna'
    assert result['transfer']['training_admission']is False
    assert result['collections'][next(iter(result['collections']))]['runtime_model']=='gpt-6-luna'


@pytest.mark.parametrize('mutation',['actor','provider','oracle','trajectory','unclosed',
                                     'changed-output','protected-related-world'])
def test_provider_source_rejects_false_success_actor_or_custody(tmp_path,monkeypatch,mutation):
    args,protection=fixture(tmp_path,monkeypatch,mutation)
    with pytest.raises(ValueError):
        source.source_binding(args,protection_reader=lambda _:protection)
