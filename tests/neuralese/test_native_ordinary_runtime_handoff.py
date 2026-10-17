"""Retained ordinary capacity is authenticated by a genuinely observed owner."""
import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from test_native_bare_runtime_handoff import bare
from test_native_later_runtime_handoff import later, _assert_next_update
from natlang_neuralese.rl import native_runtime_handoff as runtime, native_raw_handoff as handoff
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.recipe import native_recipe_binding
from natlang_neuralese.common.jsonio import write_json_durable as write


@pytest.fixture
def ordinary(bare,tmp_path):
    x=bare;root=tmp_path/'ordinary-owner';root.mkdir()
    saved=copy.deepcopy(x['saved']);options=saved['training']['options']
    options.pop('memory_qualification',None)
    parent=json.loads(Path(x['handoff']['path']).read_text())['checkpoint']
    options.update(batch_graphs=1,batch_tokens=32768,checkpoint_layers=True,chunk_tokens=1024,
        checkpoint_input_offload=False,share_causal_prefixes=False,fp8_training='off',
        fp8_recipe=None,fp8_amax_history_len=16,continuation={'checkpoint_sha256':parent['sha256']})
    identity={key:options[key]for key in('batch_graphs','batch_tokens','checkpoint_layers','chunk_tokens','checkpoint_input_offload')}
    identity.update(model_config=saved['model_config'],runtime={'total_memory_bytes':20<<30},
                    fp8={'kind':'off','recipe':None,'amax_history_len':16,'input_gradient_precision':None})
    # Protocol evidence only: these fixtures never claim production GPU capacity.
    proof=root/'capacity-proof.json';write(proof,{'schema':'natlang.native-memory-qualification/1',
        'scope':'CPU authority fixture, not measured production GPU capacity','passed':True,
        'identity':identity,'runtime_code_sha256':resident.digest(resident.code_inventory(x['newpackage'])),
        'parent_checkpoint_sha256':parent['sha256'],'optimizer_updates':0,
        'numerical_reference':{'passed':True},'capacity':{'passed':True,'cases':[
            {'status':'passed','forward_tokens':12,'kv_slots':128}]}})
    certificate=root/'capacity.json';write(certificate,{'schema':'natlang.native-resident-geometry-qualification/1',
        'passed':True,'identity':identity,'runtime_code_sha256':resident.digest(resident.code_inventory(x['newpackage'])),
        'limits':{'forward_tokens':12,'kv_slots':128},'evidence':resident.binding(proof)})
    recipe=json.loads(Path(x['bare']['recipe']['path']).read_text())
    stage=next(s for s in recipe['stages']if s['id']=='sft')
    stage['parameters']['resident_geometry_qualification']=str(certificate)
    recipe_path=root/'recipe.json';write(recipe_path,recipe)
    config=json.loads(x['config'].read_text());rb=native_recipe_binding(recipe_path,config['recipe_lineage_sha256'])
    options.update(resident_geometry_qualification=str(certificate),recipe_binding=rb)
    trainer_argv=[sys.executable,'-c','import sys;sys.stdin.buffer.read()',
        '--out',str(root),'--resident-geometry-qualification',str(certificate),'--stage-id','sft','--recipe-binding',json.dumps(rb)]
    argv_path=root/'argv.json';write(argv_path,trainer_argv)
    continuation=root/'continuation.json';write(continuation,{'parent':parent,'recipe':rb,'certificate':resident.binding(certificate)})
    script=root/'supervisor.py';script.write_text("import subprocess,sys,json;p=subprocess.Popen(json.loads(sys.argv[1]),stdin=subprocess.PIPE);print('ready',flush=True);sys.stdin.buffer.read();p.communicate()")
    intent_path=root/'launch-intent.json';write(intent_path,{'continuation':resident.binding(continuation),'argv_binding':resident.binding(argv_path),
        'supervisor':resident.binding(script),'env':{'PYTHONPATH':str(x['newpackage'].parent)}})
    supervisor=[sys.executable,str(script),json.dumps(trainer_argv)]
    process=subprocess.Popen(supervisor,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
    try:
        assert process.stdout.readline().strip()=='ready'
        owner=handoff.observe(process.pid)
        original=root/'launch.json';write(original,{'owner':owner,'argv':trainer_argv,'intent':resident.binding(intent_path)})
        spawn=root/'spawn.json';write(spawn,{'identity':owner,'intent':resident.binding(intent_path),'script':resident.binding(script)})
        observed=root/'observed-launch.json';write(observed,{'at':time.time(),'pid':owner['pid'],'argv':supervisor,
            'identity':owner,'original_launch':resident.binding(original),'spawn':resident.binding(spawn)})
        drain_intent=handoff.capture_drain_intent(resident.binding(observed),root/'drain-intent.json')
        slot=root/'checkpoint.pt';torch.save(saved,slot);pin=root/'parent.pt';shutil.copyfile(slot,pin)
        process.communicate();assert process.returncode==0
        terminal=root/'exit.json';write(terminal,{'at':time.time(),'returncode':0})
        drained=handoff.complete_drain(drain_intent,resident.binding(terminal),resident.binding(pin),root/'handoff.json',original_slot=resident.binding(slot))
    finally:
        if process.poll()is None:process.communicate()
    state=json.loads((x['out']/'state.json').read_text());state['checkpoint']=str(pin);write(x['out']/'state.json',state)
    authority={'recipe':resident.binding(recipe_path),'package':str(x['newpackage']),'ordinary_capacity':resident.binding(certificate)}
    x.update(saved=saved,pin=pin,slot=slot,handoff=drained,bare=authority,capacity=certificate,capacity_proof=proof,
             owner_continuation=continuation,owner_argv=argv_path,owner_supervisor=script)
    x['proof']=runtime.prepare_runtime_handoff(x['bootstrap'],drained,None,x['config'],root/'runtime.json',bare_direct_owner=authority)
    x['declaration']['runtime_handoff']=x['proof'];write(x['plan'],{'models':[x['declaration']]})
    return x


def test_ordinary_capacity_rejoins_without_fabricated_new_max_or_resident(ordinary):
    x=ordinary;before={p:p.read_bytes()for p in(x['pin'],x['config'],x['root']/'cycle.json',x['out']/'state.json')}
    value=runtime.verify_runtime_handoff(x['proof'],bootstrap=x['bootstrap'])
    assert value['bare_direct_owner']['ordinary_capacity']
    assert not value['bare_direct_owner'].get('qualification')
    assert all(p.read_bytes()==raw for p,raw in before.items())
    # Original operation metadata may legitimately be reused; exact snapshots survive.
    x['owner_continuation'].write_text('{"later":true}')
    x['owner_supervisor'].unlink()
    runtime.verify_runtime_handoff(x['proof'],bootstrap=x['bootstrap'])


@pytest.mark.parametrize('later',['lfm','mellum'],indirect=True)
def test_both_real_optimizer_next_updates_survive_retained_capacity_authority(ordinary):
    _assert_next_update(ordinary)


@pytest.mark.parametrize('change',['certificate','packing','profile','parent','objective','unpassed','measured-cases','launch-snapshot','mixed'])
def test_ordinary_capacity_cannot_borrow_unexecuted_or_incompatible_proof(ordinary,tmp_path,change):
    x=ordinary;value=json.loads(Path(x['proof']['path']).read_text());authority=value['bare_direct_owner']
    saved=copy.deepcopy(x['saved']);record,_=handoff.verify_handoff(x['handoff'])
    if change=='certificate':authority['requested']['ordinary_capacity']['sha256']='0'*64
    elif change=='packing':saved['training']['options']['batch_tokens']+=1
    elif change=='profile':saved['model_config']['native_ordinary_precision']='changed'
    elif change=='parent':saved['training']['options']['continuation']['checkpoint_sha256']='0'*64
    elif change=='objective':saved['training']['options']['objective_spec']='unqualified-objective'
    elif change in('unpassed','measured-cases'):
        proof=json.loads(x['capacity_proof'].read_text())
        if change=='unpassed':proof['numerical_reference']['passed']=False
        else:proof['capacity']['cases'][0]['kv_slots']=129
        write(x['capacity_proof'],proof)
        certificate=json.loads(Path(authority['ordinary_capacity']['path']).read_text());certificate['evidence']=resident.binding(x['capacity_proof'])
        forged=tmp_path/'forged-capacity.json';write(forged,certificate);authority['ordinary_capacity']=resident.binding(forged)
    elif change=='launch-snapshot':authority['launch_control']['argv']=authority['launch_control']['spawn']
    else:authority['qualification']=x['bare']['ordinary_capacity']
    with pytest.raises((ValueError,KeyError)):
        runtime._bare_authority(authority,record,saved,json.loads(x['config'].read_text()))
