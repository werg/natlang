"""Normal metadata adoption, not CUDA/model readiness qualification."""
import copy
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/coreweave'))
import prepare_native_scheduler_adoption as m
from prepare_native_actor_role import prepare as normal_prepare
from test_native_role_scheduler_deployment import source
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.native_resident import code_inventory


def setup(tmp):
    plans,_,curriculum,provider,states,runtime=source(tmp)
    materials={};inputs={}
    for role in plans:
        recipe=tmp/(role+'-recipe.json')
        policy=m.read(m.binding(curriculum))['models'][role+':base']['task_eval']
        write(recipe,{'id':role+'-normal-recipe','stages':[{'id':'full','kind':'native_state_sft','parameters':{'task_eval':policy}}]})
    boot_path=tmp/'rl-bootstrap.json';boot=json.loads(boot_path.read_text())
    for row in boot['models']:row['production_recipe']=m.binding(tmp/(row['model']+'-recipe.json'))
    write(boot_path,boot)
    for role,ref in list(plans.items()):
        plan=m.read(ref)
        for row in plan['models']:
            row['actor_runtime']=str(runtime)
            row['production_recipe']=m.binding(tmp/(row['model']+'-recipe.json'))
        write(Path(ref['path']),plan);plans[role]=m.binding(ref['path'])
    def emit(p,v):p.parent.mkdir(parents=True,exist_ok=True);write(p,v);return m.binding(p)
    for role in plans:
        root=runtime/'role-control'/role
        refs={name:emit(root/(name+'.json'),{'fixture':name})for name in
              ('binder','control_loader','control_manifest','role_mechanism_authorities')}
        refs['role_mechanism_authorities']=emit(root/'role_mechanism_authorities.json',
            {name:row['verification_mechanism']for name,row in m.read(plans[role])['role_exchange']['roles'].items()})
        controller=tmp/'controller.py'
        material={**refs,'controller':m.binding(controller),'model':role,'runtime':str(runtime),
            'actor_runtime':str(runtime),'runtime_code_sha256':canonical_json_sha256_hex(code_inventory(runtime/'training/neuralese/natlang_neuralese')),
            'controls':{}}
        material['publisher_loader']=emit(root/'publisher-loader.json',{'fixture':'bound loader'})
        material['publisher_manifest']=emit(root/'publishing-manifest.json',{
            'controls':{},'role_authorities':material['role_mechanism_authorities']})
        materials[role]=emit(root/'material.json',material)
        review=emit(tmp/(role+'-review.json'),{'passed':True,'controller':material['controller'],'binder':material['binder']})
        inputs[role]=emit(tmp/(role+'-inputs.json'),{'entry_root':str(tmp/role/'future-owner'),
            'bootstrap':plans[role],'root_review':review,'provider_readiness':m.binding(provider),
            'publication_configuration':{},'frozen_refs':[]})
    record={'schema':'natlang.native-actor-coordinator-composition/1','actor_runtime':str(runtime),
        'inventory':code_inventory(runtime/'training/neuralese/natlang_neuralese'),
        'code':canonical_json_sha256_hex(code_inventory(runtime/'training/neuralese/natlang_neuralese')),
        'role_materials':materials,
        'files':{str(p.relative_to(runtime)):{'sha256':m.binding(p)['sha256'],'bytes':p.stat().st_size}
                 for p in runtime.rglob('*')if p.is_file()}}
    construction=emit(runtime/'construction.json',record)
    spec=tmp/'adoption.json';write(spec,{'schema':m.SCHEMA,'composition':construction,
        'roles':{role:{'inputs':inputs[role],'raw_state':states[role]}for role in plans},
        'scheduler':{'runtime':str(runtime),'python':sys.executable,'curriculum':m.binding(curriculum),
            'provider':m.binding(provider),'rl_bootstrap':m.binding(tmp/'rl-bootstrap.json'),'handoff_lease_seconds':180}})
    return spec,runtime,plans


def publish(tmp,prepared,runtime):
    proposal=m.read(prepared['proposal']);result={}
    record=json.loads((runtime/'construction.json').read_text())
    for row in proposal['source_plan_successions']:
        role=row['role'];plan=m.read(row['successor']);declaration=next(v for v in plan['models']if v['model']==role)
        root=tmp/role/'published';root.mkdir(parents=True);material=m.read(record['role_materials'][role])
        spec=root/'controller-spec.json';write(spec,{**material,'numerical_package':str(runtime/'training/neuralese/natlang_neuralese'),
            'numerical_code_sha256':record['code']})
        entered=root/'controller-entered.json';write(entered,{'spec':m.binding(spec),'plan':row['successor'],'whole_recipe_completed':False})
        config=root/'config.json';write(config,{'production_recipe':declaration['production_recipe']['path'],
            'supervised_quantum_control':{'plan':row['successor']},'supervised_stage_quantum_control':{'plan':row['successor']}})
        result[role]={'spec':m.binding(spec),'entered':m.binding(entered),'raw_config':m.binding(config)}
    path=tmp/'published.json';write(path,result);return path


def test_normal_both_role_inputs_then_actual_adoption_before_launch(tmp_path):
    spec,runtime,plans=setup(tmp_path)
    original={role:Path(ref['path']).read_bytes()for role,ref in plans.items()}
    prepared_ref=m.prepare(spec,tmp_path/'proposal');prepared=m.read(prepared_ref)
    for role,ref in prepared['normal_inputs'].items():
        inputs=m.read(ref);plan=m.read(inputs['bootstrap'])
        assert plan['snapshot_preparation'] and plan['consumer_preparation_handoff']
        assert plan['models']==json.loads(original[role])['models']
        # Call the real existing normal preparation utility, without an owner.
        output=Path(prepared['normal_prepare_commands'][role][-1])
        normal_prepare(runtime,role,ref['path'],output)
        assert json.loads(output.read_text())['bootstrap']==inputs['bootstrap']
    published=publish(tmp_path,prepared,runtime)
    ready_ref=m.reconcile(prepared_ref['path'],published,tmp_path/'adopted');ready=m.read(ready_ref)
    assert ready['owner_changes']==ready['checkpoint_decodes']==0
    assert ready['activated'] is ready['readiness_granted'] is False
    relay=m.read(ready['relay']);controller=m.read(ready['controller'])
    for role,plan in ready['plans'].items():assert relay['hosts'][role]['plan']==plan
    assert controller['raw_rl_bootstrap']==ready['plans']['mellum']
    assert controller['manage_raw_bootstrap'] is False
    assert '--config' in ready['controller_command']
    assert 'relay_command'not in ready  # credentialed source host, not consumer Python
    assert original=={role:Path(ref['path']).read_bytes()for role,ref in plans.items()}


@pytest.mark.parametrize('fault',['raw-old-plan','missing-snapshot-flag','old-code','wrong-controller','commands'])
def test_published_conflicts_cannot_activate_scheduler(tmp_path,fault):
    spec,runtime,_=setup(tmp_path);prepared_ref=m.prepare(spec,tmp_path/'proposal');prepared=m.read(prepared_ref)
    published=publish(tmp_path,prepared,runtime);items=json.loads(published.read_text());item=items['lfm']
    key='raw_config'if fault=='raw-old-plan'else'spec'if fault in('old-code','wrong-controller')else'entered'
    path=Path(item[key]['path']);value=json.loads(path.read_text())
    if fault=='raw-old-plan':value['supervised_quantum_control']['plan']=m.binding(spec)
    elif fault=='old-code':value['numerical_code_sha256']='0'*64
    elif fault=='wrong-controller':value['controller']=m.binding(spec)
    else:
        plan=m.read(value['plan'])
        if fault=='missing-snapshot-flag':plan.pop('snapshot_preparation')
        else:next(v for v in plan['models']if v['model']=='lfm')['commands']={'actor_command':['other']}
        replacement=tmp_path/'changed-plan.json';write(replacement,plan);value['plan']=m.binding(replacement)
    write(path,value);item[key]=m.binding(path);write(published,items)
    with pytest.raises(ValueError):m.reconcile(prepared_ref['path'],published,tmp_path/'forbidden')
    assert not(tmp_path/'forbidden').exists()


def test_missing_publication_waits_without_creating_current_proofs(tmp_path):
    spec,runtime,_=setup(tmp_path);prepared_ref=m.prepare(spec,tmp_path/'proposal');prepared=m.read(prepared_ref)
    published=publish(tmp_path,prepared,runtime);items=json.loads(published.read_text())
    Path(items['mellum']['entered']['path']).unlink()
    result=m.reconcile(prepared_ref['path'],published,tmp_path/'unused')
    assert result['status']=='waiting-for-normal-role-adoption' and result['roles']==['mellum']
    assert not(tmp_path/'unused').exists()


def test_host_relay_reuses_exact_remote_metadata_with_local_sdk_launch(tmp_path):
    spec,runtime,_=setup(tmp_path);prepared_ref=m.prepare(spec,tmp_path/'proposal');prepared=m.read(prepared_ref)
    published=publish(tmp_path,prepared,runtime)
    ready_ref=m.reconcile(prepared_ref['path'],published,tmp_path/'adopted');ready=m.read(ready_ref)
    from natlang_neuralese.common.hashing import sha256_hex
    store={}
    def remote(name,value):
        path='/work/native-production/fixture/'+name
        raw=json.dumps(value).encode();store[path]=raw
        return {'path':path,'sha256':sha256_hex(raw),'bytes':len(raw)}
    remote_relay=remote('relay.json',m.read(ready['relay']))
    remote_ready=remote('ready.json',{**ready,'relay':remote_relay})
    class IO:
        config=m.read(ready['relay'])['provider']
        def read(self,sid,path):assert sid=='mellum-existing';return store[path]
    staged=m.read(m.stage_host_relay(IO(),'mellum-existing',remote_ready,runtime,tmp_path/'sourcehost',sys.executable))
    assert staged['command'][1]==str(runtime/'scripts/coreweave/native_role_exchange.py')
    local=m.read(staged['relay']);remote=m.read(ready['relay'])
    assert local.pop('out')==str(tmp_path/'sourcehost/state')
    remote.pop('out');assert local==remote
    assert staged['activated'] is False


def test_only_real_owned_rows_replace_stale_peer_copies(tmp_path):
    spec,runtime,plans=setup(tmp_path);value=json.loads(spec.read_text())
    for role,item in value['roles'].items():
        inputs=m.read(item['inputs']);plan=m.read(inputs['bootstrap'])
        peer=next(row for row in plan['models']if row['model']!=role)
        peer['production_recipe']=m.binding(spec)  # stale peer is never authoritative
        peer['commands']={'learner_command':['obsolete-peer-command']}
        for assignment in plan['role_exchange']['roles'].values():
            assignment['verification_mechanism']={key:'b'*64 for key in assignment['verification_mechanism']}
        path=tmp_path/(role+'-new-base-plan.json');write(path,plan)
        inputs['bootstrap']=m.binding(path)
        ip=tmp_path/(role+'-new-normal-inputs.json');write(ip,inputs);item['inputs']=m.binding(ip)
    write(spec,value)
    prepared=m.read(m.prepare(spec,tmp_path/'proposal'))
    result=m.read(prepared['proposal']);actual={row['role']:m.read(row['successor'])for row in result['source_plan_successions']}
    for plan in actual.values():
        for row in plan['models']:
            original=next(x for x in m.read(plans[row['model']])['models']if x['model']==row['model'])
            assert row==original
        assert all(set(v['verification_mechanism'].values())=={'a'*64}for v in plan['role_exchange']['roles'].values())


def test_normal_role_cannot_silently_keep_old_actor_package(tmp_path):
    spec,runtime,_=setup(tmp_path);value=json.loads(spec.read_text())
    item=value['roles']['lfm'];inputs=m.read(item['inputs']);plan=m.read(inputs['bootstrap'])
    next(row for row in plan['models']if row['model']=='lfm')['actor_runtime']='old-package'
    path=tmp_path/'old-plan.json';write(path,plan);inputs['bootstrap']=m.binding(path)
    ip=tmp_path/'old-inputs.json';write(ip,inputs);item['inputs']=m.binding(ip);write(spec,value)
    with pytest.raises(ValueError,match='later runtime'):m.prepare(spec,tmp_path/'proposal')


def test_semantic_override_does_not_hide_unmeasured_raw_foundation(tmp_path):
    spec,runtime,plans=setup(tmp_path);value=json.loads(spec.read_text())
    inputs=m.read(value['roles']['mellum']['inputs']);plan=m.read(inputs['bootstrap'])
    row=next(row for row in plan['models']if row['model']=='mellum')
    recipe=m.read(row['production_recipe']);recipe['stages'][0]['parameters']['task_eval']['reference_successes']=None
    path=tmp_path/'unmeasured-recipe.json';write(path,recipe);row['production_recipe']=m.binding(path)
    pp=tmp_path/'unmeasured-plan.json';write(pp,plan);inputs['bootstrap']=m.binding(pp)
    ip=tmp_path/'unmeasured-inputs.json';write(ip,inputs);value['roles']['mellum']['inputs']=m.binding(ip);write(spec,value)
    with pytest.raises(ValueError,match='foundation recipe'):m.prepare(spec,tmp_path/'proposal')


@pytest.mark.parametrize('fault',[None,'source','base','review','unpassed','missing-source',
    'null-source','null-base-source','empty-source','null-review','stale-pyc'])
def test_explicit_host_snapshot_producer_preserves_numerical_package_and_binds_inputs(tmp_path,fault):
    spec,runtime,plans=setup(tmp_path)
    # This is the actual shared producer, loaded under its own source origin.
    from natlang_neuralese.train import native_deployment
    source=tmp_path/'snapshot-producer.py';source.write_bytes(Path(native_deployment.__file__).read_bytes())
    if fault=='stale-pyc':
        import os,py_compile
        body=source.read_bytes().replace(b'from __future__ import annotations\n',
            b'from __future__ import annotations\nHOST_SOURCE_REVISION=1\n')+b'\nassert HOST_SOURCE_REVISION==2\n'
        source.write_bytes(body)
        stat=source.stat();py_compile.compile(str(source),doraise=True)
        source.write_bytes(body.replace(b'HOST_SOURCE_REVISION=1',b'HOST_SOURCE_REVISION=2'))
        os.utime(source,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    base=runtime/'training/neuralese/natlang_neuralese/train/native_deployment.py'
    base.parent.mkdir(parents=True,exist_ok=True);base.write_text('BASE=1\n')
    record=json.loads((runtime/'construction.json').read_text())
    record['inventory']=code_inventory(runtime/'training/neuralese/natlang_neuralese')
    record['code']=canonical_json_sha256_hex(record['inventory'])
    record['files'][str(base.relative_to(runtime))]={'sha256':m.binding(base)['sha256'],'bytes':base.stat().st_size}
    write(runtime/'construction.json',record)
    request=json.loads(spec.read_text());request['composition']=m.binding(runtime/'construction.json')
    request['scheduler']['curriculum']=m.binding(Path(request['scheduler']['curriculum']['path']))
    curriculum=Path(request['scheduler']['curriculum']['path']);value=json.loads(curriculum.read_text())
    value['adaptive_refresh']={'updates_per_phase':16};write(curriculum,value)
    request['scheduler']['curriculum']=m.binding(curriculum)
    review=tmp_path/'source-review.json';write(review,{'schema':'natlang.native-scheduler-producer-review/1',
        'passed':True,'source':m.binding(source),'base_source':m.binding(base)})
    request['scheduling_producer']={'source':m.binding(source),'base_source':m.binding(base),'review':m.binding(review)}
    if fault=='source':source.write_text('BAD=1\n')
    elif fault=='base':request['scheduling_producer']['base_source']=m.binding(source)
    elif fault in ('review','unpassed','missing-source','null-source','null-base-source','empty-source','null-review'):
        value=json.loads(review.read_text())
        if fault=='review':value['source']=m.binding(base)
        elif fault=='missing-source':value.pop('source')
        elif fault=='null-source':value['source']=None
        elif fault=='null-base-source':value['base_source']=None
        elif fault=='empty-source':value['source']={}
        elif fault=='null-review':value=None
        else:value['passed']=False
        write(review,value);request['scheduling_producer']['review']=m.binding(review)
    write(spec,request)
    before=base.read_bytes()
    if fault not in (None,'stale-pyc'):
        with pytest.raises(ValueError):m.prepare(spec,tmp_path/'prepared')
    else:
        result=m.read(m.prepare(spec,tmp_path/'prepared'));proposal=m.read(result['proposal'])
        controller=m.read(proposal['controller']);snapshot=json.loads(Path(controller['curriculum']).read_text())
        assert 'adaptive_refresh'not in snapshot
        full=m.read(controller['snapshot_successor']['curriculum'])
        assert full['adaptive_refresh']=={'updates_per_phase':16}
        assert all(ref in result['operational_inputs']for ref in (*request['scheduling_producer'].values(),m.binding(m.__file__)))
        for item in result['normal_inputs'].values():
            assert all(ref in m.read(item)['frozen_refs']for ref in result['operational_inputs'])
    assert base.read_bytes()==before
