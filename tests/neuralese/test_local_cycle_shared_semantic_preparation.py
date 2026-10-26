"""Fresh shared-stage declaration plumbing; fixtures grant no readiness."""
import json
import sys
from pathlib import Path
import pytest
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.train.native_deployment import binding
from scripts.coreweave.prepare_local_cycle_shared_semantic import build_role_declarations


def source_rows(tmp_path):
    result={}
    for role in ('lfm','mellum'):
        recipe=tmp_path/f'{role}-recipe.json'
        cases=tmp_path/f'{role}-cases.jsonl';cases.write_text('{"split":"train"}\n')
        seed=tmp_path/f'{role}-seed.jsonl';seed.write_text('{"graph":"train"}\n')
        held=tmp_path/f'{role}-held.jsonl';held.write_text('{"graph":"held"}\n')
        foundation_rows=tmp_path/f'{role}-foundation.jsonl';foundation_rows.write_text('{"graph":"foundation"}\n')
        foundation=tmp_path/f'{role}-foundation.json'
        foundation_source=(str(foundation_rows) if role=='lfm' else '/work/resume/inputs/mellum-foundation.jsonl')
        write(foundation,{'schema':'natlang.native-graph-sources/1','sources':[{'kind':'semantic-pairs',
            'path':foundation_source,'sha256':binding(foundation_rows)['sha256']}]})
        recipe_value={'schema':'natlang.production-recipe/1','id':role+'-recipe','input_bindings':{
            'train':{'path':str(seed),'sha256':binding(seed)['sha256']},
            'held':{'path':str(held),'sha256':binding(held)['sha256']},
            'foundation':{'path':str(foundation),'sha256':binding(foundation)['sha256']}},
            'stages':[{'id':'native_full_sft','kind':'native_state_sft','inputs':{'native_graphs':'train',
                'held_native_graphs':'held','foundation_native_graphs':'foundation'},
                'parameters':{'communication_foundation':True,'epochs':1.,'forward_atol':1e-4,'gradient_atol':1e-5}}]}
        write(recipe,recipe_value)
        participant=('lfm2' if role=='lfm' else 'mellum')+':base-'+role
        checkpoint_file=tmp_path/f'{role}-checkpoint.pt';checkpoint_file.write_bytes((role+' checkpoint').encode())
        checkpoint=binding(checkpoint_file)['sha256'];receipt=tmp_path/f'{role}-readiness.json'
        ready=role=='lfm'
        foundation_receipt={'schema':'natlang.native-communication-foundation/1','complete':True,
            'epochs':1.,'graph_visits':1,'required_graph_visits':1,
            'sources':{'sha256':binding(foundation)['sha256']}}
        write(receipt,{'schema':'natlang.native-training-readiness/1','model_id':participant,
            'stage_id':'native_full_sft','native_identity':{'checkpoint_sha256':checkpoint},
            'checkpoint':{'sha256':checkpoint},'recipe':{'sha256':binding(recipe)['sha256']},
            'foundation_ready':ready,'communication_foundation':foundation_receipt if ready else None})
        remote_paths={str(recipe):'/work/native-production/mellum/recipe.json',
            str(cases):'/work/native-production/mellum/cases.jsonl',
            str(seed):'/work/native-production/mellum/seed.jsonl',
            str(held):'/work/native-production/mellum/held.jsonl',
            str(foundation):'/work/native-production/mellum/foundation.json',
            str(foundation_rows):foundation_source}
        config=tmp_path/f'{role}-config.json';write(config,{'model':role,'base':str(tmp_path/f'{role}-base'),
            'runtime':str(tmp_path/'ts-host'),
            'production_recipe':(remote_paths[str(recipe)]if role=='mellum'else str(recipe)),
            'production_stage':'native_full_sft',
            'cases':(remote_paths[str(cases)]if role=='mellum'else str(cases)),
            'seed_native_graphs':(remote_paths[str(seed)]if role=='mellum'else str(seed)),
            'held_native_graphs':(remote_paths[str(held)]if role=='mellum'else str(held))})
        state=tmp_path/f'{role}-state.json';cursor=tmp_path/f'{role}-cursor.json'
        write(state,{'schema':'natlang.rl-production-state/1','checkpoint':str(checkpoint_file)})
        write(cursor,{'checkpoint':{'path':str(checkpoint_file),'sha256':checkpoint},'cycle':1,'phase':'rl'})
        result[role]={'config':binding(config),'source_receipt':binding(receipt),
            'owner_state':binding(state),'owner_cursor':binding(cursor),
            'readiness':str(tmp_path/role/'qualified-native-training-receipt.json'),'recipe':binding(recipe),
            'recipe_target':remote_paths[str(recipe)]if role=='mellum'else str(recipe),
            'source_host_id':role+'-owner','provider':{'sandbox_id':role+'-owner'},
            'provider_queue':str(tmp_path/'bridge'),'port':8097 if role=='lfm' else 8098}
        if role=='mellum':
            source_map=tmp_path/'mellum-source-map.json';write(source_map,{'schema':'natlang.staged-source-map/1',
                'paths':remote_paths})
            result[role]['source_map']=binding(source_map)
    return result


def test_builds_two_fresh_bootstrap_forms_without_claiming_readiness(tmp_path):
    raw,standard,qualification,deployed=build_role_declarations(source_rows(tmp_path))
    assert [row['model']for row in raw]==['lfm','mellum']
    assert [row['model']for row in standard]==['lfm','mellum']
    assert all('handoff_requirement'in row for row in raw)
    assert all('handoff_requirement'not in row for row in standard)
    assert all(row['foundation_readiness_policy']=='on-shared-request' for row in raw)
    assert all(len(row['foundation_qualification_generation'])==64 for row in raw)
    assert all(row['native_participant']==row['handoff_requirement']['model_id'] for row in raw)
    assert qualification['lfm']['foundation_ready'] is False
    assert qualification['lfm']['source_receipt_foundation_ready'] is True
    assert qualification['mellum']['foundation_ready'] is False
    assert all(Path(row['seed_native_graphs']['path']).is_file() for row in raw)
    assert raw[1]['source_identity']['source_map']['sha256']==source_rows(tmp_path)['mellum']['source_map']['sha256']
    assert deployed['mellum']['production_recipe']['path']=='/work/native-production/mellum/recipe.json'
    assert deployed['mellum']['seed_native_graphs']['path']=='/work/native-production/mellum/seed.jsonl'
    assert (deployed['mellum']['handoff_requirement']['foundation_sources']['path']==
        '/work/native-production/mellum/foundation.json')


@pytest.mark.parametrize('fault',['participant','foundation','trainer-recipe','qualification','owner-cursor','source-map'])
def test_fresh_declarations_reject_unbound_or_mismatched_foundation_identity(tmp_path,fault):
    inputs=source_rows(tmp_path)
    if fault=='participant':
        value=json.loads(Path(inputs['lfm']['source_receipt']['path']).read_text());value['model_id']='mellum:other';write(inputs['lfm']['source_receipt']['path'],value)
        inputs['lfm']['source_receipt']=binding(inputs['lfm']['source_receipt']['path'])
    elif fault=='foundation':
        value=json.loads(Path(inputs['mellum']['recipe']['path']).read_text());value['stages'][0]['parameters'].pop('communication_foundation');write(inputs['mellum']['recipe']['path'],value)
        inputs['mellum']['recipe']=binding(inputs['mellum']['recipe']['path'])
    elif fault=='trainer-recipe':inputs['mellum']['recipe']['declared_path']='old-recipe.json'
    elif fault=='owner-cursor':
        value=json.loads(Path(inputs['mellum']['owner_cursor']['path']).read_text())
        value['checkpoint']['path']='/tmp/other-checkpoint.pt'
        write(inputs['mellum']['owner_cursor']['path'],value)
        inputs['mellum']['owner_cursor']=binding(inputs['mellum']['owner_cursor']['path'])
    elif fault=='source-map':
        value=json.loads(Path(inputs['mellum']['source_map']['path']).read_text())
        source=next(path for path,target in value['paths'].items()if target=='/work/resume/inputs/mellum-foundation.jsonl')
        value['paths'][str(Path(inputs['mellum']['config']['path']).parent/'mellum-held.jsonl')]=value['paths'].pop(source)
        write(inputs['mellum']['source_map']['path'],value)
        inputs['mellum']['source_map']=binding(inputs['mellum']['source_map']['path'])
    else:
        value=json.loads(Path(inputs['lfm']['source_receipt']['path']).read_text());value['native_identity']['checkpoint_sha256']='b'*64;write(inputs['lfm']['source_receipt']['path'],value)
        inputs['lfm']['source_receipt']=binding(inputs['lfm']['source_receipt']['path'])
    with pytest.raises((ValueError,KeyError)):
        build_role_declarations(inputs)


def test_full_preparation_materializes_current_two_host_launch_and_boundary_plan(tmp_path):
    from natlang_neuralese.data.native_pair_schema import PAIR_SCHEMA
    from scripts.coreweave.prepare_local_cycle_shared_semantic import prepare
    inputs=source_rows(tmp_path)
    runtime=tmp_path/'runtime';package=runtime/'training/neuralese/natlang_neuralese'
    (package/'train').mkdir(parents=True);(package/'train/native_readiness.py').write_text('# fixture runtime\n')
    (runtime/'scripts').mkdir(parents=True);(runtime/'scripts/run_bonsai_queue.py').write_text('# exact TaskEval collector fixture\n')
    controller=tmp_path/'controller.py';controller.write_text('def load_controls(spec): return spec\n')
    train=tmp_path/'train-pairs.jsonl';held=tmp_path/'held-pairs.jsonl'
    for path,split in ((train,'train'),(held,'test')):
        path.write_text(json.dumps({'schema':PAIR_SCHEMA,'id':split+':pair','split':split,
            'source_groups':[split+':group'],'models':{'lfm:base-lfm':{},'mellum:base-mellum':{}}})+'\n')
    curriculum=tmp_path/'curriculum.json'
    write(curriculum,{'schema':'natlang.native-semantic-training/1','backbone_lr':1e-5,'vae_lr':1e-4,
        'latent_width':8,'hidden_width':16,'checkpoint_every':2,'eval_every':1,
        'adaptive_refresh':{'updates_per_phase':16},'pairs':binding(train),'held_pairs':binding(held),
        'phases':[{'id':kind,'kind':kind,'epochs':1}for kind in ('vae','joint','distill')]})
    local_id='dgx-local';cloud_id='mellum-current'
    provider={'coordinator_id':cloud_id,'receipts':str(tmp_path/'receipts'),
        'sandboxes':{cloud_id:{'runtime':str(runtime),'write_roots':['/work/native-production',str(runtime)],
            'read_roots':['/work/resume/inputs']}},
        'local_source_host':{'source_host_id':local_id,'runtime':str(runtime),'python':sys.executable,
            'read_roots':[str(tmp_path)],'write_roots':[str(tmp_path)]}}
    provider_path=tmp_path/'provider.json';write(provider_path,provider)
    role_inputs={}
    for role,host in (('lfm',local_id),('mellum',cloud_id)):
        row=inputs[role]
        config_path=tmp_path/f'{role}-config.json';cycle_path=tmp_path/f'{role}-cycle.json'
        control_path=tmp_path/f'{role}-control.json';config_out=(str(tmp_path/f'{role}/production')if role=='lfm'
            else'/work/native-production/mellum/production')
        config=json.loads(Path(row['config']['path']).read_text())
        config.update({'runtime':str(runtime/'ts-host'),'out':config_out,
            'actor_command':[sys.executable],'actor_environment':{'NATLANG_REPO':str(runtime)},
            'native_actor_mode':'raw-state','trainer_parameters':{'task_eval':{'packet':binding(tmp_path/f'{role}-packet.json') if (tmp_path/f'{role}-packet.json').exists() else binding(tmp_path/f'{role}-cases.jsonl'),
                'cases':[0],'reference_successes':0,'reference_provenance':'current owner fixture'}}})
        (tmp_path/f'{role}-packet.json').write_text('protected test packet\n')
        config['trainer_parameters']['task_eval']['packet']=binding(tmp_path/f'{role}-packet.json')
        write(config_path,config)
        row['config']=binding(config_path)
        cycle_out=(str(tmp_path/f'{role}/cycles')if role=='lfm'else'/work/native-production/mellum/cycles')
        cfg_target=str(config_path)if role=='lfm'else'/work/native-production/mellum/config.json'
        write(cycle_path,{'schema':'natlang.local-raw-training-plan/1','out':cycle_out,'rl_config':cfg_target,
            'optimizer_owner_lock':str(tmp_path/f'{role}/owner.lock'),'supervised_quantum_updates':250,
            'supervised_stage_quantum_updates':{'native_full_sft':250}})
        write(control_path,{'model':role,'controller':binding(controller),'numerical_package':str(package)})
        target_root=(str(tmp_path/f'{role}')if role=='lfm'else'/work/native-production/mellum')
        row.update({'source_host_id':host,'config_target':cfg_target,
            'cycle_plan':binding(cycle_path),'cycle_plan_target':str(Path(target_root)/'cycles/plan.json'),
            'state_path':config_out+'/state.json','cursor_path':cycle_out+'/cursor.json',
            'readiness':(str(Path(target_root)/'qualified/native-training-receipt.json')if role=='mellum'
                else row['readiness']),
            'controls':binding(control_path),
            'controls_target':str(Path(target_root)/'controls.json'),'plan_target':str(Path(target_root)/'role-plan.json'),
            'successor_plan_target':str(Path(target_root)/'successor-role-plan.json'),
            'owner_identity_path':cycle_out+'/owner-identity.json','runtime':str(runtime),'python':sys.executable})
        role_inputs[role]=row
    derived_models,_,_,_=build_role_declarations(role_inputs)
    model_by_role={row['model']:row for row in derived_models}
    task_packets={}
    for role in ('lfm','mellum'):
        packet=model_by_role[role]
        config=json.loads(Path(role_inputs[role]['config']['path']).read_text())
        packet_ref=config['trainer_parameters']['task_eval']['packet']
        task_packets[role]={'source':packet_ref,'local':binding(packet_ref['path'],packet_ref),
            'target':f'/work/native-production/tasks/{role}-protected.jsonl'}
    graph_relocations={}
    curriculum_value=json.loads(curriculum.read_text())
    for key in ('pairs','held_pairs'):
        graph_relocations[key]={'source':curriculum_value[key],'local':binding(curriculum_value[key]['path']),
            'target':f'/work/native-production/shared/graphs/{key}.jsonl'}
    spec={'schema':'natlang.local-cycle-shared-semantic-input/1','runtime':str(runtime),
        'provider':binding(provider_path),'curriculum':binding(curriculum),'roles':role_inputs,'consumer':'mellum',
        'task_packet_relocations':task_packets,'curriculum_graph_relocations':graph_relocations,
        'exchange_directory':'/work/native-production/shared/exchange',
        'semantic_consumer_out':'/work/native-production/shared/semantic',
        'semantic_curriculum_target':'/work/native-production/shared/semantic/full-curriculum.json',
        'snapshot_curriculum_target':'/work/native-production/shared/semantic/snapshot-curriculum.json',
        'standard_bootstrap_target':'/work/native-production/shared/semantic/rl-bootstrap.json',
        'controller_target':'/work/native-production/shared/semantic/controller.json',
        'handoff_lease_seconds':180}
    spec_path=tmp_path/'spec.json';write(spec_path,spec)
    prepared=prepare(spec_path,tmp_path/'prepared')
    assert prepared['activated'] is False and prepared['readiness_granted'] is False
    assert prepared['qualification']['lfm']['foundation_ready'] is False
    assert prepared['qualification']['lfm']['source_receipt_foundation_ready'] is True
    assert prepared['qualification']['mellum']['foundation_ready'] is False
    assert prepared['commands']['lfm_cycle']['mode']=='local-frozen-runtime-subprocess'
    assert prepared['commands']['mellum_cycle']['mode']=='existing-cloud-provider-exec'
    assert prepared['owner_handoffs']['lfm']['applied'] is False
    assert any(item['kind']=='snapshot-curriculum' for item in prepared['stage_files'])
    assert sum(item['kind']=='task-eval-python-collector' for item in prepared['stage_files'])==2
    assert sum(item['kind']=='task-eval-packet' for item in prepared['stage_files'])==2
    assert sum(item['kind'] in ('paired-pairs','paired-held_pairs') for item in prepared['stage_files'])==2
    consumer_curriculum=next(item['source']for item in prepared['stage_files']if item['kind']=='full-paired-curriculum')
    consumer_value=json.loads(Path(consumer_curriculum['path']).read_text())
    assert consumer_value['pairs']['path']=='/work/native-production/shared/graphs/pairs.jsonl'
    assert consumer_value['held_pairs']['path']=='/work/native-production/shared/graphs/held_pairs.jsonl'
    (runtime/'scripts/run_bonsai_queue.py').unlink()
    with pytest.raises(ValueError,match='TaskEval Python collector'):
        prepare(spec_path,tmp_path/'missing-collector-prepared')
