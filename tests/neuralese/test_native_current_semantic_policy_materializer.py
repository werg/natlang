import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts/coreweave'))
from prepare_native_scheduler_adoption import materialize_current_task_policy
from natlang_neuralese.common.jsonio import write_json_durable as write
from natlang_neuralese.train.native_deployment import binding
from natlang_neuralese.rl.native_role_release import EXCHANGE,MECHANISM_FILES


def source(tmp_path, *, cloud_packet=False):
    from natlang_neuralese.data.native_pair_schema import PAIR_SCHEMA
    train=tmp_path/'train.jsonl';held=tmp_path/'held.jsonl'
    for path,split in ((train,'train'),(held,'test')):
        path.write_text(json.dumps({'schema':PAIR_SCHEMA,'id':split+':pair','split':split,
            'source_groups':[split+':group'],'models':{'lfm:base':{},'mellum:base':{}}})+'\n')
    packets={}
    for role,floor in (('lfm',0),('mellum',2)):
        packet=tmp_path/(role+'-protected.jsonl');packet.write_text(role+' protected rows\n')
        local=binding(packet)
        declared=({'path':'/work/resume/protected.jsonl','sha256':local['sha256']} if cloud_packet and role=='mellum' else local)
        packets[role]={'declared':declared,'local':local}
    curriculum=tmp_path/'curriculum.json'
    write(curriculum,{'schema':'natlang.native-semantic-training/1','backbone_lr':1e-5,'vae_lr':1e-4,
        'latent_width':8,'hidden_width':16,'checkpoint_every':4,'eval_every':2,
        'adaptive_refresh':{'updates_per_phase':16},'pairs':binding(train),'held_pairs':binding(held),
        'held_eval_rows':1,'phases':[{'id':kind,'kind':kind,'epochs':1}for kind in ('vae','joint','distill')]})
    plans={};configs={};relocations={}
    assignments={role:{'owned_models':[role],'source_host_id':role+'-host',
        'verification_mechanism':{name:'a'*64 for name in MECHANISM_FILES}}for role in ('lfm','mellum')}
    for role,floor in (('lfm',0),('mellum',2)):
        models=[{'model':name,'native_participant':name+':base',
            'handoff_requirement':{'model_id':name+':base'}}for name in ('lfm','mellum')]
        plan=tmp_path/(role+'-plan.json')
        write(plan,{'schema':'natlang.native-raw-rl-bootstrap/1','models':models,'owned_models':[role],
            'role_exchange':{'schema':EXCHANGE,'role':role,'roles':assignments,
                'directory':str(tmp_path/'exchange'),'semantic_consumer':'mellum'}})
        plans[role]=binding(plan)
        task={'packet':packets[role]['declared'],'cases':[0],'reference_successes':floor,
            'reference_provenance':role+' current config baseline'}
        config=tmp_path/(role+'-config.json')
        write(config,{'schema':'natlang.rl-production-config/1','trainer_parameters':{'task_eval':task}})
        configs[role]=str(config)
        if packets[role]['declared']!=packets[role]['local']:
            relocations[role]={'source':packets[role]['declared'],'local':packets[role]['local']}
    return curriculum,plans,configs,relocations,train,held


def test_materializes_both_current_task_policies_and_exact_existing_policy_cadence(tmp_path):
    curriculum,plans,configs,relocations,train,held=source(tmp_path,cloud_packet=True)
    result=materialize_current_task_policy(curriculum,plans,configs,'mellum',tmp_path/'out',relocations)
    prepared=json.loads(Path(result['path']).read_text())
    value=json.loads(Path(prepared['curriculum']['path']).read_text())
    assert value['task_eval']==value['models']['mellum:base']['task_eval']
    assert value['models']['lfm:base']['task_eval']['reference_successes']==0
    assert value['models']['mellum:base']['task_eval']['reference_successes']==2
    assert value['models']['mellum:base']['task_eval']['packet']['path']=='/work/resume/protected.jsonl'
    assert value['operator_bindings']['current_roles']['mellum']['task_packet']['relocation']['local']['sha256']==\
        value['models']['mellum:base']['task_eval']['packet']['sha256']
    assert value['policy_schedule']=={'id':'policy','epochs':1,'every_semantic_updates':1,'updates_per_refresh':16}
    assert value['pairs']==binding(train) and value['held_pairs']==binding(held)
    assert prepared['activated'] is False and prepared['readiness_granted'] is False


def test_materializer_rejects_unmeasured_floor_and_wrong_relocated_packet(tmp_path):
    curriculum,plans,configs,relocations,*_=source(tmp_path,cloud_packet=True)
    config=Path(configs['lfm']);value=json.loads(config.read_text())
    value['trainer_parameters']['task_eval']['reference_successes']=None;write(config,value)
    with pytest.raises(ValueError,match='no measured/explicit task floor'):
        materialize_current_task_policy(curriculum,plans,configs,'mellum',tmp_path/'bad-floor',relocations)
    second=tmp_path/'second';second.mkdir()
    curriculum,plans,configs,relocations,*_=source(second,cloud_packet=True)
    relocations['mellum']['local']['sha256']='0'*64
    with pytest.raises(ValueError):
        materialize_current_task_policy(curriculum,plans,configs,'mellum',tmp_path/'bad-relocation',relocations)


def test_mellum_task_policy_resolves_from_exact_staged_recipe(tmp_path):
    curriculum,plans,configs,relocations,*_=source(tmp_path,cloud_packet=True)
    config_path=Path(configs['mellum']);config=json.loads(config_path.read_text())
    task=config['trainer_parameters'].pop('task_eval')
    recipe=tmp_path/'mellum-recipe.json'
    write(recipe,{'stages':[{'id':'native_full_sft','kind':'native_state_sft',
        'parameters':{'task_eval':task}}]})
    config['production_stage']='native_full_sft'
    config['trainer_parameters']['recipe_binding']={'path':'/work/resume/recipe.json',
        'sha256':binding(recipe)['sha256']}
    write(config_path,config)
    result=materialize_current_task_policy(curriculum,plans,configs,'mellum',tmp_path/'out',
        relocations,{'mellum':str(recipe)})
    prepared=json.loads(Path(result['path']).read_text())
    value=json.loads(Path(prepared['curriculum']['path']).read_text())
    assert value['models']['mellum:base']['task_eval']['reference_successes']==2
    assert prepared['task_eval_sources']['mellum']['recipe']['declared']['path']=='/work/resume/recipe.json'


def test_local_task_packet_can_be_staged_to_the_cloud_consumer(tmp_path):
    curriculum,plans,configs,relocations,*_=source(tmp_path)
    packet=json.loads(Path(configs['lfm']).read_text())['trainer_parameters']['task_eval']['packet']
    local=binding(packet['path'],packet)
    relocations['lfm']={'source':packet,'local':local,'target':'/work/native-production/tasks/lfm-protected.jsonl'}
    result=materialize_current_task_policy(curriculum,plans,configs,'mellum',tmp_path/'out',relocations)
    prepared=json.loads(Path(result['path']).read_text())
    value=json.loads(Path(prepared['curriculum']['path']).read_text())
    assert value['models']['lfm:base']['task_eval']['packet']['path']=='/work/native-production/tasks/lfm-protected.jsonl'
    assert prepared['staged_task_packets']==[{'role':'lfm','source':local,
        'target':'/work/native-production/tasks/lfm-protected.jsonl'}]
