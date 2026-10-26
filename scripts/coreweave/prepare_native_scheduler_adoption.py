#!/usr/bin/env python3
"""Prepare normal role adoption, then reconcile real published scheduler inputs.

Metadata only: no SDK, owner signaling, checkpoint decoding or model allocation.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys

REPOSITORY=Path(os.environ.get('NATLANG_REPO',Path(__file__).resolve().parents[2])).resolve()
sys.path[:0]=[str(REPOSITORY/'training/neuralese'),str(REPOSITORY/'scripts/coreweave')]
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.hashing import sha256_hex,canonical_json_sha256_hex
from natlang_neuralese.train.native_deployment import binding, prepare_role_scheduler
from natlang_neuralese.rl import native_role_release as roles
from natlang_neuralese.train.recipe import resolve_recipe_data,effective_stage_parameters
from natlang_neuralese.eval.task_cadence import parse_task_eval
from compose_native_actor_runtime import bound
from verify_native_actor_runtime import verify_files

SCHEMA='natlang.native-scheduler-adoption/1'


def read(ref):
    binding(ref['path'], ref)
    return json.loads(Path(ref['path']).read_bytes())


def emit(path,value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists() and json.loads(path.read_bytes())!=value:
        raise ValueError('immutable scheduler adoption output changed: '+str(path))
    write_json_durable(path,value)
    return binding(path)


def scheduling_producer(spec):
    selected=spec.get('scheduling_producer')
    if selected is None:return prepare_role_scheduler,[]
    if not isinstance(selected,dict)or set(selected)!={'source','base_source','review'}:
        raise ValueError('scheduler producer needs exact source/base/review bindings')
    source=bound(selected['source']);base=bound(selected['base_source']);review=read(selected['review'])
    binding(source,selected['source']);binding(base,selected['base_source'])
    actual=Path(spec['scheduler']['runtime'])/'training/neuralese/natlang_neuralese/train/native_deployment.py'
    if binding(base)!=binding(actual):raise ValueError('scheduler producer base differs from actual runtime')
    if (not isinstance(review,dict)
            or not {'schema','passed','source','base_source'}<=set(review)
            or review.get('schema')!='natlang.native-scheduler-producer-review/1' or review.get('passed')is not True
            or any(not isinstance(review.get(key),dict) or not {'path','sha256'}<=set(review[key])
                for key in ('source','base_source'))
            or binding(source,review.get('source'))!=binding(source,selected['source'])
            or binding(base,review.get('base_source'))!=binding(base,selected['base_source'])):
        raise ValueError('scheduler producer lacks its exact reviewed source/base authority')
    # A separately named host producer; never overwrite a loaded runtime module
    # or claim that its file belongs to the numerical composition inventory.
    import importlib.util
    name='natlang_neuralese.train._reviewed_scheduler_'+selected['source']['sha256']
    module_spec=importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(module_spec)
    raw=source.read_bytes()
    if sha256_hex(raw)!=selected['source']['sha256']:
        raise ValueError('scheduler producer source changed before execution')
    exec(compile(raw,str(source),'exec'),module.__dict__)
    return module.prepare_role_scheduler,[selected['source'],selected['base_source'],selected['review'],binding(__file__)]


def scheduler(spec,plans,controls,out,bootstrap):
    common=spec['scheduler']
    for name in ('curriculum','provider'):read(common[name])
    producer,inputs=scheduling_producer(spec)
    result=producer(plans,controls,common['curriculum']['path'],common['provider']['path'],out,
        raw_states={role:item['raw_state']for role,item in spec['roles'].items()},
        runtime=common['runtime'],python=common['python'],rl_bootstrap=bootstrap,
        handoff_lease_seconds=common['handoff_lease_seconds'],snapshot_parent=common.get('snapshot_parent'))
    if inputs:
        result['operational_inputs']=list({item['path']:item for item in
            [*result.get('operational_inputs',[]),*inputs]}.values())
        write_json_durable(out/'prepared.json',result)
    return result


def owned_bootstrap(reference,plans):
    """Retain RL knobs; use each normal source's own authenticated inputs."""
    bootstrap=read(reference)
    if {row['model']for row in bootstrap['models']}!=set(plans):
        raise ValueError('whole-graph bootstrap omitted an actual source owner')
    for row in bootstrap['models']:
        actual=next(value for value in read(plans[row['model']])['models']if value['model']==row['model'])
        for key in ('native_participant','production_recipe','cases','seed_native_graphs','held_native_graphs'):
            if key=='native_participant' and row[key]!=actual[key]:
                raise ValueError('normal succession substituted its RL participant identity')
            row[key]=copy.deepcopy(actual[key])
    return bootstrap


def check_foundation_task_policy(row,curriculum):
    """Joint overrides alone cannot repair a null RAW foundation reference."""
    model=row['handoff_requirement']['model_id']
    selected=parse_task_eval(curriculum['models'][model]['task_eval'])
    recipe=resolve_recipe_data(bound(row['production_recipe']))
    stages={stage['id']:stage for stage in recipe['stages']}
    for name in row['handoff_requirement']['stage_ids']:
        actual=parse_task_eval(effective_stage_parameters(stages[name],[],recipe).get('task_eval'))
        if (actual is None or actual['reference_successes'] is None
                or any(actual.get(key)!=selected.get(key)for key in
                       ('reference_successes','reference_provenance','cases','prompt_rendering'))
                or actual['packet']['sha256']!=selected['packet']['sha256']):
            raise ValueError('normal foundation recipe must adopt its selected actual task reference: '+row['model'])


def materialize_current_task_policy(curriculum_path,plan_refs,config_paths,consumer,out,
                                    task_packets=None,recipe_paths=None):
    """Bind semantic task checks and policy cadence to the two current owners.

    The paired graph curriculum remains authoritative for phases, optimizer
    routing, data, held rows and adaptive refresh. Only task-eval declarations
    are copied from each actual current trainer config. The whole-graph policy
    schedule is the existing native-deployment default, recorded explicitly.
    """
    if set(plan_refs)!=set(config_paths) or set(plan_refs)!={'lfm','mellum'} or consumer not in plan_refs:
        raise ValueError('semantic task materialization requires both current owners and its consumer')
    curriculum_ref=binding(curriculum_path)
    value=read(curriculum_ref)
    if value.get('schema')!='natlang.native-semantic-training/1':
        raise ValueError('task materialization needs the current paired semantic curriculum')
    models={}
    task_policies={};task_packets=task_packets or {}
    recipe_paths=recipe_paths or {}
    if not isinstance(task_packets,dict)or set(task_packets)-{'lfm','mellum'}:
        raise ValueError('task packet relocation map names an unknown model role')
    if not isinstance(recipe_paths,dict)or set(recipe_paths)-{'lfm','mellum'}:
        raise ValueError('recipe relocation map names an unknown model role')
    for role in ('lfm','mellum'):
        plan_ref=plan_refs[role];plan=read(plan_ref)
        owned,exchange=roles.ownership(plan)
        if exchange is None or exchange.get('role')!=role or len(owned)!=1:
            raise ValueError('task materialization needs the exact current single-owner role plan')
        participant=next(row for row in plan['models'] if row['model'] in owned)
        model_id=participant.get('handoff_requirement',{}).get('model_id')
        if not isinstance(model_id,str) or not model_id:
            raise ValueError('current role plan omitted its exact participant identity')
        config_ref=binding(config_paths[role]);config=read(config_ref)
        parameters=config.get('trainer_parameters') or {}
        # Local-cycle owners may carry the active task policy in the
        # continuation/seed parameter block while the resident trainer block
        # records only the original learner defaults. Preserve the policy the
        # source owner actually launches; do not synthesize cases or floors.
        seed_parameters=config.get('seed_trainer_parameters') or {}
        declaration=(parameters.get('task_eval') or config.get('task_eval')
            or seed_parameters.get('task_eval'))
        recipe_evidence=None
        if not declaration:
            recipe_decl=parameters.get('recipe_binding') or seed_parameters.get('recipe_binding')
            if not isinstance(recipe_decl,dict):
                raise ValueError('current '+role+' config omitted the task policy and exact recipe binding')
            recipe_path=recipe_paths.get(role)
            if recipe_path is None:
                if Path(recipe_decl['path']).is_file():recipe_path=recipe_decl['path']
                else:raise ValueError('current '+role+' frozen recipe bytes need an explicit staged path')
            actual_recipe=binding(recipe_path)
            if actual_recipe['sha256']!=recipe_decl['sha256']:
                raise ValueError('current '+role+' staged recipe differs from its exact config binding')
            recipe=json.loads(Path(actual_recipe['path']).read_bytes())
            stage_id=(config.get('production_stage') or parameters.get('stage_id')
                or seed_parameters.get('stage_id'))
            stages=[stage for stage in recipe.get('stages',[])if stage.get('id')==stage_id
                and stage.get('kind')=='native_state_sft']
            if len(stages)!=1:
                raise ValueError('current '+role+' recipe omitted its exact active native-state stage')
            declaration=effective_stage_parameters(stages[0],[],recipe).get('task_eval')
            if not declaration:
                raise ValueError('current '+role+' native-state recipe stage omitted task evaluation')
            recipe_evidence={'declared':recipe_decl,'local':actual_recipe}
        if not declaration:
            raise ValueError('current '+role+' config omitted its actual task-eval policy')
        from natlang_neuralese.eval.task_cadence import parse_task_eval
        selected=parse_task_eval(copy.deepcopy(declaration))
        # A measured reference is useful when the current owner has one. Some
        # legitimate recovery configs declare only an absolute min_successes
        # floor; preserve that exact policy and leave the reference unmeasured.
        # The evaluator keeps such runs ineligible for reference-based task
        # qualification, so this never manufactures a capability baseline.
        explicit_min=declaration.get('min_successes')
        explicit_floor=(type(explicit_min)is int and explicit_min>=0)
        if selected.get('reference_successes') is None and not explicit_floor:
            raise ValueError('current '+role+' task policy has neither a measured reference nor an explicit minimum-success floor')
        packet=selected['packet'];packet_evidence={'declared':copy.deepcopy(packet)}
        relocation=task_packets.get(role)
        if relocation is not None:
            if (not isinstance(relocation,dict)or set(relocation)-{'source','local','target'}
                    or not {'source','local'}<=set(relocation)
                    or relocation['source']!=packet):
                raise ValueError('current '+role+' task packet relocation changed its declared source binding')
            actual=binding(relocation['local']['path'],relocation['local'])
            if actual['sha256']!=packet['sha256']:
                raise ValueError('current '+role+' relocated task packet differs from its exact declared SHA')
            packet_evidence['relocation']={'source':relocation['source'],'local':actual}
            if 'target'in relocation:
                target=relocation['target']
                if not isinstance(target,str)or not Path(target).is_absolute()or '..'in Path(target).parts:
                    raise ValueError('task packet consumer target must be an exact absolute path')
                selected['packet']={**selected['packet'],'path':target}
                packet_evidence['relocation']['target']=target
        else:
            checked=binding(packet['path'],packet)
            if any(checked.get(key)!=packet.get(key)for key in ('path','sha256')):
                raise ValueError('current '+role+' task packet binding changed')
            packet_evidence['actual']=checked
        models[model_id]={'task_eval':selected}
        task_policies[role]={'plan':plan_ref,'config':config_ref,'participant':model_id,
            'task_eval':selected,'task_eval_sha256':canonical_json_sha256_hex(selected),
            'task_packet':packet_evidence,**({'recipe':recipe_evidence}if recipe_evidence else {})}
    if len(models)!=2:
        raise ValueError('current role plans do not name two distinct semantic participants')
    value['models']={**(value.get('models')or{}),**models}
    value['task_eval']=copy.deepcopy(models[task_policies[consumer]['participant']]['task_eval'])
    if not value.get('policy_schedule'):
        refresh=value.get('adaptive_refresh')or{}
        updates=refresh.get('updates_per_phase')
        if type(updates)is not int or updates<=0:
            raise ValueError('paired curriculum omitted its existing positive adaptive policy quantum')
        value['policy_schedule']={'id':'policy','epochs':1,'every_semantic_updates':1,
            'updates_per_refresh':updates}
    else:
        updates=(value.get('adaptive_refresh')or{}).get('updates_per_phase')
        if updates is not None and value['policy_schedule'].get('updates_per_refresh')!=updates:
            raise ValueError('existing policy cadence conflicts with the paired curriculum refresh quantum')
    value['operator_bindings']={'schema':'natlang.native-semantic-current-policy-bindings/1',
        'base_curriculum':curriculum_ref,'consumer':consumer,'current_roles':task_policies,
        'policy_schedule_source':'native-deployment-default-v1'}
    from natlang_neuralese.train.native_semantic_stage import validate_config
    validate_config(value)
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('semantic policy materialization needs a new immutable namespace')
    out.mkdir(parents=True)
    target=out/'curriculum.json';write_json_durable(target,value)
    return emit(out/'prepared.json',{'schema':'natlang.native-current-semantic-policy/1',
        'curriculum':binding(target),'inputs':{'base_curriculum':curriculum_ref,
            'plans':{key:binding(plan_refs[key]['path'],plan_refs[key])for key in plan_refs},
            'configs':{key:binding(config_paths[key])for key in config_paths}},
        'consumer':consumer,'task_eval_sources':task_policies,
        'policy_schedule':value['policy_schedule'],'owner_changes':0,'readiness_granted':False,
        'staged_task_packets':[{'role':role,'source':proof['task_packet']['relocation']['local'],
            'target':proof['task_packet']['relocation']['target']}
            for role,proof in task_policies.items()if 'target'in proof['task_packet'].get('relocation',{})],
        'activated':False})


def prepare(spec_path,out):
    spec=json.loads(Path(spec_path).read_bytes())
    if (set(spec)-{'scheduling_producer'}!={'schema','composition','roles','scheduler'} or spec['schema']!=SCHEMA
            or set(spec['roles'])!={'lfm','mellum'}):
        raise ValueError('scheduler adoption requires both explicit normal role inputs')
    construction=bound(spec['composition'])
    record=verify_files(construction.parent)
    if record['actor_runtime']!=spec['scheduler']['runtime']:
        raise ValueError('scheduler adoption runtime differs from its reviewed construction')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('scheduler adoption needs a new immutable output namespace')
    inputs={};plans={};controls={};materials={}
    for role,item in spec['roles'].items():
        if set(item)!={'inputs','raw_state'}:raise ValueError('unexpected role adoption fields')
        inputs[role]=read(item['inputs'])
        material=read(record['role_materials'][role]);materials[role]=material
        # Authentication of controller/binder review, historical helpers and all
        # normal provider/publication inputs remains with the normal utility.
        for key in ('bootstrap','root_review','provider_readiness'):read(inputs[role][key])
        for reference in inputs[role]['frozen_refs']:bound(reference)
        plans[role]=inputs[role]['bootstrap']
        if material['runtime']!=str(construction.parent):raise ValueError('role runtime is not physically installed')
        controls[role]=emit(out/(role+'-prospective-controls.json'),{
            **material,'numerical_package':str(construction.parent/'training/neuralese/natlang_neuralese')})
    # Each independently prepared role can still carry a stale peer copy. Join
    # only the real own declarations, never a guessed peer recipe/checkpoint.
    source={role:read(ref)for role,ref in plans.items()}
    own={};assignment=None
    for role,plan in source.items():
        owned,exchange=roles.ownership(plan)
        if owned!={role} or exchange['role']!=role:
            raise ValueError('normal role inputs substituted optimizer ownership')
        actual=copy.deepcopy(exchange['roles'])
        for value in actual.values():value.pop('verification_mechanism',None)
        if assignment is not None and assignment!=actual:
            raise ValueError('normal role inputs disagree on actual source-host assignments')
        assignment=actual
        own[role]=copy.deepcopy(next(row for row in plan['models']if row['model']==role))
        check_foundation_task_policy(own[role],read(spec['scheduler']['curriculum']))
        if own[role].get('actor_runtime')!=str(construction.parent):
            raise ValueError('normal role invocation has not selected the reviewed later runtime')
    authorities=[read(materials[role]['role_mechanism_authorities'])for role in source]
    if authorities[0]!=authorities[1] or set(authorities[0])!=set(source):
        raise ValueError('normal role materials disagree on their actual six-mechanism maps')
    for role,plan in source.items():
        successor=copy.deepcopy(plan)
        successor['models']=[copy.deepcopy(own[row['model']])for row in plan['models']]
        for name,value in successor['role_exchange']['roles'].items():
            value['verification_mechanism']=authorities[0][name]
        plans[role]=emit(out/(role+'-coherent-normal-source-plan.json'),successor)
    bootstrap=emit(out/'coherent-whole-graph-rl-bootstrap.json',owned_bootstrap(spec['scheduler']['rl_bootstrap'],plans))
    result=scheduler(spec,plans,controls,out/'scheduler-proposal',bootstrap)
    successors={item['role']:item['successor']for item in result['source_plan_successions']}
    commands={};normal={}
    for role in inputs:
        value=copy.deepcopy(inputs[role]);value['bootstrap']=successors[role]
        value['frozen_refs'] += [spec['roles'][role]['inputs'],spec['composition'],plans[role],successors[role],
            *result.get('operational_inputs',[])]
        normal[role]=emit(out/(role+'-normal-role-inputs.json'),value)
        commands[role]=[spec['scheduler']['python'],str(construction.parent/'scripts/coreweave/prepare_native_actor_role.py'),
            '--composition',str(construction.parent),'--role',role,'--inputs',normal[role]['path'],
            '--out',str(Path(materials[role]['binder']['path']).parent/'binder-config.json')]
    return emit(out/'prepared.json',{'schema':SCHEMA,'inputs':binding(spec_path),'normal_inputs':normal,
        'normal_prepare_commands':commands,'proposal':binding(out/'scheduler-proposal/prepared.json'),
        'operational_inputs':result.get('operational_inputs',[]),
        'owner_changes':0,'activated':False,'readiness_granted':False})


def reconcile(prepared_path,published_path,out):
    prepared=json.loads(Path(prepared_path).read_bytes())
    if prepared.get('schema')!=SCHEMA:raise ValueError('unknown scheduler adoption receipt')
    spec=read(prepared['inputs']);proposal=read(prepared['proposal'])
    construction=bound(spec['composition']);record=verify_files(construction.parent)
    expected={item['role']:read(item['successor'])for item in proposal['source_plan_successions']}
    published=json.loads(Path(published_path).read_bytes())
    if set(published)!={'lfm','mellum'}:raise ValueError('adoption needs both actual published role contexts')
    plans={};controls={};pending=[]
    for role,item in published.items():
        if set(item)!={'spec','entered','raw_config'}:raise ValueError('published role context fields changed')
        if any(not Path(ref['path']).exists()for ref in item.values()):
            pending.append(role);continue
        control=read(item['spec']);entered=read(item['entered']);raw=read(item['raw_config'])
        material=read(record['role_materials'][role])
        if (control['model']!=role or binding(entered['spec']['path'],entered['spec'])!=binding(item['spec']['path'])
                or entered.get('whole_recipe_completed')is not False
                or control.get('numerical_code_sha256')!=record['code']
                or binding(control['controller']['path'],control['controller'])!=binding(material['controller']['path'],material['controller'])
                or control['numerical_package']!=str(Path(spec['scheduler']['runtime'])/'training/neuralese/natlang_neuralese')):
            raise ValueError('published controller is not this actual normal R6 successor')
        current=entered['plan'];plan=read(current)
        if roles.contract(plan)!=roles.contract(expected[role]):
            raise ValueError('adopted role changed its actual cohort or six-mechanism authorities')
        if any(plan.get(key)is not True for key in ('snapshot_preparation','consumer_preparation_handoff')):
            raise ValueError('normal owner has not adopted scheduler preparation flags')
        row=next(row for row in plan['models']if row['model']==role)
        check_foundation_task_policy(row,read(spec['scheduler']['curriculum']))
        previous=next(row for row in expected[role]['models']if row['model']==role)
        if row.get('commands')!=previous.get('commands'):
            raise ValueError('adopted normal role commands differ from reviewed future commands')
        references=[raw.get(name,{}).get('plan')for name in ('supervised_quantum_control','supervised_stage_quantum_control')]
        if any(not ref or binding(ref['path'],ref)!=binding(current['path'],current)for ref in references):
            raise ValueError('actual RAW configuration has not consumed its normal adopted plan')
        if binding(raw['production_recipe'])!=binding(row['production_recipe']['path'],row['production_recipe']):
            raise ValueError('actual RAW recipe differs from its preserved owned declaration')
        plans[role]=current;controls[role]=item['spec']
    if pending:return {'status':'waiting-for-normal-role-adoption','roles':pending,'activated':False}
    # Actual normal succession may preserve a newer additive recipe than the
    # proposal. Rebind only actual owned RL inputs; no saved credit is inferred.
    bootstrap=owned_bootstrap(spec['scheduler']['rl_bootstrap'],plans)
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('reconciled scheduler needs a fresh immutable namespace')
    bootstrap_ref=emit(out/'actual-whole-graph-rl-bootstrap.json',bootstrap)
    result=scheduler(spec,plans,controls,out/'scheduler',bootstrap_ref)
    # Use the actual adopted paths, never the preparation copies, in ownership calls.
    relay=read(result['relay']);controller=read(result['controller'])
    if result.get('operational_inputs'):controller['operational_inputs']=result['operational_inputs']
    for role,host in relay['hosts'].items():host['plan']=plans[role]
    consumer=result['contract']['semantic_consumer'];controller['raw_rl_bootstrap']=plans[consumer]
    relay_ref=emit(out/'relay-adopted.json',relay);controller_ref=emit(out/'controller-adopted.json',controller)
    runtime=Path(spec['scheduler']['runtime']);python=spec['scheduler']['python']
    construction=read(spec['composition'])
    return emit(out/'ready-to-launch.json',{'schema':SCHEMA,'prepared':binding(prepared_path),
        'published':binding(published_path),'plans':plans,'controls':controls,
        'relay':relay_ref,'controller':controller_ref,
        'controller_command':[python,'-m','natlang_neuralese.train.native_readiness','--config',controller_ref['path']],
        'controller_environment':{'PYTHONPATH':str(runtime/'training/neuralese')+':'+str(runtime),'NATLANG_REPO':str(runtime)},
        'controller_workdir':str(runtime),'consumer':consumer,'code':construction['code'],
        'relay_requires_sourcehost_sdk':True,'activated':False,'readiness_granted':False,
        'owner_changes':0,'checkpoint_decodes':0})


def stage_host_relay(io,sid,ready_reference,composition,out,python):
    """Read only small published metadata through the existing credential owner.

    This runs on the SDK source host after reconciliation, never in a sandbox.
    No remote exec, launch, owner write or model artifact transfer is involved.
    """
    from native_role_exchange import _read_bound
    record=verify_files(Path(composition))
    ready=json.loads(_read_bound(io,sid,ready_reference))
    if ready.get('schema')!=SCHEMA or ready.get('code')!=record['code']:
        raise ValueError('host relay source differs from the reconciled controller package')
    relay=json.loads(_read_bound(io,sid,ready['relay']))
    transport=copy.deepcopy(io.config)
    if set(transport['sandboxes'])!=set(relay['provider']['sandboxes']) or (
            ('local_source_host'in transport)!=('local_source_host'in relay['provider'])):
        raise ValueError('host relay transport substituted existing source-host resources')
    for source,resource in transport['sandboxes'].items():
        # The existing preparer adds the freshly authenticated role controls.
        # Resource identity, runtime, environment and ferry remain exact.
        resource['role_controls']=relay['provider']['sandboxes'][source]['role_controls']
    if 'local_source_host'in transport:
        local_id=transport['local_source_host']['source_host_id']
        expected=relay['provider']['local_source_host']
        if local_id!=expected['source_host_id']:
            raise ValueError('host relay substituted its actual local source owner')
        transport['local_source_host']['role_controls']=expected['role_controls']
    if (sid!=relay['provider']['coordinator_id'] or sid not in io.config['sandboxes']
            or relay['provider']!=transport):
        raise ValueError('host relay transport/provider differs from its actual consumer')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('host relay needs a new immutable namespace')
    original=emit(out/'consumer-relay.json',relay)
    relay['out']=str(out/'state')
    local=emit(out/'relay.json',relay)
    root=Path(composition).resolve()
    return emit(out/'prepared.json',{'schema':SCHEMA,'remote_ready':ready_reference,
        'original_relay':original,'relay':local,'code':record['code'],
        'command':[python,str(root/'scripts/coreweave/native_role_exchange.py'),'--config',local['path']],
        'environment':{'PYTHONPATH':str(root/'training/neuralese')+':'+str(root),'NATLANG_REPO':str(root)},
        'workdir':str(root),'activated':False,'owner_changes':0,'checkpoint_decodes':0})


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='mode',required=True)
    policy=commands.add_parser('materialize-current-policy',
        help='derive exact joint task checks from both actual current trainer configs')
    policy.add_argument('--curriculum',type=Path,required=True)
    policy.add_argument('--lfm-plan',type=Path,required=True);policy.add_argument('--mellum-plan',type=Path,required=True)
    policy.add_argument('--lfm-config',type=Path,required=True);policy.add_argument('--mellum-config',type=Path,required=True)
    policy.add_argument('--consumer',choices=('lfm','mellum'),required=True)
    policy.add_argument('--task-packet-relocations',type=Path,
        help='optional exact role-> {source:{path,sha256}, local:{path,sha256}} map for cloud packet custody')
    policy.add_argument('--lfm-recipe',type=Path,help='staged exact LFM recipe when task eval is not in its config')
    policy.add_argument('--mellum-recipe',type=Path,help='staged exact Mellum recipe when task eval is not in its config')
    prep=commands.add_parser('prepare');prep.add_argument('--spec',type=Path,required=True)
    check=commands.add_parser('reconcile');check.add_argument('--prepared',type=Path,required=True);check.add_argument('--published',type=Path,required=True)
    host=commands.add_parser('stage-host-relay')
    for name in ('provider','ready','composition'):host.add_argument('--'+name,type=Path,required=True)
    host.add_argument('--source-host',required=True);host.add_argument('--python',required=True)
    for command in (prep,check,host):command.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='materialize-current-policy':
        plans={role:binding(getattr(args,role+'_plan')) for role in ('lfm','mellum')}
        configs={role:str(getattr(args,role+'_config').resolve(strict=True)) for role in ('lfm','mellum')}
        relocations=json.loads(args.task_packet_relocations.read_bytes()) if args.task_packet_relocations else None
        recipes={role:str(getattr(args,role+'_recipe'))for role in ('lfm','mellum')
            if getattr(args,role+'_recipe') is not None}
        result=materialize_current_task_policy(args.curriculum,plans,configs,args.consumer,args.out,
            relocations,recipes)
    elif args.mode=='stage-host-relay':
        from provider_worker import SDKIO, cwctl
        cwctl.load_env_file();cwctl.require_key()
        result=stage_host_relay(SDKIO(json.loads(args.provider.read_bytes())),args.source_host,
            json.loads(args.ready.read_bytes()),args.composition,args.out,args.python)
    else:result=prepare(args.spec,args.out)if args.mode=='prepare'else reconcile(args.prepared,args.published,args.out)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
