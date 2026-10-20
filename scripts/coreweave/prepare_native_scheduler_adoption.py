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
    if (not {'schema','passed','source','base_source'}<=set(review)
            or review.get('schema')!='natlang.native-scheduler-producer-review/1' or review.get('passed')is not True
            or binding(source,review.get('source'))!=binding(source,selected['source'])
            or binding(base,review.get('base_source'))!=binding(base,selected['base_source'])):
        raise ValueError('scheduler producer lacks its exact reviewed source/base authority')
    # A separately named host producer; never overwrite a loaded runtime module
    # or claim that its file belongs to the numerical composition inventory.
    import importlib.util
    name='natlang_neuralese.train._reviewed_scheduler_'+selected['source']['sha256']
    module_spec=importlib.util.spec_from_file_location(name,source)
    module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
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
    if set(transport['sandboxes'])!=set(relay['provider']['sandboxes']):
        raise ValueError('host relay transport substituted existing source-host resources')
    for source,resource in transport['sandboxes'].items():
        # The existing preparer adds the freshly authenticated role controls.
        # Resource identity, runtime, environment and ferry remain exact.
        resource['role_controls']=relay['provider']['sandboxes'][source]['role_controls']
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
    prep=commands.add_parser('prepare');prep.add_argument('--spec',type=Path,required=True)
    check=commands.add_parser('reconcile');check.add_argument('--prepared',type=Path,required=True);check.add_argument('--published',type=Path,required=True)
    host=commands.add_parser('stage-host-relay')
    for name in ('provider','ready','composition'):host.add_argument('--'+name,type=Path,required=True)
    host.add_argument('--source-host',required=True);host.add_argument('--python',required=True)
    for command in (prep,check,host):command.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='stage-host-relay':
        from provider_worker import SDKIO, cwctl
        cwctl.load_env_file();cwctl.require_key()
        result=stage_host_relay(SDKIO(json.loads(args.provider.read_bytes())),args.source_host,
            json.loads(args.ready.read_bytes()),args.composition,args.out,args.python)
    else:result=prepare(args.spec,args.out)if args.mode=='prepare'else reconcile(args.prepared,args.published,args.out)
    print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
