"""Bind the existing foundation rendezvous and RL controller to real cloud assets.

Run on the coordinator sandbox after the owner mirrors both release recipes and
source closures. No provider credentials, resources, models or trainers here.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

from ..common.hashing import sha256_file_hex_cached, canonical_json_sha256_hex
from ..common.jsonio import write_json_durable, iter_jsonl_rows
from .native_readiness import _bound
from .recipe import resolve_recipe_data, resolve_stage_inputs, effective_stage_parameters, recipe_identity_sha256


def binding(path, expected=None):
    path = Path(path).resolve(strict=True)
    value={'path': str(path), 'sha256': sha256_file_hex_cached(path)}
    if expected is not None and (not isinstance(expected,dict)
            or set(expected) not in ({'path','sha256'},{'path','sha256','bytes'})
            or value!={key:expected.get(key)for key in ('path','sha256')}
            or ('bytes'in expected and(type(expected['bytes'])is not int
                or expected['bytes']<0 or expected['bytes']!=path.stat().st_size))):
        raise ValueError('native deployment artifact binding changed')
    return value


def prepare_role_scheduler(plans, controls, curriculum_path, provider_path, out, *,
                           raw_states, runtime, rl_bootstrap, handoff_lease_seconds, python=sys.executable):
    """Bind existing external owners to the existing passive semantic scheduler.

    Run against mirrored metadata on the actual semantic consumer. The returned
    source plans still require the normal owner succession before activation.
    This function never changes an existing plan, owner or training artifact.
    """
    from ..rl import native_role_release as roles
    from .native_semantic_stage import validate_config
    if set(plans)!=set(controls) or set(plans)!=set(raw_states) or len(plans)<2:
        raise ValueError('role scheduling needs every actual owner plan, controls and raw state')
    loaded={name:json.loads(Path(binding(ref['path'],ref)['path']).read_text()) for name,ref in plans.items()}
    common=None;owned_rows={};hosts={}
    provider=json.loads(Path(provider_path).read_text())
    for name,plan in loaded.items():
        owned,exchange=roles.ownership(plan)
        if exchange is None or exchange['role']!=name or len(owned)!=1:
            raise ValueError('role scheduling requires exact existing single-student owners')
        actual=roles.contract(plan)
        if common is not None and actual!=common:
            raise ValueError('role scheduling requires coherent actual role authorities and cohorts')
        common=actual
        row=next(row for row in plan['models']if row['model']in owned)
        if (row.get('foundation_readiness_policy')!='on-shared-request'
                or not row.get('foundation_qualification_generation')
                or row['handoff_requirement'].get('readiness_boundary')!='communication-foundation'):
            raise ValueError('role scheduling requires explicit on-request communication readiness')
        sid=exchange['roles'][name]['source_host_id']
        if sid not in provider.get('sandboxes',{}):
            raise ValueError('role scheduler cannot allocate or substitute a source host')
        control_path=Path(binding(controls[name]['path'],controls[name])['path'])
        spec=json.loads(control_path.read_text())
        if spec.get('model')!=name or not spec.get('controller'):
            raise ValueError('role scheduling lost its actual publishing controller')
        binding(spec['controller']['path'],spec['controller'])
        package=Path(spec['numerical_package']).resolve(strict=True)
        if provider['sandboxes'][sid]['runtime']!=str(package.parents[2]):
            raise ValueError('role provider runtime differs from its actual publishing controls')
        provider['sandboxes'][sid]['role_controls']={**controls[name],
            'bytes':control_path.stat().st_size}
        state=Path(raw_states[name])
        if not state.is_absolute() or '..'in state.parts:
            raise ValueError('role scheduler needs each explicit actual RAW state path')
        owned_rows[row['model']]=copy.deepcopy(row)
        hosts[name]={'source_host_id':sid,'raw_state':str(state)}
    if set(hosts)!=set(common['roles']):
        raise ValueError('role scheduler omitted an actual source-host owner')
    semantic=validate_config(json.loads(Path(curriculum_path).read_text()))
    if semantic.get('policy_training'):
        raise ValueError('new snapshot preparation cannot inherit an unrelated policy episode batch')
    for key in ('pairs','held_pairs'):_bound(semantic[key])
    if not semantic.get('task_eval') or not semantic.get('policy_schedule'):
        raise ValueError('joint scheduling needs actual task checks and whole-graph policy cadence')
    if type(handoff_lease_seconds)not in(int,float) or not 30<handoff_lease_seconds:
        raise ValueError('role scheduler needs an explicit renewable handoff lease longer than two relay polls')
    bootstrap_path=Path(binding(rl_bootstrap['path'],rl_bootstrap)['path'])
    bootstrap=json.loads(bootstrap_path.read_text())
    if (bootstrap.get('schema')!='natlang.native-rl-bootstrap/1'
            or {row['model']for row in bootstrap.get('models',[])}!=set(plans)
            or len(bootstrap['models'])!=len(plans)):
        raise ValueError('role scheduler requires the complete existing whole-graph RL bootstrap')
    for row in bootstrap['models']:
        original=owned_rows[row['model']]
        for key in ('native_participant','production_recipe','cases','seed_native_graphs','held_native_graphs'):
            if row.get(key)!=original.get(key) or not row.get(key):
                raise ValueError('joint RL bootstrap changed actual owned inputs: '+key)
        for key in ('production_recipe','cases','seed_native_graphs','held_native_graphs'):_bound(row[key])
    runtime=Path(runtime).resolve(strict=True)
    if not (runtime/'training/neuralese/natlang_neuralese/train/native_readiness.py').is_file():
        raise ValueError('role scheduler requires its actual complete controller runtime')
    consumer=common['semantic_consumer']
    if provider.get('coordinator_id')!=hosts[consumer]['source_host_id']:
        raise ValueError('semantic scheduling belongs to its existing declared consumer')
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('role scheduler output must be a new immutable declaration')
    out.mkdir(parents=True,exist_ok=True)
    source_plans={};adoptions=[]
    for name,plan in loaded.items():
        successor=copy.deepcopy(plan);successor['snapshot_preparation']=True
        path=out/(name+'-snapshot-source-plan.json');write_json_durable(path,successor)
        source_plans[name]=binding(path)
        hosts[name]['plan']=source_plans[name]
        adoptions.append({'role':name,'previous':plans[name],'successor':source_plans[name],
            'owned_declaration_unchanged':True,'applied':False})
    snapshot=copy.deepcopy(semantic);snapshot['snapshot_preparation']=True
    snapshot['inherit_optimizer']=True
    curriculum=out/'snapshot-curriculum.json';write_json_durable(curriculum,snapshot)
    full=out/'joint-curriculum.json';write_json_durable(full,semantic)
    relay=out/'relay.json'
    write_json_durable(relay,{'schema':'natlang.native-role-exchange-service/1',
        'hosts':hosts,'provider':provider,'out':str(out/'relay-state'),'poll_seconds':15,
        'handoff_lease_seconds':handoff_lease_seconds})
    controller=out/'controller.json'
    write_json_durable(controller,{'schema':'natlang.native-shared-stage-controller/1',
        'out':str(out/'semantic'),'semantic_out':str(out/'semantic/preparation'),
        'curriculum':str(curriculum),'poll_seconds':60,
        'snapshot_successor':{'curriculum':binding(full),'rl_bootstrap':rl_bootstrap,
            'command':[python,'-m','natlang_neuralese.train.native_semantic_stage',
                '--config','{joint_curriculum}','--participants','{participants}',
                '--out','{out}','--device','cuda']},
        'participants':[row['handoff_requirement'] for row in owned_rows.values()],
        'raw_rl_bootstrap':source_plans[consumer],
        'raw_rl_out':str(Path(raw_states[consumer]).parent),'manage_raw_bootstrap':False,
        'command':[python,'-m','natlang_neuralese.train.native_semantic_stage',
            '--config',str(curriculum),'--participants','{participants}',
            '--out','{out}','--device','cuda'],
        'workdir':str(runtime)})
    receipt={'schema':'natlang.native-role-scheduler-preparation/1',
        'controller':binding(controller),'relay':binding(relay),
        'source_plan_successions':adoptions,'contract':common,
        'inputs':{'plans':plans,'controls':controls,'curriculum':binding(curriculum_path),
            'provider':binding(provider_path),'rl_bootstrap':rl_bootstrap},'owner_changes':0,'checkpoint_decodes':0,
        'model_allocations':0,'readiness_granted':False,'activated':False}
    write_json_durable(out/'prepared.json',receipt)
    return receipt


def prepare(release_path, curriculum_path, out, *, participants, cases=None, task_source_manifest=None,
            runtime, namespace='/work/native-production',
            refresh_curriculum_path=None):
    """Create runnable configuration without inventing future trained identities."""
    release_path, curriculum_path, out = map(Path, (release_path, curriculum_path, out))
    records = json.loads(release_path.read_text())
    if {row['role'] for row in records} != {'lfm', 'mellum'} or len(records) != 2:
        raise ValueError('native deployment requires exactly the two current training releases')
    if set(participants) != {'lfm', 'mellum'} or len(set(participants.values())) != 2:
        raise ValueError('native deployment must name both exact distinct participants')
    runtime = Path(runtime).resolve(strict=True)
    if not (runtime/'ts-host/dist/teacher/cli.js').is_file():
        raise ValueError('native deployment requires the actual built runtime')
    task_source = None
    if task_source_manifest is None and cases is None:
        task_source_manifest = Path(__file__).resolve().parents[4] / 'runs/decision-rich-train-20261016-v2-curriculum/task-source-manifest.json'
    if task_source_manifest is not None:
        from ..rl.task_source import resolve_task_source
        task_source = resolve_task_source(task_source_manifest)
        if cases is not None and Path(cases).resolve(strict=True) != task_source['cases']:
            raise ValueError('explicit RL cases differ from the selected admitted task-source manifest')
        cases = task_source['cases']
    if cases is None:
        raise ValueError('native deployment requires an admitted RL task source')
    cases_binding = binding(cases)
    case_rows = [row for _, row in iter_jsonl_rows(cases)]
    if not case_rows or any(row.get('split') != 'train' for row in case_rows):
        raise ValueError('native deployment requires the actual admitted nonempty train-only case packet')
    semantic = json.loads(curriculum_path.read_text())
    refresh = json.loads(Path(refresh_curriculum_path).read_text()) if refresh_curriculum_path else semantic
    for declaration in (semantic, refresh):
        if declaration.get('schema') != 'natlang.native-semantic-training/1':
            raise ValueError('native deployment requires the declared semantic curriculum')
        for key in ('pairs', 'held_pairs'):
            _bound(declaration[key])
        # A refresh quantum controls safe handoff, never production completion.
        # Collected whole episodes supply the actual policy_training sources;
        # an empty source declaration must not masquerade as trained exposure.
        declaration.setdefault('policy_schedule', {'id':'policy', 'epochs':1,
            'every_semantic_updates':1, 'updates_per_refresh':16})
    out = out.resolve(); out.mkdir(parents=True, exist_ok=True)
    queue = str(Path(namespace)/'shared/bridge-ipc')
    client = [sys.executable, '-m', 'natlang_neuralese.rl.native_bridge', '--queue', queue,
              '--request', '{request}', '--result', '{result}', '--stop-file', '{stop_file}']
    models, requirements = [], []
    for record in sorted(records, key=lambda row: row['role']):
        role = record['role']; line = 'lfm25' if role == 'lfm' else 'mellum21'
        recipe_path = _bound({'path': record['recipe_path'], 'sha256': record['recipe_sha256']})
        adoption_binding = record.get('resident_adoption')
        if adoption_binding:
            from .native_resident import current
            adoption_path = _bound(adoption_binding)
            adoption = current(adoption_path.parent)
            if (not adoption or adoption_path.name != 'current.json' or adoption['payload']['recipe']['path'] != str(recipe_path)
                    or adoption['payload']['recipe']['file_sha256'] != record['recipe_sha256']):
                raise ValueError('resident release does not bind its actual current recipe')
        recipe = resolve_recipe_data(recipe_path)
        stages = [stage for stage in recipe['stages'] if stage['kind'] == 'native_state_sft']
        if len(stages) != 1:
            raise ValueError('foundation release must identify its single shared native SFT stage')
        stage = stages[0]
        inputs = resolve_stage_inputs(recipe, stage, {})
        parameters = effective_stage_parameters(stage, [], recipe)
        cadence = parameters.get('task_eval_every', 0)
        if not parameters.get('task_eval') or type(cadence) is not int or cadence < 0:
            raise ValueError('native deployment requires actual boundary task evaluation and nonnegative routine cadence')
        # Zero disables routine synchronous pauses, while the native trainer
        # still evaluates exact-current foundation/refresh/final weights.
        for key in ('native_graphs', 'held_native_graphs', 'foundation_native_graphs'):
            _bound(inputs[key])
        if (Path(inputs['foundation_native_graphs']['path']).resolve() != Path(record['foundation_manifest']['path']).resolve()
                or inputs['foundation_native_graphs']['sha256'] != record['foundation_manifest']['sha256']):
            raise ValueError('release foundation differs from the actual training recipe')
        if role == 'mellum':
            for declaration in (semantic, refresh):
                declaration['task_eval'] = parameters['task_eval']
        declared_base = record.get('base') or parameters.get('base')
        if not declared_base:
            raise ValueError('native deployment requires an explicit actual base path from release or recipe')
        base = declared_base['path'] if isinstance(declared_base, dict) else declared_base
        from ..model.native_model import base_inventory
        inventory = base_inventory(base)
        inventory_sha = canonical_json_sha256_hex(inventory)
        if (inventory_sha != participants[role].split(':', 1)[-1]
                or isinstance(declared_base, dict) and declared_base.get('inventory_sha256') != inventory_sha):
            raise ValueError('native deployment resident base inventory differs from its exact participant')
        for declaration in (semantic, refresh):
            declaration.setdefault('models', {}).setdefault(participants[role], {})['base'] = str(base)
        models.append({'model': role, 'native_participant': participants[role],
            'production_recipe': binding(recipe_path), 'production_stage': stage['id'],
            'seed_native_graphs': inputs['native_graphs'], 'held_native_graphs': inputs['held_native_graphs'],
            'cases': cases_binding, 'runtime': str(runtime/'ts-host'), 'base': base,
            **({'task_source': {'manifest': str(task_source['manifest']),
                'manifest_sha256': task_source['manifest_sha256'], 'cases_sha256': task_source['cases_sha256'],
                'rows': task_source['rows'], 'audit_sha256': task_source['audit_sha256'],
                'selection_proof_sha256': task_source['selection_proof_sha256']}} if task_source else {}),
            'base_inventory_sha256': inventory_sha,
            'port': 8097 if role == 'lfm' else 8098, 'provider_queue': queue,
            'provider': {'sandbox_id': record['sandbox_id']}})
        requirements.append({'model_id': participants[role],
            'readiness': str(Path(namespace)/line/'native_full_sft/native-training-receipt.json'),
            'trainer_exit': str(Path(record['control'])/'exit.json'),
            'trainer_launch': binding(Path(record['control'])/'launch.json'),
            'trainer_recipe': str(recipe_path),
            **({'resident_adoption': adoption_binding} if adoption_binding else {}),
            'recipe_sha256': recipe_identity_sha256(recipe_path), 'stage_ids': [stage['id']],
            'readiness_boundary': 'communication-foundation',
            'foundation_sources': inputs['foundation_native_graphs'],
            'foundation_epochs': parameters.get('epochs', 1.), 'forward_atol': 1e-4, 'gradient_atol': 1e-5})
    curriculum = out/'curriculum.json'; write_json_durable(curriculum, semantic)
    refresh_curriculum = curriculum
    if refresh_curriculum_path:
        refresh_curriculum = out/'refresh-curriculum.json'
        write_json_durable(refresh_curriculum, refresh)
    bootstrap = out/'rl-bootstrap.json'
    write_json_durable(bootstrap, {'schema': 'natlang.native-rl-bootstrap/1', 'models': models,
        'semantic_refresh_config': binding(refresh_curriculum), 'refresh_command': client,
        'refresh_provider': next(model['provider'] for model in models if model['model'] == 'mellum')})
    raw_bootstrap=out/'raw-rl-bootstrap.json'
    write_json_durable(raw_bootstrap, {'schema':'natlang.native-raw-rl-bootstrap/1',
        'models':[{**model,'handoff_requirement':requirement}
                  for model,requirement in zip(models,requirements)]})
    controller = out/'controller.json'
    shared = Path(namespace)/'shared'
    write_json_durable(controller, {'schema': 'natlang.native-shared-stage-controller/1',
        'out': str(shared/'native_shared_stage'), 'curriculum': str(curriculum),
        'participants': requirements, 'poll_seconds': 60,
        'command': [sys.executable, '-m', 'natlang_neuralese.train.native_semantic_stage',
            '--config', str(curriculum), '--participants', '{participants}', '--out', '{out}', '--device', 'cuda'],
        'rl_bootstrap': binding(bootstrap), 'rl_out': str(shared/'native_rl'), 'rl_device': 'cuda',
        'raw_rl_bootstrap':binding(raw_bootstrap),
        'deployment_inputs': {'release': binding(release_path), 'curriculum': binding(curriculum_path),
            **({'refresh_curriculum':binding(refresh_curriculum_path)} if refresh_curriculum_path else {})}})
    return controller


def main(argv=None):
    argv=list(sys.argv[1:]if argv is None else argv)
    if argv and argv[0]=='role-scheduler':
        parser=argparse.ArgumentParser(description='Bind the existing source-host role scheduler')
        parser.add_argument('--spec',type=Path,required=True)
        parser.add_argument('--out',type=Path,required=True)
        args=parser.parse_args(argv[1:]);spec=json.loads(args.spec.read_text())
        if (set(spec)!={'schema','plans','controls','curriculum','provider','raw_states','runtime','python','rl_bootstrap','handoff_lease_seconds'}
                or spec['schema']!='natlang.native-role-scheduler-inputs/1'):
            raise ValueError('role scheduling requires the complete explicit metadata input contract')
        for key in ('curriculum','provider'):binding(spec[key]['path'],spec[key])
        result=prepare_role_scheduler(spec['plans'],spec['controls'],spec['curriculum']['path'],
            spec['provider']['path'],args.out,raw_states=spec['raw_states'],runtime=spec['runtime'],python=spec['python'],
            rl_bootstrap=spec['rl_bootstrap'],handoff_lease_seconds=spec['handoff_lease_seconds'])
        print(json.dumps(result,sort_keys=True));return result
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('release', 'curriculum', 'out', 'runtime'):
        parser.add_argument('--'+flag, type=Path, required=True)
    parser.add_argument('--cases', type=Path, help='explicit admitted train task packet; defaults to the bound 128-row RL source')
    parser.add_argument('--task-source-manifest', type=Path,
                        help='integrity-bound admitted RL task source; defaults to decision-rich-train-20261016-v2-curriculum')
    parser.add_argument('--participant', action='append', required=True, help='role=exact architecture:base-inventory identity')
    parser.add_argument('--namespace', default='/work/native-production')
    parser.add_argument('--refresh-curriculum', type=Path,
                        help='expanded refresh source; keeps the first shared foundation curriculum fixed')
    args = parser.parse_args(argv)
    participants = dict(item.split('=', 1) for item in args.participant)
    print(prepare(args.release, args.curriculum, args.out, participants=participants, cases=args.cases,
        task_source_manifest=args.task_source_manifest,
        runtime=args.runtime, namespace=args.namespace, refresh_curriculum_path=args.refresh_curriculum))


if __name__ == '__main__':
    main()
