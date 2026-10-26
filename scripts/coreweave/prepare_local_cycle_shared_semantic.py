#!/usr/bin/env python3
"""Prepare the two-owner local-LFM/cloud-Mellum semantic handoff, metadata only.

This prepares fresh role plans, their exact local-cycle adapters, the paired
semantic task policy, and source-host staging manifests. It never signals an
owner, contacts CoreWeave, reads model tensors, launches a trainer, or grants
readiness. The caller must first supply current bound role/bootstrap/control
inputs captured from each actual owner.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'training/neuralese'),str(ROOT/'scripts/coreweave')]
from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.train.native_deployment import binding,prepare_role_scheduler
from natlang_neuralese.train.recipe import (resolve_recipe_data,resolve_stage_inputs,effective_stage_parameters,
    recipe_identity_sha256,recipe_file_dependencies)
from natlang_neuralese.rl.native_role_release import EXCHANGE
from prepare_native_scheduler_adoption import materialize_current_task_policy

SCHEMA='natlang.local-cycle-shared-semantic-input/1'


def read(reference):
    checked=binding(reference['path'],reference)
    return json.loads(Path(checked['path']).read_bytes())


def emit(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('local-cycle shared-semantic outputs need a fresh immutable namespace')
    write_json_durable(path,value)
    reference=binding(path)
    return {**reference,'bytes':Path(reference['path']).stat().st_size}


def _under(path,roots):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:return False
    return any(path==Path(root) or Path(root) in path.parents for root in roots)


def _host_roots(provider,role,host_id):
    if role=='lfm':
        host=provider.get('local_source_host') or {}
        if host.get('source_host_id')!=host_id:raise ValueError('LFM source host is not the configured local owner')
        roots=host.get('write_roots')
        if not roots:raise ValueError('local source host has no explicit writable staging roots')
        return roots
    host=(provider.get('sandboxes')or{}).get(host_id) or {}
    roots=host.get('write_roots') or ['/work/native-production']
    return roots


def _private_export_inventory_path(provider,role,participant):
    """Select a writable local receipt index with a declared read route."""
    host=provider.get('local_source_host')or{}
    read_roots=[Path(value).resolve()for value in host.get('read_roots',[])]
    write_roots=[Path(value).resolve()for value in host.get('write_roots',[])]
    explicit=host.get('private_export_inventories',{}).get(role)
    if explicit is not None:
        path=Path(explicit)
        if (not path.is_absolute()or'..'in path.parts or str(path)!=str(path.resolve(strict=False))
                or not any(path.resolve(strict=False)==root or root in path.resolve(strict=False).parents for root in read_roots)
                or not any(path.resolve(strict=False)==root or root in path.resolve(strict=False).parents for root in write_roots)):
            raise ValueError('explicit private export inventory must be readable and writable under the local source host roots')
        return str(path.resolve(strict=False))
    roots=[root for root in write_roots if any(root==read or read in root.parents for read in read_roots)]
    if not roots:
        raise ValueError('local source owner needs a root shared by private export inventory readers and writers')
    digest=canonical_json_sha256_hex({'role':role,'participant':participant})[:24]
    return str(roots[0]/'control'/'private-export-inventories'/(role+'-'+digest+'.json'))


def _bundle_export_descriptor(role,item,participant,provider,inventory_path):
    writable=_host_roots(provider,role,item['source_host_id'])
    root=Path(writable[0])/('control/native-shared-participant-exports/'+
        canonical_json_sha256_hex({'role':role,'participant':participant})[:24])
    entrypoint='scripts.coreweave.native_shared_participant_export'
    manifest=str(root/'current-prepared.json')
    common=[item['python'],'-m',entrypoint]
    source_provider=provider.get('local_source_host')or{}
    export_python=source_provider.get('exporter_python',source_provider.get('python'))
    export_runtime=source_provider.get('runtime')
    if not isinstance(export_python,str)or not Path(export_python).is_absolute()or not isinstance(export_runtime,str):
        raise ValueError('private full-state export needs the exact local owner Python/runtime to keep uploader credentials host-side')
    export_source_mode='local' if role=='lfm' else 'sandbox'
    export_manifest=manifest if role=='lfm' else str(Path(source_provider['write_roots'][0])/
        ('control/native-shared-participant-exports/'+canonical_json_sha256_hex(
            {'role':role,'participant':participant})[:24]+'/current-prepared.json'))
    export_command=[export_python,'-m',entrypoint,'export','--manifest',export_manifest,
        '--inventory',inventory_path,'--role',role,'--exporter','{bound_private_exporter}',
        '--exporter-python','{bound_private_exporter_python}',
        '--source-mode',export_source_mode]
    if role=='mellum':export_command.extend(['--sandbox-id',item['source_host_id']])
    return {'schema':'natlang.native-shared-participant-export-follow/1','role':role,
        'participant':participant,'source_host_id':item['source_host_id'],
        'state_path':item['state_path'],'cursor_path':item['cursor_path'],
        'runtime':item['runtime'],'python':item['python'],
        'export_root':str(root),'private_export_inventory':inventory_path,
        'entrypoint':entrypoint,'manifest_path':manifest,
        'command_context':{'cwd':item['runtime'],'PYTHONPATH':[item['runtime']+'/training/neuralese',item['runtime']]},
        'prepare_command':[*common,'prepare-current','--state',item['state_path'],
            '--participant',participant,'--out-root',str(root),'--manifest',manifest],
        'private_export_runner':{'source_host_id':source_provider.get('source_host_id'),
            'runtime':export_runtime,'python':export_python,
            'source_worker_python':source_provider.get('python'),
            'exporter_python_is_distinct':export_python!=source_provider.get('python'),
            'manifest_source_host_id':item['source_host_id'],
            'source_manifest_path':manifest,'local_manifest_path':export_manifest,
            'manifest_must_be_read_back_before_export':role=='mellum',
            'manifest_transport':({'mode':'bounded-provider-metadata-read','source_host_id':item['source_host_id'],
                'source_path':manifest,'local_target':export_manifest,'maximum_bytes':16*1024**2}
                if role=='mellum' else None),
            'source_mode':export_source_mode},
        'export_command_template':export_command,
        'pending_command':[*common,'pending','--manifest',manifest,'--inventory',inventory_path,'--role',role],
        'record_receipt_command_template':[*common,'record','--manifest',manifest,
            '--inventory',inventory_path,'--role',role,'--source','{source_path}','--receipt','{receipt_path}'],
        'private_export_contract':'Every bundle member, including the selected participant shard, global semantic/training shards, bundle manifest, and tokenizer/control files, needs an exact private receipt with independent full GET before the atomic inventory append.',
        'training_credit':0,'readiness_granted':False,'activated':False}


def _source_map_routes(reference):
    if reference is None:return {},{},None
    path=Path(binding(reference['path'],reference)['path'])
    value=json.loads(path.read_bytes())
    if value.get('schema')!='natlang.staged-source-map/1' or not isinstance(value.get('paths'),dict):
        raise ValueError('role source map has an unsupported schema')
    routes={};forward={}
    for source,target in value['paths'].items():
        routes.setdefault(target,[]).append(source);forward.setdefault(source,[]).append(target)
    return routes,forward,binding(path,reference)


def _resolve_local_source(path,expected_sha,routes):
    candidate=Path(path)
    if candidate.is_file() and binding(candidate)['sha256']==expected_sha:return str(candidate.resolve())
    matches=[Path(source).resolve()for source in routes.get(str(path),[])
        if Path(source).is_file() and binding(source)['sha256']==expected_sha]
    if len(matches)!=1:raise ValueError('recipe source has no unique byte-exact local copy: '+str(path))
    return str(matches[0])


def _resolve_config_file(path,routes):
    candidate=Path(path)
    if candidate.is_file():return binding(candidate)
    matches=[Path(source).resolve()for source in routes.get(str(path),[])if Path(source).is_file()]
    if len(matches)!=1:raise ValueError('configured owner file has no unique exact source-map copy: '+str(path))
    return binding(matches[0])


def build_role_declarations(role_inputs):
    """Derive fresh bootstrap rows from exact owner config, recipe and readiness receipt bytes."""
    if set(role_inputs)!={'lfm','mellum'}:raise ValueError('fresh role construction requires both actual owners')
    raw_models=[]
    qualification={};deployment_models={}
    for role in ('lfm','mellum'):
        item=role_inputs[role]
        required={'config','readiness','owner_state','owner_cursor','recipe','recipe_target',
            'source_host_id','provider','provider_queue','port'}
        if (not required<=set(item) or ('source_receipt' in item)==('source_checkpoint_identity' in item)
                or 'model'in item or 'handoff_requirement'in item):
            raise ValueError('role declaration needs exact config, receipt, recipe and placement bindings: '+role)
        config_path=Path(binding(item['config']['path'],item['config'])['path'])
        config=json.loads(config_path.read_bytes())
        source_evidence_key='source_receipt' if 'source_receipt' in item else 'source_checkpoint_identity'
        source_evidence_ref=binding(item[source_evidence_key]['path'],item[source_evidence_key])
        source_evidence_path=Path(source_evidence_ref['path'])
        source_evidence=json.loads(source_evidence_path.read_bytes())
        owner_state_ref=binding(item['owner_state']['path'],item['owner_state'])
        owner_cursor_ref=binding(item['owner_cursor']['path'],item['owner_cursor'])
        owner_state=json.loads(Path(owner_state_ref['path']).read_bytes())
        owner_cursor=json.loads(Path(owner_cursor_ref['path']).read_bytes())
        if (owner_state.get('schema')!='natlang.rl-production-state/1'
                or not isinstance(owner_state.get('checkpoint'),str)
                or not isinstance(owner_cursor.get('checkpoint'),dict)
                or owner_state['checkpoint']!=owner_cursor['checkpoint'].get('path')):
            raise ValueError('owner state and cycle cursor do not bind the same active checkpoint path')
        recipe_path=Path(binding(item['recipe']['path'],item['recipe'])['path'])
        declared_recipe=config.get('production_recipe')
        if not isinstance(declared_recipe,str)or item['recipe_target']!=declared_recipe:
            raise ValueError('role recipe relocation differs from actual config production_recipe')
        if config.get('model')!=role:raise ValueError('owner config model differs from requested role')
        mechanism_ref=None
        mechanism=None
        if item.get('mechanism_evidence') is not None:
            mechanism_ref=binding(item['mechanism_evidence']['path'],item['mechanism_evidence'])
            mechanism_record=json.loads(Path(mechanism_ref['path']).read_bytes())
            from natlang_neuralese.rl.native_role_release import MECHANISM_FILES
            if (mechanism_record.get('schema')!='natlang.local-cycle-runtime-mechanism-evidence/1'
                    or mechanism_record.get('runtime')!=item.get('runtime')
                    or not isinstance(mechanism_record.get('mechanism'),dict)
                    or set(mechanism_record['mechanism'])!=set(MECHANISM_FILES)
                    or any(not isinstance(value,str)or re.fullmatch(r'[0-9a-f]{64}',value)is None
                        for value in mechanism_record['mechanism'].values())
                    or mechanism_record.get('mechanism_inventory_sha256')!=canonical_json_sha256_hex({
                        'runtime':mechanism_record.get('runtime'),'mechanism':mechanism_record['mechanism']})):
                raise ValueError('source host mechanism evidence does not bind its declared learner runtime and mechanism inventory')
            mechanism=mechanism_record['mechanism']
        receipt=None
        if source_evidence_key=='source_receipt':
            receipt=source_evidence
            if (receipt.get('schema')!='natlang.native-training-readiness/1'
                    or receipt.get('native_identity',{}).get('checkpoint_sha256')!=receipt.get('checkpoint',{}).get('sha256')):
                raise ValueError('owner readiness receipt has inconsistent native checkpoint identity')
            checkpoint_sha=receipt.get('checkpoint',{}).get('sha256')
            if receipt.get('checkpoint',{}).get('path')!=owner_cursor.get('checkpoint',{}).get('path'):
                raise ValueError('owner readiness receipt path differs from the current owner checkpoint')
            model_id=receipt.get('model_id')
            source_recipe_sha=receipt.get('recipe',{}).get('sha256')
        else:
            identity=source_evidence
            if identity.get('schema')!='natlang.local-cycle-source-checkpoint-identity/1':
                raise ValueError('unqualified owner head needs the explicit metadata-only source-identity schema')
            metadata_ref=binding(identity['metadata_verification']['path'],identity['metadata_verification'])
            private_ref=binding(identity['private_receipt']['path'],identity['private_receipt'])
            prior_ref=binding(identity['prior_training_receipt']['path'],identity['prior_training_receipt'])
            metadata=json.loads(Path(metadata_ref['path']).read_bytes())
            private=json.loads(Path(private_ref['path']).read_bytes())
            prior=json.loads(Path(prior_ref['path']).read_bytes())
            checkpoint_binding=identity.get('checkpoint')
            if (metadata.get('passed') is not True or metadata.get('tensor_payload_bytes_materialized')!=0
                    or metadata.get('optimizer_rng_exposure_present') is not True
                    or not isinstance(checkpoint_binding,dict)
                    or checkpoint_binding.get('path')!=owner_cursor.get('checkpoint',{}).get('path')
                    or metadata.get('identity',{}).get('sha256')!=checkpoint_binding.get('sha256')
                    or metadata.get('architecture')!=role
                    or metadata.get('base_inventory_sha256')!=identity.get('base_inventory_sha256')
                    or metadata.get('global_step')!=identity.get('global_step')
                    or metadata.get('recipe_binding')!=identity.get('checkpoint_recipe')
                    or metadata.get('private_receipt')!=private_ref['path']
                    or private.get('identity',{}).get('sha256')!=checkpoint_binding.get('sha256')
                    or private.get('identity',{}).get('bytes')!=metadata.get('identity',{}).get('bytes')
                    or private.get('role')!=role or private.get('private') is not True
                    or private.get('recovery_verification') is not True
                    or private.get('independent_verification',{}).get('sha256')!=checkpoint_binding.get('sha256')
                    or private.get('independent_verification',{}).get('bytes')!=metadata.get('identity',{}).get('bytes')
                    or prior.get('schema')!='natlang.native-training-readiness/1'
                    or prior.get('model_id')!=identity['architecture']+':'+identity['base_inventory_sha256']
                    or prior.get('native_identity',{}).get('checkpoint_sha256')!=prior.get('checkpoint',{}).get('sha256')):
                raise ValueError('metadata-only owner head lacks exact independent checkpoint/optimizer custody')
            checkpoint_sha=checkpoint_binding['sha256']
            model_id=identity['architecture']+':'+identity['base_inventory_sha256']
            source_recipe_sha=identity['checkpoint_recipe'].get('sha256')
        if checkpoint_sha!=owner_cursor.get('checkpoint',{}).get('sha256'):
            raise ValueError('source identity evidence is not for the current owner checkpoint')
        if not isinstance(model_id,str)or ':' not in model_id or model_id.split(':',1)[0]!=('lfm2' if role=='lfm' else 'mellum'):
            raise ValueError('owner receipt model identity does not match the requested role')
        recipe=resolve_recipe_data(recipe_path);policy_sha=recipe_identity_sha256(recipe_path)
        if (source_evidence_key=='source_receipt' and source_recipe_sha not in (None,policy_sha)):
            raise ValueError('owner receipt names a different production recipe identity')
        matching=[stage for stage in recipe['stages']if stage.get('id')==config.get('production_stage')]
        if len(matching)!=1 or matching[0].get('kind')!='native_state_sft':
            raise ValueError('current configured stage must resolve to one native full-state SFT stage')
        stage=matching[0];routes,forward,source_map_ref=_source_map_routes(item.get('source_map'))
        overrides={}
        for name,spec in recipe.get('input_bindings',{}).items():
            if 'path'in spec:overrides[name]=_resolve_local_source(spec['path'],spec['sha256'],routes)
        def route(manifest,child):
            child=Path(child)
            if child.is_file():return child
            matches=[Path(source)for source in routes.get(str(child),[])if Path(source).is_file()]
            return matches[0] if len(matches)==1 else child
        resolved=resolve_stage_inputs(recipe,stage,{},overrides=overrides,source_route=route)
        inputs={key:resolved[key]for key in ('native_graphs','held_native_graphs','foundation_native_graphs')}
        params=effective_stage_parameters(stage,[],recipe)
        if not params.get('communication_foundation'):
            raise ValueError('production stage does not declare its communication-foundation objective')
        foundation=inputs['foundation_native_graphs'];manifest=json.loads(Path(foundation['path']).read_bytes())
        ready=receipt is not None and receipt.get('foundation_ready') is True
        if ready:
            completed=receipt.get('communication_foundation')or{}
            if (completed.get('complete')is not True or (completed.get('sources')or{}).get('sha256')!=foundation['sha256']
                    or type(completed.get('graph_visits'))is not int
                    or completed['graph_visits']<completed.get('required_graph_visits',1)):
                raise ValueError('ready receipt does not prove full exposure of its declared foundation source')
        generation=canonical_json_sha256_hex({'schema':'natlang.foundation-qualification-generation/1',
            'role':role,'model_id':model_id,'source_identity_sha256':source_evidence_ref['sha256'],
            'recipe_sha256':policy_sha,'stage_id':stage['id'],'foundation_sha256':foundation['sha256']})
        cases_ref=_resolve_config_file(config['cases'],routes)
        model={'model':role,'native_participant':model_id,'production_recipe':binding(recipe_path),
            'production_stage':stage['id'],'seed_native_graphs':inputs['native_graphs'],
            'held_native_graphs':inputs['held_native_graphs'],'cases':cases_ref,
            'base':config['base'],'runtime':config['runtime'],'provider':copy.deepcopy(item['provider']),
            'port':item['port'],'provider_queue':item['provider_queue'],
            'base_inventory_sha256':model_id.split(':',1)[1],
            'foundation_qualification_generation':generation,
            'foundation_readiness_policy':'on-shared-request',
            'source_identity':{'config':binding(config_path,item['config']),
                source_evidence_key:source_evidence_ref,'owner_state':owner_state_ref,
                'owner_cursor':owner_cursor_ref,'recipe':binding(recipe_path,item['recipe']),
                'source_map':source_map_ref,'runtime_mechanism_evidence':mechanism_ref,
                'declared_paths':{'production_recipe':declared_recipe,'cases':config['cases'],
                    'seed_native_graphs':config.get('seed_native_graphs'),
                    'held_native_graphs':config.get('held_native_graphs')}}}
        requirement={'model_id':model_id,'readiness':item['readiness'],
            'trainer_recipe':item['recipe']['path'],'recipe_sha256':policy_sha,'stage_ids':[stage['id']],
            'readiness_boundary':'communication-foundation','foundation_sources':foundation,
            'foundation_epochs':params.get('epochs',1.),'forward_atol':params.get('forward_atol',1e-4),
            'gradient_atol':params.get('gradient_atol',1e-5)}
        qualification[role]={'foundation_ready':False,'source_receipt_foundation_ready':ready,
            'current_owner_checkpoint_sha256':owner_cursor['checkpoint']['sha256'],
            'source_receipt_checkpoint_sha256':checkpoint_sha,
            'receipt_matches_current_owner_checkpoint':checkpoint_sha==owner_cursor['checkpoint']['sha256'],
            'source_evidence_kind':source_evidence_key,
            'source_evidence_sha256':source_evidence_ref['sha256'],
            'qualification_generation':generation,
            'foundation_sha256':foundation['sha256'],'foundation_rows':manifest.get('rows')}
        def deployment_ref(local,declared_path):
            local={**binding(local['path'],{'path':local['path'],'sha256':local['sha256']}),
                'dependencies':local.get('dependencies',[])}
            targets=forward.get(local['path'],[])
            if len(set(targets))>1:raise ValueError('source map gives one exact file multiple remote paths')
            target=targets[0]if targets else declared_path
            if role=='mellum' and(not isinstance(target,str)or not target.startswith('/work/')):
                raise ValueError('Mellum source has no exact remote path in its config or source map')
            deps=[]
            for dep in local.get('dependencies',[]):
                local_dep=binding(dep['path'],dep);mapped=forward.get(local_dep['path'],[])
                if len(set(mapped))>1:raise ValueError('source map gives one graph dependency multiple remote paths')
                remote=mapped[0]if mapped else dep['path']
                if role=='mellum'and not remote.startswith('/work/'):
                    raise ValueError('Mellum graph dependency has no exact remote source path')
                deps.append({**local_dep,'path':remote})
            return {**local,'path':target,'dependencies':deps}
        def declared_role_input(role_key):
            name=inputs[role_key].get('binding')
            return recipe.get('input_bindings',{}).get(name,{}).get('path',inputs[role_key]['path'])
        deployed=copy.deepcopy(model)
        deployed['production_recipe']={**binding(recipe_path),'path':declared_recipe}
        deployed['cases']=deployment_ref(cases_ref,config['cases'])
        deployed['seed_native_graphs']=deployment_ref(inputs['native_graphs'],declared_role_input('native_graphs'))
        deployed['held_native_graphs']=deployment_ref(inputs['held_native_graphs'],declared_role_input('held_native_graphs'))
        deployed['handoff_requirement']={**requirement,
            'trainer_recipe':declared_recipe,
            'foundation_sources':deployment_ref(foundation,declared_role_input('foundation_native_graphs'))}
        deployment_models[role]=deployed
        model['handoff_requirement']=requirement
        raw_models.append(model)
    standard_models=[{key:value for key,value in row.items()if key!='handoff_requirement'}
        for row in raw_models]
    return raw_models,standard_models,qualification,deployment_models


def prepare(spec_path,out):
    spec=json.loads(Path(spec_path).read_bytes())
    required={'schema','runtime','provider','curriculum','curriculum_graph_relocations','roles','semantic_consumer_out',
            'semantic_curriculum_target','snapshot_curriculum_target','standard_bootstrap_target',
            'controller_target','consumer','exchange_directory','handoff_lease_seconds'}
    if set(spec)-required-{'task_packet_relocations','recipe_paths','runtime_source_archive','operator_inputs'} or not required<=set(spec) or spec['schema']!=SCHEMA:
        raise ValueError('local-cycle semantic preparation requires its full explicit two-owner input contract')
    if set(spec['roles'])!={'lfm','mellum'} or spec['consumer']!='mellum':
        raise ValueError('current placement requires local LFM, cloud Mellum consumer, and both source owners')
    runtime=Path(spec['runtime']).resolve(strict=True)
    if not (runtime/'training/neuralese/natlang_neuralese/train/native_readiness.py').is_file():
        raise ValueError('local-cycle semantic preparation needs the exact installed native runtime')
    collector=runtime/'scripts/run_bonsai_queue.py'
    if not collector.is_file() or collector.is_symlink():
        raise ValueError('local-cycle runtime is missing its exact TaskEval Python collector asset')
    collector_ref={**binding(collector),'bytes':collector.stat().st_size}
    provider=read(spec['provider'])
    if spec.get('operator_inputs') is not None:binding(spec['operator_inputs']['path'],spec['operator_inputs'])
    # The owners are local-cycle learners and intentionally have no old role
    # bootstrap. Build fresh declarations from each exact current model row and
    # source-qualified foundation requirement supplied in the descriptor.
    raw_models,standard_models,qualification,deployment_models=build_role_declarations(spec['roles'])
    by_role={row['model']:row for row in raw_models}
    deployed_by_role={row['model']:row for row in deployment_models.values()}
    ids={role:row.get('handoff_requirement',{}).get('model_id')for role,row in by_role.items()}
    if any(not isinstance(value,str)or not value for value in ids.values())or len(set(ids.values()))!=2:
        raise ValueError('RAW bootstrap does not preserve distinct actual participant identities')
    assignments={role:{'owned_models':[role],'source_host_id':spec['roles'][role]['source_host_id']}
        for role in ('lfm','mellum')}
    for role,item in spec['roles'].items():
        mechanism_ref=item.get('mechanism_evidence')
        if mechanism_ref is not None:
            mechanism_value=read(mechanism_ref)['mechanism']
            assignments[role]['verification_mechanism']=mechanism_value
    if (not isinstance(assignments['lfm']['source_host_id'],str)
            or assignments['lfm']['source_host_id']!=provider.get('local_source_host',{}).get('source_host_id')
            or assignments['mellum']['source_host_id'] not in provider.get('sandboxes',{})):
        raise ValueError('one owner must be the declared local DGX host and Mellum the existing cloud source host')
    if provider.get('coordinator_id')!=assignments['mellum']['source_host_id']:
        raise ValueError('the existing Mellum owner must remain the semantic consumer/coordinator')
    declared_exchange=spec['exchange_directory']
    if isinstance(declared_exchange,str):
        exchange_dirs={role:declared_exchange for role in ('lfm','mellum')}
    elif isinstance(declared_exchange,dict) and set(declared_exchange)=={'lfm','mellum'}:
        exchange_dirs=dict(declared_exchange)
    else:
        raise ValueError('exchange directories must be one legacy path or one exact path per source owner')
    for role,path in exchange_dirs.items():
        host=(provider.get('local_source_host') if role=='lfm'
            else provider.get('sandboxes',{}).get(spec['roles'][role]['source_host_id']))
        if (not isinstance(path,str)or not Path(path).is_absolute()or '..'in Path(path).parts
                or not _under(path,_host_roots(provider,role,spec['roles'][role]['source_host_id']))):
            raise ValueError('source-role exchange path must be under that exact owner’s declared staging roots')
    out=Path(out).resolve()
    if out.exists():raise FileExistsError('local-cycle semantic preparation needs a new immutable output directory')
    out.mkdir(parents=True)
    plans={};controls={};control_refs={};raw_states={};config_paths={};plan_targets={};cycle_targets={};owner_handoffs={}
    private_export_inventories={}
    stage_files=[]
    runtime_source_deployment=None
    runtime_source=spec.get('runtime_source_archive')
    if runtime_source is not None:
        if (not isinstance(runtime_source,dict)
                or set(runtime_source)!={'source','target','runtime_root','source_revision','archive_schema',
                    'must_extract_before_role_resume'}
                or runtime_source.get('archive_schema')!='git-archive-tar-v1'
                or runtime_source.get('must_extract_before_role_resume') is not True
                or not isinstance(runtime_source.get('source_revision'),str)
                or not runtime_source['source_revision']
                or not isinstance(runtime_source.get('runtime_root'),str)
                or not Path(runtime_source['runtime_root']).is_absolute()
                or '..'in Path(runtime_source['runtime_root']).parts):
            raise ValueError('future shared Python runtime needs its exact immutable archive and extraction target')
        archive=binding(runtime_source['source']['path'],runtime_source['source'])
        archive={**archive,'bytes':Path(archive['path']).stat().st_size}
        cloud_roots=_host_roots(provider,'mellum',spec['roles']['mellum']['source_host_id'])
        if not _under(runtime_source['target'],cloud_roots):
            raise ValueError('future Mellum Python runtime archive escaped its current source host writable roots')
        stage_files.append({'role':'mellum','kind':'python-runtime-source-archive',
            'target':runtime_source['target'],'source':archive})
        runtime_source_deployment={'schema':'natlang.native-source-runtime-deployment/1',
            'runtime_root':runtime_source['runtime_root'],'source_revision':runtime_source['source_revision'],
            'archive':archive,'archive_target':runtime_source['target'],
            'installed':False,'required_before_role_resume':True,
            'preserves_actor_runtime_binding':True,
            'actor_runtime_config_binding':by_role['mellum']['source_identity']['config'],
            'install_argv':{'mkdir':['mkdir','-p',runtime_source['runtime_root']],
                'extract':['tar','--extract','--file',runtime_source['target'],
                    '--directory',runtime_source['runtime_root']]}}
    # TaskEval imports this Python entrypoint through the frozen logical
    # NATLANG_REPO on each owner. Bind and stage the exact collector alongside
    # the already-inventoried compiled TypeScript task runtime.
    collector_assets={}
    for role in ('lfm','mellum'):
        item=spec['roles'][role];host=(provider.get('local_source_host') if role=='lfm'
            else provider['sandboxes'][item['source_host_id']])
        owner_config_path=Path(binding(item['config']['path'],item['config'])['path'])
        owner_config=json.loads(owner_config_path.read_bytes())
        actor_environment=owner_config.get('actor_environment')or{}
        actor_runtime=actor_environment.get('NATLANG_REPO')
        if (not isinstance(actor_runtime,str)or not Path(actor_runtime).is_absolute()
                or '..'in Path(actor_runtime).parts):
            raise ValueError('TaskEval collector needs the exact actor logical NATLANG_REPO from its current owner config')
        runtime_root=Path(actor_runtime)
        target=str(runtime_root/'scripts/run_bonsai_queue.py')
        readable=[*_host_roots(provider,role,item['source_host_id']),*host.get('read_roots',[]),
            *host.get('readonly_input_roots',[])]
        if not _under(target,readable):
            raise ValueError('TaskEval collector target is outside its actor runtime and owner read roots')
        collector_assets[role]={'path':target,'actor_runtime_root':actor_runtime,
            'source_python_runtime':host.get('runtime'),'kind':'task-source-file',
            'sha256':collector_ref['sha256'],'bytes':collector_ref['bytes']}
        if _under(target,_host_roots(provider,role,item['source_host_id'])):
            stage_files.append({'role':role,'source_host_id':assignments['lfm']['source_host_id'],
                'kind':'task-eval-python-collector','target':target,'source':collector_ref})
    # The receiver has the staged curriculum tree but the role plan must refer
    # to the matching cloud paths. Transfer exact local copies to those bound
    # targets; the relay's IO adapter will verify-before-skip existing files.
    cloud_role='mellum';cloud_host=assignments[cloud_role]['source_host_id']
    cloud_spec=provider['sandboxes'][cloud_host]
    cloud_roots=[*_host_roots(provider,cloud_role,cloud_host),*cloud_spec.get('read_roots',[]),
        *cloud_spec.get('readonly_input_roots',[])]
    local_model=by_role[cloud_role];remote_model=deployed_by_role[cloud_role]
    for field in ('production_recipe','cases','seed_native_graphs','held_native_graphs','foundation_sources'):
        local_value=(local_model['source_identity']['recipe'] if field=='production_recipe'
            else local_model['handoff_requirement']['foundation_sources'] if field=='foundation_sources'
            else local_model[field])
        remote_value=(remote_model['handoff_requirement']['foundation_sources'] if field=='foundation_sources'
            else remote_model[field])
        if field=='production_recipe':
            from natlang_neuralese.train.recipe import recipe_file_dependencies
            for dep in recipe_file_dependencies(local_value['path']):
                source=dep['path'];relative=Path(source).resolve().relative_to(Path(local_value['path']).resolve().parent)
                target=(remote_value['path'] if Path(source).resolve()==Path(local_value['path']).resolve()
                    else str(Path(remote_value['path']).parent/relative))
                reference={**binding(source), 'bytes':Path(source).stat().st_size}
                if not _under(target,cloud_roots):
                    raise ValueError('Mellum recipe closure target escaped cloud writable roots')
                stage_files.append({'role':cloud_role,'source_host_id':assignments['lfm']['source_host_id'],
                    'kind':'production-recipe-input','target':target,'source':reference})
            continue
        local_value={**binding(local_value['path'],{'path':local_value['path'],'sha256':local_value['sha256']}),
            'dependencies':local_value.get('dependencies',[])}
        remote_dependencies=remote_value.get('dependencies',[])
        if len(local_value['dependencies'])!=len(remote_dependencies):
            raise ValueError('cloud graph source dependency closure changed during path relocation')
        entries=[(local_value,remote_value)]
        entries.extend(zip(local_value['dependencies'],remote_dependencies))
        for source_ref,target_ref in entries:
            target=target_ref['path']
            if not _under(target,cloud_roots):
                raise ValueError('Mellum graph source target escaped cloud writable roots')
            reference={**binding(source_ref['path'],{'path':source_ref['path'],'sha256':source_ref['sha256']}),
                'bytes':Path(source_ref['path']).stat().st_size}
            if source_ref['sha256']!=target_ref['sha256']:
                raise ValueError('Mellum graph source target changed exact content identity')
            stage_files.append({'role':cloud_role,'source_host_id':assignments['lfm']['source_host_id'],
                'kind':'current-production-input','target':target,'source':reference})
    for role in ('lfm','mellum'):
        item=spec['roles'][role]
        required={'source_host_id','config','owner_state','owner_cursor','config_target',
            'cycle_plan','cycle_plan_target','state_path','cursor_path','readiness',
            'plan_target','successor_plan_target','owner_identity_path','runtime','python','recipe','recipe_target','provider',
            'provider_queue','port'}
        if (not required<=set(item) or set(item)-required-{'source_map','source_receipt',
                'source_checkpoint_identity','controls','controls_target','mechanism_evidence'}
                or ('source_receipt' in item)==('source_checkpoint_identity' in item)):
            raise ValueError('source role descriptor is incomplete: '+role)
        config_path=Path(binding(item['config']['path'],item['config'])['path'])
        config=json.loads(config_path.read_bytes())
        current_plan=read(item['cycle_plan'])
        cfg_target=item['config_target'];cycle_target=item['cycle_plan_target']
        host_roots=_host_roots(provider,role,item['source_host_id'])
        if (not isinstance(item['runtime'],str) or not Path(item['runtime']).is_absolute()
                or not isinstance(item['python'],str) or not Path(item['python']).is_absolute()):
            raise ValueError('each source host must bind its own exact runtime and Python executable')
        host=(provider['local_source_host'] if role=='lfm'
            else provider['sandboxes'][item['source_host_id']])
        if item['runtime']!=host.get('runtime'):
            raise ValueError('source role runtime differs from its actual configured host')
        configured_python=host.get('python')
        if configured_python and item['python']!=configured_python:
            raise ValueError('source role learner Python differs from the exact owner worker runtime')
        target_paths=(cfg_target,cycle_target,item['state_path'],item['cursor_path'],item['readiness'],
            item['plan_target'],item['successor_plan_target'],item['owner_identity_path'])
        if item.get('controls') is not None:
            if not isinstance(item.get('controls_target'),str):
                raise ValueError('a supplied role-control source needs an exact target path')
            target_paths=(*target_paths,item['controls_target'])
        if (not all(isinstance(value,str)and Path(value).is_absolute()and '..'not in Path(value).parts
                for value in target_paths)
                or any(not _under(value,host_roots)for value in target_paths)):
            raise ValueError('source role targets must be exact absolute paths under that owner writable root')
        if current_plan.get('rl_config')!=config.get('path',cfg_target):
            # A transported cloud config keeps its actual remote path while the
            # local LFM config may itself already be at the target path.
            if role=='mellum' and current_plan.get('rl_config')!=cfg_target:
                raise ValueError('Mellum cycle plan and captured current config path disagree')
            if role=='lfm' and not Path(current_plan.get('rl_config','')).is_file():
                raise ValueError('LFM cycle plan does not point at an existing current config')
        if role=='lfm' and current_plan['rl_config']!=cfg_target:
            raise ValueError('local LFM config target must equal its actual current config path')
        if current_plan['out']+'/cursor.json'!=item['cursor_path']:
            raise ValueError('cycle cursor path differs from the actual current cycle plan')
        if config['out']+'/state.json'!=item['state_path']:
            raise ValueError('production state path differs from the actual current model config')
        if str(Path(current_plan['out'])/'owner-identity.json')!=item['owner_identity_path']:
            raise ValueError('owner identity path differs from the actual current cycle owner namespace')
        if current_plan.get('optimizer_owner_lock') is None:
            raise ValueError('actual cycle plan omitted its optimizer owner lock')
        row=by_role[role]
        requirement=row['handoff_requirement']
        model=row
        for key in ('production_recipe','cases','seed_native_graphs','held_native_graphs'):
            config_key={'production_recipe':'production_recipe','cases':'cases',
                'seed_native_graphs':'seed_native_graphs','held_native_graphs':'held_native_graphs'}[key]
            allowed={model[key]['path']}
            if key=='production_recipe':allowed.add(item['recipe_target'])
            declared=model.get('source_identity',{}).get('declared_paths',{}).get(config_key)
            if isinstance(declared,str):allowed.add(declared)
            if config.get(config_key) not in (None,*allowed):
                raise ValueError('fresh role declaration source differs from the actual current config: '+config_key)
        if model.get('base')!=config.get('base') or model.get('runtime')!=config.get('runtime'):
            raise ValueError('fresh role model base/runtime differs from the actual current config')
        if item['readiness']!=requirement.get('readiness'):
            raise ValueError('role readiness path differs from the exact current RAW bootstrap requirement')
        base={'schema':'natlang.native-raw-rl-bootstrap/1','models':list(by_role.values()),'owned_models':[role],
            'role_exchange':{'schema':EXCHANGE,'role':role,
                'roles':copy.deepcopy(assignments),'directory':exchange_dirs[role],
                'semantic_consumer':'mellum'}}
        plan_target=item['plan_target']
        if not isinstance(plan_target,str)or not _under(plan_target,host_roots):
            raise ValueError('source role plan target must remain under its source owner writable root')
        local_plan_path=out/'source-plans'/(role+'-base-plan.json')
        source={'schema':'natlang.native-local-cycle-source/1',
            'cycles_plan':cycle_target,
            'rl_config':{'path':cfg_target,'sha256':binding(config_path,item['config'])['sha256']},
            'readiness':item['readiness'],'state_path':item['state_path'],
            'cursor_path':item['cursor_path'],'optimizer_owner_lock':current_plan['optimizer_owner_lock'],
            'owner_identity_path':item['owner_identity_path'],
            'return_directory':exchange_dirs[role]}
        base['local_cycle_source']=source
        # The role plan is immutable once emitted; bind the full descriptor in
        # its first write rather than rewriting a previously bound artifact.
        base_local_ref=emit(local_plan_path,base)
        deployed_base=copy.deepcopy(base);deployed_base['models']=copy.deepcopy(list(deployed_by_role.values()))
        deployed_base_path=out/'source-plans'/(role+'-deployment-plan.json')
        deployed_base_local_ref=emit(deployed_base_path,deployed_base)
        base_remote_ref={'path':plan_target,'sha256':deployed_base_local_ref['sha256'],
            'bytes':Path(deployed_base_path).stat().st_size}
        cycle_plan=copy.deepcopy(current_plan)
        cycle_plan['rl_config']=cfg_target
        cycle_plan['local_cycle_role_plan']=base_remote_ref
        cycle_path=out/'cycle-plans'/(role+'-cycles.json')
        cycle_local_ref=emit(cycle_path,cycle_plan)
        control_ref=None
        if item.get('controls') is not None:
            control_ref=binding(item['controls']['path'],item['controls'])
            control_ref={**control_ref,'bytes':Path(control_ref['path']).stat().st_size}
            if not _under(item['controls_target'],host_roots):
                raise ValueError('source role control target must remain under its owner writable root')
        plans[role]=base_local_ref;controls[role]=item.get('controls');control_refs[role]=control_ref
        raw_states[role]=item['state_path'];config_paths[role]=str(config_path)
        owner_handoffs[role]={'current_config':binding(config_path,item['config']),
            'current_cycle_plan':binding(item['cycle_plan']['path'],item['cycle_plan']),
            'optimizer_owner_lock':current_plan['optimizer_owner_lock'],
            'owner_identity_path':item['owner_identity_path'],'required_boundary':'whole-update-or-stage',
            'new_cycle_plan':{'path':cycle_target,'sha256':cycle_local_ref['sha256'],
                'bytes':Path(cycle_path).stat().st_size},'applied':False}
        inventory_path=_private_export_inventory_path(provider,role,ids[role])
        private_export_inventories[role]=inventory_path
        owner_handoffs[role]['full_state_export']=_bundle_export_descriptor(
            role,item,ids[role],provider,inventory_path)
        plan_targets[role]={'base':base_remote_ref,'successor':item['successor_plan_target']}
        cycle_targets[role]={'path':cycle_target,'sha256':cycle_local_ref['sha256'],
            'bytes':Path(cycle_path).stat().st_size}
        stage_files.extend([{'role':role,'source_host_id':assignments['lfm']['source_host_id'],
                'kind':'base-role-plan','target':plan_target,'source':deployed_base_local_ref},
            {'role':role,'kind':'cycle-plan','target':cycle_target,'source':cycle_local_ref}])
        if control_ref is not None:
            stage_files.append({'role':role,'kind':'role-controls','target':item['controls_target'],'source':control_ref})
    task_relocations=spec.get('task_packet_relocations')
    recipe_paths={role:item for role,item in spec.get('recipe_paths',{}).items()}
    curriculum_result=materialize_current_task_policy(spec['curriculum']['path'],plans,config_paths,'mellum',
        out/'current-task-policy',task_relocations,recipe_paths)
    curriculum_prepared=json.loads(Path(curriculum_result['path']).read_bytes())
    curriculum_ref=curriculum_prepared['curriculum']
    mellum_roots=_host_roots(provider,'mellum',assignments['mellum']['source_host_id'])
    for key in ('semantic_consumer_out','semantic_curriculum_target','snapshot_curriculum_target',
            'standard_bootstrap_target','controller_target'):
        if not _under(spec[key],mellum_roots):
            raise ValueError('semantic consumer target must remain under its current writable host root: '+key)
    base_curriculum=read(spec['curriculum'])
    graph_relocations=spec['curriculum_graph_relocations']
    if set(graph_relocations)!={'pairs','held_pairs'}:
        raise ValueError('paired curriculum transport must preserve both exact train and held sources')
    remote_curriculum=json.loads(Path(curriculum_ref['path']).read_bytes());graph_proofs={}
    for key in ('pairs','held_pairs'):
        relocation=graph_relocations[key]
        if not isinstance(relocation,dict)or set(relocation)!={'source','local','target'} or relocation['source']!=base_curriculum[key]:
            raise ValueError('paired graph transport differs from the exact source curriculum: '+key)
        local_ref=binding(relocation['local']['path'],relocation['local'])
        local_ref={**local_ref,'bytes':Path(local_ref['path']).stat().st_size}
        if local_ref['sha256']!=relocation['source']['sha256']:
            raise ValueError('paired graph local copy differs from exact source curriculum: '+key)
        target=relocation['target']
        if not _under(target,mellum_roots):
            raise ValueError('paired train/held source target escaped Mellum writable roots: '+key)
        remote_curriculum[key]={'path':target,'sha256':local_ref['sha256'],'bytes':local_ref['bytes']}
        graph_proofs[key]={'declared':relocation['source'],'local':local_ref,'target':target}
        stage_files.append({'role':'mellum','kind':'paired-'+key,'target':target,'source':local_ref})
    remote_curriculum.setdefault('operator_bindings',{})['paired_source_transport']={
        'schema':'natlang.native-paired-curriculum-source-transport/1','sources':graph_proofs}
    curriculum_ref=emit(out/'remote-consumer'/'full-curriculum.json',remote_curriculum)
    stage_files.append({'role':'mellum','kind':'full-paired-curriculum',
        'target':spec['semantic_curriculum_target'],'source':curriculum_ref})
    for item in curriculum_prepared.get('staged_task_packets',[]):
        if not _under(item['target'],mellum_roots):
            raise ValueError('relocated protected task packet escaped Mellum writable root')
        packet_source={**item['source'],'bytes':Path(item['source']['path']).stat().st_size}
        stage_files.append({'role':item['role'],'kind':'task-eval-packet',
            'target':item['target'],'source':packet_source})
    consumer_host=provider['sandboxes'][assignments['mellum']['source_host_id']]
    consumer_roots=[*consumer_host.get('read_roots',[]),*consumer_host.get('readonly_input_roots',[]),
        *consumer_host.get('write_roots',[])]
    staged_packet_targets={row['target']for row in curriculum_prepared.get('staged_task_packets',[])}
    for role,proof in curriculum_prepared['task_eval_sources'].items():
        packet_path=proof['task_eval']['packet']['path']
        if packet_path in staged_packet_targets:continue
        if not _under(packet_path,consumer_roots):
            raise ValueError('semantic task packet is not present in a declared consumer input root and has no staged relocation')
    # The existing scheduler validator reads all declared packet bytes locally.
    # Validate local relocated copies there, then bind the separately staged
    # consumer curriculum to its exact remote target paths.
    validation_curriculum=json.loads(Path(curriculum_prepared['curriculum']['path']).read_bytes())
    for proof in curriculum_prepared['task_eval_sources'].values():
        relocation=proof['task_packet'].get('relocation')or{}
        if 'target'in relocation:
            model_id=proof['participant']
            validation_curriculum['models'][model_id]['task_eval']['packet']=relocation['local']
            if model_id==curriculum_prepared['task_eval_sources']['mellum']['participant']:
                validation_curriculum['task_eval']=copy.deepcopy(validation_curriculum['models'][model_id]['task_eval'])
    validation_path=out/'current-task-policy'/'validation-curriculum.json'
    write_json_durable(validation_path,validation_curriculum)
    queue=str(Path(exchange_dirs['mellum'])/'bridge-ipc')
    bridge=[spec['roles']['mellum']['python'],'-m','natlang_neuralese.rl.native_bridge','--queue',queue,
        '--request','{request}','--result','{result}','--stop-file','{stop_file}']
    standard={'schema':'natlang.native-rl-bootstrap/1','models':standard_models,
        'semantic_refresh_config':curriculum_ref,'refresh_command':bridge,
        'refresh_provider':by_role['mellum'].get('provider')}
    if not standard['refresh_provider']:
        raise ValueError('current Mellum model descriptor must retain its actual provider binding')
    standard_path=out/'native-rl-bootstrap.json'
    standard_ref=emit(standard_path,standard)
    deployed_standard={'schema':'natlang.native-rl-bootstrap/1',
        'models':[{key:value for key,value in row.items()if key!='handoff_requirement'}
            for row in deployed_by_role.values()],
        'semantic_refresh_config':curriculum_ref,'refresh_command':bridge,
        'refresh_provider':by_role['mellum'].get('provider')}
    deployed_standard_ref=emit(out/'native-rl-bootstrap-deployment.json',deployed_standard)
    standard_target_ref={'path':spec['standard_bootstrap_target'],'sha256':deployed_standard_ref['sha256'],
        'bytes':Path(deployed_standard_ref['path']).stat().st_size}
    stage_files.append({'role':'mellum','kind':'standard-rl-bootstrap',
        'source_host_id':assignments['mellum']['source_host_id'],
        'target':spec['standard_bootstrap_target'],'source':deployed_standard_ref})
    result=prepare_role_scheduler(plans,controls,str(validation_path),spec['provider']['path'],
        out/'scheduler',raw_states=raw_states,runtime=runtime,python=spec['roles']['mellum']['python'],
        rl_bootstrap=standard_ref,handoff_lease_seconds=spec['handoff_lease_seconds'])
    successor_by_role={}
    for row in result['source_plan_successions']:
        role=row['role'];value=json.loads(Path(row['successor']['path']).read_bytes())
        value['models']=copy.deepcopy(list(deployed_by_role.values()))
        successor_local=emit(out/'source-plans'/(role+'-successor-deployment.json'),value)
        successor_by_role[role]=successor_local
    relay=json.loads(Path(result['relay']['path']).read_bytes())
    controller=json.loads(Path(result['controller']['path']).read_bytes())
    controller['workdir']=spec['roles']['mellum']['runtime']
    controller['command'][0]=spec['roles']['mellum']['python']
    controller['snapshot_successor']['command'][0]=spec['roles']['mellum']['python']
    snapshot_value=json.loads((Path(result['controller']['path']).parent/'snapshot-curriculum.json').read_text())
    remote_full_value=json.loads(Path(curriculum_ref['path']).read_text())
    snapshot_value['models']=copy.deepcopy(remote_full_value['models'])
    snapshot_value['task_eval']=copy.deepcopy(remote_full_value['task_eval'])
    snapshot_remote_path=out/'remote-consumer'/'snapshot-curriculum.json'
    snapshot_curriculum_local=emit(snapshot_remote_path,snapshot_value)
    snapshot_curriculum_remote={'path':spec['snapshot_curriculum_target'],
        'sha256':snapshot_curriculum_local['sha256'],'bytes':Path(snapshot_curriculum_local['path']).stat().st_size}
    full_curriculum_remote={'path':spec['semantic_curriculum_target'],
        'sha256':curriculum_ref['sha256'],'bytes':Path(curriculum_ref['path']).stat().st_size}
    controller['curriculum']=spec['snapshot_curriculum_target']
    controller['snapshot_successor']['curriculum']=full_curriculum_remote
    controller['out']=spec['semantic_consumer_out']
    controller['semantic_out']=spec['semantic_consumer_out']+'/semantic/preparation'
    controller['raw_rl_out']=str(Path(spec['roles']['mellum']['state_path']).parent)
    stage_files.append({'role':'mellum','kind':'snapshot-curriculum',
        'target':spec['snapshot_curriculum_target'],'source':snapshot_curriculum_local})
    for role in ('lfm','mellum'):
        successor_local=successor_by_role[role]
        successor_path=Path(successor_local['path']);target=spec['roles'][role]['successor_plan_target']
        successor_local={**successor_local,'bytes':successor_path.stat().st_size}
        roots=_host_roots(provider,role,assignments[role]['source_host_id'])
        if not _under(target,roots):
            raise ValueError('scheduler successor target must remain under its owner writable root')
        successor_remote={'path':target,'sha256':successor_local['sha256'],'bytes':successor_local['bytes']}
        relay['hosts'][role]['plan']=successor_remote
        resource=(relay['provider']['sandboxes'][assignments[role]['source_host_id']]
            if assignments[role]['source_host_id'] in relay['provider']['sandboxes']
            else relay['provider']['local_source_host'])
        if control_refs[role] is not None:
            resource['role_controls']={'path':spec['roles'][role]['controls_target'],
                'sha256':control_refs[role]['sha256'],'bytes':Path(control_refs[role]['path']).stat().st_size}
        else:
            resource.pop('role_controls',None)
        relay['local_cycle_returns']['role_plans'][role]=successor_remote
        stage_files.append({'role':role,'kind':'successor-role-plan','target':target,'source':successor_local})
    controller['raw_rl_bootstrap']=relay['hosts']['mellum']['plan']
    controller['snapshot_successor']['rl_bootstrap']=controller['raw_rl_bootstrap']
    relay['preparation_curriculum']=snapshot_curriculum_remote
    relay_provider=relay.setdefault('provider',copy.deepcopy(provider))
    local_provider=relay_provider.get('local_source_host')
    if not isinstance(local_provider,dict):
        raise ValueError('source relay lost its exact local owner provider declaration')
    local_provider['private_export_inventory']=private_export_inventories['lfm']
    local_provider['private_export_role']='lfm'
    relay['task_eval_collector_assets']=collector_assets
    relay['local_cycle_returns']['completion_path']=spec['semantic_consumer_out']+'/joint/shared-stage-completion.json'
    for staged in stage_files:
        staged.setdefault('source_host_id',assignments['lfm']['source_host_id'])
        staged['target_host_id']=assignments[staged['role']]['source_host_id']
    stage_declaration={'schema':'natlang.native-role-preparation-files/1',
        'source_host_id':assignments['lfm']['source_host_id'],'files':copy.deepcopy(stage_files),
        'receipt_path':spec['semantic_consumer_out']+'/prepared-files-receipt.json'}
    stage_declaration['inventory_sha256']=canonical_json_sha256_hex({
        'schema':stage_declaration['schema'],'source_host_id':stage_declaration['source_host_id'],
        'files':stage_declaration['files']})
    relay['preparation_stage_files']=stage_declaration
    relay_path=out/'scheduler'/'relay-local-cloud.json';controller_path=out/'scheduler'/'controller-local-cloud.json'
    relay_ref=emit(relay_path,relay);controller_ref=emit(controller_path,controller)
    controller_remote_ref={'path':spec['controller_target'],'sha256':controller_ref['sha256'],
        'bytes':Path(controller_ref['path']).stat().st_size}
    stage_files.append({'role':'mellum','kind':'semantic-controller',
        'target':spec['controller_target'],'source':controller_ref})
    prepared_value={'schema':SCHEMA,
        'inputs':{'spec':binding(spec_path),'runtime':str(runtime),'provider':spec['provider'],
            **({'operator_inputs':spec['operator_inputs']}if spec.get('operator_inputs')is not None else {}),
            'curriculum':spec['curriculum'],'generated_standard_rl_bootstrap':standard_ref},
        'participant_ids':ids,'qualification':qualification,
        'source_hosts':{role:assignments[role]['source_host_id']for role in assignments},
        'curriculum':{'local':curriculum_ref,'consumer':{'path':spec['semantic_curriculum_target'],
            'sha256':curriculum_ref['sha256'],'bytes':Path(curriculum_ref['path']).stat().st_size}},
        'relay':relay_ref,'controller':{'local':controller_ref,'consumer':controller_remote_ref},
        'task_eval_python_collectors':collector_assets,
        'future_source_runtime_deployment':runtime_source_deployment,
        'source_plan_targets':plan_targets,'cycle_plan_targets':cycle_targets,
        'owner_handoffs':owner_handoffs,
        'full_state_private_exports':{role:owner_handoffs[role]['full_state_export']for role in ('lfm','mellum')},
        'stage_files':stage_files,
        'commands':{'lfm_cycle':{'mode':'local-frozen-runtime-subprocess',
                'source_host_id':assignments['lfm']['source_host_id'],'python':spec['roles']['lfm']['python'],
                'argv':[spec['roles']['lfm']['python'],'-m','natlang_neuralese.rl.local_cycles','--plan',cycle_targets['lfm']['path']],
                'cwd':spec['roles']['lfm']['runtime']},
            'mellum_cycle':{'mode':'existing-cloud-provider-exec',
                'source_host_id':assignments['mellum']['source_host_id'],'python':spec['roles']['mellum']['python'],
                'argv':[spec['roles']['mellum']['python'],'-m','natlang_neuralese.rl.local_cycles','--plan',cycle_targets['mellum']['path']],
                'cwd':spec['roles']['mellum']['runtime']},
            'relay':[spec['roles']['lfm']['python'],str(runtime/'scripts/coreweave/native_role_exchange.py'),
                '--config',relay_ref['path']],
            'semantic_controller':[spec['roles']['mellum']['python'],'-m',
                'natlang_neuralese.train.native_readiness','--config',spec['controller_target']]},
        'owner_changes':0,'checkpoint_decodes':0,'model_allocations':0,'readiness_granted':False,
        'activated':False,'qualification_claimed':False}
    prepared_value['prepared']=emit(out/'prepared.json',prepared_value)
    return prepared_value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(prepare(args.spec,args.out),sort_keys=True))


if __name__=='__main__':main()
