"""Rebind only the capacity proof of the same queued, uncredited append."""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

from . import native_resident as resident
from ..common.hashing import sha256_hex
from ..common.jsonio import write_bytes_durable


@contextmanager
def _cache_root(root):
    """Scope the existing index verifier to this isolated CPU helper call."""
    name='NATLANG_ROW_INDEX_CACHE';previous=os.environ.get(name)
    os.environ[name]=str(root)
    try:yield
    finally:
        if previous is None:os.environ.pop(name,None)
        else:os.environ[name]=previous


def _relocate_indexes(indexes, old_root):
    from ..common import row_index
    actual=row_index.cache_dir().resolve();old_root=Path(old_root).resolve()
    with _cache_root(old_root):resident._check_indexes(indexes)
    actual.mkdir(parents=True,exist_ok=True)
    relocated=[];joins=[]
    for witness in indexes:
        source=resident.bound(witness['cache'])
        target=row_index.entry_path('native-graphs',resident.digest(witness['key'])).resolve()
        if not target.exists():
            fd,name=tempfile.mkstemp(prefix=target.name+'.',suffix='.tmp',dir=actual)
            temporary=Path(name)
            try:
                with os.fdopen(fd,'wb') as out,source.open('rb') as stream:
                    shutil.copyfileobj(stream,out,length=1024*1024);out.flush();os.fsync(out.fileno())
                if resident.sha256_file_hex(temporary)!=witness['cache']['sha256']:
                    raise ValueError('resident index changed while copying verified bytes')
                # Only this immutable new copy is linked for no-clobber atomic
                # publication. The original cache is never hardlinked.
                try:os.link(temporary,target)
                except FileExistsError:pass
            finally:temporary.unlink(missing_ok=True)
        replacement={**witness,'cache':resident.binding(target)}
        resident._check_indexes([replacement])
        with target.open('rb') as stream:os.fsync(stream.fileno())
        relocated.append(replacement)
        joins.append({'old':witness['cache'],'new':replacement['cache'],
            'key':witness['key'],'metadata_sha256':witness['metadata_sha256']})
    fd=os.open(actual,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    resident._check_indexes(relocated)
    return relocated,{'old_root':str(old_root),'learner_root':str(actual),'witnesses':joins}


def refresh(directory, request, *, runtime_package=None, old_cache_root=None, publish=True):
    if type(publish) is not bool:
        raise ValueError('resident geometry refresh publish must be an explicit boolean')
    directory=Path(directory).resolve()
    package=Path(runtime_package or Path(resident.__file__).resolve().parents[1])
    with resident.locked(directory):
        state=resident.current(directory)
        plan=(state['payload']['plan'] if state else
            json.loads((directory.parent/'recipe-plan.json').read_text()))
        predecessor=state['generation'] if state else resident.digest(plan)
        initial_ref=resident.binding(directory/'initial.json')
        initial_bytes=resident.bound(initial_ref).read_bytes()
        if sha256_hex(initial_bytes)!=initial_ref['sha256']:
            raise ValueError('resident initial changed while observing geometry authority')
        initial=json.loads(initial_bytes)
        pointer=json.loads((directory/'pending.json').read_text())
    value=json.loads(resident.bound(request).read_text())
    payload=value.get('payload')
    if value.get('schema')!=resident.SCHEMA or resident.digest(payload)!=value.get('generation'):
        raise ValueError('resident geometry refresh requires an intact prepared request')
    stage=payload['stage_id']
    if (payload['predecessor']!=predecessor or initial['stage_id']!=stage
            or not state and initial['plan_sha256']!=resident.digest(plan)):
        raise ValueError('resident geometry refresh current stage/generation differs')
    if resident.code_inventory(package)!=plan['code']:
        raise ValueError('resident geometry refresh runtime differs')
    resident._contract(plan,payload['plan'],stage)
    if payload['geometry_identity']!=initial['geometry_identity']:
        raise ValueError('resident geometry refresh requires the same actual geometry identity')
    for role,old_role in (('train','native_graphs'),('held','held_native_graphs')):
        expected=payload['plan']['stage_inputs'][stage][old_role]
        if resident.binding(resident.bound(payload[role]))!={k:expected[k] for k in ('path','sha256')}:
            raise ValueError('resident geometry refresh source differs from prepared plan')
        prior=plan['stage_inputs'][stage][old_role]
        if resident.binding(resident.bound(payload['prior_'+role]))!={k:prior[k] for k in ('path','sha256')}:
            raise ValueError('resident geometry refresh predecessor source differs')
    from .recipe import load_recipe,native_recipe_binding,resolve_stage_inputs
    authority=state['payload']['recipe'] if state else initial['recipe']
    if (native_recipe_binding(authority['path'],plan['recipe_sha256'])!=authority
            or load_recipe(Path(authority['path']))!=plan['recipe']):
        raise ValueError('resident geometry refresh current recipe authority changed')
    recipe=payload['recipe']
    resident.bound({'path':recipe['path'],'sha256':recipe['file_sha256']})
    if (native_recipe_binding(recipe['path'],plan['recipe_sha256'])!=recipe
            or load_recipe(Path(recipe['path']))!=payload['plan']['recipe']):
        raise ValueError('resident geometry refresh prepared recipe changed')
    resolved={s['id']:({} if s.get('admitted') is False else
        resolve_stage_inputs(payload['plan']['recipe'],s,{'records':None,'pieces':None,'text_data':None}))
        for s in payload['plan']['recipe']['stages']}
    if resolved!=payload['plan']['stage_inputs']:
        raise ValueError('resident geometry refresh resolved inputs changed')
    for reference in (payload['geometry_qualification'],initial['geometry_qualification']):
        limits=resident.geometry_capacity(reference,identity=initial['geometry_identity'],code=plan['code'])
        if (set(payload['geometry'])!=set(limits)
                or any(type(v) is not int or v<0 for v in payload['geometry'].values())
                or any(payload['geometry'][key]>limits[key] for key in limits)):
            raise ValueError('resident geometry refresh exceeds qualified whole-graph geometry')
    if old_cache_root is None:
        resident._check_indexes(payload['indexes']);indexes=payload['indexes'];relocation=None
    else:indexes,relocation=_relocate_indexes(payload['indexes'],old_cache_root)
    replacement=copy.deepcopy(value)
    replacement['payload']['geometry_qualification']=initial['geometry_qualification']
    replacement['payload']['indexes']=indexes
    replacement['generation']=resident.digest(replacement['payload'])
    requests=directory/'requests'
    new_path=requests/(replacement['generation']+'.json')
    proof_path=requests/(replacement['generation']+'-'+request['sha256']+'-geometry-refresh.json')
    with resident.locked(directory):
        actual=resident.current(directory)
        generation=actual['generation'] if actual else resident.digest(json.loads(
            (directory.parent/'recipe-plan.json').read_text()))
        if generation!=predecessor or resident.binding(directory/'initial.json')!=initial_ref:
            raise ValueError('resident geometry refresh authority changed during verification')
        # A retry may finish only this exact replacement of this exact request.
        actual_pointer=json.loads((directory/'pending.json').read_text())
        if actual_pointer!=pointer:
            raise ValueError('resident pending obligation changed during geometry refresh')
        if pointer!=request:
            if not new_path.exists() or pointer!=resident.binding(new_path):
                raise ValueError('resident geometry refresh cannot supersede another pending obligation')
        if new_path.exists():
            if json.loads(new_path.read_text())!=replacement:
                raise ValueError('resident geometry refresh immutable request differs')
        else:resident.write_json(new_path,replacement)
        new_ref=resident.binding(new_path)
        proof={'schema':'natlang.native-resident-geometry-refresh/1','request':request,
            'replacement':new_ref,'predecessor':predecessor,
            'old_geometry_qualification':payload['geometry_qualification'],
            'new_geometry_qualification':initial['geometry_qualification'],
            'payload_changes':['geometry_qualification']+(['indexes'] if indexes!=payload['indexes'] else []),
            'scope':'same prepared append; no earned exposure or numerical requalification'}
        if relocation:proof['index_cache_relocation']=relocation
        if proof_path.exists():
            previous=json.loads(proof_path.read_text())
            if {k:v for k,v in previous.items() if k not in ('initial','original_initial')}!=proof:
                raise ValueError('resident geometry refresh immutable receipt differs')
            historical=resident.bound(previous['initial']).read_bytes()
            if (sha256_hex(historical)!=previous['original_initial']['sha256']
                    or previous['original_initial']['path']!=initial_ref['path']):
                raise ValueError('resident geometry refresh historical authority snapshot differs')
            historical=json.loads(historical)
            if any(historical[key]!=initial[key] for key in
                    ('recipe','stage_id','plan_sha256','geometry_identity','geometry_qualification')):
                raise ValueError('resident geometry refresh historical authority contract differs')
        else:
            snapshot=requests/('geometry-initial-'+initial_ref['sha256']+'.json')
            if snapshot.exists():
                if snapshot.read_bytes()!=initial_bytes:
                    raise ValueError('resident geometry refresh immutable authority snapshot differs')
            else:write_bytes_durable(snapshot,initial_bytes)
            resident.write_json(proof_path,{**proof,'initial':resident.binding(snapshot),
                'original_initial':initial_ref})
        if publish:resident.write_json(directory/'pending.json',new_ref)
        return {'request':new_ref,'receipt':resident.binding(proof_path),'published':publish}


def recover_saved_workspace(directory, *, checkpoint, qualification, failed_request,
                            failed_result, captured_owner, out, reason, runtime_package=None):
    """Recover only qualified FFN workspace metadata; keep the failed owner failed."""
    import time
    from ..rl import native_bridge as bridge, native_raw_handoff as hand
    from ..rl.native_raw_policy import require_raw_training
    from .checkpoint_policy import CheckpointPolicy

    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('saved workspace recovery needs its practical reason')
    directory = Path(directory).resolve()
    package = Path(runtime_package or Path(resident.__file__).resolve().parents[1])
    out = Path(out)
    control = out.parent / (out.stem + '-control')
    for reference in (checkpoint, qualification, failed_request, failed_result, captured_owner):
        path = resident.bound(reference)
        if 'bytes' in reference and path.stat().st_size != reference['bytes']:
            raise ValueError('saved workspace recovery input byte count changed')
    request = bridge.read_request(resident.bound(failed_request))
    result = json.loads(resident.bound(failed_result).read_bytes())
    config = json.loads(resident.bound(request['config']).read_bytes())
    intent, _, identities = hand._captured_owner_controls(captured_owner)
    if (request.get('kind') != 'round' or result.get('schema') != bridge.RESULT_SCHEMA
            or result.get('status') != 'failed' or result.get('kind') != 'round'
            or result.get('request_id') != request['request_id']
            or result.get('request_sha256') != failed_request['sha256']
            or Path(config.get('resident_data_directory', '')).resolve() != directory
            or Path(config['out']).resolve() != Path(request['out']).resolve()
            or any(checkpoint[key] != request['parent_checkpoint'][key] for key in ('path', 'sha256'))):
        raise ValueError('saved workspace recovery lost its actual failed operation/parent')

    learner_out = Path(request['out']) / 'learner'
    if ((learner_out / 'progress.json').exists()
            or any(Path(item['path']) == learner_out / 'progress.json' for item in result.get('artifacts', []))
            or any(Path(item['path']) == learner_out / 'checkpoint.pt' and item['sha256'] != checkpoint['sha256']
                   for item in result.get('artifacts', []))):
        raise ValueError('saved workspace recovery cannot discard newer optimizer progress')

    def require_exited():
        if any(hand.same_process(owner, actual := hand.observe(owner['pid']))
               and actual['state'] not in ('Z', 'X') for owner in identities):
            raise ValueError('saved workspace recovery requires every actual owner to exit')

    def artifact(reference):
        actual = resident.binding(resident.bound(reference))
        if not any(all(item.get(key) == actual[key] for key in ('path', 'sha256'))
                   for item in result.get('artifacts', [])):
            raise ValueError('workspace qualification is absent from actual failed operation artifacts')
        return json.loads(resident.bound(reference).read_bytes())

    require_exited()
    transaction = {'checkpoint': checkpoint, 'qualification': qualification,
        'failed_request': failed_request, 'failed_result': failed_result,
        'captured_owner': captured_owner, 'reason': reason, 'directory': str(directory)}
    with resident.locked(directory):
        if out.exists():
            committed = json.loads(out.read_bytes())
            old = json.loads(resident.bound(committed['intent']).read_bytes())
            if old.get('transaction') != transaction:
                raise ValueError('saved workspace recovery retry changed its exact transaction')
            candidate = json.loads(resident.bound(committed['candidate']).read_bytes())
            actual = json.loads((directory / 'initial.json').read_bytes())
            # A later normal restart may update owner/history but must retain
            # the exact installed workspace and this immutable recovery intent.
            if any(actual.get(key) != candidate.get(key) for key in
                   ('geometry_identity', 'geometry_qualification', 'geometry_recoveries')):
                raise ValueError('saved workspace recovery was superseded by another authority')
            return resident.binding(out)
        intent_path = control / 'intent.json'
        if intent_path.exists():
            unfinished = json.loads(intent_path.read_bytes())
            if (unfinished.get('schema') != 'natlang.saved-workspace-recovery-intent/1'
                    or unfinished.get('transaction') != transaction
                    or unfinished.get('actor_source') != resident.binding(__file__)):
                raise ValueError('saved workspace recovery unfinished transaction changed')
            before = json.loads(resident.bound(unfinished['original_initial']).read_bytes())
            expected = {**before, 'geometry_identity': unfinished['new_geometry_identity'],
                'geometry_qualification': unfinished['new_geometry_qualification'],
                'geometry_recoveries': [*(before.get('geometry_recoveries') or []), resident.binding(intent_path)]}
            current_initial = json.loads((directory / 'initial.json').read_bytes())
            if current_initial == expected:
                if (resident.binding(directory / 'current.json') != unfinished['original_current']
                        or (resident.binding(directory / 'pending.json') if (directory / 'pending.json').exists() else None)
                           != unfinished['retained_pending']):
                    raise ValueError('resident data authority changed during unfinished workspace commit')
                candidate_path = control / 'candidate-initial.json'
                if json.loads(resident.bound(resident.binding(candidate_path)).read_bytes()) != expected:
                    raise ValueError('saved workspace candidate changed during unfinished commit')
                resident.write_json(out, {'schema': 'natlang.saved-workspace-recovery/1',
                    'intent': resident.binding(intent_path), 'candidate': resident.binding(candidate_path),
                    'installed_initial': resident.binding(directory / 'initial.json'),
                    'changed_fields': ['geometry_identity', 'geometry_qualification', 'geometry_recoveries'],
                    'training_credit': 0, 'readiness_granted': False, 'original_operation_still_failed': True})
                return resident.binding(out)
            if current_initial != before:
                raise ValueError('saved workspace recovery found another initial authority')
        initial_ref = resident.binding(directory / 'initial.json')
        initial = json.loads(resident.bound(initial_ref).read_bytes())
        state_ref = resident.binding(directory / 'current.json')
        state = resident.current(directory)
        pending = resident.binding(directory / 'pending.json') if (directory / 'pending.json').exists() else None
        owner = next((item for item in identities if hand.same_process(initial.get('owner', {}), item)), None)
        if owner is None or resident.owner_alive(initial.get('owner') or {}):
            raise ValueError('resident initial lost its actual exited captured optimizer owner')
        argv = hand.process_argv(owner)
        learner = Path(request['out']) / 'learner'
        if (argv.count('--out') != 1 or argv[argv.index('--out') + 1] != str(learner)
                or 'natlang_neuralese.train.native_state_sft' not in argv
                or config.get('production_stage') != initial['stage_id']
                or state['payload']['stage_id'] != initial['stage_id']):
            raise ValueError('saved workspace recovery captured a different actual trainer/stage')
    plan = state['payload']['plan']
    if resident.code_inventory(package) != plan['code']:
        raise ValueError('saved workspace recovery frozen runtime changed')
    certificate = artifact(qualification)
    proposed = certificate.get('identity')
    previous = initial.get('geometry_identity')
    if (not isinstance(proposed, dict) or not isinstance(previous, dict)
            or {k: v for k, v in proposed.items() if k != 'ffn_chunk_tokens'} !=
               {k: v for k, v in previous.items() if k != 'ffn_chunk_tokens'}
            or type(proposed.get('ffn_chunk_tokens')) is not int
            or proposed['ffn_chunk_tokens'] < 1
            or config['trainer_parameters'].get('ffn_chunk_tokens') != proposed['ffn_chunk_tokens']):
        raise ValueError('saved workspace recovery permits only the explicitly qualified FFN chunk change')
    limits = resident.geometry_capacity(qualification, identity=proposed, code=plan['code'])
    proof = artifact(certificate['evidence'])
    if (proof.get('schema') != 'natlang.native-memory-qualification/1'
            or proof.get('passed') is not True or proof.get('optimizer_updates') != 0
            or proof.get('parent_checkpoint_sha256') != checkpoint['sha256']
            or proof.get('identity') != proposed or proof.get('runtime_code_sha256') != resident.digest(plan['code'])
            or (proof.get('numerical_reference') or {}).get('passed') is not True
            or (proof.get('capacity') or {}).get('passed') is not True
            or proof.get('request') not in config['trainer_parameters'].get('memory_qualification', [])):
        raise ValueError('saved workspace recovery needs actual unchanged-parent numerical/capacity proof')
    memory_request = json.loads(resident.bound(proof['request']).read_bytes())
    cases = (proof.get('capacity') or {}).get('cases') or []
    expected_cases = memory_request.get('cases') or []
    if (not cases or len(cases) != len(expected_cases)
            or any(case.get('status') != 'passed' or
                   any(case.get(key) != expected.get(key) for key in ('id', 'forward_tokens', 'kv_slots'))
                   for case, expected in zip(cases, expected_cases))
            or limits != {key: max(case[key] for case in cases) for key in limits}):
        raise ValueError('saved workspace recovery certificate lost its actual complete maximum graph cases')
    if any(state['payload']['geometry'][key] > limits[key] for key in limits):
        raise ValueError('saved workspace recovery capacity does not cover the adopted data geometry')
    saved = CheckpointPolicy.load(resident.bound(checkpoint))
    training = require_raw_training(saved)
    counts = training.get('exposure_counts') or {}
    rows = training.get('source_row_bindings') or {}
    order = training.get('source_row_order') or []
    adoption = (training.get('options') or {}).get('resident_adoption') or {}
    if (saved.get('model_config') != proposed.get('model_config')
            or not rows or set(counts) != set(rows) or len(order) != len(rows) or set(order) != set(rows)
            or any(type(n) is not int or n < 0 for n in counts.values())
            or not training.get('held_row_bindings') or saved.get('step') != training.get('global_step')
            or training['global_step'] < state['at_global_step']
            or adoption != {'generation': state['generation'], 'history': state['history']}
            or any(training['source_receipts'][index]['sha256'] != state['payload'][role]['sha256']
                   for index, role in enumerate(('train', 'held')))):
        raise ValueError('saved workspace recovery lost actual saved adoption/exposure/source authority')
    del saved
    with resident.locked(directory):
        require_exited()
        if (resident.binding(directory / 'initial.json') != initial_ref
                or resident.binding(directory / 'current.json') != state_ref
                or (resident.binding(directory / 'pending.json') if (directory / 'pending.json').exists() else None) != pending):
            raise ValueError('resident authority changed during saved workspace verification')
        control.mkdir(parents=True, exist_ok=True)
        before = control / 'original-initial.json'
        write_bytes_durable(before, resident.bound(initial_ref).read_bytes())
        intent_path = control / 'intent.json'
        recovery = {'schema': 'natlang.saved-workspace-recovery-intent/1', 'at': time.time(),
            'transaction': transaction, 'actor_source': resident.binding(__file__), 'original_initial': resident.binding(before),
            'original_current': state_ref, 'retained_pending': pending,
            'new_geometry_identity': proposed, 'new_geometry_qualification': qualification,
            'saved_global_step': training['global_step'], 'training_credit': 0,
            'readiness_granted': False, 'original_operation_still_failed': True}
        if intent_path.exists():
            prior_intent = json.loads(intent_path.read_bytes())
            if {k: v for k, v in prior_intent.items() if k != 'at'} != {k: v for k, v in recovery.items() if k != 'at'}:
                raise ValueError('saved workspace recovery intent changed on retry')
        else:
            resident.write_json(intent_path, recovery)
        candidate = {**initial, 'geometry_identity': proposed, 'geometry_qualification': qualification,
            'geometry_recoveries': [*(initial.get('geometry_recoveries') or []), resident.binding(intent_path)]}
        candidate_path = control / 'candidate-initial.json'
        resident.write_json(candidate_path, candidate)
        resident.write_json(directory / 'initial.json', candidate)
        resident.write_json(out, {'schema': 'natlang.saved-workspace-recovery/1',
            'intent': resident.binding(intent_path), 'candidate': resident.binding(candidate_path),
            'installed_initial': resident.binding(directory / 'initial.json'),
            'changed_fields': ['geometry_identity', 'geometry_qualification', 'geometry_recoveries'],
            'training_credit': 0, 'readiness_granted': False, 'original_operation_still_failed': True})
    return resident.binding(out)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--request',type=Path,required=True)
    parser.add_argument('--request-sha256',required=True)
    parser.add_argument('--runtime-package',type=Path)
    parser.add_argument('--old-cache-root',type=Path)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    print(json.dumps(refresh(args.directory,{'path':str(args.request.resolve()),'sha256':args.request_sha256},
        runtime_package=args.runtime_package,old_cache_root=args.old_cache_root,publish=not args.prepare_only)))
    return 0


if __name__=='__main__':raise SystemExit(main())
