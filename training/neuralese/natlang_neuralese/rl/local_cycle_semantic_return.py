"""Authenticate and adopt a participant descendant returned by shared training.

The local-cycle owner is distinct from the consumer that ran the shared VAE,
joint and distillation phases. This module is the source-host boundary between
those controllers: it accepts only the exact released participant, portable
full-state bundle, semantic adapter and completion produced by that handoff.
It changes controller pointers only; all training state comes from the verified
participant checkpoint.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from ..model.native_bundle import verify_bundle
from .native_raw_handoff import binding


RETURN_SCHEMA = 'natlang.native-local-cycle-semantic-return/1'


def _bound(ref):
    path = Path(ref['path']).resolve(strict=True)
    actual = binding(path)
    if actual != {key: ref[key] for key in ('path', 'sha256')}:
        raise ValueError('semantic return artifact changed its exact binding')
    if 'bytes' in ref and (type(ref['bytes']) is not int or ref['bytes'] != path.stat().st_size):
        raise ValueError('semantic return artifact changed its recorded size')
    return path


def _checkpoint_state(path, participant):
    """Load the selected participant through the ordinary native checkpoint loader."""
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..model.native_shared_checkpoint import SHARED_SCHEMA
    state = CheckpointPolicy.load(Path(path).resolve(strict=True), shared_participant=participant)
    if state.get('schema')==SHARED_SCHEMA:
        selected=(state.get('participants')or{}).get(participant)
        global_training=state.get('training')or{}
        optimizer=(selected or {}).get('optimizer')
        source_rng=(selected or {}).get('source_rng')
        options=(selected or {}).get('training_options')or(global_training.get('options')or{})
    else:
        selected=state
        global_training=state.get('training')or{}
        optimizer=global_training.get('optimizer')
        source_rng=global_training.get('source_rng')or global_training.get('rng')
        options=global_training.get('options')or state.get('training_options')or{}
    if not isinstance(selected, dict):raise ValueError('semantic checkpoint omitted its selected participant payload')
    if selected.get('optimizer_pruned') or selected.get('inference_only'):
        raise ValueError('semantic return must contain resumable participant optimizer state')
    if not optimizer or not source_rng:
        raise ValueError('semantic return omitted participant optimizer/RNG/exposure state')
    if not isinstance(global_training.get('rng'),dict):
        raise ValueError('semantic return omitted the actual shared trainer RNG needed for resume')
    if not isinstance(options, dict):
        raise ValueError('semantic return omitted participant exposure/continuation bindings')
    normalized={**selected,'optimizer':optimizer,'source_rng':source_rng,
        'sft_progress':selected.get('sft_progress',global_training.get('sft_progress')),
        'global_step':selected.get('global_step',global_training.get('global_step',global_training.get('step')))}
    return normalized, {'participant':normalized,'training_options':options,
        'source_rng':source_rng,'sft_progress':normalized.get('sft_progress'),
        'global_training':global_training}


def validate_return(receipt_path, *, participant, parent_checkpoint, role_plan, role_release):
    """Verify an immutable return receipt against live source-owner authority."""
    from . import native_role_release as roles

    receipt_path = Path(receipt_path).resolve(strict=True)
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('schema') != RETURN_SCHEMA or receipt.get('participant') != participant:
        raise ValueError('semantic return changed schema or participant selection')
    if receipt.get('role_plan') != binding(role_plan) or receipt.get('role_release') != role_release:
        raise ValueError('semantic return changed its source-host release authority')
    source_role, released = roles.validate(role_release, json.loads(Path(role_plan).read_text()), kind='released')
    if participant not in released:
        raise ValueError('semantic return release omitted the selected participant')
    actual_parent = binding(parent_checkpoint)
    if (released[participant]['checkpoint']['sha256'] != actual_parent['sha256']
            or receipt.get('parent_checkpoint')!=released[participant]['checkpoint']):
        raise ValueError('semantic return is not descended from the currently released parent')

    completion_path = _bound(receipt['completion'])
    completion = json.loads(completion_path.read_text())
    if (completion.get('schema') != 'natlang.native-shared-stage-completion/1'
            or completion.get('completed') is not True
            or completion.get('production_ready') is not True
            or completion.get('preparation_complete') is True):
        raise ValueError('semantic return lacks completed production shared-stage work')
    expected_models={row['handoff_requirement']['model_id']
        for row in json.loads(Path(role_plan).read_text())['models']}
    from ..train.native_readiness import validate_completion
    key=(completion.get('participants_signature'),completion.get('curriculum_signature'))
    validate_completion(completion_path,{key:{'models':sorted(expected_models),
        'execution_signature':canonical_json_sha256_hex({'completion':receipt['completion']['sha256']})}})
    if (completion.get('checkpoint', {}).get('sha256') != receipt.get('checkpoint', {}).get('sha256')
            or completion.get('adapter', {}).get('sha256') != receipt.get('semantic_adapter', {}).get('sha256')):
        raise ValueError('semantic return is not bound to the completed shared checkpoint and adapter')
    if receipt.get('completion_sha256') != receipt['completion']['sha256']:
        raise ValueError('semantic return completion digest was substituted')

    bundle_manifest = _bound(receipt['bundle'])
    bundle = verify_bundle(bundle_manifest.parent)
    if bundle.get('participant') != participant or not bundle.get('semantic_adapter'):
        raise ValueError('semantic return bundle lost its participant or semantic control')
    if (bundle['checkpoint']['sha256']!=completion['checkpoint']['sha256']
            or bundle['semantic_adapter']['sha256']!=completion['adapter']['sha256']
            or receipt.get('semantic_adapter')!=completion['adapter']):
        raise ValueError('participant bundle changed the completed shared checkpoint/adapter bytes')
    checkpoint_path = Path(bundle['checkpoint']['resolved'])
    checkpoint_ref = binding(checkpoint_path)
    receipt_checkpoint=receipt.get('checkpoint')
    if (not isinstance(receipt_checkpoint,dict)
            or set(receipt_checkpoint)-{'path','sha256','bytes'}
            or any(receipt_checkpoint.get(key)!=checkpoint_ref[key] for key in ('path','sha256'))
            or ('bytes'in receipt_checkpoint and
                (type(receipt_checkpoint['bytes'])is not int
                 or receipt_checkpoint['bytes']!=checkpoint_path.stat().st_size))):
        raise ValueError('semantic return checkpoint differs from its verified portable bundle')

    before, before_training = _checkpoint_state(parent_checkpoint, participant)
    after, after_training = _checkpoint_state(checkpoint_path, participant)
    after_identity = after.get('source_identity') or {}
    released_identity=released[participant]['native_identity']
    before_config=before.get('model_config')or{};after_config=after.get('model_config')or{}
    if (after_identity!=released_identity
            or after_identity.get('checkpoint_sha256')!=actual_parent['sha256']
            or before_config.get('architecture')!=after_config.get('architecture')
            or before_config.get('base_inventory_sha256')!=after_config.get('base_inventory_sha256')
            or after_config.get('architecture')!=released_identity.get('architecture')):
        raise ValueError('semantic return changed the actual model/base lineage')
    if before.get('sft_progress')!=after.get('sft_progress'):
        raise ValueError('semantic return changed inherited SFT/DPO exposure without a native supervised stage')
    if (type(before.get('global_step'))is not int or type(after.get('global_step'))is not int
            or after['global_step']<before['global_step']):
        raise ValueError('semantic return regressed the participant optimizer update cursor')
    options = after_training.get('training_options') or {}
    ancestors = options.get('native_snapshot_ancestors') or []
    before_options=(before.get('training_options')or(before.get('training')or{}).get('options')or{})
    if ancestors!=before_options.get('native_snapshot_ancestors',[]):
        raise ValueError('semantic return changed the participant native snapshot ancestry')
    from ..train.checkpoint_offload import _same_values
    _same_values(before_training['global_training']['rng'],after_training['source_rng'])
    _require_optimizer_continuation(before.get('optimizer')or(before.get('training')or{}).get('optimizer'),
        after.get('optimizer'))

    adapter_path = Path(bundle['semantic_adapter']['resolved'])
    adapter = json.loads(adapter_path.read_text())
    if (adapter.get('schema') != 'natlang.native-semantic-adapter/1'
            or adapter.get('checkpoint', {}).get('sha256') != checkpoint_ref['sha256']
            or participant not in adapter.get('models', {})):
        raise ValueError('semantic return adapter is not bound to its exact participant checkpoint')
    model = adapter['models'][participant]
    if (model.get('checkpoint', {}).get('sha256') != checkpoint_ref['sha256']
            or model.get('native_identity', {}).get('checkpoint_sha256') != checkpoint_ref['sha256']):
        raise ValueError('semantic return adapter substituted participant inference identity')
    if model.get('mandatory_sft') != _obligations({'sft_progress': after.get('sft_progress')}):
        raise ValueError('semantic return adapter lost or fabricated remaining SFT/DPO debt')

    return {'receipt': receipt, 'bundle': bundle, 'checkpoint': checkpoint_ref,
            'semantic_adapter': {'path': str(adapter_path), 'sha256': sha256_file_hex_cached(adapter_path)},
            'participant': participant, 'role': source_role, 'parent': actual_parent,
            'completion': receipt['completion'],'semantic_refresh_config':completion['curriculum']}


def _obligations(training):
    from ..train.native_continuation import progress_obligations
    progress = training.get('sft_progress')
    if not isinstance(progress, dict):
        raise ValueError('semantic return omitted full native SFT progress')
    return progress_obligations(progress)


def _require_optimizer_continuation(before,after):
    if not isinstance(before,dict)or not isinstance(after,dict):
        raise ValueError('semantic return omitted the inherited participant optimizer')
    old_groups=before.get('param_groups')or[];new_groups=after.get('param_groups')or[]
    if len(old_groups)!=len(new_groups):raise ValueError('semantic return changed optimizer parameter-group structure')
    for old,new in zip(old_groups,new_groups):
        if (old.get('param_names')!=new.get('param_names')
                or len(old.get('params',[]))!=len(new.get('params',[]))):
            raise ValueError('semantic return changed optimizer parameter routing')
    old_state=before.get('state')or{};new_state=after.get('state')or{}
    if not set(old_state)<=set(new_state):raise ValueError('semantic return reset inherited optimizer moments')
    for key,old in old_state.items():
        new=new_state[key]
        if isinstance(old,dict)and 'step'in old:
            left,right=old['step'],new.get('step')
            left=float(left.item())if hasattr(left,'item')else float(left)
            right=float(right.item())if hasattr(right,'item')else float(right)
            if right<left:raise ValueError('semantic return regressed optimizer age')


def adopt_at_boundary(config, plan, cursor, *, state_path, cursor_path):
    """Adopt an already transported return only at a quiescent local-cycle boundary.

    A durable intent makes the two controller pointer writes restartable. The
    verified participant bundle remains immutable and is the actual saved state.
    """
    descriptor = plan.get('semantic_return')
    if not descriptor:
        return None
    participant = config.get('native_participant')
    if not participant:
        raise ValueError('semantic return needs the production actor participant identity')
    pointer_path=Path(descriptor['return_directory'])/(participant+'-return-current.json')
    if not pointer_path.is_file():
        return None
    pointer=json.loads(pointer_path.read_text())
    if (pointer.get('schema')!='natlang.native-local-cycle-return-pointer/1'
            or pointer.get('participant')!=participant or pointer.get('training_credit')!=0
            or type(pointer.get('ordinal'))is not int or pointer['ordinal']<1):
        raise ValueError('local-cycle semantic return pointer changed its owner or generation')
    receipt_ref=pointer.get('receipt')
    receipt_path=_bound(receipt_ref)
    state_path, cursor_path = Path(state_path), Path(cursor_path)
    if not state_path.is_file() or not cursor_path.is_file():
        return None
    state = json.loads(state_path.read_text())
    if (any(key.startswith(('active_update_','partial_update_'))for key in state)
            or cursor.get('phase')not in('rl','supervised')):
        raise ValueError('semantic return may be adopted only at a complete round/stage boundary')
    receipt_binding = binding(receipt_path)
    intent_path = Path(descriptor['return_directory'])/'return-intents'/(receipt_binding['sha256']+'.json')
    if (state.get('semantic_return') or {}).get('receipt') == receipt_binding:
        if not intent_path.is_file():
            raise ValueError('semantic return state committed without its durable transaction intent')
        intent=json.loads(intent_path.read_text())
        current_cursor=json.loads(cursor_path.read_text())
        if (intent.get('schema')!='natlang.native-local-cycle-return-intent/1'
                or intent.get('state')!=state or intent.get('receipt')!=receipt_binding
                or intent.get('training_credit')!=0
                or intent.get('state_sha256')!=canonical_json_sha256_hex(intent.get('state'))
                or intent.get('cursor_sha256')!=canonical_json_sha256_hex(intent.get('cursor'))):
            raise ValueError('semantic return recovery intent differs from committed controller state')
        if current_cursor==intent['cursor']:
            return None
        if binding(cursor_path)!=intent['before_cursor'] or current_cursor!=intent.get('before_cursor_value'):
            raise ValueError('semantic return recovery found an unknown cursor successor')
        write_json_durable(cursor_path,intent['cursor'])
        return {'checkpoint': intent['checkpoint'], 'semantic_adapter': intent['semantic_adapter'],
                'receipt': receipt_binding, 'training_credit': 0}
    checkpoint = state.get('checkpoint')
    if not checkpoint:
        raise ValueError('local-cycle return needs its actual current full-state parent')
    role_plan = Path(descriptor['role_plan']).resolve(strict=True)
    role_plan_value = json.loads(role_plan.read_text())
    receipt_value=json.loads(receipt_path.read_text())
    role_module=__import__('natlang_neuralese.rl.native_role_release',fromlist=['ownership'])
    _, exchange = role_module.ownership(role_plan_value)
    if exchange is None or exchange['role'] != descriptor.get('role'):
        raise ValueError('semantic return does not name this local source-host role')
    receipt_role_plan=Path(receipt_value['role_plan']['path']).resolve(strict=True)
    if binding(receipt_role_plan)!=receipt_value['role_plan']:
        # The receipt may omit the optional byte count; path/SHA still must be
        # the exact immutable source plan named by the receipt.
        if any(binding(receipt_role_plan).get(key)!=receipt_value['role_plan'].get(key)
                for key in ('path','sha256')):
            raise ValueError('semantic return role plan bytes changed')
    if receipt_role_plan!=role_plan:
        prior=role_module._local_cycle_role_plan_predecessor(plan,
            role_module._local_cycle_source(role_plan_value),receipt_role_plan)
        if (prior is None or Path(prior['path']).resolve()!=role_plan
                or prior.get('sha256')!=sha256_file_hex_cached(role_plan)):
            raise ValueError('semantic return role plan is not the exact frozen-preparation successor')
    else:
        receipt_role_plan=role_plan
    role_release = json.loads((Path(exchange['directory']) / (exchange['role'] + '-status.json')).read_text())
    previous=(state.get('semantic_return')or{}).get('receipt')
    if (pointer.get('previous')!=previous
            or pointer['ordinal']!=(1 if previous is None else int((state.get('semantic_return')or{}).get('ordinal',0))+1)):
        raise ValueError('semantic return pointer skipped or repeated an earned participant descendant')
    validated = validate_return(receipt_path, participant=participant, parent_checkpoint=checkpoint,
                                role_plan=receipt_role_plan, role_release=role_release)
    semantic_adapter = validated['semantic_adapter']
    next_state = {**state, 'checkpoint': validated['checkpoint']['path'],
                  'semantic_adapter': semantic_adapter['path'],
                  'native_participant':participant,
                  'semantic_refresh_config':validated['semantic_refresh_config']['path'],
                  'semantic_return': {'receipt': binding(receipt_path), 'parent': validated['parent'],
                      'checkpoint': validated['checkpoint'], 'semantic_adapter': semantic_adapter,
                      'completion': validated['completion'],'ordinal':pointer['ordinal']}}
    next_cursor = dict(cursor)
    next_cursor['checkpoint'] = validated['checkpoint']
    next_cursor['semantic_adapter'] = semantic_adapter
    next_cursor['semantic_return_receipt'] = receipt_binding
    # Cycle, phase, round target, stage index and accumulated failures/debts are
    # intentionally carried verbatim. Returned checkpoint state owns exposure.
    intent = {'schema': 'natlang.native-local-cycle-return-intent/1',
        'participant': participant, 'before_state': binding(state_path),
        'before_cursor': binding(cursor_path), 'receipt': receipt_binding,
        'checkpoint': validated['checkpoint'], 'semantic_adapter': semantic_adapter,
        'state': next_state, 'cursor': next_cursor,
        'state_sha256': canonical_json_sha256_hex(next_state),
        'cursor_sha256': canonical_json_sha256_hex(next_cursor), 'training_credit': 0}
    intent['before_cursor_value']=cursor
    intent_path.parent.mkdir(parents=True,exist_ok=True)
    if intent_path.exists():
        if json.loads(intent_path.read_text()) != intent:
            raise ValueError('semantic return retry changed its durable adoption intent')
    else:
        write_json_durable(intent_path, intent)
    current_state = json.loads(state_path.read_text())
    current_cursor = json.loads(cursor_path.read_text())
    if current_state not in (state, next_state) or current_cursor not in (cursor, next_cursor):
        raise ValueError('semantic return found an unknown current controller successor')
    if current_state != next_state:
        write_json_durable(state_path, next_state)
    if current_cursor != next_cursor:
        write_json_durable(cursor_path, next_cursor)
    return {'checkpoint': validated['checkpoint'], 'semantic_adapter': semantic_adapter,
            'receipt': binding(receipt_path), 'training_credit': 0}
