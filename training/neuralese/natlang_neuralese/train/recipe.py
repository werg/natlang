"""Declared, gated training stages with frozen code and explicit artifact handoff.

Stage implementations are shared Python modules. Extend HANDLERS with a typed
handler and declare its dependencies in a JSON recipe. No shell command strings.
An unsuccessful stage cannot advance its dependents or issue a certificate.
"""
import argparse
import hashlib
import json
import math
import re
import shutil
import signal
import subprocess
import sys
import time
import os
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex
from ..common.hashing import sha256_file_hex_cached as sha  # cached by (realpath, size, mtime_ns)


HANDLERS = {
    # Neuralese direct v1 (plans/neuralese/NEURALESE_DIRECT_V1.md): plain target SFT from a base model with the
    # NeuraleseProjection; the TokenChoiceProjection is declared per stage (off in v1).
    # Its data is either prepared text (text_data, the shared declared input of neuralese-direct-v2) or records and
    # pieces rendered by gold_text_rows: missing_stage_inputs() states that alternative.
    'neuralese_direct': {'module':'natlang_neuralese.train.direct_sft', 'required_inputs':set(),
                         'optional_inputs':{'records','pieces','text_data','heads','continue_from',
                                            'continue_identity','continue_monitor_reference',
                                            'continued_text_data'},
                         'parameters':{'steps','epochs','batch_tokens','tokens','target_tokens','prefix_tokens','batch','eval_batch','eval_every',
                           'checkpoint_every','held_documents','context_weight','feedback_weight','backbone_lr',
                           'latent_optimizer','latent_lr','head_lr','neuralese_weight','projection_blocks',
                           'projection_hidden','projection_kernel','token_choice_projection','moe_kernel','checkpoint_layers','drift_rows','drift_every','drift_gate_every','drift_gate_rise','drift_ce_windows','gate_response_kl',
                           'drift_warn','drift_stop','drift_agreement_warn','drift_agreement_stop','drift_gate','gate_held_rows','monitor',
                           'value_margin','seed','window_execution','latent_skip_idle','latent_cautious','latent_beta2_half_life_tokens',
                           'latent_ns','latent_weight_decay','grad_spike_clip','stop_at','weight_drift_base','save_state',
                           'fp8_training','fp8_recipe','fp8_amax_history_len',
                           'gradient_audit_steps','saved_tensor_audit_steps','gradient_digest_steps',
                           'state_hash_steps','sdpa_backend'},'result':'report.json'},
    'core_text_warmup': {'module':'natlang_neuralese.train.text_warmup', 'required_inputs':{'records','pieces'}, 'optional_inputs':{'text_data','continue_from','heads','student_checkpoint'},
                        'parameters':{'text_data','student_checkpoint','batch','steps','tokens','prefix_tokens','target_tokens','cutoff',
                          'cohort_weights','qualification_cohort','context_weight','feedback_weight',
                          'backbone_training','rank','optimizer','lr','sketch_lr','embedding_weight','sketch_weight','text_weight',
                          'projection_patience','projection_min_evals','projection_min_improvement',
                          'backbone_ramp_evals','pass_ramp_evals','checkpoint_every','eval_every','latent_optimizer','latent_step_in_backward','moe_kernel',
                          'held_documents','seed','checkpoint_layers',
                          'max_ce_delta','max_relative_mse','min_agreement','consecutive_gates',
                          'input_map_kernel','input_map_rank','ar_feedback_fixup','channel_consistency_weight','qat_latent_lr',
                          'member_weight','member_tokens','member_eval_windows','member_mask_system','read_adapter',
                          'optimizer_state','optimizer_added','quantization_steps','report_column_fraction',
                          'drift_rows','drift_warn','drift_stop','drift_agreement_warn','drift_agreement_stop'},'result':'heads.pt'},
    'text_warmup_runtime': {'module':'natlang_neuralese.eval.text_warmup_runtime', 'required_inputs':{'records'}, 'optional_inputs':{'heads'},
                            'parameters':set(),'result':'report.json'},
    'raw_recurrence_training': {'module': 'natlang_neuralese.train.trajectories', 'required_inputs':{'records','pieces'}, 'optional_inputs':{'heads','direct_artifact_manifest','continue_from','direct_parent_proof'},
                                'parameters': {'steps', 'batch', 'lr', 'rank', 'lora_lr', 'backbone_lr', 'backbone_training', 'max_tokens',
                                               'train', 'eval', 'handover', 'write_curriculum', 'max_writes',
                                               'write_depth', 'tokens_per_vector', 'heads_lr', 'distill',
                                               'crisp_weight', 'memory_gb', 'backward_policy', 'graph_memory_gb',
                                               'graph_headroom_gb', 'checkpoint_layers', 'checkpoint_attention_only',
                                               'staged_checkpoint_attention_only', 'checkpoint_elide_rng', 'producer_batch_size',
                                               'producer_batch_memory_gb', 'ffn_chunk_tokens',
                                               'optimizer', 'checkpoint_every', 'eval_every', 'seed', 'writer_text_weight',
                                               'max_write_vectors', 'content_transport', 'writer_length_policy', 'writer_supervision', 'stop_supervision',
                                               'sketch_gradient', 'sketch_target_weight', 'sketch_target_backbone_scale',
                                               'projection_anchor_weight', 'projection_anchor_backbone_scale',
                                               'projection_anchor_decay_steps', 'read_adapter',
                                               'local_stage_batch_size', 'train_control_rows', 'token_cache_mib',
                                               'qat_latent_lr', 'latent_optimizer', 'latent_step_in_backward', 'moe_kernel', 'member_weight', 'member_tokens', 'member_eval',
                                               'member_mask_system', 'member_full_weight',
                                               'view', 'view_window', 'context_weight', 'feedback_weight',
                                               'context_coverage', 'cohort_weights', 'qualification_cohort',
                                               'stop_pg', 'purpose_contrast', 'purpose_margin', 'view_tokens_per_vector',
                                               'optimizer_state', 'optimizer_added', 'direct_base_snapshot', 'curriculum_change',
                                               'latent_lr', 'latent_ns', 'latent_weight_decay', 'grad_spike_clip',
                                               'monitor', 'drift_rows', 'drift_every', 'drift_gate_every', 'drift_gate_rise',
                                               'gate_response_kl', 'drift_warn', 'drift_stop', 'drift_agreement_warn',
                                               'drift_agreement_stop', 'drift_gate', 'gate_held_rows'},
                                'result': 'checkpoint.pt'},
    # The view operator gate (eval/view_gate.py; TRAINING_RECIPE.md "The view operator stage") on the exact weights of
    # the view stage it requires (that stage's recurrence checkpoint).
    'view_operator_gate': {'module': 'natlang_neuralese.eval.view_gate', 'required_inputs': {'records'},
                           'optional_inputs': {'pieces', 'harness_records', 'harness_pieces'},
                           'parameters': {'split', 'per_artifact', 'harness_limit', 'seed', 'view_window',
                                          'min_reconstruction_recovery', 'min_qa_recovery', 'min_shuffle_margin',
                                          'min_purpose_margin', 'max_next_action_delta'},
                           'result': 'view-gate.json'},
    'raw_runtime_qualification': {'module': 'natlang_neuralese.eval.raw_port_handoff', 'required_inputs':{'records'}, 'optional_inputs':set(),
                                  'parameters': {'limit', 'max_length'}, 'result': 'heads.pt'},
    'verified_heads_handoff': {'module':'natlang_neuralese.train.verified_heads_handoff',
                               'required_inputs':{'warmup_checkpoint','warmup_manifest','warmup_heads','warmup_runtime_report'},
                               'optional_inputs':set(),
                               'parameters':{'max_ce_delta','max_relative_mse','min_agreement'},'result':'report.json'},
    'token_identity': {'module': 'natlang_neuralese.eval.foundation', 'required_inputs':{'records'}, 'optional_inputs':set(),
                       'parameters': {'limit', 'max_tokens'}, 'result': 'identity.json'},
    'causal_embedding_distillation': {'module': 'natlang_neuralese.train.causal_bootstrap', 'required_inputs':{'records'}, 'optional_inputs':{'pieces'},
                                     'parameters': {'cutoff', 'steps', 'batch', 'lr', 'tokens', 'contexts',
                                                    'context_tokens', 'eval_every', 'checkpoint_every',
                                                    'seed', 'agreement_gate', 'kl_gate', 'source_fraction',
                                                    'argmax_weight', 'continue_from', 'stop_on_gate'}, 'result': 'best-checkpoint.pt'},
}
DIRECT_STAGE_SCHEMA = 'natlang.neuralese-declared-direct-stage/1'

INPUT_BINDING_NAME = re.compile(r'[a-z][a-z0-9_.-]*')


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_cohort_weights(weights):
    """Explicit training fractions per named cohort (the shared text_supervision semantics). Naming a cohort here
    admits nothing; admission is the recipe's and the corpus registry's."""
    if (not isinstance(weights, dict) or not weights or
            any(not isinstance(name, str) or not name for name in weights) or
            any(not _finite_number(value) or value <= 0 for value in weights.values()) or
            not math.isclose(sum(weights.values()), 1., abs_tol=1e-8)):
        raise ValueError('cohort_weights must map nonempty cohort names to positive fractions that sum to one')


def validate_recurrence_parameters(parameters):
    """Typed values of the trajectory trainer options that recipes declare (train.trajectories flags)."""
    if 'view' in parameters and parameters['view'] not in ('preview', 'written'):
        raise ValueError('view must be preview or written')
    if 'view_window' in parameters and (type(parameters['view_window']) is not int or parameters['view_window'] < 1):
        raise ValueError('view_window must be a positive integer')
    for name in ('context_weight', 'distill'):
        if name in parameters and (not _finite_number(parameters[name]) or parameters[name] < 0):
            raise ValueError(name + ' must be finite and nonnegative')
    if 'feedback_weight' in parameters and (not _finite_number(parameters['feedback_weight']) or
                                            not 0 <= parameters['feedback_weight'] <= 1):
        raise ValueError('feedback_weight must be finite and between zero and one')
    if 'context_coverage' in parameters and parameters['context_coverage'] not in ('last-reply', 'records'):
        raise ValueError('context_coverage must be last-reply or records')
    if 'cohort_weights' in parameters:
        validate_cohort_weights(parameters['cohort_weights'])
    for name in ('stop_pg', 'purpose_contrast', 'purpose_margin', 'view_tokens_per_vector'):
        if name in parameters and (not _finite_number(parameters[name]) or parameters[name] < 0):
            raise ValueError(name + ' must be finite and nonnegative')
    if parameters.get('purpose_contrast') and parameters.get('view') != 'written':
        raise ValueError('purpose_contrast requires view written')
    if 'qualification_cohort' in parameters:
        cohort = parameters['qualification_cohort']
        if not isinstance(cohort, str) or not cohort:
            raise ValueError('qualification_cohort must be a nonempty cohort name')
        if 'cohort_weights' in parameters and cohort not in parameters['cohort_weights']:
            raise ValueError('qualification_cohort must be one of the cohort_weights cohorts')


def refuse_retired_parameters(parameters):
    retired = sorted({'checkpoint_minutes', 'eval_minutes'} & set(parameters if isinstance(parameters, dict) else ()))
    if retired:
        raise ValueError(f'{", ".join(retired)} retired (owner 2026-10-10): declare eval_every and checkpoint_every in '
                         'steps instead (steps and checkpoint_every multiples of eval_every)')


def validate_stage_parameters(kind, parameters):
    """Validate parameter shapes whose CLI actions are not scalar values."""
    if kind not in ('core_text_warmup', 'raw_recurrence_training'):
        return
    if kind == 'raw_recurrence_training':
        validate_recurrence_parameters(parameters)
    if 'moe_kernel' in parameters and parameters['moe_kernel'] not in ('loop', 'grouped'):
        raise ValueError('moe_kernel must be loop or grouped')
    if ('optimizer_state' in parameters and
            parameters['optimizer_state'] not in ('restore', 'fresh')):
        raise ValueError('optimizer_state must be restore or fresh')
    if 'optimizer_added' in parameters:
        prefixes = parameters['optimizer_added']
        if not isinstance(prefixes, list) or any(
                not isinstance(prefix, str) or not prefix for prefix in prefixes):
            raise ValueError('optimizer_added must be a list of nonempty parameter-name prefixes')


def validate_input_bindings(recipe):
    """Validate the optional, content-pinned dataset catalog and stage role map."""
    catalog = recipe.get('input_bindings', {})
    if not isinstance(catalog, dict):
        raise ValueError('input_bindings must be an object')
    for name, binding in catalog.items():
        if not isinstance(name, str) or not INPUT_BINDING_NAME.fullmatch(name) or not isinstance(binding, dict):
            raise ValueError('invalid named input binding')
        if 'artifact' in binding:
            # A registered Neuralese artifact (training/neuralese_artifacts.json): its immutable manifest pins the
            # bytes, so the recipe names the id (and the file when the artifact has several).
            if not set(binding) <= {'artifact', 'file'} or not isinstance(binding['artifact'], str) \
                    or not isinstance(binding.get('file', ''), str):
                raise ValueError('an artifact input binding has only artifact and optional file')
            continue
        if set(binding) != {'path', 'sha256'} or not isinstance(binding['path'], str) or not binding['path']:
            raise ValueError('each input binding requires only path and sha256 (or artifact)')
        if not isinstance(binding['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', binding['sha256']):
            raise ValueError('input binding sha256 must be lowercase hex')
    referenced = set()
    for stage in recipe['stages']:
        if 'inputs' not in stage:
            continue
        selected = stage['inputs']
        if not isinstance(selected, dict):
            raise ValueError('stage inputs must map input roles to named bindings')
        kind = stage['kind']
        allowed = HANDLERS[kind]['required_inputs'] | HANDLERS[kind]['optional_inputs']
        if not set(selected) <= allowed or missing_stage_inputs(kind, selected):
            raise ValueError('stage input roles do not satisfy handler contract: ' + stage['id'])
        if any(not isinstance(name, str) or name not in catalog for name in selected.values()):
            raise ValueError('stage refers to an unknown named input binding')
        referenced.update(selected.values())
    if set(catalog) - referenced:
        raise ValueError('unused named input binding(s): ' + ', '.join(sorted(set(catalog) - referenced)))


def missing_stage_inputs(kind, roles):
    """Required input roles a stage of ``kind`` lacks, given the roles it has."""
    missing = HANDLERS[kind]['required_inputs'] - set(roles)
    if kind == 'neuralese_direct' and 'text_data' not in roles and 'records' not in roles:
        missing = missing | {'text_data or records'}
    if kind == 'neuralese_direct' and 'continue_from' in roles:
        missing |= {'continue_identity', 'continue_monitor_reference'} - set(roles)
    if kind == 'neuralese_direct' and 'continued_text_data' in roles and 'text_data' not in roles:
        missing |= {'text_data (a data continuation moves text_data)'}
    return missing


# A later neuralese_direct phase (its ``requires`` names a neuralese_direct stage of the same lineage) continues from
# that predecessor phase: its final full state plus the two sidecars direct_sft requires with --continue-from (the
# lineage's step-0 identity qualification, which the phase inherits, and the base model's drift reference). These
# roles are bound implicitly from the predecessor's output directory, never declared as recipe inputs; one map serves
# the plan check (load_recipe) and the child argv (main).
DIRECT_PHASE_CONTINUATION = {'continue_from': 'checkpoint.pt', 'continue_identity': 'identity-check.json',
                             'continue_monitor_reference': 'monitor-reference.pt'}


def direct_phase_predecessor(recipe, stage):
    """The id of the neuralese_direct phase ``stage`` continues from, or None."""
    if stage.get('kind') != 'neuralese_direct':
        return None
    kinds = {other.get('id'): other.get('kind') for other in recipe['stages']}
    phases = [name for name in stage.get('requires', []) if kinds.get(name) == 'neuralese_direct']
    if len(phases) > 1:
        raise ValueError('a neuralese_direct stage continues from at most one direct predecessor: ' + stage['id'])
    return phases[0] if phases else None


def direct_phase_continuation(directory):
    """The continuation inputs of a later direct phase, resolved in its predecessor phase's output directory."""
    return {role: {'binding': None, 'path': str(Path(directory) / name)}
            for role, name in DIRECT_PHASE_CONTINUATION.items()}


def check_direct_phase_continuation(recipe, stage):
    """Plan check: a continued direct phase has every input its handler needs once the implicit roles are bound."""
    if direct_phase_predecessor(recipe, stage) is None:
        return
    declared = set(stage.get('inputs', {}))
    if declared & {'continue_from', 'continue_identity', 'continue_monitor_reference'}:
        raise ValueError('a continued direct phase binds its continuation from its predecessor phase, not from '
                         'declared inputs: ' + stage['id'])
    missing = missing_stage_inputs(stage['kind'], declared | set(DIRECT_PHASE_CONTINUATION))
    if missing:
        raise ValueError('continued direct phase ' + stage['id'] + ' lacks input(s): ' + ', '.join(sorted(missing)))


def parse_input_binding_overrides(values):
    overrides = {}
    for value in values or []:
        name, sep, path = value.partition('=')
        if not sep or not INPUT_BINDING_NAME.fullmatch(name) or not path:
            raise ValueError('--input-binding must be NAME=PATH')
        if name in overrides:
            raise ValueError('duplicate --input-binding: ' + name)
        overrides[name] = path
    return overrides


def binding_path(value):
    """A binding's file: recipe paths are relative to the repository root (``common.paths`` root ``repo``), never to
    the runner's working directory; absolute paths and overrides stay as given."""
    from ..common.paths import root
    path = Path(value)
    return (path if path.is_absolute() else root('repo') / path).resolve()


def resolve_stage_inputs(recipe, stage, defaults, overrides=None):
    """Return hashed, exact inputs for one stage; overrides change location, never identity."""
    selected = stage.get('inputs')
    if selected is None:
        result = {}
        supported = HANDLERS[stage['kind']]['required_inputs'] | HANDLERS[stage['kind']]['optional_inputs']
        for role, path in defaults.items():
            if path is not None and role in supported:
                resolved = Path(path).resolve()
                result[role] = {'binding': None, 'path': str(resolved), 'sha256': sha(resolved)}
        missing = missing_stage_inputs(stage['kind'], result)
        if missing:
            raise ValueError('missing required stage input(s): ' + ', '.join(sorted(missing)))
        return result
    catalog = recipe.get('input_bindings', {})
    overrides = overrides or {}
    unknown = set(overrides) - set(catalog)
    if unknown:
        raise ValueError('override for undeclared input binding: ' + ', '.join(sorted(unknown)))
    result = {}
    for role, name in selected.items():
        spec = catalog[name]
        if 'artifact' in spec:
            from ..artifacts import resolve as resolve_artifact

            stored, expected = resolve_artifact(spec['artifact'], spec.get('file'))
            path = binding_path(overrides.get(name, stored))
            actual_sha = sha(path)
            if actual_sha != expected:
                raise ValueError('input binding content hash mismatch: ' + name)
            result[role] = {'binding': name, 'artifact': spec['artifact'], 'path': str(path), 'sha256': actual_sha}
            continue
        path = binding_path(overrides.get(name, spec['path']))
        actual_sha = sha(path)
        if actual_sha != spec['sha256']:
            raise ValueError('input binding content hash mismatch: ' + name)
        result[role] = {'binding': name, 'path': str(path), 'sha256': actual_sha}
    return result


def _staged_inputs(resolved):
    """Data inputs resolving to the external HDD are read from a verified NVMe copy; load-once state is read in place
    (common/hdd_guard.py ``input_path``)."""
    from ..common import hdd_guard
    return {role: ({**value, 'path': hdd_guard.input_path(role, value['path'], value.get('sha256'))}
                   if isinstance(value, dict) and 'path' in value and role != 'direct_artifact_manifest' else value) for role, value in resolved.items()}


def stage_input_args(resolved, kind):
    """Serialize already-validated file roles to the shared handler CLIs."""
    args = []
    supported = HANDLERS[kind]['required_inputs'] | HANDLERS[kind]['optional_inputs']
    resolved = _staged_inputs(resolved)
    for role in ('records', 'pieces', 'text_data', 'continued_text_data', 'continue_from', 'continue_identity',
                 'continue_monitor_reference', 'student_checkpoint', 'warmup_checkpoint',
                 'warmup_manifest', 'warmup_heads', 'warmup_runtime_report', 'harness_records', 'harness_pieces',
                 'direct_parent_proof'):
        if role in resolved and role in supported:
            args += ['--' + role.replace('_', '-'), resolved[role]['path']]
    if 'direct_artifact_manifest' in resolved and 'direct_artifact_manifest' in supported:
        manifest = Path(resolved['direct_artifact_manifest']['path'])
        if manifest.name != 'manifest.json':
            raise ValueError('direct_artifact_manifest must bind the artifact manifest.json file')
        args += ['--direct-artifact', str(manifest.parent)]
    return args


def mapped_fixup_continuation(stage, reports):
    """Resolve the immediately declared mapped warm-up's full-state and heads outputs."""
    predecessors = [report for report in reports
                    if report['id'] in stage['requires'] and
                    report['kind'] == 'core_text_warmup']
    if len(predecessors) != 1:
        raise ValueError('AR feedback fixup requires exactly one completed mapped warm-up')
    heads = Path(predecessors[0]['artifact'])
    checkpoint = heads.parent / 'checkpoint.pt'
    return checkpoint, heads


def effective_stage_parameters(stage, reports, recipe=None):
    """Resolve fixup ``steps`` as new updates above its exact predecessor step."""
    parameter_defaults = (recipe or {}).get('stage_parameter_defaults', {})
    parameters = dict(parameter_defaults.get(stage['kind'], {}))
    parameters.update(stage['parameters'])
    if stage['kind'] == 'core_text_warmup' and parameters.get('ar_feedback_fixup',False):
        predecessors = [report for report in reports
                        if report['id'] in stage['requires'] and
                        report['kind'] == 'core_text_warmup']
        if len(predecessors) != 1:
            raise ValueError('AR feedback fixup requires exactly one completed mapped warm-up')
        start_step = predecessors[0].get('gate',{}).get('step')
        updates = parameters.get('steps')
        if type(start_step) is not int or type(updates) is not int or updates < 1:
            raise ValueError('AR fixup requires a predecessor step and positive update count')
        parameters['steps'] = start_step + updates
    return parameters


DELETE_MARKER = '$delete'
INHERITANCE_KEYS = ('extends', 'overrides')


def _is_delete(value):
    return isinstance(value, dict) and value == {DELETE_MARKER: True}


def deep_merge(base, override):
    """Merge ``override`` onto ``base`` without mutating either.

    Objects merge key by key, recursively. Every other value (lists, strings, numbers, booleans, null) replaces the
    base value whole: lists never concatenate. The object ``{"$delete": true}`` removes the key from the result.
    """
    if not isinstance(base, dict) or not isinstance(override, dict):
        return json.loads(json.dumps(override))
    merged = json.loads(json.dumps(base))
    for key, value in override.items():
        if _is_delete(value):
            if key not in merged:
                raise ValueError('override deletes an absent key: ' + key)
            del merged[key]
        elif key in merged:
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = json.loads(json.dumps(value))
    return merged


def resolve_recipe_data(path, _chain=()):
    """Return the canonical, fully inherited recipe object for ``path`` (``extends`` and ``overrides`` consumed).

    ``extends`` names a recipe id; the parent is the sibling file in the same directory whose ``id`` equals it.
    ``overrides`` is an object. Its ``stages`` member maps stage ids to objects deep-merged into the parent's stage of
    that id (``{"$delete": true}`` drops the stage; an unknown id is an error); ``stages_added`` lists new stages, each
    ``{"after": stage id, "stage": {...}}`` inserted after that stage (or ``{"before": ...}``, before it); every other member deep-merges into the top
    level. The child's own ``id`` and ``description`` always win. Resolution happens before validation, so the
    handler parameter whitelist applies to the merged result.
    """
    path = Path(path).resolve()
    recipe = json.loads(path.read_text())
    if 'extends' not in recipe:
        if 'overrides' in recipe:
            raise ValueError('overrides without extends')
        return recipe
    if recipe.get('schema') == DIRECT_STAGE_SCHEMA:
        raise ValueError('direct-stage recipes do not support extends')
    parent_id, overrides = recipe['extends'], recipe.get('overrides', {})
    if not isinstance(parent_id, str) or not isinstance(overrides, dict) or not isinstance(recipe.get('id'), str):
        raise ValueError('extends must name a recipe id, overrides must be an object, and the recipe needs an id')
    chain = _chain + (recipe['id'],)
    if parent_id in chain:
        raise ValueError('recipe extends cycle: ' + ' -> '.join(chain + (parent_id,)))
    candidates = []
    for sibling in sorted(path.parent.glob('*.json')):
        try:
            if json.loads(sibling.read_text()).get('id') == parent_id:
                candidates.append(sibling)
        except (ValueError, OSError):
            continue
    if len(candidates) != 1:
        raise ValueError('extends must match exactly one recipe id in the same directory: ' + parent_id)
    parent = resolve_recipe_data(candidates[0], chain)
    if parent.get('schema') != recipe.get('schema') or not isinstance(parent.get('stages'), list):
        raise ValueError('recipe and parent must share a multi-stage schema')
    top = {key: value for key, value in overrides.items() if key not in ('stages', 'stages_added')}
    for forbidden in ('id', 'schema', 'extends', 'overrides'):
        if forbidden in top:
            raise ValueError('overrides may not set ' + forbidden)
    resolved = deep_merge(parent, top)
    stage_overrides = overrides.get('stages', {})
    if not isinstance(stage_overrides, dict):
        raise ValueError('overrides.stages must map stage ids to objects')
    known = {stage['id'] for stage in resolved['stages']}
    if set(stage_overrides) - known:
        raise ValueError('override for unknown stage: ' + ', '.join(sorted(set(stage_overrides) - known)))
    stages = []
    for stage in resolved['stages']:
        change = stage_overrides.get(stage['id'])
        if _is_delete(change):
            continue
        if change is not None and not isinstance(change, dict):
            raise ValueError('stage override must be an object: ' + stage['id'])
        if change is not None and 'id' in change:
            raise ValueError('stage overrides may not change a stage id')
        stages.append(deep_merge(stage, change) if change is not None else stage)
    # ``stages_added``: new stages, each inserted after the named stage (an addition, never a change of the parent's).
    added = overrides.get('stages_added', [])
    if not isinstance(added, list):
        raise ValueError('overrides.stages_added must be a list of {"after", "stage"} objects')
    for entry in added:
        if (not isinstance(entry, dict) or set(entry) not in ({'after', 'stage'}, {'before', 'stage'})
                or not isinstance(entry['stage'], dict) or not isinstance(entry['stage'].get('id'), str)):
            raise ValueError('each added stage is {"after" or "before": stage id, "stage": object with an id}')
        ids = [stage['id'] for stage in stages]
        if entry['stage']['id'] in ids or entry['stage']['id'] in stage_overrides:
            raise ValueError('added stage id already exists: ' + entry['stage']['id'])
        anchor = entry.get('after', entry.get('before'))
        if anchor not in ids:
            raise ValueError('added stage is placed next to an unknown stage: ' + str(anchor))
        stages.insert(ids.index(anchor) + ('after' in entry), json.loads(json.dumps(entry['stage'])))
    resolved['stages'] = stages
    resolved['id'] = recipe['id']
    if 'description' in recipe:
        resolved['description'] = recipe['description']
    else:
        resolved.pop('description', None)
    for key in INHERITANCE_KEYS:
        resolved.pop(key, None)
    return resolved


def recipe_identity_sha256(path):
    """The recipe hash recorded in plans, receipts and certificates.

    A plain recipe keeps its file-bytes SHA-256 (receipts of earlier runs stay comparable). A recipe that uses
    ``extends`` is hashed as the canonical JSON of its resolved content (natlang_neuralese.common.hashing).
    """
    if 'extends' in json.loads(Path(path).read_text()):
        return canonical_json_sha256_hex(resolve_recipe_data(path))
    return sha(path)


DIRECT_IDENTITY_QUALIFICATION = 'neuralese_direct_step0_identity'


def load_recipe(path):
    recipe = resolve_recipe_data(path)
    if recipe.get('schema') == DIRECT_STAGE_SCHEMA:
        validate_direct_stage_recipe(recipe)
        return recipe
    if recipe.get('schema') != 'natlang.neuralese-training-recipe/1' or not recipe.get('stages'):
        raise ValueError('invalid or empty training recipe')
    parameter_defaults = recipe.get('stage_parameter_defaults', {})
    if not isinstance(parameter_defaults, dict):
        raise ValueError('stage_parameter_defaults must map handler names to parameter objects')
    for kind, defaults in parameter_defaults.items():
        refuse_retired_parameters(defaults)
        if (kind not in HANDLERS or not isinstance(defaults, dict) or
                not set(defaults) <= HANDLERS[kind]['parameters']):
            raise ValueError('invalid stage parameter defaults for ' + str(kind))
        validate_stage_parameters(kind, defaults)
        if not any(stage.get('kind') == kind for stage in recipe['stages']):
            raise ValueError('unused stage parameter defaults for ' + str(kind))
    declared, complete, identity_stages, embedding_stages, runtime_stages = set(), set(), set(), set(), set()
    for stage in recipe['stages']:
        name, kind = stage.get('id'), stage.get('kind')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name) or name in declared:
            raise ValueError('stage id must be unique and safe')
        if kind not in HANDLERS:
            raise ValueError('unknown stage implementation: ' + str(kind))
        required = stage.get('requires')
        if not isinstance(required, list) or len(set(required)) != len(required) or not set(required) <= complete:
            raise ValueError('stage dependencies must precede the stage')
        check_direct_phase_continuation(recipe, stage)
        parameters = stage.get('parameters')
        refuse_retired_parameters(parameters)
        if not isinstance(parameters, dict) or not set(parameters) <= HANDLERS[kind]['parameters']:
            raise ValueError('unknown stage parameters')
        validate_stage_parameters(kind, parameters)
        if 'text_data' in parameters and 'text_data' in stage.get('inputs', {}):
            raise ValueError('text_data must be declared either as a named stage input or a legacy parameter')
        if kind == 'core_text_warmup':
            ar_feedback_fixup = parameters.get('ar_feedback_fixup', False)
            if type(ar_feedback_fixup) is not bool:
                raise ValueError('ar_feedback_fixup must be boolean')
            if ar_feedback_fixup and not isinstance(stage.get('inputs', {}), dict):
                raise ValueError('AR feedback fixup stage inputs must be an object')
            if ar_feedback_fixup and not any(
                    prior['id'] in required and prior['kind']=='core_text_warmup' and
                    not prior.get('parameters',{}).get('ar_feedback_fixup',False)
                    for prior in recipe['stages']):
                raise ValueError('AR feedback fixup requires a preceding mapped-input warm-up stage')
            for option_name in ('input_map_kernel', 'input_map_rank'):
                value = parameters.get(option_name, 4 if option_name == 'input_map_kernel' else 64)
                if type(value) is not int or value < 1:
                    raise ValueError('invalid core text warm-up ' + option_name)
        if kind == 'token_identity':
            identity_stages.add(name)
        if kind == 'causal_embedding_distillation' and not set(required) & identity_stages:
            raise ValueError('embedding distillation requires an explicit token identity gate')
        if kind == 'causal_embedding_distillation':
            embedding_stages.add(name)
        if kind == 'raw_runtime_qualification' and not set(required) & embedding_stages:
            raise ValueError('runtime qualification requires an explicit embedding foundation')
        if kind == 'raw_runtime_qualification':
            runtime_stages.add(name)
        if kind == 'verified_heads_handoff' and not set(required) & runtime_stages:
            raise ValueError('verified heads handoff requires the raw runtime qualification stage')
        if kind == 'core_text_warmup' and 'continue_from' in stage.get('inputs', {}) and not stage.get('parameters',{}).get('ar_feedback_fixup',False):
            handoffs = [s for s in recipe['stages'] if s['id'] in required and s['kind']=='verified_heads_handoff']
            if (len(handoffs) != 1 or
                    handoffs[0].get('inputs', {}).get('warmup_checkpoint') != stage['inputs']['continue_from'] or
                    handoffs[0].get('inputs', {}).get('warmup_heads') != stage['inputs'].get('heads')):
                raise ValueError('continuation checkpoint and heads must be the exact files validated by a required verified_heads_handoff')
        if kind == 'core_text_warmup' and 'heads' in stage.get('inputs', {}) and 'continue_from' not in stage.get('inputs', {}):
            raise ValueError('stage-specific heads input requires a verified full-state continuation checkpoint')
        if kind == 'core_text_warmup' and stage.get('parameters',{}).get('ar_feedback_fixup',False):
            predecessors = [s for s in recipe['stages'] if s['id'] in required and
                           s['kind']=='core_text_warmup' and
                           not s.get('parameters',{}).get('ar_feedback_fixup',False)]
            if len(predecessors) != 1:
                raise ValueError('AR feedback fixup requires exactly one mapped warm-up predecessor')
        if kind == 'text_warmup_runtime' and 'heads' not in stage.get('inputs', {}):
            if not any(s['id'] in required and s['kind'] == 'core_text_warmup' for s in recipe['stages']):
                raise ValueError('adapted runtime requires a core warm-up predecessor or exact stage heads binding')
        if kind == 'raw_recurrence_training':
            runtime_dependencies = [s for s in recipe['stages']
                                    if s['id'] in required and s['kind'] == 'text_warmup_runtime']
            if not runtime_dependencies:
                raise ValueError('recurrence training requires an adapted warm-up runtime gate')
            stage_inputs = stage.get('inputs', {})
            if 'heads' in stage_inputs and 'direct_artifact_manifest' in stage_inputs:
                raise ValueError('recurrence stage must choose exactly one of heads or direct_artifact_manifest')
            if not ({'heads', 'direct_artifact_manifest'} & set(stage_inputs)) and not any(
                    s['id'] in required and s['kind'] == 'core_text_warmup' for s in recipe['stages']):
                raise ValueError('recurrence requires a core warm-up predecessor, exact stage heads, or direct artifact manifest')
        if kind == 'view_operator_gate' and not any(
                s['id'] in required and s['kind'] == 'raw_recurrence_training' for s in recipe['stages']):
            raise ValueError('the view operator gate requires the view stage (a raw_recurrence_training stage) it measures')
        declared.add(name)
        complete.add(name)
    # Neuralese direct v1 (owner decision 2026-10-11, AGENTS.md "Neuralese training foundation"): a lineage made only of
    # neuralese_direct stages starts from the base model at an exact identity with no warm-up; each stage's step-0
    # identity check (direct_sft: NeuraleseProjection equals the soft-token mixture, TokenChoiceProjection absent or
    # the identity, so logits equal the base model's) is its identity qualification. Older lineages keep the rule.
    direct_lineage = all(s['kind'] == 'neuralese_direct' for s in recipe['stages'])
    if direct_lineage and recipe.get('identity_qualification') != DIRECT_IDENTITY_QUALIFICATION:
        raise ValueError('a neuralese_direct lineage must declare identity_qualification ' +
                         repr(DIRECT_IDENTITY_QUALIFICATION))
    if not direct_lineage and (not identity_stages or
                               not any(s['kind'] == 'causal_embedding_distillation' for s in recipe['stages'])):
        raise ValueError('neuralese recipe must declare identity and embedding distillation stages')
    validate_input_bindings(recipe)
    declared_runtime_environment(recipe)
    if 'family' in recipe:
        # The nested family as an in-stage objective (owner 2026-10-12; maple/family.py): it ramps on the stage
        # curriculum the quantization component declares.
        from ..maple.family import validate_family
        validate_family(recipe['family'], {stage['id']: stage['kind'] for stage in recipe['stages']})
        progress = (recipe.get('quantization') or {}).get('stage_progress', {})
        if any(stage_id not in progress for stage_id in recipe['family']['stages']):
            raise ValueError('a family stage needs a curriculum: name it in quantization.stage_progress')
    if 'objective_continuation' in recipe:
        validate_objective_continuation(recipe)
    if 'quantization' in recipe:
        # QAT woven into the stages (owner 2026-10-10; train/quantization.py): one component, inherited by every
        # stage it names.
        from .quantization import validate_spec
        validate_spec(recipe['quantization'], {stage['id']: stage['kind'] for stage in recipe['stages']})
    if 'init' in recipe:
        from .quantization import validate_init
        validate_init(recipe['init'], recipe.get('quantization'))
    return recipe


_TRUSTED_RUNTIME_ENVIRONMENT = {'NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE': '1'}


def declared_runtime_environment(recipe):
    """Validate the small allowlist of recipe-declared runtime environment switches."""
    runtime = recipe.get('runtime')
    environment = runtime.get('environment', {}) if isinstance(runtime, dict) else {}
    if not isinstance(environment, dict) or any(
            key not in _TRUSTED_RUNTIME_ENVIRONMENT or value != _TRUSTED_RUNTIME_ENVIRONMENT[key]
            for key, value in environment.items()):
        raise ValueError('runtime.environment contains an unsupported declared setting')
    if environment:
        stages = ([recipe.get('stage', {})] if isinstance(recipe.get('stage'), dict)
                  else recipe.get('stages', []))
        te_enabled = any(
            stage.get('kind') == 'neuralese_direct' and
            stage.get('parameters', recipe.get('parameters', {})).get('fp8_training') == 'te-fp8'
            for stage in stages if isinstance(stage, dict))
        if not te_enabled:
            raise ValueError('trusted TE extra-state restore requires a declared te-fp8 neuralese_direct stage')
    return dict(environment)


def validate_direct_stage_recipe(recipe):
    """Validate a single shared handler stage with exact external lineage pins.

    This path is for a narrowly scoped repair that reuses previously qualified
    weights and therefore must not rerun unrelated foundation stages. It uses
    the same handler registry, input-role contracts, and typed parameter CLI as
    a full recipe.
    """
    if recipe.get('execution') != 'direct_shared_stage' or not isinstance(recipe.get('id'), str):
        raise ValueError('direct stage must declare its execution mode and recipe id')
    stage = recipe.get('stage')
    if not isinstance(stage, dict) or set(stage) != {'id', 'kind'}:
        raise ValueError('direct stage needs exactly an id and registered handler kind')
    if not re.fullmatch(r'[a-z][a-z0-9_-]*', stage['id']) or stage['kind'] not in HANDLERS:
        raise ValueError('invalid direct stage identity or handler')
    handler = HANDLERS[stage['kind']]
    inputs = recipe.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError('direct stage requires pinned input bindings')
    roles = set(inputs)
    if missing_stage_inputs(stage['kind'], roles) or not roles <= handler['required_inputs'] | handler['optional_inputs']:
        raise ValueError('direct stage input roles do not satisfy the shared handler contract')
    for role, binding in inputs.items():
        if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'} or
                not isinstance(binding['path'], str) or not binding['path'] or
                not isinstance(binding['sha256'], str) or
                not re.fullmatch(r'[0-9a-f]{64}', binding['sha256'])):
            raise ValueError('invalid direct stage input binding: ' + role)
    parameters = recipe.get('parameters')
    if not isinstance(parameters, dict) or not set(parameters) <= handler['parameters']:
        raise ValueError('direct stage contains unsupported handler parameters')
    validate_stage_parameters(stage['kind'], parameters)
    runtime = recipe.get('runtime')
    if (not isinstance(runtime, dict) or not re.fullmatch(r'sha256:[0-9a-f]{64}', runtime.get('image', '')) or
            not isinstance(runtime.get('package_code'), dict) or not runtime['package_code']):
        raise ValueError('direct stage requires a pinned image and shared package code inventory')
    declared_runtime_environment(recipe)
    lineage = recipe.get('lineage')
    if not isinstance(lineage, dict) or not lineage.get('scope'):
        raise ValueError('direct stage requires an explicit lineage scope')
    evidence = lineage.get('evidence', [])
    if not isinstance(evidence, list):
        raise ValueError('direct-stage lineage evidence must be a list')
    for item in evidence:
        if (not isinstance(item, dict) or set(item) != {'role', 'path', 'sha256'} or
                not isinstance(item['role'], str) or not item['role'] or
                not isinstance(item['path'], str) or not item['path'] or
                not isinstance(item['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
            raise ValueError('invalid direct-stage lineage evidence binding')
    if stage['kind'] == 'raw_recurrence_training':
        runtime_reports = [item for item in evidence if item['role'] == 'adapted_runtime']
        if len(runtime_reports) != 1:
            raise ValueError('direct recurrence requires exactly one pinned adapted_runtime lineage report')
        has_heads = 'heads' in inputs
        has_direct_artifact = 'direct_artifact_manifest' in inputs
        if has_heads == has_direct_artifact:
            raise ValueError('direct recurrence requires exactly one of heads or direct_artifact_manifest')
        if has_direct_artifact and Path(inputs['direct_artifact_manifest']['path']).name != 'manifest.json':
            raise ValueError('direct_artifact_manifest must bind the artifact manifest.json file')
    outputs = recipe.get('outputs')
    if not isinstance(outputs, dict) or not isinstance(outputs.get('directory'), str):
        raise ValueError('direct stage requires a declared output directory')
    return recipe


def verify_direct_stage_lineage_gates(stage, inputs, lineage_evidence):
    """Bind direct recurrence to the exact adapted-runtime gate for its heads.

    A direct stage has no in-recipe predecessor graph, so the runtime report is
    an explicit, hashed lineage input. Validate it before creating the output
    directory or starting the trainer.
    """
    if stage['kind'] != 'raw_recurrence_training':
        return []
    reports = [item for item in lineage_evidence if item['role'] == 'adapted_runtime']
    if len(reports) != 1:
        raise ValueError('direct recurrence requires exactly one pinned adapted_runtime lineage report')
    item = reports[0]
    try:
        report = json.loads(Path(item['path']).read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError('adapted_runtime lineage evidence is not a readable JSON report') from exc
    if not isinstance(report, dict):
        raise ValueError('adapted_runtime lineage report has an unsupported schema')
    if 'direct_artifact_manifest' in inputs:
        manifest_sha = inputs['direct_artifact_manifest']['sha256']
        if (report.get('schema') != 'natlang.neuralese-direct-runtime-qualification/1' or
                report.get('runtime_qualified') is not True or
                report.get('artifact_manifest_sha256') != manifest_sha):
            raise ValueError('adapted_runtime report does not qualify the exact direct artifact manifest')
        return [{'role': 'adapted_runtime', 'report_sha256': item['sha256'],
                 'direct_artifact_manifest_sha256': manifest_sha, 'runtime_qualified': True}]
    if report.get('schema') != 'natlang.latent-sketch-diagnostic/1':
        raise ValueError('adapted_runtime lineage report has an unsupported schema')
    require_gate(report, 'text_warmup_runtime')
    heads_sha = inputs['heads']['sha256']
    if report.get('parent_sha256') != heads_sha:
        raise ValueError('adapted_runtime lineage report belongs to different heads weights')
    if report.get('weights_unmodified') is not True or report.get('retained_trained_heads') is not True:
        raise ValueError('adapted_runtime lineage report does not attest retained, unmodified trained heads')
    return [{'role': 'adapted_runtime', 'report_sha256': item['sha256'],
             'heads_sha256': heads_sha, 'runtime_qualified': True,
             'output_reference_qualified': True, 'weights_unmodified': True,
             'retained_trained_heads': True}]


# Objective components an active stage may change by a declared objective continuation (``objective_continuation``;
# the trainer's direct_sft.OBJECTIVE_KEYS): the curriculum continues from the checkpoint's update without a jump.
OBJECTIVE_COMPONENTS = ('quantization', 'family')


def validate_objective_continuation(recipe):
    """``objective_continuation``: {"stage": the active neuralese_direct stage, "previous": {component: canonical
    sha256 of the component the stage trained with, or null when it had none}, "description"}. The declaration
    pins exactly what the running stage trained with, as a data continuation pins the data it trained on."""
    declared = recipe['objective_continuation']
    if not isinstance(declared, dict) or set(declared) - {'stage', 'previous', 'description'}:
        raise ValueError('objective_continuation is {stage, previous, description}')
    kinds = {stage['id']: stage['kind'] for stage in recipe['stages']}
    if kinds.get(declared.get('stage')) != 'neuralese_direct':
        raise ValueError('objective_continuation names a neuralese_direct stage of the recipe')
    previous = declared.get('previous')
    if (not isinstance(previous, dict) or not previous or set(previous) - set(OBJECTIVE_COMPONENTS) or
            not all(v is None or isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v) for v in previous.values())):
        raise ValueError('objective_continuation.previous maps ' + '/'.join(OBJECTIVE_COMPONENTS) +
                         ' to the canonical sha256 of the component the stage trained with (null: none)')


def objective_component_sha256(component):
    """The canonical SHA-256 an objective continuation pins a recipe component by (None when absent)."""
    return None if component is None else canonical_json_sha256_hex(component)


def objective_handoff(old_recipe, new_recipe, out):
    """A handoff that changes objective components (``OBJECTIVE_COMPONENTS``) must declare an objective continuation
    of the active stage pinning the old components. Finished stages keep their curriculum (their stage_progress and
    family membership); the active stage is the declared one; unstarted stages may change freely. The trainer then
    continues the curriculum from the checkpoint's update and refuses any λ or member-weight jump
    (direct_sft.objective_continuation). Returns (record or None, old recipe, new recipe) with the components and the
    declaration removed when they changed."""
    changed = [key for key in OBJECTIVE_COMPONENTS if old_recipe.get(key) != new_recipe.get(key)]
    if not changed:
        return None, old_recipe, new_recipe
    declared = new_recipe.get('objective_continuation')
    if not isinstance(declared, dict):
        raise ValueError('handoff refused: objective component(s) ' + ', '.join(changed) + ' changed without a '
                         'declared objective_continuation')
    stage_id = declared['stage']
    if not stage_started(out, stage_id) or stage_finished(out, stage_id):
        raise ValueError('handoff refused: objective_continuation names ' + stage_id + ', which is not the active '
                         'stage (started, unfinished)')
    for key in changed:
        if declared['previous'].get(key, 'missing') != objective_component_sha256(old_recipe.get(key)):
            raise ValueError('handoff refused: objective_continuation.previous.' + key + ' is not the running '
                             'lineage\'s component')
    for stage in new_recipe.get('stages', []):
        if not stage_finished(out, stage['id']):
            continue
        for key, member in (('quantization', lambda c, s: (c or {}).get('stage_progress', {}).get(s)),
                            ('family', lambda c, s: s in (c or {}).get('stages', []))):
            if member(old_recipe.get(key), stage['id']) != member(new_recipe.get(key), stage['id']):
                raise ValueError('handoff refused: finished stage ' + stage['id'] + ' keeps its ' + key)
    old_value, new_value = json.loads(json.dumps(old_recipe)), json.loads(json.dumps(new_recipe))
    for value in (old_value, new_value):
        for key in changed + ['objective_continuation']:
            value.pop(key, None)
    record = {'stage': stage_id, 'changed': changed,
              'previous_sha256': {key: objective_component_sha256(old_recipe.get(key)) for key in changed},
              'sha256': {key: objective_component_sha256(new_recipe.get(key)) for key in changed}}
    return record, old_value, new_value


def stage_objective_args(recipe, stage):
    """The stage's family component and objective-continuation declaration as trainer arguments."""
    from ..maple.family import stage_family
    command = []
    component = stage_family(recipe.get('family'), stage['id'])
    if component is not None:
        command += ['--family', json.dumps(component, sort_keys=True, separators=(',', ':'))]
    declared = recipe.get('objective_continuation')
    if isinstance(declared, dict) and declared.get('stage') == stage['id']:
        command += ['--objective-continuation', json.dumps({'stage': declared['stage'],
                                                            'previous': declared['previous']},
                                                           sort_keys=True, separators=(',', ':'))]
    return command


def stage_quantization(recipe, stage, parameters, reports, init_mix=None):
    """The recipe's quantization component for one stage, with its update range: a mapped warm-up from 0 to its
    steps, an AR fixup from its predecessor's step to its effective steps, a recurrence stage from 0 to its steps."""
    from .quantization import stage_component
    spec = recipe.get('quantization')
    if not spec or stage['id'] not in spec.get('stage_progress', {}):
        return None
    start = 0
    if stage['kind'] == 'core_text_warmup' and parameters.get('ar_feedback_fixup', False):
        start = next(r for r in reports if r['id'] in stage['requires'] and r['kind'] == 'core_text_warmup')['gate']['step']
    end = parameters.get('steps', 4096 if stage['kind'] == 'core_text_warmup' else 500)  # the trainers' defaults
    if stage['kind'] == 'neuralese_direct' and 'epochs' in parameters:
        if 'steps' in parameters or 'quantization_steps' in parameters:
            raise ValueError('a neuralese_direct stage declares its budget as epochs or steps, not both')
        end = 'derived'  # direct_sft derives the update count from its data and packing
        return stage_component(spec, stage['id'], start=start, end=end, init_mix=init_mix)
    if 'quantization_steps' in parameters:
        # The update range the stage's progress span maps onto, when the stage runs longer (an extension, e.g.
        # raw-recurrence-mellum-v12): updates past it hold the stage-end mixes, so extending a stage never moves its
        # ramp back.
        span = parameters['quantization_steps']
        if parameters.get('ar_feedback_fixup', False):
            raise ValueError('quantization_steps is not defined for an AR feedback fixup')
        if type(span) is not int or not start < span <= end:
            raise ValueError('quantization_steps must be an integer update count in (start, steps]')
        end = span
    return stage_component(spec, stage['id'], start=start, end=end, init_mix=init_mix)


def require_gate(report, kind):
    quantization = report.get('quantization') if isinstance(report, dict) else None
    if isinstance(quantization, dict) and quantization.get('precisions_qualified') is False:
        raise ValueError('stage did not pass every required precision: ' +
                         ', '.join(sorted(set(quantization.get('precisions_required', [])) -
                                          set(quantization.get('precisions_passed', [])))))
    if kind == 'token_identity':
        if report.get('token_aligned_reference_passed') is not True:
            raise ValueError('token identity gate failed')
    elif kind == 'causal_embedding_distillation':
        if report.get('feedback_gate_passed') is not True:
            raise ValueError('embedding distillation gate failed')
    elif kind == 'raw_runtime_qualification':
        if report.get('runtime_transport_passed') is not True:
            raise ValueError('raw runtime transport gate failed')
    elif kind == 'verified_heads_handoff':
        if report.get('handoff_verified') is not True:
            raise ValueError('exact trained heads handoff gate failed')
    elif kind == 'raw_recurrence_training':
        if report.get('training_stage_completed') is not True or report.get('errors') != 0:
            raise ValueError('raw recurrence training stage incomplete or errored')
    elif kind=='core_text_warmup':
        if report.get('qualified') is not True:raise ValueError('text warm-up alignment goal not met')
    elif kind=='neuralese_direct':
        if report.get('qualified') is not True:
            raise ValueError('Neuralese direct stage not qualified (held CE below base, no drift stop, behaviour gate)')
    elif kind=='text_warmup_runtime':
        if report.get('runtime_qualified') is not True or report.get('output_reference_qualified') is not True:
            raise ValueError('adapted warm-up runtime/output reference gate failed')
    elif kind == 'view_operator_gate':
        if report.get('view_gate_passed') is not True:
            raise ValueError('view operator gate failed')
    else:
        raise ValueError('no gate adapter for stage')


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


# Stage parameters that change how a stage computes, not what it optimizes: a lineage may switch them in place
# (``--handoff``). Each trainer also lists them among its resume-operational options.
OPERATIONAL_STAGE_PARAMETERS = frozenset({'eval_every', 'checkpoint_every', 'moe_kernel',
                                          # the execution backend; the trainer refuses a switch that loses FP8 state
                                          'fp8_training', 'checkpoint_layers', 'window_execution', 'report_column_fraction', 'drift_rows', 'drift_every', 'drift_gate_every', 'drift_gate_rise', 'drift_ce_windows', 'gate_response_kl',
                                          'drift_gate_now',  # retired by drift_gate_rise (b1cd6632); a lineage that declared it hands off
                                          'drift_warn', 'drift_stop', 'drift_agreement_warn', 'drift_agreement_stop'})
# Stage parameters the runner consumes itself (never passed to a trainer).
RUNNER_STAGE_PARAMETERS = frozenset({'quantization_steps'})


def direct_epoch_budget_change(stage_kind, old_parameters, new_parameters):
    """Return an epoch-only budget change for a direct stage.

    The trainer rebuilds both schedules from the pinned inputs and proves the
    checkpoint's consumed batches are an exact prefix of the new schedule.
    This helper only permits that narrowly scoped request to reach the trainer.
    """
    old_parameters, new_parameters = dict(old_parameters), dict(new_parameters)
    old_epochs, new_epochs = old_parameters.pop('epochs', None), new_parameters.pop('epochs', None)
    if old_parameters != new_parameters:
        return None
    if stage_kind != 'neuralese_direct':
        return None
    if (type(old_epochs) not in (int, float) or type(new_epochs) not in (int, float) or
            not math.isfinite(old_epochs) or not math.isfinite(new_epochs) or
            old_epochs <= 0 or new_epochs <= 0 or new_epochs == old_epochs):
        return None
    return {'old_epochs': old_epochs, 'new_epochs': new_epochs,
            'verification': 'trainer-reconstructs-exact-schedule-prefix-and-checkpoint-step'}


def direct_stage_epoch_handoffs(old_recipe, new_recipe):
    """Validate epoch-only budget changes on matching neuralese_direct stages."""
    old_stages = {stage.get('id'): stage for stage in old_recipe.get('stages', [])}
    changes = {}
    for stage in new_recipe.get('stages', []):
        old = old_stages.get(stage.get('id'))
        if old is None or old.get('kind') != stage.get('kind'):
            continue
        kind = stage.get('kind')
        old_parameters = dict(old_recipe.get('stage_parameter_defaults', {}).get(kind, {}))
        old_parameters.update(old.get('parameters', {}))
        new_parameters = dict(new_recipe.get('stage_parameter_defaults', {}).get(kind, {}))
        new_parameters.update(stage.get('parameters', {}))
        for key in OPERATIONAL_STAGE_PARAMETERS:
            old_parameters.pop(key, None)
            new_parameters.pop(key, None)
        if old_parameters.get('epochs') != new_parameters.get('epochs'):
            proof = direct_epoch_budget_change(kind, old_parameters, new_parameters)
            if proof is None:
                return None
            changes[stage['id']] = proof
    return changes


def recipe_handoff_epoch_changes(old_recipe, new_recipe):
    """Require recipe identity to differ only by runtime code, operational parameters, or direct-stage epochs."""
    changes = direct_stage_epoch_handoffs(old_recipe, new_recipe)
    if changes is None:
        raise ValueError('handoff refused: recipe changes outside the direct-stage epoch budget')
    old = _without_direct_epochs(_without_package_code(_without_operational(old_recipe)))
    new = _without_direct_epochs(_without_package_code(_without_operational(new_recipe)))
    if old != new:
        raise ValueError('handoff refused: the recipe changes more than operational stage parameters, '
                         'runtime.package_code, and neuralese_direct epoch budgets')
    return changes


def _without_direct_epochs(recipe):
    value = json.loads(json.dumps(recipe))
    defaults = value.get('stage_parameter_defaults', {})
    if isinstance(defaults, dict):
        direct = defaults.get('neuralese_direct')
        if isinstance(direct, dict):
            direct.pop('epochs', None)
    for stage in value.get('stages', []):
        if stage.get('kind') == 'neuralese_direct':
            stage.get('parameters', {}).pop('epochs', None)
    return value


def _stage_steps(recipe, stage):
    """(steps, quantization update range) of a stage as declared (no AR fixup offset)."""
    parameters = dict(recipe.get('stage_parameter_defaults', {}).get(stage.get('kind'), {}))
    parameters.update(stage.get('parameters', {}))
    steps = parameters.get('steps')
    return steps, parameters.get('quantization_steps', steps)


def stage_extensions(existing, recipe, out):
    """The stages a handoff extends: ``steps`` raised on a stage that has no qualified report yet, its quantization
    update range unchanged (``quantization_steps``), so the stage continues from its state with its ramp held at the
    stage-end mixes. Anything else about steps (a shorter stage, an extended qualified stage, a moved ramp) is refused.
    Returns {stage id: [old steps, new steps]}."""
    old_stages = {stage['id']: stage for stage in existing.get('stages', [])}
    extended = {}
    for stage in recipe.get('stages', []):
        old = old_stages.get(stage['id'])
        if old is None:
            continue
        (old_steps, old_span), (new_steps, new_span) = _stage_steps(existing, old), _stage_steps(recipe, stage)
        if old_steps == new_steps and old_span == new_span:
            continue
        if (stage.get('kind') != 'core_text_warmup' or stage.get('parameters', {}).get('ar_feedback_fixup', False)
                or type(old_steps) is not int or type(new_steps) is not int or new_steps <= old_steps):
            raise ValueError('handoff refused: only a mapped text warm-up may be extended (more steps): ' + stage['id'])
        if old_span != new_span:
            raise ValueError('handoff refused: extending ' + stage['id'] + ' moves its quantization ramp; '
                             'set quantization_steps to the old steps (' + str(old_span) + ')')
        if (Path(out) / (stage['id'] + '-report.json')).exists():
            raise ValueError('handoff refused: ' + stage['id'] + ' already qualified; it cannot be extended')
        extended[stage['id']] = [old_steps, new_steps]
    return extended


def _without_extensions(recipe, extended):
    value = json.loads(json.dumps(recipe))
    for stage in value.get('stages', []):
        if stage['id'] in extended:
            for key in ('steps', 'quantization_steps'):
                stage.get('parameters', {}).pop(key, None)
    return value


def _without_operational(recipe):
    """The resolved recipe minus its name, description, ancestry and operational stage parameters."""
    value = json.loads(json.dumps(recipe))
    for key in ('id', 'description', 'extends'):
        value.pop(key, None)
    for defaults in value.get('stage_parameter_defaults', {}).values():
        for key in OPERATIONAL_STAGE_PARAMETERS:
            defaults.pop(key, None)
    value['stage_parameter_defaults'] = {k: v for k, v in value.get('stage_parameter_defaults', {}).items() if v}
    for stage in value.get('stages', []):
        for key in OPERATIONAL_STAGE_PARAMETERS:
            stage.get('parameters', {}).pop(key, None)
    return value


def _without_package_code(recipe):
    """Remove only the frozen source-code inventory from a recipe comparison."""
    value = json.loads(json.dumps(recipe))
    runtime = value.get('runtime')
    if isinstance(runtime, dict):
        runtime.pop('package_code', None)
    return value


def stage_started(out, stage_id):
    """A stage has started once its output directory or its report exists; its inputs are then fixed."""
    out = Path(out)
    return (out / stage_id).exists() or (out / (stage_id + '-report.json')).exists()


def stage_finished(out, stage_id):
    return (Path(out) / (stage_id + '-report.json')).exists()


DATA_CONTINUATION_ROLE = 'continued_text_data'


def _data_continuation(old, new):
    """Whether a started stage's new inputs are a declared data continuation of its old ones: the stage now reads new
    ``text_data`` and declares ``continued_text_data`` as exactly the text data it trained on (content), every other
    role unchanged."""
    if 'text_data' not in old or 'text_data' not in new or DATA_CONTINUATION_ROLE not in new:
        return False
    if new[DATA_CONTINUATION_ROLE].get('sha256') != old['text_data'].get('sha256'):
        return False
    if new['text_data'].get('sha256') == old['text_data'].get('sha256'):
        return False
    rest = lambda roles: {k: v for k, v in roles.items() if k not in ('text_data', DATA_CONTINUATION_ROLE)}
    return rest(old) == rest(new)


def unstarted_stage_input_changes(existing, plan, out):
    """The stages whose resolved inputs a handoff changes. A later stage that has not started may take new content for
    its declared input roles (e.g. a newer data rendering for a not-yet-reached phase). The active (started,
    unfinished) stage may change its inputs only by a declared data continuation (``continued_text_data`` naming the
    text data it trained on; direct_sft continues its update counter, weights and optimizer on the new data, the seen
    documents counted as seen). A finished stage's inputs stay fixed. Returns {stage id: {'kind', 'old', 'new'}}."""
    old_inputs, new_inputs = existing.get('stage_inputs'), plan.get('stage_inputs')
    if old_inputs == new_inputs:
        return {}
    if not isinstance(old_inputs, dict) or not isinstance(new_inputs, dict) or set(old_inputs) != set(new_inputs):
        raise ValueError('handoff refused: stage_inputs changed; use a new stage lineage')
    changes = {}
    for stage_id in sorted(new_inputs):
        old, new = old_inputs[stage_id], new_inputs[stage_id]
        if old == new:
            continue
        if stage_started(out, stage_id):
            if stage_finished(out, stage_id) or not _data_continuation(old, new):
                raise ValueError('handoff refused: inputs of started stage ' + stage_id + ' changed; only a stage '
                                 'that has not started may take new inputs, and the active stage only by a declared '
                                 'data continuation (' + DATA_CONTINUATION_ROLE + ' = the text data it trained on)')
            changes[stage_id] = {'kind': 'data_continuation', 'old': old, 'new': new}
            continue
        if set(old) != set(new) or any(old[role].get('binding') != new[role].get('binding') for role in new):
            raise ValueError('handoff refused: ' + stage_id + ' changes its input roles or bindings; only their '
                             'declared content may change')
        changes[stage_id] = {'kind': 'unstarted_stage', 'old': old, 'new': new}
    return changes


def _bindings_of(recipe, stage_ids):
    return {name for stage in recipe.get('stages', []) if stage['id'] in stage_ids
            for name in (stage.get('inputs') or {}).values()}


def _without_unstarted_bindings(old_recipe, new_recipe, changes):
    """Both recipes minus the input bindings that changed for the changed stages only (and a data continuation's role
    map); refuse a changed binding that another started or unchanged stage also reads."""
    old_catalog, new_catalog = old_recipe.get('input_bindings', {}), new_recipe.get('input_bindings', {})
    changed = {name for name in set(old_catalog) | set(new_catalog) if old_catalog.get(name) != new_catalog.get(name)}
    if not changes:
        return old_recipe, new_recipe
    others = [stage['id'] for stage in new_recipe.get('stages', []) if stage['id'] not in changes]
    shared = changed & (_bindings_of(old_recipe, others) | _bindings_of(new_recipe, others))
    if shared:
        raise ValueError('handoff refused: changed input binding(s) ' + ', '.join(sorted(shared)) +
                         ' also feed a started or unchanged stage')
    if not changed <= _bindings_of(new_recipe, changes) | _bindings_of(old_recipe, changes):
        raise ValueError('handoff refused: input bindings changed outside the unstarted stages')
    old_value, new_value = json.loads(json.dumps(old_recipe)), json.loads(json.dumps(new_recipe))
    for value in (old_value, new_value):
        for name in changed:
            value.get('input_bindings', {}).pop(name, None)
        for stage in value.get('stages', []):
            if changes.get(stage['id'], {}).get('kind') == 'data_continuation':
                stage.pop('inputs', None)  # its role map moves text_data and declares continued_text_data
    return old_value, new_value


def _inputs_without(inputs, stage_inputs, changes, side):
    """The plan's input identity map minus files read only by the changed (unstarted) stages."""
    kept = {value['path'] for stage_id, roles in stage_inputs.items() if stage_id not in changes
            for value in roles.values()}
    dropped = {value['path'] for change in changes.values() for value in change[side].values()} - kept
    return {path: digest for path, digest in inputs.items() if path not in dropped}


def lineage_handoff(existing, plan, frozen, package, plan_path, out=None, dry_run=False):
    """Resume a lineage on the newest code and an operationally changed recipe (owner: jobs switch to the newest code
    at once). Device and the inputs of every started stage must be unchanged and the recipes equal apart from
    operational stage parameters, so the lineage's identity (``recipe_sha256``, which its stage reports carry) stays;
    the frozen runtime is replaced, the superseded one kept beside it, and the change appended to the plan's
    ``handoffs``. Each trainer then checks its own resume identity and logs its code handoff. A handoff may also
    extend an unqualified mapped warm-up (``stage_extensions``); the extension is recorded under ``extended``. A
    later stage that has not started may take new content for its declared inputs, and the active stage new text data
    by a declared data continuation (``unstarted_stage_input_changes``; recorded under ``stage_input_changes`` with
    the previous input map, which reports of earlier stages carry; the trainer records the seen documents and the new
    budget in its data-continuation.json and the stage report).
    ``dry_run`` validates and returns the updated plan without touching the run directory."""
    out = Path(out if out is not None else plan_path.parent)
    if existing.get('device') != plan.get('device'):
        raise ValueError('handoff refused: device changed; use a new stage lineage')
    input_changes = unstarted_stage_input_changes(existing, plan, out)
    if (_inputs_without(existing.get('inputs'), existing.get('stage_inputs') or {}, input_changes, 'old') !=
            _inputs_without(plan.get('inputs'), plan.get('stage_inputs') or {}, input_changes, 'new')):
        raise ValueError('handoff refused: inputs changed; use a new stage lineage')
    extended = stage_extensions(existing['recipe'], plan['recipe'], out)
    old_handoff_recipe = _without_extensions(existing['recipe'], extended)
    new_handoff_recipe = _without_extensions(plan['recipe'], extended)
    old_handoff_recipe, new_handoff_recipe = _without_unstarted_bindings(
        old_handoff_recipe, new_handoff_recipe, input_changes)
    objective_change, old_handoff_recipe, new_handoff_recipe = objective_handoff(
        old_handoff_recipe, new_handoff_recipe, out)
    epoch_budget_changes = recipe_handoff_epoch_changes(old_handoff_recipe, new_handoff_recipe)
    old_code = existing['code']
    old_runtime = existing['recipe'].get('runtime', {})
    new_runtime = plan['recipe'].get('runtime', {})
    old_pin = old_runtime.get('package_code') if isinstance(old_runtime, dict) else None
    new_pin = new_runtime.get('package_code') if isinstance(new_runtime, dict) else None
    if old_pin is not None or new_pin is not None:
        if not isinstance(old_pin, dict) or not isinstance(new_pin, dict):
            raise ValueError('handoff refused: runtime.package_code must be pinned on both recipe versions')
        if old_pin != old_code:
            raise ValueError('handoff refused: old recipe code pin differs from its recorded frozen runtime')
        if not frozen.is_dir():
            raise ValueError('handoff refused: old frozen runtime is missing')
        actual_old = {str(path.relative_to(frozen)): sha(path) for path in sorted(frozen.rglob('*.py'))}
        if actual_old != old_code:
            raise ValueError('handoff refused: old frozen runtime differs from its recorded code inventory')
        actual_new = {str(path.relative_to(package)): sha(path) for path in sorted(package.rglob('*.py'))}
        if actual_new != new_pin:
            raise ValueError('handoff refused: new recipe code pin differs from the current shared package')
        if old_code == actual_new:
            raise ValueError('handoff refused: runtime.package_code has not changed')
    previous = frozen.parent.parent / ('runtime-superseded-' + time.strftime('%Y%m%dT%H%M%S'))
    if dry_run:
        new_code = {str(path.relative_to(package)): sha(path) for path in package.rglob('*.py')}
    else:
        frozen.parent.rename(previous)
        shutil.copytree(package, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        new_code = {str(path.relative_to(frozen)): sha(path) for path in frozen.rglob('*.py')}
    record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'recipe_id': plan['recipe'].get('id'),
              'recipe_sha256': plan['recipe_sha256'], 'superseded_runtime': previous.name,
              'code_changed': sorted(k for k in old_code.keys() & new_code.keys() if old_code[k] != new_code[k]),
              'code_added': sorted(new_code.keys() - old_code.keys()),
              'code_removed': sorted(old_code.keys() - new_code.keys())}
    if extended:
        record['extended'] = extended
    if epoch_budget_changes:
        record['epoch_budget_changes'] = epoch_budget_changes
    if objective_change:
        record['objective_continuation'] = objective_change
    updated = dict(existing, recipe=plan['recipe'], code=new_code,
                   handoffs=list(existing.get('handoffs', [])) + [record])
    if input_changes:
        record['stage_input_changes'] = input_changes
        record['previous_inputs'] = existing['inputs']
        updated.update(inputs=plan['inputs'], stage_inputs=plan['stage_inputs'])
    if dry_run:
        return updated
    write_json(plan_path, updated)
    print(json.dumps({'event': 'lineage_handoff', **{k: v for k, v in record.items() if not k.startswith('code_')},
                      'code_files_changed': len(record['code_changed']) + len(record['code_added'])
                      + len(record['code_removed'])}), flush=True)
    return updated


class _ChildSignalForwarder:
    """Forward container termination signals to the active training child.

    Container PID 1 can ignore the default SIGTERM disposition. Recipe runners
    therefore install a handler and explicitly notify their child, allowing the
    trainer to finish its current update and write its resumable checkpoint.
    The same object is used by direct and multi-stage recipes.
    """
    def __init__(self):
        self.stopped = [False]
        self.child = [None]
        self._previous = {}

    def _handle(self, sig, _frame):
        self.stopped[0] = True
        child = self.child[0]
        self._send_if_running(child, sig)

    @staticmethod
    def _send_if_running(child, sig):
        if child is None:
            return
        try:
            if child.poll() is None:
                child.send_signal(sig)
        except ProcessLookupError:
            # The child can exit between poll() and send_signal().
            pass

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def attach(self, child):
        self.child[0] = child
        # A signal can arrive after the handler is installed but before Popen
        # returns. Deliver it once the process handle becomes available.
        if self.stopped[0]:
            self._send_if_running(child, signal.SIGTERM)
        return child

    def detach(self, child):
        if self.child[0] is child:
            self.child[0] = None

    def close(self):
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        self._previous.clear()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()
        return False

    def run(self, command, *, env=None):
        child = self.attach(subprocess.Popen(command, env=env))
        try:
            return subprocess.CompletedProcess(command, child.wait())
        finally:
            self.detach(child)


def stage_parameter_args(parameters):
    """Serialize a validated stage's typed options into CLI arguments."""
    command = []
    for key, value in parameters.items():
        if key=='cohort_weights':
            if not isinstance(value,dict):
                raise ValueError('cohort_weights must be an object of explicit fractions')
            command += ['--cohort-weights',json.dumps(value,sort_keys=True,separators=(',',':'))]
            continue
        if key in ('optimizer_added', 'curriculum_change'):
            if not isinstance(value, list) or any(not isinstance(prefix, str) or not prefix for prefix in value):
                raise ValueError(key + ' must be a list of nonempty strings')
            for prefix in value:
                command += ['--' + key.replace('_', '-'), prefix]
            continue
        if isinstance(value, bool):
            command += ['--' + ('' if value else 'no-') + key.replace('_', '-')]
        elif isinstance(value, dict):  # a declared JSON component (e.g. neuralese_direct's monitor)
            command += ['--' + key.replace('_', '-'), json.dumps(value, sort_keys=True, separators=(',', ':'))]
        else:
            command += ['--' + key.replace('_', '-'), str(value)]
    return command


def recurrence_completion_gate(checkpoint, steps):
    """Read committed progress, rather than infer completion from a clean exit."""
    import torch
    state = torch.load(checkpoint, mmap=True, weights_only=False, map_location='cpu')
    if state.get('schema') != 'natlang.neuralese_recurrence_checkpoint/1':
        raise ValueError('wrong recurrence artifact schema')
    return {'training_stage_completed': state['step'] >= steps,
            'step': state['step'], 'errors': state['errors'], 'semantic_channel_qualified': False}


def run_declared_direct_stage(recipe, recipe_path, output, launch_metadata, device, handoff=False):
    """Freeze, bind, and run one explicitly declared shared stage.

    Direct stages are only appropriate when lineage evidence already qualifies
    their prerequisites and rerunning those independent stages would change the
    experiment. The declaration still uses the shared handler registry and
    records exact input/code/runtime pins before the child starts.
    """
    validate_direct_stage_recipe(recipe)
    output = Path(output).resolve()
    from ..common import hdd_guard
    hdd_guard.require_nvme(output)
    declared_output = Path(recipe['outputs']['directory']).resolve()
    if output != declared_output:
        raise ValueError('direct-stage output differs from its declaration')
    previous = None
    if output.exists():
        previous = json.loads((output / 'direct-stage-plan.json').read_text())
    elif recipe['stage']['kind'] == 'neuralese_direct' or 'direct_artifact_manifest' in recipe['inputs']:
        # A fresh launch of a line's direct or end-to-end stage is checked against the shared base (LINE_PARITY.md).
        from .line_parity import check_declared_stage
        check_declared_stage(recipe)
    metadata_path = Path(launch_metadata).resolve()
    if metadata_path.exists() or output == metadata_path or output in metadata_path.parents:
        raise ValueError('direct-stage launch metadata must be a fresh external path')

    inputs = {}
    for role, binding in recipe['inputs'].items():
        path = Path(binding['path']).resolve()
        if sha(path) != binding['sha256']:
            raise ValueError('direct-stage input hash mismatch: ' + role)
        inputs[role] = {'path': str(path), 'sha256': binding['sha256']}
    lineage_evidence = []
    for item in recipe['lineage'].get('evidence', []):
        path = Path(item['path']).resolve()
        if sha(path) != item['sha256']:
            raise ValueError('direct-stage lineage evidence hash mismatch: ' + item['role'])
        lineage_evidence.append({'role': item['role'], 'path': str(path), 'sha256': item['sha256']})
    lineage_gate_verifications = verify_direct_stage_lineage_gates(
        recipe['stage'], inputs, lineage_evidence)

    package = Path(__file__).parents[1]
    actual_code = {str(path.relative_to(package)): sha(path)
                   for path in sorted(package.rglob('*.py'))}
    if actual_code != recipe['runtime']['package_code']:
        raise ValueError('direct-stage shared package differs from its frozen code inventory')

    stage = recipe['stage']
    stage_output = output / stage['id']
    handler = HANDLERS[stage['kind']]
    resume = None
    if previous is not None:
        for key, value in [('stage', stage), ('inputs', inputs), ('device', device)]:
            if previous.get(key) != value:
                raise ValueError('direct-stage resume changes ' + key)
        old_parameters, new_parameters = dict(previous['parameters']), dict(recipe['parameters'])
        if handoff:
            for key in OPERATIONAL_STAGE_PARAMETERS:
                old_parameters.pop(key, None)
                new_parameters.pop(key, None)
        epoch_budget_handoff = (direct_epoch_budget_change(
            stage['kind'], old_parameters, new_parameters) if handoff else None)
        if old_parameters != new_parameters and epoch_budget_handoff is None:
            raise ValueError('direct-stage resume changes training parameters')
        if previous['frozen_code'] != actual_code and not handoff:
            raise ValueError('direct-stage code change requires --handoff')
        checkpoint = stage_output / 'checkpoint.pt'
        if not checkpoint.is_file():
            raise ValueError('direct-stage resume requires the committed handler checkpoint')
        history = output / 'resume-history' / (time.strftime('%Y%m%dT%H%M%S') + '-' + str(os.getpid()))
        history.mkdir(parents=True)
        for name in ('direct-stage-plan.json', 'direct-stage-result.json'):
            path = output / name
            if path.is_file():
                shutil.copyfile(path, history / name)
        resume = {'checkpoint_path': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
                  'previous_plan_path': str(history / 'direct-stage-plan.json'),
                  'previous_plan_sha256': sha(history / 'direct-stage-plan.json'),
                  'code_handoff': previous['frozen_code'] != actual_code}
        if epoch_budget_handoff is not None:
            resume['epoch_budget_handoff'] = epoch_budget_handoff
        if resume['code_handoff']:
            (output / 'runtime').rename(history / 'runtime')
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / 'runtime' / 'natlang_neuralese'
    if not frozen.exists():
        shutil.copytree(package, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    frozen_code = {str(path.relative_to(frozen)): sha(path)
                   for path in sorted(frozen.rglob('*.py'))}
    if frozen_code != actual_code:
        raise ValueError('direct-stage runtime snapshot differs from reviewed package bytes')
    # The handler owns its fresh output boundary; pre-creating this path breaks
    # handlers that reject an existing --out directory before creating it.
    stage_output = output / stage['id']
    handler = HANDLERS[stage['kind']]
    command = [sys.executable, '-m', handler['module']]
    if 'heads' in inputs:
        command += ['--heads', hdd_guard.input_path('heads', inputs['heads']['path'], inputs['heads']['sha256'])]
    command += stage_input_args(inputs, stage['kind'])
    command += ['--out', str(stage_output), '--device', device]
    parameters = dict(recipe['parameters'])
    if 'text_data' in inputs:
        parameters.pop('text_data', None)
    command += stage_parameter_args(parameters)
    plan = {'schema': 'natlang.neuralese-direct-stage-plan/1', 'recipe_id': recipe['id'],
            'recipe_path': str(Path(recipe_path).resolve()), 'recipe_sha256': recipe_identity_sha256(recipe_path),
            'stage': stage, 'inputs': inputs, 'device': device,
            'runtime_image': recipe['runtime']['image'], 'frozen_code': frozen_code,
            'parameters': parameters, 'child_argv': command, 'stage_output': str(stage_output),
            'lineage': recipe['lineage'], 'lineage_evidence': lineage_evidence,
            'lineage_gate_verifications': lineage_gate_verifications,
            'runtime_environment': declared_runtime_environment(recipe)}
    if hdd_guard.RECORDS:
        plan['hdd_staged_inputs'] = list(hdd_guard.RECORDS)
    if resume is not None:
        plan['resume'] = resume
    write_json(output / 'direct-stage-plan.json', plan)
    write_json(metadata_path, {'schema': 'natlang.neuralese-direct-stage-launch-intent/1',
                               'recipe_id': recipe['id'], 'recipe_sha256': plan['recipe_sha256'],
                               'plan_path': str(output / 'direct-stage-plan.json'),
                               'plan_sha256': sha(output / 'direct-stage-plan.json'),
                               'output_path': str(output), 'device': device,
                               'input_sha256': {role: value['sha256'] for role, value in inputs.items()},
                               'lineage_evidence_sha256': {item['role']: item['sha256'] for item in lineage_evidence},
                               'runtime_image': recipe['runtime']['image'],
                               'frozen_code_count': len(frozen_code),
                               'frozen_code_manifest_sha256': hashlib.sha256(
                                   json.dumps(frozen_code, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                               'child_argv': command})
    environment = dict(os.environ)
    declared_environment = declared_runtime_environment(recipe)
    if 'NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE' not in declared_environment:
        environment.pop('NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE', None)
    environment.update(declared_environment)
    environment['PYTHONPATH'] = str(frozen.parent) + os.pathsep + environment.get('PYTHONPATH', '')
    print(json.dumps({'stage': stage['id'], 'command': command}), flush=True)
    with _ChildSignalForwarder() as signal_forwarder:
        result = signal_forwarder.run(command, env=environment)
    stage_report = stage_output / 'report.json'
    report = json.loads(stage_report.read_text()) if stage_report.is_file() else None
    if stage['kind'] == 'raw_recurrence_training' and (stage_output / handler['result']).is_file():
        report = recurrence_completion_gate(stage_output / handler['result'], parameters.get('steps', 500))
    qualified = False
    if result.returncode == 0 and report is not None:
        try:
            require_gate(report, stage['kind'])
            qualified = True
        except ValueError:
            qualified = False
    outputs = {str(path.relative_to(output)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
               for path in sorted(stage_output.rglob('*')) if path.is_file()}
    write_json(output / 'direct-stage-result.json',
               {'schema': 'natlang.neuralese-direct-stage-result/1', 'recipe_id': recipe['id'],
                'stage_id': stage['id'], 'process_exit_code': result.returncode,
                'interrupted': signal_forwarder.stopped[0],
                'process_finished': True,
                'completed': result.returncode == 0 and not signal_forwarder.stopped[0] and
                    (bool(report and report.get('training_stage_completed', False)) if stage['kind'] == 'raw_recurrence_training'
                     else stage['kind'] != 'neuralese_direct' or bool(report and report.get('completed', False))),
                'qualified': qualified,
                'gate_scope': recipe['lineage'].get('qualification_scope'),
                'report': report, 'outputs': outputs})
    if signal_forwarder.stopped[0]:
        print(json.dumps({'status':'interrupted','stage':stage['id'],
                          'child_exit_code':result.returncode}),flush=True)
    return result.returncode


def require_foundation(certificate, *, heads, checkpoint):
    """Downstream API: validate exact handoff, not an unrelated passed report.

    This validates foundation only. A runtime must separately qualify its own
    transport/gradient replay, then requalify when its backbone changes.
    """
    proof = json.loads(Path(certificate).read_text())
    if proof.get('schema') != 'natlang.neuralese-foundation-certificate/1' or proof.get('qualified') is not True:
        raise ValueError('qualified foundation certificate required')
    if proof['heads_sha256'] != sha(heads) or proof['feedback_checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('foundation certificate belongs to different weights')
    for stage in proof['stages']:
        # Registered snapshots may relocate a recipe directory across machines.
        # Resolve its adjacent reports first; exact hashes still bind all bytes.
        adjacent = Path(certificate).parent / Path(stage['report']).name
        report_path = adjacent if adjacent.is_file() else Path(stage['report'])
        if sha(report_path) != stage['report_sha256']:
            raise ValueError('foundation stage report changed')
        report = json.loads(report_path.read_text())
        require_gate(report['gate'], report['kind'])
    kinds = {stage['kind'] for stage in proof['stages']}
    if not {'token_identity', 'causal_embedding_distillation'} <= kinds:
        raise ValueError('foundation stages missing')
    return proof


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == 'resolve':
        resolver = argparse.ArgumentParser(prog='recipe resolve', description='print the resolved recipe and its hash')
        resolver.add_argument('recipe_id', help='recipe id or path to a recipe JSON file')
        resolver.add_argument('--dir', type=Path, default=Path(__file__).parents[2] / 'recipes')
        options = resolver.parse_args(argv[1:])
        path = Path(options.recipe_id)
        if not path.is_file():
            matches = [f for f in sorted(options.dir.glob('*.json'))
                       if json.loads(f.read_text()).get('id') == options.recipe_id]
            if len(matches) != 1:
                resolver.error('no unique recipe with id ' + options.recipe_id)
            path = matches[0]
        recipe = load_recipe(path)
        print(json.dumps(recipe, indent=2, ensure_ascii=False))
        print(json.dumps({'recipe_id': recipe['id'], 'recipe_sha256': recipe_identity_sha256(path),
                          'source_file': str(path), 'inherits': 'extends' in json.loads(path.read_text())}))
        return 0
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--heads', type=Path,
                        help='required by multi-stage recipes; direct stages pin heads in their input manifest')
    parser.add_argument('--records', type=Path, help='legacy shared fallback; prefer stage inputs in declared recipes')
    parser.add_argument('--pieces', type=Path, help='legacy shared fallback; prefer stage inputs in declared recipes')
    parser.add_argument('--text-data', type=Path, help='legacy shared fallback for text warm-up stages')
    parser.add_argument('--input-binding', action='append', default=[], metavar='NAME=PATH',
                        help='relocate a declared named input while preserving its recipe-pinned SHA-256')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--launch-metadata', type=Path,
                        help='write an immutable launch-intent receipt outside --out before running')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--until', help='run through a declared stage, retaining resumable state')
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--handoff', action='store_true',
                        help='resume an existing lineage on the newest code (the frozen runtime is replaced) and on a '
                             'recipe that differs only in operational stage parameters and schedule-verified '
                             'neuralese_direct epoch budgets; recorded in recipe-plan.json "handoffs"')
    parser.add_argument('--plan-only', action='store_true',
                        help='resolve and hash every stage input and validate the resume (or --handoff) against the '
                             'existing recipe-plan.json, print the resulting plan, and exit without writing or launching')
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    if recipe.get('schema') == DIRECT_STAGE_SCHEMA:
        if args.inspect:
            print(json.dumps(recipe, indent=2))
            return 0
        if args.input_binding or args.records or args.pieces or args.text_data or args.until or args.heads:
            parser.error('direct-stage recipes use only their pinned inputs and declared output')
        if args.launch_metadata is None:
            parser.error('direct-stage execution requires --launch-metadata')
        return run_declared_direct_stage(recipe, args.recipe, args.out, args.launch_metadata, args.device, args.handoff)
    if args.heads is None:
        parser.error('multi-stage training recipes require --heads')
    if args.until and args.until not in {stage['id'] for stage in recipe['stages']}:
        parser.error('unknown stopping stage')
    if args.inspect:
        print(json.dumps(recipe, indent=2))
        return
    if (args.records or args.pieces or args.text_data) and all(
            'inputs' in stage for stage in recipe['stages'] if stage.get('admitted') is not False):
        # Line parity (plans/neuralese/LINE_PARITY.md): a recipe that declares its data is the only source of it.
        parser.error('this recipe declares its data inputs, so --records/--pieces/--text-data cannot replace them; '
                     'relocate a declared file with --input-binding NAME=PATH (its sha256 must match)')
    args.out = args.out.resolve()
    args.heads = args.heads.resolve()
    from ..common import hdd_guard
    hdd_guard.require_nvme(args.out)
    overrides = parse_input_binding_overrides(args.input_binding)
    defaults = {'records': args.records, 'pieces': args.pieces, 'text_data': args.text_data}
    stage_inputs = {}
    for stage in recipe['stages']:
        stage_defaults = dict(defaults)
        if 'text_data' in stage['parameters']:
            stage_defaults['text_data'] = stage['parameters']['text_data']
        # A held stage (admitted: false) never runs in this lineage; its inputs are neither resolved nor hashed.
        stage_inputs[stage['id']] = ({} if stage.get('admitted') is False else
                                     resolve_stage_inputs(recipe, stage, stage_defaults, overrides))
    inputs = {str(args.heads): sha(args.heads)}
    for path in (args.records, args.pieces, args.text_data):
        if path is not None:
            resolved = path.resolve()
            inputs[str(resolved)] = sha(resolved)
    for resolved in stage_inputs.values():
        for value in resolved.values():
            inputs[value['path']] = value['sha256']
    package = Path(__file__).parents[1]
    frozen = args.out / 'runtime' / 'natlang_neuralese'
    plan_path = args.out / 'recipe-plan.json'
    plan = {'schema': 'natlang.neuralese-recipe-plan/1', 'recipe': recipe,
            'recipe_sha256': recipe_identity_sha256(args.recipe), 'inputs': inputs, 'device': args.device}
    declared_environment = declared_runtime_environment(recipe)
    if declared_environment:
        plan['runtime_environment'] = declared_environment
    if any('inputs' in stage for stage in recipe['stages']):
        plan['stage_inputs'] = stage_inputs
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if args.handoff:
            plan = lineage_handoff(existing, plan, frozen, package, plan_path, args.out, dry_run=args.plan_only)
        else:
            if (existing.get('recipe') == plan['recipe'] and existing.get('recipe_sha256') != plan['recipe_sha256'] and
                    'extends' in json.loads(args.recipe.read_text())):
                # A recipe converted to extends/overrides after this lineage began: identical resolved content keeps
                # the lineage and its originally recorded recipe hash.
                plan['recipe_sha256'] = existing['recipe_sha256']
            if any(existing[key] != value for key, value in plan.items()):
                raise ValueError('recipe or input identity changed; use a new stage lineage (or --handoff for '
                                 'newer code and operational recipe parameters)')
            if {str(path.relative_to(frozen)): sha(path) for path in frozen.rglob('*.py')} != existing['code']:
                raise ValueError('frozen recipe runtime changed')
            plan = existing
    else:
        if args.out.exists():
            raise ValueError('fresh recipe directory required')
        if args.plan_only:
            plan['code'] = {str(path.relative_to(package)): sha(path) for path in package.rglob('*.py')}
    if args.plan_only:
        print(json.dumps({'event': 'plan_only', 'fresh': not plan_path.exists(), 'handoff': args.handoff,
                          'recipe_sha256': plan['recipe_sha256'], 'stage_inputs': plan.get('stage_inputs'),
                          'stages_started': [stage['id'] for stage in recipe['stages']
                                             if stage_started(args.out, stage['id'])],
                          'handoff_record': ({k: v for k, v in plan['handoffs'][-1].items() if k != 'previous_inputs'}
                                             if args.handoff and plan.get('handoffs') else None)},
                         indent=1), flush=True)
    elif not plan_path.exists():
        shutil.copytree(package, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        plan['code'] = {str(path.relative_to(frozen)): sha(path) for path in frozen.rglob('*.py')}
        write_json(plan_path, plan)
    if args.launch_metadata is not None and not args.plan_only:
        metadata = args.launch_metadata.resolve()
        if metadata == args.out or args.out in metadata.parents:
            raise ValueError('--launch-metadata must be outside the recipe output directory')
        if metadata.exists():
            raise ValueError('launch metadata path already exists; choose a fresh immutable sidecar')
        write_json(metadata, {'schema': 'natlang.neuralese-recipe-launch-intent/1',
                              'recipe_path': str(args.recipe.resolve()),
                              'recipe_sha256': plan['recipe_sha256'],
                              'output_path': str(args.out),
                              'heads_path': str(args.heads),
                              'heads_sha256': sha(args.heads),
                              'device': args.device,
                              'until': args.until,
                              'recipe_plan_path': str(plan_path),
                              'recipe_plan_sha256': sha(plan_path),
                              'frozen_runtime_sha256': plan.get('code', {}),
                              'stage_inputs': stage_inputs,
                              'runtime_environment': declared_runtime_environment(recipe)})
    init_mix = {}
    if recipe.get('quantization') or recipe.get('init'):
        # The lineage's init decision is recorded in its heads (foundation_heads --init-receipt): check it against
        # the recipe's ``init`` and carry the latents' ramp position into every quantized stage.
        import torch
        from .quantization import init_mix_for
        heads_identity = torch.load(args.heads, map_location='cpu', weights_only=False, mmap=True).get('backbone')
        init_mix = init_mix_for(recipe, heads_identity)
    with _ChildSignalForwarder() as signal_forwarder:
        stopped = signal_forwarder.stopped
        reports = []
        feedback_checkpoint = None
        for stage in recipe['stages']:
            if stopped[0]:
                print(json.dumps({'status': 'interrupted_before_child_start', 'stage': stage['id']}), flush=True)
                return
            directory = args.out / stage['id']
            report_path = args.out / (stage['id'] + '-report.json')
            kind = stage['kind']
            if report_path.exists():
                report = json.loads(report_path.read_text())
                require_gate(report['gate'], kind)
                # A report written before a handoff gave unstarted stages new inputs carries the previous input map.
                lineage_inputs = [inputs] + [record['previous_inputs'] for record in plan.get('handoffs', [])
                                             if 'previous_inputs' in record]
                if report['recipe_sha256'] != plan['recipe_sha256'] or report['inputs'] not in lineage_inputs:
                    raise ValueError('stage report belongs to another recipe/input lineage')
                if sha(report['artifact']) != report['artifact_sha256']:
                    raise ValueError('qualified artifact changed')
            else:
                if stage.get('admitted') is False:
                    # A declared stage whose data or method is held (e.g. the view operator stage pending licence
                    # review): it never runs until its declaration admits it; run earlier stages with --until.
                    raise ValueError('stage is declared but not admitted: ' + stage['id'] +
                                     (' (' + stage['admission'] + ')' if isinstance(stage.get('admission'), str) else ''))
                for dependency in stage['requires']:
                    predecessor = next(report for report in reports if report['id'] == dependency)
                    require_gate(predecessor['gate'], predecessor['kind'])
                output = directory / HANDLERS[kind]['result']
                stage_heads = args.heads
                continuation_checkpoint = None
                if kind == 'core_text_warmup' and 'heads' in stage_inputs[stage['id']]:
                    stage_heads = stage_inputs[stage['id']]['heads']['path']
                if (kind == 'core_text_warmup' and
                        stage['parameters'].get('ar_feedback_fixup',False) and
                        'heads' not in stage_inputs[stage['id']]):
                    continuation_checkpoint, stage_heads = mapped_fixup_continuation(stage,reports)
                    continuation_checkpoint = str(continuation_checkpoint)
                direct_artifact_stage = (kind == 'raw_recurrence_training' and
                                         'direct_artifact_manifest' in stage_inputs[stage['id']])
                if kind in ('core_text_warmup','text_warmup_runtime','raw_recurrence_training') and not direct_artifact_stage:
                    has_exact_stage_heads = 'heads' in stage_inputs[stage['id']]
                    if has_exact_stage_heads:
                        stage_heads = stage_inputs[stage['id']]['heads']['path']
                    elif kind == 'core_text_warmup':
                        if not stage['parameters'].get('ar_feedback_fixup',False):
                            predecessor_kind = 'raw_runtime_qualification'
                            predecessor = next(r for r in reversed(reports)
                                               if r['id'] in stage['requires'] and r['kind'] == predecessor_kind)
                            stage_heads = predecessor['artifact']
                    else:
                        predecessor_kind = 'core_text_warmup'
                        predecessor=next(r for r in reversed(reports) if r['id'] in stage['requires'] and r['kind']==predecessor_kind)
                        stage_heads=predecessor['artifact']
                command = [sys.executable, '-m', HANDLERS[kind]['module']]
                if not direct_artifact_stage:
                    command += ['--heads', hdd_guard.input_path('heads', stage_heads)]
                command += stage_input_args(stage_inputs[stage['id']], kind) + ['--out',
                           str(output if kind == 'token_identity' else directory), '--device', args.device]
                previous_phase = direct_phase_predecessor(recipe, stage)
                if previous_phase is not None:
                    # A later phase of a direct lineage continues from its predecessor phase's final full state and
                    # sidecars (DIRECT_PHASE_CONTINUATION); refuse before the child loads anything if one is absent.
                    predecessor = next(r for r in reports if r['id'] == previous_phase)
                    continuation = direct_phase_continuation(Path(predecessor['artifact']).parent)
                    absent = sorted(value['path'] for value in continuation.values() if not Path(value['path']).is_file())
                    if absent:
                        raise ValueError('continued direct phase ' + stage['id'] + ' lacks predecessor output(s): ' +
                                         ', '.join(absent))
                    command += stage_input_args(continuation, kind)
                if continuation_checkpoint:
                    command += ['--continue-from', continuation_checkpoint]
                if kind == 'raw_runtime_qualification':
                    command += ['--checkpoint', feedback_checkpoint, '--certificate', str(args.out / 'foundation-certificate.json')]
                if kind == 'view_operator_gate':
                    measured = next(r for r in reversed(reports)
                                    if r['id'] in stage['requires'] and r['kind'] == 'raw_recurrence_training')
                    command += ['--checkpoint', measured['artifact']]
                if kind == 'verified_heads_handoff':
                    raw = next(r for r in reversed(reports) if r['id'] in stage['requires'] and r['kind']=='raw_runtime_qualification')
                    command += ['--raw-runtime-report', str(Path(raw['artifact']).parent / 'runtime-report.json')]
                stage_parameters = effective_stage_parameters(stage,reports,recipe)
                if 'text_data' in stage_inputs[stage['id']]:
                    stage_parameters = {key: value for key, value in stage_parameters.items() if key != 'text_data'}
                command += stage_parameter_args({key: value for key, value in stage_parameters.items()
                                                 if key not in RUNNER_STAGE_PARAMETERS})
                component = stage_quantization(recipe, stage, stage_parameters, reports, init_mix)
                if component is not None:
                    command += ['--quantization', json.dumps(component, sort_keys=True, separators=(',', ':'))]
                if kind == 'neuralese_direct':
                    command += stage_objective_args(recipe, stage)
                environment = dict(os.environ)
                declared_environment = declared_runtime_environment(recipe)
                if 'NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE' not in declared_environment:
                    environment.pop('NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE', None)
                environment.update(declared_environment)
                environment['PYTHONPATH'] = str(frozen.parent) + os.pathsep + environment.get('PYTHONPATH', '')
                if args.plan_only:
                    # The next stage's own data plan on the current code (budget, data continuation, QAT range, held
                    # set), from its exact child arguments; a trainer without a plan-only mode is listed only.
                    print(json.dumps({'stage': stage['id'], 'plan_only_command': command}), flush=True)
                    if kind != 'neuralese_direct':
                        return 0
                    environment['PYTHONPATH'] = str(package.parent) + os.pathsep + os.environ.get('PYTHONPATH', '')
                    return subprocess.run(command + ['--plan-only'], env=environment).returncode
                if hdd_guard.RECORDS:
                    write_json(args.out / 'hdd-staged-inputs.json', hdd_guard.RECORDS)
                print(json.dumps({'stage': stage['id'], 'command': command}), flush=True)
                result = signal_forwarder.run(command, env=environment)
                code = result.returncode
                if stopped[0]:
                    print(json.dumps({'status': 'interrupted', 'stage': stage['id'],
                                      'child_exit_code': code}), flush=True)
                    return
                if code:
                    raise RuntimeError('stage failed: ' + stage['id'])
                if kind == 'token_identity':
                    gate = json.loads(output.read_text())
                elif kind in ('raw_runtime_qualification','text_warmup_runtime'):
                    gate = json.loads((directory / ('runtime-report.json' if kind=='raw_runtime_qualification' else 'report.json')).read_text())
                elif kind=='verified_heads_handoff':
                    gate=json.loads((directory/'report.json').read_text())
                elif kind in ('core_text_warmup','neuralese_direct'):
                    gate=json.loads((directory/'report.json').read_text())
                elif kind == 'view_operator_gate':
                    gate = json.loads(output.read_text())
                else:
                    import torch
                    state = torch.load(output, mmap=True, weights_only=False, map_location='cpu')
                    if kind == 'raw_recurrence_training':
                        if state.get('schema') != 'natlang.neuralese_recurrence_checkpoint/1':
                            raise ValueError('wrong recurrence artifact schema')
                        if state['identity']['files'].get(str(Path(stage_heads).resolve())) != sha(stage_heads):
                            raise ValueError('recurrence artifact runtime handoff differs')
                        gate = recurrence_completion_gate(output, stage['parameters'].get('steps', 500))
                    else:
                        gate = state['best']
                        if state['identity']['inputs'].get(str(args.heads)) != sha(args.heads):
                            raise ValueError('bootstrap checkpoint backbone identity differs')
                report = {'id': stage['id'], 'kind': kind, 'recipe_sha256': plan['recipe_sha256'],
                          'inputs': inputs, 'gate': gate, 'artifact': str(output), 'artifact_sha256': sha(output)}
                if kind == 'neuralese_direct' and gate.get('data_continuation') is not None:
                    report['data_continuation'] = gate['data_continuation']  # previous data, seen count, new budget
                write_json(args.out / (stage['id'] + '-attempt.json'), report)
                require_gate(gate, kind)
                write_json(report_path, report)
            reports.append(report)
            if kind == 'causal_embedding_distillation':
                feedback_checkpoint = report['artifact']
            if kind == 'causal_embedding_distillation':
                certificate = {'schema': 'natlang.neuralese-foundation-certificate/1', 'qualified': True,
                               'runtime_qualified': False, 'heads_sha256': sha(args.heads),
                               'feedback_checkpoint': feedback_checkpoint, 'feedback_checkpoint_sha256': sha(feedback_checkpoint),
                               'recipe_sha256': plan['recipe_sha256'],
                               'stages': [{'kind': report['kind'], 'report': str(args.out / (report['id'] + '-report.json')),
                                           'report_sha256': sha(args.out / (report['id'] + '-report.json'))} for report in reports if report['kind'] in {'token_identity', 'causal_embedding_distillation'}]}
                write_json(args.out / 'foundation-certificate.json', certificate)
            if kind == 'neuralese_direct':
                stage_report = json.loads(Path(report['artifact']).read_text())
                identity = stage_report.get('identity_check') or {}
                certificate = {'schema': 'natlang.neuralese-direct-certificate/1',
                               'qualified': bool(stage_report.get('qualified')) and identity.get('passed') is True,
                               'identity_qualification': {'kind': DIRECT_IDENTITY_QUALIFICATION, **identity},
                               'owner_decision': 'Neuralese direct v1, 2026-10-11 (AGENTS.md "Neuralese training '
                                                 'foundation")',
                               'heads_sha256': sha(args.heads), 'recipe_sha256': plan['recipe_sha256'],
                               'stage': {'id': stage['id'], 'report': str(report_path), 'report_sha256': sha(report_path),
                                         'artifact_sha256': report['artifact_sha256']}}
                write_json(args.out / 'direct-certificate.json', certificate)
            if args.until == stage['id']:
                return
    print(json.dumps({'status': 'recipe_completed', 'foundation_runtime_qualified': any(r['kind'] == 'raw_runtime_qualification' for r in reports)}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
