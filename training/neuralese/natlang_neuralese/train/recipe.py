"""Declared, gated training stages with frozen code and explicit artifact handoff.

Stage implementations are shared Python modules. Extend HANDLERS with a typed
handler and declare its dependencies in a JSON recipe. No shell command strings.
Failed mechanisms cannot advance. Native stage receipts bind their source,
optimizer state, held split, and exact actor/runtime identity.
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
    # One recipe-owned rendezvous launches the shared curriculum after both
    # native SFT receipts become ready. Participant state belongs to the
    # controller because readiness arrives asynchronously.
    'native_shared_stage_controller': {'module': 'natlang_neuralese.train.native_readiness',
        'required_inputs': {'controller_config'}, 'optional_inputs': set(),
        'parameters': set(), 'result': 'shared-stage-completion.json'},
    'native_semantic_training': {'module': 'natlang_neuralese.train.native_semantic_stage',
        'required_inputs': {'semantic_config', 'native_participants'}, 'optional_inputs': {'resume'},
        'parameters': set(), 'result': 'checkpoint.pt'},
    # Native positional-KV/hybrid-state training (shared-state migration,
    # plans/neuralese/NATIVE_STATE_SFT_DECOMPOSITION_20261015.md). The input
    # graph files are already admitted and carry explicit split provenance.
    'native_state_sft': {'module': 'natlang_neuralese.train.native_state_sft',
                         'required_inputs': {'native_graphs', 'held_native_graphs'},
                         'optional_inputs': {'continue_from', 'continued_native_graphs', 'continued_held_native_graphs', 'foundation_native_graphs', 'foundation_qualification_request', 'withdrawn_native_graphs', 'pending_native_sources', 'unavailable_native_sources', 'source_classification_migration', 'ce_supervision_policy'},
                         'parameters': {'base', 'rank', 'dtype', 'native_linear_precision', 'native_attention_precision', 'epochs', 'batch_graphs', 'batch_tokens', 'learning_rate', 'weight_decay',
                             'latent_ns', 'eval_every', 'held_eval_tokens', 'held_eval_graphs', 'checkpoint_every', 'checkpoint_seconds', 'async_checkpoint', 'checkpoint_layers', 'checkpoint_direct_token_limit', 'checkpoint_input_offload', 'checkpoint_input_offload_directory', 'chunk_tokens', 'ffn_chunk_tokens',
                             'fp8_training', 'fp8_recipe', 'fp8_amax_history_len', 'learned_row_learning_rate', 'interface_row_learning_rate', 'native_optimizer_migration', 'native_ordinary_precision', 'native_norm_execution', 'native_shared_execution', 'history_reasoning_policy', 'seed', 'task_eval', 'objective_continuation',
                             'task_eval_every', 'forward_atol', 'gradient_atol', 'qualify_training_backend', 'resident_geometry_qualification', 'memory_qualification', 'communication_foundation', 'foundation_communication_batches', 'urgent_cohort_fairness', 'refresh_at_checkpoint', 'refresh_foundation_readiness', 'foundation_readiness_policy', 'refresh_after_updates', 'learn_reasoning_boundaries', 'share_causal_prefixes', 'cotangent_accumulator_backend'}, 'result': 'checkpoint.pt'},

}

# Native offline preference inherits the exact full-state SFT parameter contract.
RL_DEFAULTS_FROM = {'native_offline_preference': 'native_state_sft'}

INPUT_BINDING_NAME = re.compile(r'[a-z][a-z0-9_.-]*')


def _finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)






def refuse_retired_parameters(parameters):
    retired = sorted({'checkpoint_minutes', 'eval_minutes'} & set(parameters if isinstance(parameters, dict) else ()))
    if retired:
        raise ValueError(f'{", ".join(retired)} retired (owner 2026-10-10): declare eval_every and checkpoint_every in '
                         'steps instead (steps and checkpoint_every multiples of eval_every)')


def validate_stage_parameters(kind, parameters):
    """Validate parameter shapes whose CLI actions are not scalar values."""
    if kind=='native_coefficient_rl':
        if 'epochs' in parameters and (type(parameters['epochs']) is not int or parameters['epochs']<1):
            raise ValueError('native coefficient epochs must be a positive integer')
        if parameters.get('fp8_training','off') not in {'off','triton-experts'}:
            raise ValueError('native coefficient FP8 backend must be an actually qualified shared expert backend')
        if 'coefficient_optimizer' in parameters and (not isinstance(parameters['coefficient_optimizer'],dict)
                or parameters['coefficient_optimizer'].get('name') not in {'adam','sgd'}):
            raise ValueError('native coefficient optimizer must declare the shared named Adam/SGD mechanism')
    if kind in {'native_state_sft','native_state_rl','native_offline_preference'}:
        if 'native_norm_execution' in parameters and parameters['native_norm_execution'] not in ('source-v1','stable-v1'):
            raise ValueError('unsupported native norm execution profile')
        if 'native_shared_execution' in parameters:
            from ..model.shared_execution import NATIVE_SHARED_EXECUTIONS
            if parameters['native_shared_execution'] not in NATIVE_SHARED_EXECUTIONS:
                raise ValueError('unsupported native shared execution profile')
        if 'history_reasoning_policy' in parameters and parameters['history_reasoning_policy'] != 'keep':
            raise ValueError('history_reasoning_policy currently supports the explicit keep profile only')
        from ..model.native_parameter_precision import ORDINARY_PRECISION_PROFILES
        if ('native_ordinary_precision' in parameters
                and parameters['native_ordinary_precision'] not in ORDINARY_PRECISION_PROFILES):
            raise ValueError('unsupported native ordinary precision profile')
        if 'native_attention_precision' in parameters:
            from ..model.native_precision import NATIVE_ATTENTION_PRECISIONS
            if parameters['native_attention_precision'] not in NATIVE_ATTENTION_PRECISIONS:
                raise ValueError('unsupported native attention precision')
        from .native_optimizer_migration import validate_native_optimizer_migration
        validate_native_optimizer_migration(parameters.get('native_optimizer_migration'))
        from .native_learning_rates import validate_learned_row_rate
        for name in ('learned_row_learning_rate','interface_row_learning_rate'):
            if name in parameters:
                validate_learned_row_rate(parameters[name],name=name)
    if kind in {'native_state_sft','native_state_rl','native_offline_preference'} and 'checkpoint_input_offload' in parameters:
        if parameters['checkpoint_input_offload'] is not None and type(parameters['checkpoint_input_offload']) is not bool:
            raise ValueError('checkpoint_input_offload must be a boolean or omitted for inheritance')
    if kind in {'native_state_sft','native_state_rl','native_offline_preference'}:
        cotangent_backend = parameters.get('cotangent_accumulator_backend')
        if cotangent_backend is not None and cotangent_backend not in ('cpu_file', 'device_fp32'):
            raise ValueError('cotangent_accumulator_backend must be cpu_file or device_fp32')
        direct=parameters.get('checkpoint_direct_token_limit',0)
        if type(direct) is not int or direct<0:
            raise ValueError('checkpoint_direct_token_limit must be a nonnegative integer')
        ffn=parameters.get('ffn_chunk_tokens')
        if ffn is not None and(type(ffn)is not int or ffn<0):
            raise ValueError('ffn_chunk_tokens must be a nonnegative integer or omitted for inheritance')
        directory = parameters.get('checkpoint_input_offload_directory')
        if directory is not None and not isinstance(directory,str):
            raise ValueError('checkpoint_input_offload_directory must be a string or omitted for inheritance')
    if kind=='native_offline_preference':
        beta=parameters.get('preference_beta');margin=parameters.get('preference_margin')
        if beta is not None and (type(beta) not in (int,float) or not math.isfinite(beta) or beta<=0):
            raise ValueError('offline preference beta must be finite and positive')
        if margin is not None and (type(margin) not in (int,float) or not math.isfinite(margin)):
            raise ValueError('offline preference margin must be finite')
    if kind in {'native_state_sft','native_state_rl','native_offline_preference'} and parameters.get('memory_qualification') is not None:
        declared=parameters['memory_qualification']
        values=declared if isinstance(declared,list) else [declared]
        if not values:
            raise ValueError('memory_qualification requires a nonempty binding list')
        for value in values:
            if (not isinstance(value,dict) or set(value)!={'path','sha256'}
                    or not isinstance(value['path'],str) or not value['path']
                    or not isinstance(value['sha256'],str) or not re.fullmatch(r'[0-9a-f]{64}',value['sha256'])):
                raise ValueError('memory_qualification must be an exact path/sha256 binding')


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
        if kind not in HANDLERS:
            raise ValueError('unknown stage implementation: ' + str(kind))
        allowed = HANDLERS[kind]['required_inputs'] | HANDLERS[kind]['optional_inputs']
        if not set(selected) <= allowed or missing_stage_inputs(kind, selected):
            raise ValueError('stage input roles do not satisfy handler contract: ' + stage['id'])
        if any(not isinstance(name, str) or name not in catalog for name in selected.values()):
            raise ValueError('stage refers to an unknown named input binding')
        referenced.update(selected.values())
    if set(catalog) - referenced:
        raise ValueError('unused named input binding(s): ' + ', '.join(sorted(set(catalog) - referenced)))


def missing_stage_inputs(kind, roles):
    """Required input roles a current native stage lacks."""
    if kind not in HANDLERS:
        raise ValueError('unknown stage implementation: ' + str(kind))
    return HANDLERS[kind]['required_inputs'] - set(roles)


NATIVE_TRAINING_KINDS = frozenset({'native_state_sft', 'native_state_rl','native_offline_preference'})
HANDLERS['native_state_rl'] = {
    'module': 'natlang_neuralese.rl.native_stage',
    'required_inputs': {'rollouts', 'held_native_graphs', 'continue_from'},
    'optional_inputs': {'rl_block_bundle'},
    'parameters': set(HANDLERS['native_state_sft']['parameters']) |
        {'rl_tokenizer', 'rl_mode', 'native_participant'},
    'result': 'checkpoint.pt'}

HANDLERS['native_offline_preference'] = {
    'module':'natlang_neuralese.train.native_preference',
    'required_inputs':{'native_graphs','held_native_graphs','objective_spec'},
    'optional_inputs':{'continue_from','continued_native_graphs','continued_held_native_graphs','pending_native_sources'},
    'parameters':(set(HANDLERS['native_state_sft']['parameters']) -
        {'communication_foundation','foundation_communication_batches','refresh_foundation_readiness'}) |
        {'preference_beta','preference_margin'},
    'result':'checkpoint.pt'}

HANDLERS['native_coefficient_rl'] = {
    'module':'natlang_neuralese.rl.native_stage',
    'required_inputs':{'rollouts','held_native_graphs','continue_from','coefficient_policy'},
    'optional_inputs':{'rl_block_bundle'},
    'parameters':{'rl_tokenizer','rl_mode','epochs','base','native_participant','device','coefficient_optimizer',
                  'task_eval','task_eval_every','held_eval_graphs','held_eval_tokens','chunk_tokens',
                  'checkpoint_layers','fp8_training','fp8_recipe'},
    'result':'checkpoint.pt'}
COEFFICIENT_NATIVE_KINDS=frozenset({'native_coefficient_rl'})
SHARED_NATIVE_KINDS = frozenset({'native_semantic_training', 'native_shared_stage_controller'})
HEADLESS_NATIVE_KINDS = NATIVE_TRAINING_KINDS | SHARED_NATIVE_KINDS | COEFFICIENT_NATIVE_KINDS
NATIVE_IDENTITY_QUALIFICATION = 'native_state_runtime_qualified'


def native_lineage(recipe):
    stages = recipe.get('stages') or []
    return bool(stages) and all(stage.get('kind') in HEADLESS_NATIVE_KINDS for stage in stages)








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
        return _bind_native_source_dependencies(result)
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
    return _bind_native_source_dependencies(result)


def native_source_dependencies(path, *, parents=()):
    """Validate portable immutable graph manifests and enumerate every actual input."""
    path = Path(path).resolve()
    if path.suffix != '.json':
        return []
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != 'natlang.native-graph-sources/1':
        raise ValueError('native graph JSON input must be a bound source manifest')
    if path in parents or not manifest.get('sources'):
        raise ValueError('native graph manifest is empty or cyclic')
    dependencies = {}
    for source in manifest['sources']:
        if source.get('kind') not in {'native-graphs', 'semantic-pairs'}:
            raise ValueError('native graph manifest source kind must be explicit')
        child = Path(source['path'])
        child = child if child.is_absolute() else path.parent / child
        from ..data.native_source_tree import _staged_path
        child = _staged_path(path, child)
        child = child.resolve()
        digest = sha(child)
        if digest != source.get('sha256'):
            raise ValueError('native graph manifest dependency content hash mismatch: ' + str(child))
        dependencies[str(child)] = digest
        if source['kind'] == 'native-graphs':
            for value in native_source_dependencies(child, parents=(*parents, path)):
                dependencies[value['path']] = value['sha256']
    return [{'path': child, 'sha256': digest} for child, digest in dependencies.items()]


def _bind_native_source_dependencies(inputs):
    if 'ce_supervision_policy' in inputs:
        from .native_supervision import ce_policy_dependencies
        reference=inputs['ce_supervision_policy']
        dependencies=[item for item in ce_policy_dependencies(reference)
            if Path(item['path']).resolve()!=Path(reference['path']).resolve()]
        if dependencies:reference['dependencies']=dependencies
    if 'source_classification_migration' in inputs:
        from .native_classification_continuation import classification_proof_dependencies
        inputs['source_classification_migration']['dependencies'] = classification_proof_dependencies(
            inputs['source_classification_migration'])
    if 'pending_native_sources' in inputs:
        from .native_pending_sources import pending_source_dependencies
        inputs['pending_native_sources']['dependencies'] = pending_source_dependencies(inputs['pending_native_sources']['path'])
    if 'withdrawn_native_graphs' in inputs:
        proof=Path(inputs['withdrawn_native_graphs']['path'])
        value=json.loads(proof.read_text())
        if value.get('schema')!='natlang.native-graph-withdrawal/1':
            raise ValueError('native withdrawal input has an unsupported schema')
        evidence=value['evidence'];path=Path(evidence['path'])
        if not path.is_absolute():path=proof.parent/path
        path=path.resolve(strict=True)
        if sha(path)!=evidence['sha256']:
            raise ValueError('native withdrawal evidence content hash mismatch')
        inputs['withdrawn_native_graphs']['dependencies']=[{'path':str(path),'sha256':evidence['sha256']}]
    for role in ('native_graphs', 'held_native_graphs', 'continued_native_graphs', 'continued_held_native_graphs', 'foundation_native_graphs'):
        if role in inputs:
            dependencies = native_source_dependencies(inputs[role]['path'])
            if dependencies:
                inputs[role]['dependencies'] = dependencies
    return inputs


def _binding_files(value):
    return [value, *value.get('dependencies', [])]


def _staged_inputs(resolved):
    """Data inputs resolving to the external HDD are read from a verified NVMe copy; load-once state is read in place
    (common/hdd_guard.py ``input_path``)."""
    from ..common import hdd_guard
    return {role: ({**value, 'path': hdd_guard.input_path(role, value['path'], value.get('sha256'),
                                                         value.get('dependencies'))}
                   if isinstance(value, dict) and 'path' in value else value) for role, value in resolved.items()}


def require_input_decoders(stage_inputs):
    """Fail plan-only before creating a lineage if a declared compressed input cannot be read."""
    compressed = sorted({value['path'] for resolved in stage_inputs.values() for value in resolved.values()
                         if isinstance(value, dict) for value in _binding_files(value)
                         if str(value.get('path', '')).endswith('.zst')})
    if compressed and shutil.which('zstd') is None:
        raise RuntimeError('zstd CLI is required for compressed training inputs; install or mount it before '
                           'plan-only or launch (first input: ' + compressed[0] + ')')


def stage_input_args(resolved, kind):
    """Serialize already-validated file roles to the shared handler CLIs."""
    args = []
    supported = HANDLERS[kind]['required_inputs'] | HANDLERS[kind]['optional_inputs']
    resolved = _staged_inputs(resolved)
    for role, flag in (('semantic_config', '--config'), ('controller_config', '--config'),
                       ('native_participants', '--participants'), ('resume', '--resume')):
        if role in resolved and role in supported:
            args += [flag, resolved[role]['path']]
    for role in ('rollouts', 'native_graphs', 'held_native_graphs', 'continued_native_graphs',
                 'continued_held_native_graphs', 'foundation_native_graphs', 'foundation_qualification_request',
                 'withdrawn_native_graphs', 'pending_native_sources', 'unavailable_native_sources', 'source_classification_migration',
                 'ce_supervision_policy', 'objective_spec', 'rl_block_bundle', 'continue_from'):
        if role in resolved and role in supported:
            args += ['--' + role.replace('_', '-'), resolved[role]['path']]
    return args


def validate_shared_controller_output(config_path, stage_directory):
    """Bind controller-owned durable state to this exact recipe stage output."""
    config = json.loads(Path(config_path).read_text())
    if config.get('schema') != 'natlang.native-shared-stage-controller/1':
        raise ValueError('native shared controller input has the wrong schema')
    if Path(config.get('out', '')).resolve() != Path(stage_directory).resolve():
        raise ValueError('native shared controller output must equal its declared recipe stage directory')
    return config




def effective_stage_parameters(stage, reports, recipe=None):
    """Resolve fixup ``steps`` as new updates above its exact predecessor step."""
    parameter_defaults = (recipe or {}).get('stage_parameter_defaults', {})
    inherited=parameter_defaults.get(RL_DEFAULTS_FROM.get(stage['kind']), {})
    if stage['kind']=='native_offline_preference':
        inherited={key:value for key,value in inherited.items()
                   if key in HANDLERS[stage['kind']]['parameters'] and key!='share_causal_prefixes'}
    parameters = dict(inherited)
    parameters.update(parameter_defaults.get(stage['kind'], {}))
    parameters.update(stage['parameters'])

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


def resolve_recipe_data(path, _chain=(), *, _files=None):
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
    if _files is not None:_files.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
    if 'extends' not in recipe:
        if 'overrides' in recipe:
            raise ValueError('overrides without extends')
        return recipe
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
    parent = resolve_recipe_data(candidates[0], chain, _files=_files)
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


def recipe_file_dependencies(path):
    """Exact files consumed by the existing inherited recipe resolver."""
    files=[]
    resolve_recipe_data(path,_files=files)
    return files


def recipe_identity_sha256(path):
    """The recipe hash recorded in plans, receipts and certificates.

    A plain recipe keeps its file-bytes SHA-256 (receipts of earlier runs stay comparable). A recipe that uses
    ``extends`` is hashed as the canonical JSON of its resolved content (natlang_neuralese.common.hashing).
    """
    if 'extends' in json.loads(Path(path).read_text()):
        return canonical_json_sha256_hex(resolve_recipe_data(path))
    return sha(path)


def native_recipe_binding(path, lineage_sha256):
    """Bind current training policy separately from the preserved run lineage."""
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': recipe_identity_sha256(path),
            'file_sha256': sha(path), 'lineage_sha256': lineage_sha256}


def native_recipe_provenance(value):
    """Exact saved provenance metadata; this does not re-qualify recipe file bytes."""
    if value is None:return None
    if isinstance(value,str):value=json.loads(value)
    if (not isinstance(value,dict) or not isinstance(value.get('path'),str) or not value['path']
            or any(not isinstance(value.get(key),str) or len(value[key])!=64
                   or any(char not in '0123456789abcdef' for char in value[key])
                   for key in ['sha256']+[key for key in ('file_sha256','lineage_sha256') if key in value])):
        raise ValueError('native recipe provenance needs exact current policy and preserved lineage identities')
    return json.loads(json.dumps(value))


def training_recipe_binding(state, *, participant=None):
    """Read one model's own options, never the joint semantic config's identity."""
    if participant is not None:
        entry=state.get('participants',{}).get(participant)
        if entry is None:raise ValueError('recipe provenance requires its actual shared participant')
        options=entry.get('training_options') or {}
    else:
        options=((state.get('training') or state.get('shared_training') or {}).get('options') or {})
    return native_recipe_provenance(options.get('recipe_binding'))


def checkpoint_recipe_binding(path, *, participant):
    from .checkpoint_policy import CheckpointPolicy
    return training_recipe_binding(CheckpointPolicy.load(path,shared_participant=participant),participant=participant)


def load_recipe(path):
    recipe = resolve_recipe_data(path)
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
    declared, complete = set(), set()
    for stage in recipe['stages']:
        name, kind = stage.get('id'), stage.get('kind')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name) or name in declared:
            raise ValueError('stage id must be unique and safe')
        if kind not in HANDLERS:
            raise ValueError('unknown stage implementation: ' + str(kind))
        required = stage.get('requires')
        if not isinstance(required, list) or len(set(required)) != len(required) or not set(required) <= complete:
            raise ValueError('stage dependencies must precede the stage')
        parameters = stage.get('parameters')
        refuse_retired_parameters(parameters)
        if not isinstance(parameters, dict) or not set(parameters) <= HANDLERS[kind]['parameters']:
            raise ValueError('unknown stage parameters')
        validate_stage_parameters(kind, parameters)
        declared.add(name)
        complete.add(name)
    if not native_lineage(recipe):
        raise ValueError('only native positional-KV/hybrid-state recipes are supported')
    if recipe.get('identity_qualification') != NATIVE_IDENTITY_QUALIFICATION:
        raise ValueError('native-state recipe requires identity_qualification ' + NATIVE_IDENTITY_QUALIFICATION)
    retired={'e2e_acceptance','e2e_mix','family','objective_continuation','quantization','init'} & set(recipe)
    if retired:
        raise ValueError('projected recipe fields are retired: '+', '.join(sorted(retired)))
    validate_input_bindings(recipe)
    declared_runtime_environment(recipe)
    return recipe


_TRUSTED_RUNTIME_ENVIRONMENT = {'NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE': '1'}


def python_package_environment(environment, package_parent):
    """Bind module execution to its declared package rather than the caller's cwd."""
    result = dict(environment)
    paths = [str(package_parent)] + [path for path in result.get('PYTHONPATH', '').split(os.pathsep) if path]
    result['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
    result['PYTHONSAFEPATH'] = '1'
    return result


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
            stage.get('kind') in NATIVE_TRAINING_KINDS and
            stage.get('parameters', recipe.get('parameters', {})).get('fp8_training') == 'te-fp8'
            for stage in stages if isinstance(stage, dict))
        if not te_enabled:
            raise ValueError('trusted TE extra-state restore requires a declared native te-fp8 stage')
    return dict(environment)






def require_gate(report, kind, *, training_progress=False):
    """Require the exact completed receipt emitted by a current native stage."""
    if kind in COEFFICIENT_NATIVE_KINDS:
        if (report.get('schema') != 'natlang.native-coefficient-training-report/1'
                or report.get('completed') is not True
                or report.get('qualification_complete') is not True
                or report.get('base_checkpoint_unchanged') is not True):
            raise ValueError('native coefficient policy lacks completed exposure and held/task checks')
        return
    if kind in NATIVE_TRAINING_KINDS:
        if report.get('schema') != 'natlang.neuralese-native-sft-report/1' or report.get('completed') is not True:
            raise ValueError('native-state training has not completed its declared graph exposure')
        if report.get('numerics_ok') is not True:
            raise ValueError('native-state training reported a numerical failure')
        return
    if kind in SHARED_NATIVE_KINDS:
        if (report.get('schema') != 'natlang.native-shared-stage-completion/1'
                or report.get('completed') is not True or report.get('production_ready') is not True):
            raise ValueError('shared native curriculum has not completed actual exposure/runtime/task checks')
        return
    raise ValueError('no native gate adapter for stage: ' + str(kind))


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


# Stage parameters that change how a stage computes, not what it optimizes: a lineage may switch them in place
# (``--handoff``). Each trainer also lists them among its resume-operational options.
OPERATIONAL_STAGE_PARAMETERS = frozenset({'refresh_foundation_readiness', 'foundation_readiness_policy', 'refresh_after_updates', 'eval_every', 'checkpoint_every', 'checkpoint_seconds', 'async_checkpoint', 'sdpa_backend', 'deterministic_training', 'fp8_training', 'checkpoint_layers', 'checkpoint_direct_token_limit', 'task_eval', 'task_eval_every', 'qualify_training_backend', 'memory_qualification', 'refresh_at_checkpoint', 'checkpoint_input_offload', 'checkpoint_input_offload_directory'})

def native_stage_parameter_handoffs(old_recipe, new_recipe, out):
    """Allow only current native trainers' explicit resume-safe parameter changes."""
    tuning = {
              'native_state_sft': ('epochs', 'batch_graphs', 'batch_tokens', 'chunk_tokens', 'learning_rate', 'learned_row_learning_rate', 'interface_row_learning_rate', 'native_optimizer_migration', 'native_ordinary_precision', 'native_norm_execution', 'native_shared_execution', 'history_reasoning_policy', 'weight_decay', 'latent_ns', 'held_eval_tokens', 'held_eval_graphs', 'communication_foundation', 'foundation_communication_batches', 'urgent_cohort_fairness', 'learn_reasoning_boundaries', 'native_linear_precision', 'native_attention_precision', 'share_causal_prefixes', 'fp8_recipe', 'resident_geometry_qualification', 'checkpoint_input_offload', 'checkpoint_input_offload_directory', 'ffn_chunk_tokens', 'memory_qualification', 'cotangent_accumulator_backend', 'refresh_foundation_readiness', 'foundation_readiness_policy', 'refresh_after_updates', 'eval_every', 'checkpoint_every', 'checkpoint_seconds', 'async_checkpoint', 'task_eval', 'task_eval_every', 'qualify_training_backend'),
              'native_offline_preference': ('preference_beta', 'preference_margin', 'epochs', 'batch_graphs', 'batch_tokens', 'chunk_tokens', 'learning_rate', 'learned_row_learning_rate', 'interface_row_learning_rate', 'native_optimizer_migration', 'native_ordinary_precision', 'native_norm_execution', 'native_shared_execution', 'history_reasoning_policy', 'weight_decay', 'latent_ns', 'held_eval_tokens', 'held_eval_graphs', 'communication_foundation', 'foundation_communication_batches', 'urgent_cohort_fairness', 'learn_reasoning_boundaries', 'native_linear_precision', 'native_attention_precision', 'share_causal_prefixes', 'fp8_recipe', 'resident_geometry_qualification', 'checkpoint_input_offload', 'checkpoint_input_offload_directory', 'ffn_chunk_tokens', 'memory_qualification', 'cotangent_accumulator_backend'),
              'native_state_rl': ('epochs', 'batch_graphs', 'batch_tokens', 'learning_rate', 'learned_row_learning_rate', 'interface_row_learning_rate', 'native_optimizer_migration', 'native_ordinary_precision', 'native_norm_execution', 'native_shared_execution', 'history_reasoning_policy', 'weight_decay', 'latent_ns', 'held_eval_tokens', 'held_eval_graphs', 'native_linear_precision', 'native_attention_precision', 'checkpoint_input_offload', 'checkpoint_input_offload_directory', 'ffn_chunk_tokens', 'memory_qualification', 'cotangent_accumulator_backend', 'refresh_foundation_readiness', 'foundation_readiness_policy', 'refresh_after_updates', 'eval_every', 'checkpoint_every', 'checkpoint_seconds', 'async_checkpoint', 'task_eval', 'task_eval_every', 'qualify_training_backend')}
    def parameters(recipe, stage):
        merged = dict(recipe.get('stage_parameter_defaults', {}).get(stage.get('kind'), {}))
        merged.update(stage.get('parameters', {}))
        return merged
    old_stages = {stage.get('id'): stage for stage in old_recipe.get('stages', [])}
    changes = {}
    for stage in new_recipe.get('stages', []):
        old = old_stages.get(stage.get('id'))
        kind = stage.get('kind')
        if kind not in tuning or old is None or old.get('kind') != kind:
            continue
        before, after = parameters(old_recipe, old), parameters(new_recipe, stage)
        if before.get('native_ordinary_precision') != after.get('native_ordinary_precision'):
            from ..model.native_parameter_precision import ORDINARY_PRECISION_PROFILES
            if (before.get('native_ordinary_precision') is not None
                    or after.get('native_ordinary_precision') not in ORDINARY_PRECISION_PROFILES
                    or not after.get('memory_qualification')
                    or after.get('memory_qualification') == before.get('memory_qualification')):
                raise ValueError('handoff refused: norm storage upgrade requires its explicit FP32 ordinary-precision profile and fresh restored-model memory qualification')
        old_cotangent_backend = before.get('cotangent_accumulator_backend') or 'cpu_file'
        new_cotangent_backend = after.get('cotangent_accumulator_backend') or old_cotangent_backend
        if old_cotangent_backend != new_cotangent_backend:
            if (new_cotangent_backend not in ('cpu_file', 'device_fp32')
                    or not after.get('memory_qualification')
                    or after.get('memory_qualification') == before.get('memory_qualification')):
                raise ValueError('handoff refused: cotangent accumulator changes require a fresh exact-parent memory qualification')
        if kind in {'native_state_sft','native_offline_preference'} and before.get('fp8_recipe', 'delayed') != after.get('fp8_recipe', 'delayed'):
            from .fp8_backend import expert_precision_transition
            def backend_identity(values):
                return {'kind': values.get('fp8_training', 'off'),
                        'recipe': values.get('fp8_recipe', 'delayed'),
                        'amax_history_len': values.get('fp8_amax_history_len', 16)}
            transition = expert_precision_transition(backend_identity(before), backend_identity(after))
            if (transition is None or after.get('qualify_training_backend') is not True
                    or any(after.get(key) is None for key in ('forward_atol', 'gradient_atol'))):
                raise ValueError('handoff refused: expert recipe changes require an exact stateless transition '
                                 'and restored-model backend qualification with declared tolerances')
        changed = {key: [before.get(key), after.get(key)] for key in tuning[kind]
                   if key != 'qualify_training_backend' and before.get(key) != after.get(key)}
        if changed:
            if stage_finished(out, stage['id']):
                raise ValueError('handoff refused: finished native stage ' + stage['id'] + ' keeps its training contract')
            changes[stage['id']] = changed
    if not changes:
        return changes, old_recipe, new_recipe
    stripped = []
    for recipe in (old_recipe, new_recipe):
        value = json.loads(json.dumps(recipe))
        for kind, keys in tuning.items():
            defaults = value.get('stage_parameter_defaults', {}).get(kind)
            for container in [defaults] + [stage.get('parameters') for stage in value.get('stages', [])
                                          if stage.get('kind') == kind]:
                if isinstance(container, dict):
                    for key in keys:
                        container.pop(key, None)
        stripped.append(value)
    return changes, stripped[0], stripped[1]










def _without_operational(recipe):
    """The resolved recipe minus its name, description, ancestry and operational stage parameters."""
    value = json.loads(json.dumps(recipe))
    # ``backbone`` names the base weights of a FRESH lineage (owner 2026-10-12); a handoff keeps its checkpoint.
    for key in ('id', 'description', 'extends', 'backbone'):
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




def _data_continuation(old, new):
    """Whether changed train/held graph manifests retain their exact predecessor custody."""
    if 'native_graphs' in old and 'native_graphs' in new:
        foundation = 'foundation_native_graphs'
        if old.get(foundation) is not None and old[foundation] != new.get(foundation):
            return False  # New data must not move an already declared finish line.
        if new.get(foundation) is not None and not all(
                new[foundation].get(key) for key in ('path', 'sha256', 'binding')):
            return False
        roles = ('native_graphs', 'held_native_graphs')
        changed = False
        for role in roles:
            predecessor = 'continued_' + role
            if old.get(role) != new.get(role):
                changed = True
                if (not old.get(role) or not new.get(role)
                        or new.get(predecessor, {}).get('sha256') != old[role].get('sha256')):
                    return False
            elif predecessor in new and new[predecessor].get('sha256') != old[role].get('sha256'):
                return False
        withdrawal='withdrawn_native_graphs'
        if new.get(withdrawal) is not None and not all(new[withdrawal].get(key) for key in ('path','sha256','binding')):
            return False
        if old.get(withdrawal) is not None and new.get(withdrawal) is None:
            return False
        classification='source_classification_migration'
        classification_changed=old.get(classification)!=new.get(classification)
        if classification_changed and not _classification_proof_extension(old,new):
            return False
        excluded = set(roles) | {'continued_' + role for role in roles} | {foundation,withdrawal}
        if classification_changed:excluded.add(classification)
        return changed and {k: v for k, v in old.items() if k not in excluded} == {
            k: v for k, v in new.items() if k not in excluded}
    return False


def _priority_only_native_source_continuation(old, new):
    """Permit a drained scheduling-only revision of an existing train manifest.

    The source files and their order remain byte-bound. Only explicit top-level
    source priority may rise, and the old train manifest is retained as the
    declared predecessor. The trainer independently verifies every row ID,
    row binding and order against the checkpoint before continuing.
    """
    role = 'native_graphs'
    old_ref, new_ref = old.get(role), new.get(role)
    if (not isinstance(old_ref, dict) or not isinstance(new_ref, dict)
            or new_ref.get('sha256') == old_ref.get('sha256')
            or new.get('continued_native_graphs', {}).get('sha256') != old_ref.get('sha256')
            or old.get('held_native_graphs') != new.get('held_native_graphs')):
        return False
    def manifest(reference):
        if not all(reference.get(key) for key in ('path', 'sha256', 'binding')):
            return None
        path = Path(reference['path']).resolve(strict=True)
        if path.suffix.lower() != '.json':
            return None
        if sha(path) != reference['sha256']:
            raise ValueError('native scheduling continuation manifest changed')
        value = json.loads(path.read_text())
        if value.get('schema') != 'natlang.native-graph-sources/1' or not isinstance(value.get('sources'), list):
            return None
        return path, value
    def compare(old_path, old_manifest, new_path, new_manifest, ancestors=frozenset()):
        if (not isinstance(old_manifest, dict) or not isinstance(new_manifest, dict)
                or set(old_manifest) != set(new_manifest)
                or old_manifest.get('schema') != 'natlang.native-graph-sources/1'
                or new_manifest.get('schema') != 'natlang.native-graph-sources/1'):
            return False, False
        pair=(old_path.resolve(),new_path.resolve())
        if pair in ancestors:
            return False, False
        ancestors=ancestors|{pair}
        prior_rows, next_rows = old_manifest.get('sources'), new_manifest.get('sources')
        if not isinstance(prior_rows, list) or not isinstance(next_rows, list) or len(prior_rows) != len(next_rows):
            return False, False
        # All manifest metadata except the ordered source declarations is exact.
        if {key:value for key,value in old_manifest.items() if key != 'sources'} != {
                key:value for key,value in new_manifest.items() if key != 'sources'}:
            return False, False
        any_changed = False
        for prior, current in zip(prior_rows, next_rows):
            if not isinstance(prior, dict) or not isinstance(current, dict):
                return False, False
            prior_priority, current_priority = prior.get('training_priority', 0), current.get('training_priority', 0)
            if (type(prior_priority) is not int or prior_priority < 0 or type(current_priority) is not int
                    or current_priority < prior_priority):
                return False, False
            priority_changed = current_priority != prior_priority
            if priority_changed:
                any_changed = True
            old_child_path = Path(prior['path'])
            if not old_child_path.is_absolute(): old_child_path = old_path.parent / old_child_path
            new_child_path = Path(current['path'])
            if not new_child_path.is_absolute(): new_child_path = new_path.parent / new_child_path
            identity_fields = {key:value for key,value in prior.items()
                               if key not in {'training_priority', 'path', 'sha256'}}
            current_fields = {key:value for key,value in current.items()
                              if key not in {'training_priority', 'path', 'sha256'}}
            if identity_fields != current_fields:
                return False, False
            if (old_child_path.resolve() == new_child_path.resolve()
                    and prior.get('sha256') == current.get('sha256')):
                continue
            # A changed child binding is allowed only when it is itself a
            # source manifest whose sole semantic delta is a priority increase.
            old_child, new_child = manifest({'path':str(old_child_path), 'sha256':prior.get('sha256'), 'binding':'nested'}), \
                manifest({'path':str(new_child_path), 'sha256':current.get('sha256'), 'binding':'nested'})
            if old_child is None or new_child is None:
                return False, False
            valid, child_changed = compare(old_child[0], old_child[1], new_child[0], new_child[1], ancestors)
            if not valid or not child_changed:
                return False, False
            any_changed = True
        return True, any_changed
    before, after = manifest(old_ref), manifest(new_ref)
    if not before or not after:
        return False
    valid, changed = compare(before[0], before[1], after[0], after[1])
    if not valid or not changed:
        return False
    def effective_leaves(path, value, ancestry=frozenset()):
        path=path.resolve()
        if path in ancestry:
            return None
        ancestry=ancestry|{path}
        result={}
        for source in value['sources']:
            source_path=Path(source['path'])
            if not source_path.is_absolute(): source_path=path.parent/source_path
            child=manifest({'path':str(source_path),'sha256':source.get('sha256'),'binding':'nested'})
            if child is not None:
                leaves=effective_leaves(child[0],child[1],ancestry)
                if leaves is None:
                    return None
                override=source.get('training_priority')
                for key,priority in leaves.items():
                    result[key]=override if override is not None else priority
            else:
                identity=(str(source_path.resolve()),source.get('sha256'),source.get('kind'),source.get('model'))
                result[identity]=source.get('training_priority',0)
        return result
    old_priorities=effective_leaves(before[0],before[1])
    new_priorities=effective_leaves(after[0],after[1])
    if old_priorities is None or new_priorities is None or old_priorities.keys()!=new_priorities.keys():
        return False
    if (any(type(priority) is not int or priority < 0 for priority in (*old_priorities.values(),*new_priorities.values()))
            or any(new_priorities[key] < priority for key,priority in old_priorities.items())
            or not any(new_priorities[key] > priority for key,priority in old_priorities.items())):
        return False
    return {key: value for key, value in old.items() if key not in {'native_graphs', 'continued_native_graphs'}} == {
        key: value for key, value in new.items() if key not in {'native_graphs', 'continued_native_graphs'}}


def _pending_source_continuation(old, new):
    """A drained recipe may add mandatory work without changing active sources.

    Kept separate from _data_continuation: resident polling does not load new
    pending declarations, so accepting this there would lose obligations.
    """
    role = 'pending_native_sources'
    if ('native_graphs' not in old or 'native_graphs' not in new
            or role not in new or old.get(role) == new[role]
            or {k:v for k,v in old.items() if k != role} !=
               {k:v for k,v in new.items() if k != role}):
        return False
    from .native_pending_sources import load_pending_sources
    def load(reference):
        if not all(reference.get(key) for key in ('path','sha256','binding')):
            raise ValueError('pending continuation needs an exact declared binding')
        path = Path(reference['path']).resolve(strict=True)
        if sha(path) != reference['sha256']:
            raise ValueError('pending continuation declaration changed')
        value = json.loads(path.read_text())
        return load_pending_sources(path, tokenizer_fingerprint=value['tokenizer_fingerprint']), value['tokenizer_fingerprint']
    added, fingerprint = load(new[role])
    if role in old:
        prior, old_fingerprint = load(old[role])
        if old_fingerprint != fingerprint or any(added['rows'].get(key) != row
                for key,row in prior['rows'].items()):
            return False
        dependencies = {(item['path'],item['sha256']) for item in added['dependencies']}
        if any((item['path'],item['sha256']) not in dependencies for item in prior['dependencies']):
            return False
    return True



def _ce_supervision_input_continuation(old, new):
    """Bind explicit CE algebra while preserving the source continuation guards."""
    role = 'ce_supervision_policy'
    if 'native_graphs' not in old or role not in new or old.get(role) == new[role]:
        return False
    before = {key:value for key,value in old.items() if key != role}
    after = {key:value for key,value in new.items() if key != role}
    if before != after and not (_data_continuation(before, after)
            or _priority_only_native_source_continuation(before, after)
            or _pending_source_continuation(before, after)
            or _combined_native_source_continuation(before, after)
            or _classification_input_continuation(before, after)):
        return False
    from .native_supervision import load_policy
    reference = new[role]
    if not all(reference.get(key) for key in ('path', 'sha256', 'binding')):
        raise ValueError('CE supervision continuation requires its exact declared input binding')
    if load_policy(reference).get('require_eligible_sources', True) is not True:
        raise ValueError('production CE supervision continuation requires eligible-source coverage')
    return True


def _classification_input_continuation(old, new):
    """Bind a metadata-only classification proof while preserving every data input.

    This only admits the proof input. The trainer independently validates all
    cache rows against its real loaded source before reattributing existing visits.
    """
    role = 'source_classification_migration'
    if ('native_graphs' not in old or role not in new or old.get(role) == new[role]
            or {k:v for k,v in old.items() if k != role} !=
               {k:v for k,v in new.items() if k != role}):
        return False
    _classification_proof(new)
    return True


def _classification_reader_relocation(old, new):
    """Compare the same historical proof after byte-identical reader relocation."""
    role = 'source_classification_migration'
    if (role not in old or role not in new
            or {k:v for k,v in old.items() if k != role} !=
               {k:v for k,v in new.items() if k != role}
            or {k:v for k,v in old[role].items() if k != 'dependencies'} !=
               {k:v for k,v in new[role].items() if k != 'dependencies'}):
        return False
    from . import native_classification_continuation as continuation
    from scripts.neuralese_data import continue_native_source_indexes as adapter
    from .native_source_classification import _binding, _checked
    readers = {_binding(module.__file__)['sha256']: Path(module.__file__).name
               for module in (continuation, adapter)}
    def retained(reference):
        result = {}
        for value in reference.get('dependencies', []):
            path = _checked(value)
            if readers.get(value['sha256']) == path.name:
                continue
            result[str(path)] = value
        return result
    if retained(old[role]) != retained(new[role]):
        return False
    _classification_proof(new)
    return True


def same_replayed_stage_inputs(recorded, replayed):
    """Compare historical inputs using the existing checked reader relocation."""
    return recorded.keys() == replayed.keys() and all(
        recorded[stage] == replayed[stage]
        or _classification_reader_relocation(recorded[stage], replayed[stage])
        for stage in recorded)


def same_replayed_recipe_plan(recorded, replayed):
    """Preserve exact plans across authenticated byte-identical reader moves."""
    if recorded == replayed:
        return True
    if ({k: v for k, v in recorded.items() if k not in ('stage_inputs', 'inputs')}
            != {k: v for k, v in replayed.items() if k not in ('stage_inputs', 'inputs')}):
        return False
    old, new = recorded['stage_inputs'], replayed['stage_inputs']
    if not same_replayed_stage_inputs(old, new):
        return False
    # The flat inventory must still be precisely derived from each plan's real
    # stage inputs. No other missing, added or changed dependency is excused.
    for plan in (recorded, replayed):
        inputs = {item['path']: item['sha256'] for roles in plan['stage_inputs'].values()
                  for value in roles.values() for item in _binding_files(value)}
        if plan['inputs'] != inputs:
            return False
    return True


def _classification_proof(inputs):
    """Authenticate the existing metadata proof independently of input changes."""
    from .native_source_classification import SCHEMA, _checked, _binding, POLICY_FILES
    from . import native_source_classification as producer
    reference = inputs['source_classification_migration']
    if not all(reference.get(key) for key in ('path', 'sha256', 'binding')):
        raise ValueError('classification continuation needs an exact declared binding')
    proof = json.loads(_checked(reference).read_text())
    if (proof.get('schema') != SCHEMA
            or proof.get('train_sha256') != inputs['native_graphs']['sha256']
            or not proof.get('indexes') or not proof.get('held_indexes')
            or proof.get('optimizer_updates') != 0 or proof.get('student_pointer_writes') != 0
            or proof.get('model_allocations') != 0 or proof.get('graph_position_expansions') != 0):
        raise ValueError('classification continuation proof differs from current source or metadata-only work')
    if proof.get('held', {}).get('sha256') != inputs.get('held_native_graphs', {}).get('sha256'):
        raise ValueError('classification continuation held source differs')
    current_producer = _binding(producer.__file__)
    for part in proof['indexes'] + proof['held_indexes']:
        historical_producer = part.get('producer') or {}
        if (any(historical_producer.get(key) != current_producer[key] for key in ('sha256', 'bytes'))
                or set(part.get('old_policy', {})) != set(POLICY_FILES)):
            raise ValueError('classification continuation producer/policy differs')
    # Authenticate each exact part artifact once within this call, retaining
    # the existing classifier's binding checks and a final stamp check for all
    # reused files. Hundreds of parts commonly share the same policy files.
    # Objective/predecessor continuation has its own independent verifier; this
    # check preserves the original current-proof scope instead of adding work.
    from .native_classification_continuation import classification_proof_dependencies
    try:
        classification_proof_dependencies(reference,follow_continuations=False)
    except OSError as error:
        raise ValueError('classification continuation artifact unavailable') from error
    return proof


def _classification_proof_extension(old,new):
    """Append new proof parts with data; retain every admitted prior part."""
    role='source_classification_migration'
    if (not isinstance(old.get(role),dict) or not isinstance(new.get(role),dict)
            or old[role]==new[role] or old.get('held_native_graphs')!=new.get('held_native_graphs')
            or old.get('native_graphs',{}).get('sha256')==new.get('native_graphs',{}).get('sha256')
            or new.get('continued_native_graphs',{}).get('sha256')!=old.get('native_graphs',{}).get('sha256')):
        return False
    prior,current=_classification_proof(old),_classification_proof(new)
    count=len(prior['indexes'])
    if (len(current['indexes'])>count and current['indexes'][:count]==prior['indexes']
            and current['held_indexes']==prior['held_indexes'] and current['held']==prior['held']):
        return True
    from .native_classification_continuation import verify_objective_proof_extension
    return verify_objective_proof_extension(old, new, prior, current)


def _combined_native_source_continuation(old, new):
    """Compose exact active and pending continuations at a drained handoff only.

    Resident polling deliberately keeps using _data_continuation, since its
    transaction cannot independently load new pending obligations.
    """
    role = 'pending_native_sources'
    if role not in new or old.get(role) == new[role]:
        return False
    pending_only = {**old, role: new[role]}
    return (_data_continuation(pending_only, new)
            and _pending_source_continuation(old, pending_only))


def appended_training_stages(old_recipe, new_recipe, out, *, started_stage_ids=()):
    """Future dependent stages may extend a running recipe without rewriting its existing stages."""
    old = old_recipe.get('stages', [])
    new = new_recipe.get('stages', [])
    old_ids = [stage['id'] for stage in old]
    new_ids = [stage['id'] for stage in new]
    if new_ids[:len(old_ids)] != old_ids or len(set(new_ids)) != len(new_ids):
        raise ValueError('handoff refused: existing stage order/identity must stay; only append future stages')
    added = {}
    available = set(old_ids)
    for stage in new[len(old):]:
        sid = stage['id']
        if sid in started_stage_ids or stage_started(out, sid):
            raise ValueError('handoff refused: appended stage already has output: ' + sid)
        dependencies = stage.get('requires') or []
        if not dependencies or not set(dependencies) <= available:
            raise ValueError('handoff refused: appended stage must depend on existing or earlier stages: ' + sid)
        added[sid] = stage
        available.add(sid)
    return added


def activated_existing_training_stages(old_recipe, new_recipe, out, *, started_stage_ids=()):
    """Existing declared stages may be admitted only before they have produced any output."""
    old = {stage['id']: stage for stage in old_recipe.get('stages', [])}
    activated = {}
    for stage in new_recipe.get('stages', []):
        sid = stage['id']
        previous = old.get(sid)
        if previous is None or previous.get('admitted') is not False or stage.get('admitted') is False:
            continue
        if stage.get('admitted') is not True:
            raise ValueError('handoff refused: admission activation must explicitly set admitted:true for ' + sid)
        if sid in started_stage_ids or stage_started(out, sid):
            raise ValueError('handoff refused: cannot activate already-started stage ' + sid)
        if not stage.get('inputs'):
            raise ValueError('handoff refused: activated stage has no declared inputs: ' + sid)
        activated[sid] = {'kind': 'admitted_stage'}
    return activated


def unstarted_stage_input_changes(existing, plan, out, *, started_stage_ids=()):
    """Track new stage inputs, allowing started native stages only verified source continuations."""
    old_inputs, new_inputs = existing.get('stage_inputs'), plan.get('stage_inputs')
    appended = appended_training_stages(existing['recipe'], plan['recipe'], out, started_stage_ids=started_stage_ids)
    activated = activated_existing_training_stages(existing['recipe'], plan['recipe'], out, started_stage_ids=started_stage_ids)
    if old_inputs == new_inputs and not appended and not activated:
        return {}
    if (not isinstance(old_inputs, dict) or not isinstance(new_inputs, dict)
            or not set(old_inputs) <= set(new_inputs)
            or set(new_inputs) - set(old_inputs) != set(appended)):
        raise ValueError('handoff refused: stage_inputs changed outside appended future stages')
    changes = {}
    for stage_id in sorted(new_inputs):
        old, new = old_inputs.get(stage_id, {}), new_inputs[stage_id]
        if stage_id in appended:
            changes[stage_id] = {'kind': 'appended_stage', 'old': {}, 'new': new}
            continue
        if stage_id in activated:
            declared = next(s for s in plan['recipe']['stages'] if s['id'] == stage_id)
            if not new or set(new) != set(declared['inputs']):
                raise ValueError('handoff refused: activated stage ' + stage_id +
                                 ' must resolve every declared input binding')
            for role, binding in declared['inputs'].items():
                value = new[role]
                if value.get('binding') != binding:
                    raise ValueError('handoff refused: activated stage ' + stage_id +
                                     ' input binding does not match its recipe')
                spec = plan['recipe'].get('input_bindings', {}).get(binding)
                if not isinstance(spec, dict):
                    raise ValueError('handoff refused: activated stage ' + stage_id +
                                     ' input is not backed by its declared binding')
                if spec.get('artifact'):
                    if value.get('artifact') != spec['artifact']:
                        raise ValueError('handoff refused: activated stage ' + stage_id +
                                         ' artifact does not match its declared binding')
                elif value.get('sha256') != spec.get('sha256'):
                    raise ValueError('handoff refused: activated stage ' + stage_id +
                                     ' input hash does not match its declared binding')
                if plan.get('inputs', {}).get(value.get('path')) != value.get('sha256'):
                    raise ValueError('handoff refused: activated stage ' + stage_id +
                                     ' input is absent from the plan input identity map')
            changes[stage_id] = {'kind': 'admitted_stage', 'old': old, 'new': new}
            continue
        if old == new:
            continue
        if stage_id in started_stage_ids or stage_started(out, stage_id):
            if stage_finished(out, stage_id) or not (
                    _priority_only_native_source_continuation(old, new) or _data_continuation(old, new)
                    or _pending_source_continuation(old, new)
                    or _combined_native_source_continuation(old, new)
                    or (next(s for s in plan['recipe']['stages'] if s['id'] == stage_id)['kind'] == 'native_state_sft'
                        and (_classification_input_continuation(old, new)
                             or _ce_supervision_input_continuation(old, new)))):
                raise ValueError('handoff refused: inputs of started stage ' + stage_id + ' changed; only a stage '
                                 'that has not started may take new inputs, and the active stage only by an explicit metadata classification or declared '
                                 'data continuation that retains the exact authenticated native predecessor')
            changes[stage_id] = {'kind': ('ce_supervision_continuation'
                if _ce_supervision_input_continuation(old, new) else 'metadata_classification_continuation'
                if _classification_input_continuation(old, new) else 'scheduling_priority_continuation'
                if _priority_only_native_source_continuation(old, new) else 'data_continuation'),
                'old': old, 'new': new}
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
            if changes.get(stage['id'], {}).get('kind') in {'data_continuation', 'scheduling_priority_continuation', 'metadata_classification_continuation', 'ce_supervision_continuation'}:
                stage.pop('inputs', None)  # exact old inputs are preserved by the continuation predecessor binding
    return old_value, new_value


def _without_activated_stage_policy(old_recipe, new_recipe, activated):
    """Ignore only admission and future objective membership for newly admitted unstarted stages."""
    if not activated:
        return old_recipe, new_recipe
    values = [json.loads(json.dumps(value)) for value in (old_recipe, new_recipe)]
    for value in values:
        for stage in value.get('stages', []):
            if stage['id'] in activated:
                stage['admitted'] = False
    return values


def _inputs_without(inputs, stage_inputs, changes, side):
    """The plan's input identity map minus files read only by the changed (unstarted) stages."""
    kept = {value['path'] for stage_id, roles in stage_inputs.items() if stage_id not in changes
            for value in roles.values() for value in _binding_files(value)}
    dropped = {value['path'] for change in changes.values() for value in change[side].values()
               for value in _binding_files(value)} - kept
    return {path: digest for path, digest in inputs.items() if path not in dropped}


def lineage_handoff(existing, plan, frozen, package, plan_path, out=None, dry_run=False, *, started_stage_ids=()):
    """Resume a lineage on the newest code and an operationally changed recipe (owner: jobs switch to the newest code
    at once). Device and the inputs of every started stage must be unchanged and the recipes equal apart from
    operational stage parameters, so the lineage's identity (``recipe_sha256``, which its stage reports carry) stays;
    the frozen runtime is replaced, the superseded one kept beside it, and the change appended to the plan's
    ``handoffs``. Each trainer then checks its own resume identity and logs its code handoff. A handoff may also
    append future native stages and use declared hash-bound native source continuations. Started and finished inputs
    remain bound to their recorded evidence.
    ``dry_run`` validates and returns the updated plan without touching the run directory."""
    out = Path(out if out is not None else plan_path.parent)
    if (not isinstance(started_stage_ids,(tuple,list)) or len(set(started_stage_ids))!=len(started_stage_ids)
            or not set(started_stage_ids)<={stage['id']for stage in existing['recipe']['stages']}):
        raise ValueError('handoff refused: authenticated started stages differ from actual old plan')
    if existing.get('device') != plan.get('device'):
        raise ValueError('handoff refused: device changed; use a new stage lineage')
    appended = appended_training_stages(existing['recipe'], plan['recipe'], out, started_stage_ids=started_stage_ids)
    activated = activated_existing_training_stages(existing['recipe'], plan['recipe'], out, started_stage_ids=started_stage_ids)
    input_changes = unstarted_stage_input_changes(existing, plan, out, started_stage_ids=started_stage_ids)
    if (_inputs_without(existing.get('inputs'), existing.get('stage_inputs') or {}, input_changes, 'old') !=
            _inputs_without(plan.get('inputs'), plan.get('stage_inputs') or {}, input_changes, 'new')):
        raise ValueError('handoff refused: inputs changed; use a new stage lineage')
    old_handoff_recipe = existing['recipe']
    new_handoff_recipe = plan['recipe']
    old_handoff_recipe, new_handoff_recipe = _without_unstarted_bindings(
        old_handoff_recipe, new_handoff_recipe, input_changes)
    old_handoff_recipe, new_handoff_recipe = _without_activated_stage_policy(
        old_handoff_recipe, new_handoff_recipe, activated)
    if appended:
        compared = []
        for value in (old_handoff_recipe, new_handoff_recipe):
            value = json.loads(json.dumps(value))
            value['stages'] = [stage for stage in value['stages'] if stage['id'] not in appended]
            used_bindings = {binding for stage in value['stages']
                             for binding in (stage.get('inputs') or {}).values()}
            if isinstance(value.get('input_bindings'), dict):
                value['input_bindings'] = {name: binding for name, binding in value['input_bindings'].items()
                                           if name in used_bindings}
                if not value['input_bindings']:
                    value.pop('input_bindings')
            compared.append(value)
        old_handoff_recipe, new_handoff_recipe = compared
        old_kinds = {stage['kind'] for stage in old_handoff_recipe.get('stages', [])}
        added_kinds = {stage['kind'] for stage in appended.values()} - old_kinds
        for kind in added_kinds:
            new_handoff_recipe.get('stage_parameter_defaults', {}).pop(kind, None)
    if activated:
        record_activated = sorted(activated)
    else:
        record_activated = []
    native_parameter_changes, old_handoff_recipe, new_handoff_recipe = native_stage_parameter_handoffs(
        old_handoff_recipe, new_handoff_recipe, out)
    old_contract = _without_package_code(_without_operational(old_handoff_recipe))
    new_contract = _without_package_code(_without_operational(new_handoff_recipe))
    if old_contract != new_contract:
        raise ValueError('handoff refused: recipe changes more than operational native stage parameters, input continuation, or future-stage append')
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
    code_changed = old_code != actual_new if old_pin is not None or new_pin is not None else True
    previous = (frozen.parent.parent / ('runtime-superseded-' + time.strftime('%Y%m%dT%H%M%S'))
                if code_changed else None)
    if dry_run:
        new_code = {str(path.relative_to(package)): sha(path) for path in package.rglob('*.py')}
    elif code_changed:
        frozen.parent.rename(previous)
        shutil.copytree(package, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        new_code = {str(path.relative_to(frozen)): sha(path) for path in frozen.rglob('*.py')}
    else:
        new_code = old_code
    record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S%z'), 'recipe_id': plan['recipe'].get('id'),
              'recipe_sha256': plan['recipe_sha256'], 'superseded_runtime': previous.name if previous else None,
              'code_changed': sorted(k for k in old_code.keys() & new_code.keys() if old_code[k] != new_code[k]),
              'code_added': sorted(new_code.keys() - old_code.keys()),
              'code_removed': sorted(old_code.keys() - new_code.keys())}
    if started_stage_ids:
        record['authenticated_started_stages'] = list(started_stage_ids)
    if appended:
        record['appended_stages'] = appended
    if record_activated:
        record['activated_stages'] = record_activated
    if native_parameter_changes:
        record['native_parameter_changes'] = native_parameter_changes
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
    The same object is used by the native multi-stage recipe runner.
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
            from .loop import wait_forwarding_signals
            return subprocess.CompletedProcess(command, wait_forwarding_signals(child))
        finally:
            self.detach(child)


def stage_parameter_args(parameters):
    """Serialize a validated native stage's typed options into CLI arguments."""
    command = []
    for key, value in parameters.items():
        if key == 'memory_qualification':
            if value is not None:
                command += ['--memory-qualification', json.dumps(value, sort_keys=True, separators=(',', ':'))]
            continue
        if key == 'cotangent_accumulator_backend' and value is None:
            continue
        if isinstance(value, bool):
            command += ['--' + ('' if value else 'no-') + key.replace('_', '-')]
        elif isinstance(value, (dict, list)):
            command += ['--' + key.replace('_', '-'), json.dumps(value, sort_keys=True, separators=(',', ':'))]
        else:
            command += ['--' + key.replace('_', '-'), str(value)]
    return command


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
                             'recipe that differs only in operational stage parameters; recorded in recipe-plan.json "handoffs"')
    parser.add_argument('--plan-only', action='store_true',
                        help='resolve and hash every stage input and validate the resume (or --handoff) against the '
                             'existing recipe-plan.json, print the resulting plan, and exit without writing or launching')
    parser.add_argument('--recover-native-checkpoint',type=Path,
                        help='install a missing native recipe plan from its exact full checkpoint and saved recipe authority')
    parser.add_argument('--recover-native-checkpoint-sha256',
                        help='independently verified full parent SHA required for missing-plan recovery')
    args = parser.parse_args(argv)
    if (args.recover_native_checkpoint is None)!=(args.recover_native_checkpoint_sha256 is None):
        parser.error('native checkpoint recovery needs both its path and independently verified SHA')
    from .native_resident import recover as recover_resident
    resident_state = recover_resident(args.out / 'resident-data') if (args.out / 'resident-data/current.json').exists() else None
    if resident_state and args.handoff:
        raise ValueError('retire resident generation with its exact saved checkpoint and trainer terminal receipt before runtime handoff')
    if resident_state and not args.handoff:
        allowed = {item['recipe']['path'] for item in resident_state['history']}
        allowed.add(resident_state['payload']['prior_recipe']['path'])
        if str(args.recipe.resolve()) not in allowed:
            raise ValueError('resident recovery was launched with an unrelated recipe')
        args.recipe = Path(resident_state['payload']['recipe']['path'])
    recipe = load_recipe(args.recipe)
    native = native_lineage(recipe)
    if not native:
        parser.error('only native positional-KV/hybrid-state recipes are executable')
    if args.until and args.until not in {stage['id'] for stage in recipe['stages']}:
        parser.error('unknown stopping stage')
    if args.inspect:
        print(json.dumps(recipe, indent=2))
        return
    args.out = args.out.resolve()
    from ..common import hdd_guard
    hdd_guard.require_nvme(args.out)
    overrides = parse_input_binding_overrides(args.input_binding)
    stage_inputs = {}
    for stage in recipe['stages']:
        stage_defaults = {}
        # A held stage (admitted: false) never runs in this lineage; its inputs are neither resolved nor hashed.
        stage_inputs[stage['id']] = ({} if stage.get('admitted') is False else
                                     resolve_stage_inputs(recipe, stage, stage_defaults, overrides))
    require_input_decoders(stage_inputs)
    inputs = {}
    for resolved in stage_inputs.values():
        for value in resolved.values():
            for dependency in _binding_files(value):
                inputs[dependency['path']] = dependency['sha256']
    package = Path(__file__).parents[1]
    frozen = args.out / 'runtime' / 'natlang_neuralese'
    plan_path = args.out / 'recipe-plan.json'
    plan = {'schema': 'natlang.neuralese-recipe-plan/1', 'recipe': recipe,
            'recipe_sha256': recipe_identity_sha256(args.recipe), 'inputs': inputs, 'device': args.device}
    if args.recover_native_checkpoint is not None:
        if args.handoff or resident_state or plan_path.exists():
            raise ValueError('checkpoint plan recovery requires a missing plan and no resident generation or handoff')
        from .native_plan_recovery import checkpoint_plan_recovery
        recovery=checkpoint_plan_recovery(args.recover_native_checkpoint,args.recipe,recipe,stage_inputs,
            expected_sha256=args.recover_native_checkpoint_sha256)
        plan['checkpoint_recovery']=recovery
        plan['recipe_sha256']=recovery['lineage_sha256']
    if resident_state and not args.handoff:
        plan['recipe_sha256'] = resident_state['payload']['plan']['recipe_sha256']
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
            if existing.get('checkpoint_recovery') and not existing.get('handoffs') and not resident_state:
                from .native_plan_recovery import checkpoint_plan_recovery
                recovery=checkpoint_plan_recovery(existing['checkpoint_recovery']['checkpoint']['path'],
                    args.recipe,recipe,stage_inputs,
                    expected_sha256=existing['checkpoint_recovery']['checkpoint']['sha256'])
                if recovery!=existing['checkpoint_recovery']:
                    raise ValueError('installed checkpoint recovery authority changed')
                plan['checkpoint_recovery']=recovery
                plan['recipe_sha256']=recovery['lineage_sha256']
            approved_current = (existing.get('handoffs') or [{}])[-1].get('recipe_sha256')
            if (existing.get('recipe') == plan['recipe'] and existing.get('recipe_sha256') != plan['recipe_sha256'] and
                    ('extends' in json.loads(args.recipe.read_text()) or approved_current == plan['recipe_sha256'])):
                # An already committed exact handoff (or equivalent inheritance
                # conversion) retains lineage on normal restart. Every other
                # plan field and the frozen runtime remain checked below.
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
                          'checkpoint_recovery': plan.get('checkpoint_recovery'),
                          'stages_started': [stage['id'] for stage in recipe['stages']
                                             if stage_started(args.out, stage['id'])],
                          'handoff_record': ({k: v for k, v in plan['handoffs'][-1].items() if k != 'previous_inputs'}
                                             if args.handoff and plan.get('handoffs') else None)},
                         indent=1), flush=True)
        return 0
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
                              'device': args.device,
                              'until': args.until,
                              'recipe_plan_path': str(plan_path),
                              'recipe_plan_sha256': sha(plan_path),
                              'frozen_runtime_sha256': plan.get('code', {}),
                              'stage_inputs': stage_inputs,
                              'runtime_environment': declared_runtime_environment(recipe)})
    with _ChildSignalForwarder() as signal_forwarder:
        stopped = signal_forwarder.stopped
        reports = []
        for stage in recipe['stages']:
            if stopped[0]:
                print(json.dumps({'status': 'interrupted_before_child_start', 'stage': stage['id']}), flush=True)
                return 128 + signal.SIGTERM
            directory = args.out / stage['id']
            report_path = args.out / (stage['id'] + '-report.json')
            kind = stage['kind']
            if report_path.exists():
                report = json.loads(report_path.read_text())
                require_gate(report['gate'], kind)
                lineage_inputs = [inputs] + [item['previous_inputs'] for item in plan.get('handoffs', [])
                                             if 'previous_inputs' in item]
                if report.get('recipe_sha256') != plan['recipe_sha256'] or report.get('inputs') not in lineage_inputs:
                    raise ValueError('stage report belongs to another recipe/input lineage')
                if sha(report['artifact']) != report['artifact_sha256']:
                    raise ValueError('qualified artifact changed')
            else:
                if stage.get('admitted') is False:
                    raise ValueError('stage is declared but not admitted: ' + stage['id'] +
                                     (' (' + stage['admission'] + ')' if isinstance(stage.get('admission'), str) else ''))
                for dependency in stage['requires']:
                    predecessor = next(item for item in reports if item['id'] == dependency)
                    require_gate(predecessor['gate'], predecessor['kind'])
                output = directory / HANDLERS[kind]['result']
                command = [sys.executable, '-m', HANDLERS[kind]['module']]
                resolved = stage_inputs[stage['id']]
                command += stage_input_args(resolved, kind)
                if kind == 'native_shared_stage_controller':
                    validate_shared_controller_output(resolved['controller_config']['path'], directory)
                else:
                    command += ['--out', str(directory), '--device', args.device]
                native_predecessors = [name for name in stage['requires']
                    if next((item['kind'] for item in recipe['stages'] if item['id'] == name), None)
                       in NATIVE_TRAINING_KINDS]
                if kind in NATIVE_TRAINING_KINDS and native_predecessors:
                    if len(native_predecessors) != 1:
                        raise ValueError('native stage continuation requires one native-state predecessor')
                    if 'continue_from' in resolved:
                        raise ValueError('native stage cannot bind both a native predecessor and independent continue_from')
                    predecessor = next(item for item in reports if item['id'] == native_predecessors[0])
                    command += ['--continue-from', predecessor['artifact']]
                recovery = plan.get('checkpoint_recovery')
                if recovery and stage['id'] == recovery['stage_id']:
                    if '--continue-from' in command:
                        raise ValueError('checkpoint recovery conflicts with another stage continuation')
                    command += ['--continue-from', recovery['checkpoint']['path']]
                stage_parameters = effective_stage_parameters(stage, reports, recipe)
                from . import hardware_profile
                profiled, profile_environment, profile_record = hardware_profile.for_stage(
                    kind, stage_parameters, args.device)
                if profile_record is not None:
                    print(json.dumps({'event': 'hardware_profile', 'stage': stage['id'], **profile_record}), flush=True)
                command += stage_parameter_args(profiled)
                if kind in NATIVE_TRAINING_KINDS:
                    prerequisites = []
                    for dependency in stage['requires']:
                        prior = next((item for item in reports if item['id'] == dependency), None)
                        if prior is None or prior['kind'] not in NATIVE_TRAINING_KINDS:
                            continue
                        receipt_path = Path(prior['artifact']).parent / 'native-training-receipt.json'
                        if not receipt_path.is_file():
                            raise ValueError('native predecessor lacks exposure/readiness receipt: ' + prior['id'])
                        receipt = json.loads(receipt_path.read_text())
                        prerequisites.append({'id': prior['id'], 'completed': receipt.get('completed') is True,
                            'exposure_receipt': {'path': str(receipt_path.resolve()), 'sha256': sha(receipt_path)}})
                    if kind == 'native_state_sft':
                        command += ['--resident-data-directory', str(args.out / 'resident-data')]
                    recipe_binding = native_recipe_binding(args.recipe, plan['recipe_sha256'])
                    command += ['--recipe-binding', json.dumps(recipe_binding, sort_keys=True, separators=(',', ':')),
                                '--stage-id', stage['id'], '--prerequisite-stages',
                                json.dumps(prerequisites, sort_keys=True, separators=(',', ':'))]
                environment = dict(os.environ)
                declared_environment = declared_runtime_environment(recipe)
                if 'NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE' not in declared_environment:
                    environment.pop('NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE', None)
                environment.update(profile_environment)
                environment.update(declared_environment)
                environment = python_package_environment(environment, frozen.parent)
                if args.plan_only:
                    print(json.dumps({'stage': stage['id'], 'plan_only_command': command}), flush=True)
                    return 0
                if hdd_guard.RECORDS:
                    write_json(args.out / 'hdd-staged-inputs.json', hdd_guard.RECORDS)
                print(json.dumps({'stage': stage['id'], 'command': command}), flush=True)
                result = signal_forwarder.run(command, env=environment)
                if stopped[0]:
                    print(json.dumps({'status': 'interrupted', 'stage': stage['id'],
                                      'child_exit_code': result.returncode}), flush=True)
                    return result.returncode or (128 + signal.SIGTERM)
                if result.returncode:
                    raise RuntimeError('native stage failed: ' + stage['id'])
                if kind in NATIVE_TRAINING_KINDS | COEFFICIENT_NATIVE_KINDS:
                    gate = json.loads((directory / 'report.json').read_text())
                else:
                    gate = json.loads((directory / 'shared-stage-completion.json').read_text())
                report = {'id': stage['id'], 'kind': kind, 'recipe_sha256': plan['recipe_sha256'],
                          'inputs': inputs, 'gate': gate, 'artifact': str(output), 'artifact_sha256': sha(output)}
                write_json(args.out / (stage['id'] + '-attempt.json'), report)
                require_gate(gate, kind)
                write_json(report_path, report)
            reports.append(report)
            if args.until == stage['id']:
                return 0
    print(json.dumps({'status': 'recipe_completed', 'completed_stages': [report['id'] for report in reports]}), flush=True)


if __name__ == '__main__':
    sys.exit(main())
