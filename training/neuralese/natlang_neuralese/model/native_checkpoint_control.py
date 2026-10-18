"""Explicit source-bound controls derived from one verified full native decode."""
from __future__ import annotations
import hashlib
import inspect
import json
import sys
import types
from pathlib import Path

from ..common.hashing import sha256_file_hex, sha256_file_hex_cached
from .native_composite import _durable_immutable

SCHEMA = 'natlang.native-checkpoint-control/1'
_HASHING_SIDECAR = None


def install_control_hashing(reference):
    """Bootstrap an explicitly reviewed hash helper, never checkpoint-selected code.

    A qualified old runtime may bind the published strong hasher through its
    control manifest without replacing its ordinary source-policy module.
    """
    global sha256_file_hex_cached, sha256_file_hex, _HASHING_SIDECAR
    path = checked_file(reference)
    raw = path.read_bytes()
    if len(raw) != reference['bytes'] or hashlib.sha256(raw).hexdigest() != reference['sha256']:
        raise ValueError('checkpoint control hashing helper changed before execution')
    name = 'natlang_neuralese.common.checkpoint_control_hashing_' + reference['sha256']
    module = types.ModuleType(name)
    module.__package__ = 'natlang_neuralese.common'
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    if 'trust_archive_receipt' not in inspect.signature(module.sha256_file_hex_cached).parameters:
        raise ValueError('checkpoint control hashing helper lacks strong content identity')
    sha256_file_hex_cached = module.sha256_file_hex_cached
    sha256_file_hex = module.sha256_file_hex
    _HASHING_SIDECAR = dict(reference)
    checked_file(reference)
    return dict(reference)


def _strong_digest(path):
    # Older qualified packages lack the opt-out from archive receipts. Reuse
    # their existing full hasher instead of changing the seven-file source
    # policy or trusting a weaker receipt for this explicit control boundary.
    if 'trust_archive_receipt' in inspect.signature(sha256_file_hex_cached).parameters:
        return sha256_file_hex_cached(path, trust_archive_receipt=False)
    return sha256_file_hex(path)


def _file_binding(path):
    path = Path(path).resolve(strict=True)
    before = source_stat(path)
    digest = _strong_digest(path)
    if source_stat(path) != before:
        raise ValueError('checkpoint control file changed during its hash')
    return {'path': str(path), 'bytes': before[2], 'sha256': digest}


def source_stat(path):
    stat = Path(path).stat()
    return tuple(getattr(stat, key) for key in
                 ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))


def checked_file(reference):
    if (not isinstance(reference, dict) or set(reference) != {'path', 'sha256', 'bytes'}
            or type(reference['bytes']) is not int or reference['bytes'] <= 0):
        raise ValueError('checkpoint control needs exact path/SHA/byte bindings')
    path = Path(reference['path']).resolve(strict=True)
    if _file_binding(path) != reference:
        raise ValueError('checkpoint control file differs from its binding')
    return path


def read_checkpoint_control(checkpoint, reference, *, dependency_producer=None):
    """Authenticate an explicitly selected derivative; never deserialize training state."""
    path = checked_file(reference)
    value = json.loads(path.read_text())
    source = _file_binding(checkpoint)
    resume = value.get('resumable_source') or {}
    if (value.get('schema') != SCHEMA or value.get('training_authority') != 'full-source-only'
            or any(source[key] != value.get('source', {}).get(key) for key in ('sha256', 'bytes'))
            or resume.get('optimizer_present') is not True or resume.get('rng_present') is not True
            or resume.get('source_receipts_present') is not True
            or resume.get('schema') != 'natlang.neuralese-native-sft-training/1'
            or type(value.get('step')) is not int or value['step'] < 0
            or resume.get('global_step') != value['step']):
        raise ValueError('checkpoint controls do not attest this full resumable source')
    checked_file(value['producer'])
    checked_file(value['dependency_producer'])
    if value.get('hashing_sidecar') is not None:
        checked_file(value['hashing_sidecar'])
    if dependency_producer is not None and value['dependency_producer'] != _file_binding(dependency_producer):
        raise ValueError('checkpoint control dependency verifier changed')
    from .native_bundle import ACTOR_VIEW_SCHEMA
    view_path = checked_file(value['actor_view'])
    view = json.loads(view_path.read_text())
    if (view.get('schema') != ACTOR_VIEW_SCHEMA or view.get('training_resume') is not False
            or view.get('step') != value['step'] or view.get('model_config') != value['model_config']
            or any(view.get('source', {}).get(key) != source[key] for key in ('sha256', 'bytes'))):
        raise ValueError('checkpoint controls changed their actual source-bound actor view')
    checked_file(view['mechanism'])
    raw = value['raw_policy']
    if (raw['native_identity']['checkpoint_sha256'] != source['sha256']
            or raw['native_identity']['tokenizer_fingerprint'] != value['model_config']['tokenizer_fingerprint']
            or raw['native_identity']['architecture'] != value['model_config']['architecture']):
        raise ValueError('checkpoint controls changed sampled native identity')
    sidecar = checked_file(raw['history_reasoning_binding'])
    history = json.loads(sidecar.read_text())
    if (history.get('checkpoint_sha256') != source['sha256']
            or history.get('tokenizer_fingerprint') != raw['native_identity']['tokenizer_fingerprint']
            or history.get('history_reasoning') != raw['history_reasoning']):
        raise ValueError('checkpoint controls changed saved tokenizer history')
    for child in value['tokenizer_files']:
        checked_file(child)
    closure = value.get('dependencies')
    if not isinstance(closure, dict) or set(closure) != {'files', 'assets'}:
        raise ValueError('checkpoint controls lack the exact consumed dependency closure')
    if not all(isinstance(closure[key], list) for key in ('files', 'assets')):
        raise ValueError('checkpoint control dependency inventories must be lists')
    return value


def checkpoint_control_dependencies(reference, *, checkpoint):
    """The immutable control/view/tokenizer mechanism closure, excluding full-source alias."""
    path = checked_file(reference)
    value = json.loads(path.read_text())
    if value.get('schema') != SCHEMA:
        raise ValueError('unsupported checkpoint control dependency schema')
    from .native_bundle import actor_view_dependencies
    view = json.loads(checked_file(value['actor_view']).read_text())
    view_dependencies = actor_view_dependencies(value['actor_view'], metadata_only=True,
                                               checkpoint=checkpoint)
    return [reference, value['producer'], value['dependency_producer'],
            *([value['hashing_sidecar']] if value.get('hashing_sidecar') is not None else []),
            view['mechanism'], *value['tokenizer_files'],
            *view_dependencies[:2]]  # The already-bound actual full parent stays separate.


def write_checkpoint_control(checkpoint, out, *, validated_state, decoded_source_stat,
                             actor_view, dependency_collector,
                             namespace='/work/native-production', readonly_input_roots=()):
    """Use only the actual already-loaded saved object; unsupported source forms return None."""
    from .native_model import CHECKPOINT_SCHEMA
    from ..rl.native_raw_policy import require_raw_training, prepare_raw_policy
    source_path = Path(checkpoint).resolve(strict=True)
    if validated_state.get('schema') != CHECKPOINT_SCHEMA:
        return None
    if (validated_state.get('optimizer_pruned') or validated_state.get('inference_only')
            or source_stat(source_path) != tuple(decoded_source_stat)):
        raise ValueError('checkpoint control source differs from its verified full decode')
    training = require_raw_training(validated_state)
    if not training.get('source_receipts'):
        raise ValueError('checkpoint control source has no saved source receipts')
    step = validated_state.get('step')
    if type(step) is not int or step < 0 or training.get('global_step', step) != step:
        raise ValueError('checkpoint control source has inconsistent actual saved steps')
    source = _file_binding(source_path)
    out = Path(out).resolve()/source['sha256']
    out.mkdir(parents=True, exist_ok=True)
    # Reuse the exact default raw preparation with its in-process saved object;
    # it validates optimizer/RNG and derives identity from the actual tensors.
    raw = prepare_raw_policy(source_path, base=validated_state['model_config']['base'],
        tokenizer_out=out/'tokenizer', verified_checkpoint=(source, validated_state))
    raw['history_reasoning_binding'] = _file_binding(raw['history_reasoning_binding']['path'])
    dependency_module = __import__('sys').modules[dependency_collector.__module__]
    dependencies = dependency_collector({}, namespace=namespace,
        readonly_input_roots=readonly_input_roots, _checkpoint_payload=validated_state)
    value = {'schema': SCHEMA, 'source': source, 'step': step,
             'model_config': validated_state['model_config'], 'raw_policy': raw,
             'resumable_source': {'schema': training['schema'], 'global_step': step,
                 'optimizer_present': True, 'rng_present': True, 'source_receipts_present': True},
             'training_authority': 'full-source-only', 'actor_view': actor_view,
             'dependencies': dependencies,
             'dependency_context': {'namespace': str(Path(namespace).absolute()),
                                    'readonly_input_roots': sorted(readonly_input_roots)},
             'producer': _file_binding(__file__),
             'hashing_sidecar': _HASHING_SIDECAR,
             'dependency_producer': _file_binding(dependency_module.__file__),
             'tokenizer_files': [_file_binding(child) for child in sorted((out/'tokenizer').rglob('*'))
                                 if child.is_file()]}
    if source_stat(source_path) != tuple(decoded_source_stat):
        raise ValueError('checkpoint control source changed during derivation')
    content = (json.dumps(value, sort_keys=True, indent=2)+'\n').encode()
    path = out/(hashlib.sha256(content).hexdigest()+'.json')
    _durable_immutable(path, content)
    reference = _file_binding(path)
    read_checkpoint_control(source_path, reference, dependency_producer=dependency_module.__file__)
    pointer = (json.dumps(reference, sort_keys=True, indent=2)+'\n').encode()
    _durable_immutable(out/'control-current.json', pointer)
    return reference


def resolve_checkpoint_controls(config, checkpoint):
    """Freeze one optional published control pointer into this round's config copy."""
    directory = config.get('native_checkpoint_control_directory')
    if directory is None:
        return config
    root = Path(directory)
    if not root.is_absolute() or '..' in root.parts or root.resolve() != root:
        raise ValueError('checkpoint control directory must be an explicit canonical path')
    result = dict(config)
    result.pop('native_checkpoint_control_directory')
    source = _file_binding(checkpoint['path'])
    if any(source[key] != checkpoint.get(key) for key in ('sha256', 'bytes')):
        raise ValueError('checkpoint control selection changed its exact round parent')
    pointer = root/source['sha256']/'control-current.json'
    if not pointer.exists():
        return result  # This sealed round takes the unchanged full path.
    reference = json.loads(pointer.read_text())
    path = checked_file(reference)
    if not path.is_relative_to(pointer.parent.resolve()):
        raise ValueError('checkpoint control pointer escaped its exact source directory')
    read_checkpoint_control(source['path'], reference)
    controls = dict(result.get('native_checkpoint_controls') or {})
    if source['sha256'] in controls and controls[source['sha256']] != reference:
        raise ValueError('checkpoint control pointer conflicts with an explicit selector')
    controls[source['sha256']] = reference
    result['native_checkpoint_controls'] = controls
    result['native_checkpoint_control_resolution'] = {
        'source': source, 'control': reference, 'pointer_sha256': _file_binding(pointer)['sha256'],
        'scope': 'Frozen optional derivative selection; the mutable publisher pointer is not a later consumed artifact'}
    return result
