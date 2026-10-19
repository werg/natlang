"""Bind each supervised command to the publication exporter without decoding weights."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

from ..common.jsonio import write_bytes_durable, write_json_durable
from ..common.hashing import canonical_json_sha256_hex as digest
from ..train.native_resident import binding, code_inventory
from ..model.native_checkpoint_control import validate_publication_selection

SCHEMA = 'natlang.owned-publication-owner/1'
POINTER_SCHEMA = 'natlang.owned-publication-owner-current/1'
TRAINER_MODULE = 'natlang_neuralese.train.native_state_sft'
TRAINER_MODULES = frozenset({TRAINER_MODULE, 'natlang_neuralese.train.native_preference'})


def _file_binding(path):
    ref = binding(path)
    ref['bytes'] = Path(ref['path']).stat().st_size
    return ref


def _write_immutable(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or path.read_bytes() != raw:
            raise ValueError('publication owner generation already has different bytes')
        return
    write_bytes_durable(path, raw)
    if path.is_symlink() or path.read_bytes() != raw:
        raise ValueError('publication owner generation changed during durable write')


def _arg(command, name):
    indices = [i for i, value in enumerate(command) if value == name]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError('publication owner command needs exactly one ' + name)
    return command[indices[0] + 1]


def _trainer_invocation(command):
    """Find the exact Python module invocation in production's expanded argv."""
    if not isinstance(command, list) or not command or any(not isinstance(arg, str) for arg in command):
        raise ValueError('publication owner command must be a string argv list')
    index = 0
    environment = {}
    if command[0] == 'env':
        index = 1
        while index < len(command) and '=' in command[index] and not command[index].startswith('-'):
            key, value = command[index].split('=', 1)
            if not key or key in environment:
                raise ValueError('publication owner command has invalid environment prefix')
            environment[key] = value
            index += 1
    if index >= len(command):
        raise ValueError('publication owner command lacks Python executable')
    executable = Path(command[index]).name
    if executable not in {'python', 'python3'} and not executable.startswith('python3.'):
        raise ValueError('publication owner command is not a direct Python invocation')
    index += 1
    if index < len(command) and command[index] == '-u':
        index += 1
    if index + 1 >= len(command) or command[index] != '-m':
        raise ValueError('publication owner command must use Python -m')
    module = command[index + 1]
    if module not in TRAINER_MODULES:
        raise ValueError('publication owner command is not a supported native trainer')
    return module, environment, index + 2


def emit_current(config, command, *, out, selected_checkpoint, runtime_root=None):
    """Publish one exact quantum intent; it is not a liveness or progress claim."""
    pointer_value = config.get('native_checkpoint_publication_owner_pointer')
    if pointer_value is None:
        return None
    selection = validate_publication_selection(config.get('native_checkpoint_publication'))
    role = config.get('native_checkpoint_publication_role')
    if role is None:
        role = config.get('native_participant')
    if role not in ('lfm', 'mellum'):
        raise ValueError('publication owner needs its exact model role')
    declared_role = config.get('native_participant')
    model = config.get('model')
    if declared_role is not None and declared_role != role:
        raise ValueError('publication owner role differs from its semantic participant')
    if model is not None and model != role:
        raise ValueError('publication owner role differs from the configured model')
    if declared_role is None and config.get('native_actor_mode') == 'raw-state':
        if config.get('native_checkpoint_publication_role') != role:
            raise ValueError('raw-state publication needs an explicit operational model role')
    pointer = Path(pointer_value)
    if not pointer.is_absolute() or pointer.is_symlink():
        raise ValueError('publication owner pointer must be an absolute nonsymlink path')
    if pointer.exists():
        existing_pointer = json.loads(pointer.read_text())
        if (existing_pointer.get('schema') != POINTER_SCHEMA
                or existing_pointer.get('role') != role):
            raise ValueError('publication owner pointer belongs to another role/schema')
    command = list(command)
    trainer_module, environment, module_index = _trainer_invocation(command)
    from ..train import native_resident as resident
    from ..model import native_checkpoint_control as checkpoint_control
    configured_runtime = config.get('native_checkpoint_publication_runtime_root')
    if runtime_root is None:
        runtime_root = configured_runtime or Path(__file__).resolve().parents[4]
    runtime = Path(runtime_root).resolve(strict=True)
    if configured_runtime is not None and Path(configured_runtime).resolve(strict=True) != runtime:
        raise ValueError('publication owner runtime differs from its explicit runtime binding')
    command_runtime = environment.get('NATLANG_REPO')
    if command_runtime is not None and Path(command_runtime).resolve(strict=True) != runtime:
        raise ValueError('publication owner runtime differs from the expanded trainer environment')
    if configured_runtime is not None and command_runtime is None:
        raise ValueError('explicit publication runtime is not bound by the expanded trainer environment')
    if configured_runtime is None and config.get('runtime') is not None and Path(config['runtime']).resolve(strict=True) != runtime:
        raise ValueError('publication owner runtime differs from its bound operation config')
    package = runtime / 'training/neuralese/natlang_neuralese'
    if (Path(resident.__file__).resolve() != package / 'train/native_resident.py'
            or Path(checkpoint_control.__file__).resolve() != package / 'model/native_checkpoint_control.py'):
        raise ValueError('publication owner dependencies are not imported from the bound runtime')
    from ..train.native_resident import digest as canonical_digest
    runtime_code = code_inventory(package)
    runtime_sha = canonical_digest(runtime_code)
    if any(not Path(selection[key]['path']).resolve(strict=True).is_relative_to(runtime)
           for key in ('producer', 'dependency_producer')):
        raise ValueError('publication producers escaped the executing runtime')
    parent = _file_binding(selected_checkpoint)
    if Path(_arg(command, '--continue-from')).resolve() != Path(parent['path']):
        raise ValueError('publication owner selected parent differs from the actual trainer command')
    output = Path(out).resolve()
    command_output = Path(_arg(command, '--out')).resolve()
    if not command_output.is_relative_to(output):
        raise ValueError('publication owner writer output escaped its supervised job directory')
    if '--checkpoint-publication' not in command:
        raise ValueError('publication owner command disabled its declared publisher')
    encoded_roots = json.loads(_arg(command, '--checkpoint-publication-readonly-input-roots'))
    encoded_namespace = _arg(command, '--checkpoint-publication-namespace')
    if (encoded_roots != selection['readonly_input_roots']
            or encoded_namespace != selection['namespace']):
        raise ValueError('publication owner selection differs from actual trainer arguments')
    command_raw = (json.dumps(command, sort_keys=True, indent=2) + '\n').encode()
    command_sha = hashlib.sha256(command_raw).hexdigest()
    command_path = output / 'publication-owner' / 'commands' / (command_sha + '.json')
    _write_immutable(command_path, command_raw)
    operation = {key: config[key] for key in ('request_id', 'round_id', 'cycle_id',
        'production_stage', 'recipe_lineage_sha256') if key in config}
    operation.update(out=str(output), model_role=role, trainer_module=trainer_module)
    if config.get('native_participant') is not None:
        operation['participant'] = config['native_participant']
    if config.get('runtime') is not None:
        operation['orchestration_runtime'] = str(config['runtime'])
    namespace = Path(selection['namespace']).resolve()
    publication_spec = {'schema': 'natlang.owned-publication-export/1',
        'role': role, 'runtime': str(runtime), 'runtime_code_sha256': runtime_sha,
        'production_root': str(namespace), 'control_root': str(namespace/'control'),
        'effective_argv': {**_file_binding(command_path), 'allowed_output_root': str(output),
            'writer_output': str(command_output), 'runtime_from_command_env': command_runtime is not None},
        'parent': parent, 'publication': selection,
        'pin_directory': str(namespace.parent/'production/durable-exports/current-owned-publications'/role),
        'reserve_bytes': int(config.get('native_checkpoint_publication_reserve_bytes',20*1024**3))}
    body = {'schema': SCHEMA, 'role': role, 'runtime': str(runtime),
        'runtime_code_sha256': runtime_sha, 'package': {'path': str(package), 'code': runtime_code},
        'operation': operation, 'publication_spec': publication_spec}
    generation = digest(body)
    body['generation_sha256'] = generation
    manifest_raw = (json.dumps(body, sort_keys=True, indent=2) + '\n').encode()
    manifest = output / 'publication-owner' / generation / 'manifest.json'
    _write_immutable(manifest, manifest_raw)
    pointer.parent.mkdir(parents=True,exist_ok=True)
    write_json_durable(pointer, {'schema': POINTER_SCHEMA, 'role': role,
        'generation_sha256': generation, 'owner': _file_binding(manifest)})
    return {'generation_sha256': generation, 'manifest': _file_binding(manifest),
        'pointer': str(pointer), 'parent': parent, 'runtime_code_sha256': runtime_sha,
        'publication_roots': selection['readonly_input_roots'], 'qualification_credit': False}


def read_owner(pointer, *, role):
    """Local verifier used by controller-side tests and metadata-only inspectors."""
    pointer = Path(pointer)
    before = pointer.stat()
    raw = pointer.read_bytes()
    selected = json.loads(raw)
    if selected.get('schema') != POINTER_SCHEMA or selected.get('role') != role:
        raise ValueError('publication owner pointer belongs to another role/schema')
    ref = selected['owner']
    raw_manifest_path = Path(ref['path'])
    if raw_manifest_path.is_symlink():
        raise ValueError('publication owner manifest may not be a symlink')
    manifest_path = raw_manifest_path.resolve(strict=True)
    if (manifest_path.stat().st_size != ref.get('bytes')
            or hashlib.sha256(manifest_path.read_bytes()).hexdigest() != ref['sha256']):
        raise ValueError('publication owner manifest binding changed')
    manifest = json.loads(manifest_path.read_bytes())
    generation = manifest.pop('generation_sha256', None)
    if digest(manifest) != generation or generation != selected['generation_sha256']:
        raise ValueError('publication owner generation digest differs')
    manifest['generation_sha256'] = generation
    if (manifest.get('schema') != SCHEMA or manifest.get('role') != role
            or manifest.get('publication_spec', {}).get('role') != role
            or manifest['publication_spec'].get('runtime') != manifest.get('runtime')
            or manifest['publication_spec'].get('runtime_code_sha256') != manifest.get('runtime_code_sha256')):
        raise ValueError('publication owner manifest role/schema differs')
    if pointer.stat() != before or pointer.read_bytes() != raw:
        raise ValueError('publication owner pointer changed during read')
    return manifest, {'path': str(manifest_path), 'sha256': ref['sha256'], 'bytes': ref['bytes']}
