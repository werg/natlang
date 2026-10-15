"""Identity-safe native optimizer steps published as immutable composite checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from ..model.native_artifact import decode_native_tensor_artifact
from ..model.native_composite import write_native_composite_checkpoint
from ..common.hashing import sha256_file_hex_cached
from .chat import RequestError


def _optimizer_parent(engine, state_id):
    """A resumed optimizer artifact is valid only when the current manifest names it."""
    if not state_id:
        return None
    path = Path(engine.native_checkpoint)
    try:
        manifest = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError('optimizer state can only resume from its exact composite checkpoint')
    binding = manifest.get('optimizer_state_artifact')
    if not binding or binding.get('id') != state_id:
        raise ValueError('optimizer state artifact is not bound by the current checkpoint')
    record = engine.store.get_native_tensor_artifact(state_id)
    if record is None:
        raise ValueError('optimizer state artifact is unavailable on this serving process')
    return record.artifact


def native_optimizer_step(engine, body, owner):
    """Apply SGD/Adam to the compact native interface and atomically publish its new identity.

    The live model is restored before returning. Full backbone gradients remain
    explicitly unsupported until the full-checkpoint writer can persist them.
    """
    if body.get('schema') != 'natlang.native-optimizer-request/1':
        raise RequestError('neuralese-native-optim-schema', 'native optimizer requests need schema natlang.native-optimizer-request/1')
    try:
        identity = engine.native_identity
        if body.get('native_identity') != identity:
            raise ValueError('optimizer request is bound to another exact live checkpoint')
        gradient_id = body.get('gradient_id')
        record = engine.store.get_native_tensor_artifact(gradient_id) if isinstance(gradient_id, str) else None
        if record is None:
            raise ValueError('optimizer request needs an available named native gradient artifact')
        gradient = record.artifact
        if gradient.kind != 'parameter-gradient' or gradient.native_identity != identity:
            raise ValueError('gradient kind or model identity differs from the live checkpoint')
        names = set(gradient.tensors)
        allowed = {'input_rows', 'adapter.down.weight', 'adapter.up.weight', 'output_rows'}
        if not names or not names <= allowed:
            raise ValueError('gradient names must select only restorable native interface parameters')
        parameters = dict(engine.shared_model.named_parameters())
        if not names <= parameters.keys():
            raise ValueError('gradient names are not parameters of the live native interface')
        options = body.get('optimizer')
        if not isinstance(options, dict) or options.get('name') not in {'sgd', 'adam'}:
            raise ValueError('optimizer name must be sgd or adam')
        name = options['name']
        lr = float(options.get('learning_rate', 1e-3))
        weight_decay = float(options.get('weight_decay', 0.0))
        if not torch.isfinite(torch.tensor([lr, weight_decay])).all() or lr <= 0 or weight_decay < 0:
            raise ValueError('optimizer learning rate and weight decay must be finite and in range')
        beta1, beta2 = float(options.get('beta1', .9)), float(options.get('beta2', .999))
        eps = float(options.get('eps', 1e-8))
        if name == 'adam' and (not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or eps <= 0):
            raise ValueError('Adam beta values and epsilon are out of range')
        old_state = _optimizer_parent(engine, body.get('optimizer_state_id'))
        if old_state is not None:
            if old_state.kind != 'optimizer-state' or old_state.native_identity['architecture'] != identity['architecture']:
                raise ValueError('optimizer state artifact is malformed or belongs to another architecture')
            prior_step = int((old_state.provenance or {}).get('step', 0))
            if (old_state.provenance or {}).get('optimizer') != name:
                raise ValueError('optimizer algorithm cannot change while restoring its state')
            state_tensors = old_state.tensors
        else:
            prior_step, state_tensors = 0, {}
        step = prior_step + 1
        if name == 'sgd':
            next_state = {'optimizer.step': torch.tensor([step], dtype=torch.float32)}
        else:
            next_state = {}
        snapshots = {}
        with torch.no_grad():
            for param_name in sorted(names):
                parameter = parameters[param_name]
                grad = gradient.tensors[param_name].to(device=parameter.device, dtype=parameter.dtype)
                if tuple(grad.shape) != tuple(parameter.shape) or not torch.isfinite(grad).all():
                    raise ValueError(f'gradient shape or values are invalid for {param_name}')
                snapshots[param_name] = parameter.detach().clone()
                g = grad + weight_decay * parameter if weight_decay else grad
                if name == 'sgd':
                    update = g
                else:
                    mname, vname = f'm.{param_name}', f'v.{param_name}'
                    m = state_tensors.get(mname, torch.zeros_like(parameter)).to(parameter)
                    v = state_tensors.get(vname, torch.zeros_like(parameter)).to(parameter)
                    m = beta1*m + (1-beta1)*g
                    v = beta2*v + (1-beta2)*g.square()
                    next_state[mname], next_state[vname] = m.detach().cpu(), v.detach().cpu()
                    update = (m / (1-beta1**step)) / ((v / (1-beta2**step)).sqrt() + eps)
                parameter.add_(update, alpha=-lr)
            try:
                receipt = write_native_composite_checkpoint(engine.native_checkpoint,
                    Path(engine.native_checkpoint).resolve().parent / 'native-updates', engine.shared_model,
                    engine.model_config, base_identity=identity, source_gradient_id=gradient_id,
                    optimizer={'name': name, 'learning_rate': lr, 'weight_decay': weight_decay,
                        'beta1': beta1, 'beta2': beta2, 'eps': eps, 'step': step},
                    optimizer_tensors=next_state,
                    update_provenance={'objective': gradient.provenance.get('objective'),
                        'gradient_artifact_id': gradient_id, 'updated_parameter_names': sorted(names),
                        'optimizer_state_parent_id': body.get('optimizer_state_id')})
            finally:
                for param_name, old in snapshots.items():
                    parameters[param_name].copy_(old)
        update_record = engine.store.put_native_tensor_artifact(
            decode_native_tensor_artifact((Path(receipt['path']).parent /
                f"{receipt['parameter_artifact_id']}.safetensors").read_bytes())[0], owner)
        state_id = _optimizer_state_id(receipt['path'])
        if state_id:
            state_binding = json.loads(Path(receipt['path']).read_text())['optimizer_state_artifact']
            state_bytes = (Path(receipt['path']).parent / state_binding['path']).read_bytes()
            engine.store.put_native_tensor_artifact(decode_native_tensor_artifact(state_bytes)[0], owner)
        return {'checkpoint': receipt, 'parameters': update_record.meta(),
                'optimizer_state_id': state_id,
                'native_identity': receipt['native_identity'], 'step': step}
    except RequestError:
        raise
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        raise RequestError('neuralese-native-optim-invalid', str(error)) from error


def _optimizer_state_id(manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    binding = manifest.get('optimizer_state_artifact')
    return binding['id'] if binding else None
