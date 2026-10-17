"""Identity-safe native optimizer steps published as immutable composite checkpoints."""
from __future__ import annotations

import json
import copy
from pathlib import Path

import torch

from ..model.native_artifact import decode_native_tensor_artifact
from ..model.native_composite import write_native_composite_checkpoint, native_update_directory
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


def _copy_parameter_targets(model, names):
    """Copy the module tree and selected tied parameters; share untouched storage."""
    available = dict(model.named_parameters())
    if not names or not names <= available.keys():
        raise ValueError('gradient names must select actual native model parameters')
    selected = {id(available[name]): torch.nn.Parameter(available[name].detach().clone(), requires_grad=True)
                for name in names}
    modules = {}
    def clone(module):
        if id(module) in modules:
            return modules[id(module)]
        result = copy.copy(module)
        modules[id(module)] = result
        result._parameters = {name:selected.get(id(value),value) for name,value in module._parameters.items()}
        result._modules = {name:clone(child) if child is not None else None for name,child in module._modules.items()}
        return result
    return clone(model)


def native_optimizer_step(engine, body, owner):
    with engine.native_execution_lock:
        return _native_optimizer_step_impl(engine, body, owner)


def _native_optimizer_step_impl(engine, body, owner):
    """Publish named backbone/interface updates without modifying the live actor."""
    if getattr(engine,'native_actor_view',None) is not None:
        raise RequestError('neuralese-native-optim-invalid',
                           'Restore the complete parent checkpoint before requesting optimizer updates.')
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
        if gradient.kind=='state-gradient':
            from ..model.native_state_derivation import optimize_state
            result,_,_=optimize_state(engine,record,body.get('optimizer'),body.get('optimizer_state_id'),owner,
                program_id=(gradient.provenance or {}).get('derivative_program_id'))
            return {**result,'native_identity':identity,'native_state':engine.store.get_native_state(result['native_state_id']).meta()}
        if gradient.kind == 'adapter-gradient':
            from ..model.native_adapters import optimize_adapter
            with engine.native_execution_lock:
                return optimize_adapter(engine, gradient, body.get('optimizer'), body.get('optimizer_state_id'), owner)
        if gradient.kind != 'parameter-gradient' or gradient.native_identity != identity:
            raise ValueError('gradient kind or model identity differs from the live checkpoint')
        names = set(gradient.tensors)
        value_id=(gradient.provenance or {}).get('parameter_value_id')
        if value_id is not None:
            parent=engine.store.get_native_tensor_artifact(value_id)
            live=dict(engine.shared_model.named_parameters())
            if (parent is None or parent.artifact.kind!='parameter-set' or parent.artifact.native_identity!=identity
                    or not names<=parent.artifact.tensors.keys() or any(name not in live
                    or not torch.equal(live[name].detach().cpu(),parent.artifact.tensors[name]) for name in names)):
                raise ValueError('gradient belongs to another exact functional parameter value; apply it through its bound derivative program')
        updated_model = _copy_parameter_targets(engine.shared_model, names)
        parameters = dict(updated_model.named_parameters())
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
        if name == 'adam' and (not 0 <= beta1 < 1 or not 0 <= beta2 < 1
                               or not torch.isfinite(torch.tensor(eps)) or eps <= 0):
            raise ValueError('Adam beta values and epsilon are out of range')
        momentum = float(options.get('momentum',0.))
        if not torch.isfinite(torch.tensor(momentum)) or momentum < 0:
            raise ValueError('SGD momentum must be finite and nonnegative')
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
        from ..train.optim import named_service_optimizer
        from ..train.loop import capture_training_rng_state, accumulate_gradients, commit_optimizer_step
        named = [(param_name,parameters[param_name]) for param_name in sorted(names)]
        optimizer = named_service_optimizer(named,name=name,learning_rate=lr,weight_decay=weight_decay,
            beta1=beta1,beta2=beta2,eps=eps,momentum=momentum)
        gradients = []
        for param_name,parameter in named:
            grad = gradient.tensors[param_name].to(device=parameter.device,dtype=parameter.dtype)
            if grad.shape != parameter.shape or not torch.isfinite(grad).all():
                raise ValueError(f'gradient shape or values are invalid for {param_name}')
            gradients.append(grad)
            if name == 'adam' and 'm.'+param_name in state_tensors:
                values = {'exp_avg':state_tensors['m.'+param_name], 'exp_avg_sq':state_tensors['v.'+param_name]}
                if any(value.shape != parameter.shape for value in values.values()):
                    raise ValueError(f'optimizer state shape differs for {param_name}')
                optimizer.state[parameter] = {key:value.to(parameter).clone() for key,value in values.items()}
                optimizer.state[parameter]['step'] = state_tensors.get('step.'+param_name,
                    torch.tensor([prior_step],dtype=torch.float32)).reshape(()).cpu().clone()
            if name == 'sgd' and 'momentum.'+param_name in state_tensors:
                value = state_tensors['momentum.'+param_name]
                if value.shape != parameter.shape:
                    raise ValueError(f'optimizer state shape differs for {param_name}')
                optimizer.state[parameter]['momentum_buffer'] = value.to(parameter).clone()
        accumulate_gradients([value for _,value in named],gradients)
        commit_optimizer_step(optimizer)
        # Retain omitted parameter state, including its own step counter; a
        # later request can return to it without silently restarting Adam.
        next_state = {key:value.detach().cpu().clone() for key,value in state_tensors.items()}
        next_state['optimizer.step'] = torch.tensor([step],dtype=torch.float64)
        for param_name,parameter in named:
            values = optimizer.state.get(parameter,{})
            if name == 'adam':
                next_state['m.'+param_name] = values['exp_avg'].detach().cpu()
                next_state['v.'+param_name] = values['exp_avg_sq'].detach().cpu()
                next_state['step.'+param_name] = values['step'].detach().cpu().reshape(1)
            elif 'momentum_buffer' in values:
                next_state['momentum.'+param_name] = values['momentum_buffer'].detach().cpu()
        device = next(updated_model.parameters()).device
        prior_training = getattr(engine,'native_training_state',{})
        rng = prior_training.get('rng') or capture_training_rng_state(device)
        from ..train.native_continuation import collect_sft_progress
        training = {**{key:value for key,value in prior_training.items() if key.startswith('service_')},
                    'sft_progress':collect_sft_progress(prior_training),
                    'global_step':int(prior_training.get('global_step',prior_training.get('step',0)))+1,
                    'rng':rng, 'service_rng_policy':'training-stream-unchanged; chat-stream-per-request',
                    'service_optimizer_step':step,
                    'service_parent_checkpoint_sha256':identity['checkpoint_sha256']}
        receipt = write_native_composite_checkpoint(engine.native_checkpoint,
            native_update_directory(engine.native_checkpoint), updated_model,
            engine.model_config, base_identity=identity, source_gradient_id=gradient_id,
            optimizer={'name':name,'learning_rate':lr,'weight_decay':weight_decay,
                'beta1':beta1,'beta2':beta2,'eps':eps,'momentum':momentum,'step':step},
            optimizer_tensors=next_state,updated_parameter_names=sorted(names),parent_model=engine.shared_model,
            participant=(getattr(engine,'rl_policy_binding',{}) or {}).get('native_participant'),
            training_state=training,
            update_provenance={'objective':gradient.provenance.get('objective'),
                'gradient_artifact_id':gradient_id,'updated_parameter_names':sorted(names),
                'optimizer_state_parent_id':body.get('optimizer_state_id')})
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
