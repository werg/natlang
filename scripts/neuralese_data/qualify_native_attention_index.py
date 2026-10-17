#!/usr/bin/env python3
"""Compare the exact old/new attention-index path at a real saved parent."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from scripts.neuralese_data.qualify_native_causal_prefixes import load_qualification_model
from natlang_neuralese.common.checkpoint_offload import configure_checkpoint_input_offload
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.eval.native_memory import capacity_identity, qualify_capacity
from natlang_neuralese.model import shared_state
from natlang_neuralese.train.fp8_backend import make_backend
from natlang_neuralese.train.native_resident import binding, bound, code_inventory
from natlang_neuralese.train.native_state_sft import make_native_optimizer


def diagnostic_optimizer(model):
    """Reapply the same full latent-layer selection as production after loading."""
    for name, parameter in model.named_parameters():
        if name.startswith(('input_rows', 'output_rows', 'extra_input_rows', 'extra_output_rows', 'adapter.')):
            parameter.requires_grad_(True)
    optimizer, named = make_native_optimizer(model, lr=3e-6, latent_ns='cubic5',
        learned_row_lr=3e-4, interface_row_lr=3e-4)
    if not any(name.startswith('backbone.') for name, _ in named):
        raise ValueError('paired diagnostic omitted production backbone trainables')
    if any(not parameter.requires_grad for _, parameter in named):
        raise ValueError('paired diagnostic production trainables remain frozen')
    return optimizer


def gradient_name_aliases(model, optimizer):
    """Join current optimizer names and legacy observer aliases exactly."""
    canonical, owners = {}, {}
    for name, parameter in optimizer._native_named_parameters:
        if not parameter.requires_grad:
            continue
        identity = id(parameter)
        if ((name in owners and owners[name] != identity)
                or (identity in canonical and canonical[identity] != name)):
            raise ValueError('ambiguous diagnostic optimizer inventory: ' + name)
        owners[name], canonical[identity] = identity, name
    names = {name: name for name in canonical.values()}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) not in canonical:
            raise ValueError('diagnostic trainable absent from optimizer inventory: ' + name)
        actual = canonical[id(parameter)]
        if name in names and names[name] != actual:
            raise ValueError('ambiguous diagnostic gradient parameter name: ' + name)
        names[name] = actual
    return names


def tensor_digest(value):
    value = value.detach()
    if not value.is_contiguous():
        raise ValueError('paired index qualification requires contiguous exact tensor bytes')
    raw = value.view(torch.uint8).reshape(-1)
    digest = hashlib.sha256()
    for chunk in raw.split(64 << 20):
        digest.update(memoryview(chunk.cpu().numpy()))
    return {'dtype': str(value.dtype), 'shape': list(value.shape), 'sha256': digest.hexdigest()}


def reference_module(old_package, new_package):
    old, new = code_inventory(old_package), code_inventory(new_package)
    changed = {name for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    if changed != {'model/shared_state.py'}:
        raise ValueError('paired index reference changed another model/backend/source module')
    spec = importlib.util.spec_from_file_location('natlang_neuralese.model._index_old_reference',
        Path(old_package)/'model/shared_state.py')
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module  # dataclasses resolve postponed annotations through the registered module.
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise
    return module, old, new


def paired_case(model, optimizer, example, reference, *, backend=None, checkpoint_layers=True,
                chunk_tokens=1024, checkpoint_sha256, source_binding, trace=None):
    """Existing zero-update capacity machinery owns every execution/state guard."""
    candidate = shared_state.forward_graph
    trainable = {name: {'shape': list(parameter.shape), 'dtype': str(parameter.dtype)}
                 for name, parameter in optimizer._native_named_parameters if parameter.requires_grad}
    names = gradient_name_aliases(model, optimizer)
    if not any(name.startswith('backbone.') for name in trainable):
        raise ValueError('paired diagnostic needs production backbone gradients')
    values, hashes, outputs, geometries = {}, {}, {}, {}
    try:
        for label, implementation in [('old', reference.forward_graph), ('new', candidate)]:
            shared_state.forward_graph = implementation
            gradients, observed = {}, []
            def result(module, args, output):
                observed.append(tensor_digest(output['h_final']))
                plan = args[0]
                geometries[label] = {'forward_tokens': plan.token_count, 'kv_slots': len(plan.slot_sources)}
            hook = model.register_forward_hook(result)
            try:
                values[label] = qualify_capacity(model, optimizer, [example], backend=backend,
                    checkpoint_layers=checkpoint_layers, chunk_tokens=chunk_tokens,
                    checkpoint_sha256=checkpoint_sha256, source_binding=source_binding,
                    gradient_observer=lambda name, value: gradients.update({names[name]: tensor_digest(value)}))
            finally:
                hook.remove()
            if len(observed) != 1:
                raise ValueError('paired index path must execute exactly one explicit graph forward')
            hashes[label], outputs[label] = gradients, observed[0]
            if not any(name.startswith('backbone.') for name in gradients):
                raise ValueError('paired backward observed no production backbone adjoints')
        passed = (all(value['passed'] for value in values.values()) and hashes['old'] == hashes['new']
            and outputs['old'] == outputs['new'] and geometries['old'] == geometries['new']
            and values['old']['cases'][0]['loss'] == values['new']['cases'][0]['loss'])
        if not passed:
            return {'passed': False, 'parity': values, 'gradient_names': sorted(hashes['old'])}
        profiler = (torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False)
                    if trace is not None else contextlib.nullcontext())
        timings = {}
        with profiler as active:
            for label, implementation in [('old', reference.forward_graph), ('new', candidate)]:
                shared_state.forward_graph = implementation
                if next(model.parameters()).is_cuda:
                    torch.cuda.synchronize()
                with torch.profiler.record_function('attention-index-' + label):
                    result = qualify_capacity(model, optimizer, [example], backend=backend,
                        checkpoint_layers=checkpoint_layers, chunk_tokens=chunk_tokens,
                        checkpoint_sha256=checkpoint_sha256, source_binding=source_binding)
                if not result['passed']:
                    raise ValueError('timed zero-update normal batch failed')
                timings[label] = result['cases'][0]['seconds']
        if trace is not None:
            active.export_chrome_trace(str(trace))
        return {'passed': True, 'batch': geometries['new'],
            'old_seconds': timings['old'], 'new_seconds': timings['new'],
            'outputs_bitwise': True, 'loss_bitwise': True, 'all_named_adjoints_bitwise': True,
            'gradient_parameters': len(hashes['old']), 'gradient_digest': canonical_json_sha256_hex(hashes['old']),
            'production_trainable_inventory': trainable,
            'observed_gradient_names': sorted(hashes['new']),
            'unobserved_trainable_names': sorted(trainable.keys() - hashes['new'].keys()),
            'optimizer_updates': 0, 'optimizer_state_unchanged': True, 'rng_restored': True,
            'source_credits_unchanged': True, 'cuda_synchronized': next(model.parameters()).is_cuda,
            'timing_scope': 'whole-graph forward/backward with existing finite-gradient checks and release; no SGD',
            'diagnostic_optimizer': 'fresh unarmed observer only; immutable training-parent optimizer untouched',
            'parity': values, **({'trace': binding(trace)} if trace is not None else {})}
    finally:
        shared_state.forward_graph = candidate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'checkpoint', 'source', 'old-package', 'new-package', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--capacity-certificate', required=True, help='exact JSON path/SHA binding')
    parser.add_argument('--model', choices=['lfm', 'mellum'], required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--native-participant')
    args = parser.parse_args(argv)
    certificate = json.loads(bound(json.loads(args.capacity_certificate)).read_text())
    identity = certificate['identity']; args.linear_precision = identity['model_config']['native_linear_precision']
    reference, old_code, new_code = reference_module(args.old_package, args.new_package)
    if Path(shared_state.__file__).resolve() != (args.new_package/'model/shared_state.py').resolve():
        raise ValueError('paired index candidate package is not the actually imported numerical runtime')
    if certificate['runtime_code_sha256'] != canonical_json_sha256_hex(old_code):
        raise ValueError('paired index source capacity belongs to another old package')
    model, tokenizer, roles, config, parent, options = load_qualification_model(args)
    backend = None
    if args.model == 'mellum':
        backend = make_backend(identity['fp8']['kind'], model.port.hf, device='cuda',
            recipe=identity['fp8']['recipe'], amax_history_len=identity['fp8']['amax_history_len'])
        if options.get('extra_state') is not None:
            backend.restore_extra_state(options['extra_state'])
        model.port.activation_checkpoint = backend.checkpoint
    optimizer = diagnostic_optimizer(model)
    configure_checkpoint_input_offload(model.port, enabled=identity['checkpoint_input_offload'])
    model.share_causal_prefixes = identity.get('share_causal_prefixes', False)
    actual_args = argparse.Namespace(**{key: identity[key] for key in
        ('batch_graphs', 'batch_tokens', 'checkpoint_layers', 'chunk_tokens', 'checkpoint_input_offload')},
        device='cuda', share_causal_prefixes=model.share_causal_prefixes,
        fp8_training=identity['fp8']['kind'], fp8_recipe=identity['fp8']['recipe'],
        fp8_amax_history_len=identity['fp8']['amax_history_len'])
    if capacity_identity(config, actual_args, backend) != identity:
        raise ValueError('actual parent/backend/geometry differs from reviewed capacity identity')
    source = NativeGraphSource(args.source, required_split='train', tokenizer_fingerprint=config['tokenizer_fingerprint'],
        role_ids=config['marker_ids'])
    example, = source.iter_indices([args.index])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = paired_case(model, optimizer, example, reference, backend=backend,
        checkpoint_layers=identity['checkpoint_layers'], chunk_tokens=identity['chunk_tokens'],
        checkpoint_sha256=parent['sha256'], source_binding=binding(args.source),
        trace=args.output.with_suffix('.trace.json'))
    result.update(schema='natlang.attention-index-production-paired/1', identity=identity,
        old_runtime_code_sha256=canonical_json_sha256_hex(old_code),
        new_runtime_code_sha256=canonical_json_sha256_hex(new_code),
        parent_checkpoint_sha256=parent['sha256'], source=binding(args.source),
        source_row={key: source.entries[args.index][key] for key in ('id', 'binding')},
        entrypoint=binding(Path(__file__)), state_scope='separate diagnostic; parent weights/optimizer/RNG/exposure files untouched')
    write_json_durable(args.output, result)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
