"""Native graph gradients returned as identity-bound named tensor artifacts.

The request is a graph, not a legacy projected-value GradSession. Graph
execution and loss are delegated to the shared native graph objective; this
module only binds exact stored imports, selects real trainable tensors, and
durably names the resulting gradients.
"""
from __future__ import annotations

from dataclasses import replace

import torch

from ..common.hashing import canonical_json_sha256_hex
from ..data.native_graph import graph_loss, parse_graph
from ..model.native_artifact import NativeTensorArtifact
from .chat import RequestError


def _identity(engine):
    return {'checkpoint_sha256': engine.checkpoint_sha256,
            'tokenizer_fingerprint': engine.model_config['tokenizer_fingerprint'],
            'embedding_weight_sha256': engine.embedding_weight_sha256,
            'architecture': engine.model_config['architecture']}


def _leaf_block(block):
    """Clone each imported floating tensor as a leaf while preserving IDs."""
    layers = []
    leaves = {}
    for index, layer in enumerate(block.layers):
        copied = dict(layer)
        for name in ('raw_k', 'v', 'gated_inputs'):
            if name in copied:
                value = copied[name].detach().clone().requires_grad_(True)
                copied[name] = value
                leaves[f'layers.{index}.{name}'] = value
        layers.append(copied)
    # Imported embeddings/token IDs/positions are provenance and logical input
    # metadata; raw KV and hybrid sequence tensors are the consumed state.
    return replace(block, layers=tuple(layers)), leaves


def _parameter_targets(model, requested):
    available = dict(model.named_parameters())
    if not isinstance(requested, list) or not requested or any(not isinstance(name, str) for name in requested):
        raise RequestError('neuralese-native-grad-target', 'parameter gradients require explicit trainable parameter names')
    if len(set(requested)) != len(requested) or any(name not in available for name in requested):
        raise RequestError('neuralese-native-grad-target', 'requested names must uniquely select native model parameters')
    return {name: available[name] for name in requested}


def native_graph_gradient(engine, body, owner):
    """Evaluate one actual native graph and return a durable named gradient.

    `target` selects either named native model parameters or one exact imported
    NativeStateBlock. No implicit parameter inventory or lossy block flattening
    is allowed, so callers can pair the result with a declared optimizer/update.
    """
    if body.get('schema') != 'natlang.native-gradient-request/1':
        raise RequestError('neuralese-native-grad-schema', 'native graph gradients require schema natlang.native-gradient-request/1')
    if body.get('objective', 'graph-ce') != 'graph-ce':
        raise RequestError('neuralese-native-grad-objective', 'only the shared native graph CE objective is available on this endpoint')
    row = body.get('graph')
    if not isinstance(row, dict):
        raise RequestError('neuralese-native-grad-graph', 'request must include one complete native graph row')
    restore_grad_flags = {}
    try:
        example = parse_graph(row, tokenizer_fingerprint=engine.model_config['tokenizer_fingerprint'])
        if example.split != 'train':
            raise ValueError('native graph parameter learning accepts train rows only')
        graph_sha256 = canonical_json_sha256_hex(row)
        if body.get('graph_sha256') not in (None, graph_sha256):
            raise ValueError('declared graph digest differs from the supplied graph row')
        expected = _identity(engine)
        imported, leaf_map = {}, {}
        for name, binding in example.external_states.items():
            record = engine.store.get_native_state(binding['id'])
            if record is None:
                raise ValueError(f'unknown external native state {binding["id"]}')
            block = record.block
            if len(block.token_ids) != binding['token_count']:
                raise ValueError(f'external state token count changed for {name}')
            block.require_native(**expected)
            imported[name], leaves = _leaf_block(block)
            leaf_map.update({f'external.{name}.{key}': value for key, value in leaves.items()})

        target = body.get('target')
        if not isinstance(target, dict):
            raise ValueError('target must identify model parameters or one imported native state')
        kind = target.get('kind')
        if kind == 'model-parameters':
            selected = _parameter_targets(engine.shared_model, target.get('names'))
            # Serving factories freeze the model for generation. Enable only
            # the explicitly named gradient targets during this locked call.
            restore_grad_flags = {parameter: parameter.requires_grad for parameter in selected.values()}
            for parameter in restore_grad_flags:
                parameter.requires_grad_(True)
            target_tensors = selected
            artifact_kind = 'parameter-gradient'
            source_id = None
        elif kind == 'native-state':
            state_id = target.get('id')
            if not isinstance(state_id, str):
                raise ValueError('native-state target needs its exact stored state ID')
            matching = [name for name, binding in example.external_states.items() if binding['id'] == state_id]
            if len(matching) != 1:
                raise ValueError('native-state gradient target must occur exactly once as an imported graph state')
            prefix = f'external.{matching[0]}.'
            available = {name.removeprefix(prefix): value for name, value in leaf_map.items() if name.startswith(prefix)}
            requested = target.get('names', sorted(available))
            if (not isinstance(requested, list) or not requested or len(set(requested)) != len(requested)
                    or any(name not in available for name in requested)):
                raise ValueError('native-state gradient names must select consumed named KV or hybrid-state tensors')
            target_tensors = {name: available[name] for name in requested}
            artifact_kind = 'state-gradient'
            source_id = state_id
        else:
            raise ValueError('target.kind must be model-parameters or native-state')
        if not target_tensors:
            raise ValueError('gradient target has no tensors')
        loss, stats = graph_loss(engine.shared_model, example, external_blocks=imported,
                                 external_identity=expected)
        grads = torch.autograd.grad(loss, tuple(target_tensors.values()), allow_unused=True)
        tensors = {name: grad.detach() for name, grad in zip(target_tensors, grads) if grad is not None}
        if set(tensors) != set(target_tensors):
            absent = sorted(set(target_tensors) - set(tensors))
            raise ValueError(f'objective has no gradient path to requested tensors: {absent}')
        artifact = NativeTensorArtifact(artifact_kind, expected, tensors, source_id,
            {'objective': 'native-graph-token-ce/1', 'graph_id': example.id,
             'graph_sha256': graph_sha256, 'target_kind': kind,
             'target_names': sorted(tensors), 'loss': float(loss.detach()), 'stats': stats}).validate()
        stored = engine.store.put_native_tensor_artifact(artifact, owner)
        return {'loss': float(loss.detach()), 'stats': stats, 'gradient': stored.meta(),
                'native_identity': expected, 'objective': 'native-graph-token-ce/1'}
    except RequestError:
        raise
    except (ValueError, KeyError, TypeError, RuntimeError) as error:
        raise RequestError('neuralese-native-grad-invalid', str(error)) from error
    finally:
        for parameter, enabled in restore_grad_flags.items():
            parameter.requires_grad_(enabled)
