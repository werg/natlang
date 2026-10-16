"""Shared full-state supervised training for positional-KV/hybrid-state graphs.

The declared recipe supplies admitted, role-aware native graph JSONL. This
stage trains the exact model used by the native serving path; legacy projected
records are rejected by the shared graph parser.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable as write_json
from ..model.native_model import (CHECKPOINT_SCHEMA, load_native_checkpoint, load_native_model, model_state,
                                  native_identity)
from .checkpoint_policy import CheckpointPolicy
from .loop import capture_training_rng_state, restore_training_rng_state
from .native_source_streams import scheduling_rank, stream_pending_batches
from .native_continuation import (collect_sft_progress, restore_sft_progress, progress_obligations,
    communication_foundation, raw_communication_rows)

from ..data.native_sources import NativeGraphSource, _jsonl, _row_bindings, _require_communication_graph

TRAINING_SCHEMA = 'natlang.neuralese-native-sft-training/1'


def _load_examples(path, tokenizer_fingerprint, *, required_split):
    from ..data.native_graph import parse_graph
    examples, ids = [], set()
    for row in _jsonl(path):
        example = parse_graph(row, tokenizer_fingerprint=tokenizer_fingerprint)
        if example.split != required_split:
            raise ValueError(f'{path}: expected {required_split} rows, got {example.split!r} in {example.id}')
        if example.id in ids:
            raise ValueError(f'{path}: duplicate example id {example.id!r}')
        ids.add(example.id)
        examples.append(example)
    if not examples:
        raise ValueError(f'{path}: no {required_split} native graph rows')
    return examples


def communication_exposure(entries, counts, quotas):
    selected = [entry for entry in entries if entry.get('communication')]
    return {'graphs': len(selected),
            'required_graph_visits': sum(quotas[entry['id']] for entry in selected),
            'graph_visits': sum(counts.get(entry['id'], 0) for entry in selected),
            'complete': bool(selected) and all(counts.get(entry['id'], 0) >= quotas[entry['id']] > 0
                                              for entry in selected)}


def batch_boundaries(entries, *, batch_graphs=1, batch_tokens=0):
    """Sequential whole-graph batches; a long graph remains one intact batch."""
    if batch_graphs < 1 or batch_tokens < 0:
        raise ValueError('native batching requires positive graph count and nonnegative token budget')
    boundaries, start, tokens = [], 0, 0
    for index, entry in enumerate(entries):
        size = entry['forward_tokens']
        if index > start and ((batch_tokens and tokens + size > batch_tokens)
                              or (not batch_tokens and index-start >= batch_graphs)):
            boundaries.append((start, index)); start, tokens = index, 0
        tokens += size
    if start < len(entries):
        boundaries.append((start, len(entries)))
    return boundaries


def pending_exposure_batches(entries, exposure_counts, *, epochs, batch_graphs=1, batch_tokens=0):
    """Schedule only missing row visits, including rows appended after completion.

    Communication starts first and alternates with explicitly prioritized
    ordinary data; priority-zero ordinary data follows both streams. Preserve
    source order within each class. Completed visits remain credits across
    changed packing and appended sources; fractional epochs cover the declared
    source prefix. The checkpoint's actual row counts determine what remains.
    """
    whole = int(epochs)
    fractional = math.ceil((epochs - whole) * len(entries))
    quotas = {entry['id']: max(int(bool(entry.get('communication'))), whole + (index < fractional))
              for index, entry in enumerate(entries)}
    batches = []
    pending = {visit: [index for index, entry in enumerate(entries)
                       if exposure_counts.get(entry['id'], 0) <= visit < quotas[entry['id']]]
               for visit in range(math.ceil(epochs))}
    for rank in sorted({scheduling_rank(entry) for entry in entries}):
        for visit, remaining in pending.items():
            indices = [index for index in remaining if scheduling_rank(entries[index]) == rank]
            selected = [entries[index] for index in indices]
            boundaries = batch_boundaries(selected, batch_graphs=batch_graphs, batch_tokens=batch_tokens)
            for left, right in boundaries:
                last = right == len(indices) and all(scheduling_rank(entries[index]) <= rank for index in remaining)
                batches.append((visit, indices[left:right], last))
    # A positive source priority must take effect while a large communication
    # cohort is still pending, rather than waiting for its entire completion.
    # Only reorder already packed batches: quotas and authenticated credits do
    # not depend on this policy, and no row is replayed or omitted.
    communication, urgent, ordinary = [], [], []
    for batch in batches:
        entry = entries[batch[1][0]]
        destination = (communication if entry.get('communication') else
                       urgent if entry.get('training_priority', 0) > 0 else ordinary)
        destination.append(batch)
    if urgent and communication:
        interleaved = []
        for index in range(max(len(communication), len(urgent))):
            if index < len(communication):
                interleaved.append(communication[index])
            if index < len(urgent):
                interleaved.append(urgent[index])
        batches = interleaved + ordinary
        last_by_visit = {visit: index for index, (visit, _, _) in enumerate(batches)}
        batches = [(visit, indices, last_by_visit[visit] == index)
                   for index, (visit, indices, _) in enumerate(batches)]
    return quotas, batches


def make_native_optimizer(model, *, lr, weight_decay=.1, latent_ns='quintic', in_backward=True, learned_row_lr=None, interface_row_lr=None):
    """Reuse the production port optimizer and its in-backward MuonSR memory path."""
    from .backbone_policy import configure_backbone_training
    from .optim import PortMuonAdamW, latent_entries, latent_options, latent_split

    backbone_named = [('backbone.' + name, parameter)
                      for name, parameter in configure_backbone_training(model.port, 'latent')]
    # The native role rows and embedding-space adapter are learned parts of the
    # native interface. The frozen vocabulary/readout and final norm follow the
    # existing shared full-layer backbone policy.
    interface_named = [('interface.' + name, parameter) for name, parameter in model.named_parameters()
                       if name.startswith(('input_rows', 'output_rows', 'extra_input_rows', 'extra_output_rows', 'adapter.')) and parameter.requires_grad]
    named = backbone_named + interface_named
    latent, ordinary = latent_split(named, 'muonsr')
    if not latent:
        raise ValueError('native SFT found no MuonSR backbone/interface matrices')
    # Register finite-gradient checks before the optimizer's post-accumulate
    # hooks. MuonSR updates a matrix as soon as its gradient is complete.
    if in_backward:
        for name, parameter in latent:
            def finite_before_muon(value, name=name):
                if value.grad is None:
                    return
                finite = torch.isfinite(value.grad).all()
                if value.grad.device.type == 'cuda' and hasattr(torch, '_assert_async'):
                    # Keep the per-matrix hook off the host synchronization path
                    # on large dense/MoE models; this precedes update kernels.
                    torch._assert_async(finite, f'nonfinite native gradient in {name}')
                elif not bool(finite):
                    _raise_nonfinite(name)
            parameter.register_post_accumulate_grad_hook(finite_before_muon)
    optimizer = PortMuonAdamW(ordinary, lr=lr, vocab_size=int(model.port.embedding_weight.shape[0]),
        embedding_ids={id(model.port.embedding_weight)},
        latent=latent_entries(latent, 'muonsr', lr), latent_optimizer='muonsr',
        latent_options=latent_options('muonsr', weight_decay=weight_decay, ns=latent_ns))
    optimizer._native_named_parameters = named
    from .native_learning_rates import apply_native_learning_rates
    apply_native_learning_rates(optimizer,backbone_lr=lr,learned_row_lr=learned_row_lr,interface_row_lr=interface_row_lr)
    if in_backward and optimizer.latent is not None:
        optimizer.latent.step_in_backward(gated=True)
    return optimizer, named


def _raise_nonfinite(name):
    raise FloatingPointError(f'nonfinite native gradient in {name}; update was not committed')


def _source_receipt(path, examples):
    counts = {}
    for example in examples:
        for group in example.source_groups:
            value = counts.setdefault(group, {'examples': 0, 'supervised_tokens': 0,
                                               'supervised_weight': 0.})
            value['examples'] += 1
            value['supervised_tokens'] += sum(weight > 0 for weight in example.target_weights)
            value['supervised_weight'] += sum(example.target_weights)
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file_hex_cached(path),
            'rows': len(examples), 'by_source_group': counts}


def _source_hash(examples):
    digest = hashlib.sha256()
    for item in examples:
        row = {'id': item.id, 'groups': item.source_groups, 'split': item.split,
               'tokenizer': item.tokenizer_fingerprint, 'targets': item.target_weights,
               'nodes': [{'name': node.name, 'token_ids': node.token_ids,
                          'imports': [edge.__dict__ for edge in node.imports],
                          'producing': node.producing, 'logical_producing': node.logical_producing}
                         for node in item.plan.invocations]}
        value = json.dumps(row, sort_keys=True, separators=(',', ':')).encode()
        digest.update(len(value).to_bytes(8, 'big'))
        digest.update(value)
    return digest.hexdigest()


def pack_native_graphs(examples):
    """Join disjoint examples into one graph plan so shared weights update once."""
    from ..model.shared_state import Invocation, ImportSpan, compile_graph
    invocations, weights, ownership, external_sizes, external_name_maps = [], [], [], {}, []
    for graph_index, example in enumerate(examples):
        names = {node.name: f'b{graph_index}/{node.name}' for node in example.plan.invocations}
        external_names = {name: f'b{graph_index}/external/{name}' for name in example.plan.external_names}
        names.update(external_names)
        external_name_maps.append(external_names)
        external_sizes.update({names[name]: size for name, size in
                               zip(example.plan.external_names, example.plan.external_sizes)})
        for node in example.plan.invocations:
            imports = tuple(ImportSpan(source=names[edge.source], start=edge.start,
                                       stop=edge.stop, at=edge.at, logical=edge.logical)
                            for edge in node.imports)
            invocations.append(Invocation(name=names[node.name], token_ids=node.token_ids,
                imports=imports, producing=node.producing,
                logical_producing=node.logical_producing))
        weights.extend(example.target_weights)
        ownership.append((len(invocations) - len(example.plan.invocations), len(invocations)))
    packed = compile_graph(invocations, external_sizes=external_sizes)
    return packed, tuple(weights), tuple(ownership), tuple(external_name_maps)


def native_training_update(model, optimizer, batch, *, backend=None, checkpoint_layers=True,
                           chunk_tokens=128, objective=None):
    """One shared complete production update, also used by throughput diagnostics."""
    if backend is not None:
        backend.reset_row_counts()
    optimizer.zero_grad(set_to_none=True)
    sparse_before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()
                     if name in ('input_rows', 'output_rows', 'extra_input_rows', 'extra_output_rows')
                     and parameter.requires_grad}
    if optimizer.latent is not None:
        context = optimizer.latent.in_backward()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    try:
        with context:
            if backend is not None:
                with backend.autocast():
                    loss, details = packed_graph_loss(model, batch,
                        checkpoint_layers=checkpoint_layers, chunk_tokens=chunk_tokens,
                        objective=objective, share_causal_prefixes=getattr(model, 'share_causal_prefixes', False))
            else:
                loss, details = packed_graph_loss(model, batch,
                    checkpoint_layers=checkpoint_layers, chunk_tokens=chunk_tokens,
                    objective=objective, share_causal_prefixes=getattr(model, 'share_causal_prefixes', False))
            loss.backward()
        if optimizer.latent is not None:
            optimizer.latent._armed = None
        params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if params:
            norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite native gradient norm')
        optimizer.step()
        model.native_sparse_updates = sparse_row_updates(model, sparse_before)
    except Exception:
        if optimizer.latent is not None:
            optimizer.latent._armed = None
        optimizer.zero_grad(set_to_none=True)
        raise
    return loss, details


@torch.no_grad()
def sparse_row_updates(model, before):
    """Report actual gradient and committed update per learned vocabulary row.

    These tiny tables are the only snapshots. No vocabulary-sized copy, RNG
    draw, extra forward, optimizer action or training-policy decision occurs.
    """
    result = {}
    for name, previous in before.items():
        parameter = getattr(model, name)
        old, current = previous.float(), parameter.detach().float()
        gradient = parameter.grad
        gradient_present = gradient is not None
        gradient = torch.zeros_like(current) if gradient is None else gradient.detach().float()
        delta = current - old
        values = torch.stack((gradient.norm(dim=-1), delta.norm(dim=-1),
                              current.norm(dim=-1), (current != old).sum(dim=-1).float(),
                              (gradient * delta).sum(dim=-1)), dim=-1).cpu().tolist()
        ids = model.token_override_ids if name.startswith('extra_') else model.marker_ids
        result[name] = {'dtype': str(parameter.dtype), 'rows': [
            {'token_id': token, 'gradient_l2': row[0], 'update_l2': row[1],
             'gradient_present': gradient_present,
             'gradient_update_dot': row[4] if gradient_present else None,
             'gradient_update_cosine': row[4] / (row[0] * row[1])
                 if gradient_present and row[0] > 0 and row[1] > 0 else None,
             'weight_l2': row[2], 'changed_elements': int(row[3]), 'elements': current.shape[-1]}
            for token, row in zip(ids, values)]}
    return result


def packed_graph_loss(model, examples, *, checkpoint_layers=True, chunk_tokens=128, objective=None,
                      share_causal_prefixes=False):
    """One backbone graph forward for a minibatch; mean the original per-graph losses."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    from .native_prefix_packing import pack_causal_prefixes
    shared = pack_causal_prefixes(examples) if share_causal_prefixes and objective is None else None
    if shared is None:
        plan, target_weights, ownership, external_name_maps = pack_native_graphs(examples)
        physical_rows = tuple(tuple(range(plan.offsets[first],
            plan.offsets[last-1]+len(plan.invocations[last-1].token_ids))) for first,last in ownership)
    else:
        plan, physical_rows = shared
    model.port.checkpoint_layers = checkpoint_layers
    if objective is not None:
        row_offsets = {example.id: {'node_start': first_node, 'node_stop': last_node,
        'physical_start': plan.offsets[first_node],
        'physical_stop': plan.offsets[last_node - 1] + len(plan.invocations[last_node - 1].token_ids),
        'external_names': external_name_maps[index]}
        for index, (example, (first_node, last_node)) in enumerate(zip(examples, ownership))}
        context = {'plan': plan, 'examples': examples, 'target_weights': target_weights,
               'ownership': ownership, 'row_offsets': row_offsets,
               'external_names': {example.id: external_name_maps[index]
                                  for index, example in enumerate(examples)},
               'external_states': {example.id: getattr(example, 'external_states', {})
                                   for example in examples}}
        forward_inputs = objective.forward_inputs(context)
    else:
        forward_inputs = {}
    result = model(plan, retain_states=False, **(forward_inputs or {}))
    if objective is not None:
        loss, details = objective(model, result, context)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss.detach()):
            raise FloatingPointError('native policy objective must return a finite scalar loss')
        return loss, details
    hidden = result['h_final'][0]
    losses, details = [], []
    boundary_ids = set((*getattr(model, 'marker_ids', ()), *getattr(model, 'token_override_ids', ())))
    for graph_index, example in enumerate(examples):
        queries, targets, weights = [], [], []
        boundary_targets = {str(token): {'count': 0, 'weight': 0.} for token in sorted(boundary_ids)}
        for node, offset in zip(example.plan.invocations, example.plan.offsets):
            for local in range(1, len(node.token_ids)):
                target = offset + local
                weight = example.target_weights[target]
                if weight > 0:
                    queries.append(physical_rows[graph_index][target - 1])
                    token = int(node.token_ids[local])
                    targets.append(token)
                    weights.append(weight)
                    if token in boundary_ids:
                        boundary_targets[str(token)]['count'] += 1
                        boundary_targets[str(token)]['weight'] += float(weight)
        if not queries:
            raise ValueError(f'native graph {examples[graph_index].id!r} has no supervised target')
        query_ids = torch.tensor(queries, device=hidden.device)
        target_ids = torch.tensor(targets, device=hidden.device)
        token_weights = torch.tensor(weights, device=hidden.device, dtype=torch.float32)
        total = hidden.new_zeros((), dtype=torch.float32)
        correct = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(queries), chunk_tokens):
            stop = start + chunk_tokens
            def loss_chunk(states, labels, selected):
                logits = model.logits(states)
                ce = F.cross_entropy(logits.float(), labels, reduction='none')
                return ((ce * selected).sum(),
                        ((logits.argmax(-1) == labels).float() * selected).sum())
            chunk_loss, chunk_correct = checkpoint(loss_chunk, hidden[query_ids[start:stop]],
                target_ids[start:stop], token_weights[start:stop], use_reentrant=False)
            total = total + chunk_loss
            correct = correct + chunk_correct
        denominator = token_weights.sum()
        losses.append(total / denominator)
        details.append({'weight': float(denominator.detach()),
                        'boundary_targets': boundary_targets,
                        'supervised_tokens': len(queries),
                        'packed_physical_tokens': plan.token_count,
                        'original_physical_tokens': sum(item.plan.token_count for item in examples),
                        'accuracy': float((correct / denominator).detach())})
    return torch.stack(losses).mean(), details


def append_only_rows(previous, current, *, label):
    """Prove a continuation retained every prior row byte-semantically and only added rows."""
    if not previous:
        raise ValueError(f'native SFT checkpoint has no exact {label} row bindings')
    changed = [key for key, value in previous.items() if current.get(key) != value]
    if changed:
        raise ValueError(f'native SFT {label} continuation modified/removed prior rows: ' + ', '.join(changed[:8]))
    return set(current) - set(previous)


def scheduled_indices(examples, *, seed, epoch, prior_ids=None):
    """Stable epoch order; on append, preserve the old permutation as an exact prefix."""
    order = sorted(range(len(examples)), key=lambda i: hashlib.sha256(
        f'{seed}:{epoch}:{examples[i].id}'.encode()).digest())
    if prior_ids is None:
        return order
    old = [index for index in order if examples[index].id in prior_ids]
    new = [index for index in order if examples[index].id not in prior_ids]
    return old + new


def _evaluate(model, examples, *, checkpoint_layers, chunk_tokens, backend=None, graph_classes=None):
    from ..data.native_graph import graph_loss
    previous_mode = model.training
    model.eval()
    scope = backend.native_modules() if backend is not None else contextlib.nullcontext()
    boundary_ids = tuple(dict.fromkeys((*getattr(model, 'marker_ids', ()),
                                        *getattr(model, 'token_override_ids', ()))))
    boundary_sums = {str(token): dict.fromkeys(
        ('count', 'weight', 'nll_sum', 'correct_weight', 'probability_sum'), 0.)
        for token in boundary_ids}
    classes = {}
    try:
        with scope, torch.no_grad():
            total_weight, total_loss, total_correct, total_tokens, total_graphs, forward_tokens = 0., 0., 0., 0, 0, 0
            for example in examples:
                loss, detail = graph_loss(model, example, checkpoint_layers=checkpoint_layers,
                                          chunk_tokens=chunk_tokens, boundary_token_ids=boundary_ids)
                for token, sums in detail['boundary_sums'].items():
                    for key, value in sums.items():
                        boundary_sums[token][key] += value
                weight = detail['weight']
                if graph_classes is not None:
                    name = graph_classes[example.id]
                    sums = classes.setdefault(name, {'loss_sum': 0., 'correct_sum': 0., 'weight': 0.,
                        'graphs': 0, 'supervised_tokens': 0, 'forward_tokens': 0,
                        'boundaries': {token: dict.fromkeys(values, 0.) for token, values in boundary_sums.items()}})
                    sums['loss_sum'] += float(loss) * weight
                    sums['correct_sum'] += detail['accuracy'] * weight
                    sums['weight'] += weight
                    sums['graphs'] += 1
                    sums['supervised_tokens'] += detail['supervised_tokens']
                    sums['forward_tokens'] += example.plan.token_count
                    for token, values in detail['boundary_sums'].items():
                        for key, value in values.items():
                            sums['boundaries'][token][key] += value
                total_loss += float(loss) * weight
                total_correct += detail['accuracy'] * weight
                total_weight += weight
                total_tokens += detail['supervised_tokens']
                total_graphs += 1
                forward_tokens += example.plan.token_count
    finally:
        model.train(previous_mode)
    if not math.isfinite(total_loss) or total_weight <= 0:
        raise FloatingPointError('native held evaluation is nonfinite or empty')
    def summarize_boundaries(values):
        return {token: {**sums, 'count': int(sums['count']),
        'mean_nll': sums['nll_sum']/sums['weight'] if sums['weight'] else None,
        'weighted_accuracy': sums['correct_weight']/sums['weight'] if sums['weight'] else None,
        'mean_probability': sums['probability_sum']/sums['weight'] if sums['weight'] else None}
        for token, sums in values.items()}
    boundaries = summarize_boundaries(boundary_sums)
    return {'loss': total_loss / total_weight, 'weighted_accuracy': total_correct / total_weight,
            'weight': total_weight, 'supervised_tokens': total_tokens, 'examples': total_graphs,
            'forward_tokens': forward_tokens, 'boundaries': boundaries,
            'classes': {name: {**sums, 'boundaries': summarize_boundaries(sums['boundaries']), 'loss': sums['loss_sum']/sums['weight'] if sums['weight'] else None,
                'weighted_accuracy': sums['correct_sum']/sums['weight'] if sums['weight'] else None}
                for name, sums in classes.items()}}


def held_cadence_indices(entries, cursor=0, *, token_budget=0, graph_budget=0):
    """Compatibility view of the shared selector's single-class rotation."""
    from .native_held import held_selection
    if cursor < 0 or not entries:
        raise ValueError('held cadence needs nonempty data and nonnegative cursor')
    position = cursor % len(entries)
    coverage = {'cursor': position, 'completed_sweeps': cursor//len(entries),
                'graph_visits': cursor}
    indices, state, _ = held_selection([{**e, 'communication': False} for e in entries],
        coverage, source_sha256=None, token_budget=token_budget, graph_budget=graph_budget)
    return sorted(indices, key=lambda i: (i-position) % len(entries)), state['cursor']


def _evaluate_cadence(model, source, *, args, coverage=None, final=False, backend=None,
                      previous_result=None):
    from .native_held import held_selection
    coverage = coverage or {}
    indices, next_coverage, selection = held_selection(source.entries, coverage,
        source_sha256=source.file_sha256, token_budget=args.held_eval_tokens,
        graph_budget=args.held_eval_graphs, final=final, previous_result=previous_result)
    cursor = next_coverage['cursor']
    started = time.monotonic()
    result = _evaluate(model, source.iter_indices(indices), checkpoint_layers=args.checkpoint_layers,
                       chunk_tokens=args.chunk_tokens, backend=backend,
                       graph_classes={source.entries[i]['id']: ('communication' if source.entries[i].get('communication')
                                      else 'ordinary') for i in indices})
    result.update(scope='full' if len(indices) == len(source) else 'cadence-subset',
                  source_sha256=source.file_sha256,
                  source_graphs=len(source), cursor_before=int(coverage.get('cursor', 0)),
                  cursor_after=cursor, seconds=time.monotonic()-started,
                  selection=selection)
    return result, next_coverage


def _save(policy, model, optimizer, *, config, identity, step, epoch, cursor, source_receipts,
          source_hash, source_row_bindings, held_row_bindings, exposure_counts, global_step, rng, metrics,
          architecture, options, fp8_state=None, sft_progress=None):
    state = {'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model), 'step': global_step,
             'training': {'schema': TRAINING_SCHEMA, 'identity': identity, 'step': step,
                          'global_step': global_step, 'epoch': epoch,
                          'source_row_bindings': source_row_bindings,
                          'source_row_order': options.get('_source_row_order', []),
                          'held_row_bindings': held_row_bindings,
                          'exposure_counts': exposure_counts,
                          'cursor': cursor, 'source_hash': source_hash, 'source_receipts': source_receipts,
                          'optimizer': optimizer.state_dict(), 'rng': rng, 'metrics': metrics,
                          'options': options, 'fp8': fp8_state, 'sft_progress': sft_progress or {}}}
    state['training']['sft_progress'] = collect_sft_progress(state['training'])
    if sft_progress is not None:
        sft_progress.clear()
        sft_progress.update(state['training']['sft_progress'])
    policy.save(state, metric=metrics.get('held', {}).get('loss'))


def qualify_precision_transition(model, tokenizer, roles, config, *, previous_precision,
                                 checkpoint_sha256, forward_atol, gradient_atol, device, backend=None):
    """Check changed production arithmetic on the restored live weights, once."""
    if config.get('native_linear_precision', 'ieee') != 'bf16x6' or previous_precision == 'bf16x6':
        return None
    if forward_atol is None or gradient_atol is None:
        raise ValueError('native arithmetic transition requires declared forward and gradient tolerances')
    from ..eval.native_state import qualify_existing
    saved_rng = capture_training_rng_state(device)
    try:
        with backend.native_modules() if backend is not None else contextlib.nullcontext():
            report = qualify_existing(model, tokenizer, roles, config,
                checkpoint_sha256=checkpoint_sha256, forward_atol=forward_atol, gradient_atol=gradient_atol)
    finally:
        restore_training_rng_state(saved_rng, device)
    if not report['passed']:
        raise ValueError('native arithmetic transition failed the unchanged cached-reference checks: ' +
                         json.dumps(report, sort_keys=True))
    return {'schema': 'natlang.native-arithmetic-transition-check/1',
            'from': previous_precision, 'to': config['native_linear_precision'],
            'parent_weights_sha256': checkpoint_sha256, 'runtime_reference': report}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-graphs', type=Path, required=True)
    parser.add_argument('--foundation-native-graphs', type=Path)
    parser.add_argument('--refresh-at-checkpoint', action=argparse.BooleanOptionalAction, default=False,
        help='return an explicit incomplete continuation for semantic refresh at the next declared checkpoint boundary')
    parser.add_argument('--learn-reasoning-boundaries', action=argparse.BooleanOptionalAction, default=False,
                        help='explicitly add learned existing reasoning token rows to an old parent')
    parser.add_argument('--communication-foundation', action=argparse.BooleanOptionalAction, default=False,
        help='pause at the complete admitted communication cohort with mandatory ordinary credits retained')
    parser.add_argument('--held-native-graphs', type=Path, required=True)
    parser.add_argument('--continued-native-graphs', type=Path)
    parser.add_argument('--continued-held-native-graphs', type=Path)
    parser.add_argument('--withdrawn-native-graphs', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--native-participant')
    parser.add_argument('--objective-continuation', action='store_true',
        help='explicit new source/objective stage; preserve full optimizer/RNG and start local exposure at zero')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--epochs', type=float, default=1.)
    parser.add_argument('--batch-graphs', type=int, default=1)
    parser.add_argument('--batch-tokens', type=int, default=0,
        help='pack whole sequential graphs to this token volume instead of graph-count batching')
    parser.add_argument('--share-causal-prefixes', action=argparse.BooleanOptionalAction, default=False,
        help='reuse exact ordinary token prefixes with unchanged targets, losses and exposure')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default=None,
        help='fresh backbone precision; required for a new lineage, resumes retain exact checkpoint dtype')
    from ..model.native_precision import PRODUCTION_NATIVE_LINEAR_PRECISIONS
    parser.add_argument('--native-linear-precision', choices=PRODUCTION_NATIVE_LINEAR_PRECISIONS, default=None,
        help='explicit native GEMM arithmetic transition; preserves source dtype, optimizer, RNG and row exposure')
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--learned-row-learning-rate',type=float,default=None)
    parser.add_argument('--interface-row-learning-rate',type=float,default=None,
        help='override only the four Neuralese role-token input/output tables; reasoning rows retain their inherited rate')
    parser.add_argument('--weight-decay', type=float, default=.1)
    parser.add_argument('--latent-ns', choices=('quintic', 'cubic5'), default='quintic')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--held-eval-tokens', type=int, default=0,
        help='rotating whole-graph held-token budget per cadence; final evaluation always covers all held data')
    parser.add_argument('--held-eval-graphs', type=int, default=0)
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--checkpoint-seconds', type=float, default=3600.)
    parser.add_argument('--checkpoint-input-offload', action=argparse.BooleanOptionalAction, default=None,
        help='store nonreentrant layer-checkpoint inputs on host RAM; omission inherits the saved mode')
    parser.add_argument('--checkpoint-layers', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--chunk-tokens', type=int, default=128)
    parser.add_argument('--fp8-training', choices=('off', 'te-fp8', 'triton-experts'), default='off')
    parser.add_argument('--fp8-recipe', choices=('delayed', 'current', 'current-bf16-dx'), default='delayed')
    parser.add_argument('--fp8-amax-history-len', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--task-eval', default=None)
    parser.add_argument('--task-eval-every', type=int, default=0)
    parser.add_argument('--forward-atol', type=float, default=None)
    parser.add_argument('--gradient-atol', type=float, default=None)
    parser.add_argument('--qualify-training-backend', action=argparse.BooleanOptionalAction, default=False,
        help='qualify actual enabled backend on restored live weights before useful updates; preserve RNG/state')
    parser.add_argument('--resident-data-directory', type=Path)
    parser.add_argument('--resident-geometry-qualification', type=Path)
    parser.add_argument('--memory-qualification', default=None,
        help='bound prepared new-source capacity request JSON; checked read-only after exact restore')
    parser.add_argument('--recipe-binding', default=None)
    parser.add_argument('--stage-id', default=None)
    parser.add_argument('--prerequisite-stages', default='[]')
    parser.add_argument('--objective-spec', type=Path, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    resident_state = None
    if args.resident_data_directory is not None:
        from .native_resident import recover
        resident_state = recover(args.resident_data_directory)
        if resident_state and resident_state['payload']['stage_id'] == args.stage_id:
            args.native_graphs = Path(resident_state['payload']['train']['path'])
            args.held_native_graphs = Path(resident_state['payload']['held']['path'])
            args.recipe_binding = json.dumps(resident_state['payload']['recipe'])
    checkpoint_options = {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}
    if not (math.isfinite(args.epochs) and args.epochs > 0) or min(args.batch_graphs, args.eval_every,
            args.chunk_tokens, args.rank) < 1 or min(args.checkpoint_every, args.batch_tokens, args.held_eval_tokens, args.held_eval_graphs) < 0:
        raise ValueError('native SFT requires positive epochs/batch/eval/chunk/rank and nonnegative checkpoint cadence')
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds < 0:
        raise ValueError('checkpoint-seconds must be finite and nonnegative')
    if args.checkpoint_every and args.checkpoint_every % args.eval_every:
        raise ValueError('checkpoint cadence must be an evaluation cadence multiple')
    tolerances = [value for value in (args.forward_atol, args.gradient_atol) if value is not None]
    if args.task_eval_every < 0 or any(not math.isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError('task cadence must be nonnegative and supplied native numerical tolerances positive')
    if args.qualify_training_backend and (args.fp8_training == 'off' or
            args.forward_atol is None or args.gradient_atol is None):
        raise ValueError('training backend qualification needs an enabled backend and declared tolerances')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    policy = CheckpointPolicy(args.out, every_steps=args.checkpoint_every,
                              every_seconds=args.checkpoint_seconds, prune_best=False)
    policy.install_signal_handlers()
    resume = None
    parent_continuation = False
    shared_resume = None
    if args.continue_from and not policy.slot.exists():
        model, tokenizer, roles, config, parent_state = load_native_checkpoint(args.continue_from, base=args.base,
                                                                          device=args.device, participant=args.native_participant)
        shared_resume = parent_state.get('shared_training')
        resume = parent_state.get('training')
        if shared_resume is not None:
            if not args.objective_continuation or not shared_resume.get('optimizer') or not shared_resume.get('rng'):
                raise ValueError('shared native parent requires declared objective continuation and full optimizer/RNG')
        elif not isinstance(resume, dict) or resume.get('schema') != TRAINING_SCHEMA:
            raise ValueError('native SFT continuation lacks exact optimizer/schedule/RNG state')
        parent_continuation = not policy.slot.exists()
        if parent_continuation:
            del parent_state
    elif policy.slot.exists():
        model, tokenizer, roles, config, state = load_native_checkpoint(policy.slot, base=args.base,
                                                                         device=args.device)
        if state.get('schema') != CHECKPOINT_SCHEMA:
            raise ValueError('rolling slot is not a native-state checkpoint')
        resume = state.get('training')
    else:
        if args.base is None:
            raise ValueError('fresh native SFT needs --base')
        if args.dtype is None:
            raise ValueError('fresh native SFT requires explicit --dtype (QAD LFM float32; Mellum bfloat16)')
        model, tokenizer, roles, config = load_native_model(args.base, rank=args.rank,
            device=args.device, dtype=getattr(torch, args.dtype))
    token_upgrade = None
    if args.learn_reasoning_boundaries:
        from ..model.native_model import upgrade_reasoning_tokens
        token_upgrade = upgrade_reasoning_tokens(model, tokenizer, config)
        if token_upgrade['added'] or token_upgrade.get('promoted_parameters'):
            checkpoint_options['token_row_upgrade'] = token_upgrade
    from ..model.native_precision import set_native_linear_precision, resolve_native_training_precision
    previous_precision = config.get('native_linear_precision', 'ieee')
    requested_precision = resolve_native_training_precision(previous_precision, args.native_linear_precision)
    set_native_linear_precision(model, requested_precision)
    config['native_linear_precision'] = requested_precision
    saved_training = resume or shared_resume or {}
    checkpoint_options['withdrawal_history']=list((saved_training.get('options') or {}).get('withdrawal_history') or [])
    token_history = list((saved_training.get('options') or {}).get('token_row_upgrade_history') or [])
    if token_upgrade and (token_upgrade['added'] or token_upgrade.get('promoted_parameters')):
        token_history.append({**token_upgrade, 'at_global_step': int(saved_training.get('global_step', saved_training.get('step', 0))),
                              'weights_optimizer_rng_exposure': 'preserved'})
    checkpoint_options['token_row_upgrade_history'] = token_history
    carried_progress = collect_sft_progress(saved_training)
    restored_progress = False
    precision_history = list((saved_training.get('options') or {}).get('native_precision_history') or [])
    transition = {'from': previous_precision, 'to': requested_precision,
        'source_dtype': config['dtype'],
        'at_global_step': int(saved_training.get('global_step', saved_training.get('step', 0))),
        'weights_optimizer_rng_exposure': 'preserved' if saved_training else 'fresh'}
    checkpoint_options['native_precision_transition'] = transition
    if requested_precision != previous_precision:
        precision_history.append(transition)
    checkpoint_options['native_precision_history'] = precision_history
    if args.dtype is not None and config['dtype'] != str(getattr(torch, args.dtype)):
        raise ValueError('declared dtype differs from the exact restored native checkpoint')
    if resume is None and (args.continued_native_graphs or args.continued_held_native_graphs):
        raise ValueError('native data continuation predecessors require an actual resumable checkpoint')
    if config['rank'] != args.rank and not args.continue_from and not policy.slot.exists():
        raise ValueError('native rank differs from declared stage rank')
    from ..data.text_corpus import tokenizer_fingerprint
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != config['tokenizer_fingerprint']:
        raise ValueError('loaded tokenizer differs from native model configuration')
    if resident_state and resume and resident_state['payload']['stage_id'] == args.stage_id:
        prior = (resume.get('source_receipts') or [{}])[0]
        if prior.get('sha256') != resident_state['payload']['train']['sha256']:
            args.continued_native_graphs = Path(prior['path'])
            checkpoint_options['continued_native_graphs'] = str(args.continued_native_graphs)
        prior_held = (resume.get('source_receipts') or [{}, {}])[1]
        if prior_held.get('sha256') != resident_state['payload']['held']['sha256']:
            args.continued_held_native_graphs = Path(prior_held['path'])
            checkpoint_options['continued_held_native_graphs'] = str(args.continued_held_native_graphs)
    train = NativeGraphSource(args.native_graphs, fingerprint, required_split='train', role_ids=roles.ids)
    held = NativeGraphSource(args.held_native_graphs, fingerprint, required_split='test', role_ids=roles.ids)
    train_groups, held_groups = train.groups, held.groups
    if train_groups & held_groups:
        raise ValueError('native train/held source groups overlap: ' + ', '.join(sorted(train_groups & held_groups)[:8]))
    source_receipts = [train.receipt(), held.receipt()]
    source_row_bindings = train.bindings
    from .native_continuation import reject_withdrawn_targets
    reject_withdrawn_targets(saved_training,source_row_bindings)
    held_row_bindings = held.bindings
    source_hash = hashlib.sha256(json.dumps(
        [source_receipts[0]['sha256'], source_receipts[1]['sha256'],
         sorted(source_row_bindings.items()), sorted(held_row_bindings.items())],
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    source_order = train.ordered_ids()
    checkpoint_options['_communication_ids']=[entry['id'] for entry in train.entries if entry.get('communication')]
    foundation_source = (NativeGraphSource(args.foundation_native_graphs, fingerprint, required_split='train', role_ids=roles.ids)
                         if args.foundation_native_graphs else train)
    if args.communication_foundation and args.foundation_native_graphs is None:
        raise ValueError('communication foundation requires its complete declared admitted source manifest')
    if args.communication_foundation:
        from .native_continuation import verify_initial_foundation_cohort
        verify_initial_foundation_cohort(train, foundation_source,
            already_declared=bool((resume or {}).get('options', {}).get('communication_foundation')))
    objective = None
    objective_binding = None
    if args.objective_spec is not None:
        spec_sha256 = sha256_file_hex_cached(args.objective_spec)
        objective_binding = {'path': str(args.objective_spec.resolve()), 'sha256': spec_sha256}
        spec = json.loads(args.objective_spec.read_text())
        if spec.get('schema') != 'natlang.native-policy-objective/1' or spec.get('kind') != 'native-policy':
            raise ValueError('native replay objective spec must declare natlang.native-policy-objective/1 kind native-policy')
        from ..rl.native_replay import NativePolicyObjective
        objective = NativePolicyObjective(args.objective_spec)
        if parent_continuation:
            actual = native_identity(model, config, checkpoint_sha256=sha256_file_hex_cached(args.continue_from))
            if objective.identity != actual:
                raise ValueError('native policy objective differs from the actual loaded actor embedding/checkpoint identity')
    model.share_causal_prefixes = args.share_causal_prefixes
    training_identity = {'schema': TRAINING_SCHEMA, 'config': config, 'train_source_hash': source_hash,
                'schedule': 'row-visit-deficit/2',
                'held_sha256': source_receipts[1]['sha256'], 'optimizer': 'shared-muon-sr-in-backward-v1',
                'optimizer_options': {'lr': args.learning_rate, 'learned_row_lr':args.learned_row_learning_rate, 'interface_row_lr':args.interface_row_learning_rate, 'weight_decay': args.weight_decay,
                                      'latent_ns': args.latent_ns},
                'batch_graphs': args.batch_graphs, 'batch_tokens': args.batch_tokens, 'seed': args.seed, 'rank': config['rank'],
                'fp8': {'kind': args.fp8_training, 'recipe': args.fp8_recipe,
                        'amax_history_len': args.fp8_amax_history_len},
                'objective': objective_binding}
    appended_ids = set()
    restored_frame = (restore_sft_progress(shared_resume or resume or {}, args.stage_id)
                      if parent_continuation and objective is None else None)
    if restored_frame is not None:
        resume = restored_frame
        restored_progress = True
    if args.withdrawn_native_graphs:
        if resume is None or args.continued_native_graphs is None:
            raise ValueError('native withdrawal needs the actual checkpoint and exact predecessor source')
        from .native_continuation import withdraw_ordinary_rows
        old_source=NativeGraphSource(args.continued_native_graphs, fingerprint,
            required_split='train',role_ids=roles.ids)
        origins={}
        def leaf_origins(source):
            if source.kind=='manifest':
                for child in source.children:leaf_origins(child)
            else:origins.update({key:source.file_sha256 for key in source.bindings})
        leaf_origins(old_source)
        resume=withdraw_ordinary_rows(resume,args.withdrawn_native_graphs,
            current_bindings=source_row_bindings,
            model_id=config['architecture']+':'+config['base_inventory_sha256'],source_origins=origins)
        carried_progress=collect_sft_progress(resume)
        checkpoint_options['withdrawal_history']=resume['options']['withdrawal_history']
    if shared_resume is not None and not restored_progress:
        old_options = shared_resume['options']
        old_optimizer = {'weight_decay': old_options.get('weight_decay', .1),
                         'latent_ns': old_options.get('latent_ns', 'quintic')}
        resume = {'identity': {**training_identity, 'optimizer_options': old_optimizer},
            'optimizer': shared_resume['optimizer'], 'rng': shared_resume['rng'],
            'fp8': shared_resume.get('fp8'), 'step': 0, 'global_step': shared_resume['step']}
    if resume is not None:
        predecessors = dict((resume.get('options') or {}).get('source_predecessors') or {})
        for role, position in (('continued_native_graphs', 0), ('continued_held_native_graphs', 1)):
            predecessor = getattr(args, role)
            if predecessor is not None:
                prior_receipts = resume.get('source_receipts') or []
                digest = sha256_file_hex_cached(predecessor)
                # Once adopted, the original predecessor stays valid across
                # later append/priority manifests. Exact immutable row bindings
                # and the prior physical prefix are verified below.
                adopted = predecessors.get(role) == digest
                if (not adopted and (len(prior_receipts) <= position
                        or digest != prior_receipts[position]['sha256'])):
                    raise ValueError('native data continuation predecessor differs from the actual checkpoint: ' + role)
                predecessors[role] = digest
        checkpoint_options['source_predecessors'] = predecessors
        previous = resume.get('identity') or {}
        stable = ('schema', 'optimizer', 'seed', 'rank', 'fp8', 'objective')
        from .fp8_backend import expert_precision_transition
        backend_transition = expert_precision_transition(previous.get('fp8'), training_identity['fp8'])
        if backend_transition is not None:
            if not getattr(args, 'qualify_training_backend', False):
                raise ValueError('expert adjoint precision transition requires --qualify-training-backend on the restored model')
            checkpoint_options['backend_precision_transition'] = backend_transition
            training_identity['backend_precision_transition'] = backend_transition
            stable = tuple(key for key in stable if key != 'fp8')
        from ..model.native_precision import same_source_precision_identity
        previous_config = previous.get('config') or {}
        if token_upgrade and token_upgrade['token_ids']:
            previous_config = {**previous_config, 'token_override_ids': config['token_override_ids']}
            if 'token_override_dtype' in config:
                previous_config['token_override_dtype'] = config['token_override_dtype']
        # A shared/RL parent may already carry promoted rows while this CE
        # frame predates their addition. The explicit upgrade flag remains
        # idempotent; no new model promotion is needed to adopt that contract.
        if token_upgrade and 'sparse_row_dtype' in config:
            previous_config = {**previous_config, 'sparse_row_dtype':config['sparse_row_dtype']}
        if not same_source_precision_identity(previous_config, config):
            raise ValueError('native SFT continuation changes model identity beyond declared GEMM arithmetic')
        if previous.get('schedule') not in {'source-order-stream/1', 'row-visit-deficit/2'}:
            raise ValueError('native SFT checkpoint has an unsupported row-credit schedule')
        if parent_continuation and args.objective_continuation:
            stable = tuple(key for key in stable if key not in {'objective', 'batch_graphs', 'batch_tokens'})
        if any(previous.get(key, 0 if key == 'batch_tokens' else None) != training_identity.get(key) for key in stable):
            raise ValueError('native SFT continuation changes model/held/optimizer/schedule identity')
        # MuonSR retains its constructor's NS variant on state restoration;
        # switching the update approximation preserves compatible moment state.
        if restored_progress or not (parent_continuation and args.objective_continuation):
            prior_bindings = resume.get('source_row_bindings') or {}
            appended_ids = append_only_rows(prior_bindings, source_row_bindings, label='training-source')
            prior_order = resume.get('source_row_order') or []
            if source_order[:len(prior_order)] != prior_order:
                raise ValueError('native SFT continuation reordered prior training rows; exact cursor cannot be resumed')
            prior_held = resume.get('held_row_bindings') or {}
            append_only_rows(prior_held, held_row_bindings, label='held-source')
        elif shared_resume is None:
            append_only_rows(resume.get('held_row_bindings') or {}, held_row_bindings, label='held-source')
        if appended_ids:
            print(json.dumps({'event': 'native_data_append', 'prior_rows': len(prior_bindings),
                              'new_rows': len(appended_ids), 'optimizer_state': 'restored',
                              'cursor': resume.get('cursor'), 'policy': 'immutable source-order prefix then appended rows'}),
                  flush=True)
        training_identity['optimizer_options'] = {'lr': args.learning_rate,
            'learned_row_lr':args.learned_row_learning_rate, 'interface_row_lr':args.interface_row_learning_rate,
            'weight_decay': args.weight_decay, 'latent_ns': args.latent_ns,
            'continued_from': previous.get('optimizer_options')}
        training_identity['prior_source_hash'] = previous.get('train_source_hash')
    if resume is not None and not parent_continuation and (resume.get('options') or {}).get('continuation'):
        checkpoint_options['continuation'] = resume['options']['continuation']
    if parent_continuation:
        checkpoint_options['continuation'] = {'checkpoint_sha256': sha256_file_hex_cached(args.continue_from),
            'participant': args.native_participant, 'objective_transition': bool(args.objective_continuation)}
    if args.fp8_training != 'off':
        arch = config['architecture']
        expected = 'triton-experts' if arch in {'mellum', 'maple'} else 'te-fp8'
        if args.fp8_training != expected:
            raise ValueError(f'{arch} native SFT requires {expected} for the declared FP8 path')
        from .fp8_backend import make_backend
        backend = make_backend(args.fp8_training, model.port.hf, device=args.device, recipe=args.fp8_recipe,
                               amax_history_len=args.fp8_amax_history_len)
        model.port.activation_checkpoint = backend.checkpoint
    else:
        backend = None
    if config['architecture'] in {'mellum', 'maple'} and backend is None:
        raise ValueError('native Mellum/Maple production SFT requires its shared triton-experts FP8 memory path')
    inherited_offload = (saved_training.get('options') or {}).get('checkpoint_input_offload', False)
    args.checkpoint_input_offload = (inherited_offload if args.checkpoint_input_offload is None
                                     else args.checkpoint_input_offload)
    if type(args.checkpoint_input_offload) is not bool or (args.checkpoint_input_offload and not args.checkpoint_layers):
        raise ValueError('checkpoint input offload requires actual nonreentrant layer checkpointing')
    from ..common.checkpoint_offload import configure_checkpoint_input_offload
    configure_checkpoint_input_offload(model.port, enabled=args.checkpoint_input_offload)
    checkpoint_options['checkpoint_input_offload'] = args.checkpoint_input_offload
    model.port.checkpoint_layers = args.checkpoint_layers
    model.to(args.device)
    model.train()
    optimizer, named = make_native_optimizer(model, lr=args.learning_rate, learned_row_lr=args.learned_row_learning_rate, interface_row_lr=args.interface_row_learning_rate,
                                              weight_decay=args.weight_decay, latent_ns=args.latent_ns)
    task_eval = None
    task_spec = None
    if args.task_eval:
        from ..eval.task_cadence import TaskEval, parse_task_eval
        task_spec = parse_task_eval(args.task_eval)
        from ..serve.engine import Engine
        from ..serve.store import TensorStore
        # Construct the real native serving engine over the same live model,
        # avoiding a second full-width model allocation during evaluation.
        serving_identity = native_identity(model, config, checkpoint_sha256='0' * 64)
        engine = Engine(model.port, None, tokenizer, TensorStore(),
            f"native-state/{config['architecture']}", model_name=f"native-{config['architecture']}",
            device=args.device, prompt_rendering=task_spec.get('prompt_rendering', 'full'),
            shared_model=model, role_tokens=roles, model_config=config,
            checkpoint_sha256=serving_identity['checkpoint_sha256'],
            embedding_weight_sha256=serving_identity['embedding_weight_sha256'])
        task_eval = TaskEval(task_spec, args.out, engine, prompt_rendering=task_spec.get('prompt_rendering', 'full'))
    step, epoch, cursor, global_step = 0, 0, 0, 0
    metrics = {}
    exposure_counts = {key: 0 for key in source_row_bindings}
    if resume is not None:
        if token_upgrade and (token_upgrade['added'] or token_upgrade.get('promoted_parameters')):
            optimizer.load_state_dict(resume['optimizer'], allow_added_names=(
                'interface.extra_input_rows', 'interface.extra_output_rows'),
                allow_promoted_names=tuple('interface.'+name for name in token_upgrade.get('promoted_parameters', {})))
        else:
            optimizer.load_state_dict(resume['optimizer'])
        # Adaptive production may tune the step size without discarding moments.
        # The new value is recorded in optimizer_options while moments/RNG stay exact.
        from .native_learning_rates import apply_native_learning_rates
        apply_native_learning_rates(optimizer,backbone_lr=args.learning_rate,
                                    learned_row_lr=args.learned_row_learning_rate, interface_row_lr=args.interface_row_learning_rate)
        if optimizer.latent is not None:
            for group in optimizer.latent.param_groups:
                group['weight_decay'] = args.weight_decay
        if backend is not None:
            if resume.get('fp8') is None:
                raise ValueError('native SFT checkpoint lacks the declared FP8 state')
            backend.restore_extra_state(resume['fp8'])
        if parent_continuation and not restored_progress:
            # A new declared phase consumes its current full source set while
            # keeping exact weights/optimizer/RNG; its stage-local cursor begins
            # at zero and the global update counter remains monotonic. The
            # predecessor's exposure is in its own readiness receipt, so this
            # stage starts its own exposure counters at zero.
            global_step = int(resume.get('global_step', resume['step']))
            metrics = {}
        else:
            step, epoch, cursor = int(resume['step']), int(resume['epoch']), int(resume['cursor'])
            global_step = int(resume.get('global_step', step))
            exposure_counts.update({key: int(value) for key, value in (resume.get('exposure_counts') or {}).items()
                                    if key in exposure_counts})
            metrics = dict(resume.get('metrics') or {})
            if restored_progress and shared_resume is not None:
                # Credits survive other objectives; evaluations belong to the new weights.
                for key in ('held', 'held_coverage', 'task_checkpoint_sha256', 'task_native_identity'):
                    metrics.pop(key, None)
        restore_training_rng_state(resume['rng'], args.device)
        resume['optimizer'] = None
        if 'state' in locals():
            del state
        if 'parent_state' in locals():
            del parent_state
    transition_check = qualify_precision_transition(model, tokenizer, roles, config,
        previous_precision=previous_precision,
        checkpoint_sha256=sha256_file_hex_cached(args.continue_from if parent_continuation else policy.slot)
            if parent_continuation or policy.slot.exists() else config['base_inventory_sha256'],
        forward_atol=args.forward_atol, gradient_atol=args.gradient_atol,
        device=args.device, backend=backend)
    if transition_check is not None:
        checkpoint_options['native_precision_transition']['reference_check'] = transition_check
        write_json(args.out / 'precision-transition.json', transition_check)
    if args.qualify_training_backend:
        from ..eval.native_backend import qualify_training_backend
        backend_check = qualify_training_backend(model, tokenizer, roles, config, backend=backend,
            checkpoint_sha256=sha256_file_hex_cached(args.continue_from if parent_continuation else policy.slot)
                if parent_continuation or policy.slot.exists() else config['base_inventory_sha256'],
            forward_atol=args.forward_atol, gradient_atol=args.gradient_atol, device=args.device)
        checkpoint_options['training_backend_check'] = backend_check
        write_json(args.out / 'training-backend-check.json', backend_check)
    if args.memory_qualification:
        from ..eval.native_memory import run_prepared
        memory_check = run_prepared(model,optimizer,train,roles,config,args,backend=backend,
            checkpoint_sha256=sha256_file_hex_cached(args.continue_from if parent_continuation else policy.slot)
                if parent_continuation or policy.slot.exists() else config['base_inventory_sha256'])
        checkpoint_options['memory_qualification'] = {'passed':memory_check['passed'],
            'request':memory_check['request'],'parent_checkpoint_sha256':memory_check['parent_checkpoint_sha256']}
        checkpoint_options['checkpoint_input_offload'] = args.checkpoint_input_offload
        checkpoint_options['resident_geometry_qualification'] = (str(args.resident_geometry_qualification)
            if args.resident_geometry_qualification else None)
    # Actual per-row optimizer credits drive the schedule. Source identity
    # remains append-only while communication deficits execute first.
    whole_epochs = int(args.epochs)
    fractional_rows = math.ceil((args.epochs - whole_epochs) * len(train))
    target_rows = whole_epochs * len(train) + fractional_rows
    # Cursor/exposure is the durable schedule, not the old batch partition.
    # Adapt packing after a safe stop without replaying or skipping any row.
    if parent_continuation and not restored_progress:
        step = 0
    required_visits, pending_batches = pending_exposure_batches(train.entries, exposure_counts,
        epochs=args.epochs, batch_graphs=args.batch_graphs, batch_tokens=args.batch_tokens)
    target_rows = sum(required_visits.values())
    total_end = step + len(pending_batches)
    if step > total_end:
        raise ValueError('native checkpoint update is beyond the declared epoch exposure')
    resumed_step = step
    last_saved_step = step if policy.slot.is_file() and not parent_continuation else -1
    refresh_boundary = False
    if args.refresh_at_checkpoint and not (args.checkpoint_every or args.checkpoint_seconds):
        raise ValueError('semantic refresh needs a declared checkpoint step/time cadence')
    schedule = stream_pending_batches(train, pending_batches)
    resident = None
    if args.resident_data_directory is not None:
        from .native_resident import ResidentAdoption
        from ..eval.native_memory import capacity_identity
        resident = ResidentAdoption(args.resident_data_directory, stage_id=args.stage_id,
            recipe_binding=json.loads(args.recipe_binding), train=train, held=held,
            geometry_qualification=args.resident_geometry_qualification,
            geometry_identity=capacity_identity(config,args,backend))
        if resident.state:
            checkpoint_options['resident_adoption'] = {'generation': resident.generation,
                'history': resident.state['history']}
    metadata_dirty = False
    eval_at = max(1, args.eval_every)
    started = time.time()
    def foundation_exposure():
        return communication_foundation(foundation_source, exposure_counts, epochs=args.epochs,
                                        trained_bindings=source_row_bindings)
    prior_foundations = set((resume or {}).get('completed_foundations') or
        (carried_progress.get(args.stage_id or 'native-graph-ce') or {}).get('completed_foundations') or [])
    foundation_template = foundation_exposure() if args.communication_foundation else None
    foundation_missing = {key for key,row in (foundation_template or {}).get('rows',{}).items()
        if not row['admitted_to_training'] or row['required_visits']<=0 or row['graph_visits']<row['required_visits']}
    def foundation_pending():
        return bool(foundation_template and foundation_template['graphs'] and not foundation_missing and
                    foundation_template['cohort_sha256'] not in prior_foundations)
    try:
        while not policy.signaled and not foundation_pending():
            if resident is not None:
                expanded = resident.poll(global_step)
                if expanded is not None:
                    schedule.close()
                    metadata_dirty = True
                    train, held = expanded
                    args.held_native_graphs = held.path
                    held_row_bindings = held.bindings
                    args.native_graphs = train.path
                    args.recipe_binding = json.dumps(resident.recipe_binding)
                    source_row_bindings = train.bindings
                    source_order = train.ordered_ids()
                    source_receipts = [train.receipt(), held.receipt()]
                    source_hash = hashlib.sha256(json.dumps(
                        [source_receipts[0]['sha256'], source_receipts[1]['sha256'],
                         sorted(source_row_bindings.items()), sorted(held_row_bindings.items())],
                        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                    training_identity['train_source_hash'] = source_hash
                    training_identity['held_sha256'] = held.file_sha256
                    for key in source_row_bindings:
                        exposure_counts.setdefault(key, 0)
                    checkpoint_options.update(native_graphs=str(train.path), held_native_graphs=str(held.path), recipe_binding=args.recipe_binding,
                        resident_adoption={'generation': resident.generation, 'history': resident.state['history']})
                    checkpoint_options['_communication_ids'] = [entry['id'] for entry in train.entries if entry.get('communication')]
                    required_visits, pending_batches = pending_exposure_batches(train.entries, exposure_counts,
                        epochs=args.epochs, batch_graphs=args.batch_graphs, batch_tokens=args.batch_tokens)
                    total_end = step + len(pending_batches)
                    target_rows = sum(required_visits.values())
                    schedule = stream_pending_batches(train, pending_batches)
            if policy.signaled or step >= total_end:
                break
            update_started = time.monotonic()
            visit, indices, pass_end, batch = next(schedule)
            batch_tokens = sum(train.entries[index]['forward_tokens'] for index in indices)
            cursor = indices[-1] + 1
            loss, details = native_training_update(model, optimizer, batch, backend=backend,
                checkpoint_layers=args.checkpoint_layers, chunk_tokens=args.chunk_tokens, objective=objective)
            step += 1
            global_step += 1
            for example in batch:
                exposure_counts[example.id] += 1
                if example.id in foundation_missing:
                    row=foundation_template['rows'][example.id]
                    if row['admitted_to_training'] and row['required_visits']>0 and exposure_counts[example.id]>=row['required_visits']:
                        foundation_missing.remove(example.id)
            epoch = min(exposure_counts[key] for key, required in required_visits.items() if required > 0)
            if pass_end and all(exposure_counts[key] >= required for key, required in required_visits.items()):
                cursor = 0
            # Telemetry is appended after the actual optimizer update. The rolling
            # checkpoint remains the authority for committed resume/exposure; logs
            # can include work subsequently replayed after an uncheckpointed crash.
            train_loss = float(loss.detach())
            seconds = time.monotonic() - update_started
            supervised_tokens = sum(sum(weight > 0 for weight in item.target_weights) for item in batch)
            progress = {'schema': 'natlang.native-training-progress/1', 'kind': 'native',
                        'step': step, 'global_step': global_step, 'epoch': epoch, 'cursor': cursor,
                        'loss': train_loss, 'graphs': len(batch), 'forward_tokens': batch_tokens,
                        'supervised_tokens': supervised_tokens, 'target_tokens': supervised_tokens,
                        'seconds': seconds, 'forward_tokens_per_second': batch_tokens / max(seconds, 1e-9),
                        'supervised_tokens_per_second': supervised_tokens / max(seconds, 1e-9),
                        'elapsed_seconds': time.time() - started, 'timestamp': time.time(),
                        'checkpoint_step': last_saved_step,
                        'communication_exposure': communication_exposure(train.entries, exposure_counts, required_visits)}
            progress['learning_rates']=optimizer.native_learning_rates
            progress['checkpoint_input_offload'] = {'enabled': args.checkpoint_input_offload,
                'checkpoint_calls': getattr(model.port.activation_checkpoint, 'calls', 0)}
            if getattr(model, 'native_sparse_updates', None):
                progress['sparse_row_updates'] = model.native_sparse_updates
            if objective is None and details:
                physical_tokens = details[0]['packed_physical_tokens']
                progress['physical_forward_tokens'] = physical_tokens
                progress['reused_prefix_tokens'] = details[0]['original_physical_tokens'] - physical_tokens
                progress['physical_forward_tokens_per_second'] = physical_tokens / max(seconds, 1e-9)
                progress['communication_graphs'] = sum(bool(train.entries[index].get('communication')) for index in indices)
                progress['boundary_targets'] = {
                    token: {'count': sum(item['boundary_targets'][token]['count'] for item in details),
                            'weight': sum(item['boundary_targets'][token]['weight'] for item in details),
                            'objective_weight': sum(item['boundary_targets'][token]['weight'] / item['weight']
                                                    for item in details) / len(details)}
                    for token in details[0]['boundary_targets']}
            if str(args.device).startswith('cuda'):
                progress['cuda_memory_bytes'] = {'allocated': torch.cuda.memory_allocated(args.device),
                    'reserved': torch.cuda.memory_reserved(args.device),
                    'lifetime_peak_allocated': torch.cuda.max_memory_allocated(args.device)}
            with (args.out / 'train.jsonl').open('a') as log:
                log.write(json.dumps(progress, sort_keys=True) + '\n')
            write_json(args.out / 'progress.json', progress)
            print(json.dumps(progress, sort_keys=True), flush=True)
            due_checkpoint = policy.due(step, end=step == total_end)
            if policy.signaled:
                break
            due_task = task_eval is not None and (step == total_end or
                        bool(args.task_eval_every and step % args.task_eval_every == 0))
            due_eval = step % eval_at == 0 or step == total_end or due_task
            if due_eval:
                held_metrics, held_coverage = _evaluate_cadence(model, held, args=args,
                    coverage=metrics.get('held_coverage'), previous_result=metrics.get('held'), final=step == total_end, backend=backend)
                metrics = {'train_loss': train_loss, 'held': held_metrics, 'held_coverage': held_coverage,
                           'step': step, 'epoch': epoch, 'cursor': cursor,
                           'elapsed_seconds': time.time() - started,
                           'fp8': backend.inventory() if backend is not None else {'enabled': False},
                           'learning_rates':optimizer.native_learning_rates}
                write_json(args.out / 'metrics.json', metrics)
            if policy.signaled:
                break
            if due_task:
                # Bind the task proof to the checkpoint from these exact weights.
                _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
                      global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
                      source_hash=source_hash, source_row_bindings=source_row_bindings,
                      held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
                      rng=capture_training_rng_state(args.device), metrics=metrics,
                      architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
                      fp8_state=backend.extra_state() if backend is not None else None, sft_progress=carried_progress)
                last_saved_step = step
                metadata_dirty = False
                if policy.signaled:
                    break
                checkpoint_sha = sha256_file_hex_cached(policy.slot)
                serving_identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha)
                engine = task_eval.engine
                engine.checkpoint_sha256 = serving_identity['checkpoint_sha256']
                engine.embedding_weight_sha256 = serving_identity['embedding_weight_sha256']
                engine.native_identity = serving_identity
                saved_rng = capture_training_rng_state(args.device)
                model.eval()
                try:
                    if backend is not None:
                        with backend.native_modules():
                            metrics['task_eval'] = task_eval.cadence(step, final=step == total_end)
                    else:
                        metrics['task_eval'] = task_eval.cadence(step, final=step == total_end)
                finally:
                    model.train()
                    restore_training_rng_state(saved_rng, args.device)
                metrics['task_checkpoint_sha256'] = checkpoint_sha
                metrics['task_native_identity'] = serving_identity
                write_json(args.out / 'metrics.json', metrics)
            if due_checkpoint and not due_task:
                _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
                      global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
                      source_hash=source_hash, source_row_bindings=source_row_bindings,
                      held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
                      rng=capture_training_rng_state(args.device), metrics=metrics,
                      architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
                      fp8_state=backend.extra_state() if backend is not None else None, sft_progress=carried_progress)
                last_saved_step = step
                metadata_dirty = False
            if args.refresh_at_checkpoint and due_checkpoint:
                refresh_boundary = True
                break
        if step == total_end and not policy.signaled:
            try:
                next(schedule)
            except StopIteration:
                pass
            else:
                raise ValueError('native source schedule exceeds its declared updates')
    finally:
        schedule.close()
    if metadata_dirty or ((step != resumed_step or policy.signaled) and last_saved_step != step):
        if not policy.signaled and not metrics.get('held'):
            metrics['held'], metrics['held_coverage'] = _evaluate_cadence(model, held, args=args,
                coverage=metrics.get('held_coverage'), previous_result=metrics.get('held'), final=step == total_end, backend=backend)
        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
              global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
              source_hash=source_hash, source_row_bindings=source_row_bindings,
              held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
              rng=capture_training_rng_state(args.device), metrics=metrics,
              architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
              fp8_state=backend.extra_state() if backend is not None else None, sft_progress=carried_progress)
        last_saved_step = step
    completed = all(exposure_counts[key] >= visits for key, visits in required_visits.items())
    if completed and not policy.signaled and (metrics.get('held', {}).get('scope') != 'full'
            or metrics.get('held', {}).get('source_sha256') != held.file_sha256):
        metrics['held'], metrics['held_coverage'] = _evaluate_cadence(model, held, args=args,
            coverage=metrics.get('held_coverage'), previous_result=metrics.get('held'), final=True, backend=backend)
        metrics.pop('task_checkpoint_sha256', None)
        metrics.pop('task_native_identity', None)
        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
            global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
            source_hash=source_hash, source_row_bindings=source_row_bindings,
            held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
            rng=capture_training_rng_state(args.device), metrics=metrics,
            architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
            fp8_state=backend.extra_state() if backend is not None else None, sft_progress=carried_progress)
    completed = (completed and metrics.get('held', {}).get('scope') == 'full'
                 and metrics.get('held', {}).get('source_sha256') == held.file_sha256)
    foundation = foundation_exposure() if args.communication_foundation else None
    foundation_boundary = bool(foundation and foundation['complete'] and
                              foundation['cohort_sha256'] not in prior_foundations and not policy.signaled)
    # A continuation may inherit a completed checkpoint made before task
    # summaries were checkpointed. At the same final weights, make one real
    # cadence call; do not reuse an unbound log row or enter final_at's retry
    # loop merely to force an eligible score.
    if not policy.signaled and task_eval is not None and (completed or foundation_boundary or refresh_boundary) and (
            not metrics.get('task_checkpoint_sha256') or
            metrics['task_checkpoint_sha256'] != sha256_file_hex_cached(policy.slot)):
        current_sha = sha256_file_hex_cached(policy.slot)
        serving_identity = native_identity(model, config, checkpoint_sha256=current_sha)
        engine = task_eval.engine
        engine.checkpoint_sha256 = serving_identity['checkpoint_sha256']
        engine.embedding_weight_sha256 = serving_identity['embedding_weight_sha256']
        engine.native_identity = serving_identity
        saved_rng = capture_training_rng_state(args.device)
        model.eval()
        try:
            if backend is not None:
                with backend.native_modules():
                    metrics['task_eval'] = task_eval.cadence(step, final=True)
            else:
                metrics['task_eval'] = task_eval.cadence(step, final=True)
        finally:
            model.train()
            restore_training_rng_state(saved_rng, args.device)
        metrics['task_checkpoint_sha256'] = current_sha
        metrics['task_native_identity'] = serving_identity
        write_json(args.out / 'metrics.json', metrics)
    checkpoint_sha = sha256_file_hex_cached(policy.slot)
    model_identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha)
    task_summary = metrics.get('task_eval')
    task_receipt_path = None
    if task_summary:
        task_receipt_path = args.out / 'native-task-execution-receipt.json'
        task_identity = metrics.get('task_native_identity')
        from ..eval.native_state import task_execution_receipt
        task_receipt = task_execution_receipt(task_summary, task_identity,
                                               checkpoint_sha256=checkpoint_sha)
        write_json(task_receipt_path, task_receipt)
    runtime_path = args.out / 'native-runtime-certificate.json'
    if policy.signaled or (not completed and not foundation_boundary and not refresh_boundary):
        # Operational stops preserve the exact resumable checkpoint. Defer a
        # full capture/splice qualification until the declared data exposure is
        # finished; do not make interruptions pay for a second model pass.
        runtime_report = {'schema': 'natlang.native-runtime-certificate/1', 'passed': False,
                          'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
                          'deferred': True, 'reason': ('operational stop; qualification deferred until resume'
                            if policy.signaled else 'declared native graph exposure is incomplete')}
    else:
        try:
            if args.forward_atol is None or args.gradient_atol is None:
                raise ValueError('runtime qualification requires recipe-bound forward and gradient tolerances')
            from ..eval.native_state import qualify_existing
            if backend is not None:
                with backend.native_modules():
                    runtime_report = qualify_existing(model, tokenizer, roles, config,
                        checkpoint_sha256=checkpoint_sha, forward_atol=args.forward_atol,
                        gradient_atol=args.gradient_atol,
                        task_execution_receipt=(str(task_receipt_path.resolve()) if task_receipt_path else None),
                        backbone_optimizer_updates=step)
            else:
                runtime_report = qualify_existing(model, tokenizer, roles, config,
                    checkpoint_sha256=checkpoint_sha, forward_atol=args.forward_atol,
                    gradient_atol=args.gradient_atol,
                    task_execution_receipt=(str(task_receipt_path.resolve()) if task_receipt_path else None),
                    backbone_optimizer_updates=step)
        except Exception as error:  # record a real failed check without discarding the trained checkpoint
            runtime_report = {'schema': 'natlang.native-runtime-certificate/1', 'passed': False,
                              'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
                              'error': f'{type(error).__name__}: {error}'}
    write_json(runtime_path, runtime_report)
    task_receipt_ok = bool(task_receipt_path and json.loads(task_receipt_path.read_text()).get('passed'))
    recipe_binding = json.loads(args.recipe_binding) if args.recipe_binding else None
    prerequisites = json.loads(args.prerequisite_stages)
    exposure_by_source_group = {}
    for entry in train.entries:
        visits = int(exposure_counts[entry['id']])
        token_visits = visits * entry['supervised_tokens']
        for group in entry['groups']:
            counts = exposure_by_source_group.setdefault(group, {'graph_visits': 0,
                                                                   'supervised_token_visits': 0})
            counts['graph_visits'] += visits
            counts['supervised_token_visits'] += token_visits
    exposure_counts_sha256 = hashlib.sha256(json.dumps(sorted(exposure_counts.items()),
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    communication = communication_exposure(train.entries, exposure_counts, required_visits)
    mandatory_sft = progress_obligations(carried_progress)
    continuation_boundary = {'kind': 'semantic-refresh', 'reached': bool(refresh_boundary and step>resumed_step),
        'new_optimizer_updates':step-resumed_step, 'checkpoint_sha256':checkpoint_sha}
    foundation_ready = bool(foundation_boundary and runtime_report.get('passed') is True and task_receipt_ok)
    resident_receipt = ({'generation': resident.generation, 'directory': str(resident.directory.resolve()),
                         'history': resident.state['history']} if resident and resident.state else None)
    report = {'schema': 'natlang.neuralese-native-sft-report/1',
              'resident_adoption': resident_receipt,
              'communication_exposure': communication, 'completed': completed,
              'foundation_ready': foundation_ready, 'communication_foundation': foundation, 'mandatory_sft': mandatory_sft,
              'continuation_boundary': continuation_boundary,
              'raw_communication_rows':raw_communication_rows(carried_progress),
              'numerics_ok': runtime_report.get('passed') is True, 'qualified': False,
              'quality_status': 'measured-for-supervisor',
              'train_exposure': {key: sum(exposure_counts[entry['id']] * entry['supervised_tokens']
                  for entry in train.entries) for key in ('target_tokens', 'supervised_tokens')},
              'step': step, 'epoch': epoch, 'cursor': cursor, 'training_seconds': time.time() - started,
              'model_config': config, 'continuation': checkpoint_options.get('continuation'), 'train_source': source_receipts[0], 'held_source': source_receipts[1],
              'source_hash': source_hash, 'metrics': metrics, 'checkpoint': str(policy.slot),
              'checkpoint_sha256': checkpoint_sha, 'task_eval': task_summary,
              'runtime_certificate': {'path': str(runtime_path.resolve()),
                                      'sha256': sha256_file_hex_cached(runtime_path),
                                      'passed': runtime_report.get('passed') is True},
              'base_kl': {'status': 'telemetry-only', 'measured': False}}
    write_json(args.out / 'report.json', report)
    write_json(args.out / 'native-training-receipt.json', {
        'schema': 'natlang.native-training-readiness/1',
        'resident_adoption': resident_receipt,
        'communication_exposure': communication, 'foundation_ready': foundation_ready,
        'communication_foundation': foundation, 'mandatory_sft': mandatory_sft,
        'continuation_boundary':continuation_boundary,
        'raw_communication_rows':raw_communication_rows(carried_progress), 'model_id': config['architecture'] + ':' +
        config['base_inventory_sha256'], 'lineage': str(args.out.resolve()),
        'stage_id': args.stage_id or args.out.name,
        'checkpoint': {'path': str(policy.slot.resolve()), 'sha256': report['checkpoint_sha256'],
                       'schema': CHECKPOINT_SCHEMA},
        'native_identity': model_identity, 'continuation': checkpoint_options.get('continuation'),
        'runtime_certificate': {'path': str(runtime_path.resolve()),
                                'sha256': sha256_file_hex_cached(runtime_path),
                                'passed': runtime_report.get('passed') is True},
        'prerequisite_stages': prerequisites, 'recipe': recipe_binding,
        'completed': bool(completed and communication['complete'] and
                          runtime_report.get('passed') is True and task_receipt_ok and
                          all(item.get('completed') is True for item in prerequisites)),
        'task_eval': task_summary,
        'task_execution_receipt': ({'path': str(task_receipt_path.resolve()),
                                     'sha256': sha256_file_hex_cached(task_receipt_path), 'passed': task_receipt_ok}
                                    if task_receipt_path else None),
        'training_receipt': {'path': str((args.out / 'report.json').resolve()),
                                                      'sha256': sha256_file_hex_cached(args.out / 'report.json')},
        'exposure': {'train_rows': len(train), 'held_rows': len(held),
                     'supervised_tokens': sum(entry['supervised_tokens'] for entry in train.entries),
                     'graph_visits': sum(exposure_counts.values()),
                     'supervised_token_visits': sum(exposure_counts[entry['id']] *
                         entry['supervised_tokens'] for entry in train.entries),
                     'row_update_counts_sha256': exposure_counts_sha256,
                     'by_source_group': exposure_by_source_group,
                     'consumed_rows': min(target_rows, sum(exposure_counts.values())),
                     'consumed_updates': step, 'completed_epochs': epoch}})
    return 0


def _graph_loss(model, example, args):
    from ..data.native_graph import graph_loss
    return graph_loss(model, example, checkpoint_layers=args.checkpoint_layers,
                      chunk_tokens=args.chunk_tokens)


if __name__ == '__main__':
    raise SystemExit(main())
