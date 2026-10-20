"""Shared full-state supervised training for positional-KV/hybrid-state graphs.

The declared recipe supplies admitted, role-aware native graph JSONL. This
stage trains the exact model used by the native serving path; legacy projected
records are rejected by the shared graph parser.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex_cached,canonical_json_sha256_hex
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


def pending_exposure_batches(entries, exposure_counts, *, epochs, batch_graphs=1, batch_tokens=0,
                             foundation_ids=None, foundation_communication_batches=1, minimum_visits=None,
                             schedule_offset=0, cohort_ids=None):
    """Schedule only missing row visits, including rows appended after completion.

    Communication starts first and alternates with explicitly prioritized
    ordinary data. When both positive-priority and priority-zero ordinary rows
    remain, preserve a 4:1 prompt-progress/debt mix, offset by the saved update
    count so a resumed or appended schedule does not restart at the urgent lane.
    Preserve source order within each class. Completed visits remain credits
    across changed packing and appended sources; fractional epochs cover the
    declared source prefix. The checkpoint's actual row counts determine what
    remains.
    """
    if type(foundation_communication_batches) is not int or foundation_communication_batches < 1:
        raise ValueError('foundation communication batches must be a positive integer')
    if type(schedule_offset) is not int or schedule_offset < 0:
        raise ValueError('native schedule offset must be a nonnegative optimizer step')
    foundation_ids = set(foundation_ids or ())
    if cohort_ids is not None and (len(cohort_ids) != len(entries)
            or any(not isinstance(key, str) or not key for key in cohort_ids)):
        raise ValueError('cohort scheduling requires one stable identity per source row')
    if any(entry['id'] in foundation_ids and not entry.get('communication') for entry in entries):
        raise ValueError('fixed foundation priority requires genuine communication rows')
    if foundation_ids - {entry['id'] for entry in entries}:
        raise ValueError('fixed foundation priority cannot name absent source rows')
    whole = int(epochs)
    fractional = math.ceil((epochs - whole) * len(entries))
    quotas = {entry['id']: max(int(bool(entry.get('communication'))), whole + (index < fractional))
              for index, entry in enumerate(entries)}
    for key, value in (minimum_visits or {}).items():
        if type(value) is not int or value < 0:
            raise ValueError('preserved row quotas must be nonnegative integers')
        if key in quotas:quotas[key] = max(quotas[key], value)
    batches = []
    pending = {visit: [index for index, entry in enumerate(entries)
                       if exposure_counts.get(entry['id'], 0) <= visit < quotas[entry['id']]]
               for visit in range(max(quotas.values(), default=0))}
    for rank in sorted({scheduling_rank(entry) for entry in entries}):
        for visit, remaining in pending.items():
            indices = [index for index in remaining if scheduling_rank(entries[index]) == rank]
            cohorts = {}
            for index in indices:
                urgent = entries[index].get('training_priority', 0) > 0 and entries[index]['id'] not in foundation_ids
                key = cohort_ids[index] if cohort_ids is not None and urgent else None
                cohorts.setdefault(key, []).append(index)
            for selected_indices in cohorts.values():
                selected = [entries[index] for index in selected_indices]
                boundaries = batch_boundaries(selected, batch_graphs=batch_graphs, batch_tokens=batch_tokens)
                for left, right in boundaries:
                    batches.append((visit, selected_indices[left:right], False))
    # A positive source priority must take effect while a large communication
    # cohort is still pending, rather than waiting for its entire completion.
    # Only reorder already packed batches: quotas and authenticated credits do
    # not depend on this policy, and no row is replayed or omitted.
    communication, urgent, ordinary = [], [], []
    for batch in batches:
        entry = entries[batch[1][0]]
        is_foundation = bool(foundation_ids) and entry['id'] in foundation_ids
        if entry.get('communication') and (is_foundation or entry.get('training_priority', 0) == 0):
            destination = communication
        elif entry.get('training_priority', 0) > 0:
            destination = urgent
        else:
            destination = ordinary
        destination.append(batch)
    if cohort_ids is not None:
        from .native_cohort_schedule import balance_urgent_batches
        urgent = balance_urgent_batches(entries, urgent, exposure_counts, cohort_ids,
                                       foundation_ids=foundation_ids)
    def foundational(batch):
        return bool(foundation_ids) and any(entries[index]['id'] in foundation_ids for index in batch[1])
    communication.sort(key=lambda batch: not foundational(batch))
    remaining_foundation_batches = sum(foundational(batch) for batch in communication)
    if urgent and communication:
        interleaved = []
        comm_cursor = ordinary_cursor = 0
        while comm_cursor < len(communication) or ordinary_cursor < len(urgent):
            count = min(foundation_communication_batches, remaining_foundation_batches) if remaining_foundation_batches else 1
            for _ in range(count):
                if comm_cursor == len(communication):
                    break
                batch = communication[comm_cursor]
                interleaved.append(batch)
                remaining_foundation_batches -= int(foundational(batch))
                comm_cursor += 1
            if ordinary_cursor < len(urgent):
                interleaved.append(urgent[ordinary_cursor])
                ordinary_cursor += 1
        fast = interleaved
    else:
        fast = communication + urgent
    if urgent and ordinary:
        # Keep priority useful without letting an ongoing stream of new rows
        # turn already-admitted ordinary work into permanent debt. Four fast
        # batches receive slots in each five-update cycle; ordinary receives
        # the fifth. `schedule_offset` is the checkpointed optimizer step,
        # so safe stops and resident appends continue the same cycle position.
        cycle = (True, True, True, True, False)
        cursor = schedule_offset % len(cycle)
        interleaved = []
        fast_cursor = ordinary_cursor = 0
        while fast_cursor < len(fast) or ordinary_cursor < len(ordinary):
            want_fast = cycle[cursor]
            if want_fast and fast_cursor < len(fast):
                interleaved.append(fast[fast_cursor]); fast_cursor += 1
            elif not want_fast and ordinary_cursor < len(ordinary):
                interleaved.append(ordinary[ordinary_cursor]); ordinary_cursor += 1
            elif fast_cursor < len(fast):
                interleaved.append(fast[fast_cursor]); fast_cursor += 1
            else:
                interleaved.append(ordinary[ordinary_cursor]); ordinary_cursor += 1
            cursor = (cursor + 1) % len(cycle)
        batches = interleaved
    else:
        batches = fast + ordinary
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
        embedding_ids={id(model.port.embedding_weight)}, force_adamw_ids={id(parameter) for _,parameter in ordinary},
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
    from ..model.native_graph_plan import invocation_dict
    digest = hashlib.sha256()
    for item in examples:
        row = {'id': item.id, 'groups': item.source_groups, 'split': item.split,
               'tokenizer': item.tokenizer_fingerprint, 'targets': item.target_weights,
               'nodes': [{'name': node.name, 'token_ids': node.token_ids,
                          'imports': list(invocation_dict(node)['imports']),
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
                                       stop=edge.stop, at=edge.at, logical=edge.logical,stop_gradient=edge.stop_gradient)
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
        from ..model.native_weight_carrier import ParameterCotangentScope
        with ParameterCotangentScope.for_model(model, mode='vjp') as cotangents, context:
            if backend is not None:
                with backend.autocast():
                    loss, details = packed_graph_loss(model, batch,
                        checkpoint_layers=checkpoint_layers, chunk_tokens=chunk_tokens,
                        objective=objective, share_causal_prefixes=getattr(model, 'share_causal_prefixes', False),
                        collect_accuracy=False)
            else:
                loss, details = packed_graph_loss(model, batch,
                    checkpoint_layers=checkpoint_layers, chunk_tokens=chunk_tokens,
                    objective=objective, share_causal_prefixes=getattr(model, 'share_causal_prefixes', False),
                    collect_accuracy=False)
            loss.backward()
            cotangents.finish()
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


def _loss_query_state_chunks(hidden, query_ids, chunk_tokens, *, unique_queries):
    """Gather a graph's loss states once when its physical query rows are distinct.

    Repeated indices retain the old per-chunk indexing path, whose scatter-add
    accumulation order is part of the gradient law for shared physical rows.
    """
    if unique_queries:
        return hidden.index_select(0, query_ids).split(chunk_tokens)
    return tuple(hidden[query_ids[start:start + chunk_tokens]]
                 for start in range(0, query_ids.numel(), chunk_tokens))


def _native_ce_sum(model, hidden, query_ids, target_ids, weights, *, chunk_tokens, collect_accuracy, unique_queries):
    """The shared checkpointed readout kernel; callers own normalization."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    state_chunks = _loss_query_state_chunks(hidden, query_ids, chunk_tokens,
                                            unique_queries=unique_queries)
    target_chunks = target_ids.split(chunk_tokens)
    weight_chunks = weights.split(chunk_tokens)
    total = hidden.new_zeros((), dtype=torch.float32)
    correct = hidden.new_zeros((), dtype=torch.float32) if collect_accuracy else None
    for states, labels, selected in zip(state_chunks, target_chunks, weight_chunks):
        def loss_chunk(states, labels, selected, *, measure_accuracy):
            logits = model.logits(states)
            ce = F.cross_entropy(logits.float(), labels, reduction='none')
            loss_sum = (ce * selected).sum()
            if not measure_accuracy:
                return loss_sum
            return loss_sum, ((logits.argmax(-1) == labels).float() * selected).sum()
        chunk = checkpoint(loss_chunk, states, labels, selected, measure_accuracy=collect_accuracy,
            use_reentrant=False)
        if collect_accuracy:
            chunk_loss, chunk_correct = chunk
            correct = correct + chunk_correct
        else:
            chunk_loss = chunk
        total = total + chunk_loss
    return total, correct


def packed_graph_loss(model, examples, *, checkpoint_layers=True, chunk_tokens=128, objective=None,
                      share_causal_prefixes=False, collect_accuracy=True):
    """One backbone graph forward for a minibatch; mean the original per-graph losses."""
    if objective is None and any(not any(weight>0 for weight in example.target_weights) for example in examples):
        raise ValueError('native graph has no ordinary CE targets; its declared offline/policy objective is required')
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
    graph_forward = getattr(objective, 'forward_graph', None) if objective is not None else None
    result = (graph_forward(model, context, forward_inputs or {}) if graph_forward is not None else
              model(plan, retain_states=False, **(forward_inputs or {})))
    if objective is not None:
        loss, details = objective(model, result, context)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss.detach()):
            raise FloatingPointError('native policy objective must return a finite scalar loss')
        return loss, details
    hidden = result['h_final'][0]
    # BF16 head adjoints round each original use before prefix fan-in. Folding
    # those uses earlier changes that declared law, despite equal scalar CE.
    # Keep its original path; the qualified FP32 case has no such cast boundary.
    reuse_readouts = shared is not None and not collect_accuracy and hidden.dtype == torch.float32
    readout_index, readout_queries, readout_targets, readout_groups, readout_weights = {}, [], [], [], []
    readout_position = 0
    losses, details, detail_weights, detail_accuracies = [], [], [], []
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
        denominator = token_weights.sum()
        if reuse_readouts:
            for query, token in zip(queries, targets):
                key = (query, token)
                if key not in readout_index:
                    readout_index[key] = len(readout_queries)
                    readout_queries.append(query)
                    readout_targets.append(token)
                    readout_groups.append([])
                readout_groups[readout_index[key]].append(readout_position)
                readout_position += 1
            readout_weights.append(token_weights / denominator / len(examples))
        else:
            total, correct = _native_ce_sum(model, hidden, query_ids, target_ids, token_weights,
                chunk_tokens=chunk_tokens, collect_accuracy=collect_accuracy,
                unique_queries=len(set(queries)) == len(queries))
            losses.append(total / denominator)
        detail_weights.append(denominator.detach())
        if collect_accuracy:
            detail_accuracies.append((correct / denominator).detach())
        details.append({'weight': None,
                        'boundary_targets': boundary_targets,
                        'supervised_tokens': len(queries),
                        'packed_physical_tokens': plan.token_count,
                        'original_physical_tokens': sum(item.plan.token_count for item in examples),
                        'accuracy': None})
    original_readout_rows = sum(item['supervised_tokens'] for item in details)
    if reuse_readouts:
        combined = torch.cat(readout_weights)
        order = torch.tensor([position for group in readout_groups for position in group],
                             dtype=torch.long, device=hidden.device)
        lengths = torch.tensor([len(group) for group in readout_groups], dtype=torch.long, device=hidden.device)
        # Fixed original graph/position order within each exact term. Avoid
        # nondeterministic CUDA atomic addition when assembling coefficients.
        selected = torch.segment_reduce(combined.index_select(0, order), 'sum', lengths=lengths)
        total, _ = _native_ce_sum(model, hidden,
            torch.tensor(readout_queries, dtype=torch.long, device=hidden.device),
            torch.tensor(readout_targets, dtype=torch.long, device=hidden.device), selected,
            chunk_tokens=chunk_tokens, collect_accuracy=False,
            unique_queries=len(set(readout_queries)) == len(readout_queries))
        # Coefficients already include each original graph's denominator and
        # its 1/batch multiplicity. Do not normalize this scalar again.
        losses = [total]
    for detail in details:
        detail['readout_forward_rows'] = len(readout_queries) if reuse_readouts else original_readout_rows
        detail['readout_original_rows'] = original_readout_rows
    # Resolve per-graph scalars in one host transfer. Calling float(tensor)
    # inside the graph loop inserted a device synchronization for every graph
    # (and again for its accuracy), serializing the otherwise asynchronous
    # backbone forward. Training does not consume per-graph accuracy, so its
    # ordinary path reads only the denominators needed for boundary telemetry.
    if collect_accuracy:
        metrics = torch.stack((torch.stack(detail_weights), torch.stack(detail_accuracies)), dim=1)
        host_metrics = metrics.detach().cpu().tolist()
    else:
        host_weights = torch.stack(detail_weights).detach().cpu().tolist()
        host_metrics = [(weight, None) for weight in host_weights]
    for detail, (weight, accuracy) in zip(details, host_metrics):
        detail['weight'], detail['accuracy'] = float(weight), None if accuracy is None else float(accuracy)
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


def _evaluate(model, examples, *, checkpoint_layers, chunk_tokens, backend=None, graph_classes=None, objective=None):
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
                if objective is not None:
                    loss,detail=objective.evaluate(model,example,checkpoint_layers=checkpoint_layers,
                        chunk_tokens=chunk_tokens,boundary_ids=boundary_ids)
                else:
                    loss, detail = graph_loss(model, example, checkpoint_layers=checkpoint_layers,
                                              chunk_tokens=chunk_tokens, boundary_token_ids=boundary_ids)
                for token, sums in detail['boundary_sums'].items():
                    for key, value in sums.items():
                        boundary_sums[token][key] += value
                weight = detail['weight']
                if graph_classes is not None:
                    name = detail.get('metric') or graph_classes[example.id]
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
                      previous_result=None, objective=None):
    from .native_held import held_selection
    coverage = coverage or {}
    indices, next_coverage, selection = held_selection(source.entries, coverage,
        source_sha256=source.file_sha256, token_budget=args.held_eval_tokens,
        graph_budget=args.held_eval_graphs, final=final, previous_result=previous_result)
    cursor = next_coverage['cursor']
    started = time.monotonic()
    result = _evaluate(model, source.iter_indices(indices), checkpoint_layers=args.checkpoint_layers,
                       chunk_tokens=args.chunk_tokens, backend=backend, objective=objective,
                       graph_classes={source.entries[i]['id']: ('communication' if source.entries[i].get('communication')
                                      else 'ordinary') for i in indices})
    result.update(scope='full' if len(indices) == len(source) else 'cadence-subset',
                  source_sha256=source.file_sha256,
                  source_graphs=len(source), cursor_before=int(coverage.get('cursor', 0)),
                  cursor_after=cursor, seconds=time.monotonic()-started,
                  selection=selection)
    return result, next_coverage


def _retain_current_held_coverage(metrics, training, held):
    metrics.pop('held_coverage',None)
    coverage=(training.get('metrics') or {}).get('held_coverage')
    if isinstance(coverage,dict) and coverage.get('source_sha256')==held.file_sha256:
        metrics['held_coverage']=copy.deepcopy(coverage)


def _resume_identity_for_comparison(identity):
    """Normalize legacy default-backend omission without mutating checkpoint state."""
    previous = dict(identity or {})
    previous.setdefault('cotangent_accumulator_backend', 'cpu_file')
    return previous


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
                                 checkpoint_sha256, forward_atol, gradient_atol, device, backend=None,
                                 previous_attention_precision=None, previous_norm_execution=None, optimizer=None,
                                 previous_shared_execution=None):
    """Check changed production arithmetic on the restored live weights, once."""
    any_linear_changed = config.get('native_linear_precision', 'ieee') != previous_precision
    linear_changed = config.get('native_linear_precision', 'ieee') == 'bf16x6' and previous_precision != 'bf16x6'
    attention_changed = (previous_attention_precision is not None
        and config.get('native_attention_precision', 'ieee') != previous_attention_precision)
    from ..model.native_parameter_precision import norm_execution_changes_arithmetic
    norm_changed = (previous_norm_execution is not None
        and norm_execution_changes_arithmetic(model, previous_norm_execution))
    shared_changed=(previous_shared_execution is not None
        and config.get('native_shared_execution','source-v1')!=previous_shared_execution)
    if not linear_changed and not attention_changed and not norm_changed and not shared_changed:
        return None
    if forward_atol is None or gradient_atol is None:
        raise ValueError('native arithmetic transition requires declared forward and gradient tolerances')
    from ..eval.native_state import qualify_existing
    saved_rng = capture_training_rng_state(device)
    attention_report=None;attention_only_equivalent=False
    try:
        if attention_changed:
            if optimizer is None:
                raise ValueError('attention transition requires its actual production optimizer inventory')
            from ..eval.native_attention_transition import qualify_attention_transition
            attention_report=qualify_attention_transition(model,tokenizer,roles,config,optimizer=optimizer,
                previous_mode=previous_attention_precision,candidate_mode=config['native_attention_precision'],
                checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,gradient_atol=gradient_atol,
                backend=backend)
            report=attention_report['candidate_cached_reference']
            attention_only_equivalent=(not any_linear_changed and not norm_changed and not shared_changed
                and(attention_report.get('transition_equivalence')or{}).get('passed')is True)
            if not attention_report['passed']and not attention_only_equivalent:
                raise ValueError('native attention transition failed unchanged cached-reference and complete cross-mode checks: '+
                    json.dumps(attention_report,sort_keys=True))
        elif shared_changed and backend is not None:
            if optimizer is None:
                raise ValueError('shared execution transition requires actual production optimizer inventory')
            from ..eval.native_backend import qualify_training_backend
            report=qualify_training_backend(model,tokenizer,roles,config,backend=backend,
                checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,
                gradient_atol=gradient_atol,device=device,
                gradient_parameters=optimizer._native_named_parameters)['runtime_reference']
        else:
            if shared_changed and optimizer is None:
                raise ValueError('shared execution transition requires actual production optimizer inventory')
            with backend.native_modules()if backend is not None else contextlib.nullcontext():
                report=qualify_existing(model,tokenizer,roles,config,
                    checkpoint_sha256=checkpoint_sha256,forward_atol=forward_atol,gradient_atol=gradient_atol,
                    **({'gradient_parameters':optimizer._native_named_parameters}if shared_changed else{}))
    finally:
        restore_training_rng_state(saved_rng,device)
    if not report['passed']and not attention_only_equivalent:
        raise ValueError('native arithmetic transition failed the unchanged cached-reference checks: ' +
                         json.dumps(report, sort_keys=True))
    result = {'schema': 'natlang.native-arithmetic-transition-check/1',
            'from': previous_precision, 'to': config['native_linear_precision'],
            'parent_weights_sha256': checkpoint_sha256, 'runtime_reference': report}
    if attention_changed:
        result['attention'] = {'from': previous_attention_precision,
                               'to': config['native_attention_precision']}
        result['attention_complete_comparison']=attention_report
        result['attention_transition_equivalence']=attention_report['transition_equivalence']
        result['inherited_cached_reference_debt']=not report['passed']
    if norm_changed:
        result['norm_execution'] = {'from':previous_norm_execution,'to':model.native_norm_execution}
    if shared_changed:
        result['shared_execution']={'from':previous_shared_execution,'to':model.native_shared_execution}
    return result


def continuation_predecessors(training, declarations, *, stage_id):
    """Verify current or exactly witnessed historical append predecessors."""
    from .native_resident import bound
    from .recipe import load_recipe, recipe_identity_sha256, resolve_stage_inputs
    prior = dict((training.get('options') or {}).get('source_predecessors') or {})
    receipts = training.get('source_receipts') or []
    history = ((training.get('options') or {}).get('resident_adoption') or {}).get('history') or []
    for role, position in (('continued_native_graphs', 0), ('continued_held_native_graphs', 1)):
        path = declarations.get(role)
        if path is None:
            continue
        value = sha256_file_hex_cached(path)
        witnessed = False
        if prior.get(role) != value and (len(receipts) <= position or receipts[position]['sha256'] != value):
            for frame in reversed(history):
                if (len(receipts) < 2 or any((frame.get(key) or {}).get('sha256') != receipts[index]['sha256']
                        for index, key in enumerate(('train', 'held')))):
                    continue
                declared = frame['recipe']
                recipe_path = bound({'path': declared['path'],
                    'sha256': declared.get('file_sha256', declared['sha256'])})
                if recipe_identity_sha256(recipe_path) != declared['sha256']:
                    raise ValueError('resident predecessor recipe identity changed')
                recipe = load_recipe(recipe_path)
                stages = [stage for stage in recipe['stages'] if stage['id'] == stage_id
                          and stage['kind'] == 'native_state_sft']
                if len(stages) != 1:
                    continue
                inputs = resolve_stage_inputs(recipe, stages[0], {})
                witnessed = (all(inputs[key]['sha256'] == receipts[index]['sha256']
                    for index, key in enumerate(('native_graphs', 'held_native_graphs')))
                    and (inputs.get(role) or {}).get('sha256') == value)
                if witnessed:
                    break
            if not witnessed:
                raise ValueError('native data continuation predecessor differs from the actual checkpoint: ' + role)
        prior[role] = value
    return prior


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-graphs', type=Path, required=True)
    parser.add_argument('--ce-supervision-policy', type=Path,
        help='explicit source-bound balanced CE continuation; preserves raw row credits and optimizer/RNG')
    parser.add_argument('--foundation-native-graphs', type=Path)
    parser.add_argument('--pending-native-sources', action='append', default=[])
    parser.add_argument('--refresh-at-checkpoint', action=argparse.BooleanOptionalAction, default=False,
        help='return an explicit incomplete continuation for semantic refresh at the next declared checkpoint boundary')
    parser.add_argument('--refresh-after-updates', type=int,
        help='refresh after this many new optimizer updates since restoration; independent of periodic checkpoint alignment')
    parser.add_argument('--learn-reasoning-boundaries', action=argparse.BooleanOptionalAction, default=False,
                        help='explicitly add learned existing reasoning token rows to an old parent')
    parser.add_argument('--foundation-communication-batches', type=int,
        help='while the exact declared foundation is pending, communication batches per priority ordinary batch')
    parser.add_argument('--urgent-cohort-fairness', action=argparse.BooleanOptionalAction, default=False,
        help='balance positive-priority source cohorts using actual completed indexed graph token exposure')
    parser.add_argument('--source-classification-migration', type=Path,
        help='explicit source-bound index/progress classification continuation')
    parser.add_argument('--communication-foundation', action=argparse.BooleanOptionalAction, default=False,
        help='pause at the complete admitted communication cohort with mandatory ordinary credits retained')
    parser.add_argument('--refresh-foundation-readiness', action=argparse.BooleanOptionalAction, default=False,
        help='at an explicitly owned stage boundary, recheck an already earned finite cohort on current weights')
    parser.add_argument('--foundation-readiness-policy', choices=('at-boundary','on-shared-request'), default='at-boundary',
        help='continue supervised work until a shared candidate rendezvous requests exact qualification')
    parser.add_argument('--foundation-qualification-request', type=Path,
        help='immutable shared candidate request; qualification still checks actual current weights')
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
    from ..model.native_precision import PRODUCTION_NATIVE_LINEAR_PRECISIONS, NATIVE_ATTENTION_PRECISIONS
    parser.add_argument('--native-linear-precision', choices=PRODUCTION_NATIVE_LINEAR_PRECISIONS, default=None,
        help='explicit native GEMM arithmetic transition; preserves source dtype, optimizer, RNG and row exposure')
    parser.add_argument('--native-attention-precision', choices=NATIVE_ATTENTION_PRECISIONS, default=None,
        help='explicit per-model attention arithmetic; restores optimizer, RNG and row exposure before qualification')
    from .native_optimizer_migration import POLICY as NATIVE_OPTIMIZER_MIGRATION
    parser.add_argument('--native-norm-execution', choices=('source-v1','stable-v1'))
    parser.add_argument('--native-shared-execution',choices=('source-v1','stable-v1'))
    parser.add_argument('--history-reasoning-policy', choices=('keep',), default=None,
        help='explicitly render earlier assistant reasoning in complete training and serving histories')
    parser.add_argument('--native-ordinary-precision', choices=('fp32-norms-v1',))
    parser.add_argument('--native-optimizer-migration',choices=(NATIVE_OPTIMIZER_MIGRATION,),default=None,
        help='explicit new AdamW history for formerly misrouted ordinary parameters; preserves original Muon buffers and all unaffected state')
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
    parser.add_argument('--checkpoint-publication', action='store_true')
    parser.add_argument('--checkpoint-publication-namespace', default='/work/native-production')
    parser.add_argument('--checkpoint-publication-readonly-input-roots', default='[]')
    parser.add_argument('--checkpoint-input-offload', action=argparse.BooleanOptionalAction, default=None,
        help='store nonreentrant layer-checkpoint inputs on host RAM or explicit file storage; omission inherits the saved mode')
    parser.add_argument('--checkpoint-input-offload-directory', default=None,
        help='private file-backed checkpoint inputs in this directory; omission inherits, empty string selects CPU')
    parser.add_argument('--cotangent-accumulator-backend', choices=('cpu_file', 'device_fp32'),
        default=None, help='explicit FP32 parameter-cotangent storage; default preserves historical cpu_file behavior')
    parser.add_argument('--checkpoint-layers', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--chunk-tokens', type=int, default=128)
    parser.add_argument('--ffn-chunk-tokens', type=int, default=None,
        help='Existing token-local FFN workspace bound; omitted inherits saved value, zero disables chunking')
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
    parser.add_argument('--policy-capacity-request',default=None,
        help='exact admitted whole-policy source/objective MAX request; policy SGD always qualifies before updates')
    parser.add_argument('--memory-qualification', default=None,
        help='bound prepared new-source capacity request JSON; checked read-only after exact restore')
    parser.add_argument('--recipe-binding', default=None)
    parser.add_argument('--stage-id', default=None)
    parser.add_argument('--prerequisite-stages', default='[]')
    parser.add_argument('--objective-spec', type=Path, default=None)
    parser.add_argument('--preference-beta',type=float)
    parser.add_argument('--preference-margin',type=float)
    return parser


def configure_ffn_chunk_tokens(port, requested, saved_training):
    """Select the existing token-local port execution without resetting a resume."""
    value=(saved_training.get('options')or{}).get('ffn_chunk_tokens',0)if requested is None else requested
    if type(value)is not int or value<0:
        raise ValueError('ffn-chunk-tokens must be a nonnegative integer')
    port.ffn_chunk_tokens=value
    return value


def _foundation_readiness_boundary(foundation, prior_foundations, *, completed,
                                  new_optimizer_updates, policy_operation, signaled,
                                  refresh_requested=False, readiness_policy='at-boundary', qualification_requested=False):
    """Select a first, completed-continuation, or explicitly owned stage check."""
    if not foundation or foundation.get('complete') is not True or signaled:
        return False
    if readiness_policy == 'on-shared-request' and not qualification_requested:
        return False
    cohort = foundation.get('cohort_sha256')
    if not cohort:
        return False
    if cohort not in prior_foundations:
        return True
    if (refresh_requested or qualification_requested) and not policy_operation:
        return True
    # An earned finite cohort can be re-certified on newer weights after a
    # complete supervised continuation. Partial refresh checkpoints and
    # operational stops carry the old cohort unchanged without repeating the
    # full task/runtime qualification.
    return bool(completed and new_optimizer_updates > 0 and not policy_operation)


def main(argv=None):
    from .loop import StopSignal
    requested_refresh = StopSignal().install((signal.SIGUSR2,))
    try:
        return _main(argv, requested_refresh=requested_refresh)
    finally:
        requested_refresh.restore()


def _main(argv=None, *, requested_refresh):
    args = build_parser().parse_args(argv)
    if args.ffn_chunk_tokens is not None and args.ffn_chunk_tokens<0:
        raise ValueError('ffn-chunk-tokens must be a nonnegative integer')
    if args.refresh_after_updates is not None and args.refresh_after_updates < 1:
        raise ValueError('refresh-after-updates must be a positive integer')
    history_reasoning_transition = None
    if args.foundation_communication_batches is not None and args.foundation_communication_batches < 1:
        raise ValueError('foundation communication batches must be a positive integer')
    resident_state = None
    if args.resident_data_directory is not None:
        from .native_resident import recover, bound as resident_bound
        resident_state = recover(args.resident_data_directory)
        if resident_state and resident_state['payload']['stage_id'] == args.stage_id:
            args.native_graphs = Path(resident_state['payload']['train']['path'])
            args.held_native_graphs = Path(resident_state['payload']['held']['path'])
            args.recipe_binding = json.dumps(resident_state['payload']['recipe'])
            proof=resident_state['payload']['plan']['stage_inputs'][args.stage_id].get('source_classification_migration')
            args.source_classification_migration=resident_bound(proof) if proof else None
    checkpoint_options = {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}
    from .native_supervision import load_policy
    ce_binding = ({'path':str(args.ce_supervision_policy.resolve()),
                   'sha256':sha256_file_hex_cached(args.ce_supervision_policy)}
                  if args.ce_supervision_policy else None)
    ce_policy = load_policy(ce_binding)
    checkpoint_options['ce_supervision_policy_binding'] = ce_binding
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
    if args.checkpoint_publication:
        from ..model.native_checkpoint_control import saved_publication_hook
        roots=json.loads(args.checkpoint_publication_readonly_input_roots)
        if not isinstance(roots,list) or not all(isinstance(root,str) for root in roots):
            raise ValueError('checkpoint publication readonly roots must be a JSON string list')
        policy.on_saved=saved_publication_hook(namespace=args.checkpoint_publication_namespace,
                                               readonly_input_roots=roots)
    policy.install_signal_handlers()
    resume = None
    state = parent_state = None
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
            parent_state = None
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
            device=args.device, dtype=getattr(torch, args.dtype),
            native_norm_execution=args.native_norm_execution or 'stable-v1',
            history_reasoning_policy=args.history_reasoning_policy)
    if args.history_reasoning_policy is not None:
        previous_history_policy = config.get('history_reasoning_policy')
        if previous_history_policy != args.history_reasoning_policy:
            config['history_reasoning_policy'] = args.history_reasoning_policy
            history_reasoning_transition = {
                'from': previous_history_policy or 'backbone-default',
                'to': args.history_reasoning_policy,
                'model_weights_optimizer_rng_exposure': 'preserved',
                'rendering_change': True}
            # A restored parent tokenizer is already bound to its historical
            # profile. Apply an explicit continuation profile through the
            # shared binder only after recording the transition above.
            from ..serve.chat import bind_history_reasoning
            bind_history_reasoning(tokenizer, policy=args.history_reasoning_policy, allow_transition=True)
    previous_norm_declaration = config.get('native_norm_execution')
    previous_shared_declaration=config.get('native_shared_execution')
    previous_shared_execution=config.get('native_shared_execution','source-v1')
    if args.native_shared_execution is not None:
        from ..model.shared_execution import configure_shared_execution
        configure_shared_execution(model,args.native_shared_execution)
        config['native_shared_execution']=args.native_shared_execution
    previous_norm_execution = config.get('native_norm_execution', 'source-v1')
    if args.native_norm_execution is not None:
        from ..model.native_parameter_precision import configure_norm_execution
        configure_norm_execution(model, args.native_norm_execution)
        config['native_norm_execution'] = args.native_norm_execution
    norm_upgrade = None
    if args.native_ordinary_precision is not None:
        from ..model.native_parameter_precision import upgrade_norm_precision
        norm_upgrade = upgrade_norm_precision(model, config, args.native_ordinary_precision)
        checkpoint_options['ordinary_precision_upgrade'] = norm_upgrade
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
    from ..model.native_precision import set_native_attention_precision
    previous_attention_precision = config.get('native_attention_precision', 'ieee')
    attention_precision = args.native_attention_precision or previous_attention_precision
    set_native_attention_precision(model, attention_precision)
    if args.native_attention_precision is not None:
        config['native_attention_precision'] = attention_precision
    saved_training = resume or shared_resume or {}
    saved_options = saved_training.get('options') or {}
    checkpoint_options['foundation_qualification_history'] = list(saved_options.get('foundation_qualification_history') or [])
    snapshot_ancestors = list(saved_options.get('native_snapshot_ancestors') or [])
    if any(not isinstance(value,str) or len(value)!=64 or any(char not in '0123456789abcdef' for char in value)
           for value in snapshot_ancestors):
        raise ValueError('native snapshot ancestry must retain exact loaded checkpoint identities')
    if resume is not None or shared_resume is not None:
        loaded_snapshot = sha256_file_hex_cached(args.continue_from if parent_continuation else policy.slot)
        if loaded_snapshot not in snapshot_ancestors:
            snapshot_ancestors.append(loaded_snapshot)
    checkpoint_options['native_snapshot_ancestors'] = snapshot_ancestors
    previous_cotangent_backend = saved_options.get('cotangent_accumulator_backend', 'cpu_file')
    args.cotangent_accumulator_backend = (previous_cotangent_backend
        if args.cotangent_accumulator_backend is None else args.cotangent_accumulator_backend)
    if args.cotangent_accumulator_backend not in ('cpu_file', 'device_fp32'):
        raise ValueError('native checkpoint declares an unknown cotangent accumulator backend')
    if saved_training and args.cotangent_accumulator_backend != previous_cotangent_backend:
        if not args.memory_qualification:
            raise ValueError('cotangent accumulator transition requires fresh exact-parent capacity qualification')
        history = list(saved_options.get('cotangent_accumulator_history') or [])
        history.append({'from': previous_cotangent_backend,
            'to': args.cotangent_accumulator_backend,
            'at_global_step': int(saved_training.get('global_step', saved_training.get('step', 0))),
            'weights_optimizer_rng_exposure': 'preserved',
            'capacity_request': args.memory_qualification})
        checkpoint_options['cotangent_accumulator_history'] = history
    else:
        checkpoint_options['cotangent_accumulator_history'] = list(
            saved_options.get('cotangent_accumulator_history') or [])
    checkpoint_options['cotangent_accumulator_backend'] = args.cotangent_accumulator_backend
    model.native_cotangent_accumulator_backend = args.cotangent_accumulator_backend
    previous_recipe=(saved_training.get('options') or {}).get('recipe_binding')
    if previous_recipe is not None:
        old_recipe=json.loads(previous_recipe) if isinstance(previous_recipe,str) else previous_recipe
        if args.recipe_binding is None:
            args.recipe_binding=(previous_recipe if isinstance(previous_recipe,str) else json.dumps(previous_recipe))
            checkpoint_options['recipe_binding']=args.recipe_binding
        else:
            new_recipe=json.loads(args.recipe_binding)
            if old_recipe.get('lineage_sha256',old_recipe['sha256'])!=new_recipe.get('lineage_sha256',new_recipe['sha256']):
                raise ValueError('native continuation changed its preserved recipe lineage')
    checkpoint_options['withdrawal_history']=list((saved_training.get('options') or {}).get('withdrawal_history') or [])
    token_history = list((saved_training.get('options') or {}).get('token_row_upgrade_history') or [])
    if token_upgrade and (token_upgrade['added'] or token_upgrade.get('promoted_parameters')):
        token_history.append({**token_upgrade, 'at_global_step': int(saved_training.get('global_step', saved_training.get('step', 0))),
                              'weights_optimizer_rng_exposure': 'preserved'})
    checkpoint_options['token_row_upgrade_history'] = token_history
    norm_history = list((saved_training.get('options') or {}).get('ordinary_precision_history') or [])
    if norm_upgrade and norm_upgrade['previous_profile'] != norm_upgrade['profile']:
        norm_history.append({**norm_upgrade, 'at_global_step':int(saved_training.get('global_step',saved_training.get('step',0)))})
    checkpoint_options['ordinary_precision_history'] = norm_history
    execution_history = list((saved_training.get('options') or {}).get('native_norm_execution_history') or [])
    if config.get('native_norm_execution', 'source-v1') != previous_norm_execution:
        execution_history.append({'from':previous_norm_execution,'to':config['native_norm_execution'],
            'at_global_step':int(saved_training.get('global_step',saved_training.get('step',0))),
            'weights_optimizer_rng_exposure':'preserved'})
    checkpoint_options['native_norm_execution_history'] = execution_history
    shared_history=list((saved_training.get('options')or{}).get('native_shared_execution_history')or[])
    if config.get('native_shared_execution','source-v1')!=previous_shared_execution:
        shared_history.append({'from':previous_shared_execution,'to':config['native_shared_execution'],
            'at_global_step':int(saved_training.get('global_step',saved_training.get('step',0))),
            'weights_optimizer_rng_exposure':'preserved'})
    checkpoint_options['native_shared_execution_history']=shared_history
    history_profile_history = list((saved_training.get('options') or {}).get('history_reasoning_policy_history') or [])
    if history_reasoning_transition is not None:
        history_profile_history.append({**history_reasoning_transition,
            'at_global_step':int(saved_training.get('global_step',saved_training.get('step',0)))})
    checkpoint_options['history_reasoning_policy_history'] = history_profile_history
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
    attention_history = list((saved_training.get('options') or {}).get('native_attention_precision_history') or [])
    if attention_precision != previous_attention_precision:
        attention_history.append({'from':previous_attention_precision,'to':attention_precision,
            'at_global_step':int(saved_training.get('global_step',saved_training.get('step',0))),
            'weights_optimizer_rng_exposure':'preserved' if saved_training else 'fresh'})
    checkpoint_options['native_attention_precision_history'] = attention_history
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
        # A direct DPO/RAW frame describes its own objective source. Ordinary
        # continuation must use the already carried ordinary frame, which the
        # normal progress restore selects later with the same optimizer/RNG.
        source_resume = ((carried_progress.get(args.stage_id or 'native-graph-ce') or resume)
            if parent_continuation and args.objective_spec is None else resume)
        pending_reference = resident_state['payload']['plan']['stage_inputs'][args.stage_id].get('pending_native_sources')
        args.pending_native_sources = [Path(pending_reference['path'])] if pending_reference else []
        checkpoint_options['pending_native_sources'] = [str(path) for path in args.pending_native_sources]
        prior = (source_resume.get('source_receipts') or [{}])[0]
        if prior.get('sha256') != resident_state['payload']['train']['sha256']:
            args.continued_native_graphs = Path(prior['path'])
            checkpoint_options['continued_native_graphs'] = str(args.continued_native_graphs)
        prior_held = (source_resume.get('source_receipts') or [{}, {}])[1]
        if prior_held.get('sha256') != resident_state['payload']['held']['sha256']:
            args.continued_held_native_graphs = Path(prior_held['path'])
            checkpoint_options['continued_held_native_graphs'] = str(args.continued_held_native_graphs)
    train = NativeGraphSource(args.native_graphs, fingerprint, required_split='train', role_ids=roles.ids)
    held = NativeGraphSource(args.held_native_graphs, fingerprint, required_split='test', role_ids=roles.ids)
    train_groups, held_groups = train.groups, held.groups
    if train_groups & held_groups:
        raise ValueError('native train/held source groups overlap: ' + ', '.join(sorted(train_groups & held_groups)[:8]))
    from .native_operation_report import source_receipt, SourceIdentityView
    source_receipts = [source_receipt(train), source_receipt(held)]
    source_row_bindings = train.bindings
    from .native_continuation import reject_withdrawn_targets
    reject_withdrawn_targets(saved_training,source_row_bindings)
    held_row_bindings = held.bindings
    source_hash = hashlib.sha256(json.dumps(
        [source_receipts[0]['sha256'], source_receipts[1]['sha256'],
         sorted(source_row_bindings.items()), sorted(held_row_bindings.items())],
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    source_order = train.ordered_ids()
    from .native_pending_sources import load_pending_sources
    checkpoint_options['_pending_native_source_declarations'] = [
        load_pending_sources(path, tokenizer_fingerprint=fingerprint) for path in args.pending_native_sources]
    checkpoint_options['source_classification_history'] = list((saved_training.get('options') or {}).get('source_classification_history') or [])
    checkpoint_options['_communication_ids']=[entry['id'] for entry in train.entries if entry.get('communication')]
    checkpoint_options['_producer_consumer_ids'] = [entry['id'] for entry in train.entries
        if entry.get('producer_consumer', entry.get('communication', False))]
    checkpoint_options['_external_state_consumption_ids'] = [entry['id'] for entry in train.entries
        if entry.get('external_state_consumption')]
    foundation_source = (NativeGraphSource(args.foundation_native_graphs, fingerprint,
                                           required_split='train', role_ids=roles.ids)
                         if args.foundation_native_graphs else train)
    if args.foundation_communication_batches is not None and args.foundation_native_graphs is None:
        raise ValueError('foundation scheduling requires the exact declared foundation source')
    if args.communication_foundation and args.foundation_native_graphs is None:
        raise ValueError('communication foundation requires its complete declared admitted source manifest')
    if args.refresh_foundation_readiness and not args.communication_foundation:
        raise ValueError('foundation-readiness refresh requires its exact declared communication cohort')
    if args.foundation_qualification_request and (not args.communication_foundation or args.foundation_readiness_policy != 'on-shared-request'):
        raise ValueError('shared qualification request requires its explicit foundation readiness policy')
    if args.communication_foundation:
        from .native_continuation import verify_initial_foundation_cohort
        foundation_frame = carried_progress.get(args.stage_id or 'native-graph-ce') or {}
        verify_initial_foundation_cohort(train, foundation_source,
            already_declared=bool((foundation_frame.get('options') or {}).get('communication_foundation')))
    objective = None
    objective_binding = None
    if args.objective_spec is not None:
        spec_sha256 = sha256_file_hex_cached(args.objective_spec)
        objective_binding = {'path': str(args.objective_spec.resolve()), 'sha256': spec_sha256}
        spec = json.loads(args.objective_spec.read_text())
        if spec.get('schema')=='natlang.native-offline-pairwise-objective/1' and spec.get('kind')=='native-offline-pairwise':
            from .native_preference import NativePreferenceObjective
            objective=NativePreferenceObjective(args.objective_spec,beta=args.preference_beta,margin=args.preference_margin)
            objective_binding.update(kind=objective.kind,beta=objective.beta,margin=objective.margin)
        else:
            if spec.get('schema') != 'natlang.native-policy-objective/1' or spec.get('kind') != 'native-policy':
                raise ValueError('native objective spec must declare an implemented native policy or offline pairwise objective')
            from ..rl.native_replay import NativePolicyObjective
            objective = NativePolicyObjective(args.objective_spec)
            if parent_continuation:
                actual = native_identity(model, config, checkpoint_sha256=sha256_file_hex_cached(args.continue_from))
                if objective.identity != actual:
                    raise ValueError('native policy objective differs from the actual loaded actor embedding/checkpoint identity')
    mandatory_objective=bool(getattr(objective,'mandatory_progress',False))
    policy_operation=objective is not None and not mandatory_objective
    held_objective=objective if mandatory_objective else None
    if (args.preference_beta is not None or args.preference_margin is not None) and not mandatory_objective:
        raise ValueError('preference beta/margin require the declared native offline pairwise objective')
    model.share_causal_prefixes = args.share_causal_prefixes
    training_identity = {'schema': TRAINING_SCHEMA, 'config': config, 'train_source_hash': source_hash,
                'schedule': 'row-visit-deficit/2',
                'held_sha256': source_receipts[1]['sha256'], 'optimizer': 'shared-muon-sr-in-backward-v1',
                'optimizer_options': {'native_optimizer_migration':args.native_optimizer_migration,'lr': args.learning_rate, 'learned_row_lr':args.learned_row_learning_rate, 'interface_row_lr':args.interface_row_learning_rate, 'weight_decay': args.weight_decay,
                                      'latent_ns': args.latent_ns},
                'batch_graphs': args.batch_graphs, 'batch_tokens': args.batch_tokens, 'seed': args.seed, 'rank': config['rank'],
                'cotangent_accumulator_backend': args.cotangent_accumulator_backend,
                'fp8': {'kind': args.fp8_training, 'recipe': args.fp8_recipe,
                        'amax_history_len': args.fp8_amax_history_len},
                'objective': objective_binding}
    if mandatory_objective:
        training_identity['objective_kind']=objective.kind
    appended_ids = set()
    continuation_training = shared_resume or resume or {}
    restored_frame = (restore_sft_progress(continuation_training, args.stage_id,
                        collected_progress=carried_progress if continuation_training is saved_training else None)
                      if parent_continuation and (objective is None or mandatory_objective) else None)
    if restored_frame is not None:
        resume = restored_frame
        restored_progress = True
    ce_prior_options = (resume or {}).get('options') or saved_options
    previous_ce = ce_prior_options.get('ce_supervision_policy_binding')
    inherited_ce = saved_options.get('ce_supervision_policy_binding') or previous_ce
    if inherited_ce and ce_binding is None:
        ce_binding = dict(inherited_ce)
        ce_policy = load_policy(ce_binding)
        args.ce_supervision_policy = Path(ce_binding['path'])
        checkpoint_options['ce_supervision_policy'] = str(args.ce_supervision_policy)
        checkpoint_options['ce_supervision_policy_binding'] = ce_binding
    ce_history = list(ce_prior_options.get('ce_supervision_policy_history') or [])
    if previous_ce != ce_binding:
        ce_history.append({'from':previous_ce,'to':ce_binding,
            'at_global_step':int(saved_training.get('global_step',saved_training.get('step',0))),
            'weights_optimizer_rng_exposure':'preserved', 'held_ce':'original-corpus-weights'})
    checkpoint_options['ce_supervision_policy_history'] = ce_history
    train.set_ce_policy(ce_policy if objective is None else None)
    checkpoint_options['ce_supervision_active'] = ce_policy is not None and objective is None
    from .native_source_classification import (migration_classes, reclassify_progress,
                                               require_explicit_reclassification)
    class_migration = (migration_classes(args.source_classification_migration, SourceIdentityView(train))
                       if args.source_classification_migration else {})
    require_explicit_reclassification(carried_progress, train.entries, class_migration)
    if class_migration:
        carried_progress = reclassify_progress(carried_progress, class_migration)
        if resume is not None:
            resume = reclassify_progress({'active': resume}, class_migration)['active']
    if args.source_classification_migration:
        proof_sha = sha256_file_hex_cached(args.source_classification_migration)
        if not any(item['proof_sha256'] == proof_sha for item in checkpoint_options['source_classification_history']):
            checkpoint_options['source_classification_history'].append(
                {'proof_sha256': proof_sha, 'graph_visits_created': 0})
    if args.withdrawn_native_graphs:
        if resume is None:
            raise ValueError('native withdrawal needs the actual checkpoint and exact predecessor source')
        from .native_continuation import withdraw_ordinary_rows
        origins={}
        applied=any(item.get('proof_sha256')==sha256_file_hex_cached(args.withdrawn_native_graphs)
            for item in (resume.get('options') or {}).get('withdrawal_history') or [])
        if not applied:
            if args.continued_native_graphs is None:
                raise ValueError('native withdrawal needs the actual checkpoint and exact predecessor source')
            old_source=NativeGraphSource(args.continued_native_graphs, fingerprint,
                required_split='train',role_ids=roles.ids)
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
        if mandatory_objective:
            checkpoint_options['_objective_required_visits']=dict((resume.get('options') or {}).get('_objective_required_visits') or {})
            checkpoint_options['offline_objective_revisions']=list((resume.get('options') or {}).get('offline_objective_revisions') or [])
        checkpoint_options['source_predecessors'] = continuation_predecessors(resume,
            {role: getattr(args, role) for role in ('continued_native_graphs', 'continued_held_native_graphs')},
            stage_id=args.stage_id or 'native-graph-ce')
        previous = _resume_identity_for_comparison(resume.get('identity'))
        stable = ('schema', 'optimizer', 'seed', 'rank', 'fp8', 'objective',
                  'cotangent_accumulator_backend')
        if previous.get('cotangent_accumulator_backend') != args.cotangent_accumulator_backend:
            stable = tuple(key for key in stable if key != 'cotangent_accumulator_backend')
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
        if norm_upgrade:
            previous_config = {**previous_config, 'native_ordinary_precision':config['native_ordinary_precision']}
        if args.native_norm_execution is not None:
            previous_config = {**previous_config, 'native_norm_execution':config['native_norm_execution']}
        if args.history_reasoning_policy is not None:
            previous_config = {**previous_config,
                'history_reasoning_policy':config['history_reasoning_policy']}
        if not same_source_precision_identity(previous_config, config):
            raise ValueError('native SFT continuation changes model identity beyond declared GEMM arithmetic')
        if previous.get('schedule') not in {'source-order-stream/1', 'row-visit-deficit/2'}:
            raise ValueError('native SFT checkpoint has an unsupported row-credit schedule')
        offline_tweak=mandatory_objective and args.objective_continuation and previous.get('objective_kind')==objective.kind
        if offline_tweak and previous.get('objective')!=objective_binding:
            transition={'from':previous.get('objective'),'to':objective_binding,
                'optimizer_rng_exposure':'preserved','previous_stage_complete':False,
                'additional_full_pair_visits':0}
            old_frame=collect_sft_progress(resume).get(args.stage_id or 'native-graph-ce') or {}
            old_quotas=resume.get('required_visits') or old_frame.get('required_visits') or {}
            old_counts=resume.get('exposure_counts') or {}
            parameter_change=any((previous.get('objective') or {}).get(key)!=objective_binding.get(key) for key in ('beta','margin'))
            if parameter_change and old_quotas and all(old_counts.get(key,0)>=visits for key,visits in old_quotas.items()):
                transition.update(previous_stage_complete=True,additional_full_pair_visits=1)
                checkpoint_options['_objective_required_visits'].update({key:int(old_counts.get(key,0))+1
                    for key in source_row_bindings})
            checkpoint_options['offline_objective_transition']=transition
            checkpoint_options['offline_objective_revisions'].append(transition)
        if (parent_continuation and args.objective_continuation) or offline_tweak:
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
            'native_optimizer_migration':args.native_optimizer_migration,
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
    inherited_directory = (saved_training.get('options') or {}).get('checkpoint_input_offload_directory')
    if args.checkpoint_input_offload_directory is None:
        args.checkpoint_input_offload_directory = inherited_directory if args.checkpoint_input_offload else None
    args.checkpoint_input_offload_directory = args.checkpoint_input_offload_directory or None
    hook = configure_checkpoint_input_offload(model.port, enabled=args.checkpoint_input_offload,
        directory=args.checkpoint_input_offload_directory)
    args.checkpoint_input_offload_directory = getattr(hook, 'directory', None)
    checkpoint_options['checkpoint_input_offload_directory'] = args.checkpoint_input_offload_directory
    checkpoint_options['checkpoint_input_offload'] = args.checkpoint_input_offload
    args.ffn_chunk_tokens=configure_ffn_chunk_tokens(model.port,args.ffn_chunk_tokens,saved_training)
    checkpoint_options['ffn_chunk_tokens']=args.ffn_chunk_tokens
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
        task_eval = TaskEval(task_spec, args.out, engine, prompt_rendering=task_spec.get('prompt_rendering', 'full'), stop_requested=lambda: policy.signaled)
    step, epoch, cursor, global_step = 0, 0, 0, 0
    metrics = {}
    exposure_counts = {key: 0 for key in source_row_bindings}
    if resume is not None:
        if norm_upgrade:
            from ..model.native_parameter_precision import norm_parameters
        promoted_names = tuple('backbone.'+name for name in norm_parameters(model)) if norm_upgrade else ()
        promoted_names += tuple(
            'interface.'+name for name in (token_upgrade or {}).get('promoted_parameters', {}))
        added_names = ('interface.extra_input_rows', 'interface.extra_output_rows') if token_upgrade and token_upgrade['added'] else ()
        load_options = {'native_optimizer_migration': args.native_optimizer_migration}
        if added_names:load_options['allow_added_names'] = added_names
        if promoted_names:load_options['allow_promoted_names'] = promoted_names
        optimizer.load_state_dict(resume['optimizer'], **load_options)
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
            if policy_operation:_retain_current_held_coverage(metrics,saved_training,held)
        else:
            step, epoch, cursor = int(resume['step']), int(resume['epoch']), int(resume['cursor'])
            global_step = int(resume.get('global_step', step))
            exposure_counts.update({key: int(value) for key, value in (resume.get('exposure_counts') or {}).items()
                                    if key in exposure_counts})
            metrics = dict(resume.get('metrics') or {})
            if restored_progress and (shared_resume is not None
                    or (saved_training.get('options') or {}).get('stage_id')!=args.stage_id
                    or (saved_training.get('identity') or {}).get('objective')!=objective_binding):
                # Credits survive other objectives; evaluations belong to the new weights.
                for key in ('held', 'task_eval', 'task_checkpoint_sha256', 'task_native_identity'):
                    metrics.pop(key, None)
                _retain_current_held_coverage(metrics,saved_training,held)
        restore_training_rng_state(resume['rng'], args.device)
        resume['optimizer'] = None
        # Avoid locals(): its cached frame dictionary retains every setup
        # reference even after the corresponding fast local is cleared.
        state = parent_state = None
    # Keep only the small continuation values still needed below. Decoded
    # predecessor/source maps are no longer needed after optimizer/RNG restore;
    # retaining them through gradient and capacity checks wastes gigabytes.
    prior_policy_capacity = ((resume or {}).get('options') or {}).get('policy_capacity')
    admission_record = ((resume or saved_training).get('options') or {}).get('_pending_source_admission')
    prior_foundations = set((resume or {}).get('completed_foundations') or
        (carried_progress.get(args.stage_id or 'native-graph-ce') or {}).get('completed_foundations') or [])
    saved_options = class_migration = None
    saved_training = resume = shared_resume = continuation_training = restored_frame = None
    source_resume = foundation_frame = old_frame = old_options = None
    old_counts = old_quotas = prior_bindings = prior_order = prior_held = None
    old_source = origins = None
    transition_check = qualify_precision_transition(model, tokenizer, roles, config,
        previous_precision=previous_precision,
        previous_attention_precision=previous_attention_precision,
        previous_norm_execution=previous_norm_execution,
        previous_shared_execution=previous_shared_execution,
        checkpoint_sha256=sha256_file_hex_cached(args.continue_from if parent_continuation else policy.slot)
            if parent_continuation or policy.slot.exists() else config['base_inventory_sha256'],
        forward_atol=args.forward_atol, gradient_atol=args.gradient_atol,
        device=args.device, backend=backend, optimizer=optimizer)
    if transition_check is not None:
        checkpoint_options['native_precision_transition']['reference_check'] = transition_check
        write_json(args.out / 'precision-transition.json', transition_check)
        if 'attention' in transition_check:
            checkpoint_options['native_attention_precision_history'][-1]['reference_check'] = transition_check['runtime_reference']
        if 'norm_execution' in transition_check:
            checkpoint_options['native_norm_execution_history'][-1]['reference_check'] = transition_check['runtime_reference']
        if 'shared_execution'in transition_check:
            checkpoint_options['native_shared_execution_history'][-1]['reference_check']=transition_check['runtime_reference']
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
        memory_check = run_prepared(model,optimizer,train,roles,config,args,backend=backend,objective=objective,
            checkpoint_sha256=sha256_file_hex_cached(args.continue_from if parent_continuation else policy.slot)
                if parent_continuation or policy.slot.exists() else config['base_inventory_sha256'])
        if not args.resident_geometry_qualification:
            raise ValueError('explicit memory qualification produced no usable physical capacity certificate')
        checkpoint_options['memory_qualification'] = {'passed':memory_check['passed'],
            'request':memory_check['request'],'parent_checkpoint_sha256':memory_check['parent_checkpoint_sha256']}
        checkpoint_options['checkpoint_input_offload'] = args.checkpoint_input_offload
        checkpoint_options['checkpoint_input_offload_directory'] = args.checkpoint_input_offload_directory
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
    foundation_priority_ids = ({entry['id'] for entry in foundation_source.entries if entry.get('communication')}
                               if args.foundation_communication_batches is not None else None)
    from .native_continuation import _retain_pending_sources
    pending_frame = _retain_pending_sources({
        'source_row_bindings': source_row_bindings, 'source_row_order': source_order,
        'exposure_counts': dict(exposure_counts), 'options': checkpoint_options},
        carried_progress.get(args.stage_id or 'native-graph-ce') or {})
    from .native_cohort_schedule import source_cohort_ids
    urgent_cohort_ids = source_cohort_ids(train) if args.urgent_cohort_fairness else None
    required_visits, pending_batches = pending_exposure_batches(train.entries, exposure_counts,
        epochs=args.epochs, batch_graphs=args.batch_graphs, batch_tokens=args.batch_tokens,
        foundation_ids=foundation_priority_ids,
        foundation_communication_batches=args.foundation_communication_batches or 1,
        minimum_visits=pending_frame.get('required_visits'), schedule_offset=global_step,
        cohort_ids=urgent_cohort_ids)
    if objective is not None and not mandatory_objective:
        from ..eval.native_memory import qualify_policy_objective
        checkpoint_options['policy_capacity']=qualify_policy_objective(model,optimizer,train,objective,args,config,
            backend=backend,checkpoint_sha256=sha256_file_hex_cached(args.continue_from if parent_continuation else policy.slot),
            pending_batches=pending_batches,prior_proof=prior_policy_capacity)
    target_rows = sum(required_visits.values())
    total_end = step + len(pending_batches)
    if step > total_end:
        raise ValueError('native checkpoint update is beyond the declared epoch exposure')
    resumed_step = step
    last_saved_step = step if policy.slot.is_file() and not parent_continuation else -1
    refresh_boundary = False
    if args.refresh_at_checkpoint and args.refresh_after_updates is None and not (args.checkpoint_every or args.checkpoint_seconds):
        raise ValueError('semantic refresh needs a declared checkpoint step/time cadence')
    schedule = stream_pending_batches(train, pending_batches, priority_ids=foundation_priority_ids,
                                      cohort_ids=urgent_cohort_ids)
    resident = None
    if args.resident_data_directory is not None:
        from .native_resident import ResidentAdoption
        from ..eval.native_memory import capacity_identity
        resident = ResidentAdoption(args.resident_data_directory, stage_id=args.stage_id,
            recipe_binding=json.loads(args.recipe_binding), train=train, held=held,
            geometry_qualification=args.resident_geometry_qualification,
            geometry_identity=capacity_identity(config,args,backend),
            withdrawal_history=checkpoint_options['withdrawal_history'])
        if resident.state:
            checkpoint_options['resident_adoption'] = {'generation': resident.generation,
                'history': resident.state['history']}
            checkpoint_options['source_predecessors'] = {
                'continued_native_graphs': resident.state['payload']['prior_train']['sha256'],
                'continued_held_native_graphs': resident.state['payload']['prior_held']['sha256']}
    # A no-update continuation still needs its own durable checkpoint, and an
    # explicitly changed optimizer algorithm must be saved even at a completed
    # exposure boundary. Neither operation earns another graph visit.
    # Validate new declarations against carried/active exact row identities BEFORE updates.
    pending_sources_changed = (pending_frame.get('pending_source_declarations') !=
        (carried_progress.get(args.stage_id or 'native-graph-ce') or {}).get('pending_source_declarations'))
    pending_parent_sha = (sha256_file_hex_cached(policy.slot) if policy.slot.is_file() else
        ((checkpoint_options.get('continuation') or {}).get('checkpoint_sha256')))
    if pending_sources_changed and pending_parent_sha and not len(pending_batches):
        admission_record = {'schema': 'natlang.native-pending-source-admission/1',
            'parent_checkpoint_sha256': pending_parent_sha, 'new_optimizer_updates': 0,
            'global_step': global_step,
            'declarations': sorted(pending_frame.get('pending_source_declarations') or {})}
    elif len(pending_batches):
        admission_record = None  # An actual subsequent update is a new continuation.
    if admission_record is not None:
        checkpoint_options['_pending_source_admission'] = admission_record
    metadata_dirty = (last_saved_step < 0 or args.native_optimizer_migration is not None
        or args.native_shared_execution is not None and args.native_shared_execution!=previous_shared_declaration
        or config.get('native_shared_execution','source-v1')!=previous_shared_execution
        or args.native_norm_execution is not None and args.native_norm_execution != previous_norm_declaration
        or config.get('native_norm_execution', 'source-v1') != previous_norm_execution
        or norm_upgrade is not None and norm_upgrade['previous_profile'] != norm_upgrade['profile']
        or pending_frame.get('pending_source_declarations') !=
        (carried_progress.get(args.stage_id or 'native-graph-ce') or {}).get('pending_source_declarations'))
    eval_at = max(1, args.eval_every)
    started = time.time()
    def foundation_exposure():
        return communication_foundation(foundation_source, exposure_counts, epochs=args.epochs,
                                        trained_bindings=source_row_bindings)
    foundation_template = foundation_exposure() if args.communication_foundation else None
    qualification_request = None
    previous_qualification_history = checkpoint_options['foundation_qualification_history']
    qualification_history = list(previous_qualification_history)
    if args.foundation_qualification_request:
        from .native_readiness import validate_foundation_qualification_request
        qualification_request = validate_foundation_qualification_request(args.foundation_qualification_request,
            config['architecture'] + ':' + config['base_inventory_sha256'], foundation_template)
        checkpoint_options['foundation_qualification_request_binding'] = qualification_request
        if qualification_request not in qualification_history:
            qualification_history.append(qualification_request)
    checkpoint_options['foundation_qualification_history'] = qualification_history
    metadata_dirty = metadata_dirty or qualification_history != previous_qualification_history
    foundation_missing = {key for key,row in (foundation_template or {}).get('rows',{}).items()
        if not row['admitted_to_training'] or row['required_visits']<=0 or row['graph_visits']<row['required_visits']}
    # An explicitly owned refresh continues an already exposed cohort through
    # its declared update boundary before checking the new weights. Exposure
    # alone never adds a qualified completed_foundations acknowledgement.
    refresh_inherited_foundation = bool((args.refresh_foundation_readiness or args.foundation_readiness_policy == 'on-shared-request') and
        foundation_template and foundation_template['complete'])
    def foundation_pending():
        return bool(foundation_template and foundation_template['graphs'] and not foundation_missing and
                    foundation_template['cohort_sha256'] not in prior_foundations and
                    not refresh_inherited_foundation and args.foundation_readiness_policy != 'on-shared-request')
    try:
        while not policy.signaled and not foundation_pending():
            if resident is not None:
                expanded = resident.poll(global_step)
                if expanded is not None:
                    schedule.close()
                    metadata_dirty = True
                    checkpoint_options['source_predecessors'] = {
                        'continued_native_graphs': source_receipts[0]['sha256'],
                        'continued_held_native_graphs': source_receipts[1]['sha256']}
                    train, held = expanded
                    train.set_ce_policy(ce_policy if objective is None else None)
                    args.held_native_graphs = held.path
                    held_row_bindings = held.bindings
                    args.native_graphs = train.path
                    args.recipe_binding = json.dumps(resident.recipe_binding)
                    proof=resident.plan['stage_inputs'][args.stage_id].get('source_classification_migration')
                    args.source_classification_migration=resident_bound(proof) if proof else None
                    checkpoint_options['source_classification_migration']=str(args.source_classification_migration) if proof else None
                    if proof and not any(item['proof_sha256']==proof['sha256'] for item in checkpoint_options['source_classification_history']):
                        checkpoint_options['source_classification_history'].append(
                            {'proof_sha256':proof['sha256'],'graph_visits_created':0})
                    source_row_bindings = train.bindings
                    source_order = train.ordered_ids()
                    source_receipts = [source_receipt(train), source_receipt(held)]
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
                    checkpoint_options['_producer_consumer_ids'] = [entry['id'] for entry in train.entries
                        if entry.get('producer_consumer', entry.get('communication', False))]
                    checkpoint_options['_external_state_consumption_ids'] = [entry['id'] for entry in train.entries
                        if entry.get('external_state_consumption')]
                    checkpoint_options['_pending_native_source_declarations'] = resident.pending_declarations
                    args.pending_native_sources = [Path(item['binding']['path']) for item in resident.pending_declarations]
                    checkpoint_options['pending_native_sources'] = [str(path) for path in args.pending_native_sources]
                    pending_frame = _retain_pending_sources({
                        'source_row_bindings': source_row_bindings, 'source_row_order': source_order,
                        'exposure_counts': dict(exposure_counts), 'options': checkpoint_options}, pending_frame)
                    urgent_cohort_ids = source_cohort_ids(train) if args.urgent_cohort_fairness else None
                    required_visits, pending_batches = pending_exposure_batches(train.entries, exposure_counts,
                        epochs=args.epochs, batch_graphs=args.batch_graphs, batch_tokens=args.batch_tokens,
                        foundation_ids=foundation_priority_ids,
                        foundation_communication_batches=args.foundation_communication_batches or 1,
                        minimum_visits=pending_frame.get('required_visits'), schedule_offset=global_step,
                        cohort_ids=urgent_cohort_ids)
                    total_end = step + len(pending_batches)
                    target_rows = sum(required_visits.values())
                    schedule = stream_pending_batches(train, pending_batches, priority_ids=foundation_priority_ids,
                                                      cohort_ids=urgent_cohort_ids)
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
            if foundation_priority_ids is not None:
                progress['foundation_schedule'] = {'communication_batches_per_priority_ordinary': args.foundation_communication_batches,
                    'fixed_foundation_graphs': sum(train.entries[index]['id'] in foundation_priority_ids for index in indices),
                    'required_graphs': len((foundation_template or {}).get('rows', {})),
                    'completed_graphs': len((foundation_template or {}).get('rows', {})) - len(foundation_missing)}
            if mandatory_objective:progress['offline_preference']=details
            progress['learning_rates']=optimizer.native_learning_rates
            progress['checkpoint_input_offload'] = {'enabled': args.checkpoint_input_offload,
                **(model.port.activation_checkpoint.snapshot() if hasattr(model.port.activation_checkpoint, 'snapshot') else {'checkpoint_calls': 0})}
            if getattr(model, 'native_sparse_updates', None):
                progress['sparse_row_updates'] = model.native_sparse_updates
            if objective is None and details:
                physical_tokens = details[0]['packed_physical_tokens']
                progress['physical_forward_tokens'] = physical_tokens
                progress['reused_prefix_tokens'] = details[0]['original_physical_tokens'] - physical_tokens
                progress['physical_forward_tokens_per_second'] = physical_tokens / max(seconds, 1e-9)
                progress['readout_forward_rows'] = details[0]['readout_forward_rows']
                progress['readout_original_rows'] = details[0]['readout_original_rows']
                progress['reused_readout_rows'] = details[0]['readout_original_rows'] - details[0]['readout_forward_rows']
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
            relative_refresh_due = (args.refresh_after_updates is not None
                                    and step - resumed_step >= args.refresh_after_updates)
            due_checkpoint = (policy.due(step, end=step == total_end)
                              or relative_refresh_due or requested_refresh.requested)
            if policy.signaled:
                break
            due_task = task_eval is not None and ((step == total_end and not policy_operation) or
                        bool(args.task_eval_every and step % args.task_eval_every == 0))
            # The independent export/async monitor owns historical quality in
            # this mode. Only an explicit rendezvous (below) or true stage end
            # runs the synchronous exact-current-weight qualification.
            if args.foundation_readiness_policy == 'on-shared-request':
                due_task = bool(task_eval is not None and step == total_end and not policy_operation)
            due_eval = step % eval_at == 0 or step == total_end or due_task
            if due_eval:
                evaluation_rng = capture_training_rng_state(args.device)
                try:
                    held_metrics, held_coverage = _evaluate_cadence(model, held, args=args,
                        coverage=metrics.get('held_coverage'), previous_result=metrics.get('held'), final=step == total_end and not policy_operation, backend=backend, objective=held_objective)
                except Exception as error:
                    # This boundary follows a completed optimizer update and exact
                    # exposure accounting. Never checkpoint a partial failed update.
                    restore_training_rng_state(evaluation_rng, args.device)
                    failure_metrics = {'train_loss': train_loss, 'step': step,
                        'epoch': epoch, 'cursor': cursor,
                        'elapsed_seconds': time.time() - started,
                        'held_evaluation_failure': {'type': type(error).__name__, 'message': str(error)},
                        'previous_evaluation_metrics': metrics}
                    try:
                        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
                            global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
                            source_hash=source_hash, source_row_bindings=source_row_bindings,
                            held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
                            rng=evaluation_rng, metrics=failure_metrics,
                            architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
                            fp8_state=backend.extra_state() if backend is not None else None, sft_progress=carried_progress)
                    except Exception as save_error:
                        error.add_note(f'Completed-update recovery checkpoint failed: {save_error!r}')
                    raise
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
                            metrics['task_eval'] = task_eval.cadence(step, final=step == total_end and not policy_operation)
                    else:
                        metrics['task_eval'] = task_eval.cadence(step, final=step == total_end and not policy_operation)
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
            if requested_refresh.requested or relative_refresh_due or (args.refresh_after_updates is None
                                        and args.refresh_at_checkpoint and due_checkpoint):
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
                coverage=metrics.get('held_coverage'), previous_result=metrics.get('held'), final=step == total_end and not policy_operation, backend=backend, objective=held_objective)
        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
              global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
              source_hash=source_hash, source_row_bindings=source_row_bindings,
              held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
              rng=capture_training_rng_state(args.device), metrics=metrics,
              architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
              fp8_state=backend.extra_state() if backend is not None else None, sft_progress=carried_progress)
        last_saved_step = step
    completed = all(exposure_counts[key] >= visits for key, visits in required_visits.items())
    if completed and not policy_operation and not policy.signaled and (metrics.get('held', {}).get('scope') != 'full'
            or metrics.get('held', {}).get('source_sha256') != held.file_sha256):
        metrics['held'], metrics['held_coverage'] = _evaluate_cadence(model, held, args=args,
            coverage=metrics.get('held_coverage'), previous_result=metrics.get('held'), final=True, backend=backend, objective=held_objective)
        metrics.pop('task_checkpoint_sha256', None)
        metrics.pop('task_native_identity', None)
        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
            global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
            source_hash=source_hash, source_row_bindings=source_row_bindings,
            held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
            rng=capture_training_rng_state(args.device), metrics=metrics,
            architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
            fp8_state=backend.extra_state() if backend is not None else None, sft_progress=carried_progress)
    completed = (completed and (policy_operation or metrics.get('held', {}).get('scope') == 'full')
                 and metrics.get('held', {}).get('source_sha256') == held.file_sha256)
    foundation = foundation_exposure() if args.communication_foundation else None
    if qualification_request:
        from .native_readiness import _bound
        _bound(qualification_request)  # Request bytes cannot drift during SGD.
    foundation_boundary = _foundation_readiness_boundary(foundation, prior_foundations,
        completed=completed, new_optimizer_updates=step-resumed_step,
        policy_operation=policy_operation, signaled=policy.signaled,
        refresh_requested=args.refresh_foundation_readiness, readiness_policy=args.foundation_readiness_policy,
        qualification_requested=qualification_request is not None)
    # A continuation may inherit a completed checkpoint made before task
    # summaries were checkpointed. At the same final weights, make one real
    # cadence call; do not reuse an unbound log row or enter final_at's retry
    # loop merely to force an eligible score.
    if not policy.signaled and task_eval is not None and ((completed and not policy_operation) or foundation_boundary) and (
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
    completed = completed and not policy.signaled
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
    deferred_qualification = (args.foundation_readiness_policy == 'on-shared-request' and not completed and not foundation_boundary)
    if policy.signaled or deferred_qualification or (not completed and not foundation_boundary and not refresh_boundary):
        # Operational stops preserve the exact resumable checkpoint. Defer a
        # full capture/splice qualification until the declared data exposure is
        # finished; do not make interruptions pay for a second model pass.
        runtime_report = {'schema': 'natlang.native-runtime-certificate/1', 'passed': False,
                          'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
                          'deferred': True, 'reason': ('operational stop; qualification deferred until resume'
                            if policy.signaled else 'shared qualification request pending' if deferred_qualification
                            else 'declared native graph exposure is incomplete')}
    else:
        try:
            if args.forward_atol is None or args.gradient_atol is None:
                raise ValueError('runtime qualification requires recipe-bound forward and gradient tolerances')
            from ..eval.native_backend import qualify_complete_runtime
            runtime_report=qualify_complete_runtime(model,tokenizer,roles,config,
                optimizer=optimizer,backend=backend,fp8_profile=training_identity['fp8'],
                checkpoint_sha256=checkpoint_sha,forward_atol=args.forward_atol,
                gradient_atol=args.gradient_atol,device=args.device,
                task_execution_receipt=(str(task_receipt_path.resolve())if task_receipt_path else None),
                backbone_optimizer_updates=step)
        except Exception as error:  # record a real failed check without discarding the trained checkpoint
            runtime_report = {'schema': 'natlang.native-runtime-certificate/1', 'passed': False,
                              'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
                              'error': f'{type(error).__name__}: {error}'}
    qualification_profile={'model_config_sha256':canonical_json_sha256_hex(config),'fp8':training_identity['fp8']}
    runtime_report['qualification_profile']=qualification_profile
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
    completed = completed and (policy_operation or mandatory_sft['complete'])
    pending_source_admission = ({**admission_record, 'checkpoint_sha256': checkpoint_sha}
        if admission_record and admission_record['global_step'] == global_step and step == resumed_step else None)
    continuation_boundary = {'kind': 'semantic-refresh', 'reached': bool(refresh_boundary and step>resumed_step),
        'new_optimizer_updates':step-resumed_step, 'checkpoint_sha256':checkpoint_sha}
    if requested_refresh.received:
        continuation_boundary['requested_signals'] = requested_refresh.received
    foundation_ready = bool(foundation_boundary and runtime_report.get('passed') is True and task_receipt_ok)
    qualification_schedule = {'policy':args.foundation_readiness_policy,
        'request':qualification_request, 'requested':qualification_request is not None,
        'candidate':bool(foundation and foundation.get('complete') is True),
        'readiness_granted':foundation_ready,
        'historical_quality':'external-monitor-required; trainer does not attest monitor health' if args.foundation_readiness_policy == 'on-shared-request'
            else 'trainer-cadence'}
    resident_receipt = ({'generation': resident.generation, 'directory': str(resident.directory.resolve()),
                         'history': resident.state['history']} if resident and resident.state else None)
    report = {'schema': 'natlang.neuralese-native-sft-report/1',
              'qualification_profile':qualification_profile,
              'resident_adoption': resident_receipt,
              'communication_exposure': communication, 'completed': completed,
              'completion_scope':'policy-operation' if policy_operation else 'supervised-exposure',
              'held_qualification':{'required_scope':'cadence' if policy_operation else 'full',
                  'observed_scope':metrics.get('held',{}).get('scope'),
                  'source_sha256':metrics.get('held',{}).get('source_sha256')},
              'foundation_ready': foundation_ready, 'communication_foundation': foundation, 'mandatory_sft': mandatory_sft,
              'foundation_qualification_schedule':qualification_schedule,
              'pending_source_admission': pending_source_admission,
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
    from .native_operation_report import compact_training_report,compact_exposure
    checkpoint_detail={'path':str(policy.slot.resolve()),'sha256':checkpoint_sha,'bytes':policy.slot.stat().st_size}
    report=compact_training_report(report,checkpoint_detail)
    mandatory_sft=report['mandatory_sft']
    write_json(args.out / 'report.json', report)
    write_json(args.out / 'native-training-receipt.json', {
        'schema': 'natlang.native-training-readiness/1',
        'resident_adoption': resident_receipt,
        'communication_exposure': communication, 'foundation_ready': foundation_ready,
        'communication_foundation': foundation, 'mandatory_sft': mandatory_sft,
        'foundation_qualification_schedule':qualification_schedule,
              'pending_source_admission': pending_source_admission,
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
        'policy_operation_completed':bool(policy_operation and completed),
        'completed': bool(not policy_operation and completed and communication['complete'] and
                          runtime_report.get('passed') is True and task_receipt_ok and
                          all(item.get('completed') is True for item in prerequisites)),
        'task_eval': task_summary,
        'task_execution_receipt': ({'path': str(task_receipt_path.resolve()),
                                     'sha256': sha256_file_hex_cached(task_receipt_path), 'passed': task_receipt_ok}
                                    if task_receipt_path else None),
        'training_receipt': {'path': str((args.out / 'report.json').resolve()),
                                                      'sha256': sha256_file_hex_cached(args.out / 'report.json')},
        'exposure': compact_exposure({'train_rows': len(train), 'held_rows': len(held),
                     'supervised_tokens': sum(entry['supervised_tokens'] for entry in train.entries),
                     'graph_visits': sum(exposure_counts.values()),
                     'supervised_token_visits': sum(exposure_counts[entry['id']] *
                         entry['supervised_tokens'] for entry in train.entries),
                     'row_update_counts_sha256': exposure_counts_sha256,
                     'by_source_group': exposure_by_source_group,
                     'consumed_rows': min(target_rows, sum(exposure_counts.values())),
                     'consumed_updates': step, 'completed_epochs': epoch},checkpoint_detail)})
    return 0


def _graph_loss(model, example, args):
    from ..data.native_graph import graph_loss
    return graph_loss(model, example, checkpoint_layers=args.checkpoint_layers,
                      chunk_tokens=args.chunk_tokens)


if __name__ == '__main__':
    raise SystemExit(main())
