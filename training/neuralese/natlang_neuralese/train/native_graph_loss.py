"""Packed native graph loss: the shared forward/CE of whole native graphs (positional KV, hybrid state).

The native server's gradient endpoints (serve/native_grad.py) and the program-tuning trainer use it; it was the
loss core of the retired BF16 native SFT trainer.
"""
from __future__ import annotations

import contextlib

import torch


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
    from ..common import stable_softmax
    from torch.utils.checkpoint import checkpoint
    state_chunks = _loss_query_state_chunks(hidden, query_ids, chunk_tokens,
                                            unique_queries=unique_queries)
    target_chunks = target_ids.split(chunk_tokens)
    weight_chunks = weights.split(chunk_tokens)
    total = hidden.new_zeros((), dtype=torch.float32)
    correct = hidden.new_zeros((), dtype=torch.float32) if collect_accuracy else None
    numerics = stable_softmax.active_policy()  # bound now: checkpoint recomputation runs in backward
    for states, labels, selected in zip(state_chunks, target_chunks, weight_chunks):
        def loss_chunk(states, labels, selected, *, measure_accuracy):
            logits = model.logits(states)
            ce = stable_softmax.cross_entropy(logits, labels, policy=numerics)
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
                      share_causal_prefixes=False, collect_accuracy=True, allow_communication_prefixes=True):
    """One backbone graph forward for a minibatch; mean the original per-graph losses."""
    # A graph weighted by the declared own-output objective may have no own
    # outputs (e.g. only failed actions); it contributes nothing, not an error.
    # Decision readouts (data.decision_graph) are own outputs supervised over their option set.
    targeted = [any(weight>0 for weight in example.target_weights) or bool(example.decisions)
                for example in examples]
    if objective is None and any(not has and getattr(example, 'objective_categories', None) is None
                                 for has, example in zip(targeted, examples)):
        raise ValueError('native graph has no ordinary CE targets; its declared offline/policy objective is required')
    targeted_graphs = sum(targeted)
    from .native_prefix_packing import pack_causal_prefixes, ordinary_sft_physical_sharing_enabled
    # Exact raw prefixes alias physically in every dtype (prefix arithmetic v4):
    # shared rows run once and receive the cotangent fan-in of all consumers.
    # Graphs without an applicable alias keep the fixed-linear primal reuse
    # below. Capacity measures the same packed physical graph.
    if objective is None:
        share = ordinary_sft_physical_sharing_enabled(model, share_causal_prefixes)
        allow_communication = share and allow_communication_prefixes
    else:
        supports = getattr(objective, 'shares_causal_prefixes', None)
        share = bool(share_causal_prefixes and supports is not None and supports(examples))
        allow_communication = share and allow_communication_prefixes
    operator_rows = getattr(model, 'operator_rows', None)
    with_operators = operator_rows is not None and any(getattr(example, 'operators', ()) for example in examples)
    # Operator rows substitute embeddings at declared positions; keep every original row unaliased for them.
    share = share and not with_operators
    shared = pack_causal_prefixes(examples, allow_communication=allow_communication) if share else None
    if shared is None:
        plan, target_weights, ownership, external_name_maps = pack_native_graphs(examples)
        physical_rows = tuple(tuple(range(plan.offsets[first],
            plan.offsets[last-1]+len(plan.invocations[last-1].token_ids))) for first,last in ownership)
    else:
        plan, physical_rows = shared
    # Use the actual compiled physical graph after prefix sharing. Keep complete
    # episodes; this selects recomputation, never a context/token truncation.
    direct_limit = getattr(model, 'checkpoint_direct_token_limit', 0)
    model.port.checkpoint_layers = checkpoint_layers and not (direct_limit and plan.token_count <= direct_limit)
    if objective is not None:
        # Every objective reads its original-token coordinates through physical_rows; the node ownership, row
        # offsets and external-name maps exist only for the unshared layout (shared objectives never use them).
        context = {'plan': plan, 'examples': examples, 'shared_causal_prefixes': shared is not None,
                   'physical_rows': {example.id: rows for example, rows in zip(examples, physical_rows)},
                   'external_states': {example.id: getattr(example, 'external_states', {})
                                       for example in examples}}
        if shared is None:
            context.update(target_weights=target_weights, ownership=ownership,
                row_offsets={example.id: {'node_start': first_node, 'node_stop': last_node,
                    'physical_start': plan.offsets[first_node],
                    'physical_stop': plan.offsets[last_node - 1] + len(plan.invocations[last_node - 1].token_ids),
                    'external_names': external_name_maps[index]}
                    for index, (example, (first_node, last_node)) in enumerate(zip(examples, ownership))},
                external_names={example.id: external_name_maps[index] for index, example in enumerate(examples)})
        else:
            context['external_names'] = {example.id: {} for example in examples}
        forward_inputs = objective.forward_inputs(context)
    else:
        forward_inputs = {}
    if with_operators:
        if shared is not None or 'embeddings' in forward_inputs:
            raise ValueError('learned operator rows require the unshared packed layout and their own input embeddings')
        from ..model.learned_operators import operator_embeddings, resolve_declared
        spans = tuple(span for example, (first, _) in zip(examples, ownership)
                      for span in resolve_declared(getattr(example, 'operators', ()), example.plan,
                                                   operator_rows.bindings, offset=plan.offsets[first]))
        forward_inputs = {**forward_inputs, 'embeddings': operator_embeddings(model, operator_rows, plan, spans)}
    graph_forward = getattr(objective, 'forward_graph', None) if objective is not None else None
    from .native_prefix_packing import communication_primal_rows
    from ..model.native_gradient_partitions import primal_row_reuse
    primal_rows = (communication_primal_rows(examples) if shared is None and share_causal_prefixes
        and objective is None and allow_communication_prefixes
        and model.port.embedding_weight.dtype == torch.bfloat16 else None)
    # Reuse native fixed linear primals only: all original occurrence rows,
    # masks, activation casts and custom backward reductions remain intact.
    with (primal_row_reuse(plan,primal_rows) if primal_rows is not None
          else contextlib.nullcontext()) as primal_aliases:
        result = (graph_forward(model, context, forward_inputs or {}) if graph_forward is not None else
                  model(plan, retain_states=False, **(forward_inputs or {})))
    if objective is not None:
        loss, details = objective(model, result, context)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss.detach()):
            raise FloatingPointError('native policy objective must return a finite scalar loss')
        if isinstance(details, dict):
            details = {**details, 'packed_physical_tokens': plan.token_count,
                       'original_physical_tokens': sum(example.plan.token_count for example in examples)}
        return loss, details
    hidden = result['h_final'][0]
    if not targeted_graphs:
        return hidden.sum() * 0., []
    # BF16 head adjoints round each original use before prefix fan-in. Folding
    # those uses earlier changes that declared law, despite equal scalar CE.
    # Keep its original path; the qualified FP32 case has no such cast boundary.
    reuse_readouts = shared is not None and not collect_accuracy and hidden.dtype == torch.float32
    readout_index, readout_queries, readout_targets, readout_groups, readout_weights = {}, [], [], [], []
    readout_position = 0
    losses, details, detail_weights, detail_accuracies = [], [], [], []
    category_sums, decision_sums, decision_extra = [], [], []
    boundary_ids = set((*getattr(model, 'marker_ids', ()), *getattr(model, 'token_override_ids', ())))
    for graph_index, example in enumerate(examples):
        if not targeted[graph_index]:
            continue
        categories = getattr(example, 'objective_categories', None)
        scale = float(getattr(example, 'loss_scale', 1.))
        # Own-output token mass: the summed weighted CE times the scale, never divided by the graph's own mass.
        token_mass = getattr(example, 'loss_normalization', 'graph') == 'token'
        queries, targets, weights = [], [], []
        reasoning = []
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
                    if categories is not None:
                        reasoning.append(categories[target] == 2)
                    if token in boundary_ids:
                        boundary_targets[str(token)]['count'] += 1
                        boundary_targets[str(token)]['weight'] += float(weight)
        if not queries and not example.decisions:
            raise ValueError(f'native graph {examples[graph_index].id!r} has no supervised target')
        decided = None
        if example.decisions:
            from .decision_supervision import decision_terms
            decided = decision_terms(model, hidden, example, physical_rows=physical_rows[graph_index],
                                     chunk_tokens=chunk_tokens)
            decision_sums.append(torch.stack((decided[0].detach(), decided[2])))
            # A decision is one invocation of mass 1; token CE keeps its per-invocation normalization, so the
            # graph mean is (token invocations * token mean + decision CE sum) / (token invocations + decisions).
            invocations = sum(1 for node, offset in zip(example.plan.invocations, example.plan.offsets)
                              if any(weight > 0 for weight in example.target_weights[offset:offset+len(node.token_ids)]))
            token_share = invocations / (invocations + decided[1])
            decision_scale = (scale * example.decision_tokens if token_mass else
                              scale / (invocations + decided[1]))
        if not queries:
            if reuse_readouts:
                decision_extra.append(decided[0] * decision_scale / targeted_graphs)
            else:
                losses.append(decided[0] * decision_scale)
            detail_weights.append(hidden.new_zeros((), dtype=torch.float32))
            if collect_accuracy:
                detail_accuracies.append((decided[2] / decided[1]).detach())
            details.append({'weight': None, 'boundary_targets': boundary_targets, 'supervised_tokens': 0,
                            'packed_physical_tokens': plan.token_count, 'primal_projection_rows': 0,
                            'reused_primal_projection_rows': 0,
                            'original_physical_tokens': sum(item.plan.token_count for item in examples),
                            'accuracy': None, 'decisions': None,
                            **({'loss_scale': scale} if categories is not None else {})})
            continue
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
            if token_mass:
                readout_weights.append(token_weights * scale / targeted_graphs)
                if decided is not None:
                    decision_extra.append(decided[0] * decision_scale / targeted_graphs)
            elif decided is None:
                readout_weights.append(token_weights / denominator / targeted_graphs * scale)
            else:
                readout_weights.append(token_weights / denominator * token_share / targeted_graphs * scale)
                decision_extra.append(decided[0] * decision_scale / targeted_graphs)
        elif categories is not None:
            # Own-output objective: the same weighted CE, summed per category
            # so reply/action and reasoning learning are reported separately.
            parts, correct = [], None
            for selected in (False, True):
                index = [i for i, value in enumerate(reasoning) if value == selected]
                if not index:
                    parts.append((hidden.new_zeros((), dtype=torch.float32), hidden.new_zeros((), dtype=torch.float32)))
                    continue
                subset = torch.tensor(index, device=hidden.device)
                part, part_correct = _native_ce_sum(model, hidden, query_ids[subset], target_ids[subset],
                    token_weights[subset], chunk_tokens=chunk_tokens, collect_accuracy=collect_accuracy,
                    unique_queries=len({queries[i] for i in index}) == len(index))
                parts.append((part, token_weights[subset].sum()))
                if collect_accuracy:
                    correct = part_correct if correct is None else correct + part_correct
            total = parts[0][0] + parts[1][0]
            category_sums.append(torch.stack((parts[0][1], parts[0][0].detach(), parts[1][1], parts[1][0].detach())))
            if token_mass:
                losses.append(total * scale if decided is None else total * scale + decided[0] * decision_scale)
            else:
                losses.append(total / denominator * scale if decided is None else
                              total / denominator * token_share * scale + decided[0] * decision_scale)
        else:
            total, correct = _native_ce_sum(model, hidden, query_ids, target_ids, token_weights,
                chunk_tokens=chunk_tokens, collect_accuracy=collect_accuracy,
                unique_queries=len(set(queries)) == len(queries))
            if token_mass:
                losses.append(total * scale if decided is None else total * scale + decided[0] * decision_scale)
            else:
                losses.append(total / denominator * scale if decided is None else
                              total / denominator * token_share * scale + decided[0] * decision_scale)
        detail_weights.append(denominator.detach())
        if collect_accuracy:
            detail_accuracies.append((correct / denominator).detach())
        details.append({'weight': None,
                        'boundary_targets': boundary_targets,
                        'supervised_tokens': len(queries),
                        'packed_physical_tokens': plan.token_count,
                        'primal_projection_rows': primal_aliases.forward_input_rows if primal_aliases else 0,
                        'reused_primal_projection_rows': (primal_aliases.forward_input_rows-primal_aliases.forward_unique_rows)
                            if primal_aliases else 0,
                        'original_physical_tokens': sum(item.plan.token_count for item in examples),
                        'accuracy': None, **({'decisions': None} if decided is not None else {}),
                        **({'own_output': None, 'loss_scale': scale} if categories is not None else {})})
    original_readout_rows = sum(item['supervised_tokens'] for item in details)
    if reuse_readouts and not readout_queries:
        losses = [torch.stack(decision_extra).sum()]
    elif reuse_readouts:
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
        losses = [total + torch.stack(decision_extra).sum()] if decision_extra else [total]
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
    if decision_sums:
        values = iter(torch.stack(decision_sums).detach().cpu().tolist())
        for detail, example in zip(details, (item for item, has in zip(examples, targeted) if has)):
            if 'decisions' in detail:
                loss_sum, correct = next(values)
                detail['decisions'] = {'count': len(example.decisions), 'loss_sum': loss_sum, 'correct': correct}
    if category_sums:
        values = iter(torch.stack(category_sums).detach().cpu().tolist())
        for detail in details:
            if 'own_output' in detail:
                reply_weight, reply_loss, reasoning_weight, reasoning_loss = next(values)
                detail['own_output'] = {'reply': {'weight': reply_weight, 'loss_sum': reply_loss},
                                        'reasoning': {'weight': reasoning_weight, 'loss_sum': reasoning_loss}}
    return torch.stack(losses).mean(), details
