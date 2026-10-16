"""Shared-state VAE, joint write-out/consumer adaptation and distillation.

One producer graph result supplies both control supervision and its exported
state. Cross-model adaptation then uses a separate masked consumer graph.
Optimizers are deliberately deferred until the entire joint backward ends.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math

import torch
import torch.nn.functional as F

from ..data.native_graph import NativeGraphExample, parse_graph, graph_loss_from_result
from ..model.shared_state import (Invocation, ImportSpan, compile_graph, capture_block,
                                  capture_logical_span)
from ..model.state_block import NativeStateBlock
from ..model.semantic_vae import alignment_weights

from ..data.native_pair_schema import PAIR_SCHEMA, parse_pair


@dataclass
class SemanticParticipant:
    model: object
    tokenizer: object
    config: dict
    identity: dict
    optimizer: object = None
    backend: object = None
    sft_progress: dict = field(default_factory=dict)
    global_step: int = 0


def producer(participant, example, view, *, external_blocks=None, chunk_tokens=128, supervise=True):
    model = participant.model
    model.port.checkpoint_layers = True
    kwargs = ({'external_blocks': external_blocks, 'external_identity': participant.identity}
              if external_blocks else {})
    result = model(example.plan, retain_states=True, **kwargs)
    export = view['export']
    if export.get('logical', False):
        values = capture_logical_span(result, export['node'], export['start'], export['stop'])
        block = NativeStateBlock(text=export['text'], **participant.identity, **values).validate()
    else:
        block = capture_block(result, export['node'], export['start'], export['stop'],
                              text=export['text'], **participant.identity)
    ids = block.token_ids.detach().cpu().tolist()
    if participant.tokenizer.decode(ids, skip_special_tokens=False) != block.text:
        raise ValueError('semantic export text is not the actual exported graph token sequence')
    encoding = participant.tokenizer(block.text, add_special_tokens=False, return_offsets_mapping=True)
    actual_offsets = [tuple(pair) for pair in encoding['offset_mapping']]
    if list(encoding['input_ids']) != ids or actual_offsets != [tuple(pair) for pair in export['offsets']]:
        raise ValueError('semantic export token IDs/offsets differ from the exact payload tokenizer')
    # Producer controls (including the closing role token) use this result,
    # rather than a second forward with different hidden state.
    control = graph_loss_from_result(model, example, result, chunk_tokens=chunk_tokens)[0] if supervise else None
    # Real control markers give even the empty payload a causal producer
    # trajectory. They are not synthetic payload tokens or exported KV rows.
    op, close = model.marker_ids[:2]
    plan = compile_graph([Invocation('__semantic_payload', (op, *ids, close), producing=((1, 1 + len(ids)),))])
    baseline = capture_block(model(plan), '__semantic_payload', 1, 1 + len(ids),
                             text=block.text, **participant.identity)
    return block, baseline, result, control


def consumer_example(example, view, token_count):
    """Replace exactly the attested source import and retain causal ancestry."""
    export = view['export']
    nodes = {node.name: node for node in example.plan.invocations}
    target_names = set(view['consumer_nodes'])
    wire = '__semantic_wire'
    while wire in nodes or wire in example.plan.external_names:
        wire += '_'
    rewritten, replacements = {}, set()
    for name, node in nodes.items():
        imports = []
        for edge in node.imports:
            if (edge.source == export['node'] and edge.start == export['start'] and edge.stop == export['stop']
                    and edge.logical == export.get('logical', False)):
                imports.append(ImportSpan(edge.at, wire, 0, token_count,stop_gradient=edge.stop_gradient))
                replacements.add(name)
            else:
                imports.append(edge)
        rewritten[name] = Invocation(name, node.token_ids, tuple(imports), node.producing, node.logical_producing)
    included = set()
    def visit(name):
        if name in included:
            return
        included.add(name)
        for edge in rewritten[name].imports:
            if edge.source in rewritten:
                visit(edge.source)
    for name in target_names:
        visit(name)
    if not included & replacements:
        raise ValueError('semantic consumers never import the attested exported native span')
    reaches = {}
    def reaches_wire(name):
        if name not in reaches:
            reaches[name] = any(edge.source == wire or
                edge.source in rewritten and reaches_wire(edge.source) for edge in rewritten[name].imports)
        return reaches[name]
    if any(not reaches_wire(name) for name in target_names):
        raise ValueError('each declared semantic consumer must causally read the adapted native state')
    selected, weights = [], []
    for node, offset in zip(example.plan.invocations, example.plan.offsets):
        if node.name not in included:
            continue
        selected.append(rewritten[node.name])
        values = example.target_weights[offset:offset + len(node.token_ids)]
        weights.extend(values if node.name in target_names else [0.] * len(values))
    external_sizes = dict(zip(example.plan.external_names, example.plan.external_sizes))
    external_sizes[wire] = token_count
    # Retain only external observations actually needed by the selected graph.
    used = {edge.source for node in selected for edge in node.imports}
    external_sizes = {name: size for name, size in external_sizes.items() if name in used}
    plan = compile_graph(selected, external_sizes=external_sizes)
    return NativeGraphExample(example.id + ':adapted-consumers', example.split, example.source_groups,
        example.tokenizer_fingerprint, plan, tuple(weights),
        {name: value for name, value in example.external_states.items() if name in used}), wire


def _consumer_forward(participant, example, view, block, *, external_blocks=None, chunk_tokens=128):
    consumer, wire = consumer_example(example, view, len(block.token_ids))
    blocks = {**(external_blocks or {}), wire: block}
    # Unused producer observations do not enter the consumer's external layout.
    blocks = {name: blocks[name] for name in consumer.plan.external_names}
    result = participant.model(consumer.plan, external_blocks=blocks,
                               external_identity=participant.identity, retain_states=False)
    loss, details = graph_loss_from_result(participant.model, consumer, result, chunk_tokens=chunk_tokens)
    return loss, result, consumer, details


def distillation_loss(model, student, teacher, consumer, *, temperature=1., chunk_tokens=128):
    """Own-state consumer imitates detached adapted-state consumer behavior."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('distillation temperature must be finite and positive')
    own = {node.name: (node, offset) for node, offset in
           zip(student['plan'].invocations, student['plan'].offsets)}
    student_queries, teacher_queries, weights = [], [], []
    for node, offset in zip(consumer.plan.invocations, consumer.plan.offsets):
        source_node, source_offset = own[node.name]
        if source_node.token_ids != node.token_ids:
            raise ValueError('distillation student and adapted consumer tokens differ')
        for local in range(1, len(node.token_ids)):
            weight = consumer.target_weights[offset + local]
            if weight:
                student_queries.append(source_offset + local - 1)
                teacher_queries.append(offset + local - 1)
                weights.append(weight)
    if not weights:
        raise ValueError('distillation consumer has no supervised target positions')
    device = student['h_final'].device
    sq, tq = (torch.tensor(indices, device=device) for indices in (student_queries, teacher_queries))
    selected = torch.tensor(weights, dtype=torch.float32, device=device)
    total = selected.new_zeros(())
    from torch.utils.checkpoint import checkpoint
    for start in range(0, len(weights), chunk_tokens):
        stop = start + chunk_tokens
        def chunk(learned, fixed, weight):
            learned_logits = model.logits(learned).float() / temperature
            with torch.no_grad():
                fixed_probs = (model.logits(fixed).float() / temperature).softmax(-1)
            return (F.kl_div(learned_logits.log_softmax(-1), fixed_probs, reduction='none').sum(-1)
                    * weight).sum() * temperature ** 2
        total = total + checkpoint(chunk, student['h_final'][0, sq[start:stop]],
            teacher['h_final'][0, tq[start:stop]].detach(), selected[start:stop], use_reentrant=False)
    return total / selected.sum()


def joint_loss(vae, participants, row, *, phase='joint', options=None, fetch_external=None):
    """Differentiable bidirectional loss; this function never steps an optimizer."""
    options = options or {}
    for key in ('kl_weight', 'alignment_weight', 'cross_weight', 'reconstruction_weight',
                'control_weight', 'behavior_weight', 'distillation_weight'):
        if key in options and (type(options[key]) not in (int, float)
                or not math.isfinite(options[key]) or options[key] < 0):
            raise ValueError('semantic loss weights must be finite and nonnegative: ' + key)
    if phase not in {'vae', 'joint', 'distill'}:
        raise ValueError('semantic phase must be vae, joint or distill')
    parsed = parse_pair(row, participants)
    if len(parsed) < 2:
        raise ValueError('semantic training row has fewer than two currently ready participants')
    chunk_tokens = options.get('chunk_tokens', 128)
    produced, observations = {}, {}
    for name, (example, view) in parsed.items():
        participant = participants[name]
        external = {}
        for key, binding in example.external_states.items():
            if fetch_external is None:
                raise ValueError('semantic producer external state needs its verified artifact resolver')
            external[key] = fetch_external(name, binding['id'])
        observations[name] = external
        with torch.set_grad_enabled(torch.is_grad_enabled() and phase != 'vae'):
            produced[name] = producer(participant, example, view, external_blocks=external,
                chunk_tokens=chunk_tokens, supervise=phase != 'vae')
    terms, stats = [], {'pairs': [], 'phase': phase, 'models': sorted(parsed)}
    for first, second in itertools.combinations(sorted(parsed), 2):
        a, ab, ar, ac = produced[first]
        b, bb, br, bc = produced[second]
        av, bv = parsed[first][1], parsed[second][1]
        to_b, rb = alignment_weights(av['export']['offsets'], bv['export']['offsets'],
            device=a.embeddings.device, allow_nearest=options.get('allow_nearest', False))
        to_a, ra = alignment_weights(bv['export']['offsets'], av['export']['offsets'],
            device=b.embeddings.device, allow_nearest=options.get('allow_nearest', False))
        # All participants in one optimizer group must share a training device.
        # Distributed workers can supply remote captures, but cross-device
        # copies must remain explicit rather than silently detaching gradients.
        if a.embeddings.device != b.embeddings.device:
            raise ValueError('joint semantic participants require the declared common training device')
        with torch.set_grad_enabled(torch.is_grad_enabled() and phase != 'distill'):
            reconstruction, parts = vae.pair_loss(first, a, ab, second, b, bb,
                source_to_target=to_b, target_to_source=to_a,
                kl_weight=options.get('kl_weight', .001), alignment_weight=options.get('alignment_weight', .1),
                cross_weight=options.get('cross_weight', 1.), detach_targets=options.get('detach_targets', True))
        loss = reconstruction * options.get('reconstruction_weight', 1.) if phase != 'distill' else a.embeddings.new_zeros(())
        metrics = {'source': first, 'target': second, 'alignment_to_source': ra, 'alignment_to_target': rb,
                   **{key: float(parts[key].detach()) for key in
                      ('self_reconstruction', 'cross_reconstruction', 'kl', 'semantic_alignment')}}
        if phase != 'vae':
            behavior, distill = [], []
            for name, adapted, result in ((first, parts['source_block'], ar), (second, parts['target_block'], br)):
                participant, (example, view) = participants[name], parsed[name]
                with torch.set_grad_enabled(torch.is_grad_enabled() and phase == 'joint'):
                    consumer_loss, adapted_result, consumer, _ = _consumer_forward(participant,
                        example, view, adapted, external_blocks=observations[name], chunk_tokens=chunk_tokens)
                if phase == 'joint':
                    behavior.append(consumer_loss)
                else:
                    distill.append(distillation_loss(participant.model, result, adapted_result, consumer,
                        temperature=options.get('distillation_temperature', 1.), chunk_tokens=chunk_tokens))
            control = (ac + bc) * .5
            loss = loss + options.get('control_weight', 1.) * control
            metrics['control_loss'] = float(control.detach())
            if behavior:
                value = torch.stack(behavior).mean()
                loss = loss + options.get('behavior_weight', 1.) * value
                metrics['behavior_loss'] = float(value.detach())
            if distill:
                value = torch.stack(distill).mean()
                loss = loss + options.get('distillation_weight', 1.) * value
                metrics['distillation_loss'] = float(value.detach())
        terms.append(loss)
        stats['pairs'].append(metrics)
    loss = torch.stack(terms).mean()
    if loss.ndim != 0 or not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError('semantic joint objective is not finite')
    return loss, stats
