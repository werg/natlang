"""Typed, token-aligned native graph inputs shared by all backbone trainers."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import re

import torch
import torch.nn.functional as F

from ..model.shared_state import Invocation, ImportSpan, compile_graph

SCHEMA = 'natlang.neuralese-native-graph/1'


@dataclass
class NativeGraphExample:
    id: str
    split: str
    source_groups: tuple[str, ...]
    tokenizer_fingerprint: str
    plan: object
    target_weights: tuple[float, ...]
    external_states: dict = field(default_factory=dict)


def parse_graph(row, *, tokenizer_fingerprint):
    """Validate actual token supervision and graph references before any update."""
    if row.get('schema') != SCHEMA:
        raise ValueError('native graph schema is required; legacy projected records need explicit conversion')
    if row.get('tokenizer_fingerprint') != tokenizer_fingerprint:
        raise ValueError('native graph tokenizer differs from the expanded model tokenizer')
    if not isinstance(row.get('id'), str) or not row['id']:
        raise ValueError('native graph needs a source identity')
    if row.get('split') not in {'train', 'test'}:
        raise ValueError('native graph split must be explicit train or test')
    groups = row.get('source_groups')
    if not isinstance(groups, list) or not groups or any(not isinstance(group, str) or not group for group in groups):
        raise ValueError('native graph needs explicit source groups')
    nodes, weights = [], []
    for node in row.get('nodes') or []:
        ids, supervision = node.get('token_ids'), node.get('target_weights')
        if (not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids)
                or not isinstance(supervision, list) or len(supervision) != len(ids)
                or any(type(weight) not in (float, int) or not math.isfinite(weight) or weight < 0
                       for weight in supervision)):
            raise ValueError('native node needs nonnegative token IDs and matching finite target weights')
        if ids and supervision[0] != 0:
            raise ValueError('the first own token has no own predecessor query and cannot be supervised')
        imports = tuple(ImportSpan(**edge) for edge in node.get('imports', []))
        for edge in imports:
            if edge.at < len(ids) and supervision[edge.at] != 0:
                raise ValueError('a forced token after an imported state has no consumer predecessor query')
        nodes.append(Invocation(name=node['name'], token_ids=tuple(ids), imports=imports,
            producing=tuple(tuple(span) for span in node.get('producing', [])),
            logical_producing=tuple(tuple(span) for span in node.get('logical_producing', []))))
        weights.extend(float(weight) for weight in supervision)
    external = row.get('external_states') or {}
    if (not isinstance(external, dict) or any(not isinstance(name, str) or not name
            or not isinstance(binding, dict)
            or not isinstance(binding.get('id'), str) or not re.fullmatch(r'nzs1_[a-z2-7]{52}', binding['id'])
            or type(binding.get('token_count')) is not int or binding['token_count'] < 0
            for name, binding in external.items())):
        raise ValueError('native graph external states need exact content IDs and nonnegative token counts')
    plan = compile_graph(nodes, external_sizes={name: binding['token_count'] for name, binding in external.items()})
    return NativeGraphExample(row['id'], row['split'], tuple(groups), tokenizer_fingerprint,
                              plan, tuple(weights), external)


def text_graph(row, tokenizer, *, tokenizer_fingerprint, source_id, context_weight=0.05):
    """Convert an admitted text row into the same one-node graph objective.

    Existing pretokenized IDs must not be reused with the expanded tokenizer.
    The old boundary is recovered through its own token IDs only when the
    caller supplies the exact source tokenizer; otherwise use native rendering.
    """
    text = row['text']
    source_tokenizer = row.get('_source_tokenizer')
    old_ids = row.get('token_ids')
    boundary = row.get('supervised_suffix_start')
    if source_tokenizer is None or not isinstance(old_ids, list) or type(boundary) is not int:
        raise ValueError('text conversion requires its exact source tokenizer and supervised boundary')
    if not 0 <= boundary <= len(old_ids):
        raise ValueError('source supervised boundary is outside its token sequence')
    prompt = source_tokenizer.decode(old_ids[:boundary], skip_special_tokens=False)
    target = source_tokenizer.decode(old_ids[boundary:], skip_special_tokens=False)
    if prompt + target != text:
        raise ValueError('source text does not equal its exact token-boundary reconstruction')
    # Explicit role replacement must happen before this stage if old structural
    # Neuralese markers exist. Their role cannot be guessed from a flat string.
    if '<|neuralese|>' in text or '<|/neuralese|>' in text:
        raise ValueError('legacy Neuralese markers require role-aware native trajectory conversion')
    prompt_ids = tokenizer(prompt, add_special_tokens=False)['input_ids']
    target_ids = tokenizer(target, add_special_tokens=False)['input_ids']
    ids = prompt_ids + target_ids
    if not math.isfinite(context_weight) or context_weight < 0:
        raise ValueError('native context weight must be finite and nonnegative')
    weights = [context_weight] * len(prompt_ids) + [1.] * len(target_ids)
    if weights:
        weights[0] = 0.
    return {'schema': SCHEMA, 'id': source_id, 'split': row['split'],
            'source_groups': row['source_groups'], 'tokenizer_fingerprint': tokenizer_fingerprint,
            'nodes': [{'name': 'text', 'token_ids': ids, 'target_weights': weights}]}


def graph_loss(model, example, *, checkpoint_layers=True, chunk_tokens=128,
               external_blocks=None, external_identity=None):
    """One graph forward and token-weighted CE; never recursive producer replay."""
    if chunk_tokens < 1:
        raise ValueError('loss chunk size must be positive')
    model.port.checkpoint_layers = checkpoint_layers
    result = model(example.plan, external_blocks=external_blocks, external_identity=external_identity,
                   retain_states=False)
    return graph_loss_from_result(model, example, result, chunk_tokens=chunk_tokens)


def graph_loss_from_result(model, example, result, *, chunk_tokens=128):
    """Reuse a producer graph forward for supervision and state adaptation."""
    if chunk_tokens < 1:
        raise ValueError('loss chunk size must be positive')
    if result['plan'] is not example.plan:
        raise ValueError('native supervision must use its own graph result')
    hidden = result['h_final'][0]
    queries, targets, selected_weights = [], [], []
    for node, offset in zip(example.plan.invocations, example.plan.offsets):
        for local in range(1, len(node.token_ids)):
            target = offset + local
            if example.target_weights[target] > 0:
                queries.append(target - 1)
                targets.append(int(node.token_ids[local]))
                selected_weights.append(example.target_weights[target])
    if not queries:
        raise ValueError('native graph contains no supervised own-token targets')
    query_ids = torch.tensor(queries, device=hidden.device)
    target_ids = torch.tensor(targets, device=hidden.device)
    selected = torch.tensor(selected_weights, device=hidden.device, dtype=torch.float32)
    total = hidden.new_zeros((), dtype=torch.float32)
    correct = hidden.new_zeros((), dtype=torch.float32)
    def loss_chunk(states, labels, token_weights):
        logits = model.logits(states)
        ce = F.cross_entropy(logits.float(), labels, reduction='none')
        return ((ce * token_weights).sum(),
                ((logits.argmax(-1) == labels).float() * token_weights).sum())
    from torch.utils.checkpoint import checkpoint
    for start in range(0, len(queries), chunk_tokens):
        stop = start + chunk_tokens
        # Save hidden rows, not a vocabulary-sized softmax per graph token.
        # The readout/role parameters remain unchanged until this backward ends.
        chunk_loss, chunk_correct = checkpoint(loss_chunk, hidden[query_ids[start:stop]],
            target_ids[start:stop], selected[start:stop], use_reentrant=False)
        total = total + chunk_loss
        correct = correct + chunk_correct
    denominator = selected.sum()
    return total / denominator, {'supervised_tokens': len(queries), 'weight': float(denominator.detach()),
                                 'accuracy': float((correct / denominator).detach()),
                                 'physical_tokens': example.plan.token_count,
                                 'logical_slots': sum(len(sequence) for sequence in example.plan.sequences),
                                 'allocated_kv_slots': len(example.plan.slot_sources),
                                 'kv_padding_slots': sum(source < 0 for source in example.plan.slot_sources)}
