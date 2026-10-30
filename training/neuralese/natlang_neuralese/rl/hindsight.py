"""Neuralese hindsight hints: the writer's reconstruction objective (plans/neuralese/NEURALESE_HINDSIGHT_HINTS.md).

For one verified trajectory tau* and a group of K hints (one writer episode each, ``applications/guidance-writer``
``writeHint`` per call-tree node), one packed graph holds every member's hint producer invocations and tau*'s
consumer invocations, each importing its node's hint at the declared guidance place (``hinted_graph``). Per member i:

    NLL_i = - sum_{t in tau* actions} log pi(tau*_t | node inputs, z_i)          (teacher-forced, through the channel)
    R_i   = - NLL_i - lambda_len L_i - lambda_explore E_i - lambda_kl KL_i + eta o_i
    A_i   = R_i - mean_{j != i} R_j                                             (leave-one-out baseline)
    loss  = (1 / (K T)) sum_i [ NLL_i + lambda_kl KL_i - sg(A_i) sum_{t in a_i} log pi(a_i,t) ]

L_i: hint tokens, E_i: sampled exploration tokens, KL_i: the import channel's KL (model/import_channel.py), o_i the
self-teacher's oracle outcome with hint i (phase 2), a_i the writer episode's sampled actions (exploration and hint
tokens) under the actor's exact law. Crisp mechanism only: what the hints say is the natlang writer's policy.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

SCHEMA = 'hindsight-reconstruction/1'
DEFAULTS = {'schema': SCHEMA, 'k': 4, 'lambda_len': .05, 'lambda_explore': .002, 'lambda_kl': 1e-6, 'eta': 0.,
            'channel': {'sigma': .5, 'sigma_prior': 1., 'seed': 0}, 'trainable': 'writer',
            # lambda_kl: the channel KL sums over every imported dimension (LFM3: about 7e4 nats per hint token at
            # sigma 0.5, pilot 2026-10-30), so 1e-6 puts it near the length cost per token.
            # A child call with fewer own assistant-action tokens than this is trained by plain SFT on its own
            # trajectory (no hint): a useful hint would be about as long as its content. Chosen from the tau* length
            # distribution and the expected hint length (design doc, "Call trees").
            'child_hint_min_tokens': 256}


def hindsight_declaration(value=None):
    """Canonical declaration: defaults filled, every constant checked."""
    value = dict(value or {})
    if value.get('schema', SCHEMA) != SCHEMA:
        raise ValueError(f'hindsight declarations use schema {SCHEMA}')
    if set(value) - set(DEFAULTS):
        raise ValueError('hindsight-reconstruction/1 accepts only ' + ', '.join(sorted(DEFAULTS)))
    result = {**DEFAULTS, **value, 'channel': {**DEFAULTS['channel'], **(value.get('channel') or {})}}
    if type(result['k']) is not int or result['k'] < 2:
        raise ValueError('hindsight groups need at least two hints per trajectory (leave-one-out baseline)')
    for name in ('lambda_len', 'lambda_explore', 'lambda_kl', 'eta'):
        if type(result[name]) not in (int, float) or not math.isfinite(result[name]) or result[name] < 0:
            raise ValueError(f'hindsight {name} must be a finite nonnegative number')
        result[name] = float(result[name])
    if type(result['child_hint_min_tokens']) is not int or result['child_hint_min_tokens'] < 1:
        raise ValueError('hindsight child_hint_min_tokens is a positive integer')
    if result['trainable'] not in ('writer', 'all'):
        raise ValueError("hindsight trainable is 'writer' (phase 1) or 'all'")
    return result


def leave_one_out(rewards):
    """A_i = R_i - mean_{j != i} R_j."""
    count = len(rewards)
    if count < 2:
        raise ValueError('a leave-one-out baseline needs at least two hints')
    total = sum(rewards)
    return [reward - (total - reward) / (count - 1) for reward in rewards]


def hindsight_terms(nll, kl, writer_logp, *, hint_tokens, explore_tokens, outcomes, declaration, target_tokens):
    """Group loss from per-member tensors ``nll`` [K], ``kl`` [K] and ``writer_logp`` [K] (sum of log pi over a_i).

    Returns (loss, details). ``target_tokens`` is T, the count of tau* action tokens (the same for every member).
    """
    count = len(nll)
    if not (len(kl) == len(writer_logp) == len(hint_tokens) == len(explore_tokens) == len(outcomes) == count):
        raise ValueError('hindsight members disagree in count')
    if target_tokens < 1:
        raise ValueError('hindsight reconstruction needs at least one tau* action')
    d = declaration
    rewards = [-float(nll[i]) - d['lambda_len'] * hint_tokens[i] - d['lambda_explore'] * explore_tokens[i]
               - d['lambda_kl'] * float(kl[i]) + d['eta'] * outcomes[i] for i in range(count)]
    advantages = leave_one_out(rewards)
    advantage = torch.tensor(advantages, dtype=torch.float32, device=nll.device)
    pathwise = nll.float() + d['lambda_kl'] * kl.float()
    loss = (pathwise - advantage * writer_logp.float()).sum() / (count * target_tokens)
    return loss, {'nll_per_token': [float(value) / target_tokens for value in nll], 'kl': [float(value) for value in kl],
                  'rewards': rewards, 'advantages': advantages, 'hint_tokens': list(hint_tokens),
                  'explore_tokens': list(explore_tokens), 'target_tokens': target_tokens}


def node_classes(nodes, declaration):
    """Which tau* calls get a writer pass: ``{call: 'hint' | 'sft'}``.

    ``nodes``: ``{call: {'role': 'root' | 'child', 'action_tokens': own assistant-action tokens}}``. The root always gets
    a writer pass; a child gets its own independent pass over its subtrajectory when its own action tokens reach
    ``child_hint_min_tokens``, else plain SFT on its own trajectory with its actual inputs.
    """
    threshold = declaration['child_hint_min_tokens']
    result = {}
    for call, node in nodes.items():
        if node['role'] not in ('root', 'child') or type(node['action_tokens']) is not int:
            raise ValueError('tau* nodes declare role root|child and their own action token count')
        result[call] = 'hint' if node['role'] == 'root' or node['action_tokens'] >= threshold else 'sft'
    if sum(node['role'] == 'root' for node in nodes.values()) != 1:
        raise ValueError('a tau* call tree has exactly one root')
    return result


# ---------------------------------------------------------------- graph

@dataclass(frozen=True)
class Hint:
    """One node's hint: the re-encoded producer span of the writer's returned text (``node`` is a tau* invocation)."""
    node: str
    token_ids: tuple


def hint_invocation(name, hint, roles):
    """The configured writer's re-encoding of a hint (``Engine.native_write_text``): open, payload (producing), close."""
    from ..model.native_graph_plan import Invocation
    if not hint.token_ids:
        raise ValueError('a hint has at least one token')
    tokens = (roles.produce_open, *hint.token_ids, roles.produce_close)
    return Invocation(name, tokens, producing=((1, 1 + len(hint.token_ids)),))


def hinted_graph(invocations, actions, members, rendering, roles, *, prefix=()):
    """One packed graph: every member's hint producers, then tau*'s nodes once per member, each node importing its hint.

    ``invocations``: tau*'s compiled ``Invocation``s (one replay plan, in dependency order); ``actions``: its actions
    ``{turn, local_query, token_id, ...}`` (``local_query`` own-token index of the predicting position);
    ``members``: per hint group member, ``{node: Hint}`` (a node without a hint gets none). Imports between tau*'s own
    nodes are renamed to the member's copy. ``prefix``: invocations placed first unchanged (the writer episodes' turns,
    whose sampled actions carry the score-function term). Returns ``(plan, member_actions, hint_names)``: per member the
    reconstruction actions with packed ``query`` and per member its hint invocation names (channel sources).
    """
    from dataclasses import replace
    from ..model.native_graph_plan import ImportSpan, compile_graph
    from .privileged_scoring import _Edit, _rebase_plan
    names = {node.name for node in invocations}
    nodes, edits, hint_names = list(prefix), {}, []
    for number, hints in enumerate(members):
        unknown = set(hints) - names
        if unknown:
            raise ValueError('hint names no tau* invocation: ' + ', '.join(sorted(unknown)))
        suffix = f'#m{number}'
        own_hints = []
        for node_name, hint in sorted(hints.items()):
            producer = hint_invocation(node_name + suffix + '#hint', hint, roles)
            nodes.append(producer)
            own_hints.append(producer.name)
        hint_names.append(own_hints)
        for node in invocations:
            copy = replace(node, name=node.name + suffix, imports=tuple(
                replace(edge, source=edge.source + suffix) if edge.source in names else edge for edge in node.imports))
            nodes.append(copy)
            if node.name in hints:
                at, _ = rendering.insertion_point(node.token_ids)
                tokens, offset = rendering.state_turn()
                count = len(hints[node.name].token_ids)
                edits[copy.name] = _Edit(at, 0, tuple(tokens),
                                         (ImportSpan(offset, node.name + suffix + '#hint', 1, 1 + count),))
    plan = compile_graph(nodes)
    offsets = {node.name: offset for node, offset in zip(plan.invocations, plan.offsets)}
    rebased, requery = _rebase_plan(plan, edits)
    member_actions = []
    for number in range(len(members)):
        suffix = f'#m{number}'
        member_actions.append([requery({**action, 'turn': action['turn'] + suffix,
                                        'query': offsets[action['turn'] + suffix] + action['local_query']})
                               for action in actions])
    return rebased, member_actions, hint_names


def reconstruction_nll(model, hidden, actions, *, chunk_tokens=128):
    """-sum log pi(tau*_t) at temperature 1 without masks (the model's own distribution), with gradients."""
    from .native_replay import token_action_logps
    plain = [{**action, 'temperature': 1., 'allowed_token_ids': None, 'banned_token_ids': None} for action in actions]
    return -token_action_logps(model, hidden, plain, chunk_tokens=chunk_tokens).sum()


def group_loss(model, result, member_actions, hint_names, writer_logp, *, hint_tokens, explore_tokens, outcomes,
               declaration):
    """``hindsight_terms`` of one packed hinted graph's forward ``result`` (import channel active over ``hint_names``)."""
    hidden = result['h_final'][0]
    nll = torch.stack([reconstruction_nll(model, hidden, actions) for actions in member_actions])
    channel = result.get('import_channel')
    if channel is not None:
        index = {name: number for number, name in enumerate(channel['sources'])}
        kl = torch.stack([sum((channel['kl'][index[name]] for name in names), nll.new_zeros(()))
                          for names in hint_names])
    else:
        kl = torch.zeros_like(nll)
    return hindsight_terms(nll, kl, writer_logp, hint_tokens=hint_tokens, explore_tokens=explore_tokens,
                           outcomes=outcomes, declaration=declaration, target_tokens=len(member_actions[0]))
