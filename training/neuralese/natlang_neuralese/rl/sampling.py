"""Replay exact sampled token coordinates; never infer EOS from a chat template."""
from __future__ import annotations

import math

import torch


def capture_items(capture, *, require_samples=True):
    if capture.get('schema') != 'natlang.rl-sampling/1':
        raise ValueError('missing exact RL sampling capture; legacy rollouts are RFT-only')
    if capture.get('guided'):
        raise ValueError('guided rollback sampling needs mask replay; use unconstrained RL collection')
    if capture.get('temperature') != 1.0:
        raise ValueError('current exact RL replay requires temperature=1')
    prompt = []
    for kind, value in capture['prompt']:
        if kind == 'tokens':
            prompt.extend(('tok', token) for token in value)
        elif kind == 'block':
            prompt.append((kind, value))
        else:
            raise ValueError('unknown captured prompt item')
    target = [tuple(item) for item in capture['items']]
    events = capture['samples']
    previous = -1
    for event in events:
        position, token = event['item'], event['token']
        if type(position) is not int or position <= previous or position > len(target):
            raise ValueError('RL sampling positions are unordered/outside target')
        if event.get('banned') or not math.isfinite(event['logp']):
            raise ValueError('RL masked/nonfinite sampling requires explicit mask replay')
        if position == len(target):  # stop token chosen but not emitted by the server
            target.append(('tok', token))
        elif target[position][0] == 'tok' and target[position][1] != token:
            raise ValueError('RL sampled token differs from recorded target position')
        previous = position
    if not prompt or (require_samples and not events):
        raise ValueError('RL needs nonempty prompt and sampled actions')
    return prompt, target, events


def captured_logps(session, capture, *, leaves=None):
    prompt, target, events = capture_items(capture)
    if len(prompt) + len(target) > session.engine.context:
        raise ValueError('RL captured trajectory exceeds model context')
    scored = session._score(prompt, target, leaves or {}, write_terms=False)
    # _score includes a block opener only for marker ports; raw direct ports
    # read blocks without an opener in the input. Index text and block decisions
    # explicitly, including the actual terminal token from the receipt.
    offsets, cursor = {}, 0
    for index, (kind, _) in enumerate(target):
        if kind == 'tok' or session.heads.read_markers:
            offsets[index] = cursor
            cursor += 1
    logps = []
    for event in events:
        index = event['item']
        if index in offsets:
            logps.append(scored['token_logp'][offsets[index]])
        else:
            # A raw-port block opening was sampled before replacing it by its
            # payload. Score that discrete opener under its exact prefix.
            prefix = target[:index] + [('tok', event['token'])]
            logps.append(session._score(prompt, prefix, leaves or {}, write_terms=False)['token_logp'][-1])
    return torch.stack(logps), torch.tensor([e['logp'] for e in events], dtype=torch.float32,
                                          device=logps[0].device)
