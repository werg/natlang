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


def captured_logps(session, capture, *, leaves=None, return_predictions=False):
    prompt, target, events = capture_items(capture)
    def positions(items):
        return sum(1 if kind == 'tok' else session.engine.lookup(value).length +
                   (2 if session.heads.read_markers else 0) for kind, value in items)
    if positions(prompt) + positions(target) > session.engine.context:
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
    logps, predictions = [], []
    for event in events:
        index = event['item']
        if index in offsets:
            logps.append(scored['token_logp'][offsets[index]])
            if return_predictions:
                predictions.append(scored['token_logits'][offsets[index]].argmax())
        else:
            # A raw-port block opening was sampled before replacing it by its
            # payload. Score that discrete opener under its exact prefix.
            prefix = target[:index] + [('tok', event['token'])]
            opening = session._score(prompt, prefix, leaves or {}, write_terms=False)
            logps.append(opening['token_logp'][-1])
            if return_predictions:
                predictions.append(opening['token_logits'][-1].argmax())
    result = (torch.stack(logps), torch.tensor([e['logp'] for e in events], dtype=torch.float32,
                                          device=logps[0].device))
    return (*result, torch.stack(predictions)) if return_predictions else result
