"""Durable Neuralese RL block graphs using the existing wire format and replay.

The bundle contains exact content-addressed block bytes, including producer
metadata, not text approximations. Consumer scores keep recorded values and
backpropagate through the existing producer rewrite. Deterministic payloads
have no invented sampling density. This path is a pathwise surrogate on value
communication, with policy gradients on sampled text, not a claim that discrete
writer choices have acquired a new stochastic policy.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from ..common.jsonio import write_bytes_durable, write_json_durable


def block_ids(value):
    if isinstance(value, str) and re.fullmatch(r'nz1_[a-z2-7]{52}', value):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from block_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from block_ids(item)


def save_bundle(rollouts, out, *, policy_id, fetch):
    """Fetch recursively referenced blocks with Server.get_bytes or a local store."""
    from ..serve.store import decode_block
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pending = set(block_ids(rollouts))
    entries = {}
    while pending:
        block_id = pending.pop()
        if block_id in entries:
            continue
        raw = fetch(block_id)
        block = decode_block(raw)  # verifies content ID and payload dimensions
        if block.id != block_id:
            raise ValueError('RL block fetch returned a different content ID')
        destination = out / (block_id + '.nzb')
        write_bytes_durable(destination, raw)
        entries[block_id] = {'path': destination.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                             'bytes': len(raw), 'length': block.length, 'width': block.width}
        pending.update(set(block_ids(block.producer)) - entries.keys())
    manifest = {'schema': 'natlang.rl-block-bundle/1', 'policy_id': policy_id, 'blocks': entries}
    destination = out / 'manifest.json'
    write_json_durable(destination, manifest)
    return manifest


def restore_bundle(engine, manifest_path, *, policy_id):
    from ..serve.store import decode_block
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'natlang.rl-block-bundle/1' or manifest.get('policy_id') != policy_id:
        raise ValueError('RL block bundle belongs to a different policy')
    for block_id, entry in manifest['blocks'].items():
        path = manifest_path.parent / entry['path']
        if path.resolve().parent != manifest_path.parent.resolve():
            raise ValueError('RL block bundle escapes its directory')
        raw = path.read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('RL block bundle bytes changed')
        block = decode_block(raw)
        if block.id != block_id or block.width != engine.width:
            raise ValueError('RL block bundle identity/width differs from learner')
        engine.store.put(block)
    return manifest


def producer_turns(rollout):
    """All recorded policy turns can produce values read by later turns."""
    return [{'messages': t['messages'], 'tools': t.get('tools'), 'reply': t['target'],
             **({'sampling': t['sampling']} if t.get('sampling') else {})}
            for t in rollout.get('turns') or []]


def replay_scores(session, turn, producers=()):
    """Text policy likelihoods with recorded-value, differentiable producer replay."""
    session.model_gradients = True
    indexed = session._index_producers(list(producers))
    term = {'messages': turn['messages'], 'tools': turn.get('tools'), 'target': turn['target']}
    prompt, target = session._target_items(term['messages'], term['tools'], term['target'])
    leaves, memo = {}, {}
    for block_id in session._term_blocks(term):
        if block_id in indexed:
            value = session._rewritten(block_id, {}, indexed, memo)
            if value is not None:
                leaves[block_id] = value
    return session._score(prompt, target, leaves, write_terms=False)['token_logp']


def captured_producer_leaves(session, capture, producers):
    """Bind writer contexts and read edges to their exact captured coordinates."""
    from .sampling import capture_items
    session.model_gradients = True
    indexed = {}
    for producer in producers:
        if not producer.get('sampling'):
            raise ValueError('value-channel RL needs exact producer token captures')
        prompt, reply, _ = capture_items(producer['sampling'], require_samples=False)
        for at, (kind, value) in enumerate(reply):
            if kind == 'block':
                indexed.setdefault(value, (producer, prompt, reply, at))
    prompt, target, _ = capture_items(capture)
    memo = {}
    leaves = {}
    for kind, block_id in prompt + target:
        if kind == 'block' and block_id in indexed:
            value = session._rewritten(block_id, {}, indexed, memo)
            if value is not None:
                leaves[block_id] = value
    return leaves
