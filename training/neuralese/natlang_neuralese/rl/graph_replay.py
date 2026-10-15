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
    if isinstance(value, str) and re.fullmatch(r'(?:nz1|nzs1)_[a-z2-7]{52}', value):
        yield value
    elif isinstance(value, str) and value.lstrip().startswith(('{', '[')):
        # Tool-call arguments remain JSON text in stored assistant messages.
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return
        yield from block_ids(decoded)
    elif isinstance(value, dict):
        for item in value.values():
            yield from block_ids(item)
    elif isinstance(value, list):
        for item in value:
            yield from block_ids(item)


def verified_bundle_bytes(manifest_path, block_id, *, policy_id):
    """Reuse exact durable bytes when an interrupted actor lease is restarted."""
    from ..common.hashing import sha256_file_hex_cached
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'natlang.rl-block-bundle/1' or manifest.get('policy_id') != policy_id:
        raise ValueError('RL archive belongs to a different policy')
    entry = manifest['blocks'].get(block_id)
    if entry is None:
        return None
    path = (manifest_path.parent / entry['path']).resolve(strict=True)
    if (path.parent != manifest_path.parent.resolve() or path.stat().st_size != entry['bytes']
            or sha256_file_hex_cached(path) != entry['sha256']):
        raise ValueError('RL archived block path/size/hash changed')
    return path.read_bytes()


def save_bundle(rollouts, out, *, policy_id, fetch):
    """Fetch recursively referenced blocks with Server.get_bytes or a local store."""
    from ..serve.store import decode_block, decode_native_state
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    pending = set(block_ids(rollouts))
    entries = {}
    while pending:
        block_id = pending.pop()
        if block_id in entries:
            continue
        raw = fetch(block_id)
        native = block_id.startswith('nzs1_')
        if native:
            block, actual = decode_native_state(raw)
        else:
            block = decode_block(raw)
            actual = block.id
        if actual != block_id:
            raise ValueError('RL block fetch returned a different content ID')
        destination = out / (block_id + '.nzb')
        write_bytes_durable(destination, raw)
        entries[block_id] = {'path': destination.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                             'bytes': len(raw), 'length': len(block.token_ids) if native else block.length,
                             'width': block.embeddings.shape[1] if native else block.width,
                             'format': 'native-state' if native else 'legacy-projected'}
        if native:
            entries[block_id]['native_identity'] = {name: getattr(block, name) for name in (
                'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')}
        else:
            pending.update(set(block_ids(block.producer)) - entries.keys())
    manifest = {'schema': 'natlang.rl-block-bundle/1', 'policy_id': policy_id, 'blocks': entries}
    destination = out / 'manifest.json'
    write_json_durable(destination, manifest)
    return manifest


def restore_bundle(engine, manifest_path, *, policy_id):
    from ..serve.store import decode_block, decode_native_state
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
        if block_id.startswith('nzs1_'):
            if not getattr(engine, 'native_state_enabled', False):
                raise ValueError('native RL state bundle requires a native-state learner')
            block, actual = decode_native_state(raw)
            identity = {name: getattr(block, name) for name in (
                'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')}
            if actual != block_id or entry.get('native_identity') != identity:
                raise ValueError('native RL block identity differs from its archive receipt')
            # Foreign source blocks can be present in a real multi-model graph.
            # Store exact bytes; the graph/runtime must select explicit native,
            # learned-adaptation or text-reencoding consumption, never width-cast.
            engine.store.put_native_state(block)
        else:
            block = decode_block(raw)
            if block.id != block_id or block.width != engine.width:
                raise ValueError('RL block bundle identity/width differs from learner')
            engine.store.put(block)
    return manifest


def producer_turns(rollout):
    """Bind sampled invocation turns to existing execution-graph occurrences."""
    graph = rollout.get('execution_graph') or []
    nodes = {node['node']: node for node in graph if node.get('node')}
    counters, result = {}, []
    for turn in rollout.get('turns') or []:
        call = turn.get('invocation_id')
        counters[call] = counters.get(call, 0) + 1
        turn_id = f'{call}#turn{counters[call]}'
        reads, writes = {}, {}
        for node in graph:
            if node.get('turn') != turn_id:
                continue
            if node.get('kind') == 'block_write':
                writes[node['block']] = node['node']
            elif node.get('kind') == 'block_read':
                sources = [edge['node'] for edge in node.get('inputs', []) if edge.get('block') == node['block']]
                if sources and sources[0] != 'external':
                    reads[node['block']] = sources[0]
        result.append({'messages': turn['messages'], 'tools': turn.get('tools'), 'reply': turn['target'],
            'graph_turn': turn_id, 'written_nodes': writes, 'read_bindings': reads,
            **({'sampling': turn['sampling']} if turn.get('sampling') else {})})
    return result


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


def captured_producer_leaves(session, capture, producers, *, read_bindings=None):
    """Bind writer contexts and read edges to their exact captured coordinates."""
    from .sampling import capture_items
    session.model_gradients = True
    candidates, occurrences = {}, {}
    copies = [dict(producer) for producer in producers]
    for producer in copies:
        if not producer.get('sampling'):
            raise ValueError('value-channel RL needs exact producer token captures')
        prompt, reply, _ = capture_items(producer['sampling'], require_samples=False)
        for at, (kind, value) in enumerate(reply):
            if kind == 'block':
                entry = (producer, prompt, reply, at)
                candidates.setdefault(value, []).append(entry)
                node = producer.get('written_nodes', {}).get(value)
                if node:
                    occurrences[node] = entry
    def bind(bindings):
        result = {}
        for block_id, choices in candidates.items():
            if block_id in bindings:
                node = bindings[block_id]
                if node not in occurrences:
                    raise ValueError('RL graph references an uncaptured writer occurrence: ' + node)
                entry = occurrences[node]
                if entry[2][entry[3]] != ('block', block_id):
                    raise ValueError('RL graph writer occurrence has a different block')
                result[block_id] = entry
            elif len(choices) == 1:
                result[block_id] = choices[0]
            elif len({json.dumps(choice[0]['sampling'], sort_keys=True) for choice in choices}) == 1:
                result[block_id] = choices[0]
            else:
                # No silent first-writer wins on content-addressed identical values.
                result[block_id] = None
        return result
    for producer in copies:
        producer['_producer_dependencies'] = bind(producer.get('read_bindings', {}))
    indexed = bind(read_bindings or {})
    prompt, target, _ = capture_items(capture)
    memo = {}
    leaves = {}
    for kind, block_id in prompt + target:
        if kind == 'block' and block_id in indexed:
            if indexed[block_id] is None:
                raise ValueError('ambiguous RL writer occurrence requires execution graph binding')
            value = session._rewritten(block_id, {}, indexed, memo)
            if value is not None:
                leaves[block_id] = value
    return leaves
