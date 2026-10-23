"""Durable native-state RL graphs with exact producer/import occurrence bindings."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from ..common.jsonio import write_bytes_durable, write_json_durable


def block_ids(value):
    if isinstance(value, str) and re.fullmatch(r'nz1_[a-z2-7]{52}', value):
        raise ValueError('native RL archive cannot retain a projected artifact reference')
    elif isinstance(value, str) and re.fullmatch(r'(?:nzs1|nnt1)_[a-z2-7]{52}', value):
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


def verified_bundle_path(manifest_path, block_id, *, policy_id, manifest=None):
    """Authenticate one archived member without materializing its payload."""
    from ..common.hashing import sha256_file_hex_cached
    manifest_path = Path(manifest_path)
    if not manifest_path.exists() and manifest is None:
        return None
    manifest = json.loads(manifest_path.read_text()) if manifest is None else manifest
    if manifest.get('schema') != 'natlang.rl-block-bundle/1' or manifest.get('policy_id') != policy_id:
        raise ValueError('RL archive belongs to a different policy')
    entry = manifest['blocks'].get(block_id)
    if entry is None:
        return None
    path = (manifest_path.parent / entry['path']).resolve(strict=True)
    if (path.parent != manifest_path.parent.resolve() or path.stat().st_size != entry['bytes']
            or sha256_file_hex_cached(path) != entry['sha256']):
        raise ValueError('RL archived block path/size/hash changed')
    return path


def verified_bundle_bytes(manifest_path, block_id, *, policy_id, manifest=None):
    """Reuse exact durable bytes when an interrupted actor lease is restarted."""
    path=verified_bundle_path(manifest_path,block_id,policy_id=policy_id,manifest=manifest)
    return None if path is None else path.read_bytes()


class NativeBundleStore:
    """Read-only exact artifact view for the shared native provenance verifier."""
    def __init__(self,path,*,policy_id,manifest=None):
        import copy,contextvars
        self.path=Path(path).resolve(strict=manifest is None);self.policy_id=policy_id
        self.manifest=copy.deepcopy(json.loads(self.path.read_text()) if manifest is None else manifest)
        if self.manifest.get('schema')!='natlang.rl-block-bundle/1' or self.manifest.get('policy_id')!=policy_id:
            raise ValueError('native archive belongs to a different exact actor policy')
        for ident, entry in self.manifest.get('blocks', {}).items():
            expected = 'native-state' if isinstance(ident, str) and ident.startswith('nzs1_') else (
                'native-tensor' if isinstance(ident, str) and ident.startswith('nnt1_') else None)
            if expected is None or not isinstance(entry, dict) or entry.get('format') != expected:
                raise ValueError('native RL archive contains a non-native artifact entry')
        self._records=contextvars.ContextVar('native_bundle_records',default=None)

    def _record(self,ident):
        records=self._records.get()
        if records is not None and ident in records:return records[ident]
        from types import SimpleNamespace
        raw=verified_bundle_bytes(self.path,ident,policy_id=self.policy_id,manifest=self.manifest)
        if raw is None:return None
        if ident.startswith('nzs1_'):
            from ..serve.store import decode_native_state
            block,actual=decode_native_state(raw)
            identity={key:getattr(block,key) for key in ('checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256','architecture')}
            record=SimpleNamespace(id=actual,block=block)
        elif ident.startswith('nnt1_'):
            from ..model.native_artifact import decode_native_tensor_artifact
            artifact,actual=decode_native_tensor_artifact(raw)
            identity=artifact.native_identity;record=SimpleNamespace(id=actual,artifact=artifact)
        else:raise ValueError('native archive verifier requires native state or named tensor artifacts')
        if actual!=ident or self.manifest['blocks'][ident].get('native_identity')!=identity:
            raise ValueError('native archive content identity differs from its exact receipt')
        if records is not None:records[ident]=record
        return record

    def get_native_state(self,ident):
        if not ident.startswith('nzs1_'):raise ValueError('native state archive ID has the wrong kind')
        return self._record(ident)

    def get_native_tensor_artifact(self,ident):
        if not ident.startswith('nnt1_'):raise ValueError('named native archive ID has the wrong kind')
        return self._record(ident)

    def state(self,ident):
        # Share decoded ancestors within this verification, never retain every
        # episode's potentially huge KV payload across a production epoch.
        token=self._records.set({}) if self._records.get() is None else None
        try:
            record=self.get_native_state(ident)
            if record is None:raise ValueError('native archive omitted its exact state observation')
            from ..model.native_state_derivation import verify_state_derivation
            verify_state_derivation(self,record.block)
            return record.block
        finally:
            if token is not None:self._records.reset(token)


def save_bundle(rollouts, out, *, policy_id, fetch, manifest_fields=None):
    """Fetch recursively referenced blocks with Server.get_bytes or a local store."""
    from ..serve.store import decode_native_state
    from ..model.native_artifact import decode_native_tensor_artifact, native_tensor_dependencies
    from ..model.state_block import native_state_dependencies
    if manifest_fields is not None and (not isinstance(manifest_fields, dict)
            or {'schema', 'policy_id', 'blocks'} & manifest_fields.keys()):
        raise ValueError('RL bundle metadata cannot replace its exact archive authority')
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
        tensor = block_id.startswith('nnt1_')
        if not (native or tensor):
            raise ValueError('native RL archive accepts only native state or named tensor artifacts')
        if tensor:
            block, actual = decode_native_tensor_artifact(raw)
        else:
            block, actual = decode_native_state(raw)
        if actual != block_id:
            raise ValueError('RL block fetch returned a different content ID')
        destination = out / (block_id + '.nzb')
        write_bytes_durable(destination, raw)
        entries[block_id] = {'path': destination.name, 'sha256': hashlib.sha256(raw).hexdigest(),
                             'bytes': len(raw),
                             'format': 'native-tensor' if tensor else 'native-state'}
        if tensor:
            entries[block_id]['native_identity'] = block.native_identity
            pending.update(set(native_tensor_dependencies(block)) - entries.keys())
        else:
            entries[block_id].update(length=len(block.token_ids), width=block.embeddings.shape[1])
        if native:
            entries[block_id]['native_identity'] = {name: getattr(block, name) for name in (
                'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')}
            pending.update(set(native_state_dependencies(block))-entries.keys())
    manifest = {'schema': 'natlang.rl-block-bundle/1', 'policy_id': policy_id, 'blocks': entries}
    manifest.update(manifest_fields or {})
    destination = out / 'manifest.json'
    reader=NativeBundleStore(destination,policy_id=policy_id,manifest=manifest)
    for ident,record in entries.items():
        if record['format']=='native-state':reader.state(ident)
    # A self-consistent content ID is not proof of a valid derived state. Keep
    # the previous published manifest intact until all derivations validate.
    write_json_durable(destination, manifest)
    return manifest


def restore_bundle(engine, manifest_path, *, policy_id):
    from ..serve.store import decode_native_state
    from ..model.native_artifact import decode_native_tensor_artifact, native_tensor_dependencies
    from ..model.state_block import native_state_dependencies
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'natlang.rl-block-bundle/1' or manifest.get('policy_id') != policy_id:
        raise ValueError('RL block bundle belongs to a different policy')
    native={}
    for block_id, entry in manifest['blocks'].items():
        path = manifest_path.parent / entry['path']
        if path.resolve().parent != manifest_path.parent.resolve():
            raise ValueError('RL block bundle escapes its directory')
        raw = path.read_bytes()
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('RL block bundle bytes changed')
        if block_id.startswith('nnt1_'):
            if not getattr(engine, 'native_state_enabled', False):
                raise ValueError('native RL tensor bundle requires a native-state learner')
            artifact, actual = decode_native_tensor_artifact(raw)
            if actual != block_id or entry.get('native_identity') != artifact.native_identity:
                raise ValueError('native RL tensor identity differs from its archive receipt')
            dependencies=native_tensor_dependencies(artifact)
            if any(dependency not in manifest['blocks'] for dependency in dependencies):
                raise ValueError('native RL tensor bundle is missing required adapter dependencies')
            native[block_id]=dependencies
        elif block_id.startswith('nzs1_'):
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
            dependencies=native_state_dependencies(block)
            if any(ident not in manifest['blocks'] for ident in dependencies):
                raise ValueError('native RL state bundle is missing required state/adapter derivation dependencies')
            native[block_id]=dependencies
        else:
            raise ValueError('native RL bundle contains a non-native artifact ID')
    reader=NativeBundleStore(manifest_path,policy_id=policy_id,manifest=manifest)
    while native:
        ready=[ident for ident,dependencies in native.items() if not any(key in native for key in dependencies)]
        if not ready:raise ValueError('native RL archive has cyclic artifact dependencies')
        for ident in ready:
            native.pop(ident)
            record=reader._record(ident)
            if ident.startswith('nzs1_'):engine.store.put_native_state(record.block)
            else:engine.store.put_native_tensor_artifact(record.artifact)
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
