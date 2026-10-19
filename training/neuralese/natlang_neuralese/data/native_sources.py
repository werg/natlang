"""Immutable graph-source validation and indexes; independent of training and kernels."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached, sha256_file_hex
from ..common.jsonio import open_jsonl_text


def _jsonl(path):
    path = Path(path)
    with open_jsonl_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSONL') from exc


class NativeGraphSource:
    """Stream-validated graph rows with a compact in-memory identity index.

    The graph plans and token arrays are deliberately not retained. Full/PH/WT
    unions and authenticated E2E archives are large enough that keeping parsed
    plans for every row can exceed host memory by orders of magnitude.
    """
    def __init__(self, path, tokenizer_fingerprint, *, required_split, pair_model=None, role_ids=None, _parents=()):
        from .native_graph_schema import parse_graph
        from ..common import row_index
        from ..common.hashing import canonical_json_sha256_hex
        self.path = Path(path)
        if self.path.resolve() in _parents:
            raise ValueError('native graph source manifests cannot contain cycles')
        self.kind = 'semantic-pairs' if pair_model is not None else 'native-graphs'
        self.pair_model, self.role_ids = pair_model, tuple(role_ids or ())
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.required_split = required_split
        if self.path.suffix == '.json' and pair_model is None:
            self._init_manifest(_parents)
            return
        before = self.path.stat()
        self.tokenizer_fingerprint = tokenizer_fingerprint
        self.required_split = required_split
        self.file_sha256 = sha256_file_hex_cached(self.path)
        package = Path(__file__).resolve().parents[1]
        policy = canonical_json_sha256_hex({name: sha256_file_hex_cached(package / name) for name in (
            'data/native_sources.py', 'data/native_graph_schema.py', 'model/native_graph_plan.py',
            'data/native_pair_schema.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')})
        key = {'source_sha256': self.file_sha256, 'tokenizer': tokenizer_fingerprint,
               'split': required_split, 'policy_sha256': policy,
               'kind': self.kind, 'pair_model': pair_model, 'role_ids': list(self.role_ids)}
        index_path = row_index.entry_path('native-graphs', canonical_json_sha256_hex(key))
        candidates = [(index_path, key)]
        cached = None
        for candidate_path, candidate_key in candidates:
            candidate = row_index.load(candidate_path)
            if (candidate and candidate.get('key') == candidate_key
                    and candidate.get('metadata_sha256') == canonical_json_sha256_hex(candidate.get('entries'))):
                cached = candidate
                break
        if (cached and isinstance(cached.get('entries'), list)
                and cached['entries']
                and cached.get('metadata_sha256') == canonical_json_sha256_hex(cached['entries'])):
            self.entries = cached['entries']
            self.bindings = {entry['id']: entry['binding'] for entry in self.entries}
            self.groups = {group for entry in self.entries for group in entry['groups']}
            after = self.path.stat()
            after = self._checked_file_state(before, after, context='source changed while loading graph index')
            self.file_binding = (after.st_size, after.st_mtime_ns, after.st_ino)
            print(json.dumps({'kind': 'native_graph_index_cache', 'hit': True,
                              'path': str(self.path), 'rows': len(self.entries)}), flush=True)
            return
        self.entries = []
        self.bindings = {}
        self.groups = set()
        for row in _jsonl(self.path):
            example, binding = self._parse(row)
            if example.split != required_split:
                raise ValueError(f'{self.path}: expected {required_split} rows, got {example.split!r} in {example.id}')
            if example.id in self.bindings:
                raise ValueError(f'{self.path}: duplicate example id {example.id!r}')
            from .native_graph_schema import communication_classification
            visits = {'id': example.id, 'groups': example.source_groups, 'binding': binding,
                      'supervised_tokens': sum(weight > 0 for weight in example.target_weights),
                      'forward_tokens': example.plan.token_count,
                      **communication_classification(example)}
            self.entries.append(visits)
            self.bindings[example.id] = binding
            self.groups.update(example.source_groups)
            del example
        if not self.entries:
            raise ValueError(f'{self.path}: no {required_split} native graph rows')
        after = self.path.stat()
        after = self._checked_file_state(before, after, context='source changed during streaming graph validation', strict=True)
        self.file_binding = (after.st_size, after.st_mtime_ns, after.st_ino)
        row_index.save(index_path, {'key': key, 'entries': self.entries,
            'metadata_sha256': canonical_json_sha256_hex(self.entries)})

    def _checked_file_state(self, previous, state, *, context, strict=False):
        """A metadata change is safe only after stable, fresh whole-file hashing."""
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        def signature(value):
            return tuple(getattr(value, name) for name in fields)
        def binding(value):
            return (value.st_size, value.st_mtime_ns, value.st_ino)
        expected = (getattr(self, '_file_stat_signature', None) if type(previous) is tuple
                    else signature(previous))
        same = signature(state) == expected and (type(previous) is not tuple or binding(state) == previous)
        if not same:
            if strict:
                raise ValueError(f'{self.path}: ' + context)
            before = state
            actual = sha256_file_hex(self.path)
            state = self.path.stat()
            if signature(before) != signature(state) or actual != self.file_sha256:
                raise ValueError(f'{self.path}: ' + context)
        self._file_stat_signature = signature(state)
        self.file_binding = binding(state)
        return state

    def _init_manifest(self, parents):
        from .native_source_tree import initialize_manifest
        initialize_manifest(self, parents, load_child=NativeGraphSource)

    def _parse(self, row):
        from .native_graph_schema import parse_graph
        if self.kind != 'semantic-pairs':
            example = parse_graph(row, tokenizer_fingerprint=self.tokenizer_fingerprint)
            return example, _row_bindings([example])[example.id]
        from types import SimpleNamespace
        from .native_pair_schema import parse_pair
        from ..common.hashing import canonical_json_sha256_hex
        if len(self.role_ids) != 4 or len(set(self.role_ids)) != 4:
            raise ValueError('paired native SFT requires actual four model role-token IDs')
        models = row.get('models') or {}
        if not isinstance(models, dict) or self.pair_model not in models:
            raise ValueError('semantic pair lacks the selected student model view')
        participants = {name: SimpleNamespace(config={'tokenizer_fingerprint': (
            self.tokenizer_fingerprint if name == self.pair_model else view['graph']['tokenizer_fingerprint'])})
            for name, view in models.items()}
        parsed = parse_pair(row, participants, required_split=self.required_split)
        example, view = parsed[self.pair_model]
        _require_communication_graph(example, view, self.role_ids)
        provenance = view['graph'].get('source_pairing') or {}
        occurrences = provenance.get('consumed_occurrences')
        original_sha = provenance.get('consumer_original_text_sha256')
        if (not isinstance(occurrences, list) or not occurrences or not isinstance(original_sha, str)
                or len(original_sha) != 64 or any(c not in '0123456789abcdef' for c in original_sha)
                or not provenance.get('consumer_source_row_id')):
            raise ValueError('paired SFT needs source-bound consumed occurrence provenance')
        expected_imports = sorted(edge.at for node in example.plan.invocations
            if node.name in view['consumer_nodes'] for edge in node.imports)
        observed = []
        for occurrence in occurrences:
            start, stop = occurrence.get('wrapped_token_start'), occurrence.get('wrapped_token_stop')
            at = occurrence.get('consumer_import_at')
            if (type(start) is not int or type(stop) is not int or not 0 <= start <= stop
                    or type(at) is not int or at < 0
                    or occurrence.get('source_row_id') != provenance['consumer_source_row_id']
                    or occurrence.get('payload_sha256') != row['payload_sha256']):
                raise ValueError('paired SFT consumed occurrence differs from its source payload witness')
            observed.append(at)
        if sorted(observed) != expected_imports:
            raise ValueError('paired SFT consumed occurrence proof differs from actual imports')
        binding = canonical_json_sha256_hex({'graph': _row_bindings([example])[example.id],
            'paired_source': canonical_json_sha256_hex(row), 'model': self.pair_model})
        return example, binding

    def iter_indices(self, indices):
        """Yield exactly the requested order, streaming monotone runs in bounded memory."""
        indices = list(indices)
        if len(set(indices)) != len(indices) or any(type(i) is not int or not 0 <= i < len(self) for i in indices):
            raise ValueError('native requested row indices must be distinct and in range')
        start = 0
        for stop in range(1, len(indices)+1):
            if stop == len(indices) or indices[stop] < indices[stop-1]:
                run = indices[start:stop]
                if run:
                    yield from self.iter_from(run[0], selected=set(run))
                start = stop

    def __len__(self):
        return len(self.entries)

    def __iter__(self):
        return self.iter_from(0)

    def iter_from(self, start=0, *, selected=None):
        from .native_graph_schema import parse_graph
        if not 0 <= start <= len(self.entries):
            raise ValueError(f'{self.path}: row cursor outside indexed source')
        state = self.path.stat()
        state = self._checked_file_state(self.file_binding, state, context='source changed after index validation')
        if self.kind == 'manifest':
            offset = 0
            for child in self.children:
                stop = offset + len(child)
                selected_child = None if selected is None else {i-offset for i in selected if offset <= i < stop}
                if stop > start and (selected_child is None or selected_child):
                    yield from child.iter_from(max(0, start-offset), selected=selected_child)
                offset = stop
            self._checked_file_state(self.file_binding, self.path.stat(), context='source changed while streaming examples')
            return
        with open_jsonl_text(self.path) as handle:
            index = 0
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                if index >= start and (selected is None or index in selected):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f'{self.path}:{line_number}: invalid JSONL') from exc
                    example, binding = self._parse(row)
                    entry = self.entries[index]
                    if (example.id != entry['id'] or binding != entry['binding']):
                        raise ValueError(f'{self.path}:{line_number}: graph source changed after index validation')
                    yield example
                index += 1
            if index != len(self.entries):
                raise ValueError(f'{self.path}: row count changed after index validation')
        after = self.path.stat()
        after = self._checked_file_state(self.file_binding, after, context='source changed while streaming examples')

    def receipt(self):
        counts = {}
        for entry in self.entries:
            for group in entry['groups']:
                value = counts.setdefault(group, {'examples': 0, 'supervised_tokens': 0})
                value['examples'] += 1
                value['supervised_tokens'] += entry['supervised_tokens']
        return {'path': str(self.path.resolve()), 'sha256': self.file_sha256,
                'rows': len(self.entries), 'by_source_group': counts,
                'communication_graphs': sum(bool(entry.get('communication')) for entry in self.entries),
                **({'sources': [child.receipt() for child in self.children]} if self.kind == 'manifest' else {})}

    def ordered_ids(self):
        return [entry['id'] for entry in self.entries]


def _require_communication_graph(example, view, role_ids):
    """Reject marker-only and loss-free pairs: actual exported state must be read."""
    produce_open, produce_close, consume_open, consume_close = role_ids
    plan = example.plan
    nodes = {node.name: node for node in plan.invocations}
    exported = view['export']
    producer = nodes[exported['node']]
    declared = producer.logical_producing if exported.get('logical', False) else producer.producing
    if not any(start <= exported['start'] <= exported['stop'] <= stop for start, stop in declared):
        raise ValueError('paired SFT export must address an actually declared producing span')
    producing = False
    consumed = set()
    for node, offset in zip(plan.invocations, plan.offsets):
        for start, stop in node.producing:
            if not (0 < start <= stop < len(node.token_ids)
                    and node.token_ids[start-1] == produce_open and node.token_ids[stop] == produce_close):
                raise ValueError('paired SFT producer span needs actual producing boundary tokens')
            producing = True
        for edge in node.imports:
            if edge.source not in nodes:
                raise ValueError('paired SFT needs its actual in-graph producer state, not external placeholders')
            if not (0 < edge.at < len(node.token_ids) and node.token_ids[edge.at-1] == consume_open
                    and node.token_ids[edge.at] == consume_close):
                raise ValueError('paired SFT import needs actual consuming boundary tokens')
            if not any(weight > 0 for weight in example.target_weights[offset+edge.at+1:offset+len(node.token_ids)]):
                raise ValueError('paired SFT needs supervised consumer tokens after its imported state')
            consumed.add(node.name)
    source_index = next(i for i, node in enumerate(plan.invocations) if node.name == producer.name)
    bounds = plan.boundaries[source_index]
    export_bounds = ((exported['start'], exported['stop']) if exported.get('logical', False)
                     else (bounds[exported['start']], bounds[exported['stop']]))
    for name in view['consumer_nodes']:
        matching = [edge for edge in nodes[name].imports if edge.source == producer.name and
            ((edge.start, edge.stop) if edge.logical else (bounds[edge.start], bounds[edge.stop])) == export_bounds]
        if not matching:
            raise ValueError('paired SFT consumer does not import the authenticated exported span')
    if not producing or not set(view['consumer_nodes']) <= consumed:
        raise ValueError('paired SFT must train genuine producing and consuming graph nodes')


def _row_bindings(examples):
    """Exact semantic identities needed to prove an append-only data continuation."""
    from ..model.native_graph_plan import invocation_dict
    result = {}
    for item in examples:
        value = {'id': item.id, 'groups': item.source_groups, 'split': item.split,
                 'tokenizer': item.tokenizer_fingerprint, 'targets': item.target_weights,
                 'external_names': item.plan.external_names,
                 'external_sizes': item.plan.external_sizes,
                 'external_states': getattr(item, 'external_states', {}),
                 'nodes': [{'name': node.name, 'token_ids': node.token_ids,
                            'imports': list(invocation_dict(node)['imports']),
                            'producing': node.producing, 'logical_producing': node.logical_producing}
                           for node in item.plan.invocations]}
        result[item.id] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


