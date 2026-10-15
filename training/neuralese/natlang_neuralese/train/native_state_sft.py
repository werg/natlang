"""Shared full-state supervised training for positional-KV/hybrid-state graphs.

The declared recipe supplies admitted, role-aware native graph JSONL. This
stage trains the exact model used by the native serving path; legacy projected
records are rejected by the shared graph parser.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import open_jsonl_text, write_json_durable as write_json
from ..model.native_model import (CHECKPOINT_SCHEMA, load_native_checkpoint, load_native_model, model_state,
                                  native_identity)
from .checkpoint_policy import CheckpointPolicy
from .loop import capture_training_rng_state, restore_training_rng_state

TRAINING_SCHEMA = 'natlang.neuralese-native-sft-training/1'


def _jsonl(path):
    path = Path(path)
    with open_jsonl_text(path) as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f'{path}:{line_number}: invalid JSONL') from exc


def _load_examples(path, tokenizer_fingerprint, *, required_split):
    from ..data.native_graph import parse_graph
    examples, ids = [], set()
    for row in _jsonl(path):
        example = parse_graph(row, tokenizer_fingerprint=tokenizer_fingerprint)
        if example.split != required_split:
            raise ValueError(f'{path}: expected {required_split} rows, got {example.split!r} in {example.id}')
        if example.id in ids:
            raise ValueError(f'{path}: duplicate example id {example.id!r}')
        ids.add(example.id)
        examples.append(example)
    if not examples:
        raise ValueError(f'{path}: no {required_split} native graph rows')
    return examples


class NativeGraphSource:
    """Stream-validated graph rows with a compact in-memory identity index.

    The graph plans and token arrays are deliberately not retained. Full/PH/WT
    unions and authenticated E2E archives are large enough that keeping parsed
    plans for every row can exceed host memory by orders of magnitude.
    """
    def __init__(self, path, tokenizer_fingerprint, *, required_split, pair_model=None, role_ids=None, _parents=()):
        from ..data.native_graph import parse_graph
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
            'data/native_graph.py', 'model/shared_state.py', 'train/native_state_sft.py',
            'train/native_semantic.py', 'common/row_index.py', 'common/jsonio.py', 'common/hashing.py')})
        key = {'source_sha256': self.file_sha256, 'tokenizer': tokenizer_fingerprint,
               'split': required_split, 'policy_sha256': policy,
               'kind': self.kind, 'pair_model': pair_model, 'role_ids': list(self.role_ids)}
        index_path = row_index.entry_path('native-graphs', canonical_json_sha256_hex(key))
        candidates = [(index_path, key)]
        if self.kind == 'native-graphs':
            # Exact previous trainer implementation; unchanged parser/compiler
            # hashes remain mandatory. Streaming still verifies each row before
            # its optimizer update. Never reuse ordinary indexes for paired data.
            legacy_policy = {name: sha256_file_hex_cached(package/name) for name in (
                'data/native_graph.py', 'model/shared_state.py', 'common/row_index.py',
                'common/jsonio.py', 'common/hashing.py')}
            legacy_policy['train/native_state_sft.py'] = 'd8f3d6413a40f26389c90b6e5c3cc50781e3fdae70942de241301fd678327c9d'
            legacy_key = {'source_sha256': self.file_sha256, 'tokenizer': tokenizer_fingerprint,
                'split': required_split, 'policy_sha256': canonical_json_sha256_hex(legacy_policy)}
            candidates.append((row_index.entry_path('native-graphs', canonical_json_sha256_hex(legacy_key)), legacy_key))
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
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise ValueError(f'{self.path}: source changed while loading graph index')
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
            visits = {'id': example.id, 'groups': example.source_groups, 'binding': binding,
                      'supervised_tokens': sum(weight > 0 for weight in example.target_weights),
                      'forward_tokens': example.plan.token_count,
                      'communication': self.kind == 'semantic-pairs' and
                          row['models'][self.pair_model]['export']['stop'] > row['models'][self.pair_model]['export']['start']}
            self.entries.append(visits)
            self.bindings[example.id] = binding
            self.groups.update(example.source_groups)
            del example
        if not self.entries:
            raise ValueError(f'{self.path}: no {required_split} native graph rows')
        after = self.path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise ValueError(f'{self.path}: source changed during streaming graph validation')
        self.file_binding = (after.st_size, after.st_mtime_ns, after.st_ino)
        row_index.save(index_path, {'key': key, 'entries': self.entries,
            'metadata_sha256': canonical_json_sha256_hex(self.entries)})

    def _init_manifest(self, parents):
        before = self.path.stat()
        manifest = json.loads(self.path.read_text())
        if manifest.get('schema') != 'natlang.native-graph-sources/1' or not manifest.get('sources'):
            raise ValueError('native graph source manifest requires its schema and bound sources')
        self.kind = 'manifest'
        self.file_sha256 = sha256_file_hex_cached(self.path)
        self.children, self.entries, self.bindings, self.groups = [], [], {}, set()
        for binding in manifest['sources']:
            priority = binding.get('training_priority', 0)
            if type(priority) is not int or priority < 0:
                raise ValueError('native source training_priority must be a nonnegative integer')
            kind = binding.get('kind')
            if kind not in {'native-graphs', 'semantic-pairs'}:
                raise ValueError('native graph manifest source kind must be explicit')
            if kind == 'semantic-pairs' and not isinstance(binding.get('model'), str):
                raise ValueError('semantic graph source must select its exact model view')
            path = Path(binding['path'])
            if not path.is_absolute():
                path = self.path.parent / path
            if sha256_file_hex_cached(path) != binding.get('sha256'):
                raise ValueError('native graph manifest source SHA-256 mismatch: ' + str(path))
            child = NativeGraphSource(path, self.tokenizer_fingerprint, required_split=self.required_split,
                pair_model=binding.get('model') if kind == 'semantic-pairs' else None,
                role_ids=self.role_ids, _parents=(*parents, self.path.resolve()))
            if child.file_sha256 != binding['sha256']:
                raise ValueError('native graph manifest dependency changed during validation')
            if self.bindings.keys() & child.bindings.keys():
                raise ValueError('native graph manifest repeats a graph identity across sources')
            self.children.append(child)
            # Scheduling policy never changes physical source order or row
            # binding. An omitted priority inherits a nested declaration.
            self.entries.extend({**entry, 'training_priority': priority if 'training_priority' in binding
                                 else entry.get('training_priority', 0)} for entry in child.entries)
            self.bindings.update(child.bindings)
            self.groups.update(child.groups)
        state = self.path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (state.st_size, state.st_mtime_ns, state.st_ino):
            raise ValueError('native graph manifest changed during validation')
        self.file_binding = (state.st_size, state.st_mtime_ns, state.st_ino)

    def _parse(self, row):
        from ..data.native_graph import parse_graph
        if self.kind != 'semantic-pairs':
            example = parse_graph(row, tokenizer_fingerprint=self.tokenizer_fingerprint)
            return example, _row_bindings([example])[example.id]
        from types import SimpleNamespace
        from .native_semantic import parse_pair
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
        from ..data.native_graph import parse_graph
        if not 0 <= start <= len(self.entries):
            raise ValueError(f'{self.path}: row cursor outside indexed source')
        state = self.path.stat()
        if (state.st_size, state.st_mtime_ns, state.st_ino) != self.file_binding:
            raise ValueError(f'{self.path}: source changed after index validation')
        if self.kind == 'manifest':
            offset = 0
            for child in self.children:
                stop = offset + len(child)
                selected_child = None if selected is None else {i-offset for i in selected if offset <= i < stop}
                if stop > start and (selected_child is None or selected_child):
                    yield from child.iter_from(max(0, start-offset), selected=selected_child)
                offset = stop
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
        if (after.st_size, after.st_mtime_ns, after.st_ino) != self.file_binding:
            raise ValueError(f'{self.path}: source changed while streaming examples')

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


def communication_exposure(entries, counts, quotas):
    selected = [entry for entry in entries if entry.get('communication')]
    return {'graphs': len(selected),
            'required_graph_visits': sum(quotas[entry['id']] for entry in selected),
            'graph_visits': sum(counts.get(entry['id'], 0) for entry in selected),
            'complete': bool(selected) and all(counts.get(entry['id'], 0) >= quotas[entry['id']] > 0
                                              for entry in selected)}


def batch_boundaries(entries, *, batch_graphs=1, batch_tokens=0):
    """Sequential whole-graph batches; a long graph remains one intact batch."""
    if batch_graphs < 1 or batch_tokens < 0:
        raise ValueError('native batching requires positive graph count and nonnegative token budget')
    boundaries, start, tokens = [], 0, 0
    for index, entry in enumerate(entries):
        size = entry['forward_tokens']
        if index > start and ((batch_tokens and tokens + size > batch_tokens)
                              or (not batch_tokens and index-start >= batch_graphs)):
            boundaries.append((start, index)); start, tokens = index, 0
        tokens += size
    if start < len(entries):
        boundaries.append((start, len(entries)))
    return boundaries


def pending_exposure_batches(entries, exposure_counts, *, epochs, batch_graphs=1, batch_tokens=0):
    """Schedule only missing row visits, including rows appended after completion.

    Each pass schedules authenticated communication deficits first, preserving
    source order within each class. Completed visits remain credits across
    changed packing and appended sources; fractional epochs cover the declared
    source prefix. The checkpoint's actual row counts determine what remains.
    """
    whole = int(epochs)
    fractional = math.ceil((epochs - whole) * len(entries))
    quotas = {entry['id']: max(int(bool(entry.get('communication'))), whole + (index < fractional))
              for index, entry in enumerate(entries)}
    batches = []
    pending = {visit: [index for index, entry in enumerate(entries)
                       if exposure_counts.get(entry['id'], 0) <= visit < quotas[entry['id']]]
               for visit in range(math.ceil(epochs))}
    def scheduling_rank(entry):
        return (not bool(entry.get('communication')), -entry.get('training_priority', 0))
    for rank in sorted({scheduling_rank(entry) for entry in entries}):
        for visit, remaining in pending.items():
            indices = [index for index in remaining if scheduling_rank(entries[index]) == rank]
            selected = [entries[index] for index in indices]
            boundaries = batch_boundaries(selected, batch_graphs=batch_graphs, batch_tokens=batch_tokens)
            for left, right in boundaries:
                last = right == len(indices) and all(scheduling_rank(entries[index]) <= rank for index in remaining)
                batches.append((visit, indices[left:right], last))
    return quotas, batches


def make_native_optimizer(model, *, lr, weight_decay=.1, latent_ns='quintic', in_backward=True):
    """Reuse the production port optimizer and its in-backward MuonSR memory path."""
    from .backbone_policy import configure_backbone_training
    from .optim import PortMuonAdamW, latent_entries, latent_options, latent_split

    backbone_named = [('backbone.' + name, parameter)
                      for name, parameter in configure_backbone_training(model.port, 'latent')]
    # The native role rows and embedding-space adapter are learned parts of the
    # native interface. The frozen vocabulary/readout and final norm follow the
    # existing shared full-layer backbone policy.
    interface_named = [('interface.' + name, parameter) for name, parameter in model.named_parameters()
                       if name.startswith(('input_rows', 'output_rows', 'adapter.')) and parameter.requires_grad]
    named = backbone_named + interface_named
    latent, ordinary = latent_split(named, 'muonsr')
    if not latent:
        raise ValueError('native SFT found no MuonSR backbone/interface matrices')
    # Register finite-gradient checks before the optimizer's post-accumulate
    # hooks. MuonSR updates a matrix as soon as its gradient is complete.
    if in_backward:
        for name, parameter in latent:
            def finite_before_muon(value, name=name):
                if value.grad is None:
                    return
                finite = torch.isfinite(value.grad).all()
                if value.grad.device.type == 'cuda' and hasattr(torch, '_assert_async'):
                    # Keep the per-matrix hook off the host synchronization path
                    # on large dense/MoE models; this precedes update kernels.
                    torch._assert_async(finite, f'nonfinite native gradient in {name}')
                elif not bool(finite):
                    _raise_nonfinite(name)
            parameter.register_post_accumulate_grad_hook(finite_before_muon)
    optimizer = PortMuonAdamW(ordinary, lr=lr, vocab_size=int(model.port.embedding_weight.shape[0]),
        embedding_ids={id(model.port.embedding_weight)},
        latent=latent_entries(latent, 'muonsr', lr), latent_optimizer='muonsr',
        latent_options=latent_options('muonsr', weight_decay=weight_decay, ns=latent_ns))
    if in_backward and optimizer.latent is not None:
        optimizer.latent.step_in_backward(gated=True)
    return optimizer, named


def _raise_nonfinite(name):
    raise FloatingPointError(f'nonfinite native gradient in {name}; update was not committed')


def _source_receipt(path, examples):
    counts = {}
    for example in examples:
        for group in example.source_groups:
            value = counts.setdefault(group, {'examples': 0, 'supervised_tokens': 0,
                                               'supervised_weight': 0.})
            value['examples'] += 1
            value['supervised_tokens'] += sum(weight > 0 for weight in example.target_weights)
            value['supervised_weight'] += sum(example.target_weights)
    return {'path': str(Path(path).resolve()), 'sha256': sha256_file_hex_cached(path),
            'rows': len(examples), 'by_source_group': counts}


def _source_hash(examples):
    digest = hashlib.sha256()
    for item in examples:
        row = {'id': item.id, 'groups': item.source_groups, 'split': item.split,
               'tokenizer': item.tokenizer_fingerprint, 'targets': item.target_weights,
               'nodes': [{'name': node.name, 'token_ids': node.token_ids,
                          'imports': [edge.__dict__ for edge in node.imports],
                          'producing': node.producing, 'logical_producing': node.logical_producing}
                         for node in item.plan.invocations]}
        value = json.dumps(row, sort_keys=True, separators=(',', ':')).encode()
        digest.update(len(value).to_bytes(8, 'big'))
        digest.update(value)
    return digest.hexdigest()


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
                                       stop=edge.stop, at=edge.at, logical=edge.logical)
                            for edge in node.imports)
            invocations.append(Invocation(name=names[node.name], token_ids=node.token_ids,
                imports=imports, producing=node.producing,
                logical_producing=node.logical_producing))
        weights.extend(example.target_weights)
        ownership.append((len(invocations) - len(example.plan.invocations), len(invocations)))
    packed = compile_graph(invocations, external_sizes=external_sizes)
    return packed, tuple(weights), tuple(ownership), tuple(external_name_maps)


def native_training_update(model, optimizer, batch, *, backend=None, checkpoint_layers=True,
                           chunk_tokens=128, objective=None):
    """One shared complete production update, also used by throughput diagnostics."""
    if backend is not None:
        backend.reset_row_counts()
    optimizer.zero_grad(set_to_none=True)
    if optimizer.latent is not None:
        context = optimizer.latent.in_backward()
    else:
        from contextlib import nullcontext
        context = nullcontext()
    try:
        with context:
            if backend is not None:
                with backend.autocast():
                    loss, details = packed_graph_loss(model, batch,
                        checkpoint_layers=checkpoint_layers, chunk_tokens=chunk_tokens,
                        objective=objective)
            else:
                loss, details = packed_graph_loss(model, batch,
                    checkpoint_layers=checkpoint_layers, chunk_tokens=chunk_tokens,
                    objective=objective)
            loss.backward()
        if optimizer.latent is not None:
            optimizer.latent._armed = None
        params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if params:
            norm = torch.nn.utils.clip_grad_norm_(params, float('inf'))
            if not torch.isfinite(norm):
                raise FloatingPointError('nonfinite native gradient norm')
        optimizer.step()
    except Exception:
        if optimizer.latent is not None:
            optimizer.latent._armed = None
        optimizer.zero_grad(set_to_none=True)
        raise
    return loss, details


def packed_graph_loss(model, examples, *, checkpoint_layers=True, chunk_tokens=128, objective=None):
    """One backbone graph forward for a minibatch; mean the original per-graph losses."""
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint
    plan, target_weights, ownership, external_name_maps = pack_native_graphs(examples)
    model.port.checkpoint_layers = checkpoint_layers
    row_offsets = {example.id: {'node_start': first_node, 'node_stop': last_node,
        'physical_start': plan.offsets[first_node],
        'physical_stop': plan.offsets[last_node - 1] + len(plan.invocations[last_node - 1].token_ids),
        'external_names': external_name_maps[index]}
        for index, (example, (first_node, last_node)) in enumerate(zip(examples, ownership))}
    context = {'plan': plan, 'examples': examples, 'target_weights': target_weights,
               'ownership': ownership, 'row_offsets': row_offsets,
               'external_names': {example.id: external_name_maps[index]
                                  for index, example in enumerate(examples)},
               'external_states': {example.id: getattr(example, 'external_states', {})
                                   for example in examples}}
    forward_inputs = objective.forward_inputs(context) if objective is not None else {}
    result = model(plan, retain_states=False, **(forward_inputs or {}))
    if objective is not None:
        loss, details = objective(model, result, context)
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0 or not torch.isfinite(loss.detach()):
            raise FloatingPointError('native policy objective must return a finite scalar loss')
        return loss, details
    hidden = result['h_final'][0]
    losses, details = [], []
    for graph_index, (first_node, last_node) in enumerate(ownership):
        queries, targets, weights = [], [], []
        for node_index in range(first_node, last_node):
            node, offset = plan.invocations[node_index], plan.offsets[node_index]
            for local in range(1, len(node.token_ids)):
                target = offset + local
                weight = target_weights[target]
                if weight > 0:
                    queries.append(target - 1)
                    targets.append(int(node.token_ids[local]))
                    weights.append(weight)
        if not queries:
            raise ValueError(f'native graph {examples[graph_index].id!r} has no supervised target')
        query_ids = torch.tensor(queries, device=hidden.device)
        target_ids = torch.tensor(targets, device=hidden.device)
        token_weights = torch.tensor(weights, device=hidden.device, dtype=torch.float32)
        total = hidden.new_zeros((), dtype=torch.float32)
        correct = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, len(queries), chunk_tokens):
            stop = start + chunk_tokens
            def loss_chunk(states, labels, selected):
                logits = model.logits(states)
                ce = F.cross_entropy(logits.float(), labels, reduction='none')
                return ((ce * selected).sum(),
                        ((logits.argmax(-1) == labels).float() * selected).sum())
            chunk_loss, chunk_correct = checkpoint(loss_chunk, hidden[query_ids[start:stop]],
                target_ids[start:stop], token_weights[start:stop], use_reentrant=False)
            total = total + chunk_loss
            correct = correct + chunk_correct
        denominator = token_weights.sum()
        losses.append(total / denominator)
        details.append({'weight': float(denominator.detach()),
                        'supervised_tokens': len(queries),
                        'accuracy': float((correct / denominator).detach())})
    return torch.stack(losses).mean(), details


def _row_bindings(examples):
    """Exact semantic identities needed to prove an append-only data continuation."""
    result = {}
    for item in examples:
        value = {'id': item.id, 'groups': item.source_groups, 'split': item.split,
                 'tokenizer': item.tokenizer_fingerprint, 'targets': item.target_weights,
                 'external_names': item.plan.external_names,
                 'external_sizes': item.plan.external_sizes,
                 'external_states': getattr(item, 'external_states', {}),
                 'nodes': [{'name': node.name, 'token_ids': node.token_ids,
                            'imports': [edge.__dict__ for edge in node.imports],
                            'producing': node.producing, 'logical_producing': node.logical_producing}
                           for node in item.plan.invocations]}
        result[item.id] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return result


def append_only_rows(previous, current, *, label):
    """Prove a continuation retained every prior row byte-semantically and only added rows."""
    if not previous:
        raise ValueError(f'native SFT checkpoint has no exact {label} row bindings')
    changed = [key for key, value in previous.items() if current.get(key) != value]
    if changed:
        raise ValueError(f'native SFT {label} continuation modified/removed prior rows: ' + ', '.join(changed[:8]))
    return set(current) - set(previous)


def scheduled_indices(examples, *, seed, epoch, prior_ids=None):
    """Stable epoch order; on append, preserve the old permutation as an exact prefix."""
    order = sorted(range(len(examples)), key=lambda i: hashlib.sha256(
        f'{seed}:{epoch}:{examples[i].id}'.encode()).digest())
    if prior_ids is None:
        return order
    old = [index for index in order if examples[index].id in prior_ids]
    new = [index for index in order if examples[index].id not in prior_ids]
    return old + new


def _evaluate(model, examples, *, checkpoint_layers, chunk_tokens, backend=None):
    from ..data.native_graph import graph_loss
    previous_mode = model.training
    model.eval()
    scope = backend.native_modules() if backend is not None else contextlib.nullcontext()
    try:
        with scope, torch.no_grad():
            total_weight, total_loss, total_correct, total_tokens, total_graphs, forward_tokens = 0., 0., 0., 0, 0, 0
            for example in examples:
                loss, detail = graph_loss(model, example, checkpoint_layers=checkpoint_layers,
                                          chunk_tokens=chunk_tokens)
                weight = detail['weight']
                total_loss += float(loss) * weight
                total_correct += detail['accuracy'] * weight
                total_weight += weight
                total_tokens += detail['supervised_tokens']
                total_graphs += 1
                forward_tokens += example.plan.token_count
    finally:
        model.train(previous_mode)
    if not math.isfinite(total_loss) or total_weight <= 0:
        raise FloatingPointError('native held evaluation is nonfinite or empty')
    return {'loss': total_loss / total_weight, 'weighted_accuracy': total_correct / total_weight,
            'weight': total_weight, 'supervised_tokens': total_tokens, 'examples': total_graphs,
            'forward_tokens': forward_tokens}


def held_cadence_indices(entries, cursor=0, *, token_budget=0, graph_budget=0):
    """Rotate complete held graphs; never truncate a graph to fit a cadence."""
    if min(cursor, token_budget, graph_budget) < 0 or not entries:
        raise ValueError('held cadence needs nonempty data and nonnegative budgets/cursor')
    indices, tokens = [], 0
    for advance in range(len(entries)):
        index = (cursor+advance) % len(entries)
        size = entries[index]['forward_tokens']
        if indices and ((token_budget and tokens+size > token_budget)
                        or (graph_budget and len(indices) >= graph_budget)):
            break
        indices.append(index)
        tokens += size
    return indices, (indices[-1]+1) % len(entries)


def _evaluate_cadence(model, source, *, args, coverage=None, final=False, backend=None):
    coverage = coverage or {}
    if final:
        indices, cursor = list(range(len(source))), 0
    else:
        indices, cursor = held_cadence_indices(source.entries, int(coverage.get('cursor', 0)),
            token_budget=args.held_eval_tokens, graph_budget=args.held_eval_graphs)
    started = time.monotonic()
    result = _evaluate(model, source.iter_indices(indices), checkpoint_layers=args.checkpoint_layers,
                       chunk_tokens=args.chunk_tokens, backend=backend)
    result.update(scope='full' if len(indices) == len(source) else 'cadence-subset',
                  source_sha256=source.file_sha256,
                  source_graphs=len(source), cursor_before=int(coverage.get('cursor', 0)),
                  cursor_after=cursor, seconds=time.monotonic()-started)
    coverage = {'cursor': cursor, 'graph_visits': int(coverage.get('graph_visits', 0))+len(indices),
        'forward_token_visits': int(coverage.get('forward_token_visits', 0))+result['forward_tokens'],
        'completed_sweeps': int(coverage.get('completed_sweeps', 0)) +
            (1 if final else (int(coverage.get('cursor', 0)) % len(source)+len(indices))//len(source))}
    return result, coverage


def _save(policy, model, optimizer, *, config, identity, step, epoch, cursor, source_receipts,
          source_hash, source_row_bindings, held_row_bindings, exposure_counts, global_step, rng, metrics,
          architecture, options, fp8_state=None):
    state = {'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model), 'step': global_step,
             'training': {'schema': TRAINING_SCHEMA, 'identity': identity, 'step': step,
                          'global_step': global_step, 'epoch': epoch,
                          'source_row_bindings': source_row_bindings,
                          'source_row_order': options.get('_source_row_order', []),
                          'held_row_bindings': held_row_bindings,
                          'exposure_counts': exposure_counts,
                          'cursor': cursor, 'source_hash': source_hash, 'source_receipts': source_receipts,
                          'optimizer': optimizer.state_dict(), 'rng': rng, 'metrics': metrics,
                          'options': options, 'fp8': fp8_state}}
    policy.save(state, metric=metrics.get('held', {}).get('loss'))


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-graphs', type=Path, required=True)
    parser.add_argument('--held-native-graphs', type=Path, required=True)
    parser.add_argument('--continued-native-graphs', type=Path)
    parser.add_argument('--continued-held-native-graphs', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--base', type=Path)
    parser.add_argument('--continue-from', type=Path)
    parser.add_argument('--native-participant')
    parser.add_argument('--objective-continuation', action='store_true',
        help='explicit new source/objective stage; preserve full optimizer/RNG and start local exposure at zero')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--rank', type=int, default=8)
    parser.add_argument('--epochs', type=float, default=1.)
    parser.add_argument('--batch-graphs', type=int, default=1)
    parser.add_argument('--batch-tokens', type=int, default=0,
        help='pack whole sequential graphs to this token volume instead of graph-count batching')
    parser.add_argument('--dtype', choices=('float32', 'bfloat16', 'float16'), default=None,
        help='fresh backbone precision; required for a new lineage, resumes retain exact checkpoint dtype')
    from ..model.native_precision import PRODUCTION_NATIVE_LINEAR_PRECISIONS
    parser.add_argument('--native-linear-precision', choices=PRODUCTION_NATIVE_LINEAR_PRECISIONS, default=None,
        help='explicit native GEMM arithmetic transition; preserves source dtype, optimizer, RNG and row exposure')
    parser.add_argument('--learning-rate', type=float, default=3e-6)
    parser.add_argument('--weight-decay', type=float, default=.1)
    parser.add_argument('--latent-ns', choices=('quintic', 'cubic5'), default='quintic')
    parser.add_argument('--eval-every', type=int, default=100)
    parser.add_argument('--held-eval-tokens', type=int, default=0,
        help='rotating whole-graph held-token budget per cadence; final evaluation always covers all held data')
    parser.add_argument('--held-eval-graphs', type=int, default=0)
    parser.add_argument('--checkpoint-every', type=int, default=0)
    parser.add_argument('--checkpoint-seconds', type=float, default=3600.)
    parser.add_argument('--checkpoint-layers', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--chunk-tokens', type=int, default=128)
    parser.add_argument('--fp8-training', choices=('off', 'te-fp8', 'triton-experts'), default='off')
    parser.add_argument('--fp8-recipe', choices=('delayed', 'current'), default='delayed')
    parser.add_argument('--fp8-amax-history-len', type=int, default=16)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--task-eval', default=None)
    parser.add_argument('--task-eval-every', type=int, default=0)
    parser.add_argument('--forward-atol', type=float, default=None)
    parser.add_argument('--gradient-atol', type=float, default=None)
    parser.add_argument('--recipe-binding', default=None)
    parser.add_argument('--stage-id', default=None)
    parser.add_argument('--prerequisite-stages', default='[]')
    parser.add_argument('--objective-spec', type=Path, default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    checkpoint_options = {key: str(value) if isinstance(value, Path) else value
                          for key, value in vars(args).items()}
    if not (math.isfinite(args.epochs) and args.epochs > 0) or min(args.batch_graphs, args.eval_every,
            args.chunk_tokens, args.rank) < 1 or min(args.checkpoint_every, args.batch_tokens, args.held_eval_tokens, args.held_eval_graphs) < 0:
        raise ValueError('native SFT requires positive epochs/batch/eval/chunk/rank and nonnegative checkpoint cadence')
    if not math.isfinite(args.checkpoint_seconds) or args.checkpoint_seconds < 0:
        raise ValueError('checkpoint-seconds must be finite and nonnegative')
    if args.checkpoint_every and args.checkpoint_every % args.eval_every:
        raise ValueError('checkpoint cadence must be an evaluation cadence multiple')
    tolerances = [value for value in (args.forward_atol, args.gradient_atol) if value is not None]
    if args.task_eval_every < 0 or any(not math.isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError('task cadence must be nonnegative and supplied native numerical tolerances positive')
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    policy = CheckpointPolicy(args.out, every_steps=args.checkpoint_every,
                              every_seconds=args.checkpoint_seconds, prune_best=False)
    policy.install_signal_handlers()
    resume = None
    parent_continuation = False
    shared_resume = None
    if args.continue_from and not policy.slot.exists():
        model, tokenizer, roles, config, parent_state = load_native_checkpoint(args.continue_from, base=args.base,
                                                                          device=args.device, participant=args.native_participant)
        shared_resume = parent_state.get('shared_training')
        resume = parent_state.get('training')
        if shared_resume is not None:
            if not args.objective_continuation or not shared_resume.get('optimizer') or not shared_resume.get('rng'):
                raise ValueError('shared native parent requires declared objective continuation and full optimizer/RNG')
        elif not isinstance(resume, dict) or resume.get('schema') != TRAINING_SCHEMA:
            raise ValueError('native SFT continuation lacks exact optimizer/schedule/RNG state')
        parent_continuation = not policy.slot.exists()
        if parent_continuation:
            del parent_state
    elif policy.slot.exists():
        model, tokenizer, roles, config, state = load_native_checkpoint(policy.slot, base=args.base,
                                                                         device=args.device)
        if state.get('schema') != CHECKPOINT_SCHEMA:
            raise ValueError('rolling slot is not a native-state checkpoint')
        resume = state.get('training')
    else:
        if args.base is None:
            raise ValueError('fresh native SFT needs --base')
        if args.dtype is None:
            raise ValueError('fresh native SFT requires explicit --dtype (QAD LFM float32; Mellum bfloat16)')
        model, tokenizer, roles, config = load_native_model(args.base, rank=args.rank,
            device=args.device, dtype=getattr(torch, args.dtype))
    from ..model.native_precision import set_native_linear_precision, resolve_native_training_precision
    previous_precision = config.get('native_linear_precision', 'ieee')
    requested_precision = resolve_native_training_precision(previous_precision, args.native_linear_precision)
    set_native_linear_precision(model, requested_precision)
    config['native_linear_precision'] = requested_precision
    saved_training = resume or shared_resume or {}
    precision_history = list((saved_training.get('options') or {}).get('native_precision_history') or [])
    transition = {'from': previous_precision, 'to': requested_precision,
        'source_dtype': config['dtype'],
        'at_global_step': int(saved_training.get('global_step', saved_training.get('step', 0))),
        'weights_optimizer_rng_exposure': 'preserved' if saved_training else 'fresh'}
    checkpoint_options['native_precision_transition'] = transition
    if requested_precision != previous_precision:
        precision_history.append(transition)
    checkpoint_options['native_precision_history'] = precision_history
    if args.dtype is not None and config['dtype'] != str(getattr(torch, args.dtype)):
        raise ValueError('declared dtype differs from the exact restored native checkpoint')
    if resume is None and (args.continued_native_graphs or args.continued_held_native_graphs):
        raise ValueError('native data continuation predecessors require an actual resumable checkpoint')
    if config['rank'] != args.rank and not args.continue_from and not policy.slot.exists():
        raise ValueError('native rank differs from declared stage rank')
    from ..data.text_corpus import tokenizer_fingerprint
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != config['tokenizer_fingerprint']:
        raise ValueError('loaded tokenizer differs from native model configuration')
    train = NativeGraphSource(args.native_graphs, fingerprint, required_split='train', role_ids=roles.ids)
    held = NativeGraphSource(args.held_native_graphs, fingerprint, required_split='test', role_ids=roles.ids)
    train_groups, held_groups = train.groups, held.groups
    if train_groups & held_groups:
        raise ValueError('native train/held source groups overlap: ' + ', '.join(sorted(train_groups & held_groups)[:8]))
    source_receipts = [train.receipt(), held.receipt()]
    source_row_bindings = train.bindings
    held_row_bindings = held.bindings
    source_hash = hashlib.sha256(json.dumps(
        [source_receipts[0]['sha256'], source_receipts[1]['sha256'],
         sorted(source_row_bindings.items()), sorted(held_row_bindings.items())],
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    source_order = train.ordered_ids()
    objective = None
    objective_binding = None
    if args.objective_spec is not None:
        spec_sha256 = sha256_file_hex_cached(args.objective_spec)
        objective_binding = {'path': str(args.objective_spec.resolve()), 'sha256': spec_sha256}
        spec = json.loads(args.objective_spec.read_text())
        if spec.get('schema') != 'natlang.native-policy-objective/1' or spec.get('kind') != 'native-policy':
            raise ValueError('native replay objective spec must declare natlang.native-policy-objective/1 kind native-policy')
        from ..rl.native_replay import NativePolicyObjective
        objective = NativePolicyObjective(args.objective_spec)
        if parent_continuation:
            actual = native_identity(model, config, checkpoint_sha256=sha256_file_hex_cached(args.continue_from))
            if objective.identity != actual:
                raise ValueError('native policy objective differs from the actual loaded actor embedding/checkpoint identity')
    training_identity = {'schema': TRAINING_SCHEMA, 'config': config, 'train_source_hash': source_hash,
                'schedule': 'row-visit-deficit/2',
                'held_sha256': source_receipts[1]['sha256'], 'optimizer': 'shared-muon-sr-in-backward-v1',
                'optimizer_options': {'lr': args.learning_rate, 'weight_decay': args.weight_decay,
                                      'latent_ns': args.latent_ns},
                'batch_graphs': args.batch_graphs, 'batch_tokens': args.batch_tokens, 'seed': args.seed, 'rank': config['rank'],
                'fp8': {'kind': args.fp8_training, 'recipe': args.fp8_recipe,
                        'amax_history_len': args.fp8_amax_history_len},
                'objective': objective_binding}
    appended_ids = set()
    if shared_resume is not None:
        old_options = shared_resume['options']
        old_optimizer = {'weight_decay': old_options.get('weight_decay', .1),
                         'latent_ns': old_options.get('latent_ns', 'quintic')}
        resume = {'identity': {**training_identity, 'optimizer_options': old_optimizer},
            'optimizer': shared_resume['optimizer'], 'rng': shared_resume['rng'],
            'fp8': shared_resume.get('fp8'), 'step': 0, 'global_step': shared_resume['step']}
    if resume is not None:
        predecessors = dict((resume.get('options') or {}).get('source_predecessors') or {})
        for role, position in (('continued_native_graphs', 0), ('continued_held_native_graphs', 1)):
            predecessor = getattr(args, role)
            if predecessor is not None:
                prior_receipts = resume.get('source_receipts') or []
                digest = sha256_file_hex_cached(predecessor)
                # Once adopted, the original predecessor stays valid across
                # later append/priority manifests. Exact immutable row bindings
                # and the prior physical prefix are verified below.
                adopted = predecessors.get(role) == digest
                if (not adopted and (len(prior_receipts) <= position
                        or digest != prior_receipts[position]['sha256'])):
                    raise ValueError('native data continuation predecessor differs from the actual checkpoint: ' + role)
                predecessors[role] = digest
        checkpoint_options['source_predecessors'] = predecessors
        previous = resume.get('identity') or {}
        stable = ('schema', 'optimizer', 'seed', 'rank', 'fp8', 'objective')
        from ..model.native_precision import same_source_precision_identity
        if not same_source_precision_identity(previous.get('config') or {}, config):
            raise ValueError('native SFT continuation changes model identity beyond declared GEMM arithmetic')
        if previous.get('schedule') not in {'source-order-stream/1', 'row-visit-deficit/2'}:
            raise ValueError('native SFT checkpoint has an unsupported row-credit schedule')
        if parent_continuation and args.objective_continuation:
            stable = tuple(key for key in stable if key not in {'objective', 'batch_graphs', 'batch_tokens'})
        if any(previous.get(key, 0 if key == 'batch_tokens' else None) != training_identity.get(key) for key in stable):
            raise ValueError('native SFT continuation changes model/held/optimizer/schedule identity')
        # MuonSR retains its constructor's NS variant on state restoration;
        # switching the update approximation preserves compatible moment state.
        if not (parent_continuation and args.objective_continuation):
            prior_bindings = resume.get('source_row_bindings') or {}
            appended_ids = append_only_rows(prior_bindings, source_row_bindings, label='training-source')
            prior_order = resume.get('source_row_order') or []
            if source_order[:len(prior_order)] != prior_order:
                raise ValueError('native SFT continuation reordered prior training rows; exact cursor cannot be resumed')
            prior_held = resume.get('held_row_bindings') or {}
            append_only_rows(prior_held, held_row_bindings, label='held-source')
        elif shared_resume is None:
            append_only_rows(resume.get('held_row_bindings') or {}, held_row_bindings, label='held-source')
        if appended_ids:
            print(json.dumps({'event': 'native_data_append', 'prior_rows': len(prior_bindings),
                              'new_rows': len(appended_ids), 'optimizer_state': 'restored',
                              'cursor': resume.get('cursor'), 'policy': 'immutable source-order prefix then appended rows'}),
                  flush=True)
        training_identity['optimizer_options'] = {'lr': args.learning_rate,
            'weight_decay': args.weight_decay, 'latent_ns': args.latent_ns,
            'continued_from': previous.get('optimizer_options')}
        training_identity['prior_source_hash'] = previous.get('train_source_hash')
    if resume is not None and not parent_continuation and (resume.get('options') or {}).get('continuation'):
        checkpoint_options['continuation'] = resume['options']['continuation']
    if parent_continuation:
        checkpoint_options['continuation'] = {'checkpoint_sha256': sha256_file_hex_cached(args.continue_from),
            'participant': args.native_participant, 'objective_transition': bool(args.objective_continuation)}
    if args.fp8_training != 'off':
        arch = config['architecture']
        expected = 'triton-experts' if arch in {'mellum', 'maple'} else 'te-fp8'
        if args.fp8_training != expected:
            raise ValueError(f'{arch} native SFT requires {expected} for the declared FP8 path')
        from .fp8_backend import make_backend
        backend = make_backend(args.fp8_training, model.port.hf, device=args.device, recipe=args.fp8_recipe,
                               amax_history_len=args.fp8_amax_history_len)
        model.port.activation_checkpoint = backend.checkpoint
    else:
        backend = None
    if config['architecture'] in {'mellum', 'maple'} and backend is None:
        raise ValueError('native Mellum/Maple production SFT requires its shared triton-experts FP8 memory path')
    model.port.checkpoint_layers = args.checkpoint_layers
    model.to(args.device)
    model.train()
    optimizer, named = make_native_optimizer(model, lr=args.learning_rate,
                                              weight_decay=args.weight_decay, latent_ns=args.latent_ns)
    task_eval = None
    task_spec = None
    if args.task_eval:
        from ..eval.task_cadence import TaskEval, parse_task_eval
        task_spec = parse_task_eval(args.task_eval)
        from ..serve.engine import Engine
        from ..serve.store import TensorStore
        # Construct the real native serving engine over the same live model,
        # avoiding a second full-width model allocation during evaluation.
        serving_identity = native_identity(model, config, checkpoint_sha256='0' * 64)
        engine = Engine(model.port, None, tokenizer, TensorStore(),
            f"native-state/{config['architecture']}", model_name=f"native-{config['architecture']}",
            device=args.device, prompt_rendering=task_spec.get('prompt_rendering', 'full'),
            shared_model=model, role_tokens=roles, model_config=config,
            checkpoint_sha256=serving_identity['checkpoint_sha256'],
            embedding_weight_sha256=serving_identity['embedding_weight_sha256'])
        task_eval = TaskEval(task_spec, args.out, engine, prompt_rendering=task_spec.get('prompt_rendering', 'full'))
    step, epoch, cursor, global_step = 0, 0, 0, 0
    metrics = {}
    exposure_counts = {key: 0 for key in source_row_bindings}
    if resume is not None:
        optimizer.load_state_dict(resume['optimizer'])
        # Adaptive production may tune the step size without discarding moments.
        # The new value is recorded in optimizer_options while moments/RNG stay exact.
        for group in optimizer.param_groups:
            group['lr'] = args.learning_rate
            if 'qat_latent_lr' in group:
                group['qat_latent_lr'] = args.learning_rate
        if optimizer.latent is not None:
            for group in optimizer.latent.param_groups:
                group['weight_decay'] = args.weight_decay
        if backend is not None:
            if resume.get('fp8') is None:
                raise ValueError('native SFT checkpoint lacks the declared FP8 state')
            backend.restore_extra_state(resume['fp8'])
        if parent_continuation:
            # A new declared phase consumes its current full source set while
            # keeping exact weights/optimizer/RNG; its stage-local cursor begins
            # at zero and the global update counter remains monotonic. The
            # predecessor's exposure is in its own readiness receipt, so this
            # stage starts its own exposure counters at zero.
            global_step = int(resume.get('global_step', resume['step']))
            metrics = {}
        else:
            step, epoch, cursor = int(resume['step']), int(resume['epoch']), int(resume['cursor'])
            global_step = int(resume.get('global_step', step))
            exposure_counts.update({key: int(value) for key, value in (resume.get('exposure_counts') or {}).items()
                                    if key in exposure_counts})
            metrics = dict(resume.get('metrics') or {})
        restore_training_rng_state(resume['rng'], args.device)
        resume['optimizer'] = None
        if 'state' in locals():
            del state
        if 'parent_state' in locals():
            del parent_state
    # Actual per-row optimizer credits drive the schedule. Source identity
    # remains append-only while communication deficits execute first.
    whole_epochs = int(args.epochs)
    fractional_rows = math.ceil((args.epochs - whole_epochs) * len(train))
    target_rows = whole_epochs * len(train) + fractional_rows
    # Cursor/exposure is the durable schedule, not the old batch partition.
    # Adapt packing after a safe stop without replaying or skipping any row.
    if parent_continuation:
        step = 0
    required_visits, pending_batches = pending_exposure_batches(train.entries, exposure_counts,
        epochs=args.epochs, batch_graphs=args.batch_graphs, batch_tokens=args.batch_tokens)
    target_rows = sum(required_visits.values())
    total_end = step + len(pending_batches)
    if step > total_end:
        raise ValueError('native checkpoint update is beyond the declared epoch exposure')
    resumed_step = step
    last_saved_step = step if policy.slot.is_file() and not parent_continuation else -1
    source_iters = {}
    pass_indices = {}
    for visit, indices, _ in pending_batches:
        pass_indices.setdefault(visit, []).extend(indices)
    schedule = iter(pending_batches)
    eval_at = max(1, args.eval_every)
    started = time.time()
    while step < total_end and not policy.signaled:
        update_started = time.monotonic()
        visit, indices, pass_end = next(schedule)
        if visit not in source_iters:
            source_iters[visit] = train.iter_indices(pass_indices[visit])
        source_iter = source_iters[visit]
        try:
            batch = [next(source_iter) for _ in indices]
        except StopIteration as error:
            raise ValueError('native graph source ended before its validated row count') from error
        if [item.id for item in batch] != [train.entries[index]['id'] for index in indices]:
            raise ValueError('native graph stream differs from scheduled optimizer-credit order')
        batch_tokens = sum(train.entries[index]['forward_tokens'] for index in indices)
        cursor = indices[-1] + 1
        loss, details = native_training_update(model, optimizer, batch, backend=backend,
            checkpoint_layers=args.checkpoint_layers, chunk_tokens=args.chunk_tokens, objective=objective)
        step += 1
        global_step += 1
        for example in batch:
            exposure_counts[example.id] += 1
        epoch = min(exposure_counts[key] for key, required in required_visits.items() if required > 0)
        if pass_end and all(exposure_counts[key] >= required for key, required in required_visits.items()):
            cursor = 0
        # Telemetry is appended after the actual optimizer update. The rolling
        # checkpoint remains the authority for committed resume/exposure; logs
        # can include work subsequently replayed after an uncheckpointed crash.
        train_loss = float(loss.detach())
        seconds = time.monotonic() - update_started
        supervised_tokens = sum(sum(weight > 0 for weight in item.target_weights) for item in batch)
        progress = {'schema': 'natlang.native-training-progress/1', 'kind': 'native',
                    'step': step, 'global_step': global_step, 'epoch': epoch, 'cursor': cursor,
                    'loss': train_loss, 'graphs': len(batch), 'forward_tokens': batch_tokens,
                    'supervised_tokens': supervised_tokens, 'target_tokens': supervised_tokens,
                    'seconds': seconds, 'forward_tokens_per_second': batch_tokens / max(seconds, 1e-9),
                    'supervised_tokens_per_second': supervised_tokens / max(seconds, 1e-9),
                    'elapsed_seconds': time.time() - started, 'timestamp': time.time(),
                    'checkpoint_step': last_saved_step,
                    'communication_exposure': communication_exposure(train.entries, exposure_counts, required_visits)}
        if str(args.device).startswith('cuda'):
            progress['cuda_memory_bytes'] = {'allocated': torch.cuda.memory_allocated(args.device),
                'reserved': torch.cuda.memory_reserved(args.device),
                'lifetime_peak_allocated': torch.cuda.max_memory_allocated(args.device)}
        with (args.out / 'train.jsonl').open('a') as log:
            log.write(json.dumps(progress, sort_keys=True) + '\n')
        write_json(args.out / 'progress.json', progress)
        print(json.dumps(progress, sort_keys=True), flush=True)
        due_task = task_eval is not None and (step == total_end or
                    bool(args.task_eval_every and step % args.task_eval_every == 0))
        due_eval = step % eval_at == 0 or step == total_end or due_task
        if due_eval:
            held_metrics, held_coverage = _evaluate_cadence(model, held, args=args,
                coverage=metrics.get('held_coverage'), final=step == total_end, backend=backend)
            metrics = {'train_loss': train_loss, 'held': held_metrics, 'held_coverage': held_coverage,
                       'step': step, 'epoch': epoch, 'cursor': cursor,
                       'elapsed_seconds': time.time() - started,
                       'fp8': backend.inventory() if backend is not None else {'enabled': False}}
            write_json(args.out / 'metrics.json', metrics)
        if due_task:
            # Bind the task proof to the checkpoint from these exact weights.
            _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
                  global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
                  source_hash=source_hash, source_row_bindings=source_row_bindings,
                  held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
                  rng=capture_training_rng_state(args.device), metrics=metrics,
                  architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
                  fp8_state=backend.extra_state() if backend is not None else None)
            last_saved_step = step
            checkpoint_sha = sha256_file_hex_cached(policy.slot)
            serving_identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha)
            engine = task_eval.engine
            engine.checkpoint_sha256 = serving_identity['checkpoint_sha256']
            engine.embedding_weight_sha256 = serving_identity['embedding_weight_sha256']
            engine.native_identity = serving_identity
            saved_rng = capture_training_rng_state(args.device)
            model.eval()
            try:
                if backend is not None:
                    with backend.native_modules():
                        metrics['task_eval'] = task_eval.cadence(step)
                else:
                    metrics['task_eval'] = task_eval.cadence(step)
            finally:
                model.train()
                restore_training_rng_state(saved_rng, args.device)
            metrics['task_checkpoint_sha256'] = checkpoint_sha
            metrics['task_native_identity'] = serving_identity
            write_json(args.out / 'metrics.json', metrics)
        if policy.due(step, end=step == total_end) and not due_task:
            _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
                  global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
                  source_hash=source_hash, source_row_bindings=source_row_bindings,
                  held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
                  rng=capture_training_rng_state(args.device), metrics=metrics,
                  architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
                  fp8_state=backend.extra_state() if backend is not None else None)
            last_saved_step = step
    if (step != resumed_step or policy.signaled) and last_saved_step != step:
        if not policy.signaled and not metrics.get('held'):
            metrics['held'], metrics['held_coverage'] = _evaluate_cadence(model, held, args=args,
                coverage=metrics.get('held_coverage'), final=step == total_end, backend=backend)
        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
              global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
              source_hash=source_hash, source_row_bindings=source_row_bindings,
              held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
              rng=capture_training_rng_state(args.device), metrics=metrics,
              architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
              fp8_state=backend.extra_state() if backend is not None else None)
        last_saved_step = step
    completed = all(exposure_counts[key] >= visits for key, visits in required_visits.items())
    if completed and not policy.signaled and (metrics.get('held', {}).get('scope') != 'full'
            or metrics.get('held', {}).get('source_sha256') != held.file_sha256):
        metrics['held'], metrics['held_coverage'] = _evaluate_cadence(model, held, args=args,
            coverage=metrics.get('held_coverage'), final=True, backend=backend)
        metrics.pop('task_checkpoint_sha256', None)
        metrics.pop('task_native_identity', None)
        _save(policy, model, optimizer, config=config, identity=training_identity, step=step,
            global_step=global_step, epoch=epoch, cursor=cursor, source_receipts=source_receipts,
            source_hash=source_hash, source_row_bindings=source_row_bindings,
            held_row_bindings=held_row_bindings, exposure_counts=exposure_counts,
            rng=capture_training_rng_state(args.device), metrics=metrics,
            architecture=config['architecture'], options={**checkpoint_options, '_source_row_order': source_order},
            fp8_state=backend.extra_state() if backend is not None else None)
    completed = (completed and metrics.get('held', {}).get('scope') == 'full'
                 and metrics.get('held', {}).get('source_sha256') == held.file_sha256)
    # A continuation may inherit a completed checkpoint made before task
    # summaries were checkpointed. At the same final weights, make one real
    # cadence call; do not reuse an unbound log row or enter final_at's retry
    # loop merely to force an eligible score.
    if task_eval is not None and completed and not metrics.get('task_checkpoint_sha256'):
        current_sha = sha256_file_hex_cached(policy.slot)
        serving_identity = native_identity(model, config, checkpoint_sha256=current_sha)
        engine = task_eval.engine
        engine.checkpoint_sha256 = serving_identity['checkpoint_sha256']
        engine.embedding_weight_sha256 = serving_identity['embedding_weight_sha256']
        engine.native_identity = serving_identity
        saved_rng = capture_training_rng_state(args.device)
        model.eval()
        try:
            if backend is not None:
                with backend.native_modules():
                    metrics['task_eval'] = task_eval.cadence(step)
            else:
                metrics['task_eval'] = task_eval.cadence(step)
        finally:
            model.train()
            restore_training_rng_state(saved_rng, args.device)
        metrics['task_checkpoint_sha256'] = current_sha
        metrics['task_native_identity'] = serving_identity
        write_json(args.out / 'metrics.json', metrics)
    checkpoint_sha = sha256_file_hex_cached(policy.slot)
    model_identity = native_identity(model, config, checkpoint_sha256=checkpoint_sha)
    task_summary = metrics.get('task_eval')
    task_receipt_path = None
    if task_summary:
        task_receipt_path = args.out / 'native-task-execution-receipt.json'
        task_checkpoint_sha = metrics.get('task_checkpoint_sha256')
        task_identity = metrics.get('task_native_identity')
        score_path = Path(task_summary['run']) / 'task-score.json' if task_summary.get('run') else None
        task_receipt = {'schema': 'natlang.native-task-execution-receipt/1',
            'checkpoint_sha256': task_checkpoint_sha, 'native_identity': task_identity,
            'packet_sha256': task_summary.get('packet_sha256'),
            'case_indices': task_summary.get('case_indices'), 'cases': task_summary.get('cases'),
            'complete': task_summary.get('complete'), 'eligible': task_summary.get('eligible'),
            'task_successes': task_summary.get('task_successes'),
            'task_successes_are_qualification_barrier': False,
            'score_path': str(score_path.resolve()) if score_path and score_path.is_file() else None,
            'score_sha256': sha256_file_hex_cached(score_path) if score_path and score_path.is_file() else None}
        task_receipt['passed'] = bool(task_checkpoint_sha == checkpoint_sha and task_identity and
                                      task_receipt['complete'] and task_receipt['eligible'] and
                                      task_receipt['score_sha256'])
        write_json(task_receipt_path, task_receipt)
    runtime_path = args.out / 'native-runtime-certificate.json'
    if not completed:
        # Operational stops preserve the exact resumable checkpoint. Defer a
        # full capture/splice qualification until the declared data exposure is
        # finished; do not make interruptions pay for a second model pass.
        runtime_report = {'schema': 'natlang.native-runtime-certificate/1', 'passed': False,
                          'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
                          'deferred': True, 'reason': 'declared native graph exposure is incomplete'}
    else:
        try:
            if args.forward_atol is None or args.gradient_atol is None:
                raise ValueError('runtime qualification requires recipe-bound forward and gradient tolerances')
            from ..eval.native_state import qualify_existing
            if backend is not None:
                with backend.native_modules():
                    runtime_report = qualify_existing(model, tokenizer, roles, config,
                        checkpoint_sha256=checkpoint_sha, forward_atol=args.forward_atol,
                        gradient_atol=args.gradient_atol,
                        task_execution_receipt=(str(task_receipt_path.resolve()) if task_receipt_path else None),
                        backbone_optimizer_updates=step)
            else:
                runtime_report = qualify_existing(model, tokenizer, roles, config,
                    checkpoint_sha256=checkpoint_sha, forward_atol=args.forward_atol,
                    gradient_atol=args.gradient_atol,
                    task_execution_receipt=(str(task_receipt_path.resolve()) if task_receipt_path else None),
                    backbone_optimizer_updates=step)
        except Exception as error:  # record a real failed check without discarding the trained checkpoint
            runtime_report = {'schema': 'natlang.native-runtime-certificate/1', 'passed': False,
                              'checkpoint_sha256': checkpoint_sha, 'native_identity': model_identity,
                              'error': f'{type(error).__name__}: {error}'}
    write_json(runtime_path, runtime_report)
    task_receipt_ok = bool(task_receipt_path and json.loads(task_receipt_path.read_text()).get('passed'))
    recipe_binding = json.loads(args.recipe_binding) if args.recipe_binding else None
    prerequisites = json.loads(args.prerequisite_stages)
    exposure_by_source_group = {}
    for entry in train.entries:
        visits = int(exposure_counts[entry['id']])
        token_visits = visits * entry['supervised_tokens']
        for group in entry['groups']:
            counts = exposure_by_source_group.setdefault(group, {'graph_visits': 0,
                                                                   'supervised_token_visits': 0})
            counts['graph_visits'] += visits
            counts['supervised_token_visits'] += token_visits
    exposure_counts_sha256 = hashlib.sha256(json.dumps(sorted(exposure_counts.items()),
        ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    communication = communication_exposure(train.entries, exposure_counts, required_visits)
    report = {'schema': 'natlang.neuralese-native-sft-report/1',
              'communication_exposure': communication, 'completed': completed,
              'numerics_ok': runtime_report.get('passed') is True, 'qualified': False,
              'quality_status': 'measured-for-supervisor',
              'train_exposure': {key: sum(exposure_counts[entry['id']] * entry['supervised_tokens']
                  for entry in train.entries) for key in ('target_tokens', 'supervised_tokens')},
              'step': step, 'epoch': epoch, 'cursor': cursor, 'training_seconds': time.time() - started,
              'model_config': config, 'continuation': checkpoint_options.get('continuation'), 'train_source': source_receipts[0], 'held_source': source_receipts[1],
              'source_hash': source_hash, 'metrics': metrics, 'checkpoint': str(policy.slot),
              'checkpoint_sha256': checkpoint_sha, 'task_eval': task_summary,
              'runtime_certificate': {'path': str(runtime_path.resolve()),
                                      'sha256': sha256_file_hex_cached(runtime_path),
                                      'passed': runtime_report.get('passed') is True},
              'base_kl': {'status': 'telemetry-only', 'measured': False}}
    write_json(args.out / 'report.json', report)
    write_json(args.out / 'native-training-receipt.json', {
        'schema': 'natlang.native-training-readiness/1',
        'communication_exposure': communication, 'model_id': config['architecture'] + ':' +
        config['base_inventory_sha256'], 'lineage': str(args.out.resolve()),
        'stage_id': args.stage_id or args.out.name,
        'checkpoint': {'path': str(policy.slot.resolve()), 'sha256': report['checkpoint_sha256'],
                       'schema': CHECKPOINT_SCHEMA},
        'native_identity': model_identity, 'continuation': checkpoint_options.get('continuation'),
        'runtime_certificate': {'path': str(runtime_path.resolve()),
                                'sha256': sha256_file_hex_cached(runtime_path),
                                'passed': runtime_report.get('passed') is True},
        'prerequisite_stages': prerequisites, 'recipe': recipe_binding,
        'completed': bool(completed and communication['complete'] and
                          runtime_report.get('passed') is True and task_receipt_ok and
                          all(item.get('completed') is True for item in prerequisites)),
        'task_eval': task_summary,
        'task_execution_receipt': ({'path': str(task_receipt_path.resolve()),
                                     'sha256': sha256_file_hex_cached(task_receipt_path), 'passed': task_receipt_ok}
                                    if task_receipt_path else None),
        'training_receipt': {'path': str((args.out / 'report.json').resolve()),
                                                      'sha256': sha256_file_hex_cached(args.out / 'report.json')},
        'exposure': {'train_rows': len(train), 'held_rows': len(held),
                     'supervised_tokens': sum(entry['supervised_tokens'] for entry in train.entries),
                     'graph_visits': sum(exposure_counts.values()),
                     'supervised_token_visits': sum(exposure_counts[entry['id']] *
                         entry['supervised_tokens'] for entry in train.entries),
                     'row_update_counts_sha256': exposure_counts_sha256,
                     'by_source_group': exposure_by_source_group,
                     'consumed_rows': min(target_rows, sum(exposure_counts.values())),
                     'consumed_updates': step, 'completed_epochs': epoch}})
    return 0


def _graph_loss(model, example, args):
    from ..data.native_graph import graph_loss
    return graph_loss(model, example, checkpoint_layers=args.checkpoint_layers,
                      chunk_tokens=args.chunk_tokens)


if __name__ == '__main__':
    raise SystemExit(main())
