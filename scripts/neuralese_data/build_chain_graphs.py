#!/usr/bin/env python3
"""Whole-trajectory native graphs from authenticated e2e chains (LFM3), with diversity caps.

Batch driver over the shared chain machinery (``native_chain_render`` -> ``native_chain_materialize`` ->
``native_chain_assembly``; the single-trajectory entry is ``build_lfm3_checked_chain.py``). It replaces the
one-2-node-graph-per-edge conversion for communication data (audit D7, LFM3_DATA_COVERAGE_AUDIT_20261031.md):

- **One graph per chain.** Every decision of a trajectory is a node; every producer span and consumer import sits at
  its recorded position, so a producer read by several consumers (fan-out) or a consumer reading several producers
  (fan-in) is one graph, trained in one pass, and the consumers' answer decisions stay in it.
- **Reader cap per payload** (``--reader-cap``, E2E_DATA.md: 64; ``--result-reader-cap`` for computed child results).
  Edges are kept for the first N reads (input order) of a given payload body; later reads of that body stay ordinary
  text context in their node.
- **Identical-target dedupe** (``--target-cap``). A decision whose exact target (canonical JSON) has already been
  supervised N times keeps no weight (decisions writing a computed child result are exempt: their context differs); it stays in the graph only when it produces a kept edge, and then only its
  exported span keeps weight. Decisions with neither weight nor a kept edge are dropped from the chain.
- A chain left without edges is written as an ordinary multi-node graph (``ordinary-<split>``), keeping the item
  variety of its decisions; a chain left without supervised decisions is an omission.

Custody: a train chain related to any current ProtectedWorlds identity (including reserved groups and quality-only
groups), acquired-connected groups, or a native decision hold is omitted; quality-only overlaps retain their reason.
The exact current protection digest is bound into every graph build. Test chains are written as held graphs. Context
weight is zero (own outputs only).
Prompt rendering: each tool-loop call is rebuilt with the current ts-host serving prompt and tool schemas
(``data/serving_prompt.py``; ``--serving-entry mix`` draws collector/runtime per record as the SFT re-render does) and
rendered ``full``; ``--fallback program-header``
re-renders a chain whose full graph exceeds ``--max-tokens`` raw tokens. Graphs are never truncated.

Inputs (one of):
  --work W --source S     converted e2e_mine shards (W/S/shard-NNNN.{records,edges,pieces}.jsonl)
  --corpus DIR            an assembled/extracted cohort (DIR/{records,edges,chains,pieces}.jsonl)

    build_chain_graphs.py (--work W --source S | --corpus DIR) --out OUT --corpus-id ID [--workers 4]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex  # noqa: E402

LFM3 = Path.home() / '.cache/huggingface/hub/models--LiquidAI--LFM3-1.7B/snapshots/40ebe0aa6662c9d40389c2f5d565b6a2c6aebacd'
LFM3_FINGERPRINT = '133504d493a77e3121f62cc8c83699cb222ee993c46959d604c4b683a2ffdaf3'
LFM3_ROLES = [125033, 125034, 125035, 125036]
PROTECTION = data_path('data_nvme', 'dgxops/acquired-connected-source-protection-20261017-d13-d14-d32-v3/protection-manifest.json')
CODE = ('scripts/neuralese_data/build_chain_graphs.py', 'scripts/neuralese_data/native_chain_sources.py',
        'scripts/neuralese_data/e2e_mine.py', 'scripts/neuralese_data/build_verified_teacher_trajectory_graphs.py',
        'training/neuralese/natlang_neuralese/common/custody.py',
        'scripts/neuralese_data/native_chain_render.py', 'scripts/neuralese_data/native_chain_materialize.py',
        'scripts/neuralese_data/native_chain_assembly.py', 'scripts/neuralese_data/build_native_semantic_pairs_e2e.py',
        'training/neuralese/natlang_neuralese/data/text_corpus.py',
        'training/neuralese/natlang_neuralese/data/native_graph_schema.py',
        'training/neuralese/natlang_neuralese/serve/chat.py',
        'training/neuralese/natlang_neuralese/data/serving_prompt.py', 'training/prompt-pieces/serving-surface.json')
ID_PATTERN = re.compile(rb'"id"\s*:\s*"([^"]+)"')
STATE = {}
STATE_INDEX_CACHE = None  # set from --index-cache: byte offsets of records.jsonl, bound to its size/mtime


def edge_digest(edge):
    return hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def render_plan_arguments(args):
    """Arguments that alter target/edge dedupe, excluding render-only continuation policy."""
    return {key: str(sorted(value) if isinstance(value, (set, frozenset)) else value)
            for key, value in sorted(vars(args).items())
            if key not in ('workers', 'batch_chains', 'continuation_max_tokens',
                           'continuation_old_code_fingerprint')}


def split_render_retry_rows(committed_bytes, attempted, recoverable_unknown_trajectories):
    """Separate only over-cap or source-proved unknown-JSON omissions for render continuation."""
    rows = [json.loads(line) for line in committed_bytes.splitlines() if line]
    def eligible(row):
        reason = row.get('reason', '')
        trajectory = row.get('trajectory')
        recoverable_reason = (reason in {
            'render-error: producer occurrence is not the exact authenticated payload value',
            'render-error: source writer requires a jointly renderable authenticated producer span'}
            and trajectory in recoverable_unknown_trajectories)
        return (reason == 'graph-over-token-cap' or recoverable_reason) and trajectory not in attempted
    retry = [row for row in rows if eligible(row)]
    retry_ids = [row.get('trajectory') for row in retry]
    if any(not isinstance(trajectory, str) or not trajectory for trajectory in retry_ids):
        raise ValueError('over-cap omission lacks an exact trajectory identity')
    retry_set = set(retry_ids)
    retained = [row for row in rows if not eligible(row)]
    reason_counts = {}
    for row in retry:
        reason = row['reason']
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return retained, retry_ids, reason_counts


def has_recoverable_unknown_result(task, plan):
    """Confirm a committed chain has a selected exact JSON-value producer span supported by the new renderer."""
    chain, locations, edges, _pieces_path, _plan = task
    if locations is None:
        return False
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import (
        source_occurrences, structured_unknown_result_span, writer_payload,
    )
    records = read_records(locations)
    for edge in edges:
        if edge_digest(edge) not in plan['edges']:
            continue
        writer, reader = records.get(edge['writer_record_id']), records.get(edge['reader_record_id'])
        if writer is None or reader is None:
            continue
        try:
            body = writer_payload(writer, edge['write_name'], edge['body_sha256'])
            occurrence = source_occurrences(writer, reader, edge['write_name'], body, edge['body_sha256'],
                writer_occurrence_id=edge.get('writer_occurrence_id'),
                reader_occurrence_id=edge.get('reader_occurrence_id'))
            structured_unknown_result_span(writer, occurrence, body)
        except (ValueError, KeyError, TypeError):
            continue
        return True
    return False


def decision_index(record_id):
    return int(record_id.rsplit(':', 1)[1])


# ---------------------------------------------------------------- units of work: (chain, record locations, edges)

def shard_units(work, source):
    """Chains of converted e2e_mine shards; records are located by (file, offset, length)."""
    for records_path in sorted((work / source).glob('shard-*.records.jsonl')):
        base = str(records_path).removesuffix('.records.jsonl')
        edges = [json.loads(line) for line in open(base + '.edges.jsonl') if line.strip()]
        edges_by = collections.defaultdict(list)
        for edge in edges:
            edges_by[edge['trajectory']].append({**edge, 'source': edge.get('source', source)})
        located = collections.defaultdict(list)
        splits = collections.defaultdict(set)
        with open(records_path, 'rb') as stream:
            offset = 0
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    located[row['source_ref']['trajectory_id']].append((row['id'], offset, len(line)))
                    splits[row['source_ref']['trajectory_id']].add(row['split'])
                offset += len(line)
        for trajectory, items in located.items():
            if not edges_by[trajectory]:
                continue
            items.sort(key=lambda item: decision_index(item[0]))
            split = splits[trajectory].pop() if len(splits[trajectory]) == 1 else None
            chain = {'source': source, 'trajectory': trajectory, 'split': split, 'records': [i[0] for i in items],
                     'edges': len(edges_by[trajectory])}
            yield chain, {i[0]: (str(records_path), i[1], i[2]) for i in items}, edges_by[trajectory], base + '.pieces.jsonl'


def corpus_units(corpus):
    """Chains of an assembled cohort, records located by a one-pass byte-offset index of records.jsonl."""
    records_path = corpus / 'records.jsonl'
    stat = records_path.stat()
    identity = {'path': str(records_path.resolve()), 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    cache = Path(STATE_INDEX_CACHE) if STATE_INDEX_CACHE else None
    index = None
    if cache and cache.exists():
        saved = json.loads(cache.read_text())
        if saved.get('identity') == identity:
            index = {rid: (str(records_path), offset, length) for rid, (offset, length) in saved['index'].items()}
    if index is None:
        index = {}
        with open(records_path, 'rb') as stream:
            offset = 0
            for line in stream:
                match = ID_PATTERN.search(line)
                if match:
                    index[match.group(1).decode()] = (str(records_path), offset, len(line))
                offset += len(line)
        if cache:
            cache.write_text(json.dumps({'identity': identity,
                                         'index': {rid: [o, n] for rid, (_, o, n) in index.items()}}))
    edges_by = collections.defaultdict(list)
    for line in open(corpus / 'edges.jsonl'):
        if line.strip():
            edge = json.loads(line)
            edges_by[(edge.get('source'), edge['trajectory'])].append(edge)
    for line in open(corpus / 'chains.jsonl'):
        if not line.strip():
            continue
        chain = json.loads(line)
        edges = edges_by[(chain.get('source'), chain['trajectory'])]
        missing = [rid for rid in chain['records'] if rid not in index]
        if missing:
            yield chain, None, edges, str(corpus / 'pieces.jsonl')
            continue
        yield ({key: chain[key] for key in ('source', 'trajectory', 'split', 'records', 'edges')},
               {rid: index[rid] for rid in chain['records']}, edges, str(corpus / 'pieces.jsonl'))


def read_records(locations):
    out = {}
    handles = {}
    try:
        for rid, (path, offset, length) in locations.items():
            stream = handles.get(path) or handles.setdefault(path, open(path, 'rb'))
            stream.seek(offset)
            row = json.loads(stream.read(length))
            if row.get('id') != rid:
                raise ValueError(f'record index points to a different record: {rid}')
            out[rid] = row
    finally:
        for stream in handles.values():
            stream.close()
    return out


# ---------------------------------------------------------------- workers

def init_worker(config):
    import sys
    sys.path[:0] = [str(ROOT / 'training' / 'neuralese'), str(ROOT)]
    from transformers import AutoTokenizer
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
    tokenizer = AutoTokenizer.from_pretrained(str(LFM3), local_files_only=True, trust_remote_code=False)
    roles = provision(tokenizer)
    if tokenizer_fingerprint(tokenizer) != LFM3_FINGERPRINT or list(roles) != LFM3_ROLES:
        raise ValueError('LFM3 tokenizer or role marker fingerprint differs')
    STATE.update(config, tokenizer=tokenizer, roles=roles, fingerprint=LFM3_FINGERPRINT, pieces={})


def current_protection_snapshot():
    """Load exact current identities and source-group classes for every graph build by default."""
    sys.path[:0] = [str(ROOT / 'training' / 'neuralese'), str(ROOT)]
    from natlang_neuralese.common.custody import REGISTRY, ProtectedWorlds
    from scripts.neuralese_data.e2e_mine import current_protected_source_groups
    registry_path = ROOT / REGISTRY
    registry_sha = sha256_file_hex(registry_path)
    protected = ProtectedWorlds.load()
    categories, group_bindings, category_registry_sha = current_protected_source_groups()
    if category_registry_sha != registry_sha:
        raise ValueError('ProtectedWorlds registry changed while loading graph-build custody')
    return ({'registry': REGISTRY, 'registry_sha256': registry_sha,
             'protected_sha256': protected.digest,
             'source_categories': [json.loads(item) for item in protected.sources],
             'source_group_bindings': group_bindings}, protected, categories)


def current_protection_omission(records, protected, quality_groups, nonquality_groups):
    """Return the applicable current custody exclusions for one complete chain, preserving quality-only labels."""
    from scripts.neuralese_data.e2e_mine import protected_trajectories, reviewed_trajectories
    matches = protected_trajectories(records.values(), protected)
    if not matches:
        return None
    trajectories = set(matches)
    kinds = set().union(*matches.values())
    quality = reviewed_trajectories(records.values(), quality_groups)
    nonquality = reviewed_trajectories(records.values(), nonquality_groups)
    reasons = set()
    if trajectories & set(quality):
        reasons.add('source_quality_quarantine_chain')
    if (trajectories & set(nonquality) or kinds - {'group', 'binding'} or not trajectories <= set(quality)):
        reasons.add('current_protected_worlds_custody')
    return {'reason': ('current_protected_worlds_custody' if 'current_protected_worlds_custody' in reasons
                       else 'source_quality_quarantine_chain'),
            'applicable_exclusions': sorted(reasons),
            'protected_identity_kinds': sorted(kinds),
            'protected_trajectories': sorted(trajectories)}


def target_shas(unit):
    """Phase 1: exact target identity of every decision of a chain (for the global dedupe)."""
    chain, locations, edges, _ = unit
    if locations is None:
        return {}
    return {rid: canonical_json_sha256_hex(row.get('target')) for rid, row in read_records(locations).items()}


def phase1_shas(pool, units, path, inputs_sha, batch):
    """Target identities per unit, appended to ``path`` in batches so a killed run resumes phase 1 too."""
    done = []
    if path.exists():
        lines = path.read_text().splitlines()
        if lines and json.loads(lines[0]).get('inputs_sha256') == inputs_sha:
            for line in lines[1:]:
                try:
                    done.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # a torn final line from a kill
    with open(path, 'w') as stream:
        stream.write(json.dumps({'inputs_sha256': inputs_sha, 'units': len(units)}) + '\n')
        for shas in done:
            stream.write(json.dumps(shas) + '\n')
        stream.flush()
        if done:
            print(f'phase 1 resuming after {len(done)}/{len(units)} chains', flush=True)
        for start in range(len(done), len(units), batch):
            for shas in pool.map(target_shas, units[start:start + batch], chunksize=8):
                done.append(shas); stream.write(json.dumps(shas) + '\n')
            stream.flush(); os.fsync(stream.fileno())
    return done


def pieces_for(path):
    if path not in STATE['pieces']:
        STATE['pieces'] = {path: [json.loads(line) for line in open(path) if line.strip()]}
    return STATE['pieces'][path]


def render(chain, records, edges, pieces, prompt):
    from scripts.neuralese_data.native_chain_render import render_chain
    from scripts.neuralese_data.native_chain_materialize import materialize
    from scripts.neuralese_data.native_chain_assembly import assemble
    rendered = render_chain(chain, records, edges, pieces, tokenizer=STATE['tokenizer'],
        tokenizer_fingerprint=STATE['fingerprint'], prompt_rendering=prompt,
        retain_prompt_tools=prompt == 'full' and STATE['serving_entry'] is None, serving_entry=STATE['serving_entry'])
    nodes = materialize(rendered, records, edges, tokenizer=STATE['tokenizer'], role_ids=STATE['roles'],
                        context_weight=0.0)
    return assemble(chain, records, edges, nodes, tokenizer=STATE['tokenizer'],
                    tokenizer_fingerprint=STATE['fingerprint'], role_ids=STATE['roles'],
                    source_binding={**STATE['binding'], 'prompt_rendering': prompt,
                                    'serving_entry': STATE['serving_entry']})


def convert_unit(task):
    """Phase 2: render one capped chain. Returns (kind, split, graph-or-omission)."""
    from natlang_neuralese.data.native_graph_schema import parse_graph, communication_classification
    chain, locations, edges, pieces_path, plan = task
    if locations is None:
        return 'omission', chain.get('split'), {'trajectory': chain['trajectory'], 'reason': 'records-missing'}
    if chain['split'] not in ('train', 'test'):
        return 'omission', None, {'trajectory': chain['trajectory'], 'reason': 'mixed-split-trajectory'}
    records = read_records(locations)
    if chain['split'] == 'train':
        current_omission = current_protection_omission(records, STATE['current_worlds'],
                                                       STATE['current_quality_groups'],
                                                       STATE['current_nonquality_groups'])
        if current_omission:
            return 'omission', 'train', {'trajectory': chain['trajectory'], **current_omission}
        if set(records) & STATE['holds']:
            return 'omission', 'train', {'trajectory': chain['trajectory'], 'reason': 'native-decision-hold'}
        source_groups = set().union(*(set(row.get('source_groups') or []) |
                                      set((((row.get('task') or {}).get('program_ir') or {}).get('source_groups') or []))
                                      for row in records.values()))
        acquired = STATE['acquired_protection']
        if source_groups & acquired['held']:
            return 'omission', 'train', {'trajectory': chain['trajectory'], 'reason': 'acquired-held-source-group'}
        if source_groups & acquired['quality']:
            return 'omission', 'train', {'trajectory': chain['trajectory'], 'reason': 'source-quality-quarantine-chain'}
        if source_groups & acquired['pending']:
            return 'omission', 'train', {'trajectory': chain['trajectory'], 'reason': 'acquired-pending-source-review'}
    families = {row.get('family') for row in records.values()}
    if chain['split'] == 'train' and families & STATE['excluded_families']:
        return 'omission', 'train', {'trajectory': chain['trajectory'],
                                     'reason': 'family-excluded: ' + ','.join(sorted(families & STATE['excluded_families']))}
    if chain['split'] == 'train' and any(set(row.get('source_ids') or []) & STATE['held_sources']
                                         for row in records.values()):
        return 'omission', 'train', {'trajectory': chain['trajectory'], 'reason': 'family-hold-source'}
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import source_occurrences, writer_payload
    order = {rid: index for index, rid in enumerate(chain['records'])}
    kept_edges, repairs = [], collections.Counter()
    for edge in edges:
        if edge_digest(edge) not in plan['edges']:
            continue
        writer, reader = records.get(edge['writer_record_id']), records.get(edge['reader_record_id'])
        if writer is None or reader is None or order[edge['writer_record_id']] >= order[edge['reader_record_id']]:
            repairs['edge-endpoint-order'] += 1; continue
        try:  # the edge's own authentication; a failing edge leaves the chain, its decisions stay
            body = writer_payload(writer, edge['write_name'], edge['body_sha256'])
            source_occurrences(writer, reader, edge['write_name'], body, edge['body_sha256'],
                               writer_occurrence_id=edge.get('writer_occurrence_id'),
                               reader_occurrence_id=edge.get('reader_occurrence_id'))
        except (ValueError, KeyError, TypeError) as error:
            repairs['edge-unauthenticated: ' + str(error)[:60]] += 1; continue
        kept_edges.append(edge)
    pieces = pieces_for(pieces_path)
    dropped = set()
    graph, prompt, reason = None, None, None
    for attempt in range(3):
        current = [edge for edge in kept_edges if edge['writer_record_id'] not in dropped
                   and edge['reader_record_id'] not in dropped]
        writers = {edge['writer_record_id'] for edge in current}
        keep = [rid for rid in chain['records'] if rid not in dropped and (rid in plan['supervised'] or rid in writers)]
        if not keep or (not any(rid in plan['supervised'] for rid in keep) and not current):
            return 'omission', chain['split'], {'trajectory': chain['trajectory'],
                                                'reason': 'no-supervised-decision-after-dedupe'}
        sub = {**chain, 'records': keep, 'edges': len(current)}
        sub_records = {rid: records[rid] for rid in keep}
        failed = None
        for prompt in STATE['prompts']:
            try:
                graph = render(sub, sub_records, current, pieces, prompt)
            except (ValueError, KeyError, TypeError, IndexError) as error:
                graph, reason = None, 'render-error: ' + str(error)[:240]
                # A decision the renderer cannot render leaves the chain with its edges; the rest is retried.
                failed = set(re.findall(r'"id": "([^"]+)"', str(error))) & set(keep)
                break
            if sum(len(node['token_ids']) for node in graph['nodes']) <= STATE['max_tokens']:
                reason = None; break
            reason, graph = 'graph-over-token-cap', None
        if graph is not None or not failed:
            break
        dropped |= failed
        repairs['decision-unrenderable'] += len(failed)
    if graph is None:
        return 'omission', chain['split'], {'trajectory': chain['trajectory'], 'reason': reason}
    kept_edges = current
    # Identical targets beyond the cap keep no weight; a kept producer keeps weight only on its exported span.
    names = {rid: f'n{index}' for index, rid in enumerate(keep)}
    for rid, node in zip(keep, graph['nodes']):
        if node['name'] != names[rid]:
            raise ValueError('assembled node order differs from the chain record order')
        if rid not in plan['supervised']:
            inside = set()
            for start, stop in node['producing']:
                inside.update(range(start, stop))
            node['target_weights'] = [w if i in inside else 0.0 for i, w in enumerate(node['target_weights'])]
    parsed = parse_graph(graph, tokenizer_fingerprint=STATE['fingerprint'])
    from natlang_neuralese.train.native_prefix_packing import pack_causal_prefixes
    packed = pack_causal_prefixes([parsed], allow_communication=True)
    communicating = bool(kept_edges) and communication_classification(parsed)['producer_consumer']
    first = records[keep[0]]
    graph['chain_conversion'] = {'schema': 'natlang.chain-graph-conversion/1', 'prompt_rendering': prompt,
        'context_weight': 0.0, 'serving_entry': STATE['serving_entry'], 'family': first.get('family') or chain.get('family'), 'source': chain['source'],
        'source_ids': first.get('source_ids'), 'decisions_in_source_chain': len(chain['records']),
        'decisions_kept': len(keep), 'edges_in_source_chain': len(edges), 'edges_kept': len(kept_edges),
        'decisions_without_weight': sum(rid not in plan['supervised'] for rid in keep),
        'reader_cap': STATE['reader_cap'], 'target_cap': STATE['target_cap'],
        'repairs': dict(repairs)}
    stats = {'nodes': len(graph['nodes']), 'tokens': sum(len(n['token_ids']) for n in graph['nodes']),
             'supervised_tokens': sum(1 for n in graph['nodes'] for w in n['target_weights'] if w > 0),
             'imports': sum(len(n['imports']) for n in graph['nodes']),
             'producer_spans': sum(len(n['producing']) for n in graph['nodes']),
             'prefix_shared_tokens': packed[0].token_count if packed is not None else parsed.plan.token_count}
    if stats['supervised_tokens'] == 0:
        return 'omission', chain['split'], {'trajectory': chain['trajectory'], 'reason': 'no-weight-after-dedupe'}
    stats.update({'repair:' + key: value for key, value in repairs.items()})
    return ('graph' if communicating else 'ordinary'), chain['split'], {'graph': graph, 'stats': stats,
            'family': graph['chain_conversion']['family'], 'prompt': prompt}


class Writer:
    """Append-only .jsonl.tmp streams that a resumed run truncates back to its last committed offsets."""
    def __init__(self, out, resume=None):
        self.out, self.files, self.counts = out, {}, collections.Counter((resume or {}).get('counts', {}))
        for name, offset in (resume or {}).get('offsets', {}).items():
            stream = open(self.out / f'{name}.jsonl.tmp', 'r+')
            stream.truncate(offset); stream.seek(offset)
            self.files[name] = stream

    def write(self, name, row):
        if name not in self.files:
            self.files[name] = open(self.out / f'{name}.jsonl.tmp', 'w')
        self.files[name].write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        self.counts[name] += 1

    def commit(self):
        """Flush every stream; returns the offsets/counts a resumed run restores."""
        for stream in self.files.values():
            stream.flush(); os.fsync(stream.fileno())
        return {'offsets': {name: stream.tell() for name, stream in self.files.items()}, 'counts': dict(self.counts)}

    def finish(self):
        bindings = {}
        for name, stream in self.files.items():
            stream.close()
            tmp, final = self.out / f'{name}.jsonl.tmp', self.out / f'{name}.jsonl.zst'
            if os.system(f'zstd -q -T4 -6 --rm -f "{tmp}" -o "{final}"'):
                raise RuntimeError('zstd failed for ' + name)
            bindings[name] = {'path': str(final), 'rows': self.counts[name], 'sha256': sha256_file_hex(final),
                              'bytes': final.stat().st_size}
        return bindings


def main():
    import sys
    sys.path[:0] = [str(ROOT / 'training' / 'neuralese'), str(ROOT)]
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
    from scripts.neuralese_data.decision_holds import load_decision_holds
    parser = argparse.ArgumentParser()
    parser.add_argument('--work', type=Path)
    parser.add_argument('--source')
    parser.add_argument('--corpus', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--corpus-id', required=True)
    parser.add_argument('--prompt', choices=('full', 'program-header'), default='full')
    parser.add_argument('--serving-entry', choices=('mix', 'collector', 'runtime', 'recorded'), default='mix',
                        help='rebuild each tool-loop call with the current ts-host serving prompt and tool schemas '
                             '(data/serving_prompt.py); "recorded" keeps the recorded prompt')
    parser.add_argument('--fallback', choices=('program-header', 'none'), default='program-header')
    parser.add_argument('--max-tokens', type=int, default=64000,
                        help='raw node-token cap; identical prompt prefixes are shared physically in training')
    parser.add_argument('--continuation-max-tokens', type=int,
                        help='resume a committed build with this higher render cap; retries prior over-cap chains '
                             'from the immutable all-chain dedupe plan before processing remaining chains')
    parser.add_argument('--continuation-old-code-fingerprint',
                        help='aggregate SHA256 for the exact renderer code used by the predecessor run')
    parser.add_argument('--reader-cap', type=int, default=64)
    parser.add_argument('--result-reader-cap', type=int, default=512,
                        help='reader cap per payload of computed child results (--computed-kinds)')
    parser.add_argument('--computed-kinds', type=lambda v: set(v.split(',')), default={'result'},
                        help='edge kinds whose payload a child computed from its own content; their writers are exempt '
                             'from the target dedupe')
    parser.add_argument('--target-cap', type=int, default=64)
    parser.add_argument('--limit-chains', type=int)
    parser.add_argument('--index-cache', type=Path, help='reusable records.jsonl offset index (corpus input)')
    parser.add_argument('--cap-state', type=Path,
                        help='JSON counters shared by successive runs, so the caps are global across tranches')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch-chains', type=int, default=512, help='chains per committed (resumable) batch')
    parser.add_argument('--protection', type=Path, default=PROTECTION)
    parser.add_argument('--exclude-families', type=lambda v: set(v.split(',')), default=set(),
                        help='record families excluded from training (e.g. graded by a since-repaired oracle)')
    args = parser.parse_args()
    if bool(args.corpus) == bool(args.work and args.source):
        raise SystemExit('give exactly one of --corpus or --work/--source')
    if (args.out / 'receipt.json').exists():
        print('already complete:', args.out); return
    args.out.mkdir(parents=True, exist_ok=True)
    protection = current_protection(args.protection)
    current_binding, current_worlds, current_group_categories = current_protection_snapshot()
    current_quality_groups = set(current_group_categories['quality-source-groups'])
    current_nonquality_groups = set().union(*(groups for source_id, groups in current_group_categories.items()
                                               if source_id != 'quality-source-groups'))
    holds, hold_bindings = load_decision_holds(sorted((ROOT / 'training/decision-holds').glob('*.jsonl')))
    code = {path: sha256_file_hex(ROOT / path) for path in CODE}
    code_fingerprint = hashlib.sha256(''.join(code[path] for path in sorted(code)).encode()).hexdigest()
    global STATE_INDEX_CACHE
    STATE_INDEX_CACHE = args.index_cache
    if args.corpus:
        inputs = {name: {'path': str(args.corpus / name), 'sha256': sha256_file_hex(args.corpus / name)}
                  for name in ('records.jsonl', 'edges.jsonl', 'chains.jsonl', 'pieces.jsonl')}
        if (args.corpus / 'receipt.json').is_file():
            source_receipt = args.corpus / 'receipt.json'
            receipt_sha = sha256_file_hex(source_receipt)
            source_proof = json.loads(source_receipt.read_text()).get('current_protected_worlds') or {}
            current_binding['source_receipt'] = {'path': str(source_receipt.resolve()),
                                                 'sha256': receipt_sha,
                                                 'upstream_union_status': ('matches-current' if
                                                     source_proof.get('registry_sha256') == current_binding['registry_sha256']
                                                     and source_proof.get('protected_sha256') == current_binding['protected_sha256']
                                                     else 'rechecked-record-by-record')}
            inputs['receipt.json'] = current_binding['source_receipt']
        units = list(corpus_units(args.corpus))
    else:
        inputs = {'work': str(args.work / args.source)}
        units = list(shard_units(args.work, args.source))
    if args.limit_chains:
        units = units[:args.limit_chains]
    binding = {'source_corpus_id': args.corpus_id, 'inputs': inputs, 'protection': protection['proof'],
               'decision_holds': hold_bindings, 'current_protected_worlds': current_binding}
    prompts = [args.prompt] + ([args.fallback] if args.fallback != 'none' and args.fallback != args.prompt else [])
    family_holds = json.loads((ROOT / 'training/source-reviews/family-holds.json').read_text())['holds']
    excluded = set(args.exclude_families) | {'curriculum_' + hold['family'] for hold in family_holds
                                             if hold['primarySourceId'] is None}
    held_sources = {hold['primarySourceId'] for hold in family_holds if hold['primarySourceId']}
    binding['family_holds'] = {'path': 'training/source-reviews/family-holds.json',
                               'sha256': sha256_file_hex(ROOT / 'training/source-reviews/family-holds.json'),
                               'excluded_families': sorted(excluded)}
    from natlang_neuralese.data.serving_prompt import load_surface, surface_identity
    serving_entry = None if args.serving_entry == 'recorded' else args.serving_entry
    binding['serving_surface'] = surface_identity(load_surface()) if serving_entry else None
    effective_max_tokens = args.continuation_max_tokens or args.max_tokens
    if effective_max_tokens < args.max_tokens:
        raise SystemExit('--continuation-max-tokens must not reduce the original graph cap')
    config = {'serving_entry': serving_entry, 'binding': binding, 'excluded_families': excluded,
              'held_sources': held_sources, 'prompts': prompts, 'max_tokens': effective_max_tokens, 'holds': holds,
              'acquired_protection': protection, 'current_worlds': current_worlds,
              'current_quality_groups': current_quality_groups,
              'current_nonquality_groups': current_nonquality_groups,
              'reader_cap': args.reader_cap, 'target_cap': args.target_cap}
    families, totals = collections.defaultdict(collections.Counter), collections.Counter()
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(config,)) as pool:
        # Phase 1: target identities; caps are decided in input order (deterministic, worker-independent).
        target_count, body_count = collections.Counter(), collections.Counter()
        # Shared caps count every OTHER output's kept targets/payloads; this output's own earlier
        # (e.g. preempted) attempt is replaced, so a rerun never double-counts against itself.
        run_key, runs = str(args.out.resolve()), {}
        if args.cap_state and args.cap_state.exists():
            runs = json.loads(args.cap_state.read_text())['runs']
            for key, state in runs.items():
                if key != run_key:
                    target_count.update(state['targets'])
                    body_count.update({(computed == 'computed', body): n for computed, body, n in state['bodies']})
        before_targets, before_bodies = collections.Counter(target_count), collections.Counter(body_count)
        cap_state_before = {'targets': sum(target_count.values()), 'bodies': sum(body_count.values())}
        plans = []
        for unit, shas in zip(units, phase1_shas(pool, units, args.out / 'target-shas.jsonl',
                                                  canonical_json_sha256_hex(binding['inputs']), args.batch_chains * 4)):
            # A child result is computed from content only its call saw: its decision is never deduped away.
            exempt = {edge['writer_record_id'] for edge in unit[2] if edge.get('kind') in args.computed_kinds}
            supervised = set()
            for rid in unit[0]['records']:
                if rid in exempt:
                    supervised.add(rid)
                elif rid in shas and target_count[shas[rid]] < args.target_cap:
                    target_count[shas[rid]] += 1; supervised.add(rid)
            kept = set()
            for edge in unit[2]:
                computed = edge.get('kind') in args.computed_kinds
                key = (computed, edge['body_sha256'])
                if edge['reader_record_id'] in supervised and body_count[key] < (
                        args.result_reader_cap if computed else args.reader_cap):
                    body_count[key] += 1; kept.add(edge_digest(edge))
            plans.append({'supervised': supervised, 'edges': kept})
        totals['distinct_targets'] = len(target_count)
        totals['distinct_payloads'] = len({e['body_sha256'] for u in units for e in u[2]})
        totals['distinct_payloads_kept'] = len({body for _, body in body_count})
        if args.cap_state:
            tmp = args.cap_state.with_suffix('.tmp')
            runs[run_key] = {'targets': target_count - before_targets, 'bodies': [
                ['computed' if computed else 'constant', body, n] for (computed, body), n in (body_count - before_bodies).items()]}
            tmp.write_text(json.dumps({'schema': 'natlang.chain-graph-cap-state/2', 'runs': runs}))
            os.replace(tmp, args.cap_state)
        # Phase 2 runs in bounded batches committed to progress.json, so a kill resumes after the last batch.
        # Phase 1 is deterministic (other outputs' cap-state only), so the plans identity must match to resume.
        plans_sha = canonical_json_sha256_hex({'binding': binding, 'args': render_plan_arguments(args),
                                               'plans': [{'supervised': sorted(p['supervised']), 'edges': sorted(p['edges'])} for p in plans]})
        progress_path = args.out / 'progress.json'
        progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
        if progress and progress['plans_sha256'] != plans_sha:
            print('progress belongs to different plans; restarting this output', flush=True)
            progress = None
        if not progress:
            for stale in args.out.glob('*.jsonl.tmp'):
                stale.unlink()
        # A cap change is a render-policy continuation, not a new dedupe plan.
        # Retry only already committed over-cap chains, preserving every other
        # committed output and all original plan decisions. The continuation
        # cursor is committed with output offsets after each bounded batch.
        if progress and args.continuation_max_tokens and not progress.get('render_continuation'):
            if (not args.continuation_old_code_fingerprint or not re.fullmatch(
                    r'[0-9a-f]{64}', args.continuation_old_code_fingerprint)):
                raise ValueError('render continuation requires the predecessor renderer code fingerprint')
            attempted = {trajectory for item in progress.get('render_continuation_history', [])
                         for trajectory in item.get('retried_trajectories', [])}
            omission_path = args.out / 'omissions.jsonl.tmp'
            old_offset = progress['omissions']['offsets'].get('omissions', 0)
            with omission_path.open('rb') as stream:
                committed = stream.read(old_offset)
            if len(committed) != old_offset:
                raise ValueError('committed omission stream is shorter than its progress offset')
            committed_rows = [json.loads(line) for line in committed.splitlines() if line]
            candidate_ids = {row.get('trajectory') for row in committed_rows
                if row.get('reason') in {
                    'render-error: producer occurrence is not the exact authenticated payload value',
                    'render-error: source writer requires a jointly renderable authenticated producer span'}
                and row.get('trajectory') not in attempted}
            unit_by_id = {unit[0]['trajectory']: (unit, plan) for unit, plan in zip(units, plans)}
            recoverable_unknown = set()
            for trajectory in candidate_ids:
                item = unit_by_id.get(trajectory)
                if item and has_recoverable_unknown_result((*item[0], item[1]), item[1]):
                    recoverable_unknown.add(trajectory)
            retained, retry_ids, retry_reasons = split_render_retry_rows(
                committed, attempted, recoverable_unknown)
            if retry_ids:
                tmp = omission_path.with_suffix('.retry.tmp')
                with tmp.open('wb') as stream:
                    for row in retained:
                        stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')).encode() + b'\n')
                    stream.flush(); os.fsync(stream.fileno())
                os.replace(tmp, omission_path)
                retained_bytes = omission_path.stat().st_size
                progress['omissions'] = {'offsets': {'omissions': retained_bytes},
                                         'counts': {'omissions': len(retained)}}
                for reason, count in retry_reasons.items():
                    key = 'omitted:' + reason.split(':', 1)[0]
                    progress['totals'][key] = max(0, progress['totals'].get(key, 0) - count)
                progress['render_continuation'] = {'schema': 'natlang.chain-render-continuation/1',
                    'old_max_tokens': args.max_tokens, 'new_max_tokens': effective_max_tokens,
                    'old_code_fingerprint': args.continuation_old_code_fingerprint,
                    'new_code_fingerprint': code_fingerprint, 'retry_reasons': retry_reasons,
                    'retried_trajectories': retry_ids, 'cursor': 0}
                progress_path.with_suffix('.tmp').write_text(json.dumps(progress))
                os.replace(progress_path.with_suffix('.tmp'), progress_path)
            elif progress.get('render_continuation_history'):
                progress['render_continuation'] = None
        if progress and progress.get('render_continuation'):
            continuation = progress['render_continuation']
            if (not args.continuation_max_tokens or effective_max_tokens != continuation['new_max_tokens']
                    or code_fingerprint != continuation.get('new_code_fingerprint')
                    or args.continuation_old_code_fingerprint != continuation.get('old_code_fingerprint')):
                raise ValueError('resumed render continuation differs from its committed code or cap transition')
        writer = Writer(args.out, progress and progress['graphs'])
        omissions = Writer(args.out, progress and progress['omissions'])
        done = progress['done'] if progress else 0
        if progress:
            totals.update({key: value for key, value in progress['totals'].items() if not key.startswith('distinct_')})
            for family, counts in progress['families'].items():
                families[family].update(counts)
            print(f'resuming after {done}/{len(units)} chains', flush=True)
        tasks = [(*unit, plan) for unit, plan in zip(units, plans)]
        continuation = progress.get('render_continuation') if progress else None
        if continuation:
            retry_ids = continuation['retried_trajectories']
            by_trajectory = {unit[0]['trajectory']: task for unit, task in zip(units, tasks)}
            missing = [trajectory for trajectory in retry_ids if trajectory not in by_trajectory]
            if missing:
                raise ValueError('render continuation names trajectories absent from immutable source plan')
            cursor = continuation['cursor']
            while cursor < len(retry_ids):
                batch_ids = retry_ids[cursor:cursor + args.batch_chains]
                batch = [by_trajectory[trajectory] for trajectory in batch_ids]
                for kind, split, value in pool.map(convert_unit, batch, chunksize=4):
                    if kind == 'omission':
                        totals['omitted:' + value['reason'].split(':')[0]] += 1
                        omissions.write('omissions', value)
                        continue
                    name = f'{"graphs" if kind == "graph" else "ordinary"}-{split}'
                    writer.write(name, value['graph'])
                    totals[name] += 1
                    totals[f'prompt:{value["prompt"]}'] += 1
                    for key, number in value['stats'].items():
                        totals[f'{name}:{key}'] += number
                    families[value['family']][name] += 1
                    families[value['family']][f'{name}:imports'] += value['stats']['imports']
                cursor += len(batch)
                continuation['cursor'] = cursor
                state = {'plans_sha256': plans_sha, 'done': done, 'graphs': writer.commit(),
                         'omissions': omissions.commit(), 'totals': dict(totals),
                         'families': {k: dict(v) for k, v in families.items()},
                         'render_continuation': continuation,
                         'render_continuation_history': progress.get('render_continuation_history', [])}
                progress_path.with_suffix('.tmp').write_text(json.dumps(state))
                os.replace(progress_path.with_suffix('.tmp'), progress_path)
                print(f'committed over-cap retry {cursor}/{len(retry_ids)}', flush=True)
            history = list(progress.get('render_continuation_history', []))
            history.append({'schema': continuation['schema'], 'old_max_tokens': continuation['old_max_tokens'],
                'new_max_tokens': continuation['new_max_tokens'],
                'old_code_fingerprint': continuation['old_code_fingerprint'],
                'new_code_fingerprint': continuation['new_code_fingerprint'],
                'retry_reasons': continuation.get('retry_reasons', {}),
                'retried_trajectories': continuation['retried_trajectories'],
                'retried_count': len(continuation['retried_trajectories'])})
            progress = {'plans_sha256': plans_sha, 'done': done, 'graphs': writer.commit(),
                        'omissions': omissions.commit(), 'totals': dict(totals),
                        'families': {k: dict(v) for k, v in families.items()},
                        'render_continuation_history': history}
            progress_path.with_suffix('.tmp').write_text(json.dumps(progress))
            os.replace(progress_path.with_suffix('.tmp'), progress_path)
        while done < len(tasks):
            batch = tasks[done:done + args.batch_chains]
            for kind, split, value in pool.map(convert_unit, batch, chunksize=4):
                if kind == 'omission':
                    totals['omitted:' + value['reason'].split(':')[0]] += 1
                    omissions.write('omissions', value)
                    continue
                name = f'{"graphs" if kind == "graph" else "ordinary"}-{split}'
                writer.write(name, value['graph'])
                totals[name] += 1
                totals[f'prompt:{value["prompt"]}'] += 1
                for key, number in value['stats'].items():
                    totals[f'{name}:{key}'] += number
                families[value['family']][name] += 1
                families[value['family']][f'{name}:imports'] += value['stats']['imports']
            done += len(batch)
            state = {'plans_sha256': plans_sha, 'done': done, 'graphs': writer.commit(), 'omissions': omissions.commit(),
                     'totals': dict(totals), 'families': {k: dict(v) for k, v in families.items()},
                     'render_continuation_history': (progress or {}).get('render_continuation_history', [])}
            progress_path.with_suffix('.tmp').write_text(json.dumps(state))
            os.replace(progress_path.with_suffix('.tmp'), progress_path)
            print(f'committed {done}/{len(tasks)} chains', flush=True)
    files = writer.finish()
    omission_files = omissions.finish()
    if any(sha256_file_hex(ROOT / path) != digest for path, digest in code.items()):
        raise ValueError('chain renderer code changed during this conversion')
    receipt = {'schema': 'natlang.chain-graphs/1', 'corpus_id': args.corpus_id, 'source_binding': binding,
               'prompt_rendering': prompts, 'context_weight': 0.0, 'max_tokens': effective_max_tokens,
               'render_continuation_history': (progress or {}).get('render_continuation_history', []),
               'reader_cap': args.reader_cap, 'result_reader_cap': args.result_reader_cap,
               'computed_kinds': sorted(args.computed_kinds), 'target_cap': args.target_cap, 'chains': len(units),
               'cap_state': {'path': str(args.cap_state) if args.cap_state else None, 'before': cap_state_before},
               'tokenizer': {'path': str(LFM3), 'fingerprint': LFM3_FINGERPRINT, 'role_ids': LFM3_ROLES},
               'code': code, 'totals': dict(totals), 'families': {k: dict(v) for k, v in sorted(families.items())},
               'files': files, 'omissions': omission_files}
    end_binding, _, _ = current_protection_snapshot()
    if (end_binding['protected_sha256'] != current_binding['protected_sha256']
            or end_binding['registry_sha256'] != current_binding['registry_sha256']
            or end_binding['source_group_bindings'] != current_binding['source_group_bindings']):
        raise ValueError('current ProtectedWorlds union or source-group files changed during graph conversion')
    if current_binding.get('source_receipt') and sha256_file_hex(
            Path(current_binding['source_receipt']['path'])) != current_binding['source_receipt']['sha256']:
        raise ValueError('assembled source receipt changed during graph conversion')
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    (args.out / 'progress.json').unlink(missing_ok=True)
    (args.out / 'target-shas.jsonl').unlink(missing_ok=True)
    print(json.dumps({'totals': dict(totals)}))


if __name__ == '__main__':
    main()
