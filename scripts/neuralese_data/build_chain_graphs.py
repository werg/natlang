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

Custody: a train chain touching current held/quality/pending groups (acquired-connected protection projection) or a
native decision hold is omitted; test chains are written as held graphs. Context weight is zero (own outputs only).
Prompt rendering: the recorded full serving prompt with its tool schemas (``full``); ``--fallback program-header``
re-renders a chain whose full graph exceeds ``--max-tokens``. Graphs are never truncated.

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
        'scripts/neuralese_data/native_chain_render.py', 'scripts/neuralese_data/native_chain_materialize.py',
        'scripts/neuralese_data/native_chain_assembly.py', 'scripts/neuralese_data/build_native_semantic_pairs_e2e.py',
        'training/neuralese/natlang_neuralese/data/text_corpus.py',
        'training/neuralese/natlang_neuralese/data/native_graph_schema.py',
        'training/neuralese/natlang_neuralese/serve/chat.py')
ID_PATTERN = re.compile(rb'"id"\s*:\s*"([^"]+)"')
STATE = {}
STATE_INDEX_CACHE = None  # set from --index-cache: byte offsets of records.jsonl, bound to its size/mtime


def edge_digest(edge):
    return hashlib.sha256(json.dumps(edge, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


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


def target_shas(unit):
    """Phase 1: exact target identity of every decision of a chain (for the global dedupe)."""
    chain, locations, edges, _ = unit
    if locations is None:
        return {}
    return {rid: canonical_json_sha256_hex(row.get('target')) for rid, row in read_records(locations).items()}


def pieces_for(path):
    if path not in STATE['pieces']:
        STATE['pieces'] = {path: [json.loads(line) for line in open(path) if line.strip()]}
    return STATE['pieces'][path]


def render(chain, records, edges, pieces, prompt):
    from scripts.neuralese_data.native_chain_render import render_chain
    from scripts.neuralese_data.native_chain_materialize import materialize
    from scripts.neuralese_data.native_chain_assembly import assemble
    rendered = render_chain(chain, records, edges, pieces, tokenizer=STATE['tokenizer'],
        tokenizer_fingerprint=STATE['fingerprint'], prompt_rendering=prompt, retain_prompt_tools=prompt == 'full')
    nodes = materialize(rendered, records, edges, tokenizer=STATE['tokenizer'], role_ids=STATE['roles'],
                        context_weight=0.0)
    return assemble(chain, records, edges, nodes, tokenizer=STATE['tokenizer'],
                    tokenizer_fingerprint=STATE['fingerprint'], role_ids=STATE['roles'],
                    source_binding={**STATE['binding'], 'prompt_rendering': prompt})


def convert_unit(task):
    """Phase 2: render one capped chain. Returns (kind, split, graph-or-omission)."""
    from natlang_neuralese.data.native_graph_schema import parse_graph, communication_classification
    chain, locations, edges, pieces_path, plan = task
    if locations is None:
        return 'omission', chain.get('split'), {'trajectory': chain['trajectory'], 'reason': 'records-missing'}
    if chain['split'] not in ('train', 'test'):
        return 'omission', None, {'trajectory': chain['trajectory'], 'reason': 'mixed-split-trajectory'}
    records = read_records(locations)
    if chain['split'] == 'train' and (set(records) & STATE['holds'] or any(
            set(row.get('source_groups') or []) & STATE['blocked'] for row in records.values())):
        return 'omission', 'train', {'trajectory': chain['trajectory'], 'reason': 'protected-or-held'}
    families = {row.get('family') for row in records.values()}
    if chain['split'] == 'train' and families & STATE['excluded_families']:
        return 'omission', 'train', {'trajectory': chain['trajectory'],
                                     'reason': 'family-excluded: ' + ','.join(sorted(families & STATE['excluded_families']))}
    if chain['split'] == 'train' and any(set(row.get('source_ids') or []) & STATE['held_sources']
                                         for row in records.values()):
        return 'omission', 'train', {'trajectory': chain['trajectory'], 'reason': 'family-hold-source'}
    kept_edges = [edge for edge in edges if edge_digest(edge) in plan['edges']]
    writers = {edge['writer_record_id'] for edge in kept_edges}
    keep = [rid for rid in chain['records'] if rid in plan['supervised'] or rid in writers]
    if not any(rid in plan['supervised'] for rid in keep) and not kept_edges:
        return 'omission', chain['split'], {'trajectory': chain['trajectory'], 'reason': 'no-supervised-decision-after-dedupe'}
    sub = {**chain, 'records': keep, 'edges': len(kept_edges)}
    sub_records = {rid: records[rid] for rid in keep}
    pieces = pieces_for(pieces_path)
    graph, prompt, reason = None, None, None
    for prompt in STATE['prompts']:
        try:
            graph = render(sub, sub_records, kept_edges, pieces, prompt)
        except (ValueError, KeyError, TypeError, IndexError) as error:
            reason = 'render-error: ' + str(error)[:240]; graph = None; break
        if sum(len(node['token_ids']) for node in graph['nodes']) <= STATE['max_tokens']:
            reason = None; break
        reason, graph = 'graph-over-token-cap', None
    if graph is None:
        return 'omission', chain['split'], {'trajectory': chain['trajectory'], 'reason': reason}
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
    communicating = bool(kept_edges) and communication_classification(parsed)['producer_consumer']
    first = records[keep[0]]
    graph['chain_conversion'] = {'schema': 'natlang.chain-graph-conversion/1', 'prompt_rendering': prompt,
        'context_weight': 0.0, 'family': first.get('family') or chain.get('family'), 'source': chain['source'],
        'source_ids': first.get('source_ids'), 'decisions_in_source_chain': len(chain['records']),
        'decisions_kept': len(keep), 'edges_in_source_chain': len(edges), 'edges_kept': len(kept_edges),
        'decisions_without_weight': sum(rid not in plan['supervised'] for rid in keep),
        'reader_cap': STATE['reader_cap'], 'target_cap': STATE['target_cap']}
    stats = {'nodes': len(graph['nodes']), 'tokens': sum(len(n['token_ids']) for n in graph['nodes']),
             'supervised_tokens': sum(1 for n in graph['nodes'] for w in n['target_weights'] if w > 0),
             'imports': sum(len(n['imports']) for n in graph['nodes']),
             'producer_spans': sum(len(n['producing']) for n in graph['nodes'])}
    if stats['supervised_tokens'] == 0:
        return 'omission', chain['split'], {'trajectory': chain['trajectory'], 'reason': 'no-weight-after-dedupe'}
    return ('graph' if communicating else 'ordinary'), chain['split'], {'graph': graph, 'stats': stats,
            'family': graph['chain_conversion']['family'], 'prompt': prompt}


class Writer:
    def __init__(self, out):
        self.out, self.files, self.counts = out, {}, collections.Counter()

    def write(self, name, row):
        if name not in self.files:
            self.files[name] = open(self.out / f'{name}.jsonl.tmp', 'w')
        self.files[name].write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        self.counts[name] += 1

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
    parser.add_argument('--fallback', choices=('program-header', 'none'), default='program-header')
    parser.add_argument('--max-tokens', type=int, default=40000)
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
    holds, hold_bindings = load_decision_holds(sorted((ROOT / 'training/decision-holds').glob('*.jsonl')))
    code = {path: sha256_file_hex(ROOT / path) for path in CODE}
    global STATE_INDEX_CACHE
    STATE_INDEX_CACHE = args.index_cache
    if args.corpus:
        inputs = {name: {'path': str(args.corpus / name), 'sha256': sha256_file_hex(args.corpus / name)}
                  for name in ('records.jsonl', 'edges.jsonl', 'chains.jsonl', 'pieces.jsonl')}
        units = list(corpus_units(args.corpus))
    else:
        inputs = {'work': str(args.work / args.source)}
        units = list(shard_units(args.work, args.source))
    if args.limit_chains:
        units = units[:args.limit_chains]
    binding = {'source_corpus_id': args.corpus_id, 'inputs': inputs, 'protection': protection['proof'],
               'decision_holds': hold_bindings}
    prompts = [args.prompt] + ([args.fallback] if args.fallback != 'none' and args.fallback != args.prompt else [])
    family_holds = json.loads((ROOT / 'training/source-reviews/family-holds.json').read_text())['holds']
    excluded = set(args.exclude_families) | {'curriculum_' + hold['family'] for hold in family_holds
                                             if hold['primarySourceId'] is None}
    held_sources = {hold['primarySourceId'] for hold in family_holds if hold['primarySourceId']}
    binding['family_holds'] = {'path': 'training/source-reviews/family-holds.json',
                               'sha256': sha256_file_hex(ROOT / 'training/source-reviews/family-holds.json'),
                               'excluded_families': sorted(excluded)}
    config = {'binding': binding, 'excluded_families': excluded, 'held_sources': held_sources, 'prompts': prompts, 'max_tokens': args.max_tokens, 'holds': holds,
              'blocked': protection['held'] | protection['quality'] | protection['pending'],
              'reader_cap': args.reader_cap, 'target_cap': args.target_cap}
    writer, omissions, totals = Writer(args.out), Writer(args.out), collections.Counter()
    families = collections.defaultdict(collections.Counter)
    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(config,)) as pool:
        # Phase 1: target identities; caps are decided in input order (deterministic, worker-independent).
        target_count, body_count = collections.Counter(), collections.Counter()
        if args.cap_state and args.cap_state.exists():
            state = json.loads(args.cap_state.read_text())
            target_count.update(state['targets'])
            body_count.update({(computed == 'computed', body): n for computed, body, n in state['bodies']})
        cap_state_before = {'targets': sum(target_count.values()), 'bodies': sum(body_count.values())}
        plans = []
        for unit, shas in zip(units, pool.map(target_shas, units, chunksize=8)):
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
            tmp.write_text(json.dumps({'targets': target_count, 'bodies': [
                ['computed' if computed else 'constant', body, n] for (computed, body), n in body_count.items()]}))
            os.replace(tmp, args.cap_state)
        tasks = [(*unit, plan) for unit, plan in zip(units, plans)]
        for kind, split, value in pool.map(convert_unit, tasks, chunksize=4):
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
    files = writer.finish()
    omission_files = omissions.finish()
    if any(sha256_file_hex(ROOT / path) != digest for path, digest in code.items()):
        raise ValueError('chain renderer code changed during this conversion')
    receipt = {'schema': 'natlang.chain-graphs/1', 'corpus_id': args.corpus_id, 'source_binding': binding,
               'prompt_rendering': prompts, 'context_weight': 0.0, 'max_tokens': args.max_tokens,
               'reader_cap': args.reader_cap, 'result_reader_cap': args.result_reader_cap,
               'computed_kinds': sorted(args.computed_kinds), 'target_cap': args.target_cap, 'chains': len(units),
               'cap_state': {'path': str(args.cap_state) if args.cap_state else None, 'before': cap_state_before},
               'tokenizer': {'path': str(LFM3), 'fingerprint': LFM3_FINGERPRINT, 'role_ids': LFM3_ROLES},
               'code': code, 'totals': dict(totals), 'families': {k: dict(v) for k, v in sorted(families.items())},
               'files': files, 'omissions': omission_files}
    (args.out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({'totals': dict(totals)}))


if __name__ == '__main__':
    main()
