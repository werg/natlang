#!/usr/bin/env python3
"""Lower the joint-SFT native graph sources to plain per-call token rows for `exp0_sft.py train --rows` (SFT v2,
plans/neuralese/APPROACH_REVIEW_20261111.md sections 2.1/2.2/2.7).

Every text node of a kept row becomes one sequence: its exact token ids, loss on the row's own supervised tokens
(`target_weights > 0`: the model's actions and answers; prompts, tool output and feedback carry weight 0 in the source).
Weights are the lineage's declared own-output objective (`own_output_supervision.graph_weights` of the CE policy:
assistant replies/actions, reasoning at its declared share, template-forced and context tokens 0), without family scales:
the mix is set by selection. A node whose ids extend the previous node's ids of the same row is merged into that sequence (identical causal loss,
fewer forward tokens). Per-call flattening, not whole-graph: nodes that import or produce Neuralese state, and decision
readout rows, are skipped and counted. Families come from the source's logical-family map; `--drop-family` removes whole
families (the scripted templates). Read-only on the sources.

    lower_sft_sources.py --sources train-sources.json --families families.json --out DIR \
        --drop-family port-record-wrapper --drop-family native-communication-template
    lower_sft_sources.py sample --out DIR --tokenizer TOK [--per-family 2]
    lower_sft_sources.py select --data DIR --out chains.pkl --budget-tokens N --tokenizer TOK \
        --held-cases held-v2/cases.jsonl [--share FAMILY=X ...]

`select` takes a forward-token budget across families (water-filled target shares: a family smaller than its share is taken
whole and the rest goes to the others; rows sampled at random within a family, each at most once), drops every row tied to a
held task (its id, source ids, source/split groups, or 3+ distinctive 10-word shingles of one held task's inputs or world
files in the row's decoded text), and writes `exp0_sft.py --chains` sequences (segments = runs of positive weight).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import multiprocessing
import pickle
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))


def leaves(manifest_path):
    """Leaf graph files of a (nested) native-graph-sources manifest, in order."""
    manifest = json.loads(Path(manifest_path).read_text())
    if manifest.get('schema') != 'natlang.native-graph-sources/1':
        raise ValueError(f'{manifest_path}: not a native graph source manifest')
    out = []
    for binding in manifest['sources']:
        path = Path(binding['path'])
        if not path.is_absolute():
            path = Path(manifest_path).parent / path
        out += leaves(path) if path.suffix == '.json' else [str(path)]
    return out


def _flatten(row, objective, stats):
    """Sequences of one graph row: [(ids int32, weights float32, node names)]."""
    if row.get('decisions'):
        stats['skip_rows_decision'] += 1
        return []
    from natlang_neuralese.train.own_output_supervision import graph_weights
    try:
        flat, _ = graph_weights(row, objective)
    except ValueError as error:
        stats['skip_rows_objective_error'] += 1
        stats[f'objective_error:{str(error)[:80]}'] += 1
        return []
    flat = np.asarray(flat, dtype=np.float32)
    sequences, current, offset = [], None, 0
    for node in row.get('nodes') or []:
        node_weights = flat[offset:offset + len(node['token_ids'])]
        offset += len(node['token_ids'])
        if node.get('imports') or node.get('producing') or node.get('logical_producing'):
            stats['skip_nodes_neuralese'] += 1
            current = None
            continue
        ids = np.asarray(node['token_ids'], dtype=np.int32)
        weights = node_weights
        if not (weights > 0).any():
            stats['skip_nodes_unsupervised'] += 1
            continue
        if current is not None and len(ids) >= len(current[0]) and np.array_equal(ids[:len(current[0])], current[0]):
            merged = weights.copy()
            merged[:len(current[1])] = np.maximum(merged[:len(current[1])], current[1])
            current[0], current[1] = ids, merged
            current[2].append(node['name'])
            stats['merged_nodes'] += 1
            continue
        current = [ids, weights, [node['name']]]
        sequences.append(current)
    stats['text_nodes'] += sum(len(s[2]) for s in sequences)
    return [tuple(s) for s in sequences]


def lower_leaf(job):
    path, families, drop, out_dir, index, args_split, policy_path = job
    from natlang_neuralese.train.own_output_supervision import resolve
    objective = resolve(json.loads(Path(policy_path).read_text())['objective'])
    from natlang_neuralese.common.jsonio import open_jsonl_text
    stats = collections.Counter()
    weight_values = collections.Counter()
    kept = 0
    out = Path(out_dir) / f'leaf-{index:04d}.pkl'
    partial = out.with_suffix('.pkl.partial')
    with open_jsonl_text(Path(path)) as handle, partial.open('wb') as sink:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            stats['rows'] += 1
            family = families.get(row['id'], 'unmapped')
            stats[f'rows:{family}'] += 1
            if family in drop or row.get('split') != args_split:
                stats['skip_rows_dropped_family' if family in drop else 'skip_rows_not_train'] += 1
                continue
            for ids, weights, names in _flatten(row, objective, stats):
                positive = weights[weights > 0]
                weight_values.update(np.round(positive, 4).tolist())
                pickle.dump({'id': row['id'], 'family': family, 'nodes': names, 'groups': row.get('source_groups') or [],
                             'ids': ids, 'w': weights}, sink, protocol=4)       # a stream of pickles: bounded memory
                kept += 1
    result = {'path': path, 'stats': dict(stats), 'weights': dict(weight_values.most_common(12)), 'kept': kept}
    out.with_suffix('.json').write_text(json.dumps(result))
    partial.rename(out)                       # a leaf is complete only once renamed: `build` resumes after a stop
    return result


def build(args):
    out = Path(args.out)
    (out / 'leaves').mkdir(parents=True, exist_ok=True)
    families = {}
    if args.families:
        families = {row['id']: row['family'] for row in json.loads(Path(args.families).read_text())['rows']}
    paths = leaves(args.sources)
    jobs = [(p, families, set(args.drop_family), str(out / 'leaves'), i, args.split, args.ce_policy) for i, p in enumerate(paths)]
    jobs = [job for job in jobs if not (out / 'leaves' / f'leaf-{job[4]:04d}.pkl').exists()]
    jobs.sort(key=lambda job: -Path(job[0]).stat().st_size)          # the large leaves start first
    with multiprocessing.Pool(args.workers) as pool:
        for result in pool.imap_unordered(lower_leaf, jobs):
            print(json.dumps({'leaf': result['path'], 'kept_sequences': result['kept'], 'rows': result['stats'].get('rows', 0)}), flush=True)
    total, weights = collections.Counter(), collections.Counter()
    for done in sorted((out / 'leaves').glob('leaf-*.json')):
        result = json.loads(done.read_text())
        total.update(result['stats'])
        weights.update(result['weights'])
    summary = {'sources': str(Path(args.sources).resolve()), 'families_map': args.families and str(Path(args.families).resolve()), 'split': args.split, 'ce_policy': str(Path(args.ce_policy).resolve()),
               'dropped_families': sorted(args.drop_family), 'leaves': len(paths), 'stats': dict(total),
               'positive_weight_values': dict(weights.most_common(12))}
    (out / 'lowering.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def iter_rows(out):
    for path in sorted((Path(out) / 'leaves').glob('leaf-*.pkl')):
        with path.open('rb') as handle:
            while True:
                try:
                    yield pickle.load(handle)
                except EOFError:
                    break


def sample(args):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    seen = collections.Counter()
    for row in iter_rows(args.out):
        if seen[row['family']] >= args.per_family:
            continue
        seen[row['family']] += 1
        supervised = row['w'] > 0
        text = tokenizer.decode(row['ids'].tolist())
        target = tokenizer.decode(row['ids'][supervised].tolist())
        print(f"===== {row['family']} {row['id']} tokens={len(row['ids'])} supervised={int(supervised.sum())}")
        print(text[:args.chars])
        print('..... SUPERVISED:')
        print(target[:args.chars // 2])


MAX_LEN = 32000
DEFAULT_SHARES = {'natlang': .16, 'natlang-inline': .16, 'natlang-chains': .14, 'native-communication': .05,
                  'native-communication-rest27': .10, 'agent-sessions': .10, 'coding-sessions': .03, 'coding-swe': .07,
                  'acquired-d13': .07, 'web-research': .08}
SHINGLE = 10


def _words(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def _shingles(text):
    words = _words(text)
    return {hashlib.blake2b(' '.join(words[i:i + SHINGLE]).encode(), digest_size=8).digest()
            for i in range(len(words) - SHINGLE + 1)}


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def held_probes(cases_paths):
    """(identity strings, shingle -> held task) of the held tasks: ids, source ids/groups, split groups; distinctive
    shingles of their inputs and world files (a shingle in two or more held tasks is boilerplate, not identity)."""
    identities, owners = set(), collections.defaultdict(set)
    for path in cases_paths:
        for line in Path(path).read_text().splitlines():
            case = json.loads(line)
            identities.update([case['id'], *case.get('source_ids', []), *case.get('source_groups', []),
                               (case.get('curriculum') or {}).get('split_group') or case['id']])
            semantics = case.get('semantics') or {}
            for text in _strings([semantics.get('inputs'), semantics.get('folder_files'), semantics.get('services')]):
                for shingle in _shingles(text):
                    owners[shingle].add(case['id'])
    distinctive = {shingle: next(iter(tasks)) for shingle, tasks in owners.items() if len(tasks) == 1}
    return identities, distinctive


def _text_hits(job):
    tokenizer_path, items, distinctive = job
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    flagged = []
    for key, ids in items:
        hits = collections.Counter(distinctive[s] for s in _shingles(tokenizer.decode(ids.tolist())) if s in distinctive)
        task, count = hits.most_common(1)[0] if hits else (None, 0)
        if count >= 3:
            flagged.append((key, task, count))
    return flagged


def to_chain(row):
    """exp0 chain: segments are the runs of positive weight (the first token is never a target)."""
    positive = np.concatenate([[False], row['w'] > 0, [False]])
    edges = np.flatnonzero(positive[1:] != positive[:-1])
    segs = [(int(a), int(b - a), {'sample': row['ids'][a:b], 'weight': row['w'][a:b]}) for a, b in zip(edges[::2], edges[1::2])]
    return {'ids': row['ids'], 'segs': segs, 'task': row['id'], 'round': row['family']}


def select(args):
    rng = np.random.default_rng(args.seed)
    shares = dict(DEFAULT_SHARES)
    for item in args.share:
        name, value = item.split('=')
        shares[name] = float(value)
    identities, distinctive = held_probes(args.held_cases)
    for path in args.held_ids:
        identities.update(json.loads(Path(path).read_text()))
    index, stats = [], collections.Counter()
    for ordinal, row in enumerate(iter_rows(args.data)):
        length = len(row['ids'])
        if length > MAX_LEN:
            stats['skip_over_max_len'] += 1
            continue
        if identities & {row['id'], *row['groups']}:
            stats['skip_held_identity'] += 1
            continue
        index.append((ordinal, row['family'], length, float(row['w'].sum()), int((row['w'] > 0).sum())))
    families = sorted({family for _, family, *_ in index})
    totals = {f: sum(length for _, family, length, *_ in index if family == f) for f in families}
    # water-fill: families smaller than their share are taken whole
    allocation, open_families, budget = {}, set(families), float(args.budget_tokens)
    while open_families:
        weight = sum(shares.get(f, 0.) for f in open_families)
        if weight <= 0:
            break
        full = {f for f in open_families if totals[f] <= budget * shares.get(f, 0.) / weight}
        if not full:
            for f in open_families:
                allocation[f] = budget * shares.get(f, 0.) / weight
            break
        for f in full:
            allocation[f] = totals[f]
            budget -= totals[f]
        open_families -= full
    chosen = set()
    for f in families:
        rows = [item for item in index if item[1] == f]
        order = rng.permutation(len(rows))
        used = 0
        for i in order:
            if used >= allocation.get(f, 0):
                break
            chosen.add(rows[i][0])
            used += rows[i][2]
    picked = [row for ordinal, row in enumerate(iter_rows(args.data)) if ordinal in chosen]
    # text-level held check on the chosen rows
    flagged = []
    if args.tokenizer:
        items = [((row['id'], i), row['ids']) for i, row in enumerate(picked)]
        jobs = [(args.tokenizer, items[k::args.workers], distinctive) for k in range(args.workers)]
        with multiprocessing.Pool(args.workers) as pool:
            for part in pool.imap_unordered(_text_hits, jobs):
                flagged += part
    drop = {key[1] for key, _, _ in flagged}
    picked = [row for i, row in enumerate(picked) if i not in drop]
    rng.shuffle(picked)
    chains = [to_chain(row) for row in picked]
    summary = collections.defaultdict(lambda: collections.Counter())
    for row in picked:
        item = summary[row['family']]
        item['rows'] += 1
        item['forward_tokens'] += len(row['ids'])
        item['target_tokens'] += int((row['w'] > 0).sum())
        item['target_mass'] += float(row['w'].sum())
    meta = {'data': str(Path(args.data).resolve()), 'budget_tokens': args.budget_tokens, 'shares': shares, 'seed': args.seed,
            'held_cases': [str(Path(p).resolve()) for p in args.held_cases], 'held_ids': args.held_ids, 'held_identities': len(identities),
            'held_distinctive_shingles': len(distinctive), 'stats': dict(stats),
            'available_forward_tokens': totals, 'allocation': {f: int(v) for f, v in allocation.items()},
            'held_text_flagged': [{'row': key[0], 'held_task': task, 'shingles': count} for key, task, count in flagged],
            'families': {f: {k: round(v, 1) for k, v in item.items()} for f, item in sorted(summary.items())},
            'total': {'rows': len(picked), 'forward_tokens': int(sum(len(r['ids']) for r in picked)),
                      'target_tokens': int(sum(int((r['w'] > 0).sum()) for r in picked))}}
    with open(args.out, 'wb') as handle:
        pickle.dump({'chains': chains, 'meta': meta}, handle, protocol=4)
    Path(args.out).with_suffix('.json').write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build')
    p.add_argument('--sources', required=True)
    p.add_argument('--families', help='logical-family map (rows without one are family "unmapped")')
    p.add_argument('--split', default='train')
    p.add_argument('--ce-policy', required=True, help='CE supervision policy whose own-output objective weights the tokens')
    p.add_argument('--out', required=True)
    p.add_argument('--drop-family', action='append', default=[])
    p.add_argument('--workers', type=int, default=12)
    p = sub.add_parser('sample')
    p.add_argument('--out', required=True)
    p.add_argument('--tokenizer', required=True)
    p.add_argument('--per-family', type=int, default=2)
    p.add_argument('--chars', type=int, default=4000)
    p = sub.add_parser('select')
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--budget-tokens', type=int, required=True, help='forward tokens to select (all families)')
    p.add_argument('--share', action='append', default=[], help='FAMILY=SHARE (overrides the default target shares)')
    p.add_argument('--held-cases', action='append', default=[], help='held task cases.jsonl (repeatable)')
    p.add_argument('--held-ids', action='append', default=[], help='JSON list of held ids and siblings (e.g. held-v2/exclusion-ids.json)')
    p.add_argument('--tokenizer', help='decode chosen rows for the text-level held check (skipped without)')
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--seed', type=int, default=20261112)
    args = parser.parse_args()
    {'build': build, 'sample': sample, 'select': select}[args.command](args)


if __name__ == '__main__':
    main()
