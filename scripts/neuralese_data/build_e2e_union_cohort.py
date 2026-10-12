#!/usr/bin/env python3
"""End-to-end Neuralese stage records from every union source with child-result write/read sites.

Selects both endpoints of every authenticated producer->consumer gradient edge
(train/trajectories.py authenticated_recurrence_context_view, the rule that selected the 1243-record cohort) over the
sources below, first source wins per id; lines are copied byte-for-byte. Splits are the rows' own; cross-split edges
and train edges sharing a source group with a held document of the target union are dropped. Writes records.jsonl,
pieces.jsonl, edges.jsonl and receipt.json to the output directory. Run under the memory ledger (CPU, ~3 GB).
"""
import json, sys, os
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402
from natlang_neuralese.train.trajectories import authenticated_recurrence_context_view  # noqa: E402
R = str(ROOT) + '/'
ID = 'neuralese-e2e-union-cohort-20261012-v1'
OUT = resolve_str('data_nvme', 'natlang-corpora', ID)
os.makedirs(OUT, exist_ok=True)
SOURCES = [
 ('assembly-v3', 'runs/v18-v19-native-soft-edge-review-20261009-v1/cumulative-full-assembly-v3/recurrence-records.jsonl',
  'runs/v18-v19-native-soft-edge-review-20261009-v1/cumulative-full-assembly-v3/recurrence-pieces.jsonl'),
 ('luna-v17-v9-v3', 'runs/luna-v17-actual-quality-proposal-20261010-v9-portable-context-view-v1/assembled-v3/native-records.jsonl',
  'runs/luna-v17-actual-quality-proposal-20261010-v9-portable-context-view-v1/assembled-v3/native-pieces.jsonl'),
 ('free-v28', 'runs/free-provider-generation-20261009-v1/held-v28-visible-input-native-text-v10-target-binding/native-turns-render-input.jsonl',
  'runs/free-provider-generation-20261009-v1/held-v28-visible-input-native-text-v10-target-binding/pieces.jsonl'),
 ('free-v29', 'runs/free-provider-generation-20261009-v1/v29-root-ordinary-text-admission-v1/selected-native-turns-render-input.jsonl',
  'runs/free-provider-generation-20261009-v1/held-v29-visible-input-native-text-v2/pieces.jsonl'),
]
tables = {}
for name, rp, pp in SOURCES:
    tables[name] = {json.loads(l)['name']: (l if l.endswith('\n') else l + '\n') for l in open(R+pp)}
base = {n: json.loads(l)['text'] for n, l in tables['assembly-v3'].items()}
conflict = {n for name, t in tables.items() if name != 'assembly-v3' for n, l in t.items() if n in base and json.loads(l)['text'] != base[n]}
lines, origin, held_conflict = {}, {}, []
for name, rp, pp in SOURCES:
    for l in open(R+rp):
        if not l.strip(): continue
        r = json.loads(l)
        if r['id'] in origin: continue
        if name != 'assembly-v3' and any(f'"name": "{n}"' in l or f'"name":"{n}"' in l for n in conflict):
            held_conflict.append(r['id']); origin[r['id']] = 'held:piece_version_conflict'; continue
        origin[r['id']] = name; lines[r['id']] = l if l.endswith('\n') else l + '\n'
pieces = dict(base)
piece_lines = dict(tables['assembly-v3'])
for name, t in tables.items():
    for n, l in t.items():
        if n not in pieces: pieces[n] = json.loads(l)['text']; piece_lines[n] = l
records = [json.loads(l) for l in lines.values()]
prepared, review = authenticated_recurrence_context_view(records, pieces)
split = {r['id']: r['split'] for r in records}; groups = {r['id']: set(r.get('source_groups') or []) for r in records}
edges = set()
for d in review['decisions']:
    for ref in d.get('references') or []:
        if ref.get('gradient_edge'): edges.add((ref.get('writer_record_id'), d['id'], ref.get('name'), ref.get('body_sha256')))
held_groups = set()
U = R+'runs/natlang-target-union-20261011-r3'
for rend in ('full', 'program-header', 'whole-trajectory'):
    for l in open(f'{U}/{rend}/lfm25/text-data.jsonl'):
        if '"test"' in l:
            t = json.loads(l)
            if t['split'] == 'test': held_groups.update(t['source_groups'])
kept, dropped = [], Counter()
for w, rd, n, b in sorted(edges, key=lambda e: tuple(str(x) for x in e)):
    if split.get(w) != split.get(rd): dropped['cross_split'] += 1; continue
    if split[w] == 'train' and ((groups[w] | groups[rd]) & held_groups): dropped['train_shares_union_held_group'] += 1; continue
    kept.append({'writer_record_id': w, 'reader_record_id': rd, 'write_name': n, 'body_sha256': b, 'split': split[rd],
                 'source': origin[rd]})
ends = []
order = {i: k for k, i in enumerate(lines)}
for i in sorted({x for e in kept for x in (e['writer_record_id'], e['reader_record_id'])}, key=lambda i: order[i]):
    ends.append(i)
with open(f'{OUT}/records.jsonl', 'w') as f:
    for i in ends: f.write(lines[i])
with open(f'{OUT}/pieces.jsonl', 'w') as f:
    for n, l in piece_lines.items(): f.write(l)  # the full merged table (assembly-v3 wins every name; no conflicts among the rest)
with open(f'{OUT}/edges.jsonl', 'w') as f:
    for e in kept: f.write(json.dumps(e, sort_keys=True) + '\n')
receipt = {
 'schema': 'natlang.neuralese-e2e-cohort/1', 'id': ID,
 'format': 'natlang.teacher_training_turn.native/1 records consumed by train/trajectories.py --records/--pieces (--handover written, --direct-artifact); every line byte-identical to its source line; tokenizer-agnostic (the trainer tokenizes per line), so one file serves both lines in lockstep',
 'selection': 'both endpoints of every gradient_edge=true reference of authenticated_recurrence_context_view (train/trajectories.py) over the union of the sources below, first source wins per id; the same rule that selected the 1243 cohort (full-candidate-closure-v1), which this cohort contains byte-identically',
 'splits': 'each record keeps its own split and source_groups (the unions inherit the same rows); cross-split edges dropped; train edges whose endpoints share a source group with any held document of union r3 (all renderings) dropped',
 'sources': [{'name': n, 'records': rp, 'records_sha256': sha256_file_hex(R+rp), 'pieces': pp, 'pieces_sha256': sha256_file_hex(R+pp)} for n, rp, pp in SOURCES],
 'piece_policy': f'assembly-v3 piece table (the pinned e2e_pieces) wins every name; {len(conflict)} names differ in a later table ({sorted(conflict)}); later-source records referencing them are held (piece_version_conflict) rather than read with another version',
 'held_piece_version_conflict_records': len(held_conflict),
 'review': {k: review[k] for k in ('source_records', 'prepared_records', 'excluded_context_records', 'prepared_by_split', 'excluded_by_split')},
 'excluded_reasons': Counter(e['reason'][:100] for e in review['excluded']),
 'edges': len(kept), 'edges_by_split': Counter(e['split'] for e in kept), 'edges_by_source': Counter(e['source'] for e in kept),
 'edges_dropped': dropped,
 'records': len(ends), 'records_by_split': Counter(split[i] for i in ends), 'records_by_source': Counter(origin[i] for i in ends),
 'readers_by_split': Counter(split[e['reader_record_id']] for e in kept),
 'files': {n: {'sha256': sha256_file_hex(f'{OUT}/{n}'), 'bytes': os.path.getsize(f'{OUT}/{n}')}
           for n in ('records.jsonl', 'pieces.jsonl', 'edges.jsonl')},
}
json.dump(receipt, open(f'{OUT}/receipt.json', 'w'), indent=1, sort_keys=True, default=dict)
print(json.dumps({k: receipt[k] for k in ('records', 'records_by_split', 'records_by_source', 'edges', 'edges_by_split', 'edges_by_source', 'edges_dropped', 'held_piece_version_conflict_records', 'files')}, default=dict, indent=1))
