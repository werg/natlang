#!/usr/bin/env python3
"""The shared natlang target training set: one registry entry listing every member render and every pending piece.

Reads ``plans/neuralese/TARGET_SET_20261011.json`` (members: registry ids of renders in the shared layout
``<path>/<rendering>/<tokenizer>/text-data.jsonl``; pending: datasets not integrated yet, with the exact reason and
conversion) and writes the registry entry ``natlang-target-set-20261011`` with per-member, per-rendering and
per-tokenizer documents and supervised target tokens, and their totals. Run again whenever a piece lands.
Cross-corpus decontamination and exact deduplication of the members are done by build_native_turn_union.py when
the set is materialized as one file per rendering and tokenizer.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'training' / 'neuralese_corpora.json'
SPEC = ROOT / 'plans' / 'neuralese' / 'TARGET_SET_20261011.json'
BASE_ID = 'natlang-target-set-20261011'  # revisions: BASE_ID-r<N> (manifests are immutable snapshots)


def file_stats(path: Path) -> dict:
    import hashlib
    out = {'documents': 0, 'train_documents': 0, 'test_documents': 0, 'train_tokens': 0, 'train_supervised_tokens': 0}
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for line in f:
            h.update(line)
            row = json.loads(line)
            n = len(row['token_ids'])
            out['documents'] += 1
            out[row['split'] + '_documents'] += 1
            if row['split'] == 'train':
                out['train_tokens'] += n
                if row.get('supervised_suffix_start') is not None:
                    out['train_supervised_tokens'] += n - row['supervised_suffix_start']
    out['sha256'] = h.hexdigest()
    return out


def main():
    spec = json.loads(SPEC.read_text())
    spec['revision'] = int(spec.get('revision', 0)) + 1
    ID = f"{BASE_ID}-r{spec['revision']}"
    registry = json.loads(REGISTRY.read_text())
    by_id = {c['id']: c for c in registry['corpora']}
    members, totals = [], {}
    for mid in spec['members']:
        entry = by_id[mid]
        if entry.get('training_admission') is not True:
            raise SystemExit(f'member {mid} is not training-admitted')
        renders = {}
        for rendering, toks in entry['renders'].items():
            for tok, stats in toks.items():
                split = stats['by_split']
                row = {'file': f"{entry['path']}/{stats['file']}", 'sha256': stats['sha256'],
                       'documents': stats['documents'],
                       'train_documents': split.get('train', {}).get('documents', 0),
                       'test_documents': split.get('test', {}).get('documents', 0),
                       'train_tokens': split.get('train', {}).get('tokens', 0),
                       'train_supervised_tokens': split.get('train', {}).get('supervised_tokens', 0)}
                renders.setdefault(rendering, {})[tok] = row
                t = totals.setdefault(rendering, {}).setdefault(tok, {'documents': 0, 'train_documents': 0,
                                                                     'test_documents': 0, 'train_tokens': 0,
                                                                     'train_supervised_tokens': 0})
                for k in t:
                    t[k] += row[k]
        members.append({'id': mid, 'derived_from': entry.get('derived_from'), 'renders': renders})
    for ext in spec.get('external_members', []):  # registered text sets in their own layout (4962 union, harness)
        renders = {}
        for rendering, toks in ext['files'].items():
            for tok, path in toks.items():
                row = file_stats(ROOT / path)
                row['file'] = path
                renders.setdefault(rendering, {})[tok] = row
                t = totals.setdefault(rendering, {}).setdefault(tok, {'documents': 0, 'train_documents': 0,
                                                                     'test_documents': 0, 'train_tokens': 0,
                                                                     'train_supervised_tokens': 0})
                for k in t:
                    t[k] += row[k]
        members.append({'id': ext['id'], 'registry_ids': ext['registry_ids'], 'note': ext.get('note', ''),
                        'renders': renders})
    entry = {
        'id': ID, 'owner': 'dgx', 'path': 'plans/neuralese/target-set', 'include': [f'{ID}.json'],
        'source_class': 'shared-natlang-target-training-set-manifest',
        'admission': 'manifest of training-admitted member renders (owner 2026-10-11: expand the target set by '
                     'everything found that can be integrated); exclusions only protected eval sets, held splits '
                     'and defect rows under repair',
        'training_admission': True,
        'training_admission_reason': 'lists training-admitted members only; consume through the union files once '
                                     'built (build_native_turn_union.py), or member files for early starts',
        'split': 'per member; held stays held; cross-member source-group decontamination at union build',
        'members': members, 'totals_before_cross_member_dedup': totals,
        'pending': spec['pending'],
        'notes': spec.get('notes', ''),
    }
    registry = json.loads(REGISTRY.read_text())  # re-read: other agents may have registered meanwhile
    for c in registry['corpora']:
        if c['id'] == BASE_ID or c['id'].startswith(BASE_ID + '-r'):
            c['superseded_by'] = ID
            if c['id'] == BASE_ID:
                c['retired'] = True  # its included spec file is the living spec, so its snapshot no longer verifies
    registry['corpora'] = [c for c in registry['corpora'] if c['id'] != ID] + [entry]
    SPEC.write_text(json.dumps(spec, indent=2, ensure_ascii=False) + '\n')
    snapshot = ROOT / 'plans' / 'neuralese' / 'target-set' / f'{ID}.json'  # immutable per revision
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    snapshot.write_text(json.dumps({**spec, 'members_resolved': members, 'totals_before_cross_member_dedup': totals},
                                   indent=1, ensure_ascii=False) + '\n')
    print(ID)
    REGISTRY.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(totals, indent=1))


if __name__ == '__main__':
    main()
