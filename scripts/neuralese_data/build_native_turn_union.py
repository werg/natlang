#!/usr/bin/env python3
"""One shared training set from rendered native text members, per rendering and tokenizer twin.

Members are rendered text files (``text-data.jsonl``/``text.jsonl`` rows of the shared ``load_text_rows`` schema),
named ``NAME:RENDERING:TOKENIZER=PATH``. For every rendering the union holds the SAME documents for each tokenizer, in
the same order (twin alignment by document id). Rules:

1. Held stays held: every member's test documents are kept; the source groups and text digests of all held documents
   of all members (every tokenizer) are collected first.
2. Source-group-disjoint decontamination across corpora: a train document sharing a source group with any held
   document, or whose text equals a held document's, is dropped.
3. Exact duplicates across members (same split and text in a tokenizer) are kept once, first member wins.
4. Twin alignment: a document id missing for a tokenizer, or dropped for any tokenizer, is dropped for all.
5. Document ids are unique per rendering: an id already taken by an earlier member is dropped from the later one.

Each row gets ``corpus`` (the member name) when it has none. Omissions are written with reasons; the receipt counts
documents, tokens and supervised target tokens per member, rendering and tokenizer.

Usage (CPU, natlang-neuralese image, under the memory ledger):
  build_native_turn_union.py --out-dir OUT --tokenizer lfm25=DIR --tokenizer mellum21=DIR \
      --member NAME:RENDERING:TOKENIZER=PATH [...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))
from render_native_turn_corpus import canonical, file_sha, role_counter, sha, whole_trajectory_target_tokens  # noqa: E402

SCHEMA = 'natlang.native-text-union/1'


def rows(path: Path):
    """Rows of a text file; ``.zst`` files (compressed full renders of the static campaign) stream through zstd."""
    if str(path).endswith('.zst'):
        import subprocess
        proc = subprocess.Popen(['zstd', '-dcq', '--long=27', str(path)], stdout=subprocess.PIPE, text=True)
        try:
            for line in proc.stdout:
                if line.strip():
                    yield line, json.loads(line)
        finally:
            proc.stdout.close()
            if proc.wait() not in (0, None):
                raise RuntimeError(f'zstd failed on {path}')
        return
    with open(path) as f:
        for line in f:
            if line.strip():
                yield line, json.loads(line)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--tokenizer', action='append', required=True, help='NAME=DIR')
    p.add_argument('--member', action='append', required=True, help='NAME:RENDERING:TOKENIZER=PATH')
    a = p.parse_args(argv)
    t0 = time.time()
    tokenizer_dirs = dict(item.split('=', 1) for item in a.tokenizer)
    members = defaultdict(dict)  # rendering -> [(name, {tok: path})] in order
    order = defaultdict(list)
    for item in a.member:
        spec, path = item.split('=', 1)
        name, rendering, tok = spec.split(':')
        if tok not in tokenizer_dirs:
            p.error(f'unknown tokenizer {tok} in {item}')
        if name not in members[rendering]:
            order[rendering].append(name)
            members[rendering][name] = {}
        members[rendering][name][tok] = Path(path)
    for rendering, by_name in members.items():
        for name, paths in by_name.items():
            if set(paths) != set(tokenizer_dirs):
                p.error(f'member {name}:{rendering} lacks a tokenizer twin: has {sorted(paths)}')
    # Pass 1: held groups and held texts over every member, rendering and tokenizer.
    held_groups, held_texts = set(), set()
    for rendering, by_name in members.items():
        for name, paths in by_name.items():
            for tok, path in paths.items():
                for _, row in rows(path):
                    if row['split'] == 'test':
                        held_groups.update(row['source_groups'])
                        held_texts.add(sha(row['text'].encode()))
    print(json.dumps({'held_groups': len(held_groups), 'held_texts': len(held_texts),
                      'seconds': round(time.time() - t0)}), flush=True)
    from build_neuralese_gold_text_delta import load_pinned_tokenizer
    counters = {tok: role_counter(load_pinned_tokenizer(Path(d))) for tok, d in tokenizer_dirs.items()}
    # The union is a streamed, filtered concatenation of its members: bind each member file it read (a compressed
    # member is bound by its file hash; the union is then the only plain copy of it).
    member_files = {rendering: {name: {tok: {'path': str(path), 'bytes': Path(path).stat().st_size,
                                             'sha256': file_sha(Path(path))}
                                       for tok, path in paths.items()}
                                for name, paths in by_name.items()}
                    for rendering, by_name in members.items()}
    receipt = {'schema': SCHEMA, 'renderings': {}, 'held_groups': len(held_groups), 'held_texts': len(held_texts),
               'member_files': member_files,
               'tokenizers': {tok: str(Path(d).resolve()) for tok, d in tokenizer_dirs.items()},
               'code': {str(f.relative_to(ROOT)): file_sha(f) for f in (
                   Path(__file__).resolve(), ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py')}}
    for rendering in members:
        # Pass 2: decide per member and document id (all tokenizers), then write aligned files.
        drop = {}  # (member, id) -> reason
        ids = {}
        seen = {tok: set() for tok in tokenizer_dirs}
        claimed = {}  # document id -> first member holding it (ids are unique in the union; first member wins)
        for name in order[rendering]:
            per_tok = {}
            for tok, path in members[rendering][name].items():
                per_tok[tok] = []
                for _, row in rows(path):
                    rid = row['id']
                    per_tok[tok].append(rid)
                    reason = None
                    if row['split'] == 'train':
                        if set(row['source_groups']) & held_groups:
                            reason = 'train_document_shares_held_source_group'
                        elif sha(row['text'].encode()) in held_texts:
                            reason = 'train_copy_of_held_document'
                    key = (row['split'], sha(row['text'].encode()))
                    if reason is None and key in seen[tok]:
                        reason = f'exact_duplicate_document ({tok})'
                    seen[tok].add(key)
                    if reason is not None and (name, rid) not in drop:
                        drop[(name, rid)] = reason
            id_sets = {tok: set(v) for tok, v in per_tok.items()}
            common = set.intersection(*id_sets.values())
            for tok, v in id_sets.items():
                for rid in v - common:
                    drop.setdefault((name, rid), f'missing_tokenizer_twin (only {tok})')
            # The same document id from two members (e.g. two registered renders of one source, one not declared
            # superseded; r3 held 19,921 program-header ids twice with different texts) would give the trainer two
            # different documents under one id: the later member's copy is dropped.
            for rid in set.union(*id_sets.values()):
                if rid in claimed:
                    drop[(name, rid)] = f'duplicate_document_id (first in {claimed[rid]})'
                else:
                    claimed[rid] = name
            ids[name] = per_tok
        out = a.out_dir / rendering
        out.mkdir(parents=True, exist_ok=True)
        rendering_receipt = {'members': order[rendering], 'by_tokenizer': {}}
        omissions = [{'member': n, 'id': i, 'reason': r} for (n, i), r in drop.items()]
        (out / 'omissions.jsonl').write_text(''.join(canonical(o) + '\n' for o in omissions))
        rendering_receipt['omissions'] = dict(Counter(f"{o['member']}: {o['reason'].split(' (')[0]}" for o in omissions))
        for tok in tokenizer_dirs:
            target = out / tok / 'text-data.jsonl'
            target.parent.mkdir(parents=True, exist_ok=True)
            stats = {'documents': 0, 'by_member': {}, 'by_split': {}}
            order_check = []
            with open(target, 'w') as dst:
                for name in order[rendering]:
                    m = stats['by_member'].setdefault(name, {'documents': 0, 'tokens': 0, 'supervised_tokens': 0,
                                                             'train_documents': 0, 'test_documents': 0})
                    for line, row in rows(members[rendering][name][tok]):
                        if (name, row['id']) in drop:
                            continue
                        if 'corpus' not in row:
                            row['corpus'] = name
                            line = canonical(row) + '\n'
                        n = len(row['token_ids'])
                        supervised = (n - row['supervised_suffix_start'] if row.get('supervised_suffix_start') is not None
                                      else whole_trajectory_target_tokens(row, counters[tok])[0])
                        for bucket in (m, stats['by_split'].setdefault(row['split'], {'documents': 0, 'tokens': 0,
                                                                                      'supervised_tokens': 0})):
                            bucket['documents'] += 1
                            bucket['tokens'] += n
                            bucket['supervised_tokens'] += supervised
                        m[row['split'] + '_documents'] += 1
                        stats['documents'] += 1
                        order_check.append((row['id'], row['split'], tuple(row['source_groups'])))
                        dst.write(line if line.endswith('\n') else line + '\n')
            stats['text_data_sha256'] = file_sha(target)
            stats['identity_sha256'] = sha(canonical(order_check).encode())
            rendering_receipt['by_tokenizer'][tok] = stats
            print(json.dumps({rendering: tok, 'documents': stats['documents'], 'by_split': stats['by_split'],
                              'seconds': round(time.time() - t0)}), flush=True)
        identities = {s['identity_sha256'] for s in rendering_receipt['by_tokenizer'].values()}
        rendering_receipt['twin_identity_equal'] = len(identities) == 1
        if len(identities) != 1:
            raise SystemExit(f'{rendering}: tokenizer twins differ in id/split/groups order')
        receipt['renderings'][rendering] = rendering_receipt
    receipt['seconds'] = round(time.time() - t0)
    (a.out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
