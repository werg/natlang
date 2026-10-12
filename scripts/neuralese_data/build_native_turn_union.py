#!/usr/bin/env python3
"""One shared training set from rendered native text members, per rendering and tokenizer twin.

Members are rendered text files (``text-data.jsonl``/``text.jsonl`` rows of the shared ``load_text_rows`` schema),
named ``NAME:RENDERING:TOKENIZER=PATH``. For every rendering the union holds the SAME documents for each tokenizer, in
the same order (twin alignment by document id). Rules:

1. Held stays held: every member's test documents are kept; the source groups and text digests of all held documents
   of all members (every tokenizer) are collected first.
2. Source-group-disjoint decontamination across corpora: a train document sharing a source group with any held
   document, or whose text equals a held document's, is dropped.
3. Only complete, otherwise-eligible tokenizer twins participate in duplicate resolution.
   Exact duplicates across eligible members are kept once, first member wins unless a later member declares that it
   supersedes the first.
4. Twin alignment: a document id missing for a tokenizer, or dropped for any tokenizer, is dropped for all.
5. Document ids are unique per rendering. Conflicting split/source groups preserve a held/test copy and explicitly
   omit the other copy. Distinct payloads with the same id keep the first eligible copy unless a later member
   explicitly supersedes it.

Each row gets ``corpus`` (the member name) when it has none. Omissions are written with reasons; the receipt counts
documents, tokens and supervised target tokens per member, rendering and tokenizer.

Usage (CPU, natlang-neuralese image, under the memory ledger):
  build_native_turn_union.py --out-dir OUT --tokenizer lfm25=DIR --tokenizer mellum21=DIR \
      --member NAME:RENDERING:TOKENIZER=PATH [...] [--supersedes NEWER=OLDER ...]
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

SCHEMA = 'natlang.native-text-union/2'


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


def _row_identity(row: dict) -> dict:
    return {
        'split': row['split'],
        'source_groups': tuple(sorted(row['source_groups'])),
        'text_sha256': sha(row['text'].encode()),
    }


def _drop(drop: dict, key: tuple[str, str], reason: str, **details) -> None:
    if key not in drop:
        drop[key] = {'reason': reason, **details}


def plan_member_rows(order: list[str], member_rows: dict, supersedes: dict,
                     tokenizer_names: list[str], held_groups: set, held_texts: set,
                     member_duplicates: dict | None = None) -> tuple[set, dict]:
    """Return (selected (member,id) pairs, omissions) after twin and source-safety checks.

    ``member_rows`` holds only row identities, not rendered text/token payloads. A candidate becomes eligible only
    after it has a consistent row in every tokenizer and passes the held-source filters. Duplicate resolution sees
    eligible candidates only, so an incomplete or held row cannot reserve a document ID or text hash.
    """
    drop = {}
    eligible_by_id = defaultdict(list)
    observed_by_id = defaultdict(list)
    member_duplicates = member_duplicates or {}
    for name in order:
        per_tok = member_rows[name]
        id_sets = {tok: set(per_tok.get(tok, {})) for tok in tokenizer_names}
        common = set.intersection(*id_sets.values()) if id_sets else set()
        all_ids = set.union(*id_sets.values()) if id_sets else set()
        for rid in sorted(all_ids):
            key = (name, rid)
            observations = [per_tok[tok][rid] for tok in tokenizer_names if rid in per_tok.get(tok, {})]
            observed_by_id[rid].extend({'member': name, 'identity': identity} for identity in observations)
            if rid in member_duplicates.get(name, set()):
                _drop(drop, key, 'duplicate_document_id_within_member')
                continue
            if rid not in common:
                present_tok = [tok for tok in tokenizer_names if rid in id_sets[tok]]
                _drop(drop, key, 'missing_tokenizer_twin', present_in=present_tok)
                continue
            signatures = [per_tok[tok][rid] for tok in tokenizer_names]
            if any(identity != signatures[0] for identity in signatures[1:]):
                _drop(drop, key, 'tokenizer_twin_identity_mismatch',
                      tokenizer_text_sha256={tok: per_tok[tok][rid]['text_sha256'] for tok in tokenizer_names})
                continue
            identity = signatures[0]
            if identity['split'] == 'train' and set(identity['source_groups']) & held_groups:
                _drop(drop, key, 'train_document_shares_held_source_group',
                      text_sha256=identity['text_sha256'])
                continue
            if identity['split'] == 'train' and identity['text_sha256'] in held_texts:
                _drop(drop, key, 'train_copy_of_held_document', text_sha256=identity['text_sha256'])
                continue
            eligible_by_id[rid].append({'member': name, 'identity': identity})

    # First choose one eligible owner for each ID. A held/test copy wins a split/group conflict. Within a consistent
    # identity, a later row replaces its current owner only when the later member explicitly names that owner in
    # ``supersedes``; every other conflict is an explicit, hash-bound omission.
    selected_by_id = {}
    for rid, candidates in eligible_by_id.items():
        candidates.sort(key=lambda candidate: order.index(candidate['member']))
        winner = candidates[0]
        winner_identity = winner['identity']
        observations = observed_by_id[rid]
        conflict = any(
            candidate['identity']['split'] != winner_identity['split'] or
            candidate['identity']['source_groups'] != winner_identity['source_groups']
            for candidate in observations
        )
        if conflict:
            tests = [candidate for candidate in candidates if candidate['identity']['split'] == 'test']
            if tests:
                winner = tests[0]
            elif any(candidate['identity']['split'] == 'test' for candidate in observations):
                # A test copy still controls the split even when that copy has no complete eligible twin. It cannot
                # claim the ID or be emitted, but an eligible train copy with a conflicting identity is not safe.
                for candidate in candidates:
                    held = next(candidate for candidate in observations if candidate['identity']['split'] == 'test')
                    _drop(drop, (candidate['member'], rid), 'conflicting_split_duplicate_id',
                          held_member=held['member'], held_text_sha256=held['identity']['text_sha256'],
                          omitted_split=candidate['identity']['split'],
                          omitted_source_groups=list(candidate['identity']['source_groups']),
                          omitted_text_sha256=candidate['identity']['text_sha256'])
                continue
            else:
                # With no held/test observation, keep the first eligible representation and omit conflicting eligible
                # siblings. Ineligible rows never become ID owners.
                winner = candidates[0]
            for candidate in candidates:
                if candidate is winner:
                    continue
                _drop(drop, (candidate['member'], rid), 'conflicting_split_duplicate_id',
                      kept_member=winner['member'], kept_split=winner['identity']['split'],
                      kept_source_groups=list(winner['identity']['source_groups']),
                      kept_text_sha256=winner['identity']['text_sha256'],
                      omitted_split=candidate['identity']['split'],
                      omitted_source_groups=list(candidate['identity']['source_groups']),
                      omitted_text_sha256=candidate['identity']['text_sha256'])
            selected_by_id[rid] = winner
            continue

        for candidate in candidates[1:]:
            current = winner
            if current['member'] in supersedes.get(candidate['member'], set()):
                _drop(drop, (current['member'], rid), 'superseded_duplicate_document_id',
                      kept_member=candidate['member'], superseded_by=candidate['member'],
                      kept_text_sha256=candidate['identity']['text_sha256'],
                      omitted_text_sha256=current['identity']['text_sha256'])
                winner = candidate
            else:
                _drop(drop, (candidate['member'], rid), 'duplicate_document_id',
                      kept_member=current['member'], kept_text_sha256=current['identity']['text_sha256'],
                      omitted_text_sha256=candidate['identity']['text_sha256'])
        selected_by_id[rid] = winner

    # Exact-text de-duplication runs only over the eligible per-ID winners. The first selected member wins unless
    # the later candidate explicitly supersedes that text owner.
    selected = set()
    seen_text = {}
    for rid, candidate in sorted(selected_by_id.items(), key=lambda item: order.index(item[1]['member'])):
        key = (candidate['identity']['split'], candidate['identity']['text_sha256'])
        if key in seen_text:
            kept_member, kept_id = seen_text[key]
            if kept_member in supersedes.get(candidate['member'], set()):
                selected.discard((kept_member, kept_id))
                _drop(drop, (kept_member, kept_id), 'superseded_exact_duplicate_document',
                      kept_member=candidate['member'], superseded_by=candidate['member'], kept_id=rid,
                      kept_text_sha256=key[1], omitted_text_sha256=key[1])
                selected.add((candidate['member'], rid))
                seen_text[key] = (candidate['member'], rid)
            else:
                _drop(drop, (candidate['member'], rid), 'exact_duplicate_document',
                      kept_member=kept_member, kept_id=kept_id,
                      kept_text_sha256=key[1], omitted_text_sha256=key[1])
                continue
        else:
            seen_text[key] = (candidate['member'], rid)
        selected.add((candidate['member'], rid))
    return selected, drop


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--tokenizer', action='append', required=True, help='NAME=DIR')
    p.add_argument('--member', action='append', required=True, help='NAME:RENDERING:TOKENIZER=PATH')
    p.add_argument('--supersedes', action='append', default=[],
                   help='MEMBER=OLDER_MEMBER; only the named later member may replace the older member')
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
    supersedes = defaultdict(set)
    for item in a.supersedes:
        try:
            newer, older = item.split('=', 1)
        except ValueError:
            p.error(f'invalid --supersedes {item!r}; expected MEMBER=OLDER_MEMBER')
        if not newer or not older or newer == older:
            p.error(f'invalid --supersedes {item!r}; member names must be distinct and non-empty')
        supersedes[newer].add(older)
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
               'member_supersedes': {rendering: {name: sorted(supersedes.get(name, set())) for name in names}
                                     for rendering, names in order.items()},
               'tokenizers': {tok: str(Path(d).resolve()) for tok, d in tokenizer_dirs.items()},
               'code': {str(f.relative_to(ROOT)): file_sha(f) for f in (
                   Path(__file__).resolve(), ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py')}}
    for rendering in members:
        # Pass 2: index compact row identities in both tokenizer files before resolving any duplicate. This prevents
        # an incomplete or held copy from claiming an ID/text hash before a later complete eligible twin is seen.
        member_rows, member_duplicates = {}, {}
        for name in order[rendering]:
            per_tok, duplicates = {}, set()
            for tok, path in members[rendering][name].items():
                per_tok[tok] = {}
                for _, row in rows(path):
                    rid = row['id']
                    if rid in per_tok[tok]:
                        duplicates.add(rid)
                    else:
                        per_tok[tok][rid] = _row_identity(row)
            member_rows[name] = per_tok
            member_duplicates[name] = duplicates
        selected, drop = plan_member_rows(order[rendering], member_rows, supersedes,
                                          list(tokenizer_dirs), held_groups, held_texts,
                                          member_duplicates=member_duplicates)
        out = a.out_dir / rendering
        out.mkdir(parents=True, exist_ok=True)
        rendering_receipt = {'members': order[rendering], 'by_tokenizer': {}}
        omissions = [{'member': n, 'id': i, **details} for (n, i), details in sorted(drop.items())]
        (out / 'omissions.jsonl').write_text(''.join(canonical(o) + '\n' for o in omissions))
        rendering_receipt['omissions'] = dict(Counter(f"{o['member']}: {o['reason']}" for o in omissions))
        rendering_receipt['duplicate_id_conflicts'] = [o for o in omissions
                                                        if o['reason'] == 'conflicting_split_duplicate_id']
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
                        if (name, row['id']) not in selected:
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
