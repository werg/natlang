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
import os
import subprocess
from contextlib import contextmanager
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))
from render_native_turn_corpus import canonical, file_sha, role_counter, sha, whole_trajectory_target_tokens  # noqa: E402
from text_tokenizer_twin import document_identity, IDENTITY_FIELDS, PROVENANCE_FIELDS  # noqa: E402
from source_review import pending_unpinned_source_groups, registry_sha256  # noqa: E402

SCHEMA = 'natlang.native-text-union/2'


@contextmanager
def union_text_writer(path, compressed=False):
    """Exact UTF-8 output, optionally streamed through zstd without a raw working copy."""
    digest = hashlib.sha256()
    class Writer:
        decoded_bytes = 0
        def write(self, text):
            data = text.encode('utf-8')
            stream.write(data)
            digest.update(data)
            self.decoded_bytes += len(data)
    writer = Writer()
    with open(path, 'wb') as raw:
        proc = subprocess.Popen(['zstd', '-q', '-3', '--long=27', '-T2', '-c'],
                                stdin=subprocess.PIPE, stdout=raw) if compressed else None
        stream = proc.stdin if proc else raw
        try:
            yield writer
        except BaseException:
            if proc is not None:
                proc.terminate()
            raise
        finally:
            if proc is not None:
                stream.close()
                result = proc.wait()
        if proc is not None and result:
            raise RuntimeError(f'union zstd encoder failed: {result}')
        raw.flush()
        os.fsync(raw.fileno())
    writer.decoded_sha256 = digest.hexdigest()
    if compressed:
        from natlang_neuralese.common.jsonio import open_jsonl_text
        actual, size = hashlib.sha256(), 0
        with open_jsonl_text(path) as decoded:
            for line in decoded:
                data = line.encode('utf-8')
                actual.update(data)
                size += len(data)
        if (actual.hexdigest(), size) != (writer.decoded_sha256, writer.decoded_bytes):
            raise ValueError('compressed union decoded identity mismatch')


def rows(path: Path):
    """Exact literal-LF rows, plain or compressed through the shared text opener."""
    from natlang_neuralese.common.jsonio import open_jsonl_text
    with open_jsonl_text(path) as stream:
        for line in stream:
            if line.strip():
                yield line, json.loads(line)


def _row_identity(row: dict) -> dict:
    return {
        'split': row['split'],
        'source_groups': tuple(sorted(row['source_groups'])),
        'text_sha256': sha(row['text'].encode()),
        'source_identity_sha256': (sha(canonical(document_identity(row)).encode())
                                   if row.get('source_record_ids') else None),
    }


def _same_twin_identity(left, right):
    if left.get('source_identity_sha256') is not None or right.get('source_identity_sha256') is not None:
        return (left['split'] == right['split'] and left['source_groups'] == right['source_groups']
                and left.get('source_identity_sha256') is not None
                and left.get('source_identity_sha256') == right.get('source_identity_sha256'))
    # Legacy rows without source provenance can only prove a twin by exact text.
    return left == right


def _drop(drop: dict, key: tuple[str, str], reason: str, **details) -> None:
    if key not in drop:
        drop[key] = {'reason': reason, **details}
        return
    if drop[key]['reason'] == reason:
        return
    additional = drop[key].setdefault('additional_reasons', [])
    record = {'reason': reason, **details}
    if record not in additional:
        additional.append(record)


def plan_member_rows(order: list[str], member_rows: dict, supersedes: dict,
                     tokenizer_names: list[str], held_groups: set, held_texts: set,
                     member_duplicates: dict | None = None,
                     member_duplicate_observations: dict | None = None,
                     source_review_groups: set[str] | None = None) -> tuple[set, dict]:
    """Return (selected (member,id) pairs, omissions) after twin and source-safety checks.

    ``member_rows`` holds only row identities, not rendered text/token payloads. A candidate becomes eligible only
    after it has a consistent row in every tokenizer and passes the held-source filters. Duplicate resolution sees
    eligible candidates only, so an incomplete or held row cannot reserve a document ID or text hash.
    """
    drop = {}
    eligible_by_id = defaultdict(list)
    observed_by_id = defaultdict(list)
    member_duplicates = member_duplicates or {}
    member_duplicate_observations = member_duplicate_observations or {}
    source_review_groups = source_review_groups or set()
    member_rank = {name: index for index, name in enumerate(order)}
    for name in order:
        per_tok = member_rows[name]
        id_sets = {tok: set(per_tok.get(tok, {})) for tok in tokenizer_names}
        common = set.intersection(*id_sets.values()) if id_sets else set()
        all_ids = set.union(*id_sets.values()) if id_sets else set()
        for rid in sorted(all_ids):
            key = (name, rid)
            observations = [per_tok[tok][rid] for tok in tokenizer_names if rid in per_tok.get(tok, {})]
            observed_by_id[rid].extend({'member': name, 'identity': identity} for identity in observations)
            observed_by_id[rid].extend(
                {'member': name, 'identity': identity}
                for identity in member_duplicate_observations.get(name, {}).get(rid, []))
            if rid in member_duplicates.get(name, set()):
                _drop(drop, key, 'duplicate_document_id_within_member')
                continue
            if rid not in common:
                present_tok = [tok for tok in tokenizer_names if rid in id_sets[tok]]
                _drop(drop, key, 'missing_tokenizer_twin', present_in=present_tok)
                continue
            signatures = [per_tok[tok][rid] for tok in tokenizer_names]
            if any(not _same_twin_identity(signatures[0], identity) for identity in signatures[1:]):
                _drop(drop, key, 'tokenizer_twin_identity_mismatch',
                      tokenizer_text_sha256={tok: per_tok[tok][rid]['text_sha256'] for tok in tokenizer_names},
                      tokenizer_source_identity_sha256={tok: per_tok[tok][rid].get('source_identity_sha256')
                                                        for tok in tokenizer_names})
                continue
            identity = {**signatures[0], 'twin_text_sha256': tuple(
                (tok, per_tok[tok][rid]['text_sha256']) for tok in tokenizer_names)}
            pending_source_groups = sorted(set(identity['source_groups']) & source_review_groups)
            if identity['split'] == 'train' and pending_source_groups:
                _drop(drop, key, 'source_review_pending', source_groups=pending_source_groups,
                      text_sha256=identity['text_sha256'])
                continue
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
        candidates.sort(key=lambda candidate: member_rank[candidate['member']])
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
                held_observation = next(candidate for candidate in observations
                                        if candidate['identity']['split'] == 'test')
                for candidate in candidates:
                    _drop(drop, (candidate['member'], rid), 'conflicting_split_duplicate_id',
                          held_member=held_observation['member'],
                          held_text_sha256=held_observation['identity']['text_sha256'],
                          omitted_split=candidate['identity']['split'],
                          omitted_source_groups=list(candidate['identity']['source_groups']),
                          omitted_text_sha256=candidate['identity']['text_sha256'])
                for observation in observations:
                    if observation['identity']['split'] != 'test':
                        _drop(drop, (observation['member'], rid), 'conflicting_split_duplicate_id',
                              held_member=held_observation['member'],
                              held_text_sha256=held_observation['identity']['text_sha256'],
                              omitted_split=observation['identity']['split'],
                              omitted_source_groups=list(observation['identity']['source_groups']),
                              omitted_text_sha256=observation['identity']['text_sha256'])
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
            for observation in observations:
                if (observation['member'] != winner['member'] and
                        (observation['identity']['split'] != winner['identity']['split'] or
                         observation['identity']['source_groups'] != winner['identity']['source_groups'])):
                    _drop(drop, (observation['member'], rid), 'conflicting_split_duplicate_id',
                          kept_member=winner['member'], kept_split=winner['identity']['split'],
                          kept_source_groups=list(winner['identity']['source_groups']),
                          kept_text_sha256=winner['identity']['text_sha256'],
                          omitted_split=observation['identity']['split'],
                          omitted_source_groups=list(observation['identity']['source_groups']),
                          omitted_text_sha256=observation['identity']['text_sha256'])
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
    for rid, candidate in sorted(selected_by_id.items(), key=lambda item: member_rank[item[1]['member']]):
        key = (candidate['identity']['split'], candidate['identity']['twin_text_sha256'])
        if key in seen_text:
            kept_member, kept_id = seen_text[key]
            if kept_member in supersedes.get(candidate['member'], set()):
                selected.discard((kept_member, kept_id))
                _drop(drop, (kept_member, kept_id), 'superseded_exact_duplicate_document',
                      kept_member=candidate['member'], superseded_by=candidate['member'], kept_id=rid,
                      kept_text_sha256=candidate['identity']['text_sha256'],
                      omitted_text_sha256=candidate['identity']['text_sha256'])
                selected.add((candidate['member'], rid))
                seen_text[key] = (candidate['member'], rid)
            else:
                _drop(drop, (candidate['member'], rid), 'exact_duplicate_document',
                      kept_member=kept_member, kept_id=kept_id,
                      kept_text_sha256=candidate['identity']['text_sha256'],
                      omitted_text_sha256=candidate['identity']['text_sha256'])
                continue
        else:
            seen_text[key] = (candidate['member'], rid)
        selected.add((candidate['member'], rid))
    return selected, drop


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--compress-text', action='store_true', help='write verified zstd text directly, without raw intermediates')
    p.add_argument('--tokenizer', action='append', required=True, help='NAME=DIR')
    p.add_argument('--member', action='append', required=True, help='NAME:RENDERING:TOKENIZER=PATH')
    p.add_argument('--supersedes', action='append', default=[],
                   help='MEMBER=OLDER_MEMBER; only the named later member may replace the older member')
    p.add_argument('--cohort', action='append', default=[],
                   help='MEMBER=COHORT: every row of that member gets text_cohort COHORT (the trainer\'s cohort, e.g. '
                        'agentic-tool-loop for pi/harness/SWE sessions, upweighted by --cohort-repeats)')
    a = p.parse_args(argv)
    cohorts = {}
    for item in a.cohort:
        member, _, cohort = item.partition('=')
        if not member or not cohort:
            p.error(f'invalid --cohort {item!r}; expected MEMBER=COHORT')
        cohorts[member] = cohort
    t0 = time.time()
    source_review_groups = pending_unpinned_source_groups()
    source_review_sha256 = registry_sha256()
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
               'source_review': {'holds_sha256': source_review_sha256,
                                 'pending_unpinned_source_groups': len(source_review_groups),
                                 'match': 'exact original dataset:id source_group; train rows only'},
               'member_files': member_files,
               'member_cohorts': dict(sorted(cohorts.items())),
               'tokenizer_twin_identity': {'fields': list(IDENTITY_FIELDS + PROVENANCE_FIELDS),
                                           'rendered_text_may_differ': True,
                                           'exact_text_dedup': 'tuple of all tokenizer text hashes'},
               'member_supersedes': {rendering: {name: sorted(supersedes.get(name, set())) for name in names}
                                     for rendering, names in order.items()},
               'tokenizers': {tok: str(Path(d).resolve()) for tok, d in tokenizer_dirs.items()},
               'code': {str(f.relative_to(ROOT)): file_sha(f) for f in (
                   Path(__file__).resolve(), ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py',
                   ROOT / 'scripts/neuralese_data/text_tokenizer_twin.py',
                   ROOT / 'scripts/source_review.py')}}
    for rendering in members:
        # Pass 2: index compact row identities in both tokenizer files before resolving any duplicate. This prevents
        # an incomplete or held copy from claiming an ID/text hash before a later complete eligible twin is seen.
        member_rows, member_duplicates, member_duplicate_observations = {}, {}, {}
        for name in order[rendering]:
            per_tok, duplicates, repeated_rows = {}, set(), defaultdict(list)
            for tok, path in members[rendering][name].items():
                per_tok[tok] = {}
                for _, row in rows(path):
                    rid = row['id']
                    if rid in per_tok[tok]:
                        duplicates.add(rid)
                        repeated_rows[rid].append(_row_identity(row))
                    else:
                        per_tok[tok][rid] = _row_identity(row)
            member_rows[name] = per_tok
            member_duplicates[name] = duplicates
            member_duplicate_observations[name] = repeated_rows
        selected, drop = plan_member_rows(order[rendering], member_rows, supersedes,
                                          list(tokenizer_dirs), held_groups, held_texts,
                                          member_duplicates=member_duplicates,
                                          member_duplicate_observations=member_duplicate_observations,
                                          source_review_groups=source_review_groups)
        if not selected:
            raise ValueError(f'{rendering}: no eligible tokenizer twins; cannot publish an empty training union')
        out = a.out_dir / rendering
        out.mkdir(parents=True, exist_ok=True)
        rendering_receipt = {'members': order[rendering], 'by_tokenizer': {}}
        omissions = [{'member': n, 'id': i, **details} for (n, i), details in sorted(drop.items())]
        (out / 'omissions.jsonl').write_text(''.join(canonical(o) + '\n' for o in omissions))
        rendering_receipt['omissions'] = dict(Counter(f"{o['member']}: {o['reason']}" for o in omissions))
        rendering_receipt['duplicate_id_conflicts'] = [o for o in omissions
            if o['reason'] == 'conflicting_split_duplicate_id' or any(
                detail['reason'] == 'conflicting_split_duplicate_id'
                for detail in o.get('additional_reasons', []))]
        for tok in tokenizer_dirs:
            target = out / tok / ('text-data.jsonl.zst' if a.compress_text else 'text-data.jsonl')
            target.parent.mkdir(parents=True, exist_ok=True)
            stats = {'documents': 0, 'by_member': {}, 'by_split': {}}
            order_check = []
            with union_text_writer(target, a.compress_text) as dst:
                for name in order[rendering]:
                    m = stats['by_member'].setdefault(name, {'documents': 0, 'tokens': 0, 'supervised_tokens': 0,
                                                             'train_documents': 0, 'test_documents': 0})
                    for line, row in rows(members[rendering][name][tok]):
                        if (name, row['id']) not in selected:
                            continue
                        changed = False
                        if 'corpus' not in row:
                            row['corpus'] = name
                            changed = True
                        if name in cohorts and row.get('text_cohort') != cohorts[name]:
                            row['text_cohort'] = cohorts[name]
                            changed = True
                        if changed:
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
                        c = stats.setdefault('by_cohort', {}).setdefault(row.get('text_cohort', 'native'), {
                            'documents': 0, 'train_documents': 0, 'tokens': 0, 'supervised_tokens': 0})
                        c['documents'] += 1
                        c['train_documents'] += row['split'] == 'train'
                        c['tokens'] += n
                        c['supervised_tokens'] += supervised
                        stats['documents'] += 1
                        order_check.append((row['id'], row['split'], tuple(row['source_groups'])))
                        dst.write(line if line.endswith('\n') else line + '\n')
            stats['text_data_sha256'] = file_sha(target)
            if a.compress_text:
                stats.update(text_data_file=target.name, decoded_text_data_sha256=dst.decoded_sha256,
                             decoded_text_data_bytes=dst.decoded_bytes)
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
