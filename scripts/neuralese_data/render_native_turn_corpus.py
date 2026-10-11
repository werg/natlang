#!/usr/bin/env python3
"""Render an admitted natlang-native teacher-turn corpus for both students, in three renderings.

Input: ``natlang.teacher_training_turn.native/1`` rows (one decision per row: ``messages``, ``target``, ``tools``,
``teacher_reasoning``, ``split``, ``source_groups``, ``training_admission``). Output, per tokenizer (LFM2.5 and
Mellum2.1, the same documents in the same order):

1. ``full``: one document per approved decision, the serving prompt with the recorded system prompt and tool
   schemas, then the target reply (``native_gold_packet``; loss from ``supervised_suffix_start``). Warm-up phase.
2. ``program-header``: the same decisions rendered with ``prompt_rendering='program-header'`` (generic runtime
   prose, tool docs and the return_result schema dropped; program-specific paragraphs kept verbatim). Every row is
   also derived from its full row by ``strip_generic_system_prompt.strip_row`` (its byte and token-exactness checks)
   and the two must be identical; a mismatch is an omission, never silently kept.
3. ``whole-trajectory``: one document per maximal decision chain of one call invocation (program-header rendering),
   every assistant turn with its reasoning as the template's declared history-reasoning policy renders it; no
   ``supervised_suffix_start``, so the trainer supervises the whole document with its shared role weights (tool
   output at the feedback weight, ``text_supervision.TEXT_POSITION_WEIGHT_POLICY``). A chain is used only up to the
   first decision of its invocation that is not approved, so every assistant turn in it is an approved decision.

The target turn carries the teacher's reasoning (``teacher_reasoning``) unless the source marks it untrained
(``teacher_reasoning_trained: false``) or it is empty; history turns carry theirs as recorded.

Selection (each exclusion is an omission with its reason): ``training_admission.approved`` must be true; rows of
programs named in ``--exclude-programs`` are dropped (held cases, leak defects); with ``--protected`` a program whose
IR or messages contain a protected benchmark question (normalised hash of any string, or of any sentence-like line)
or a protected id is dropped; ``--held-text`` files (rendered text of other corpora) supply held source groups, and a
train document sharing one is dropped (source-group-disjoint decontamination). Exact duplicate documents are kept once.

Usage (CPU, natlang-neuralese image, under the DGX memory ledger):
  render_native_turn_corpus.py --corpus-id ID --source F [F...] --tokenizer lfm25=DIR --tokenizer mellum21=DIR \
      --out-dir OUT [--exclude-programs F] [--protected F] [--held-text F ...] [--workers N] [--shards N] [--limit N]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))

SCHEMA = 'natlang.native-turn-render/1'
RENDERINGS = ('full', 'program-header', 'whole-trajectory')
_G: dict = {}


from natlang_neuralese.common.hashing import sha256_file_hex as file_sha, sha256_hex as sha  # noqa: E402
from natlang_neuralese.common.jsonio import canonical_json_str as canonical  # noqa: E402


# ---------------------------------------------------------------- selection

def normalize_text(text: str) -> str:
    from records import normalize_text as norm  # scripts/neuralese_data/records.py: the protected-scan normaliser
    return norm(text)


def _strings(value, out):
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, list):
        for item in value:
            _strings(item, out)
    elif isinstance(value, dict):
        for item in value.values():
            _strings(item, out)
    return out


def protected_hits(row: dict, protected: dict) -> list[str]:
    """Protected benchmark items named by this row's program: a question (normalised hash of a whole string, a line
    or a sentence, the same normalisation as splits.protected_hit) or an upstream id."""
    hashes, ids = protected['question_hashes'], protected['ids']
    hits = []
    for value in [*(row.get('source_ids') or []), *(row.get('source_groups') or [])]:
        if isinstance(value, str) and value.split(':', 1)[-1] in ids:
            hits.append('id:' + value)
    texts = _strings([row.get('task'), [m.get('content') for m in row.get('messages') or [] if m.get('role') == 'user']], [])
    seen = set()
    for text in texts:
        pieces = [text, *text.split('\n')]
        for line in text.split('\n'):
            pieces += [s for s in re.split(r'(?<=[?.!])\s+', line) if s]
            if ':' in line:
                pieces.append(line.split(':', 1)[1])
            pieces.append(line.strip(' \t-*"\''))
        for piece in pieces:
            if len(piece) < 12 or len(piece) > 4000 or piece in seen:
                continue
            seen.add(piece)
            if sha(normalize_text(piece).encode()) in hashes:
                hits.append('question:' + piece[:120])
    return hits


def reserved_needles(path: Path) -> list[tuple[str, str]]:
    """Explicit test reservations (training/source-split-reservations.json): (source id, needle) with the audit's
    rule (plans/neuralese/JEFF_RESERVATION_OVERLAP_20261011.json): the reserved task's state text normalized as the
    reservation file declares, first 60 characters; states shorter than 40 characters are skipped."""
    reservations = json.loads(path.read_text())
    wanted = {sid for r in reservations['records'] for sid in r.get('source_ids') or []}
    source = ROOT / reservations['source']['path']
    needles = []
    for line in source.read_text().splitlines():
        task = json.loads(line)
        if task.get('id') in wanted and isinstance(task.get('state'), str):
            text = reserved_normalize(task['state'])
            if len(text) >= 40:
                needles.append((task['id'], text[:60]))
    return needles


def reserved_normalize(text: str) -> str:
    return re.sub(r'\s+', ' ', text.replace('\\', ' ')).strip()


_PREFIX = 16


def reserved_hit(row: dict, needles) -> str | None:
    """First reserved source id whose needle occurs in the row's normalized program text. Needles are indexed by
    their first 16 characters, so one pass over the text finds every candidate (the same substring rule)."""
    index = needles if isinstance(needles, dict) else None
    if index is None:
        raise TypeError('reserved needles must be indexed with index_needles')
    haystack = reserved_normalize(' '.join(_strings(row.get('task'), [])))
    for i in range(len(haystack) - _PREFIX + 1):
        for sid, needle in index.get(haystack[i:i + _PREFIX], ()):
            if haystack.startswith(needle, i):
                return sid
    return None


def index_needles(needles):
    index = {}
    for sid, needle in needles:
        index.setdefault(needle[:_PREFIX], []).append((sid, needle))
    return index


def program_key(row: dict) -> str:
    return (row.get('source_ref') or {}).get('program_ir_id') or row.get('program_id') or ''


def chain_key(row: dict) -> str:
    ref = row.get('source_ref') or {}
    return f"{ref.get('trajectory_id') or row.get('teacher_trajectory_id')}#{ref.get('invocation_id') or ''}"


def registered_source_pieces(sources):
    """Load exact piece tables listed beside the source in its immutable corpus manifest."""
    registry_path = ROOT / 'training/neuralese_corpora.json'
    registry = json.loads(registry_path.read_text())
    entries = registry.get('corpora', [])
    registered = {entry['id']: entry for entry in entries}
    if len(registered) != len(entries):
        raise ValueError('duplicate corpus IDs in training/neuralese_corpora.json')
    tables, bindings = {}, []
    for source in sources:
        source_arg = Path(source)
        logical_path = source_arg if source_arg.is_absolute() else ROOT / source_arg
        source_path = logical_path.resolve(strict=True)
        try:
            relative = logical_path.relative_to(ROOT).as_posix()
        except ValueError:
            relative = str(logical_path)
        source_bytes, source_hash = source_path.stat().st_size, file_sha(source_path)
        matches, listed_paths = [], []
        for corpus_id, entry in registered.items():
            corpus_root = ROOT / entry['path']
            real_corpus_root = corpus_root.resolve()
            possible_relatives = set()
            try:
                possible_relatives.add(logical_path.relative_to(corpus_root).as_posix())
            except ValueError:
                pass
            try:
                possible_relatives.add(source_path.relative_to(real_corpus_root).as_posix())
            except ValueError:
                pass
            if not possible_relatives:
                continue
            manifest_path = ROOT / 'training/corpus-manifests' / f'{corpus_id}.json'
            if not manifest_path.is_file():
                continue
            manifest = json.loads(manifest_path.read_text())
            if manifest.get('id') != corpus_id or manifest.get('path') != entry['path']:
                raise ValueError(f'registry/manifest identity mismatch for {corpus_id}')
            listed = {item['path']: item for item in manifest.get('files', [])}
            for rel_to_corpus in possible_relatives:
                source_item = listed.get(rel_to_corpus)
                if source_item:
                    listed_paths.append((corpus_id, rel_to_corpus))
                    if source_item.get('bytes') == source_bytes and source_item.get('sha256') == source_hash:
                        matches.append((corpus_id, entry, manifest_path, manifest, listed, rel_to_corpus))
        if not matches and listed_paths:
            raise ValueError(f'registered source bytes differ from immutable manifest: {relative}')
        if not matches:
            bindings.append({'source': relative, 'source_sha256': file_sha(source_path),
                             'registry_sha256': file_sha(registry_path), 'manifest': None,
                             'piece_table': None})
            continue
        declared = []
        for corpus_id, entry, manifest_path, manifest, listed, rel_to_corpus in matches:
            parent = Path(rel_to_corpus).parent
            candidates = [f'{parent}/{name}' if str(parent) != '.' else name
                          for name in ('native-pieces.jsonl', 'pieces.jsonl')]
            found = [name for name in candidates if name in listed]
            if len(found) > 1:
                raise ValueError(f'multiple registered piece tables beside {relative}: {found}')
            item = listed[found[0]] if found else None
            declared.append((corpus_id, entry, manifest_path, listed, rel_to_corpus,
                             found[0] if found else None, item))
        table_identities = {(item.get('sha256'), item.get('bytes')) if item else None
                            for _, _, _, _, _, _, item in declared}
        if len(table_identities) > 1:
            raise ValueError(f'registered source manifests disagree on its piece table: {relative}')
        # Prefer the explicitly training-admitted registration; equivalent duplicate registrations have the same
        # sidecar identity, so this only determines provenance naming in the receipt.
        declared.sort(key=lambda row: (not bool(row[1].get('training_admission')), row[0]))
        corpus_id, entry, manifest_path, listed, rel_to_corpus, piece_rel, item = declared[0]
        binding = {'source': relative, 'source_sha256': source_hash, 'corpus_id': corpus_id,
                   'also_registered_as': [row[0] for row in declared[1:]],
                   'registry_sha256': file_sha(registry_path),
                   'manifest': str(manifest_path.relative_to(ROOT)), 'manifest_sha256': file_sha(manifest_path),
                   'piece_table': None}
        if piece_rel:
            piece_path = ROOT / entry['path'] / piece_rel
            if piece_path.stat().st_size != item.get('bytes') or file_sha(piece_path) != item.get('sha256'):
                raise ValueError(f'registered piece table hash mismatch: {piece_rel}')
            binding['piece_table'] = {'path': (Path(entry['path']) / piece_rel).as_posix(),
                                      'bytes': item['bytes'], 'sha256': item['sha256']}
            with piece_path.open() as f:
                for line_number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    name, text = row.get('name'), row.get('text')
                    if not isinstance(name, str) or not isinstance(text, str):
                        raise ValueError(f'invalid piece row {piece_rel}:{line_number}')
                    if name in tables and tables[name] != text:
                        raise ValueError(f'conflicting registered piece body: {name}')
                    tables[name] = text
        bindings.append(binding)
    return tables, bindings


def select(args, shard_dir: Path):
    """Stream the sources; write selected raw lines to trajectory shards; return omissions and blocked chains."""
    exclude = {}
    for path in args.exclude_programs or []:
        for line in path.read_text().splitlines():
            if line.strip():
                item = json.loads(line)
                exclude.setdefault(item['id'], item.get('reason_code') or 'held_case_excluded')
    needles = index_needles(reserved_needles(args.reserved)) if args.reserved else None
    reserved_cache, reserved_groups = {}, set()
    protected = json.loads(args.protected.read_text()) if args.protected else None
    shard_files = [open(shard_dir / f'{i:04d}.jsonl', 'wb') for i in range(args.shards)]
    omissions, counts = [], Counter()
    blocked = {}  # chain -> fewest messages of an unselected decision in it
    protected_cache = {}
    seen_ids = set()
    test_groups = set()
    n = 0
    for source in args.source:
        with open(source, 'rb') as f:
            for line in f:
                if not line.strip():
                    continue
                if args.limit and n >= args.limit:
                    break
                n += 1
                row = json.loads(line)
                rid = row.get('id')
                counts['source_rows'] += 1
                reason = None
                if row.get('version') != 'natlang.teacher_training_turn.native/1':
                    reason = 'not_a_native_teacher_turn'
                elif rid in seen_ids:
                    reason = 'duplicate_source_row_id'
                elif not (isinstance(row.get('training_admission'), dict)
                          and row['training_admission'].get('approved') is True):
                    reason = 'not_approved:' + str((row.get('training_admission') or {}).get('reason') or 'approved=false')[:80]
                elif {program_key(row), row.get('program_id'), *(row.get('source_program_ids') or [])} & exclude.keys():
                    hit = sorted({program_key(row), row.get('program_id'), *(row.get('source_program_ids') or [])}
                                 & exclude.keys())[0]
                    reason = exclude[hit]
                    if reason == 'jeff_reserved_test_text' and row.get('split') == 'train':
                        reserved_groups.update(row.get('source_groups') or [])
                elif needles is not None and row.get('split') == 'train' and (
                        reserved_cache.setdefault(program_key(row), reserved_hit(row, needles))):
                    reason = 'jeff_reserved_test_text:' + reserved_cache[program_key(row)]
                    reserved_groups.update(row.get('source_groups') or [])
                elif protected is not None:
                    key = program_key(row)
                    if key not in protected_cache:
                        protected_cache[key] = protected_hits(row, protected)
                    if protected_cache[key]:
                        reason = 'protected_benchmark_overlap:' + protected_cache[key][0][:120]
                if reason is not None:
                    omissions.append({'id': rid, 'program': program_key(row), 'split': row.get('split'),
                                      'reason': reason, 'stage': 'selection'})
                    counts['omitted:' + reason.split(':', 1)[0]] += 1
                    chain = chain_key(row)
                    blocked[chain] = min(blocked.get(chain, 1 << 30), len(row.get('messages') or []))
                    continue
                seen_ids.add(rid)
                if row.get('split') == 'test':
                    test_groups.update(g for g in row.get('source_groups') or [] if isinstance(g, str))
                counts['selected'] += 1
                counts['selected:' + str(row.get('split'))] += 1
                shard = int(sha(str(row.get('teacher_trajectory_id') or chain_key(row)).encode())[:8], 16) % args.shards
                shard_files[shard].write(line.rstrip(b'\n') + b'\n')
    for f in shard_files:
        f.close()
    counts['reserved_programs_hit'] = sum(bool(v) for v in reserved_cache.values())
    counts['reserved_groups'] = len(reserved_groups)
    return omissions, counts, blocked, protected_cache, test_groups, reserved_groups


# ---------------------------------------------------------------- rendering (workers)

def _init(tokenizers, pieces):
    from build_neuralese_gold_text_delta import load_pinned_tokenizer
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from strip_generic_system_prompt import strip_row
    _G['strip_row'] = strip_row
    _G['pieces'] = pieces
    _G['tok'] = {}
    for name, path in tokenizers.items():
        tok = load_pinned_tokenizer(Path(path))
        bind_history_reasoning(tok, require_declared=True)
        _G['tok'][name] = tok


def prepare_record(row: dict) -> dict:
    target = dict(row['target'])
    reasoning = row.get('teacher_reasoning')
    if (isinstance(reasoning, str) and reasoning.strip() and row.get('teacher_reasoning_trained') is not False
            and not target.get('reasoning_content') and '</think>' not in str(target.get('content') or '')):
        target['reasoning_content'] = reasoning
    return {**row, 'target': target}


def _message_digest(message) -> str:
    return sha(canonical(message).encode())


def whole_trajectory_records(records: list[dict], blocked: dict) -> list[tuple[dict, list[str]]]:
    """Maximal clean decision chains: (record whose messages+target is the document, ids of decisions it covers)."""
    groups = defaultdict(list)
    for r in records:
        groups[chain_key(r)].append(r)
    out = []
    for chain, rows in groups.items():
        limit = blocked.get(chain, 1 << 30)
        clean = sorted((r for r in rows if len(r.get('messages') or []) < limit),
                       key=lambda r: len(r.get('messages') or []))
        digests = [[_message_digest(m) for m in r.get('messages') or []] for r in clean]
        covered_by = {}
        for i, r in enumerate(clean):
            for j in range(len(clean) - 1, i, -1):
                if len(digests[j]) > len(digests[i]) and digests[j][:len(digests[i])] == digests[i]:
                    covered_by[i] = j
                    break
        for i, r in enumerate(clean):
            if i in covered_by:
                continue
            members = [clean[k]['id'] for k in range(len(clean))
                       if k == i or (len(digests[k]) < len(digests[i]) and digests[i][:len(digests[k])] == digests[k])]
            out.append((r, members))
    return out


def render_shard(job):
    shard_path, out_dir, tok_name, blocked, corpus, held_groups = job
    from natlang_neuralese.data.text_corpus import (authenticated_context_writer_sources,
                                                    authenticated_crisp_context_messages, gold_text_rows,
                                                    native_gold_document, tokenizer_fingerprint)
    from natlang_neuralese.train.trajectories import crisp_messages, handover_notes
    tok = _G['tok'][tok_name]
    records = []
    with open(shard_path, 'rb') as f:
        for line in f:
            row = json.loads(line)
            row['_source_record_sha256'] = sha(line.rstrip(b'\n'))
            records.append(prepare_record(row))
    stem = Path(shard_path).stem
    result = {'shard': stem, 'tokenizer': tok_name, 'records': len(records)}
    omissions = []
    # Source-group-disjoint decontamination: a train decision sharing a source group with any held document (this
    # corpus's test rows or another corpus's held set) is dropped; held rows never move.
    kept = []
    for r in records:
        hit = sorted(set(r.get('source_groups') or []) & held_groups) if r.get('split') == 'train' else []
        if hit:
            omissions.append({'id': r['id'], 'rendering': 'all', 'stage': 'decontamination',
                              'reason': 'train_decision_shares_held_source_group', 'group': hit[0]})
        else:
            kept.append(r)
    records = kept
    result['decontaminated'] = result['records'] - len(records)
    if not records:
        write_done(Path(out_dir), tok_name, stem, result)
        return result
    rows = {}
    for mode in ('full', 'program-header'):
        r, receipt, omitted, provenance = gold_text_rows(records, _G['pieces'], tokenizer=tok, require_independent_splits=False,
                                                         prompt_rendering=mode)
        for row, prov in zip(r, provenance):
            row['corpus'] = corpus
            row['prompt_rendering'] = mode
        rows[mode] = r
        omissions += [{**o, 'rendering': mode, 'stage': 'render'} for o in omitted]
        result['receipt_' + mode] = {k: receipt[k] for k in ('format', 'rendering', 'history_reasoning', 'supervision',
                                                             'documents', 'omitted_records',
                                                             'duplicate_same_split_documents_deduplicated',
                                                             'excluded_train_exact_held_complete_documents')}
    # Program-header parity: the renderer's rows must equal the strip script's derivation from the full rows.
    im_start, im_end = tok.convert_tokens_to_ids('<|im_start|>'), tok.convert_tokens_to_ids('<|im_end|>')
    by_id = {row['id']: row for row in rows['program-header']}
    parity = Counter()
    kept_ph = []
    for full in rows['full']:
        ph = by_id.get(full['id'])
        if ph is None:
            parity['program_header_missing'] += 1
            continue
        stripped, reason, info = _G['strip_row'](full, tok, im_start, im_end)
        if stripped is None:
            parity['strip_failed'] += 1
            omissions.append({'id': full['id'], 'rendering': 'program-header', 'stage': 'parity', 'reason': 'strip_failed: ' + reason})
            continue
        parity['retokenization_checked'] += bool(info['retokenization_checked'])
        if (stripped['text'], stripped['token_ids'], stripped['supervised_suffix_start']) != (
                ph['text'], ph['token_ids'], ph['supervised_suffix_start']):
            parity['mismatch'] += 1
            omissions.append({'id': full['id'], 'rendering': 'program-header', 'stage': 'parity',
                              'reason': 'renderer_and_strip_script_differ'})
            continue
        parity['equal'] += 1
        ph['system_prompt_mode'] = stripped['system_prompt_mode']
        kept_ph.append(ph)
    rows['program-header'] = kept_ph
    result['parity'] = dict(parity)
    # Whole trajectory.
    fingerprint = tokenizer_fingerprint(tok)
    source_hashes, writer_sources = authenticated_context_writer_sources(records)
    wt = []
    for record, members in whole_trajectory_records(records, blocked):
        try:
            messages, _, _ = authenticated_crisp_context_messages(record, _G['pieces'], writer_sources)
            target = crisp_messages([record['target']], _G['pieces'], handover_notes(record))[0]
            text, ids = native_gold_document(tok, messages, target, record.get('tools') or [],
                                             prompt_rendering='program-header')
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            omissions.append({'id': record['id'], 'rendering': 'whole-trajectory', 'stage': 'render',
                              'reason': 'native_chat_render_failed', 'detail': str(exc)[:240]})
            continue
        opening = 0  # leading assistant turns the runtime ran itself (the opening eval that declares the inputs)
        for m in messages:
            if m.get('role') == 'assistant':
                if any(c.get('id') == 'scope_0' for c in m.get('tool_calls') or []):
                    opening += 1
                    continue
                break
        wt.append({'text': text, 'token_ids': ids, 'tokenizer_sha256': fingerprint, 'split': record['split'],
                   'runtime_opening_assistant_turns': opening,
                   'source_groups': sorted(set(g for g in record.get('source_groups') or [] if isinstance(g, str) and g)),
                   'id': 'whole-trajectory:' + record['id'], 'source_record_ids': members,
                   'source_records': [{'id': m, 'sha256': source_hashes[m]} for m in members],
                   'corpus': corpus, 'prompt_rendering': 'program-header', 'supervision': 'whole-trajectory'})
    rows['whole-trajectory'] = wt
    for mode, rs in rows.items():
        path = Path(out_dir) / mode / tok_name / f'{stem}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            for row in rs:
                f.write(canonical(row) + '\n')
        result[mode] = len(rs)
    path = Path(out_dir) / 'omissions' / tok_name / f'{stem}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(canonical(o) + '\n' for o in omissions))
    write_done(Path(out_dir), tok_name, stem, result)
    return result


def write_done(work: Path, tok_name: str, stem: str, result: dict):
    """Resume marker, written last and atomically: a job with a marker is never rendered again."""
    done = work / 'done' / tok_name / f'{stem}.json'
    done.parent.mkdir(parents=True, exist_ok=True)
    tmp = done.with_suffix('.tmp')
    tmp.write_text(json.dumps(result))
    os.replace(tmp, done)


# ---------------------------------------------------------------- merge

def role_counter(tok):
    from natlang_neuralese.train.text_warmup import chat_roles
    from natlang_neuralese.train.text_supervision import ROLE_CODES

    def token_id(text):
        value = tok.convert_tokens_to_ids(text)
        return value if isinstance(value, int) and value != tok.unk_token_id else None
    start = token_id('<|im_start|>')
    role_ids = {token_id(name): name for name in ('system', 'user', 'assistant', 'tool') if token_id(name) is not None}
    prefixes = tuple(tuple(tok.encode(p, add_special_tokens=False)) for p in ('\n<tool_response>', '\n\n<tool_response>'))

    def count(ids):
        codes = chat_roles(ids, start_id=start, role_ids=role_ids, think_open=token_id('<think>'),
                           think_close=token_id('</think>'), tool_response_prefixes=prefixes)
        c = Counter(ROLE_CODES[x] for x in codes)
        return c, codes
    return count


def whole_trajectory_target_tokens(row, count_roles):
    """(decision target tokens, runtime-opening tokens, role counter) of a whole-trajectory document: its assistant
    turns (weight 1) less the leading turns the runtime ran itself (the opening eval: inputs, supervised as context
    like the user turn)."""
    c, codes = count_roles(row['token_ids'])
    runs, inside = [], False
    for code in codes:
        is_assistant = code in (4, 5)  # ROLE_CODES assistant_reasoning, assistant_reply
        if is_assistant and not inside:
            runs.append(0)
        if is_assistant:
            runs[-1] += 1
        inside = is_assistant
    skip = row.get('runtime_opening_assistant_turns', 0)
    return sum(runs[skip:]), sum(runs[:skip]), c


def held_groups_from(paths):
    groups = set()
    for path in paths or []:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get('split') == 'test':
                    groups.update(row.get('source_groups') or [])
    return groups


def merge(out: Path, mode: str, tok_name: str, shards: list[str], held_external: set, count_roles):
    """Concatenate shard outputs in shard order with the global split and dedup checks of gold_text_rows."""
    src = [out / 'work' / mode / tok_name / f'{s}.jsonl' for s in shards]
    src = [p for p in src if p.exists()]
    train_groups, test_groups, test_texts = set(), set(), set()
    for p in src:
        with open(p) as f:
            for line in f:
                row = json.loads(line)
                (train_groups if row['split'] == 'train' else test_groups).update(row['source_groups'])
                if row['split'] == 'test':
                    test_texts.add(sha(row['text'].encode()))
    overlap = train_groups & test_groups
    held = test_groups | held_external
    final = out / mode / tok_name
    final.mkdir(parents=True, exist_ok=True)
    omissions, seen = [], set()
    stats = {'documents': 0, 'by_split': {}, 'max_document_tokens': 0}
    roles_total = Counter()
    with open(final / 'text-data.jsonl', 'w') as dst:
        for p in src:
            with open(p) as f:
                for line in f:
                    row = json.loads(line)
                    if row['split'] == 'train':
                        hit = sorted(set(row['source_groups']) & held)
                        if hit:
                            omissions.append({'id': row['id'], 'reason': 'train_document_shares_held_source_group',
                                              'group': hit[0], 'internal': hit[0] in test_groups})
                            continue
                        if sha(row['text'].encode()) in test_texts:
                            omissions.append({'id': row['id'], 'reason': 'train_copy_of_held_document'})
                            continue
                    key = (row['split'], sha(row['text'].encode()), sha(canonical(row['token_ids']).encode()),
                           row.get('supervised_suffix_start'))
                    if key in seen:
                        omissions.append({'id': row['id'], 'reason': 'exact_duplicate_document'})
                        continue
                    seen.add(key)
                    n = len(row['token_ids'])
                    s = stats['by_split'].setdefault(row['split'], {'documents': 0, 'tokens': 0, 'supervised_tokens': 0,
                                                                   'assistant_tokens': 0, 'tool_tokens': 0})
                    s['documents'] += 1
                    s['tokens'] += n
                    if row.get('supervised_suffix_start') is not None:
                        s['supervised_tokens'] += n - row['supervised_suffix_start']
                    else:
                        target, opening, c = whole_trajectory_target_tokens(row, count_roles)
                        roles_total.update(c)
                        s['assistant_tokens'] += c['assistant_reasoning'] + c['assistant_reply']
                        s['tool_tokens'] += c['tool']
                        s['opening_tokens'] = s.get('opening_tokens', 0) + opening
                        s['supervised_tokens'] += target
                    stats['documents'] += 1
                    stats['max_document_tokens'] = max(stats['max_document_tokens'], n)
                    dst.write(line if line.endswith('\n') else line + '\n')
    stats['roles'] = dict(roles_total)
    stats['internal_train_test_group_overlap'] = len(overlap)
    (final / 'omissions.jsonl').write_text(''.join(canonical(o) + '\n' for o in omissions))
    stats['merge_omissions'] = dict(Counter(o['reason'] for o in omissions))
    stats['text_data_sha256'] = file_sha(final / 'text-data.jsonl')
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus-id', required=True)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--tokenizer', action='append', required=True, help='NAME=DIR')
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--exclude-programs', type=Path, nargs='*', default=[],
                   help='jsonl rows {id, reason_code?}: programs whose rows are dropped')
    p.add_argument('--reserved', type=Path, help='explicit test reservations (training/source-split-reservations.json)')
    p.add_argument('--protected', type=Path)
    p.add_argument('--held-text', type=Path, nargs='*', default=[])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--shards', type=int, default=64)
    p.add_argument('--limit', type=int)
    p.add_argument('--keep-work', action='store_true', help='keep the per-shard work files after the merge')
    a = p.parse_args(argv)
    tokenizers = dict(item.split('=', 1) for item in a.tokenizer)
    pieces, piece_bindings = registered_source_pieces(a.source)
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'receipt.json').exists():
        p.error(f'refusing to overwrite a finished render: {out}')
    t0 = time.time()
    shard_dir = out / 'work' / 'selected'
    state_path = out / 'work' / 'selection-state.json'
    if state_path.exists():  # resume: selection is deterministic and already done
        state = json.loads(state_path.read_text())
        omissions, counts, blocked = state['omissions'], Counter(state['counts']), state['blocked']
        protected_cache, test_groups, reserved_groups = state['protected'], set(state['test_groups']), set(state['reserved_groups'])
        print(json.dumps({'resumed_selection': True}), flush=True)
    else:
        shutil.rmtree(out / 'work', ignore_errors=True)
        shard_dir.mkdir(parents=True, exist_ok=True)
        omissions, counts, blocked, protected_cache, test_groups, reserved_groups = select(a, shard_dir)
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'omissions': omissions, 'counts': counts, 'blocked': blocked,
                                   'protected': protected_cache, 'test_groups': sorted(test_groups),
                                   'reserved_groups': sorted(reserved_groups)}))
        os.replace(tmp, state_path)
    held_external = held_groups_from(a.held_text)
    # Train decisions sharing a source group with a reserved-test hit go with it (whole source groups, every split's
    # program of that group): they are decontaminated exactly like held groups.
    held_groups = test_groups | held_external | reserved_groups
    print(json.dumps({'selection': counts, 'internal_held_groups': len(test_groups),
                      'external_held_groups': len(held_external), 'seconds': round(time.time() - t0)}), flush=True)
    shards = sorted(f.stem for f in shard_dir.glob('*.jsonl'))
    jobs = [(str(shard_dir / f'{s}.jsonl'), str(out / 'work'), name, blocked, a.corpus_id, held_groups)
            for s in shards for name in tokenizers]
    results = []
    pending = []
    for job in jobs:
        done = out / 'work' / 'done' / job[2] / f'{Path(job[0]).stem}.json'
        if done.exists():
            results.append(json.loads(done.read_text()))
        else:
            pending.append(job)
    print(json.dumps({'jobs': len(jobs), 'already_done': len(results)}), flush=True)
    jobs = pending
    with Pool(a.workers, initializer=_init, initargs=(tokenizers, pieces)) as pool:
        for i, res in enumerate(pool.imap_unordered(render_shard, jobs)):
            results.append(res)
            if i % 8 == 0:
                print(json.dumps({'done': i + 1, 'of': len(jobs), 'last': res.get('shard'),
                                  'seconds': round(time.time() - t0)}), flush=True)
    _init(tokenizers, pieces)
    renders = {}
    for mode in RENDERINGS:
        for name in tokenizers:
            renders[f'{mode}/{name}'] = merge(out, mode, name, shards, held_external,
                                              role_counter(_G['tok'][name]))
            print(json.dumps({mode: name, **{k: v for k, v in renders[f'{mode}/{name}'].items() if k != 'roles'}}), flush=True)
    render_omissions = Counter()
    parity = Counter()
    for res in results:
        parity.update(res.get('parity') or {})
        render_omissions[f"{res['tokenizer']}/all/decontaminated"] += res.get('decontaminated', 0)
    for name in tokenizers:
        for f in sorted((out / 'work' / 'omissions' / name).glob('*.jsonl')):
            for line in f.read_text().splitlines():
                o = json.loads(line)
                render_omissions[f"{name}/{o.get('rendering')}/{o['reason'].split(':', 1)[0]}"] += 1
    with open(out / 'selection-omissions.jsonl', 'w') as f:
        for o in omissions:
            f.write(canonical(o) + '\n')
    receipts = {k: v for res in results for k, v in res.items() if k.startswith('receipt_')}
    receipt = {
        'schema': SCHEMA, 'corpus_id': a.corpus_id,
        'sources': {str(s): {'sha256': file_sha(s), 'bytes': s.stat().st_size} for s in a.source},
        'source_piece_bindings': piece_bindings,
        'source_piece_entries': len(pieces),
        'source_piece_table_sha256': sha(canonical(pieces).encode()) if pieces else None,
        'exclude_programs': [{'path': str(e), 'sha256': file_sha(e)} for e in a.exclude_programs],
        'reserved': ({'path': str(a.reserved), 'sha256': file_sha(a.reserved)} if a.reserved else None),
        'reserved_groups': len(reserved_groups),
        'protected': ({'path': str(a.protected), 'sha256': file_sha(a.protected),
                       'programs_checked': len(protected_cache),
                       'programs_hit': sum(bool(v) for v in protected_cache.values())} if a.protected else None),
        'held_text': {str(h): file_sha(h) for h in a.held_text},
        'held_external_groups': len(held_external),
        'tokenizers': {name: {'path': str(Path(path).resolve()),
                              'files': {f: file_sha(Path(path) / f) for f in ('tokenizer.json', 'tokenizer_config.json',
                                                                              'chat_template.jinja')
                                        if (Path(path) / f).is_file()},
                              'history_reasoning': _G['tok'][name].natlang_history_reasoning}
                       for name, path in tokenizers.items()},
        'renderer_code': {str(path.relative_to(ROOT)): file_sha(path) for path in (
            Path(__file__).resolve(), ROOT / 'scripts/neuralese_data/strip_generic_system_prompt.py',
            ROOT / 'training/neuralese/natlang_neuralese/data/text_corpus.py',
            ROOT / 'training/neuralese/natlang_neuralese/data/prompt_rendering.py',
            ROOT / 'training/neuralese/natlang_neuralese/serve/chat.py')},
        'selection': dict(counts),
        'render_omissions': dict(render_omissions),
        'program_header_parity': dict(parity),
        'renders': renders,
        'shard_receipt_sample': receipts,
        'target_reasoning': 'teacher_reasoning on the target turn unless teacher_reasoning_trained is false or empty',
        'whole_trajectory': 'one document per maximal clean decision chain of one invocation (program-header); no '
                            'supervised_suffix_start: role-weighted whole-document supervision (tool output at the '
                            'shared feedback weight); assistant tokens counted as supervised target tokens',
        'seconds': round(time.time() - t0),
    }
    for name in tokenizers:  # render omissions are kept; the per-shard documents are merged
        dst = out / 'render-omissions' / name
        dst.mkdir(parents=True, exist_ok=True)
        with open(dst / 'omissions.jsonl', 'w') as f:
            for src in sorted((out / 'work' / 'omissions' / name).glob('*.jsonl')):
                f.write(src.read_text())
    if not a.keep_work:
        shutil.rmtree(out / 'work')
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=False) + '\n')
    print(json.dumps({'renders': {k: {'documents': v['documents'], 'by_split': v['by_split']} for k, v in renders.items()},
                      'parity': parity, 'render_omissions': render_omissions}, indent=1))


if __name__ == '__main__':
    main()
