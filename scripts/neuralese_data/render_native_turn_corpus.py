#!/usr/bin/env python3
"""Render an admitted natlang-native teacher-turn corpus for both students, in three renderings.

Input: ``natlang.teacher_training_turn.native/1`` rows (one decision per row: ``messages``, ``target``, ``tools``,
``teacher_reasoning``, ``split``, ``source_groups``, ``training_admission``). Output, per tokenizer (LFM2.5 and
Mellum2.1, the same documents in the same order):

1. ``full``: one document per approved decision, the serving prompt with the recorded system prompt and tool
   schemas, then the target reply (``native_gold_packet``; loss from ``supervised_suffix_start``). Warm-up phase.
   Kept only for the warm-up sample (``--full-sample``, default 5% of programs, stratified by family; shared rule,
   coordinator 2026-10-11): the generic prompt is dropped after warm-up, so the rest of a corpus never needs it.
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

Selection: a row whose split is validation, valid, val or dev is rendered as test (held; source_split keeps the name;
counted as split_mapped:NAME->test). Each exclusion is an omission with its reason: ``training_admission.approved`` must be true; rows of
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
import math
import json
import os
import re
import shutil
import subprocess
import sys
import time
import inspect
import sqlite3
import tempfile
from collections import Counter, defaultdict
from contextlib import ExitStack
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))

SCHEMA = 'natlang.native-turn-render/1'
RENDERINGS = ('full', 'program-header', 'whole-trajectory')
# The Search/OWL selection checkpoint was committed while selection and
# rendering shared one source file. Accept this exact legacy file binding once
# the selector inputs below validate byte-for-byte; render-only code changes
# must not force a 658k-row reselection.
LEGACY_SELECTOR_FILE_SHA256 = {'797e534147c258b5453c146365e53763133ac45ec468193b66fba442419bdccf'}
# The one legacy full-file selector hash above may migrate only while the
# current selector implementation still has this independently reviewed
# semantic fingerprint. This prevents the legacy file allowlist from silently
# authorizing a later selector-policy change.
LEGACY_SELECTOR_SEMANTICS_SHA256 = '69f7da7f1dc08c01ca9deb554f0275072c82e2666615d60a166e9c105eeb804b'
# One explicit migration witness for the active Search/OWL v3 renderer. Its
# historic bindings used canonical-checkout path labels even when the new
# renderer is run from a frozen checkout elsewhere.
LEGACY_SEARCH_RENDERER_SHA256 = 'b054133ae53e7d86a9202ad84e60cabed1f79a26d18aa588e3890c2c21249917'
# The exact renderer immediately before the reviewed 2026-10-17 resource-only
# fix. The fix closes SQLite before temporary-directory cleanup and bounds
# future submission; it does not alter selected rows, rendering, or outputs.
SEARCH_RESOURCE_ONLY_PARENT_RENDERER_SHA256 = '6b59b92914a93139215d7a22f0129d59ade36b533a28126cd5e927dbc23fbc5b'
from natlang_neuralese.common.paths import DEFAULTS as MACHINE_PATH_DEFAULTS  # noqa: E402

_LEGACY_REPO = Path(MACHINE_PATH_DEFAULTS['repo'])
LEGACY_SEARCH_CODE_BINDING = {
    str(_LEGACY_REPO / 'scripts/neuralese_data/render_native_turn_corpus.py'):
        LEGACY_SEARCH_RENDERER_SHA256,
    str(_LEGACY_REPO / 'training/neuralese/natlang_neuralese/data/text_corpus.py'):
        '004659afb57d4c23dd64262ed49ca4b3964938b60e701789a61b63091464773a',
    str(_LEGACY_REPO / 'scripts/neuralese_data/strip_generic_system_prompt.py'):
        'ebedd490684608188473f19eaa214d7be39dbf762a57d20e8955893741727838',
}
# Exact, reviewed semantic transition for the profile-aware shared text renderer.
# An unconfigured/default-profile load still invokes bind_history_reasoning()
# with the same backbone declaration; only a tokenizer carrying an explicit
# checkpoint profile takes the new path. Legacy Search rows can therefore reuse
# their old receipts only under the exact current file bytes below and only for
# the default profile. This is not a general dependency waiver.
LEGACY_SEARCH_TEXT_CORPUS_DEFAULT_PROFILE_TRANSITION = {
    'historical_sha256': '004659afb57d4c23dd64262ed49ca4b3964938b60e701789a61b63091464773a',
    'current_sha256': '622b4ad08257ed09d0dd10d52d97adfa1c45cc8b9c8c46bbeef57fae6caff63b',
}
# Upstream held-out split names rendered as test (recorded as selection counts split_mapped:<name>->test and as each
# row's source_split). Any other split other than train/test stays an omission with its reason.
HELD_SPLIT_ALIASES = ('validation', 'valid', 'val', 'dev')
_G: dict = {}


from natlang_neuralese.common.hashing import sha256_file_hex as file_sha, sha256_hex as sha  # noqa: E402
from natlang_neuralese.common.jsonio import (canonical_json_str as canonical, open_jsonl_text,
                                              write_json_durable, write_jsonl_text)  # noqa: E402
from natlang_neuralese.common.resumable_jsonl_frames import ResumableJsonlFrames  # noqa: E402
from scripts.neuralese_data.conversion_output_lease import acquire_output_lease  # noqa: E402


def jsonl_lines(path: Path):
    """Read LF-delimited records; Unicode U+2028 inside JSON strings is data."""
    with open_jsonl_text(path) as stream:
        yield from stream


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
    for line in jsonl_lines(source):
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
    # The program IR (task) and the user turns: a row without a program IR carries its inputs in the user turn.
    haystack = reserved_normalize(' '.join(_strings([row.get('task'), [m.get('content') for m in row.get('messages') or []
                                                                         if m.get('role') == 'user']], [])))
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
    return (row.get('source_ref') or {}).get('program_ir_id') or row.get('program_id') or row.get('id') or ''


def chain_key(row: dict) -> str:
    ref = row.get('source_ref') or {}
    return f"{ref.get('trajectory_id') or row.get('teacher_trajectory_id')}#{ref.get('invocation_id') or ''}"


def split_chain_key(row: dict) -> str:
    return canonical([chain_key(row), row.get('split')])


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


def candidate_selection_correction_ref(packet, admission):
    """Keep the packet and registry correction declaration in exact agreement."""
    packet_ref = packet.get('selection_review_pin_correction')
    if packet_ref is not None and not isinstance(packet_ref, dict):
        raise ValueError('registry candidate-selection correction reference is malformed')
    packet_sha = packet_ref.get('sha256') if isinstance(packet_ref, dict) else None
    registry_sha = admission.get('selection_review_pin_correction_sha256')
    if packet_sha != registry_sha:
        raise ValueError('registry candidate-selection correction pin mismatch')
    if packet_ref is None:
        if packet_sha is not None:
            raise ValueError('registry candidate-selection correction pin mismatch')
        return None
    if (not isinstance(packet_ref.get('path'), str) or not packet_ref['path'] or
            not isinstance(packet_sha, str) or not packet_sha):
        raise ValueError('registry candidate-selection correction reference is malformed')
    return packet_ref


def candidate_selection_owner_item_sha(owner_row, verified_item_sha):
    """Validate a supplied item digest; older receipts omit it under pinned review/registry bindings."""
    if 'item_sha256' in owner_row and owner_row['item_sha256'] != verified_item_sha:
        raise ValueError('root candidate-selection owner item hash differs from verified packet')
    return owner_row.get('item_sha256', verified_item_sha)


def registered_candidate_selection_admission(corpus_id, sources):
    """Verify a registry owner's exact admission of a conversion-only candidate packet.

    The packet bytes remain candidate-selection metadata. This returns the exact ID/target/source map which the
    renderer may promote in memory only because the registry owner admitted this corpus.
    """
    registry_path = ROOT / 'training/neuralese_corpora.json'
    registry = json.loads(registry_path.read_text())
    entries = [entry for entry in registry.get('corpora', []) if entry.get('id') == corpus_id]
    if not entries:
        # A corpus id with no registry entry (a mixed-source render of several registered sources) has no
        # candidate-selection admission to verify; its rows go through the ordinary admission checks.
        return None
    if len(entries) != 1:
        raise ValueError(f'candidate-selection admission needs one registry entry: {corpus_id}')
    entry = entries[0]
    admission = entry.get('candidate_selection_admission')
    if admission is None:
        return None
    if (entry.get('training_admission') is not True or entry.get('trace_admission') is True or
            admission.get('schema') != 'natlang.registry-candidate-selection-admission/1' or
            admission.get('decision') != 'admit-reviewed-candidate-selection' or
            admission.get('training_admission') is not True or admission.get('trace_admission') is not False):
        raise ValueError(f'candidate-selection registry admission is malformed or out of scope: {corpus_id}')
    corpus_root = (ROOT / entry['path']).resolve(strict=True)
    if any(path.resolve(strict=True).parent != corpus_root for path in sources):
        raise ValueError('candidate-selection admission must render directly from the registered packet files')
    registry_manifest_path = ROOT / 'training/corpus-manifests' / f'{corpus_id}.json'
    registry_manifest = json.loads(registry_manifest_path.read_text())
    if registry_manifest.get('id') != corpus_id or registry_manifest.get('path') != entry['path']:
        raise ValueError('candidate-selection registry manifest identity mismatch')
    registered = {item['path']: item for item in registry_manifest.get('files', [])}

    def registered_bytes(name):
        item = registered.get(name)
        path = corpus_root / name
        if not item or not path.is_file() or path.stat().st_size != item.get('bytes') or file_sha(path) != item.get('sha256'):
            raise ValueError(f'candidate-selection file differs from its immutable corpus manifest: {name}')
        return path.read_bytes()

    packet_bytes = registered_bytes('materialized-manifest.json')
    packet_sha = file_sha(corpus_root / 'materialized-manifest.json')
    if packet_sha != admission.get('packet_manifest_sha256'):
        raise ValueError('registry candidate-selection packet manifest pin mismatch')
    packet = json.loads(packet_bytes)
    if (packet.get('candidate_selection') is not True or packet.get('training_admission') is not False or
            packet.get('trace_admission') is not False):
        raise ValueError('registry candidate-selection packet has an invalid stage/admission')
    candidate_ref, semantic_ref = packet.get('candidate_selection_review'), packet.get('semantic_review')
    clearance_ref = packet.get('source_quality_clearance')
    correction_ref = candidate_selection_correction_ref(packet, admission)
    if (not candidate_ref or not semantic_ref or not clearance_ref or
            candidate_ref.get('sha256') != admission.get('candidate_selection_review_sha256') or
            semantic_ref.get('sha256') != admission.get('semantic_review_sha256') or
            packet.get('semantic_review_input_audit', {}).get('sha256') != admission.get('historical_audit_sha256') or
            clearance_ref.get('file_sha256') != admission.get('source_quality_clearance_sha256')):
        raise ValueError('registry candidate-selection review or clearance pin mismatch')
    registered_bytes(candidate_ref['path'])
    registered_bytes(semantic_ref['path'])
    registered_bytes(clearance_ref['path'])
    if correction_ref:
        registered_bytes(correction_ref['path'])
    owner_path = admission.get('owner_admission_path')
    owner_sha = admission.get('owner_admission_sha256')
    if owner_path != 'root-selected-action-admission.json' or not isinstance(owner_sha, str):
        raise ValueError('registry candidate-selection owner receipt reference is invalid')
    owner_bytes = registered_bytes(owner_path)
    if file_sha(corpus_root / owner_path) != owner_sha:
        raise ValueError('registry candidate-selection owner receipt hash mismatch')
    owner = json.loads(owner_bytes)
    if (owner.get('schema') != 'natlang.root-selected-action-admission/1' or
            owner.get('training_admission') is not True or owner.get('trace_admission') is not False or
            owner.get('packet_manifest_sha256') != packet_sha or
            owner.get('review_path') != str((Path(entry['path']) / semantic_ref['path']).as_posix()) or
            owner.get('review_sha256') != semantic_ref.get('sha256') or
            owner.get('counts', {}).get('selected_native_actions') != len(owner.get('rows', [])) or
            owner.get('counts', {}).get('train_actions') != len(owner.get('rows', [])) or
            owner.get('counts', {}).get('test_actions') != 0 or
            owner.get('counts', {}).get('whole_trajectories') != 0 or
            owner.get('counts', {}).get('new_worlds') != 0 or
            owner.get('qualifications', {}).get('learned_writer') is not False or
            owner.get('qualifications', {}).get('recurrence') is not False or
            owner.get('qualifications', {}).get('runtime') is not False or
            owner.get('integration', {}).get('active_GPU_inputs_changed') is not False or
            owner.get('review_evidence', {}).get('selection_pin_correction_sha256') !=
            admission.get('selection_review_pin_correction_sha256')):
        raise ValueError('root candidate-selection admission receipt is malformed or outside SFT-only scope')
    prefix = Path(entry['path']).as_posix() + '/'
    owner_pins = owner.get('input_pins')
    if not isinstance(owner_pins, dict) or not owner_pins:
        raise ValueError('root candidate-selection receipt has no input pins')
    for pinned_path, pin in owner_pins.items():
        if not pinned_path.startswith(prefix):
            raise ValueError('root candidate-selection receipt input escapes the registered packet')
        registered_bytes(pinned_path[len(prefix):])
        if (pin.get('sha256') != registered[pinned_path[len(prefix):]].get('sha256') or
                pin.get('bytes') != registered[pinned_path[len(prefix):]].get('bytes')):
            raise ValueError('root candidate-selection input pin differs from corpus manifest')
    source_list = {path.resolve(strict=True) for path in sources}
    if corpus_root / 'turns.jsonl' not in source_list:
        raise ValueError('candidate-selection admission must include the pinned turns.jsonl source')
    verified = subprocess.run([
        'node', str(ROOT / 'scripts/verify_candidate_selection_packet.mjs'), '--root', str(corpus_root),
        '--manifest-sha256', packet_sha,
    ], cwd=ROOT, check=True, capture_output=True, text=True)
    receipt = json.loads(verified.stdout)
    if (receipt.get('schema') != 'natlang.verified-candidate-selection-packet/1' or
            receipt.get('manifest_sha256') != packet_sha or
            receipt.get('candidate_review_sha256') != admission.get('candidate_selection_review_sha256') or
            receipt.get('semantic_review_sha256') != admission.get('semantic_review_sha256') or
            receipt.get('historical_audit_sha256') != admission.get('historical_audit_sha256') or
            receipt.get('selection_review_pin_correction_sha256') != admission.get('selection_review_pin_correction_sha256')):
        raise ValueError('shared candidate packet verifier returned different pins')
    turns_path = corpus_root / 'turns.jsonl'
    turn_rows = [json.loads(line) for line in jsonl_lines(turns_path) if line.strip()]
    turns_by_id = {row['id']: row for row in turn_rows}
    if len(turns_by_id) != len(turn_rows):
        raise ValueError('registered candidate-selection turn IDs are not unique')
    owner_rows = owner.get('rows')
    if not isinstance(owner_rows, list) or not owner_rows:
        raise ValueError('root candidate-selection receipt has no exact row list')
    owner_rows_by_id = {row.get('native_id'): row for row in owner_rows if isinstance(row, dict)}
    if len(owner_rows_by_id) != len(owner_rows):
        raise ValueError('root candidate-selection receipt has duplicate or malformed row IDs')
    actual_owner_rows = []
    for row in receipt['rows']:
        turn = turns_by_id.get(row['id'])
        if not turn:
            raise ValueError(f'root candidate-selection receipt turn is missing: {row["id"]}')
        owner_row = owner_rows_by_id.get(row['id'])
        if (owner_row is None or
                owner_row.get('decision') != 'admit-exact-selected-native-action-SFT-only' or
                owner_row.get('training_admission') is not True):
            raise ValueError(f'root candidate-selection row is not approved for exact native action SFT: {row["id"]}')
        messages_sha256 = sha(canonical(turn.get('messages')).encode())
        # The manifest pins candidate_id and invocation identity. The owner receipt
        # intentionally authorizes the native ID plus source/target/message tuple;
        # keep the row projection here identical to that stable receipt schema.
        actual_owner_rows.append({
            'id': row.get('id'), 'item_id': row.get('item_id'), 'item_sha256': row.get('item_sha256'),
            'target_sha256': row.get('target_sha256'), 'messages_sha256': messages_sha256,
            'split': row.get('split'), 'source_groups': row.get('source_groups'),
            'decision': owner_row['decision'], 'training_admission': owner_row['training_admission'],
        })
    actual_owner_rows.sort(key=lambda row: row['id'])
    actual_by_id = {row['id']: row for row in receipt['rows']}
    declared_owner_rows = [{
        'id': row.get('native_id'), 'item_id': row.get('item_id'),
        'item_sha256': candidate_selection_owner_item_sha(
            row, actual_by_id.get(row.get('native_id'), {}).get('item_sha256')),
        'target_sha256': row.get('target_sha256'), 'messages_sha256': row.get('messages_sha256'),
        'split': row.get('split'), 'source_groups': row.get('source_groups'),
        'decision': row.get('decision'), 'training_admission': row.get('training_admission'),
    } for row in owner_rows]
    declared_owner_rows.sort(key=lambda row: row['id'])
    if canonical(actual_owner_rows) != canonical(declared_owner_rows):
        raise ValueError('root exact native IDs, target/message hashes, splits, or groups differ from verified packet')
    declared_rows = admission.get('rows')
    if not isinstance(declared_rows, list) or not declared_rows:
        raise ValueError('registry candidate-selection admission has no exact row list')

    def identity(row):
        return {key: row.get(key) for key in ('id', 'item_id', 'item_sha256', 'target_sha256',
                                               'messages_sha256', 'split', 'source_groups', 'decision',
                                               'training_admission')}
    actual = {row['id']: identity(row) for row in actual_owner_rows}
    declared = {row.get('id'): identity(row) for row in declared_rows if isinstance(row, dict)}
    if (len(declared) != len(declared_rows) or len(actual) != len(actual_owner_rows) or
            canonical(actual) != canonical(declared)):
        raise ValueError('registry candidate-selection rows do not equal owner-approved native IDs/targets/messages')
    rows_sha256 = sha(canonical([actual[row_id] for row_id in sorted(actual)]).encode())
    if admission.get('rows_sha256') != rows_sha256:
        raise ValueError('registry candidate-selection exact row-list hash mismatch')
    return {'corpus_id': corpus_id, 'packet_manifest_sha256': packet_sha,
            'candidate_selection_review_sha256': admission['candidate_selection_review_sha256'],
            'semantic_review_sha256': admission['semantic_review_sha256'],
            'historical_audit_sha256': admission['historical_audit_sha256'],
            'selection_review_pin_correction_sha256': admission['selection_review_pin_correction_sha256'],
            'owner_admission_sha256': owner_sha,
            'source_quality_clearance_sha256': admission['source_quality_clearance_sha256'], 'rows': actual}


def validate_candidate_selection_row(row, approved):
    """Keep corpus approval bound to one verifier-approved child row."""
    rid = row.get('id')
    if (approved.get('decision') != 'admit-exact-selected-native-action-SFT-only' or
            approved.get('training_admission') is not True):
        raise ValueError(f'registry candidate-selection row lacks exact native-action SFT approval: {rid}')
    if (approved.get('id') != rid or
            sha(canonical(row.get('target')).encode()) != approved.get('target_sha256') or
            sha(canonical(row.get('messages')).encode()) != approved.get('messages_sha256') or
            row.get('split') != approved.get('split') or
            row.get('source_groups') != approved.get('source_groups')):
        raise ValueError(f'registry candidate-selection row changed after packet verification: {rid}')
    decision = row.get('decision')
    if isinstance(decision, dict) and decision.get('training_approved') is not True:
        raise ValueError(f'candidate-selection source row is explicitly rejected: {rid}')
    admission = row.get('training_admission')
    if isinstance(admission, dict) and admission.get('approved') is False:
        pending = (admission.get('kind') == 'candidate-selection-pending-corpus-admission' and
                   admission.get('reason') in {
                       'exact child target selected for review; corpus admission remains false',
                       'source quality remains held; exact child target is materialized for review only; '
                       'corpus admission remains false',
                   })
        if not pending:
            raise ValueError(f'candidate-selection source row has an explicit training rejection: {rid}')


def selection_input_binding(args):
    """Bind the deterministic selector and all inputs that can change its result."""
    def file_bindings(paths):
        return [{'path': str(Path(path).resolve()), 'bytes': Path(path).stat().st_size,
                 'sha256': file_sha(path)} for path in paths]
    files = {
        'sources': file_bindings(args.source),
        'include_ids': file_bindings(args.include_ids or []),
        'exclude_programs': file_bindings(args.exclude_programs or []),
        'decision_exclude_ids': file_bindings(args.decision_exclude_ids or []),
        'target_exclude_ids': file_bindings(args.target_exclude_ids or []),
        'held_text': file_bindings(args.held_text or []),
        'reserved': file_bindings([args.reserved] if args.reserved else []),
        'protected': file_bindings([args.protected] if args.protected else []),
    }
    selector_functions = ('jsonl_lines', 'normalize_text', '_strings', 'protected_hits', 'reserved_needles',
                          'reserved_normalize', 'reserved_hit', 'index_needles', 'program_key', 'chain_key',
                          'split_chain_key', 'registered_source_pieces', 'candidate_selection_correction_ref',
                          'candidate_selection_owner_item_sha', 'registered_candidate_selection_admission',
                          'validate_candidate_selection_row', '_selection_state', 'select', 'full_sample')
    selector_code = '\n'.join(inspect.getsource(globals()[name]) for name in selector_functions)
    selector_dependencies = {
        'scripts/neuralese_data/records.py': file_sha(ROOT / 'scripts/neuralese_data/records.py'),
        'training/neuralese/natlang_neuralese/common/resumable_jsonl_frames.py':
            file_sha(ROOT / 'training/neuralese/natlang_neuralese/common/resumable_jsonl_frames.py'),
    }
    selector_sha = sha(canonical({
        'functions': selector_code,
        'dependencies': selector_dependencies,
        # These module-level values affect split normalization, registered-path
        # resolution and the protected-index representation used by helpers.
        'constants': {'held_split_aliases': HELD_SPLIT_ALIASES, 'prefix_bytes': _PREFIX},
    }).encode())
    options = {
        'corpus_id': args.corpus_id,
        'include_admits': bool(args.include_admits),
        'train_only': bool(args.train_only),
        'registry_admits_reason': args.registry_admits_reason,
        'limit': args.limit,
        'shards': args.shards,
        'full_sample': args.full_sample,
        'candidate_selection_admission_sha256': (
            sha(canonical(args.candidate_selection_admission).encode('utf-8'))
            if args.candidate_selection_admission is not None else None),
        'selector_code_sha256': selector_sha,
        'selector_policy': 'native-turn-source-order-selection/1',
    }
    binding = {'schema': 'natlang.native-turn-selection-inputs/1', 'files': files, 'options': options}
    binding['sha256'] = sha(canonical(binding).encode('utf-8'))
    return binding


def _selection_state(binding, next_source, omissions, counts, strata, blocked, protected_cache,
                     seen_ids, test_groups, reserved_groups, reserved_cache, frame_writer):
    return {
        'schema': 'natlang.native-turn-selection-checkpoint/1',
        'binding': binding,
        'next_source': next_source,
        'omissions': omissions,
        'counts': dict(counts),
        'strata': {key: sorted(values) for key, values in strata.items()},
        'blocked': blocked,
        'protected': protected_cache,
        'seen_ids': sorted(seen_ids),
        'test_groups': sorted(test_groups),
        'reserved_groups': sorted(reserved_groups),
        'reserved_cache': reserved_cache,
        'frames': frame_writer.frames,
    }


def select(args, shard_dir: Path, state_path: Path, binding: dict):
    """Stream the sources; write selected raw lines to trajectory shards; return omissions and blocked chains."""
    exclude = {}
    include = None
    if args.include_ids:  # restrict to these decisions (ids or source_record_ids of rendered text rows)
        include = set()
        for path in args.include_ids:
            for line in open(path, 'rb'):
                if line.strip():
                    item = json.loads(line)
                    include.add(item.get('id'))
                    include.update(item.get('source_record_ids') or [])
    for path in args.exclude_programs or []:
        for line in jsonl_lines(path):
            if line.strip():
                item = json.loads(line)
                exclude.setdefault(item['id'], item.get('reason_code') or 'held_case_excluded')
    needles = index_needles(reserved_needles(args.reserved)) if args.reserved else None
    reserved_cache, reserved_groups = {}, set()
    protected = json.loads(args.protected.read_text()) if args.protected else None
    compressed = getattr(args, 'compress_text', False)
    existing = json.loads(state_path.read_text()) if state_path.exists() else None
    if existing is not None:
        if existing.get('schema') != 'natlang.native-turn-selection-checkpoint/1':
            raise ValueError('selection state is complete legacy data, not an incremental checkpoint')
        if existing.get('binding') != binding:
            previous = existing.get('binding')
            compatible = False
            if isinstance(previous, dict):
                previous_options = previous.get('options') or {}
                old_selector_sha = previous_options.get('selector_code_sha256')
                candidate = dict(previous)
                candidate_options = dict(previous_options)
                candidate_options['selector_code_sha256'] = binding['options']['selector_code_sha256']
                candidate['options'] = candidate_options
                candidate.pop('sha256', None)
                candidate['sha256'] = sha(canonical(candidate).encode())
                compatible = (old_selector_sha in LEGACY_SELECTOR_FILE_SHA256
                              and binding['options']['selector_code_sha256'] == LEGACY_SELECTOR_SEMANTICS_SHA256
                              and candidate == binding)
            if not compatible:
                raise ValueError('selection checkpoint inputs or selector semantics changed')
            # Record the narrower selector binding atomically. The selected
            # frame table and source frontier are unchanged.
            existing['binding'] = binding
            write_json_durable(state_path, existing)
        omissions, counts = existing['omissions'], Counter(existing['counts'])
        strata = defaultdict(set, {key: set(values) for key, values in existing['strata'].items()})
        blocked, protected_cache = existing['blocked'], existing['protected']
        seen_ids, test_groups = set(existing['seen_ids']), set(existing['test_groups'])
        reserved_groups = set(existing['reserved_groups'])
        reserved_cache = existing['reserved_cache']
        start_source = existing['next_source']
        if not isinstance(start_source, int) or not 0 <= start_source <= len(args.source):
            raise ValueError('selection checkpoint source frontier is invalid')
        frame_writer = ResumableJsonlFrames(shard_dir, args.shards, compressed=compressed,
                                            frames=existing['frames'])
        n = counts['source_rows']
    else:
        omissions, counts = [], Counter()
        strata = defaultdict(set)  # family -> selected programs (for the full-rendering warm-up sample)
        blocked = {}  # chain -> fewest messages of an unselected decision in it
        protected_cache = {}
        seen_ids, test_groups, reserved_groups = set(), set(), set()
        start_source, n = 0, 0
        frame_writer = ResumableJsonlFrames(shard_dir, args.shards, compressed=compressed)
    full_sample_programs = None
    limit_reached = False
    try:
        for source_index in range(start_source, len(args.source)):
            source = args.source[source_index]
            with open_jsonl_text(source) as f:
                for text_line in f:
                    line = text_line.encode('utf-8')
                    if not line.strip():
                        continue
                    if args.limit and n >= args.limit:
                        limit_reached = True
                        break
                    n += 1
                    row = json.loads(line)
                    rid = row.get('id')
                    counts['source_rows'] += 1
                    if row.get('split') in HELD_SPLIT_ALIASES:
                        # Held-out under another name (upstream validation/dev): the students' loaders know train and
                        # test only, so it is rendered as test (held, decontaminated against train) and the original
                        # name is kept on the row; never dropped.
                        counts[f"split_mapped:{row['split']}->test"] += 1
                        row['source_split'], row['split'] = row['split'], 'test'
                        line = json.dumps(row, ensure_ascii=False).encode()
                    reason = None
                    if include is not None and rid not in include:
                        counts['not_in_include_set'] += 1  # outside this render's scope: no omission, no chain block
                        continue
                    if include is not None and args.include_admits and not (
                            row.get('training_admission') or {}).get('approved'):
                        # Admitted by the text set it was rendered for (its root receipt), recorded per row.
                        row['training_admission'] = {**(row.get('training_admission') or {}), 'approved': True,
                                                     'admitted_through_include_set': [str(x) for x in args.include_ids]}
                        line = json.dumps(row, ensure_ascii=False).encode()
                        counts['approved_through_include_set'] += 1
                    candidate_admission = getattr(args, 'candidate_selection_admission', None)
                    if row.get('candidate_selection', {}).get('approved') is True and candidate_admission is not None:
                        approved = candidate_admission['rows'].get(rid)
                        if approved is None:
                            reason = 'not_in_registry_candidate_selection_admission'
                        else:
                            validate_candidate_selection_row(row, approved)
                            row['training_admission'] = {
                                'approved': True, 'kind': 'registry-bound-candidate-selection',
                                'status': 'admitted-exact-selected-child-action', 'corpus_id': candidate_admission['corpus_id'],
                                'packet_manifest_sha256': candidate_admission['packet_manifest_sha256'],
                                'candidate_selection_review_sha256': candidate_admission['candidate_selection_review_sha256'],
                                'semantic_review_sha256': candidate_admission['semantic_review_sha256'],
                                'historical_audit_sha256': candidate_admission['historical_audit_sha256'],
                                'selection_review_pin_correction_sha256': candidate_admission[
                                    'selection_review_pin_correction_sha256'],
                                'owner_admission_sha256': candidate_admission['owner_admission_sha256'],
                                'source_quality_clearance_sha256': candidate_admission['source_quality_clearance_sha256'],
                                'item_id': approved['item_id'], 'item_sha256': approved['item_sha256'],
                                'target_sha256': approved['target_sha256'], 'split': approved['split'],
                                'source_groups': approved['source_groups'],
                            }
                            row['decision'] = {**(row.get('decision') or {}), 'training_approved': True,
                                               'registry_candidate_selection_admission': candidate_admission['corpus_id']}
                            line = json.dumps(row, ensure_ascii=False).encode()
                            counts['approved_by_registry_candidate_selection'] += 1
                    if row.get('version') != 'natlang.teacher_training_turn.native/1':
                        reason = 'not_a_native_teacher_turn'
                    elif rid in seen_ids:
                        reason = 'duplicate_source_row_id'
                    elif args.train_only and row.get('split') != 'train':
                        reason = 'held_split_kept_out'
                    elif (args.registry_admits_reason and isinstance(row.get('training_admission'), dict)
                          and row['training_admission'].get('approved') is False
                          and str(row['training_admission'].get('reason') or '').startswith(args.registry_admits_reason)):
                        # The registry's training admission overrides this row-level hold (recorded per row).
                        row['training_admission'] = {**row['training_admission'], 'approved': True,
                                                     'row_hold_overridden_by_registry_admission': args.corpus_id}
                        line = json.dumps(row, ensure_ascii=False).encode()
                        counts['approved_by_registry_admission'] += 1
                    if reason is not None:
                        pass
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
                            reserved_cache[program_key(row)] if program_key(row) in reserved_cache
                            else reserved_cache.setdefault(program_key(row), reserved_hit(row, needles))):
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
                        if reason != 'duplicate_source_row_id':
                            # A later copy of a selected ID is not an unselected decision: its first occurrence (an
                            # earlier source, e.g. a replacement overlay) was selected. Blocking its chain cut the
                            # selected copy and every longer decision out of whole-trajectory documents (the 4962
                            # render kept 32 of 4962 rows, 2026-10-12).
                            chain = split_chain_key(row)
                            blocked[chain] = min(blocked.get(chain, 1 << 30), len(row.get('messages') or []))
                        continue
                    seen_ids.add(rid)
                    if row.get('split') == 'test':
                        test_groups.update(g for g in row.get('source_groups') or [] if isinstance(g, str))
                    counts['selected'] += 1
                    counts['selected:' + str(row.get('split'))] += 1
                    strata[str(row.get('family') or row.get('task_family') or '')].add(program_key(row))
                    shard = int(sha(str(row.get('teacher_trajectory_id') or chain_key(row)).encode())[:8], 16) % args.shards
                    frame_writer.write(shard, line.rstrip(b'\n').decode('utf-8') + '\n')
            frame_writer.finish_source(source_index)
            next_source = len(args.source) if limit_reached else source_index + 1
            write_json_durable(state_path, _selection_state(
                binding, next_source, omissions, counts, strata, blocked, protected_cache,
                seen_ids, test_groups, reserved_groups, reserved_cache, frame_writer))
            if limit_reached:
                break
    except BaseException:
        frame_writer.abort()
        raise
    else:
        frame_writer.close()
    full_sample_programs = full_sample(strata, args.full_sample)
    counts['reserved_programs_hit'] = sum(bool(v) for v in reserved_cache.values())
    counts['reserved_groups'] = len(reserved_groups)
    return omissions, counts, blocked, protected_cache, test_groups, reserved_groups, full_sample_programs


def full_sample(strata: dict, fraction: float) -> list[str] | None:
    """Programs whose decisions get the full rendering (warm-up share; shared rule, coordinator 2026-10-11): per family,
    the ceil(fraction * n) programs with the smallest sha256(program id): deterministic, stratified, at least one per
    family. None means every program (fraction >= 1)."""
    if fraction >= 1:
        return None
    chosen = []
    for programs in strata.values():
        ranked = sorted(programs, key=lambda key: sha(key.encode()))
        chosen += ranked[:max(1, math.ceil(fraction * len(ranked)))] if fraction > 0 else []
    return sorted(chosen)


# ---------------------------------------------------------------- rendering (workers)

def _init(tokenizers, pieces=None, history_policies=None):
    from build_neuralese_gold_text_delta import load_pinned_tokenizer
    from natlang_neuralese.serve.chat import bind_history_reasoning
    from strip_generic_system_prompt import strip_row
    _G['strip_row'] = strip_row
    _G['pieces'] = pieces or {}  # exact source pieces, keyed by their registered names
    _G['tok'] = {}
    _G['tokenizer_paths'] = {}
    history_policies = history_policies or {}
    if set(history_policies) - set(tokenizers):
        raise ValueError('history-reasoning policy names must have a configured tokenizer')
    _G['history_policies'] = dict(history_policies)
    for name, path in tokenizers.items():
        tok = load_pinned_tokenizer(Path(path))
        policy = history_policies.get(name)
        bind_history_reasoning(tok, policy=policy, require_declared=True,
                               allow_transition=policy is not None)
        _G['tok'][name] = tok
        _G['tokenizer_paths'][name] = Path(path).resolve()


def _render_input_binding(tokenizers, pieces, history_policies=None):
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    tokenizer_inputs = {name: _tokenizer_input_binding(name, path, tokenizer_fingerprint)
                        for name, path in sorted(tokenizers.items())}
    history_policies = history_policies or {}
    result = {'tokenizers': tokenizer_inputs,
              'pieces_sha256': sha(canonical(pieces or {}).encode()) if pieces else None}
    if history_policies:
        result['history_reasoning_policies'] = {
            name: history_policies.get(name, 'backbone-default') for name in sorted(tokenizers)}
    return result


def _history_profiles_for_tokenizers(tokenizer_paths, history_policies):
    return {name: history_policies.get(name, 'backbone-default')
            for name in sorted(tokenizer_paths)}


def _parse_history_reasoning_policies(values, tokenizer_names):
    parsed = {}
    for item in values:
        if '=' not in item:
            raise ValueError('--history-reasoning-policy expects NAME=POLICY')
        name, policy = item.split('=', 1)
        if not name or policy not in ('keep', 'last_turn_only') or name in parsed:
            raise ValueError(f'invalid or duplicate history-reasoning policy: {item!r}')
        parsed[name] = policy
    unknown = set(parsed) - set(tokenizer_names)
    if unknown:
        raise ValueError(f'history-reasoning policy names have no tokenizer: {sorted(unknown)}')
    return parsed


def _tokenizer_input_binding(name, path, fingerprint=None):
    if fingerprint is None:
        from natlang_neuralese.data.text_corpus import tokenizer_fingerprint as fingerprint
    root = Path(path).resolve()
    relevant_files = ('tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja',
                      'special_tokens_map.json', 'added_tokens.json', 'vocab.json',
                      'merges.txt', 'tokenizer.model')
    return {'fingerprint': fingerprint(_G['tok'][name]),
            'files': {filename: file_sha(root / filename) for filename in relevant_files
                      if (root / filename).is_file()}}


def _message_digest(message) -> str:
    return sha(canonical(message).encode())


def source_assistant_binding(member, actual):
    """Authenticate the action, allowing only declared reasoning and call-ID transport."""
    from natlang_neuralese.data.text_corpus import target_with_teacher_reasoning
    target = member.get('target')
    if (not isinstance(target, dict) or target.get('role') != 'assistant'
            or not isinstance(actual, dict) or actual.get('role') != 'assistant'
            or (actual.get('reasoning_content') is not None and not isinstance(actual['reasoning_content'], str))):
        return None
    def without_call_ids(message):
        normalized = dict(message)
        calls = normalized.get('tool_calls')
        if calls is None:
            return normalized, []
        if (not isinstance(calls, list) or any(not isinstance(call, dict)
                or (call.get('id') is not None and not isinstance(call['id'], str)) for call in calls)):
            return None, None
        identifiers = [call.get('id') for call in calls]
        present = [value for value in identifiers if value is not None]
        if len(present) != len(set(present)):
            return None, None
        normalized['tool_calls'] = [{key: value for key, value in call.items() if key != 'id'} for call in calls]
        return normalized, identifiers
    normalized_actual, actual_ids = without_call_ids(actual)
    if normalized_actual is None:
        return None
    candidates = [target, target_with_teacher_reasoning(member)]
    reasoning = member.get('teacher_reasoning')
    if (member.get('teacher_reasoning_trained') is False and isinstance(reasoning, str) and reasoning
            and not target.get('reasoning_content')):
        candidates.append({**target, 'reasoning_content': reasoning})
    for candidate in candidates:
        normalized_target, source_ids = without_call_ids(candidate)
        if normalized_target is not None and normalized_target == normalized_actual:
            return {'source_target_sha256': _message_digest(target),
                    'history_assistant_sha256': _message_digest(actual),
                    **({'untrained_reasoning_sha256': sha(actual['reasoning_content'].encode())}
                       if member.get('teacher_reasoning_trained') is False and actual.get('reasoning_content') else {}),
                    'tool_call_id_rebindings': [{'call_index': index, 'source_id': source_id, 'history_id': actual_id}
                                                for index, (source_id, actual_id) in enumerate(zip(source_ids, actual_ids))
                                                if source_id != actual_id]}
    return None


def source_decision_in_history(member, record):
    """An input prefix alone never proves that its admitted answer occurred."""
    prefix, history = member.get('messages') or [], record.get('messages') or []
    return (len(prefix) < len(history) and prefix == history[:len(prefix)]
            and source_assistant_binding(member, history[len(prefix)]) is not None)


def decontaminate_source_records(records, held_groups):
    """Protect exact held context as well as each record's own source groups."""
    protected = defaultdict(list)
    for record in records:
        hits = sorted(set(record.get('source_groups') or []) & held_groups)
        if hits or record.get('split') == 'test':
            protected[chain_key(record)].append((record, hits))
    kept, omissions = [], []
    for record in records:
        own_hits = sorted(set(record.get('source_groups') or []) & held_groups)
        inherited = next(((member, hits) for member, hits in protected.get(chain_key(record), [])
                          if member['id'] != record['id'] and source_decision_in_history(member, record)), None)
        if record.get('split') == 'train' and (own_hits or inherited is not None):
            omissions.append({'id': record['id'], 'rendering': 'all', 'stage': 'decontamination',
                'reason': ('train_decision_shares_held_source_group' if own_hits else
                           'train_trajectory_contains_held_source_decision'),
                **({'group': own_hits[0]} if own_hits else
                   {'held_source_record_id': inherited[0]['id'],
                    'held_source_groups': sorted(inherited[0].get('source_groups') or [])})})
        else:
            kept.append(record)
    return kept, omissions


def whole_trajectory_records(records: list[dict], blocked: dict) -> list[tuple[dict, list[str]]]:
    """Maximal clean decision chains: (record whose messages+target is the document, ids of decisions it covers)."""
    groups = defaultdict(list)
    for r in records:
        groups[(chain_key(r), r.get('split'))].append(r)
    out = []
    for (chain, _split), rows in groups.items():
        # Older completed selection states used invocation-only keys. Preserve
        # their conservative exclusions explicitly; new states are split-local.
        limit = blocked.get(canonical([chain, _split]), blocked.get(chain, 1 << 30))
        clean = sorted((r for r in rows if len(r.get('messages') or []) < limit),
                       key=lambda r: len(r.get('messages') or []))
        digests = [[_message_digest(m) for m in r.get('messages') or []] for r in clean]
        covered_by = {}
        for i, r in enumerate(clean):
            for j in range(len(clean) - 1, i, -1):
                if (len(digests[j]) > len(digests[i]) and digests[j][:len(digests[i])] == digests[i]
                        and source_decision_in_history(r, clean[j])):
                    covered_by[i] = j
                    break
        for i, r in enumerate(clean):
            if i in covered_by:
                continue
            members = [clean[k]['id'] for k in range(len(clean))
                       if k == i or (len(digests[k]) < len(digests[i]) and digests[i][:len(digests[k])] == digests[k]
                                     and source_decision_in_history(clean[k], r))]
            out.append((r, members))
    return out


def whole_trajectory_target_holds(record, members, messages, records_by_id, source_hashes, target_exclude):
    """Bind quality-held decisions to assistant ordinals without changing context."""
    held = sorted(set(members) & set(target_exclude))
    if not held:
        return {}
    targets = sorted(set(members) - set(target_exclude))
    if not targets:
        raise ValueError('whole trajectory has no independently admitted decision targets')
    original = record.get('messages') or []
    assistants = sum(message.get('role') == 'assistant' for message in original)
    if sum(message.get('role') == 'assistant' for message in messages) != assistants:
        raise ValueError('target hold mapping changed assistant count during context authentication')
    bindings, positive_bindings, seen_turns = [], [], set()
    for identifier in sorted(members):
        member = records_by_id[identifier]
        groups = member.get('source_groups')
        if (member.get('split') != record.get('split') or not isinstance(groups, list) or not groups
                or any(not isinstance(group, str) or not group for group in groups)):
            raise ValueError('target hold members require the same split and factual source groups')
        prefix = member.get('messages') or []
        if (len(prefix) > len(original) or prefix != original[:len(prefix)]
                or (identifier != record['id'] and
                    (len(prefix) == len(original) or original[len(prefix)].get('role') != 'assistant'))):
            raise ValueError('decision is not bound to an assistant in its exact source prefix')
        actual = record.get('target') if len(prefix) == len(original) else original[len(prefix)]
        assistant_binding = source_assistant_binding(member, actual)
        if assistant_binding is None:
            raise ValueError('decision target differs from the actual source assistant')
        ordinal = sum(message.get('role') == 'assistant' for message in prefix)
        if ordinal in seen_turns:
            raise ValueError('multiple source decisions claim the same assistant turn')
        seen_turns.add(ordinal)
        digest = source_hashes[identifier]
        if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
            raise ValueError('decision lacks its exact source record digest')
        (bindings if identifier in held else positive_bindings).append(
            {'id': identifier, 'source_record_sha256': digest,
             **assistant_binding, 'assistant_turn': ordinal})
    return {'assistant_turn_count': assistants + 1,
            'context_only_assistant_turns': sorted({item['assistant_turn'] for item in bindings}),
            'target_exclusion_bindings': bindings, 'target_source_bindings': positive_bindings,
            'source_record_groups': {identifier: sorted(set(records_by_id[identifier]['source_groups']))
                                     for identifier in sorted(members)},
            'target_source_record_ids': targets,
            'context_only_source_record_ids': held}


def whole_trajectory_reasoning_holds(record, members, messages, target, records_by_id, source_hashes):
    """Source-declared untrained notes stay context, including historical turns."""
    original = record.get('messages') or []
    assistants = sum(message.get('role') == 'assistant' for message in original)
    if sum(message.get('role') == 'assistant' for message in messages) != assistants:
        raise ValueError('reasoning hold mapping changed assistant count during context authentication')
    bindings = []
    for identifier in members:
        member = records_by_id[identifier]
        if member.get('teacher_reasoning_trained') is not False:
            continue
        prefix = member.get('messages') or []
        if prefix != original[:len(prefix)] or len(prefix) > len(original):
            raise ValueError('untrained reasoning source differs from its actual history prefix')
        actual = record.get('target') if identifier == record['id'] else original[len(prefix)]
        if not isinstance(actual, dict) or not actual.get('reasoning_content'):
            continue
        binding = source_assistant_binding(member, actual)
        if binding is None:
            raise ValueError('untrained reasoning action differs from its actual source assistant')
        if 'untrained_reasoning_sha256' not in binding:
            continue
        digest = source_hashes[identifier]
        if not isinstance(digest, str) or re.fullmatch('[0-9a-f]{64}', digest) is None:
            raise ValueError('untrained reasoning lacks its exact source record binding')
        bindings.append({'id': identifier, 'source_record_sha256': digest,
                         'assistant_turn': sum(message.get('role') == 'assistant' for message in prefix),
                         'teacher_reasoning_trained': False, **binding})
    if not bindings:
        return {}
    return {'reasoning_assistant_turn_count': assistants+1,
            'context_only_reasoning_turns': sorted({item['assistant_turn'] for item in bindings}),
            'reasoning_exclusion_bindings': bindings}


def bind_target_hold_policy(out, sources, holds):
    """A resumed shard may never silently retain a different target mask."""
    path = out / 'target-hold-policy.json'
    if not holds and not path.exists():
        return None
    binding = {'schema': 'natlang.native-turn-target-holds/1',
               'sources': [{'path': str(source), 'sha256': file_sha(source)} for source in sources],
               'target_exclude_ids': [{'path': str(hold), 'sha256': file_sha(hold)} for hold in holds]}
    if path.exists():
        if json.loads(path.read_text()) != binding:
            raise ValueError('render target hold policy changed; use a fresh output directory')
    else:
        if (out / 'work' / 'selection-state.json').exists() or (out / 'receipt.json').exists():
            raise ValueError('existing render has no target hold binding; use a fresh output directory')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(canonical(binding)+'\n')
        temporary.replace(path)
    return {'file': path.name, 'sha256': file_sha(path)}


def qualified_whole_trajectory_row(row):
    """Keep source/mask variants distinct before corpus-union ID selection."""
    from text_tokenizer_twin import document_dedup_identity
    legacy_id = row['id']
    digest = sha(canonical(document_dedup_identity(row)).encode())
    return {**row, 'source_row_id': legacy_id, 'id': legacy_id + ':source-sha256:' + digest}


def apply_explicit_history_profile_ids(rows, profile):
    """Give a changed explicit rendering its own IDs while retaining source IDs."""
    if profile is None:
        return rows
    if profile not in ('keep', 'last_turn_only'):
        raise ValueError('unsupported explicit history-reasoning ID profile')
    for row in rows:
        original_id = row['id']
        expected = original_id + ':history-reasoning:' + profile
        if row.get('history_reasoning_policy') not in (None, profile):
            raise ValueError('rendered row profile conflicts with the tokenizer profile')
        if any(original_id.endswith(':history-reasoning:' + name)
               for name in ('keep', 'last_turn_only')):
            raise ValueError('rendered row already has a history-reasoning profile ID')
        row.setdefault('source_row_id', original_id)
        row['id'] = expected
        row['history_reasoning_policy'] = profile
    return rows


def _render_shard_materialized(job):
    (shard_path, out_dir, tok_name, blocked, corpus, held_groups, decision_exclude, header_as_full, full_programs,
     renderings, *storage_options) = job
    compress_text = bool(storage_options and storage_options[0])
    target_exclude = storage_options[1] if len(storage_options) > 1 else set()
    writer_context_records = storage_options[2] if len(storage_options) > 2 else ()
    # Per-decision renderings are rendered only when one of them is requested; the program-header parity check needs
    # both, so with program-header requested the full rendering is rendered too (and kept per --full-sample).
    per_decision = {'full', 'program-header'} & set(renderings)
    # header_as_full: the corpus's system prompt is its program's own prompt (no generic runtime prose), so the
    # program-header rendering is the full rendering.
    from natlang_neuralese.data.text_corpus import (authenticated_context_writer_sources,
                                                    authenticated_crisp_context_messages, gold_text_rows,
                                                    native_gold_document, target_with_teacher_reasoning,
                                                    tokenizer_fingerprint)
    from natlang_neuralese.train.trajectories import crisp_messages, handover_notes
    tok = _G['tok'][tok_name]
    records = []
    with open_jsonl_text(shard_path) as f:
        for line in f:
            row = json.loads(line)
            row['_source_record_sha256'] = row.pop('_stream_source_sha256', None) or sha(line.rstrip('\n').encode('utf-8'))
            records.append(row)
    stem = Path(shard_path).name.removesuffix('.zst').removesuffix('.jsonl')
    result = {'shard': stem, 'tokenizer': tok_name, 'records': len(records)}
    omissions = []
    # Source-group-disjoint decontamination: a train decision sharing a source group with any held document (this
    # corpus's test rows or another corpus's held set) is dropped; held rows never move.
    records, decontamination_omissions = decontaminate_source_records(records, held_groups)
    omissions.extend(decontamination_omissions)
    result['decontaminated'] = result['records'] - len(records)
    if not records:
        write_done(Path(out_dir), tok_name, stem, result)
        return result
    rows = {}
    # Decisions already rendered elsewhere (--decision-exclude-ids: e.g. the 4962 union's source records) are not
    # rendered again as decision documents; they still complete their whole trajectories.
    decision_records = [r for r in records if r['id'] not in decision_exclude and r['id'] not in target_exclude]
    result['decision_excluded'] = sum(r['id'] in decision_exclude for r in records)
    result['target_excluded'] = sum(r['id'] in target_exclude for r in records)
    omissions.extend({'id': r['id'], 'rendering': 'decision', 'stage': 'quality-target-hold',
                      'reason': 'explicit_target_hold',
                      'source_record_sha256': r['_source_record_sha256'],
                      'whole_trajectory_context_eligible': True}
                     for r in records if r['id'] in target_exclude)
    for mode in (('full',) if header_as_full else ('full', 'program-header')) if per_decision else ():
        if not decision_records:
            rows[mode] = []
            continue
        r, receipt, omitted, provenance = gold_text_rows(decision_records, _G['pieces'], tokenizer=tok,
                                                         require_independent_splits=False, prompt_rendering=mode,
                                                         target_reasoning=True,
                                                         writer_context_records=writer_context_records)
        for row, prov in zip(r, provenance):
            row['corpus'] = corpus
            row['prompt_rendering'] = mode
        rows[mode] = r
        omissions += [{**o, 'rendering': mode, 'stage': 'render'} for o in omitted]
        result['receipt_' + mode] = {k: receipt[k] for k in ('format', 'rendering', 'history_reasoning', 'supervision',
                                                             'documents', 'omitted_records',
                                                             'duplicate_same_split_documents_deduplicated',
                                                             'excluded_train_exact_held_complete_documents')}
    if header_as_full and per_decision:
        rows['program-header'] = [dict(row, prompt_rendering='program-header', system_prompt_mode='program-own')
                                  for row in rows['full']]
    # Program-header parity: the renderer's rows must equal the strip script's derivation from the full rows.
    im_start, im_end = tok.convert_tokens_to_ids('<|im_start|>'), tok.convert_tokens_to_ids('<|im_end|>')
    by_id = {row['id']: row for row in rows.get('program-header', [])}
    parity = Counter()
    kept_ph = []
    for full in ([] if header_as_full or not per_decision else rows['full']):
        ph = by_id.get(full['id'])
        if ph is None:
            parity['program_header_missing'] += 1
            continue
        stripped, reason, info = _G['strip_row'](full, tok, im_start, im_end)
        if stripped is None and reason == 'no leading system turn' and (full['text'], full['token_ids']) == (
                ph['text'], ph['token_ids']):
            parity['equal_no_system_turn'] += 1  # nothing to strip: both renderings are the same document
            ph['system_prompt_mode'] = 'none'
            kept_ph.append(ph)
            continue
        if stripped is None:
            parity['strip_failed'] += 1
            omissions.append({'id': full['id'], 'rendering': 'program-header', 'stage': 'parity', 'reason': 'strip_failed: ' + reason})
            continue
        parity['retokenization_checked'] += bool(info['retokenization_checked'])
        same_text = (stripped['text'] == ph['text'] and
                     tok.decode(stripped['token_ids'][stripped['supervised_suffix_start']:]) ==
                     tok.decode(ph['token_ids'][ph['supervised_suffix_start']:]))
        if same_text and stripped['token_ids'] != ph['token_ids']:
            # Same bytes, different token boundaries: the header carries special-token spellings, which the serving
            # renderer tokenizes per escaped run while the strip splice encodes the header as one string. The
            # renderer's IDs are the serving tokenization; keep them.
            parity['equal_text_serving_token_boundaries'] += 1
            ph['system_prompt_mode'] = stripped['system_prompt_mode']
            kept_ph.append(ph)
            continue
        if (stripped['text'], stripped['token_ids'], stripped['supervised_suffix_start']) != (
                ph['text'], ph['token_ids'], ph['supervised_suffix_start']):
            parity['mismatch'] += 1
            omissions.append({'id': full['id'], 'rendering': 'program-header', 'stage': 'parity',
                              'reason': 'renderer_and_strip_script_differ'})
            continue
        parity['equal'] += 1
        ph['system_prompt_mode'] = stripped['system_prompt_mode']
        kept_ph.append(ph)
    if not header_as_full and per_decision:
        rows['program-header'] = kept_ph
    result['parity'] = dict(parity)
    # The full rendering is rendered for every decision (the program-header parity check needs it) but kept only for
    # the warm-up sample of programs; program-header and whole-trajectory cover everything.
    if full_programs is not None and per_decision:
        program_of = {r['id']: program_key(r) for r in decision_records}
        before = len(rows['full'])
        rows['full'] = [row for row in rows['full'] if program_of.get(row['id']) in full_programs]
        result['full_outside_warmup_sample'] = before - len(rows['full'])
    # Whole trajectory.
    fingerprint = tokenizer_fingerprint(tok)
    # The streaming wrapper supplies the exact source rows for external
    # writers referenced by this trajectory. Rebuild the authenticated writer
    # index from those rows, rather than retaining large body strings in the
    # shard-wide metadata index.
    context_rows = list(records)
    context_rows.extend(writer_context_records)
    source_hashes, writer_sources = authenticated_context_writer_sources(context_rows)
    records_by_id = {record['id']: record for record in records}
    wt = []
    for record, members in whole_trajectory_records(records, blocked):
        try:
            piece_map = _G['pieces']
            messages, _, _ = authenticated_crisp_context_messages(record, piece_map, writer_sources)
            target_holds = whole_trajectory_target_holds(record, members, messages,
                records_by_id, source_hashes, target_exclude)
            # Gates above bind the exact source target; the teacher reasoning is added only for rendering.
            target = crisp_messages([target_with_teacher_reasoning(record)], piece_map, handover_notes(record))[0]
            reasoning_holds = whole_trajectory_reasoning_holds(record, members, messages, target,
                records_by_id, source_hashes)
            text, ids = native_gold_document(tok, messages, target, record.get('tools') or [],
                                             prompt_rendering='full' if header_as_full else 'program-header')
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
        wt.append(qualified_whole_trajectory_row({'text': text, 'token_ids': ids, 'tokenizer_sha256': fingerprint, 'split': record['split'],
                   'runtime_opening_assistant_turns': opening,
                   'source_groups': sorted({g for member in members
                                            for g in records_by_id[member].get('source_groups') or []
                                            if isinstance(g, str) and g}),
                   'id': 'whole-trajectory:' + record['id'],
                   'source_record_ids': target_holds.get('target_source_record_ids', members),
                   'source_records': [{'id': m, 'sha256': source_hashes[m]} for m in members],
                   'source_record_groups': {m: sorted(set(records_by_id[m].get('source_groups') or []))
                                            for m in members},
                   **target_holds,
                   **reasoning_holds,
                   'corpus': corpus, 'prompt_rendering': 'program-header', 'supervision': 'whole-trajectory',
                   **({'system_prompt_mode': 'program-own'} if header_as_full else {})}))
    rows['whole-trajectory'] = wt
    for mode in [m for m in rows if m not in renderings]:  # rendered only for the parity check
        del rows[mode]
    explicit_profile = (_G.get('history_policies') or {}).get(tok_name)
    if explicit_profile is not None:
        for rs in rows.values():
            apply_explicit_history_profile_ids(rs, explicit_profile)
    for mode, rs in rows.items():
        path = Path(out_dir) / mode / tok_name / (f'{stem}.jsonl.zst' if compress_text else f'{stem}.jsonl')
        path.parent.mkdir(parents=True, exist_ok=True)
        with write_jsonl_text(path) as f:
            for row in rs:
                f.write(canonical(row) + '\n')
        if compress_text:
            path.with_suffix('').unlink(missing_ok=True)  # superseded unfinished plain output
        result[mode] = len(rs)
    path = Path(out_dir) / 'omissions' / tok_name / f'{stem}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(canonical(o) + '\n' for o in omissions))
    write_done(Path(out_dir), tok_name, stem, result)
    return result


def _trajectory_key(record):
    ref = record.get('source_ref') or {}
    trajectory = ref.get('trajectory_id') or record.get('teacher_trajectory_id')
    # Legacy records without an explicit trajectory still retain invocation
    # isolation, exactly as chain_key does.
    return str(trajectory) if trajectory is not None else chain_key(record)


def _render_job_binding(source_sha, tok_name, blocked, corpus, held_groups, decision_exclude,
                        target_exclude, header_as_full, full_programs, renderings,
                        compress_text, code_binding, input_binding=None):
    return sha(canonical({
        'source_sha256': source_sha, 'tokenizer': tok_name, 'corpus': corpus,
        'blocked': blocked, 'held_groups': sorted(held_groups),
        'decision_exclude': sorted(decision_exclude), 'target_exclude': sorted(target_exclude),
        'header_as_full': header_as_full,
        'full_programs': None if full_programs is None else sorted(full_programs),
        'renderings': list(renderings), 'compress_text': compress_text, 'code': code_binding,
        'render_inputs': input_binding,
    }).encode())


def _trajectory_render_binding(job_binding, trajectory, source_ids, source_record_hashes):
    return sha(canonical({'job': job_binding, 'trajectory': trajectory,
                          'source_ids': source_ids,
                          'source_record_hashes': source_record_hashes}).encode())


def _render_code_binding():
    renderer_path = str(Path(__file__).resolve())
    binding = {
        renderer_path: file_sha(Path(__file__).resolve()),
        str(ROOT / 'training/neuralese/natlang_neuralese/data/text_corpus.py'):
            file_sha(ROOT / 'training/neuralese/natlang_neuralese/data/text_corpus.py'),
        str(ROOT / 'scripts/neuralese_data/strip_generic_system_prompt.py'):
            file_sha(ROOT / 'scripts/neuralese_data/strip_generic_system_prompt.py'),
        str(ROOT / 'scripts/neuralese_data/conversion_output_lease.py'):
            file_sha(ROOT / 'scripts/neuralese_data/conversion_output_lease.py'),
    }
    return binding


def _resource_parent_code_binding(parent_renderer_sha256):
    if parent_renderer_sha256 != SEARCH_RESOURCE_ONLY_PARENT_RENDERER_SHA256:
        raise ValueError('renderer parent SHA lacks the exact reviewed resource-only continuation')
    binding = _render_code_binding()
    binding[str(Path(__file__).resolve())] = parent_renderer_sha256
    return binding


def _render_job_binding_from_job(job, source_digest, code_binding):
    return _render_job_binding(source_digest, job[2], job[3], job[4], job[5], job[6],
                               job[11] if len(job) > 11 else set(), job[7], job[8], job[9],
                               bool(job[10]) if len(job) > 10 else False, code_binding,
                               job[13] if len(job) > 13 else None)


def _legacy_render_job_binding(job, source_sha, renderer_sha):
    if renderer_sha != LEGACY_SEARCH_RENDERER_SHA256:
        raise ValueError('legacy renderer SHA is not an approved historical migration witness')
    dependency_paths = {
        'text_corpus.py': ROOT / 'training/neuralese/natlang_neuralese/data/text_corpus.py',
        'strip_generic_system_prompt.py': ROOT / 'scripts/neuralese_data/strip_generic_system_prompt.py',
    }
    for name, path in dependency_paths.items():
        old_path = next(Path(key) for key in LEGACY_SEARCH_CODE_BINDING if key.endswith('/' + name))
        actual_sha = file_sha(path)
        expected_sha = LEGACY_SEARCH_CODE_BINDING[str(old_path)]
        if actual_sha == expected_sha:
            continue
        if name == 'text_corpus.py':
            profile_bindings = [part.get('history_reasoning_policies') for part in job
                                if isinstance(part, dict) and 'history_reasoning_policies' in part]
            _shard, _out, tok_name = job[:3]
            selected_profile = (profile_bindings[-1].get(tok_name)
                                if profile_bindings and isinstance(profile_bindings[-1], dict) else None)
            transition = LEGACY_SEARCH_TEXT_CORPUS_DEFAULT_PROFILE_TRANSITION
            if actual_sha == transition['current_sha256'] and selected_profile in (None, 'backbone-default'):
                continue
        if actual_sha != expected_sha:
            raise ValueError(f'legacy renderer dependency differs from its historical binding: {name}')
    (_shard_path, _out_dir, tok_name, blocked, corpus, held_groups, decision_exclude,
     header_as_full, full_programs, renderings, compress_text, target_exclude, *_rest) = job
    # The approved legacy renderer (content SHA b054133a; commit cd4140a5e5)
    # did not include the later tokenizer/piece ``render_inputs`` binding
    # field. Recreate its
    # exact job-binding equation here; adding ``render_inputs: None`` would
    # silently change every legacy binding and make its trajectory receipts
    # impossible to authenticate. Per-trajectory source IDs and raw row hashes
    # are still checked by _trajectory_render_binding before any old output is
    # reused.
    return sha(canonical({
        'source_sha256': source_sha, 'tokenizer': tok_name, 'corpus': corpus,
        'blocked': blocked, 'held_groups': sorted(held_groups),
        'decision_exclude': sorted(decision_exclude), 'target_exclude': sorted(target_exclude),
        'header_as_full': header_as_full,
        'full_programs': None if full_programs is None else sorted(full_programs),
        'renderings': list(renderings), 'compress_text': compress_text,
        'code': LEGACY_SEARCH_CODE_BINDING,
    }).encode())


def _verify_trajectory_receipts(job, done_result, allowed_renderer_shas, expected_job_binding=None):
    shard_path, out_dir, tok_name = Path(job[0]), Path(job[1]), job[2]
    stem = shard_path.name.removesuffix('.zst').removesuffix('.jsonl')
    source_digest = file_sha(shard_path)
    current_code_binding = _render_code_binding()
    current_renderer_sha = current_code_binding[str(Path(__file__).resolve())]
    expected_job_binding = (expected_job_binding if expected_job_binding is not None else
                            _render_job_binding_from_job(job, source_digest, current_code_binding))
    if done_result.get('render_job_binding') != expected_job_binding:
        raise ValueError(f'done marker does not match current render job policy: {stem}/{tok_name}')
    if done_result.get('renderer_code_sha256') not in allowed_renderer_shas:
        raise ValueError(f'done marker renderer identity is not explicitly allowed: {stem}/{tok_name}')
    expected_aggregate = {}
    for mode in job[9]:
        suffix = '.jsonl.zst' if bool(job[10]) else '.jsonl'
        expected_aggregate[f'{mode}/{tok_name}/{stem}{suffix}'] = None
    expected_aggregate[f'omissions/{tok_name}/{stem}.jsonl'] = None
    recorded_aggregate = done_result.get('aggregate_outputs')
    if not isinstance(recorded_aggregate, dict) or set(recorded_aggregate) != set(expected_aggregate):
        raise ValueError(f'done marker aggregate output inventory mismatch: {stem}/{tok_name}')
    for relative, metadata in recorded_aggregate.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError(f'done marker contains unsafe aggregate output path: {relative}')
        output_path = out_dir / relative_path
        if not output_path.is_file():
            raise ValueError(f'aggregate output missing: {output_path}')
        if (not isinstance(metadata, dict) or metadata.get('bytes') != output_path.stat().st_size
                or metadata.get('sha256') != file_sha(output_path)):
            raise ValueError(f'aggregate output checksum mismatch: {output_path}')
    receipts_root = out_dir / 'trajectory-stream' / tok_name / stem / 'receipts'
    receipts = sorted(receipts_root.glob('*.json'))
    expected_count = done_result.get('trajectories_committed')
    if not isinstance(expected_count, int) or len(receipts) != expected_count:
        raise ValueError(f'done marker/trajectory receipt count mismatch: {stem}/{tok_name}')
    actual_renderer_shas = set()
    trajectory_bindings = done_result.get('trajectory_bindings')
    trajectory_renderers = done_result.get('trajectory_renderers')
    if not isinstance(trajectory_bindings, dict) or not isinstance(trajectory_renderers, dict):
        raise ValueError(f'done marker lacks per-trajectory source/code identities: {stem}/{tok_name}')
    if set(trajectory_bindings) != {path.stem for path in receipts} or set(trajectory_renderers) != set(trajectory_bindings):
        raise ValueError(f'done marker trajectory identity inventory mismatch: {stem}/{tok_name}')
    for path in receipts:
        receipt = json.loads(path.read_text())
        if receipt.get('binding') != trajectory_bindings[path.stem]:
            raise ValueError(f'trajectory source binding differs from done marker: {path}')
        if receipt.get('trajectory_sha256') != path.stem:
            raise ValueError(f'trajectory identity mismatch: {path}')
        expected_renderer_sha = trajectory_renderers[path.stem]
        if expected_renderer_sha not in allowed_renderer_shas:
            raise ValueError(f'trajectory renderer identity is not allowed: {path}')
        recorded_renderer = receipt.get('renderer_code_sha256')
        if recorded_renderer is None and expected_renderer_sha == LEGACY_SEARCH_RENDERER_SHA256:
            recorded_renderer = expected_renderer_sha
        if recorded_renderer != expected_renderer_sha:
            raise ValueError(f'trajectory renderer identity mismatch: {path}')
        actual_renderer_shas.add(recorded_renderer)
        group_dir = path.parent.parent / 'groups' / path.stem
        _verify_group_receipt(
            receipt, path.stem, group_dir, tok_name, job[9], bool(job[10]),
            expected_binding=trajectory_bindings[path.stem],
            expected_renderer_sha=expected_renderer_sha,
            allow_legacy_done_output=(
                'renderer_code_sha256' not in receipt
                and expected_renderer_sha == LEGACY_SEARCH_RENDERER_SHA256))
    declared = done_result.get('group_renderer_code_sha256s')
    if declared is not None and sorted(actual_renderer_shas) != sorted(declared):
        raise ValueError(f'done marker renderer summary differs from trajectory receipts: {stem}/{tok_name}')


def _verify_group_receipt(receipt, group_id, group_dir, tok_name, renderings, compress_text,
                          expected_binding=None, expected_renderer_sha=None,
                          allow_legacy_done_output=False):
    """Require every durable output of a trajectory group, with safe paths and exact hashes."""
    if not isinstance(receipt, dict) or receipt.get('schema') != 'natlang.native-turn-trajectory-render/1':
        raise ValueError(f'trajectory receipt schema mismatch: {group_id}')
    if receipt.get('trajectory_sha256') != group_id:
        raise ValueError(f'trajectory receipt identity mismatch: {group_id}')
    if expected_binding is not None and receipt.get('binding') != expected_binding:
        raise ValueError(f'trajectory receipt binding mismatch: {group_id}')
    if expected_renderer_sha is not None:
        got_renderer_sha = receipt.get('renderer_code_sha256', expected_renderer_sha)
        if got_renderer_sha != expected_renderer_sha:
            raise ValueError(f'trajectory receipt renderer identity mismatch: {group_id}')
    source_ids = receipt.get('source_record_ids')
    result = receipt.get('result')
    if not isinstance(source_ids, list) or not source_ids or not isinstance(result, dict):
        raise ValueError(f'trajectory receipt lacks source rows or render result: {group_id}')
    if any(not isinstance(result.get(mode), int) or result[mode] < 0 for mode in renderings):
        raise ValueError(f'trajectory receipt lacks a valid count for every rendering: {group_id}')
    suffix = '.jsonl.zst' if compress_text else '.jsonl'
    expected = {f'{mode}/{tok_name}/{group_id}{suffix}' for mode in renderings}
    expected.add(f'omissions/{tok_name}/{group_id}.jsonl')
    if allow_legacy_done_output:
        expected.add(f'done/{tok_name}/{group_id}.json')
    outputs = receipt.get('outputs')
    if not isinstance(outputs, dict) or set(outputs) != expected:
        raise ValueError(f'trajectory receipt output inventory mismatch: {group_id}')
    for relative, digest in outputs.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or '..' in relative_path.parts:
            raise ValueError(f'trajectory receipt contains unsafe output path: {relative}')
        if not isinstance(digest, str) or not re.fullmatch(r'[0-9a-f]{64}', digest):
            raise ValueError(f'trajectory receipt has invalid output SHA-256: {relative}')
        output_path = group_dir / relative_path
        if not output_path.is_file():
            raise ValueError(f'trajectory output missing: {output_path}')
        if file_sha(output_path) != digest:
            raise ValueError(f'trajectory output checksum mismatch: {output_path}')


def _verify_legacy_done_binding(job, done_result, renderer_sha, source_digest):
    if renderer_sha != LEGACY_SEARCH_RENDERER_SHA256:
        raise ValueError('legacy renderer SHA is not an approved historical migration witness')
    if done_result.get('source_sha256') not in (None, source_digest):
        raise ValueError('legacy done marker source SHA differs from selected shard')
    if done_result.get('trajectories_committed') is None:
        # The original Search renderer wrote aggregate-only markers. They have
        # no per-trajectory receipts and are never trusted as reusable output:
        # this migration queues the selected shard for a complete source-based
        # rerender, which writes the current receipt only after every group is
        # independently committed. Accept only that exact old marker shape.
        legacy_only = {'renderer_code_sha256', 'render_job_binding',
                       'aggregate_outputs', 'trajectory_bindings',
                       'trajectory_renderers'}
        if any(key in done_result for key in legacy_only):
            raise ValueError('legacy done marker has an incomplete current receipt')
        if not isinstance(done_result.get('records'), int) or done_result['records'] < 0:
            raise ValueError('legacy done marker lacks its selected-record count')
        if done_result.get('shard') != Path(job[0]).name.removesuffix('.zst').removesuffix('.jsonl'):
            raise ValueError('legacy done marker shard identity differs from selected input')
        if done_result.get('tokenizer') != job[2]:
            raise ValueError('legacy done marker tokenizer differs from selected job')
        for mode in job[9]:
            if not isinstance(done_result.get(mode), int) or done_result[mode] < 0:
                raise ValueError(f'legacy done marker lacks a valid {mode} count')
    elif not isinstance(done_result.get('trajectories_committed'), int):
        raise ValueError('legacy done marker has an invalid committed trajectory count')


def _filter_render_jobs(jobs, selected_shards):
    if selected_shards is None:
        return list(jobs)
    requested = set(selected_shards)
    available = {Path(job[0]).name.removesuffix('.zst').removesuffix('.jsonl') for job in jobs}
    unknown = sorted(requested - available)
    if unknown:
        raise ValueError(f'unknown selected shard stems: {unknown[:8]}')
    return [job for job in jobs
            if Path(job[0]).name.removesuffix('.zst').removesuffix('.jsonl') in requested]


def render_shard(job):
    """Render one selected shard with bounded, trajectory-scoped memory.

    Selection remains the existing immutable shard. Rows are spooled to a
    rebuildable SQLite index, and each complete trajectory is rendered with
    the established materialized renderer into independently checksummed
    files. A group receipt is committed only after every rendering and omission
    file is durable. Restart verifies those receipts and reuses complete groups;
    an unreceipted group is discarded and rendered again.
    """
    (shard_path, out_dir, tok_name, blocked, corpus, held_groups, decision_exclude, header_as_full, full_programs,
     renderings, *storage_options) = job
    compress_text = bool(storage_options and storage_options[0])
    target_exclude = storage_options[1] if len(storage_options) > 1 else set()
    input_binding = storage_options[3] if len(storage_options) > 3 else None
    sqlite_temp_root = Path(storage_options[4]) if len(storage_options) > 4 and storage_options[4] else None
    shard_path, out_dir = Path(shard_path), Path(out_dir)
    stem = shard_path.name.removesuffix('.zst').removesuffix('.jsonl')
    outer = out_dir / 'trajectory-stream' / tok_name / stem
    groups_root, receipts_root = outer / 'groups', outer / 'receipts'
    groups_root.mkdir(parents=True, exist_ok=True)
    receipts_root.mkdir(parents=True, exist_ok=True)
    source_file_sha = file_sha(shard_path)
    if input_binding is not None:
        from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
        actual_input_binding = {
            'tokenizers': {tok_name: _tokenizer_input_binding(
                tok_name, _G['tokenizer_paths'][tok_name], tokenizer_fingerprint)},
            'pieces_sha256': sha(canonical(_G.get('pieces') or {}).encode())
            if _G.get('pieces') else None,
        }
        if _G.get('history_policies'):
            actual_input_binding['history_reasoning_policies'] = _history_profiles_for_tokenizers(
                _G['tokenizer_paths'], _G['history_policies'])
        expected_model_binding = input_binding.get('tokenizers', {}).get(tok_name)
        if (not isinstance(expected_model_binding, dict)
                or expected_model_binding != actual_input_binding['tokenizers'][tok_name]
                or input_binding.get('pieces_sha256') != actual_input_binding['pieces_sha256']
                or input_binding.get('history_reasoning_policies') != actual_input_binding.get(
                    'history_reasoning_policies')):
            raise ValueError(f'loaded tokenizer or piece inventory changed for {tok_name}')
    code_binding = _render_code_binding()
    renderer_code_sha = code_binding[str(Path(__file__).resolve())]
    job_binding = _render_job_binding(source_file_sha, tok_name, blocked, corpus, held_groups, decision_exclude,
                                      target_exclude, header_as_full, full_programs, renderings,
                                      compress_text, code_binding, input_binding)
    legacy_renderer_sha = storage_options[2] if len(storage_options) > 2 else None
    resource_parent_sha = storage_options[5] if len(storage_options) > 5 else None
    legacy_job_binding = None
    if legacy_renderer_sha:
        legacy_job_binding = _legacy_render_job_binding(job, source_file_sha, legacy_renderer_sha)
    resource_parent_job_binding = None
    if resource_parent_sha:
        resource_parent_job_binding = _render_job_binding_from_job(
            job, source_file_sha, _resource_parent_code_binding(resource_parent_sha))

    # The SQLite row index is wholly rebuildable from the selected shard. Keep
    # it on a caller-selected fast local scratch volume when supplied; durable
    # trajectory outputs and receipts remain under `outer`.
    if sqlite_temp_root is not None:
        sqlite_temp_root.mkdir(parents=True, exist_ok=True)
    # Register the DB close after the temporary-directory cleanup callback so
    # ExitStack runs db.close() first on every exit path.  If rendering raises,
    # TemporaryDirectory must never try to remove an open SQLite database: the
    # failed cleanup can otherwise leave the deleted database reachable through
    # an unlinked worker fd for the lifetime of a retained future traceback.
    with ExitStack() as scratch:
        temporary = scratch.enter_context(tempfile.TemporaryDirectory(
            prefix=f'natlang-{stem}-', dir=sqlite_temp_root or outer))
        db = sqlite3.connect(str(Path(temporary) / 'rows.sqlite'))
        scratch.callback(db.close)
        db.execute('PRAGMA journal_mode=OFF')
        db.execute('PRAGMA synchronous=OFF')
        # Keep the per-worker page cache bounded while sorting/indexing a large
        # selected shard. Negative cache_size values are kibibytes in SQLite.
        db.execute('PRAGMA cache_size=-65536')
        db.execute('CREATE TABLE rows (seq INTEGER PRIMARY KEY, trajectory TEXT NOT NULL, row_id TEXT, '
                   'source_sha TEXT NOT NULL, payload TEXT NOT NULL)')
        db.execute('CREATE TABLE emitted (mode TEXT NOT NULL, seq INTEGER NOT NULL, group_index INTEGER NOT NULL, '
                   'row_index INTEGER NOT NULL, line TEXT NOT NULL)')
        insert = []
        total = 0
        with open_jsonl_text(shard_path) as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                source_sha = sha(line.rstrip('\n').encode('utf-8'))
                insert.append((total, _trajectory_key(record), record.get('id'), source_sha, line.rstrip('\n')))
                total += 1
                if len(insert) >= 256:
                    db.executemany('INSERT INTO rows VALUES (?, ?, ?, ?, ?)', insert)
                    db.commit()
                    insert.clear()
        if insert:
            db.executemany('INSERT INTO rows VALUES (?, ?, ?, ?, ?)', insert)
            db.commit()

        # Index creation is deliberately after the bulk load: maintaining both
        # secondary indexes for every 256-row insert caused random writes across
        # multi-gigabyte temporary databases on the external source disk.
        db.execute('CREATE INDEX rows_trajectory_seq ON rows(trajectory, seq)')
        db.execute('CREATE INDEX rows_id_seq ON rows(row_id, seq)')
        db.commit()

        trajectories_sql = ('SELECT trajectory, MIN(seq) AS first_seq FROM rows '
                            'GROUP BY trajectory ORDER BY first_seq')
        # Build the exact cross-trajectory soft-writer index one trajectory at
        # a time. Same-run provider-expanded lookups are handled within each
        # complete trajectory by the existing authenticator.
        from natlang_neuralese.data.text_corpus import authenticated_context_writer_sources
        cross_writers = {}
        for trajectory, _first_seq in db.execute(trajectories_sql):
            local = []
            for payload, source_sha in db.execute(
                    'SELECT payload, source_sha FROM rows WHERE trajectory=? ORDER BY seq', (trajectory,)):
                record = json.loads(payload)
                record['_source_record_sha256'] = source_sha
                local.append(record)
            local_hashes, local_sources = authenticated_context_writer_sources(local)
            for block_id, candidates in local_sources.items():
                current = cross_writers.setdefault(block_id, [])
                by_identity = {(item.get('writer_record_id'), item.get('body_sha256')) for item in current}
                for candidate in candidates:
                    identity = (candidate.get('writer_record_id'), candidate.get('body_sha256'))
                    if identity not in by_identity:
                        current.append({key: value for key, value in candidate.items() if key != 'body'})
                        by_identity.add(identity)
            del local, local_hashes, local_sources

        _G['cross_trajectory_writer_sources'] = cross_writers
        trajectory_count = db.execute('SELECT COUNT(*) FROM (SELECT trajectory FROM rows GROUP BY trajectory)').fetchone()[0]
        aggregate = {'shard': stem, 'tokenizer': tok_name, 'records': total,
                     'trajectory_groups': trajectory_count, 'streaming': True,
                     'source_sha256': source_file_sha, 'renderer_code_sha256': renderer_code_sha,
                     'render_job_binding': job_binding}
        group_renderer_shas = set()
        trajectory_bindings, trajectory_renderers = {}, {}

        def accumulate(result):
            for key, value in result.items():
                if key in {'shard', 'tokenizer', 'records'}:
                    continue
                if isinstance(value, int):
                    aggregate[key] = aggregate.get(key, 0) + value
                elif key.startswith('receipt_') and isinstance(value, dict):
                    receipt = aggregate.setdefault(key, {})
                    for receipt_key, receipt_value in value.items():
                        if isinstance(receipt_value, int):
                            receipt[receipt_key] = receipt.get(receipt_key, 0) + receipt_value
                        else:
                            receipt.setdefault(receipt_key, receipt_value)
                elif key == 'parity' and isinstance(value, dict):
                    parity = aggregate.setdefault('parity', {})
                    for item, count in value.items():
                        parity[item] = parity.get(item, 0) + count

        for group_index, (trajectory, _first_seq) in enumerate(db.execute(trajectories_sql)):
            group_id = sha(trajectory.encode('utf-8'))
            group_dir = groups_root / group_id
            receipt_path = receipts_root / f'{group_id}.json'
            group_rows = []
            source_record_hashes = []
            for payload, source_sha in db.execute(
                    'SELECT payload, source_sha FROM rows WHERE trajectory=? ORDER BY seq', (trajectory,)):
                record = json.loads(payload)
                record['_stream_source_sha256'] = source_sha
                group_rows.append(record)
                source_record_hashes.append(source_sha)
            source_ids = [row.get('id') for row in group_rows]
            binding = _trajectory_render_binding(job_binding, trajectory, source_ids,
                                                 source_record_hashes)
            legacy_binding = (_trajectory_render_binding(legacy_job_binding, trajectory, source_ids,
                                                          source_record_hashes)
                              if legacy_job_binding is not None else None)
            expected_receipt = None
            if receipt_path.is_file():
                expected_receipt = json.loads(receipt_path.read_text())
                receipt_binding = expected_receipt.get('binding')
                if receipt_binding == binding:
                    receipt_renderer_sha = expected_receipt.get('renderer_code_sha256', renderer_code_sha)
                    if receipt_renderer_sha != renderer_code_sha:
                        raise ValueError(f'trajectory receipt renderer identity changed: {trajectory}')
                elif legacy_binding is not None and receipt_binding == legacy_binding:
                    receipt_renderer_sha = expected_receipt.get('renderer_code_sha256', legacy_renderer_sha)
                    if receipt_renderer_sha != legacy_renderer_sha:
                        raise ValueError(f'legacy trajectory receipt renderer identity changed: {trajectory}')
                elif (resource_parent_job_binding is not None
                      and receipt_binding == _trajectory_render_binding(
                          resource_parent_job_binding, trajectory, source_ids, source_record_hashes)):
                    receipt_renderer_sha = expected_receipt.get('renderer_code_sha256', resource_parent_sha)
                    if receipt_renderer_sha != resource_parent_sha:
                        raise ValueError(f'resource-parent trajectory renderer identity changed: {trajectory}')
                else:
                    raise ValueError(f'committed trajectory binding changed: {trajectory}')
                _verify_group_receipt(
                    expected_receipt, group_id, group_dir, tok_name, renderings, compress_text,
                    expected_binding=receipt_binding, expected_renderer_sha=receipt_renderer_sha,
                    allow_legacy_done_output=('renderer_code_sha256' not in expected_receipt
                                              and receipt_renderer_sha == legacy_renderer_sha))
                if expected_receipt.get('source_record_ids') != source_ids:
                    raise ValueError(f'committed trajectory source row inventory changed: {trajectory}')
            if expected_receipt is not None:
                accumulate(expected_receipt['result'])
                group_renderer_shas.add(receipt_renderer_sha)
                trajectory_bindings[group_id] = receipt_binding
                trajectory_renderers[group_id] = receipt_renderer_sha
                del group_rows
                continue
            if group_dir.exists():
                shutil.rmtree(group_dir)
            group_dir.mkdir(parents=True)
            from natlang_neuralese.data.text_corpus import _message_neuralese_ids, _message_soft_reads
            referenced_blocks = set()
            for row in group_rows:
                referenced_blocks.update(_message_neuralese_ids(row.get('messages') or []))
                referenced_blocks.update(read.get('block_id') for read in
                                        _message_soft_reads(row.get('messages') or [])
                                        if isinstance(read.get('block_id'), str))
            writer_ids = sorted({candidate.get('writer_record_id')
                                 for block_id in referenced_blocks
                                 for candidate in cross_writers.get(block_id, [])
                                 if isinstance(candidate.get('writer_record_id'), str)}
                                - {row.get('id') for row in group_rows})
            writer_context_records = []
            for writer_id in writer_ids:
                found = db.execute('SELECT payload, source_sha FROM rows WHERE row_id=? ORDER BY seq LIMIT 1',
                                   (writer_id,)).fetchone()
                if found is None:
                    raise ValueError(f'cross-trajectory writer source disappeared from shard index: {writer_id}')
                writer_record = json.loads(found[0])
                writer_record['_source_record_sha256'] = found[1]
                writer_context_records.append(writer_record)
            temp_source = group_dir / f'{group_id}.jsonl'
            with temp_source.open('w', encoding='utf-8') as stream:
                for row in group_rows:
                    stream.write(canonical(row) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
            del group_rows
            group_job = (str(temp_source), str(group_dir), tok_name, blocked, corpus, held_groups,
                         decision_exclude, header_as_full, full_programs, renderings, compress_text,
                         target_exclude, writer_context_records)
            result = _render_shard_materialized(group_job)
            temp_source.unlink(missing_ok=True)
            outputs = {}
            for path in sorted(group_dir.rglob('*')):
                if (path.is_file() and path != receipt_path and not path.name.endswith('.tmp')
                        and path.relative_to(group_dir).parts[0] != 'done'):
                    outputs[str(path.relative_to(group_dir))] = file_sha(path)
            group_receipt = {'schema': 'natlang.native-turn-trajectory-render/1',
                             'binding': binding, 'trajectory_sha256': sha(trajectory.encode()),
                             'renderer_code_sha256': renderer_code_sha,
                             'source_record_ids': source_ids, 'outputs': outputs, 'result': result}
            write_json_durable(receipt_path, group_receipt)
            group_renderer_shas.add(renderer_code_sha)
            trajectory_bindings[group_id] = binding
            trajectory_renderers[group_id] = renderer_code_sha
            accumulate(result)

        # Sort the materialized trajectory outputs back into exact selected
        # source order without retaining rows in Python. The scratch index is
        # rebuildable; durable group frames/receipts remain the resume frontier.
        for mode in renderings:
            destination = out_dir / mode / tok_name / (f'{stem}.jsonl.zst' if compress_text else f'{stem}.jsonl')
            db.execute('DELETE FROM emitted')
            row_index = 0
            for trajectory, _first_seq in db.execute(trajectories_sql):
                group_id = sha(trajectory.encode('utf-8'))
                group_file = groups_root / group_id / mode / tok_name / (
                    f'{group_id}.jsonl.zst' if compress_text else f'{group_id}.jsonl')
                if not group_file.exists():
                    continue
                with open_jsonl_text(group_file) as stream:
                    for line in stream:
                        row = json.loads(line)
                        source_id = row.get('id')
                        if mode == 'whole-trajectory':
                            source_rows = row.get('source_records') or []
                            if source_rows:
                                source_id = source_rows[0].get('id')
                        found = db.execute('SELECT seq FROM rows WHERE row_id=? ORDER BY seq LIMIT 1',
                                           (source_id,)).fetchone()
                        seq = found[0] if found else _first_seq
                        db.execute('INSERT INTO emitted VALUES (?, ?, ?, ?, ?)',
                                   (mode, seq, row_index, row_index, line))
                        row_index += 1
                db.commit()
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = (destination.with_name(destination.stem + '.tmp.zst') if compress_text
                         else destination.with_name(destination.name + '.tmp'))
            from natlang_neuralese.common.jsonio import write_jsonl_text
            with write_jsonl_text(temporary) as stream:
                for (line,) in db.execute('SELECT line FROM emitted WHERE mode=? ORDER BY seq, group_index, row_index',
                                          (mode,)):
                    stream.write(line if line.endswith('\n') else line + '\n')
            with temporary.open('rb') as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, destination)

        omission_path = out_dir / 'omissions' / tok_name / f'{stem}.jsonl'
        db.execute('DELETE FROM emitted')
        row_index = 0
        for trajectory, first_seq in db.execute(trajectories_sql):
            group_id = sha(trajectory.encode('utf-8'))
            group_file = groups_root / group_id / 'omissions' / tok_name / f'{group_id}.jsonl'
            if not group_file.exists():
                continue
            with group_file.open(encoding='utf-8') as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    found = db.execute('SELECT seq FROM rows WHERE row_id=? ORDER BY seq LIMIT 1',
                                       (row.get('id'),)).fetchone()
                    db.execute('INSERT INTO emitted VALUES (?, ?, ?, ?, ?)',
                               ('omissions', found[0] if found else first_seq, row_index, row_index, line))
                    row_index += 1
        db.commit()
        omission_path.parent.mkdir(parents=True, exist_ok=True)
        omission_tmp = omission_path.with_name(omission_path.name + '.tmp')
        with omission_tmp.open('w', encoding='utf-8', newline='\n') as stream:
            for (line,) in db.execute("SELECT line FROM emitted WHERE mode='omissions' ORDER BY seq, group_index, row_index"):
                stream.write(line if line.endswith('\n') else line + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(omission_tmp, omission_path)
        db.close()
        aggregate_outputs = {}
        for mode in renderings:
            path = out_dir / mode / tok_name / (
                f'{stem}.jsonl.zst' if compress_text else f'{stem}.jsonl')
            aggregate_outputs[str(path.relative_to(out_dir))] = {
                'bytes': path.stat().st_size, 'sha256': file_sha(path)}
        aggregate_outputs[str(omission_path.relative_to(out_dir))] = {
            'bytes': omission_path.stat().st_size, 'sha256': file_sha(omission_path)}
        aggregate['trajectories_committed'] = trajectory_count
        aggregate['aggregate_outputs'] = aggregate_outputs
        aggregate['group_renderer_code_sha256s'] = sorted(group_renderer_shas)
        aggregate['trajectory_bindings'] = dict(sorted(trajectory_bindings.items()))
        aggregate['trajectory_renderers'] = dict(sorted(trajectory_renderers.items()))
        write_done(out_dir, tok_name, stem, aggregate)
        return aggregate


def write_done(work: Path, tok_name: str, stem: str, result: dict):
    """Resume marker, written last and atomically: a job with a marker is never rendered again."""
    done = work / 'done' / tok_name / f'{stem}.json'
    done.parent.mkdir(parents=True, exist_ok=True)
    tmp = done.with_suffix('.tmp')
    tmp.write_text(json.dumps(result))
    os.replace(tmp, done)


def _run_bounded_render_jobs(pool, jobs, workers, on_result):
    """Submit at most one worker-width of renders and stop submission on error."""
    pending_jobs = iter(jobs)
    futures = set()
    for _ in range(min(workers, len(jobs))):
        try:
            futures.add(pool.submit(render_shard, next(pending_jobs)))
        except StopIteration:
            break
    completed = 0
    while futures:
        future = next(as_completed(futures))
        futures.remove(future)
        try:
            result = future.result()
        except BaseException:
            for outstanding in futures:
                outstanding.cancel()
            raise
        completed += 1
        on_result(completed, result)
        try:
            futures.add(pool.submit(render_shard, next(pending_jobs)))
        except StopIteration:
            pass


# ---------------------------------------------------------------- merge

def role_counter(tok):
    from natlang_neuralese.train.text_supervision import ROLE_CODES, chat_roles

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
    count.reasoning_boundary_ids = (token_id('<think>'), token_id('</think>'))
    return count


def whole_trajectory_target_tokens(row, count_roles):
    """(decision target tokens, runtime-opening tokens, role counter) of a whole-trajectory document: its assistant
    turns (weight 1) less the leading turns the runtime ran itself (the opening eval: inputs, supervised as context
    like the user turn)."""
    c, codes = count_roles(row['token_ids'])
    runs, reasoning_body, inside = [], [], False
    boundaries = getattr(count_roles, 'reasoning_boundary_ids', ())
    for position, code in enumerate(codes):
        is_assistant = code in (4, 5)  # ROLE_CODES assistant_reasoning, assistant_reply
        if is_assistant and not inside:
            runs.append(0)
            reasoning_body.append(0)
        if is_assistant:
            runs[-1] += 1
            reasoning_body[-1] += int(code == 4 and row['token_ids'][position] not in boundaries)
        inside = is_assistant
    skip = row.get('runtime_opening_assistant_turns', 0)
    held = set(row.get('context_only_assistant_turns', []))
    reasoning_held = set(row.get('context_only_reasoning_turns', []))
    return sum(size - (reasoning_body[index] if index in reasoning_held else 0)
               for index, size in enumerate(runs) if index >= skip and index not in held), sum(runs[:skip]), c


def held_groups_from(paths):
    groups = set()
    for path in paths or []:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                if row.get('split') == 'test':
                    groups.update(row.get('source_groups') or [])
    return groups


def merge(out: Path, mode: str, tok_name: str, shards: list[str], held_external: set, count_roles, *, compress_text=False):
    """Concatenate shard outputs in shard order with the global split and dedup checks of gold_text_rows."""
    from text_tokenizer_twin import document_dedup_identity
    src = [out / 'work' / mode / tok_name / f'{s}.jsonl' for s in shards]
    src = [p if p.exists() else p.with_suffix('.jsonl.zst') for p in src]
    src = [p for p in src if p.exists()]
    train_groups, test_groups, test_texts = set(), set(), set()
    for p in src:
        with open_jsonl_text(p) as f:
            for line in f:
                row = json.loads(line)
                (train_groups if row['split'] == 'train' else test_groups).update(row['source_groups'])
                if row['split'] == 'test':
                    test_texts.add(sha(row['text'].encode()))
    overlap = train_groups & test_groups
    held = test_groups | held_external
    final = out / mode / tok_name
    final.mkdir(parents=True, exist_ok=True)
    omissions, seen = [], {}
    stats = {'documents': 0, 'by_split': {}, 'max_document_tokens': 0}
    roles_total = Counter()
    text_path = final / ('text-data.jsonl.zst' if compress_text else 'text-data.jsonl')
    decoded_digest, decoded_bytes = hashlib.sha256(), 0
    with write_jsonl_text(text_path) as dst:
        for p in src:
            with open_jsonl_text(p) as f:
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
                           row.get('supervised_suffix_start'), tuple(row.get('context_only_assistant_turns', [])),
                           tuple(row.get('context_only_reasoning_turns', [])),
                           canonical(document_dedup_identity(row)))
                    if key in seen:
                        # Same document as a kept one: its source decisions are covered there (lineage kept here).
                        omissions.append({'id': row['id'], 'reason': 'exact_duplicate_document',
                                          'duplicate_of': seen[key],
                                          'source_record_ids': row.get('source_record_ids', [])})
                        continue
                    seen[key] = row['id']
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
                    text = line if line.endswith('\n') else line + '\n'
                    dst.write(text)
                    data = text.encode('utf-8')
                    decoded_digest.update(data)
                    decoded_bytes += len(data)
    stats['roles'] = dict(roles_total)
    stats['internal_train_test_group_overlap'] = len(overlap)
    (final / 'omissions.jsonl').write_text(''.join(canonical(o) + '\n' for o in omissions))
    stats['merge_omissions'] = dict(Counter(o['reason'] for o in omissions))
    stats['text_data_sha256'] = file_sha(text_path)
    if compress_text:
        stats.update(text_data_file=text_path.name, decoded_text_data_sha256=decoded_digest.hexdigest(), decoded_text_data_bytes=decoded_bytes)
    return stats


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--corpus-id', required=True)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--tokenizer', action='append', default=[], help='NAME=DIR')
    p.add_argument('--history-reasoning-policy', action='append', default=[], metavar='NAME=POLICY',
                   help='opt in to a history policy for this tokenizer; omitted names use the backbone default')
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--verify-admission-only', action='store_true',
                   help='verify the complete registered candidate admission without rendering or model loading')
    p.add_argument('--exclude-programs', type=Path, nargs='*', default=[],
                   help='jsonl rows {id, reason_code?}: programs whose rows are dropped')
    p.add_argument('--program-header-as-full', action='store_true',
                   help="the system prompt is the program's own prompt, not generic runtime prose: keep it")
    p.add_argument('--pieces', type=Path, nargs='*', default=[], help='jsonl {name, text}: soft prompt pieces')
    p.add_argument('--registry-admits-reason', default='',
                   help='rows held only for a reason starting with this text are admitted by the registry entry')
    p.add_argument('--include-ids', type=Path, nargs='*', default=[],
                   help='jsonl rows with id / source_record_ids: render only these decisions')
    p.add_argument('--include-admits', action='store_true',
                   help='included rows count as approved (their admitted text set is the admission)')
    p.add_argument('--train-only', action='store_true', help='leave held rows out entirely (their held set lives elsewhere)')
    p.add_argument('--decision-exclude-ids', type=Path, nargs='*', default=[],
                   help='jsonl {id}: decisions already rendered elsewhere; used only inside whole trajectories')
    p.add_argument('--target-exclude-ids', type=Path, nargs='*', default=[],
                   help='jsonl {id}: quality-held targets; retain whole-trajectory context with zero native target weight')
    p.add_argument('--reserved', type=Path, help='explicit test reservations (training/source-split-reservations.json)')
    p.add_argument('--protected', type=Path)
    p.add_argument('--held-text', type=Path, nargs='*', default=[])
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--compress-text', action='store_true', help='compress shard and merged text while writing; exact decoded rows unchanged')
    p.add_argument('--sqlite-temp-root', type=Path,
                   help='local scratch directory for rebuildable per-shard SQLite indexes (durable outputs stay in --out-dir)')
    p.add_argument('--shards', type=int, default=64)
    p.add_argument('--limit', type=int)
    p.add_argument('--keep-work', action='store_true', help='keep the per-shard work files after the merge')
    p.add_argument('--renderings', nargs='+', choices=RENDERINGS, default=list(RENDERINGS),
                   help='renderings to produce (e.g. whole-trajectory alone for long agent trajectories, where one '
                        'document per decision repeats the context)')
    p.add_argument('--full-sample', type=float, default=0.05,
                   help='fraction of programs (stratified by family) that also get the full rendering, the warm-up '
                        'share; 1.0 renders full for every program (the old behaviour)')
    p.add_argument('--render-shards', nargs='*', default=None,
                   help='render only these existing selected shard stems; requires --render-only')
    p.add_argument('--render-only', action='store_true',
                   help='render the selected shard subset and leave merge/final publication for a later pass')
    p.add_argument('--finalize-only', action='store_true',
                   help='merge already completed shard outputs; fail unless every shard/tokenizer pair is done')
    p.add_argument('--legacy-renderer-sha256',
                   help='explicit prior renderer identity whose exact trajectory receipts may be reused on resume; '
                        'the final receipt preserves it separately from this renderer')
    p.add_argument('--resource-only-renderer-parent-sha256',
                   help='exact pre-2026-10-17 renderer SHA whose output receipts may continue across the reviewed '
                        'SQLite cleanup/fail-fast resource change')
    a = p.parse_args(argv)
    if a.render_only != (a.render_shards is not None):
        p.error('--render-only and --render-shards must be supplied together')
    if a.render_only and not a.render_shards:
        p.error('--render-only requires at least one --render-shards stem')
    if a.finalize_only and a.render_only:
        p.error('--finalize-only cannot be combined with --render-only')
    if a.legacy_renderer_sha256 is not None and not re.fullmatch(r'[0-9a-f]{64}', a.legacy_renderer_sha256):
        p.error('--legacy-renderer-sha256 must be a lowercase SHA-256 digest')
    if (a.resource_only_renderer_parent_sha256 is not None
            and a.resource_only_renderer_parent_sha256 != SEARCH_RESOURCE_ONLY_PARENT_RENDERER_SHA256):
        p.error('--resource-only-renderer-parent-sha256 must match the exact reviewed Search renderer parent')
    if a.verify_admission_only:
        verified = registered_candidate_selection_admission(a.corpus_id, a.source)
        if verified is None:
            p.error('--verify-admission-only requires registered candidate-selection admission')
        print(json.dumps(verified, sort_keys=True))
        return
    if not a.tokenizer or a.out_dir is None:
        p.error('rendering requires --tokenizer and --out-dir')
    tokenizers = dict(item.split('=', 1) for item in a.tokenizer)
    try:
        history_policies = _parse_history_reasoning_policies(a.history_reasoning_policy, tokenizers)
    except ValueError as exc:
        p.error(str(exc))
    pieces, piece_bindings = registered_source_pieces(a.source)
    a.candidate_selection_admission = registered_candidate_selection_admission(a.corpus_id, a.source)
    if a.candidate_selection_admission and a.include_admits:
        p.error('--include-admits cannot be combined with exact candidate-selection admission')
    explicit_piece_bindings = []
    for path in a.pieces:
        explicit_piece_bindings.append({'path': str(path), 'bytes': path.stat().st_size,
                                        'sha256': file_sha(path)})
        for line_number, line in enumerate(jsonl_lines(path), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            name, text = row.get('name'), row.get('text')
            if not isinstance(name, str) or not isinstance(text, str):
                p.error(f'invalid explicit piece row {path}:{line_number}')
            if name in pieces and pieces[name] != text:
                p.error(f'explicit piece table conflicts with registered source piece {name}')
            pieces[name] = text
    out = a.out_dir
    with acquire_output_lease(out):
        if (out / 'receipt.json').exists():
            p.error(f'refusing to overwrite a finished render: {out}')
        target_hold_policy = bind_target_hold_policy(out, a.source, a.target_exclude_ids)
        t0 = time.time()
        shard_dir = out / 'work' / 'selected'
        state_path = out / 'work' / 'selection-state.json'
        selection_binding = selection_input_binding(a)
        state = json.loads(state_path.read_text()) if state_path.exists() else None
        if state is not None and state.get('schema') == 'natlang.native-turn-selection-checkpoint/1':
            omissions, counts, blocked, protected_cache, test_groups, reserved_groups, full_programs = select(
                a, shard_dir, state_path, selection_binding)
            print(json.dumps({'resumed_selection': True, 'next_source': len(a.source)}), flush=True)
        elif state is not None:  # compatibility: older complete selection state
            omissions, counts, blocked = state['omissions'], Counter(state['counts']), state['blocked']
            protected_cache, test_groups, reserved_groups = state['protected'], set(state['test_groups']), set(state['reserved_groups'])
            # A selection made before --full-sample existed rendered (and keeps rendering) full for every program.
            full_programs = state.get('full_sample_programs')
            if 'full_sample_programs' not in state:
                a.full_sample = 1.0
            print(json.dumps({'resumed_selection': True}), flush=True)
        else:
            shutil.rmtree(out / 'work', ignore_errors=True)
            shard_dir.mkdir(parents=True, exist_ok=True)
            omissions, counts, blocked, protected_cache, test_groups, reserved_groups, full_programs = select(
                a, shard_dir, state_path, selection_binding)
        held_external = held_groups_from(a.held_text)
        # Train decisions sharing a source group with a reserved-test hit go with it (whole source groups, every split's
        # program of that group): they are decontaminated exactly like held groups.
        held_groups = test_groups | held_external | reserved_groups
        print(json.dumps({'selection': counts, 'internal_held_groups': len(test_groups),
                          'external_held_groups': len(held_external), 'seconds': round(time.time() - t0)}), flush=True)
        if a.compress_text:
            for selected in sorted(shard_dir.glob('*.jsonl')):
                subprocess.run(['zstd', '-q', '-3', '-T1', '--rm', '-f', str(selected)], check=True)
        shard_paths = {f.name.removesuffix('.jsonl').removesuffix('.jsonl.zst'): f for f in shard_dir.iterdir() if f.name.endswith(('.jsonl', '.jsonl.zst'))}
        shards = sorted(shard_paths)
        decision_exclude = set()
        for path in a.decision_exclude_ids:
            decision_exclude.update(json.loads(line)['id'] for line in jsonl_lines(path) if line.strip())
        target_exclude = set()
        for path in a.target_exclude_ids:
            target_exclude.update(json.loads(line)['id'] for line in jsonl_lines(path) if line.strip())
        full_set = None if full_programs is None else set(full_programs)
        _init(tokenizers, pieces, history_policies)
        input_binding = _render_input_binding(tokenizers, pieces, history_policies)
        all_jobs = [(str(shard_paths[s]), str(out / 'work'), name, blocked, a.corpus_id, held_groups, decision_exclude,
                     a.program_header_as_full, full_set, tuple(a.renderings), a.compress_text, target_exclude,
                     a.legacy_renderer_sha256, input_binding, str(a.sqlite_temp_root) if a.sqlite_temp_root else None,
                     a.resource_only_renderer_parent_sha256)
                for s in shards for name in tokenizers]
        if a.render_shards is not None:
            if len(a.render_shards) != len(set(a.render_shards)):
                p.error('--render-shards contains duplicate stems')
            try:
                jobs = _filter_render_jobs(all_jobs, a.render_shards)
            except ValueError as exc:
                p.error(str(exc))
        else:
            jobs = all_jobs
        results = []
        pending = []
        for job in jobs:
            done = out / 'work' / 'done' / job[2] / f"{Path(job[0]).name.removesuffix('.zst').removesuffix('.jsonl')}.json"
            if done.exists():
                result = json.loads(done.read_text())
                source_path = Path(job[0])
                source_digest = file_sha(source_path)
                if result.get('source_sha256') is not None and result['source_sha256'] != source_digest:
                    raise ValueError(f'completed shard source changed: {source_path}')
                if result.get('renderer_code_sha256') is None:
                    if a.legacy_renderer_sha256 is None:
                        raise ValueError(f'completed shard lacks renderer identity; supply the exact legacy SHA: {done}')
                    _verify_legacy_done_binding(job, result, a.legacy_renderer_sha256, source_digest)
                    # The legacy done marker has no per-group source hashes.
                    # Re-index this shard so the worker can recompute each old
                    # nested binding from actual rows before reusing receipts.
                    pending.append(job)
                    continue
                else:
                    current_renderer_sha = file_sha(Path(__file__).resolve())
                    parent_sha = a.resource_only_renderer_parent_sha256
                    parent_binding = None
                    if result['renderer_code_sha256'] != current_renderer_sha:
                        if parent_sha is None or result['renderer_code_sha256'] != parent_sha:
                            raise ValueError(f'completed shard renderer identity is neither current nor explicitly continued: {done}')
                        parent_binding = _render_job_binding_from_job(
                            job, source_digest, _resource_parent_code_binding(parent_sha))
                        if result.get('render_job_binding') != parent_binding:
                            raise ValueError(f'completed shard resource-parent job binding mismatch: {done}')
                    if result.get('source_sha256') is None:
                        raise ValueError(f'completed shard lacks its source identity: {done}')
                    allowed_renderer_shas = {current_renderer_sha}
                    if parent_sha is not None:
                        allowed_renderer_shas.add(parent_sha)
                    if a.legacy_renderer_sha256 is not None:
                        _legacy_render_job_binding(job, source_digest, a.legacy_renderer_sha256)
                        allowed_renderer_shas.add(a.legacy_renderer_sha256)
                    _verify_trajectory_receipts(job, result, allowed_renderer_shas,
                                                 expected_job_binding=parent_binding)
                result.setdefault('source_sha256', source_digest)
                results.append(result)
            else:
                pending.append(job)
        if a.finalize_only and pending:
            raise ValueError(f'cannot finalize: {len(pending)} shard/tokenizer outputs lack durable done receipts')
        print(json.dumps({'jobs': len(jobs), 'already_done': len(results)}), flush=True)
        jobs = [] if a.finalize_only else pending
        # ProcessPoolExecutor, not multiprocessing.Pool: a worker killed by the container's OOM killer makes the pool
        # raise BrokenProcessPool (the unit fails visibly and a relaunch resumes the finished shards) instead of
        # waiting forever for the lost shard (view-ask render, 2026-10-12).
        # Keep only one worker-width of tasks submitted at a time.  If a shard
        # fails, queued work can be cancelled and the parent reports the error
        # without first draining a corpus-sized executor queue.  Completed
        # shard receipts remain the durable resume frontier.
        with ProcessPoolExecutor(a.workers, initializer=_init,
                                 initargs=(tokenizers, pieces, history_policies)) as pool:
            def record_result(completed, res):
                results.append(res)
                if completed % 8 == 1:
                    print(json.dumps({'done': completed, 'of': len(jobs), 'last': res.get('shard'),
                                      'seconds': round(time.time() - t0)}), flush=True)
            _run_bounded_render_jobs(pool, jobs, a.workers, record_result)
        if a.render_only:
            print(json.dumps({'render_only_complete': True, 'shards': sorted(set(a.render_shards)),
                              'outputs': len(results)}), flush=True)
            return
        if not a.finalize_only and a.render_shards is not None:
            p.error('subset rendering requires --render-only')
        if len(results) != len(all_jobs):
            raise ValueError(f'finalization requires all {len(all_jobs)} shard/tokenizer receipts; found {len(results)}')
        renders = {}
        for mode in a.renderings:
            for name in tokenizers:
                renders[f'{mode}/{name}'] = merge(out, mode, name, shards, held_external,
                                                  role_counter(_G['tok'][name]), compress_text=a.compress_text)
                print(json.dumps({mode: name, **{k: v for k, v in renders[f'{mode}/{name}'].items() if k != 'roles'}}), flush=True)
        render_omissions = Counter()
        parity = Counter()
        for res in results:
            parity.update(res.get('parity') or {})
            render_omissions[f"{res['tokenizer']}/all/decontaminated"] += res.get('decontaminated', 0)
            render_omissions[f"{res['tokenizer']}/decision/already_rendered_elsewhere"] += res.get('decision_excluded', 0)
            render_omissions[f"{res['tokenizer']}/full/outside_warmup_sample"] += res.get('full_outside_warmup_sample', 0)
        for name in tokenizers:
            for f in sorted((out / 'work' / 'omissions' / name).glob('*.jsonl')):
                for line in jsonl_lines(f):
                    o = json.loads(line)
                    render_omissions[f"{name}/{o.get('rendering')}/{o['reason'].split(':', 1)[0]}"] += 1
        with open(out / 'selection-omissions.jsonl', 'w') as f:
            for o in omissions:
                f.write(canonical(o) + '\n')
        receipts = {k: v for res in results for k, v in res.items() if k.startswith('receipt_')}
        if bind_target_hold_policy(out, a.source, a.target_exclude_ids) != target_hold_policy:
            raise ValueError('render target hold policy changed before completion')
        receipt = {
            'schema': SCHEMA, 'corpus_id': a.corpus_id,
            'sources': {str(s): {'sha256': file_sha(s), 'bytes': s.stat().st_size} for s in a.source},
            'source_piece_bindings': piece_bindings,
            'source_piece_entries': len(pieces),
            'source_piece_table_sha256': sha(canonical(pieces).encode()) if pieces else None,
            'explicit_piece_bindings': explicit_piece_bindings,
            'exclude_programs': [{'path': str(e), 'sha256': file_sha(e)} for e in a.exclude_programs],
            'target_exclude_ids': (json.loads((out / target_hold_policy['file']).read_text())['target_exclude_ids']
                                   if target_hold_policy is not None else []),
            **({'target_hold_policy': target_hold_policy} if target_hold_policy is not None else {}),
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
                ROOT / 'training/prompt-pieces/pieces.json',
                ROOT / 'training/neuralese/natlang_neuralese/serve/chat.py')},
            'renderer_code_scope': 'finalizer code plus per-shard worker identities below',
            'shard_renderers': sorted(
                ({'shard': res['shard'], 'tokenizer': res['tokenizer'],
                  'source_sha256': res['source_sha256'],
                  'renderer_code_sha256': res['renderer_code_sha256'],
                  'group_renderer_code_sha256s': res.get('group_renderer_code_sha256s',
                                                         [res['renderer_code_sha256']])}
                 for res in results), key=lambda item: (item['shard'], item['tokenizer'])),
            'selection': dict(counts),
            'renderings': list(a.renderings),
            'full_sample': {'fraction': a.full_sample, 'programs': None if full_programs is None else len(full_programs),
                            'rule': 'full rendering only for a warm-up sample: per family the ceil(fraction*n) programs with '
                                    'the smallest sha256(program id), at least one per family; program-header and '
                                    'whole-trajectory cover every program (shared rule, coordinator 2026-10-11)'},
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
