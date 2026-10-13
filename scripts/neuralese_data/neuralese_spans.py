#!/usr/bin/env python3
"""Neuralese-output spans of a direct union rendering: the sidecar the mixed direct+e2e run reads.

Design: plans/neuralese/NEURALESE_DIRECT_V1.md, "The mixed run" and "Neuralese outputs in direct samples" (owner
2026-10-13). A Neuralese output is a value a record's target writes (conversion /16 producers, ``target_writes``); its
text is the producer's handover source (``handover_notes``), the text the e2e writer is sized from. A direct row (one
document of the union's ``text-data.jsonl``) names its native records in ``source_record_ids``, the same ids the e2e
cohort uses, so the join is exact.

Output, one JSON row per direct row that has at least one span:
    {"id": row id, "spans": [[begin, end, write_name, has_edge], ...]}
``begin``/``end`` are positions in the row's own ``token_ids`` (end exclusive) covering the written value inside the
row's supervised part; ``has_edge`` says whether the e2e cohort has a consumer edge for that write. The mixed run
makes edged spans context-only in direct updates (they train through e2e) and reads the others through the
TokenChoiceProjection with h detached.

    neuralese_spans.py --rows <text-data.jsonl> --cohort <e2e cohort dir> --tokenizer <dir> --out <neuralese-spans.jsonl>
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))


def value_forms(value: str) -> list[str]:
    """The written value as it can appear in a rendered row: raw, or JSON-escaped inside a string argument."""
    escaped = json.dumps(value, ensure_ascii=False)[1:-1]
    return [value] if escaped == value else [value, escaped]


def token_offsets(token_ids, text, tokenizer):
    """Character (start, end) of each token in ``text``. Re-tokenising ``text`` must reproduce ``token_ids`` (then
    the tokenizer's own offset mapping is exact); otherwise each token's decoded piece is laid end to end, which is
    exact for byte-level vocabularies whose pieces decode to whole characters, and the caller checks the result."""
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if list(encoded['input_ids']) == list(token_ids):
        return [tuple(o) for o in encoded['offset_mapping']], text
    pieces = [tokenizer.decode([t]) for t in token_ids]
    offsets, at = [], 0
    for piece in pieces:
        offsets.append((at, at + len(piece)))
        at += len(piece)
    return offsets, ''.join(pieces)


def spans_for_row(row: dict, writes: list[tuple[str, str, bool]], tokenizer) -> tuple[list, int]:
    """(spans, writes not found) of one direct row. ``writes``: (write name, value text, has edge) of the row's source
    records. A value is searched in the row's supervised part only (it also appears in later prompts as an input)."""
    token_ids = list(row['token_ids'])
    offsets, text = token_offsets(token_ids, row['text'], tokenizer)
    start_token = int(row.get('supervised_suffix_start') or 0)
    supervised_from = offsets[start_token][0] if start_token < len(offsets) else len(text)
    spans, missing = [], 0
    for name, value, has_edge in writes:
        if not isinstance(value, str) or not value:
            missing += 1
            continue
        found = None
        for form in value_forms(value):
            at = text.find(form, supervised_from)
            if at >= 0:
                found = (at, at + len(form))
                break
        if found is None:
            missing += 1
            continue
        covered = [i for i, (b, e) in enumerate(offsets) if b < found[1] and e > found[0]]
        if not covered:
            missing += 1
            continue
        spans.append([covered[0], covered[-1] + 1, name, bool(has_edge)])
    return sorted(spans), missing


def cohort_writes(cohort: Path) -> dict[str, list[tuple[str, str, bool]]]:
    """Record id -> (write name, value text, has edge) of every record in the cohort whose target writes."""
    from natlang_neuralese.train import trajectories as T
    edged = collections.defaultdict(set)
    for line in open(cohort / 'edges.jsonl'):
        edge = json.loads(line)
        edged[edge['writer_record_id']].add(edge['write_name'])
    writers = {}
    for line in open(cohort / 'records.jsonl'):
        record = json.loads(line)
        names = T.target_writes(record)
        if names:
            notes = T.handover_notes(record)
            writers[record['id']] = [(n, notes.get(n), n in edged.get(record['id'], ())) for n in sorted(names)]
    return writers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, action='append', required=True, help='e2e cohort dir (repeatable)')
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(argv)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer)
    writers = {}
    for cohort in a.cohort:
        for rid, writes in cohort_writes(cohort).items():
            writers.setdefault(rid, writes)
    stats = collections.Counter()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    partial = a.out.with_name(a.out.name + '.partial')
    with open(partial, 'w') as out:
        for line in open(a.rows):
            row = json.loads(line)
            stats['rows'] += 1
            writes = [w for rid in (row.get('source_record_ids') or [row.get('id')]) for w in writers.get(rid, ())]
            if not writes:
                continue
            spans, missing = spans_for_row(row, writes, tokenizer)
            stats['writer_rows'] += 1
            stats['writes'] += len(writes)
            stats['writes_not_found'] += missing
            stats['spans'] += len(spans)
            stats['span_tokens'] += sum(e - b for b, e, _, _ in spans)
            stats['span_tokens_with_edge'] += sum(e - b for b, e, _, edge in spans if edge)
            if spans:
                out.write(json.dumps({'id': row['id'], 'spans': spans}) + '\n')
    partial.replace(a.out)
    receipt = {'schema': 'natlang.neuralese-output-spans/1', 'rows': str(a.rows), 'cohorts': [str(c) for c in a.cohort],
               'tokenizer': a.tokenizer, 'out': str(a.out),
               'out_sha256': hashlib.sha256(a.out.read_bytes()).hexdigest(), **stats}
    a.out.with_name(a.out.stem + '.receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps(receipt), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
