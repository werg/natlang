#!/usr/bin/env python3
"""Add reference child notes to an existing task source's reader-framed training cases (span audit S1).

  add_child_notes.py --predecessor SOURCE_DIR --build BUILD_DIR [--build BUILD_DIR ...] --out NEW_SOURCE_DIR

A task source is immutable and merged from several builds, so it is not rebuilt. The source builders now emit
``curriculum.reference.children`` for their neuralese helper (``scripts/rl_tasks/child_notes.py``); this tool takes
those notes from freshly built raw cases and transplants them into the predecessor's rows, only where the rebuilt
row is identical to the predecessor row once the children are removed (same program, inputs, oracle, services and
asset hashes). A row that differs in any other way keeps no notes and is counted. Every other field, the held
packets and the assets stay byte-identical: ``held/`` and the assets are linked, not copied, and the receipt binds the
predecessor's ``cases.jsonl`` sha and the new one.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from scripts.rl_tasks.child_notes import coverage  # noqa: E402
from scripts.rl_tasks.sources import reader_notes  # noqa: E402
from natlang_neuralese.common.hashing import sha256_file_hex as sha_file  # noqa: E402

# Sources whose upstream rows are not on this machine: notes derived from the case's own inputs and gold answer.
DERIVED = {'hotpotqa': reader_notes.derived_document_notes, 'spider': reader_notes.derived_sql_notes,
           'bird': reader_notes.derived_sql_notes, 'spider-hard': reader_notes.derived_sql_notes,
           'bird-hard': reader_notes.derived_sql_notes, 'codebase-py': reader_notes.derived_file_notes,
           'codebase-natlang': reader_notes.derived_file_notes, 'clutrr-guided': reader_notes.derived_clutrr_notes,
           'babi-guided': reader_notes.derived_babi_notes, 'natlang-codebase': reader_notes.derived_function_notes}
# Derived from the case's own bound snapshot and gold even when a rebuilt row exists: the live repository has moved on.
DERIVE_FIRST = {'codebase-py', 'codebase-natlang'}


def digest_without_children(row):
    """Canonical digest of a row ignoring its reference children (the only field this tool may change)."""
    children = row['curriculum']['reference'].pop('children', None)
    try:
        return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    finally:
        if children is not None:
            row['curriculum']['reference']['children'] = children


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predecessor', required=True, type=Path)
    ap.add_argument('--build', required=True, type=Path, action='append')
    ap.add_argument('--out', required=True, type=Path)
    ap.add_argument('--assets-v3', type=Path, default=None, help='repository snapshots (default: PREDECESSOR/assets-v3)')
    a = ap.parse_args()
    context = {'assets_v3': a.assets_v3 or a.predecessor / 'assets-v3', 'cache': {}}
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; task sources are immutable')
    rebuilt = {}
    for build in a.build:
        for path in sorted(build.glob('*/raw-cases.jsonl')):
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    if row['split'] == 'train' and (row['curriculum']['reference'].get('children')):
                        rebuilt[row['id']] = (path.parent.name, row['curriculum']['reference']['children'],
                                              digest_without_children(row))
    stats = collections.defaultdict(collections.Counter)
    a.out.mkdir(parents=True)
    cases = a.out / 'cases.jsonl'
    coverage_by_family = collections.defaultdict(lambda: collections.Counter())
    rows_total = 0
    with open(a.predecessor / 'cases.jsonl') as handle, open(cases, 'w') as sink:
        for line in handle:
            row = json.loads(line)
            external = row['curriculum'].get('external') or {}
            if row['split'] == 'train' and row['id'] in rebuilt and external.get('source') not in DERIVE_FIRST:
                source, children, digest = rebuilt[row['id']]
                if not row['curriculum']['reference'].get('children') and digest_without_children(row) == digest:
                    row['curriculum']['reference']['children'] = children
                    stats[source]['transplanted'] += 1
                else:
                    stats[source]['rebuilt_row_differs'] += 1
            elif (row['split'] == 'train' and external.get('source') in DERIVED and external.get('framing') == 'neuralese-helper'
                  and not row['curriculum']['reference'].get('children')):
                children = [c for c in DERIVED[external['source']](row, context) if c]
                if children:
                    row['curriculum']['reference']['children'] = children
                    stats[external['source']]['derived'] += 1
                else:
                    stats[external['source']]['no_note'] += 1
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            rows_total += 1
            if row['split'] == 'train':
                one = coverage([row])
                if one['sites']:
                    for key in ('sites', 'covered', 'uncovered'):
                        coverage_by_family[row['family']][key] += one.get(key, 0)
                    for kind, count in one['provenance'].items():
                        coverage_by_family[row['family']]['provenance:' + kind] += count
    for entry in sorted(a.predecessor.iterdir()):
        if entry.name in ('cases.jsonl', 'receipt.json'):
            continue
        os.symlink(entry.resolve(), a.out / entry.name)
    receipt = json.loads((a.predecessor / 'receipt.json').read_text())
    receipt['predecessor'] = {'source_id': receipt.get('source_id'), 'path': str(a.predecessor.resolve()),
                              'cases_sha256': sha_file(a.predecessor / 'cases.jsonl'), 'rows': rows_total}
    receipt['source_id'] = a.out.name
    receipt['builder'] = 'scripts/rl_tasks/add_child_notes.py'
    receipt['cases'] = {**receipt['cases'], 'sha256': sha_file(cases), 'rows': rows_total}
    receipt['child_notes'] = {
        'purpose': 'reference.children notes for Neuralese producer helpers (NEURALESE_SPAN_BASIS_AUDIT_20261103 S1)',
        'builds': [str(b) for b in a.build], 'per_source': {k: dict(v) for k, v in sorted(stats.items())},
        'unchanged_apart_from': 'curriculum.reference.children; held, assets and every other file are the predecessor\'s'}
    (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    (a.out / 'child-notes-coverage.json').write_text(json.dumps({k: dict(v) for k, v in sorted(coverage_by_family.items())}, indent=1) + '\n')
    print(json.dumps({'rows': rows_total, 'per_source': receipt['child_notes']['per_source']}))


if __name__ == '__main__':
    main()
