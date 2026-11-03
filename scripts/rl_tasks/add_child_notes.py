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

# Sources whose upstream rows are not on this machine: notes derived from the case's own inputs and gold answer.
DERIVED = {'hotpotqa': reader_notes.derived_document_notes, 'spider': reader_notes.derived_sql_notes,
           'bird': reader_notes.derived_sql_notes, 'spider-hard': reader_notes.derived_sql_notes,
           'bird-hard': reader_notes.derived_sql_notes}


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def without_children(row):
    copy = json.loads(json.dumps(row))
    copy['curriculum']['reference'].pop('children', None)
    return copy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--predecessor', required=True, type=Path)
    ap.add_argument('--build', required=True, type=Path, action='append')
    ap.add_argument('--out', required=True, type=Path)
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit(f'{a.out} exists; task sources are immutable')
    rebuilt = {}
    for build in a.build:
        for path in sorted(build.glob('*/raw-cases.jsonl')):
            with open(path) as handle:
                for line in handle:
                    row = json.loads(line)
                    if row['split'] == 'train' and (row['curriculum']['reference'].get('children')):
                        rebuilt[row['id']] = (path.parent.name, row)
    stats = collections.defaultdict(collections.Counter)
    rows_out = []
    notes_added = []
    with open(a.predecessor / 'cases.jsonl') as handle:
        for line in handle:
            row = json.loads(line)
            if row['split'] == 'train' and row['id'] in rebuilt:
                source, new = rebuilt[row['id']]
                if (without_children(new) == row and not row['curriculum']['reference'].get('children')):
                    row['curriculum']['reference']['children'] = new['curriculum']['reference']['children']
                    stats[source]['transplanted'] += 1
                    notes_added.append(row)
                else:
                    stats[source]['rebuilt_row_differs'] += 1
            elif (row['split'] == 'train' and row['curriculum']['external']['source'] in DERIVED
                  and row['curriculum']['external'].get('framing') == 'neuralese-helper'
                  and not row['curriculum']['reference'].get('children')):
                source = row['curriculum']['external']['source']
                children = [c for c in DERIVED[source](row) if c]
                if children:
                    row['curriculum']['reference']['children'] = children
                    stats[source]['derived'] += 1
                else:
                    stats[source]['no_note'] += 1
            rows_out.append(row)
    a.out.mkdir(parents=True)
    cases = a.out / 'cases.jsonl'
    with open(cases, 'w') as handle:
        for row in rows_out:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    for entry in sorted(a.predecessor.iterdir()):
        if entry.name in ('cases.jsonl', 'receipt.json'):
            continue
        os.symlink(entry.resolve(), a.out / entry.name)
    receipt = json.loads((a.predecessor / 'receipt.json').read_text())
    receipt['predecessor'] = {'source_id': receipt.get('source_id'), 'path': str(a.predecessor.resolve()),
                              'cases_sha256': sha_file(a.predecessor / 'cases.jsonl'), 'rows': len(rows_out)}
    receipt['source_id'] = a.out.name
    receipt['builder'] = 'scripts/rl_tasks/add_child_notes.py'
    receipt['cases'] = {**receipt['cases'], 'sha256': sha_file(cases), 'rows': len(rows_out)}
    receipt['child_notes'] = {
        'purpose': 'reference.children notes for Neuralese producer helpers (NEURALESE_SPAN_BASIS_AUDIT_20261103 S1)',
        'builds': [str(b) for b in a.build], 'per_source': {k: dict(v) for k, v in sorted(stats.items())},
        'unchanged_apart_from': 'curriculum.reference.children; held, assets and every other file are the predecessor\'s'}
    (a.out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    by_family = collections.defaultdict(list)
    for row in rows_out:
        if row['split'] == 'train':
            by_family[row['family']].append(row)
    report = {family: coverage(rows) for family, rows in sorted(by_family.items())}
    report = {family: value for family, value in report.items() if value['sites']}
    (a.out / 'child-notes-coverage.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({'rows': len(rows_out), 'transplanted': len(notes_added), 'per_source': receipt['child_notes']['per_source']}))


if __name__ == '__main__':
    main()
