#!/usr/bin/env python3
"""Bind regrade verdicts (ts-host/scripts/inline-curriculum/regrade-family-rows.mjs) into one exact correction record.

Input: one or more regrade output directories (verdicts.jsonl). Output OUT/correction.json
(natlang.oracle-regrade-correction/1): per row the source file (path and SHA-256 of the whole file), line number,
SHA-256 of the line, row/program/trajectory ids, split, family and verdict, grouped as

  withdraw  regraded episodes that fail the repaired oracle, and every derived row (training turn, graph) that
            inherits such a verdict: these lose positive admission;
  hold      rows whose verdict cannot be established exactly (unverifiable rebuild, unresolved parent): not positive
            until regraded;
  usable    passing episodes and the derived rows inheriting a pass;
  cases     program records (RL tasks, packets) whose stored oracle is outdated: regenerate their oracle side.

Source files are never modified; the record names exactly which rows a consumer must drop, hold or may use, with the
split evidence of each row. Consumers verify each row by (path, file SHA-256, line, line SHA-256).

  build_oracle_regrade_correction.py --out DIR REGRADE-DIR...
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402

SCHEMA = 'natlang.oracle-regrade-correction/1'
GROUP = {'fail': 'withdraw', 'pass': 'usable', 'unverifiable': 'hold', 'unresolved': 'hold',
         'oracle_outdated': 'cases', 'oracle_current': 'current_cases'}
KEEP = ('file', 'line', 'row_sha256', 'kind', 'row_id', 'program_id', 'trajectory_id', 'family', 'variant', 'split',
        'oracle_era', 'old_accepted', 'verdict', 'problems', 'inherited_from', 'corpora')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('inputs', nargs='+', type=Path)
    args = parser.parse_args()
    rows, seen = [], set()
    for directory in args.inputs:
        for line in (directory / 'verdicts.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            key = (row.get('file'), row.get('line'))
            if key in seen:
                continue  # the same file can be listed by several scans or corpora
            seen.add(key)
            rows.append(row)
    groups = collections.defaultdict(list)
    for row in rows:
        group = GROUP.get(row.get('verdict'))
        if group:
            groups[group].append({k: row[k] for k in KEEP if k in row})
    bound = sorted({r['file'] for g in ('withdraw', 'hold', 'usable', 'cases') for r in groups[g]})
    files = {path: sha256_file_hex(Path(path)) for path in bound}
    for group in groups.values():
        for row in group:
            row['file_sha256'] = files.get(row['file'])
    census = collections.Counter(f"{r.get('kind')}:{r.get('family')}:{r.get('verdict')}" for r in rows)
    by_corpus = collections.defaultdict(collections.Counter)
    for row in rows:
        for corpus in row.get('corpora') or []:
            by_corpus[corpus][f"{GROUP.get(row.get('verdict'), row.get('verdict'))}:{row.get('family')}"] += 1
    record = {'schema': SCHEMA, 'inputs': [str(p) for p in args.inputs],
              'regrader': 'ts-host/scripts/inline-curriculum/regrade-family-rows.mjs',
              'oracles': {'edit_stream': 'edit_fidelity (edit-fidelity/1)', 'contract_diagnosis': 'function-effects/1'},
              'rows': len(rows), 'census': dict(sorted(census.items())),
              'by_corpus': {k: dict(v) for k, v in sorted(by_corpus.items())},
              'withdraw': groups['withdraw'], 'hold': groups['hold'], 'usable': groups['usable'], 'cases': groups['cases']}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'correction.json').write_text(json.dumps(record, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: len(record[k]) for k in ('withdraw', 'hold', 'usable', 'cases')} | {'rows': len(rows)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
