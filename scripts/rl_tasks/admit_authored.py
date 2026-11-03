#!/usr/bin/env python3
"""Admit a qualified, audited batch of authored RL tasks into the admitted pool (plans/neuralese/RL_TASK_AUTHORING_KIT.md).

  admit_authored.py BATCH_DIR --qualified qualified-2 [--novelty novelty.json] [--drop ID ...]
                    [--supersede CASE_ID ...] --reviewer TEXT [--note TEXT] [--pool DIR] [--dry-run]

Preconditions (all checked, none relaxed): the qualification report says PASS and every admitted row passed every step;
the novelty report says PASS. Admitted rows are the batch's verified converted cases, minus ``--drop`` task ids (held
back by the audit), appended to ``POOL/cases.jsonl`` under the pool's lock and written atomically (temporary file,
fsync, rename). A case id already in the pool is refused unless it is named in ``--supersede``, which replaces that row
(a corrected copy of a task). ``POOL/admissions.jsonl`` gets one entry with the file hashes, the reviewer and the ids.
Default POOL is natlang-rl-authoring/admitted under the data_hdd root (common.paths).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_repetition import write_text_atomic  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

DEFAULT_POOL = resolve_str('data_hdd', 'natlang-rl-authoring', 'admitted')


def read_rows(path):
    path = Path(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def admit(batch, qualified, novelty, drop, supersede, reviewer, note, pool, dry_run=False):
    batch, pool = Path(batch), Path(pool)
    qdir = batch / qualified
    report = json.loads((qdir / 'report.json').read_text())
    if report.get('result') != 'PASS':
        raise SystemExit(f'qualification is not PASS: {qdir / "report.json"}')
    novelty_path = batch / novelty if novelty else None
    if novelty_path is not None and json.loads(novelty_path.read_text()).get('result') != 'PASS':
        raise SystemExit(f'novelty is not PASS: {novelty_path}')
    status = report['tasks_status']
    rows = read_rows(qdir / 'verified' / 'cases.jsonl')
    dropped = set(drop or [])
    admitted, held = [], []
    for row in rows:
        entry = status.get(row['id'])
        if entry is None or entry.get('problems'):
            raise SystemExit(f'case {row["id"]} has no passing qualification entry')
        (held if entry['task'] in dropped else admitted).append(row)
    unknown = dropped - {status[row['id']]['task'] for row in rows}
    if unknown:
        raise SystemExit(f'--drop names tasks not in the batch: {sorted(unknown)}')
    supersede = set(supersede or [])
    pool.mkdir(parents=True, exist_ok=True)
    cases_path = pool / 'cases.jsonl'
    with open(pool / 'cases.jsonl.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        existing = read_rows(cases_path)
        have = {row['id'] for row in existing}
        missing = supersede - have
        if missing:
            raise SystemExit(f'--supersede names ids not in the pool: {sorted(missing)}')
        clashes = [row['id'] for row in admitted if row['id'] in have and row['id'] not in supersede]
        if clashes:
            raise SystemExit(f'case ids already admitted: {clashes[:5]}')
        merged = [row for row in existing if row['id'] not in supersede] + admitted
        summary = {'batch': str(batch), 'qualified': qualified, 'admitted': len(admitted), 'held': len(held),
                   'superseded': len(supersede), 'pool_rows': len(merged)}
        if dry_run:
            return summary
        write_text_atomic(cases_path, ''.join(json.dumps(row, sort_keys=False) + '\n' for row in merged))
        entry = {'unix': time.time(), 'batch': '/'.join(batch.parts[-2:]), 'batch_dir': str(batch), 'status': 'admitted',
                 'qualified': qualified, 'report_sha256': sha256(qdir / 'report.json'),
                 'novelty_sha256': sha256(novelty_path) if novelty_path else None, 'reviewer': reviewer, 'note': note,
                 'admitted_case_ids': [row['id'] for row in admitted], 'dropped_tasks': sorted(dropped),
                 'superseded_case_ids': sorted(supersede), 'pool_sha256': sha256(cases_path)}
        with open(pool / 'admissions.jsonl', 'a') as log:
            log.write(json.dumps(entry) + '\n')
    summary['pool_sha256'] = entry['pool_sha256']
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('batch')
    parser.add_argument('--qualified', default='qualified')
    parser.add_argument('--novelty')
    parser.add_argument('--drop', action='append', default=[])
    parser.add_argument('--supersede', action='append', default=[])
    parser.add_argument('--reviewer', required=True)
    parser.add_argument('--note', default='')
    parser.add_argument('--pool', default=DEFAULT_POOL)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(admit(args.batch, args.qualified, args.novelty, args.drop, args.supersede, args.reviewer,
                           args.note, args.pool, args.dry_run)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
