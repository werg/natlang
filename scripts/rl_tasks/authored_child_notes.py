#!/usr/bin/env python3
"""Reference notes for the Neuralese producer helpers of authored RL tasks (fallow register S1).

  authored_child_notes.py backfill [--pool DIR] [--dry-run]     add notes to the admitted train rows that can carry them

An authored Neuralese task scripts, in its smoke, what each producer helper literally writes for the call it is given
(``smoke.children``; ``smoke.replay`` runs the whole root script with those answers and the oracle judges the result,
so the scripted notes are consistent with the checked ``expected``). Where such a scripted note is input-specific it is
the right target note and JOLT+ privileged context for that producer call, in the shape
``curriculum.reference.children = [{match, value, provenance}]`` that ``rl/privileged_context`` consumes:

* a per-call script (``{"$match": [{"match": <text in the call>, "value": <note>}, ...]}``): one entry per call, matched
  by the call's opening and the identifying fragment;
* a single scripted value: only where the smoke code calls the helper once and has no loop, so the value is that call's;
* only rows whose smoke is a full replay (the oracle judged the whole scripted run); a structure-only smoke may script
  placeholders;
* never a constant: a string note shorter than ``MIN_NOTE_CHARS``, a value shared by every call of a helper, or a placeholder
  is skipped (reported, not invented). A producer without a derivable note stays uncovered.

Provenance is ``checked`` (the note is part of a literal run whose final answer the oracle verified), never ``gold``.
Held rows are not touched. ``admit_authored`` calls ``derive`` for rows it admits.
"""
from __future__ import annotations

import argparse
import collections
import fcntl
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from child_notes import CALL, declared_sites, rendered  # noqa: E402
from measure_repetition import write_text_atomic  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

DEFAULT_POOL = resolve_str('data_hdd', 'natlang-rl-authoring', 'admitted')
MIN_NOTE_CHARS = 40
LOOP = re.compile(r'\b(for|while)\s*\(|\.(map|forEach|reduce|flatMap)\(|\biterateOn\b')


def smoke_code(row):
    smoke = row['curriculum']['external']['structure']['smoke']
    return '\n'.join(step[1].get('code', '') for step in smoke.get('root', []) if step and step[0] == 'eval')


def good_note(value):
    if isinstance(value, str):
        return len(value.strip()) >= MIN_NOTE_CHARS
    return isinstance(value, (dict, list)) and bool(value)


def derive(row):
    """(children, skipped) for one authored row: reference child entries and why other producer sites were skipped."""
    children, skipped = [], collections.Counter()
    scripted = row['curriculum']['external']['structure']['smoke'].get('children') or {}
    code = smoke_code(row)
    for site in declared_sites(row):
        value = scripted.get(site)
        if value is None:
            skipped['no_smoke_script'] += 1
        elif isinstance(value, dict) and '$match' in value:
            entries = [(item['match'], item['value']) for item in value['$match']]
            notes = [json.dumps(v, sort_keys=True) for _, v in entries]
            if len(entries) > 1 and len(set(notes)) == 1:
                skipped['identical_notes'] += 1
                continue
            for match, note in entries:
                if not isinstance(match, str) or not good_note(note):
                    skipped['constant_or_short_note'] += 1
                    continue
                children.append({'match': [f'{CALL}{site}(', rendered(match)], 'value': note, 'provenance': 'checked'})
        else:
            calls = len(re.findall(rf'\b{re.escape(site)}\s*\(', code))
            if calls != 1 or LOOP.search(code):
                skipped['single_script_for_repeated_calls'] += 1
            elif not good_note(value):
                skipped['constant_or_short_note'] += 1
            else:
                children.append({'match': [f'{CALL}{site}('], 'value': value, 'provenance': 'checked'})
    return children, dict(skipped)


def with_notes(row):
    """The row with derived notes attached (train rows with producers only), and the counters."""
    if row['split'] != 'train' or not (row.get('curriculum') or {}).get('external', {}).get('structure'):
        return row, None
    if not row['curriculum']['external']['structure']['smoke'].get('replay'):
        return row, {'smoke_not_replay_verified': 1}  # a structure-only smoke script may hold placeholder notes
    if (row['curriculum'].get('reference') or {}).get('children'):
        return row, {'row_has_scripted_reference_children': 1}
    children, skipped = derive(row)
    if not children:
        return row, skipped
    reference = dict(row['curriculum']['reference'])
    reference['children'] = children
    return {**row, 'curriculum': {**row['curriculum'], 'reference': reference}}, skipped


def backfill(pool, dry_run=False):
    pool = Path(pool)
    with open(pool / 'cases.jsonl.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in (pool / 'cases.jsonl').read_text().splitlines() if line.strip()]
        out, stats, skipped = [], collections.Counter(), collections.Counter()
        for row in rows:
            sites = declared_sites(row) if row['split'] == 'train' else []
            new, why = with_notes(row)
            out.append(new)
            if not sites:
                continue
            stats['rows_with_producers'] += 1
            found = [re.match(re.escape(CALL) + r'([\w$]+)\(', c['match'][0]) for c in
                     (new['curriculum']['reference'].get('children') or [])]
            covered = {m.group(1) for m in found if m}
            stats['sites'] += len(sites)
            stats['sites_covered'] += len(covered & set(sites))
            stats['rows_with_notes'] += bool(covered)
            skipped.update(why or {})
        summary = {'rows': len(rows), **stats, 'skipped_sites': dict(skipped)}
        if not dry_run:
            write_text_atomic(pool / 'cases.jsonl', ''.join(json.dumps(row) + '\n' for row in out))
            with open(pool / 'admissions.jsonl', 'a') as log:
                log.write(json.dumps({'unix': __import__('time').time(), 'status': 'child-notes-backfill',
                                      **summary}) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    fill = sub.add_parser('backfill')
    fill.add_argument('--pool', default=DEFAULT_POOL)
    fill.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    print(json.dumps(backfill(args.pool, args.dry_run)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
