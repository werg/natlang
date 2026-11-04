#!/usr/bin/env python3
"""Reference notes for the Neuralese producer helpers of authored RL tasks (fallow register S1).

  authored_child_notes.py backfill [--pool DIR] [--dry-run]     add notes to the admitted train rows that can carry them
  authored_child_notes.py mechanical TASKS.jsonl [--pool DIR] [--dry-run]
                                                                 attach per-call notes of a corrected tasks file to admitted rows
                                                                 whose program cannot replay (provenance ``mechanical``)
  authored_child_notes.py openers [--pool DIR] [--dry-run]
                                                                 attach the literal start note of zero-argument producer
                                                                 helpers (``begin_notes()``; provenance mechanical:opener-text)
  authored_child_notes.py extract [--pool DIR] [--batch NAME ...] [--dry-run]
                                                                 attach argument-extract notes to admitted train rows whose
                                                                 ``Neuralese<string>`` producers have none, from the call
                                                                 arguments their smoke script passes (round 6)

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
from child_notes import CALL, declared_sites, fragment, rendered, string_key  # noqa: E402
from measure_repetition import write_text_atomic  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

DEFAULT_POOL = resolve_str('data_hdd', 'natlang-rl-authoring', 'admitted')
MIN_NOTE_CHARS = 40
LOOP = re.compile(r'\b(for|while)\s*\(|\.(map|forEach|reduce|flatMap)\(|\biterateOn\b')


def smoke_code(row):
    smoke = row['curriculum']['external']['structure']['smoke']
    return '\n'.join(step[1].get('code', '') for step in smoke.get('root', []) if step and step[0] == 'eval')


NAMED_ARG = re.compile(r'^([A-Za-z_]\w*): string = "(.*)"$', re.S)


def match_key(match):
    """The call-opening text a scripted ``match`` stands for: a raw fragment of an argument (escaped as the runtime prints a
    string value), or the named form ``name: string = "value"`` of a whole string argument."""
    named = NAMED_ARG.match(match)
    return string_key(named.group(1), named.group(2)) if named else rendered(match)


def site_of(entry):
    """The producer helper a reference child entry answers (None for a root-script child that matches other text)."""
    match = entry.get('match')
    match = [match] if isinstance(match, str) else match or []
    found = re.match(re.escape(CALL) + r'([\w$]+)\(', match[0]) if match else None
    return found.group(1) if found else None


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
                children.append({'match': [f'{CALL}{site}(', match_key(match)], 'value': note, 'provenance': 'checked'})
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


def mechanical(pool, tasks_path, dry_run=False, provenance='mechanical:argument-extract', replace=None):
    """Attach input-specific per-call notes from ``tasks_path`` (a corrected copy of an admitted batch whose only change is
    ``smoke.children``) to the admitted rows of the same task ids that have none. For programs that cannot be a full
    replay (their root reads typed Neuralese fields in code) the notes are deterministic extracts of each call's own
    arguments, not verified producer outputs: provenance is ``mechanical:argument-extract``, never ``checked``. The
    same exclusions as ``derive`` apply (constants, short strings, identical notes, repeated calls with one script).

    ``replace`` names a provenance (``mechanical:rule-derived``) whose existing notes the new ones supersede for the
    sites they cover: used to upgrade rule-derived notes to ``checked`` once a whole-root replay of the task (plain-typed
    copy, every producer answering the scripted record, the oracle judging the root's final answer) has passed."""
    pool = Path(pool)
    tasks = {}
    for line in Path(tasks_path).read_text().splitlines():
        if line.strip():
            task = json.loads(line)
            tasks[task['id']] = task
    with open(pool / 'cases.jsonl.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in (pool / 'cases.jsonl').read_text().splitlines() if line.strip()]
        out, stats, skipped = [], collections.Counter(), collections.Counter()
        for row in rows:
            task = tasks.get(row['id'].split(':')[2]) if row['id'].count(':') >= 3 else None
            existing = (row['curriculum'].get('reference') or {}).get('children') or []
            have = {site for site in (site_of(c) for c in existing if not (replace and c.get('provenance') == replace)) if site}
            if replace:
                swappable = {site_of(c) for c in existing if c.get('provenance') == replace}
                if task is not None and not swappable:
                    out.append(row)
                    continue
            if (task is None or row['split'] != 'train' or not (row['curriculum'].get('external') or {}).get('structure')
                    or not set(declared_sites(row)) - have):
                out.append(row)
                continue
            stats['candidate_rows'] += 1
            probe = json.loads(json.dumps(row))
            # Only per-call scripts: a single constant is never a note (derive would accept it for a one-call smoke).
            per_call = {name: value for name, value in task['smoke']['children'].items()
                        if isinstance(value, dict) and '$match' in value and name not in have}
            probe['curriculum']['external']['structure']['smoke']['children'] = per_call
            children, why = derive(probe)
            skipped.update(why)
            if not children:
                out.append(row)
                continue
            covered = {site_of(c) for c in children}
            kept = [c for c in existing if not (replace and c.get('provenance') == replace and site_of(c) in covered)]
            children = kept + [{**child, 'provenance': provenance} for child in children]
            reference = dict(row['curriculum']['reference'])
            reference['children'] = children
            out.append({**row, 'curriculum': {**row['curriculum'], 'reference': reference}})
            stats['rows_with_notes'] += 1
            stats['notes'] += len(children)
        summary = {'rows': len(rows), **stats, 'skipped_sites': dict(skipped), 'tasks_file': str(tasks_path)}
        if not dry_run and stats['rows_with_notes']:
            write_text_atomic(pool / 'cases.jsonl', ''.join(json.dumps(row) + '\n' for row in out))
            with open(pool / 'admissions.jsonl', 'a') as log:
                log.write(json.dumps({'unix': __import__('time').time(), 'status': 'child-notes-mechanical',
                                      **summary}) + '\n')
    return summary


ARG_PATH = re.compile(r'^([A-Za-z_]\w*)((?:\.[A-Za-z_]\w*|\[\d+\])*)$')
EXTRACT_PROVENANCE = 'mechanical:argument-extract'
EXTRACT_CHARS = 400


def header_of(text):
    return text.split('---')[1] if text.startswith('---') else ''


def declared_arg_names(text):
    names, in_args = [], False
    for line in header_of(text).splitlines():
        if re.match(r'^args:\s*$', line):
            in_args = True
        elif in_args and re.match(r'^\s+[A-Za-z_$][\w$]*\s*:', line):
            names.append(line.split(':', 1)[0].strip())
        elif in_args and line and not line.startswith(' '):
            in_args = False
    return names


def expand(value, tokens):
    """Every value the access path ``tokens`` (field names, and indices as wildcards over the list) reaches."""
    if not tokens:
        return [value]
    head, rest = tokens[0], tokens[1:]
    if head.startswith('['):
        return [leaf for item in (value if isinstance(value, list) else []) for leaf in expand(item, rest)]
    return expand(value[head], rest) if isinstance(value, dict) and head in value else []


def extract_text(name, value):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=False)
    text = re.sub(r'\s+', ' ', text).strip()
    return f'{name}: {text}'[:EXTRACT_CHARS].rstrip()


def argument_extract(row):
    """(children, skipped) for one admitted row: for every ``Neuralese<string>`` producer without a note, one note per
    distinct call the smoke script's argument expressions reach (an indexed access such as ``turns[0]`` stands for the whole
    list, since the root's loop calls the helper per element). A note is the call's own plain arguments, named, in order:
    an exact extract (identifiers and values), never a summary the helper would write; carried-note arguments are not
    repeated. Typed (record) producers and calls whose arguments the smoke script does not express are skipped."""
    from authored_checks import call_arguments
    children, skipped = [], collections.Counter()
    files = row['semantics']['files']
    inputs = row['semantics'].get('inputs') or {}
    code = smoke_code(row)
    sites = declared_sites(row)
    for site in sites:
        path = next(p for p in files if p.rsplit('/', 1)[-1] == site + '.nl' and p != row['semantics']['root'])
        returns = re.search(r'^returns:\s*(.*)$', header_of(files[path]), re.M).group(1)
        if 'Neuralese<string>' not in returns:
            skipped['typed_producer'] += 1
            continue
        names = declared_arg_names(files[path])
        seen, calls = set(), []
        for match in re.finditer(rf'\b{re.escape(site)}\s*\(', code):
            arguments = call_arguments(code, match.end() - 1)
            if arguments is None or len(arguments) != len(names):
                continue
            per_arg = []
            for name, expression in zip(names, arguments):
                found = ARG_PATH.match(expression.strip())
                if found and found.group(1) in inputs:
                    tokens = re.findall(r'\.[A-Za-z_]\w*|\[\d+\]', found.group(2))
                    per_arg.append([(name, v) for v in expand(inputs[found.group(1)], [t[1:] if t[0] == '.' else t for t in tokens])])
                elif found and re.fullmatch(r'[A-Za-z_]\w*', expression.strip()):
                    per_arg.append(None)  # a variable: the carried note (or a derived value): not repeated
                else:
                    try:
                        per_arg.append([(name, json.loads(expression))])
                    except ValueError:
                        per_arg.append(None)
            if all(option is None for option in per_arg):
                continue
            width = max(len(option) for option in per_arg if option is not None)
            for index in range(width):
                call = tuple(option[min(index, len(option) - 1)] for option in per_arg if option)
                key = json.dumps(call, sort_keys=True, ensure_ascii=False)
                if call and key not in seen and all(option for option in per_arg if option is not None):
                    seen.add(key)
                    calls.append(call)
        if not calls:
            skipped['no_expressible_call'] += 1
            continue
        fragments = set()
        for call in calls:
            first = next((v for _, v in call if isinstance(v, str)), None)
            text = ' | '.join(extract_text(n, v) for n, v in call)
            if len(calls) == 1:
                keys = [f'{CALL}{site}(']
            elif first is None:
                skipped['unmatchable_call'] += 1
                continue
            else:
                keys = [f'{CALL}{site}(', fragment(first, limit=90)]
                if keys[1] is None or keys[1] in fragments:
                    skipped['ambiguous_match'] += 1
                    continue
                fragments.add(keys[1])
            if len(text) >= 12:
                children.append({'match': keys, 'value': text, 'provenance': EXTRACT_PROVENANCE})
    return children, dict(skipped)


def extract(pool, batches=None, dry_run=False):
    """Attach ``argument_extract`` notes to admitted train rows with no reference children yet (optionally only rows
    admitted from the named batches, e.g. ``w33``)."""
    pool = Path(pool)
    wanted = set(batches or [])
    ids = set()
    if wanted:
        for line in (pool / 'admissions.jsonl').read_text().splitlines():
            entry = json.loads(line)
            if entry.get('status') == 'admitted' and entry['batch'].split('/')[0] in wanted:
                ids |= set(entry['admitted_case_ids'])
    with open(pool / 'cases.jsonl.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in (pool / 'cases.jsonl').read_text().splitlines() if line.strip()]
        out, stats, skipped = [], collections.Counter(), collections.Counter()
        for row in rows:
            sites = declared_sites(row) if row['split'] == 'train' and (not wanted or row['id'] in ids) else []
            if (not sites or (row['curriculum'].get('reference') or {}).get('children')
                    or not (row['curriculum'].get('external') or {}).get('structure')):
                out.append(row)
                continue
            children, why = argument_extract(row)
            skipped.update(why)
            stats['candidate_rows'] += 1
            stats['sites'] += len(sites)
            covered = {re.match(re.escape(CALL) + r'([\w$]+)\(', c['match'][0]).group(1) for c in children}
            stats['sites_covered'] += len(covered)
            if not children:
                out.append(row)
                continue
            reference = dict(row['curriculum']['reference'])
            reference['children'] = children
            out.append({**row, 'curriculum': {**row['curriculum'], 'reference': reference}})
            stats['rows_with_notes'] += 1
            stats['notes'] += len(children)
        summary = {'rows': len(rows), **stats, 'skipped': dict(skipped), 'batches': sorted(wanted)}
        if not dry_run and stats['rows_with_notes']:
            write_text_atomic(pool / 'cases.jsonl', ''.join(json.dumps(row) + '\n' for row in out))
            with open(pool / 'admissions.jsonl', 'a') as log:
                log.write(json.dumps({'unix': __import__('time').time(), 'status': 'child-notes-extract', **summary}) + '\n')
    return summary


OPENER_PROVENANCE = 'mechanical:opener-text'
OPENER_MIN_CHARS = 12


def zero_argument_sites(row):
    files = row['semantics']['files']
    found = []
    for site in declared_sites(row):
        path = next((p for p in files if p.rsplit('/', 1)[-1] == site + '.nl' and p != row['semantics']['root']), None)
        if path is not None and not declared_arg_names(files[path]):
            found.append(site)
    return found


def openers(pool, dry_run=False):
    """Attach the literal start note of each zero-argument producer helper (``begin_notes()``) that has none. Such a call
    receives no argument, so nothing in its opening can tell calls apart and no note can depend on the call's input: the
    note is the opening the helper's own text prescribes, scripted by the author in the task's smoke (a string of at least
    ``OPENER_MIN_CHARS``). It is program-level content, provenance ``mechanical:opener-text``, matched by the call's
    signature alone (a task calling the opener once per item gets the same start note each time, as the helper would write).
    Typed and unscripted openers stay uncovered."""
    pool = Path(pool)
    with open(pool / 'cases.jsonl.lock', 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in (pool / 'cases.jsonl').read_text().splitlines() if line.strip()]
        out, stats = [], collections.Counter()
        for row in rows:
            structure = (row['curriculum'].get('external') or {}).get('structure')
            if row['split'] != 'train' or not structure:
                out.append(row)
                continue
            existing = (row['curriculum'].get('reference') or {}).get('children') or []
            have = {site_of(c) for c in existing}
            scripted = structure['smoke'].get('children') or {}
            added = []
            for site in zero_argument_sites(row):
                stats['zero_argument_sites'] += 1
                if site in have:
                    stats['already_covered'] += 1
                    continue
                note = scripted.get(site)
                if not isinstance(note, str) or len(note.strip()) < OPENER_MIN_CHARS:
                    stats['no_literal_note'] += 1
                    continue
                added.append({'match': [f'{CALL}{site}('], 'value': note.strip(), 'provenance': OPENER_PROVENANCE})
            if not added:
                out.append(row)
                continue
            reference = dict(row['curriculum']['reference'])
            reference['children'] = existing + added
            out.append({**row, 'curriculum': {**row['curriculum'], 'reference': reference}})
            stats['rows_with_notes'] += 1
            stats['notes'] += len(added)
        summary = {'rows': len(rows), **stats}
        if not dry_run and stats['rows_with_notes']:
            write_text_atomic(pool / 'cases.jsonl', ''.join(json.dumps(row) + '\n' for row in out))
            with open(pool / 'admissions.jsonl', 'a') as log:
                log.write(json.dumps({'unix': __import__('time').time(), 'status': 'child-notes-openers', **summary}) + '\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    fill = sub.add_parser('backfill')
    fill.add_argument('--pool', default=DEFAULT_POOL)
    fill.add_argument('--dry-run', action='store_true')
    mech = sub.add_parser('mechanical')
    mech.add_argument('tasks')
    mech.add_argument('--pool', default=DEFAULT_POOL)
    mech.add_argument('--dry-run', action='store_true')
    mech.add_argument('--replace', metavar='PROVENANCE',
                      help='replace the existing notes of this provenance for the sites the tasks file covers (e.g. '
                           'mechanical:rule-derived after a passed whole-root replay)')
    mech.add_argument('--provenance', default='mechanical:argument-extract',
                      help='mechanical:argument-extract (extracts of the call arguments) or mechanical:rule-derived (the helper\'s '
                           'stated rule implemented in code over the call arguments, cross-checked against the task\'s own data)')
    opn = sub.add_parser('openers', help='attach the literal start note of zero-argument producer helpers')
    opn.add_argument('--pool', default=DEFAULT_POOL)
    opn.add_argument('--dry-run', action='store_true')
    ext = sub.add_parser('extract')
    ext.add_argument('--pool', default=DEFAULT_POOL)
    ext.add_argument('--batch', action='append', default=[])
    ext.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if args.cmd == 'openers':
        print(json.dumps(openers(args.pool, args.dry_run)))
        return 0
    if args.cmd == 'extract':
        print(json.dumps(extract(args.pool, args.batch, args.dry_run)))
        return 0
    if args.cmd == 'mechanical':
        print(json.dumps(mechanical(args.pool, args.tasks, args.dry_run, args.provenance, args.replace)))
        return 0
    print(json.dumps(backfill(args.pool, args.dry_run)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
