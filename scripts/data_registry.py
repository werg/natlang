#!/usr/bin/env python3
"""The training-data registry from the shell (library: natlang_neuralese.common.data_registry).

  record   --id ID --path PATH --kind KIND --format TEXT [--include GLOB].. [--input ID|PATH].. [--license TEXT]
           [--protected] [--open] [--note TEXT] [--supersedes ID]        a producer snapshot (shell producers, by hand)
  consume  --run RUN --how TEXT ITEM..                                   a consumer record (shell consumers)
  show ID | find PATH | consumers ID | list [--kind K] [--layer L] | role NAME | protected
  check    [PATH..]       exit 1 when a data path (default: the registry roles and the data directories of the roots)
                          is in no dataset
  promote  ID.. [--role NAME] [--all-closed] [--dry-run]   fold closed journal snapshots into
                          training/corpora (scripts/neuralese_data/register_corpora.py commits them
                          unattended); --role grpo-pool admits a task-supply source into training
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.common import data_registry as dr  # noqa: E402



def role_spec(reg, role, order, role_file=None):
    """A promoted dataset's role entry. Its file follows the role's current members when none is given (grpo-pool and
    grpo-memetic-tasks read cases.jsonl; grpo-sd-teacher reads sft_sources WORK directories); a new role reads
    cases.jsonl."""
    if role_file is None:
        files = [((e.get('roles') or {}).get(role) or {}).get('file') for e in reg.git.values()
                 if isinstance((e.get('roles') or {}).get(role), dict)]
        role_file = collections.Counter(files).most_common(1)[0][0] if files else 'cases.jsonl'
    return {'order': order, **({'file': role_file} if role_file else {})}

def _print(value):
    print(json.dumps(value, indent=1, ensure_ascii=False, default=str))


# Where data lives by its naming (the check's default scan): each pattern names a dataset directory, or a marker file
# whose directory is one. Anything matching that no dataset contains (or only an archive inventory) is reported.
DATA_PATTERNS = {
    'data_hdd': ('natlang-rl-*/cases.jsonl', 'natlang-rl-*/*/cases.jsonl', 'natlang-grpo-v*/run*/rollouts',
                 'external-rl-sources/*/'),
    'data_nvme': ('memetic-tasks-*/cases.jsonl', 'sft-sources-*/', 'sft-teacher-demos-*/', 'hindsight-index-*/index.jsonl',
                  'natlang-operator-rows-*/manifest.json', 'vibestudio-live-corpus/seed-*/',
                  'lfm3-rl-dgx-20261029/ops/teacher-gen-*/*/manifest.jsonl', 'lfm3-rl-dgx-20261029/ops/held-v*/manifest.json',
                  'lfm3-rl-dgx-20261029/ops/grpo-v*/run*/rollouts', 'lfm3-rl-dgx-20261029/ops/operators-v1/merged/'),
}


def default_check_paths(reg):
    import glob
    from natlang_neuralese.common.paths import root
    paths = []
    for role in ('grpo-pool', 'grpo-memetic-tasks', 'grpo-sd-teacher', 'held-eval'):
        paths += [str(p) for _, p in reg.role(role)]
    for name, patterns in DATA_PATTERNS.items():
        for pattern in patterns:
            for hit in sorted(glob.glob(str(root(name) / pattern))):
                d = hit.rstrip('/') if os.path.isdir(hit) else os.path.dirname(hit)
                if not os.path.basename(d).startswith(('pin-', 'tmp', '.')) and '/pin-' not in d:
                    paths.append(d)
    return list(dict.fromkeys(paths))


COVERAGE = REPO / 'training/data-registry-coverage-20261120.json'


def coverage_paths(path=COVERAGE):
    """The local paths of a coverage baseline (what the ops catalog found before the registry; datasets present here)."""
    rows = json.loads(Path(path).read_text())['datasets']
    return [str(dr.absolute(r['path'])) for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('record')
    r.add_argument('--id', required=True)
    r.add_argument('--path', required=True)
    r.add_argument('--kind', required=True, choices=dr.KINDS)
    r.add_argument('--format', required=True)
    r.add_argument('--include', action='append')
    r.add_argument('--exclude', action='append', default=[])
    r.add_argument('--input', action='append', default=[])
    r.add_argument('--license')
    r.add_argument('--protected', action='store_true')
    r.add_argument('--open', action='store_true', help='the dataset keeps growing (later snapshots follow)')
    r.add_argument('--note')
    r.add_argument('--supersedes')
    r.add_argument('--rows', type=int)
    r.add_argument('--producer-script', help='the producing script, when this records for one (shell builds)')
    c = sub.add_parser('consume')
    c.add_argument('--run', required=True)
    c.add_argument('--how', required=True)
    c.add_argument('--amount')
    c.add_argument('items', nargs='+')
    for name in ('show', 'consumers', 'role'):
        sub.add_parser(name).add_argument('name')
    sub.add_parser('find').add_argument('path')
    sub.add_parser('protected')
    ls = sub.add_parser('list')
    ls.add_argument('--kind')
    ls.add_argument('--layer', choices=['git', 'journal', 'git+journal'])
    ck = sub.add_parser('check')
    ck.add_argument('paths', nargs='*')
    ck.add_argument('--baseline', action='append', default=[],
                    help='coverage file (natlang.data-registry-coverage/1): every listed path must be held by a dataset')
    pr = sub.add_parser('promote')
    pr.add_argument('ids', nargs='*')
    pr.add_argument('--all-closed', action='store_true')
    pr.add_argument('--dry-run', action='store_true')
    pr.add_argument('--no-check', action='store_true', help='skip check-main --quick in the registration commit')
    pr.add_argument('--role', help='give the promoted datasets this registry role, after the role\'s current members '
                    '(grpo-pool: a task-supply source enters training at the next pin)')
    pr.add_argument('--role-file', default=None,
                    help="the file of the dataset the role reads (default: the role's existing members' file, "
                         "else cases.jsonl; a role whose members are directories, e.g. grpo-sd-teacher, takes none)")
    a = ap.parse_args(argv)
    if a.cmd == 'record':
        rec = dr.record_dataset(a.id, os.path.abspath(a.path), kind=a.kind, format=a.format, include=a.include,
                                exclude=a.exclude, inputs=a.input, license=a.license, protected=a.protected,
                                closed=not a.open, note=a.note, supersedes=a.supersedes, rows=a.rows,
                                producer={'script': a.producer_script} if a.producer_script else None)
        _print({k: rec.get(k) for k in ('id', 'path', 'manifest_sha256', 'files', 'bytes', 'rows')})
        return 0
    if a.cmd == 'consume':
        _print(dr.consume(a.items, run=a.run, how=a.how, amount=a.amount))
        return 0
    reg = dr.load()
    if a.cmd == 'show':
        ds = reg.datasets.get(a.name)
        if ds is None:
            print(f'unknown dataset {a.name}', file=sys.stderr)
            return 1
        _print({**ds, 'snapshots': len(reg.snapshots.get(a.name, [])), 'files_listed': len(reg.files(a.name)),
                'consumers': reg.consumers(a.name)})
        return 0
    if a.cmd == 'find':
        ds = reg.find(Path(a.path))
        _print({'id': ds['id'], 'path': ds['path'], 'kind': ds.get('kind'), 'layer': ds.get('layer')} if ds else None)
        return 0 if ds else 1
    if a.cmd == 'consumers':
        _print(reg.consumers(a.name))
        return 0
    if a.cmd == 'role':
        _print([{'id': e['id'], 'path': str(p)} for e, p in reg.role(a.name)])
        return 0
    if a.cmd == 'protected':
        _print(reg.protected_ids())
        return 0
    if a.cmd == 'list':
        rows = [{'id': i, 'kind': d.get('kind'), 'layer': d.get('layer'), 'path': d.get('path'), 'rows': d.get('rows'),
                 'bytes': d.get('bytes')} for i, d in reg.datasets.items()
                if (not a.kind or d.get('kind') == a.kind) and (not a.layer or d.get('layer') == a.layer)]
        _print(rows)
        return 0
    if a.cmd == 'check':
        paths = list(a.paths)
        for baseline in a.baseline:
            paths += coverage_paths(baseline)
        paths = paths or default_check_paths(reg)
        missing = [p for p in paths if (reg.find(Path(p)) or {'admission': dr.INVENTORY}).get('admission') == dr.INVENTORY]
        _print({'checked': len(paths), 'unregistered': missing})
        return 1 if missing else 0
    if a.cmd == 'promote':
        return promote(reg, a)
    return 2


def promote(reg, a) -> int:
    ids = list(a.ids)
    if a.all_closed:
        ids += [i for i, d in reg.datasets.items() if d.get('layer') == 'journal' and d.get('closed')
                and d.get('manifest_sha256') and i not in ids]
    entries = []
    for i in ids:
        snaps = [s for s in reg.snapshots.get(i, []) if s.get('manifest_sha256')]
        if not snaps:
            print(f'{i}: no journal snapshot', file=sys.stderr)
            return 1
        entry = dr.git_entry(snaps[-1])
        if i in reg.git:       # keep the curated fields; add what the producer recorded
            entry = {**entry, **reg.git[i]}
        if a.role:
            order = max([((e.get('roles') or {}).get(a.role) or {}).get('order', 0) for e in reg.git.values()
                         if a.role in (e.get('roles') or {})] + [x['roles'][a.role]['order'] for x in entries
                                                                 if a.role in (x.get('roles') or {})] + [-1]) + 1
            entry['roles'] = {**(entry.get('roles') or {}), a.role: role_spec(reg, a.role, order, a.role_file)}
        entries.append(entry)
    if a.dry_run or not entries:
        _print(entries)
        return 0
    sys.path.insert(0, str(REPO / 'scripts/neuralese_data'))
    import register_corpora
    register_corpora.register(entries, f'data registry: promote {len(entries)} dataset(s)', replace=True,
                              check=not a.no_check)
    return 0


if __name__ == '__main__':
    sys.exit(main())
