#!/usr/bin/env python3
"""List the programs an admitted authored family already has, so an extension batch adds genuinely new ones.

  authored_family_report.py FAMILY [--pool admitted/cases.jsonl]    (FAMILY with or without the `auth_` prefix)

One line per distinct functional program: tasks, worlds, domain, shape, root name, helper names and the first words of
the root instruction. A new program differs from every line (new operation, helper roles, data flow or code steps).
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_repetition import program_features, split_file  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training' / 'neuralese'))
from natlang_neuralese.common.paths import resolve  # noqa: E402

POOL = str(resolve('data_hdd', 'natlang-rl-authoring', 'admitted', 'cases.jsonl'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('family')
    parser.add_argument('--pool', default=POOL)
    args = parser.parse_args(argv)
    family = args.family if args.family.startswith('auth_') else 'auth_' + args.family
    programs = collections.OrderedDict()
    for line in Path(args.pool).read_text().splitlines():
        row = json.loads(line)
        if row['family'] != family:
            continue
        key = program_features(row)['functional']
        entry = programs.setdefault(key, {'tasks': 0, 'worlds': set(), 'row': row})
        entry['tasks'] += 1
        entry['worlds'].add(row['curriculum']['split_group'])
    domains = sorted({e['row']['curriculum']['external'].get('subject_domain') for e in programs.values()})
    print(f'{family}: {len(programs)} functional programs; domains used: {", ".join(map(str, domains))}')
    for entry in programs.values():
        row = entry['row']
        files = row['semantics']['files']
        root = row['semantics']['root']
        helpers = sorted(path.rsplit('/', 1)[-1][:-3] for path in files if path.endswith('.nl') and path != root)
        body = re.sub(r'\s+', ' ', split_file(files[root])[1]).strip()[:110]
        print(f'- {entry["tasks"]}t/{len(entry["worlds"])}w {row["curriculum"]["external"].get("subject_domain")} | '
              f'{row["curriculum"]["shape"]} | {root} | helpers: {", ".join(helpers) or "-"} | {body}')


if __name__ == '__main__':
    sys.exit(main())
