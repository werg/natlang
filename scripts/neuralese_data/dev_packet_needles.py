#!/usr/bin/env python3
"""Exact content-window needles for a task packet (separation scan against rendered training texts).

Every string leaf of each case's `semantics` (program files, services, inputs, folder files, expectations) of at least
WINDOW characters is JSON-escaped (one level, and two levels for text nested inside tool output) and cut into
non-overlapping WINDOW-character windows. Any substring of >= 2*WINDOW-1 escaped characters shared with a rendered text
contains one whole window, so a literal fixed-string scan of the texts finds it. Windows are written one per line
(escaped text has no raw newline) to NEEDLES; MAP records which cases (and how many distinct cases) own each window.

  dev_packet_needles.py CASES.jsonl --needles NEEDLES.txt --map MAP.json
  dev_packet_needles.py --classify MAP.json HITS.txt     # HITS.txt: matched windows, one per line
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

WINDOW = 64


def leaves(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from leaves(v)
    elif isinstance(value, list):
        for v in value:
            yield from leaves(v)


def esc(text: str) -> str:
    return json.dumps(text, ensure_ascii=False)[1:-1]


def windows(leaf: str):
    for text in (esc(leaf), esc(esc(leaf))):
        for i in range(0, len(text) - WINDOW + 1, WINDOW):
            yield text[i:i + WINDOW]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cases', nargs='?', type=Path)
    ap.add_argument('--needles', type=Path)
    ap.add_argument('--map', type=Path)
    ap.add_argument('--classify', nargs=2, metavar=('MAP', 'HITS'), type=Path)
    a = ap.parse_args()
    if a.classify:
        owners = json.loads(a.classify[0].read_text())
        hits = {l.rstrip('\n') for l in a.classify[1].read_text().splitlines() if l}
        unique, shared = {}, {}
        for w in hits:
            o = owners.get(w)
            if o is None:
                continue
            (unique if len(o['cases']) == 1 else shared).setdefault(o['family'], set()).update(o['cases'] if len(o['cases']) == 1 else [])
        print(json.dumps({'hit_windows': len(hits), 'case_unique_hit_windows': sum(1 for w in hits if w in owners and len(owners[w]['cases']) == 1),
                          'shared_hit_windows': sum(1 for w in hits if w in owners and len(owners[w]['cases']) > 1),
                          'cases_with_unique_hits': {k: sorted(v) for k, v in unique.items()},
                          'shared_hits_by_family': {f: sum(1 for w in hits if w in owners and len(owners[w]['cases']) > 1 and owners[w]['family'] == f)
                                                    for f in {o['family'] for o in owners.values()}}}, indent=1))
        return 0
    owners: dict[str, dict] = {}
    for line in a.cases.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        for leaf in leaves(r['semantics']):
            if len(leaf) >= WINDOW:
                for w in windows(leaf):
                    o = owners.setdefault(w, {'family': r['curriculum']['family'], 'cases': []})
                    if r['id'] not in o['cases']:
                        o['cases'].append(r['id'])
    a.needles.write_text(''.join(w + '\n' for w in owners))
    a.map.write_text(json.dumps(owners))
    print(json.dumps({'windows': len(owners), 'case_unique': sum(1 for o in owners.values() if len(o['cases']) == 1)}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
