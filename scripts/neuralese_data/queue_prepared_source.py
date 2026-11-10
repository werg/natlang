#!/usr/bin/env python3
"""Queue rows of a prepared, verified native-graph source as one ``prepared-source`` admission package.

The designed way to add prepared static data to the interleaved controller's next mandatory SFT window
(``rl/memetic_admission.py``, family ``prepared-source``): one immutable package per source in the controller's
admission queue (``<controller out>/memetic/distill/admission-queue.jsonl``). The window boundary rechecks the
current protection union per row, admits rows under the shared family share cap after the lineage's own rows, and
defers (never drops) the rest. This tool only prepares the package; it changes no recipe, config or controller state.

Rows kept: train split, LFM3 tokenizer/role geometry of the live SFT stage, listed in ``--ids`` when given, not already
in the current joint SFT source (``--exclude-ids-from`` a logical-family map), and not related to any protected world.
The package records the source file hash, the verification receipt it relies on, the selection description (and the
hash of the ids list) and the per-reason counts; ``--dry-run`` prints them without writing.

  queue_prepared_source.py --controller-out OUT --source FILE --origin TEXT --verification RECEIPT --selection TEXT \\
      --geometry packed-geometry.json [--ids FILE] [--exclude-ids-from families.json] [--limit N] [--dry-run]
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from natlang_neuralese.common.hashing import sha256_file_hex_cached  # noqa: E402
from natlang_neuralese.common.jsonio import open_jsonl_text  # noqa: E402


def read_ids(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def family_map_ids(path):
    return {row['id'] for row in json.loads(Path(path).read_text())['rows']}


def select(path, *, geometry, ids=None, exclude=frozenset(), protected=None, limit=None):
    """(rows in file order, {reason: count}) of the source rows that can be queued."""
    wanted = None if ids is None else set(ids)
    rows, skipped, seen = [], collections.Counter(), set()
    with open_jsonl_text(path) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if wanted is not None and row['id'] not in wanted:
                continue
            if row.get('split') != 'train':
                skipped['not a train row'] += 1
            elif row.get('tokenizer_fingerprint') != geometry['tokenizer']:
                skipped['tokenizer differs from the SFT stage'] += 1
            elif row['id'] in seen:
                skipped['duplicate id in the source'] += 1
            elif row['id'] in exclude:
                skipped['already in the joint SFT source'] += 1
            elif protected is not None and protected.related(row):
                skipped['related to a protected world'] += 1
            else:
                seen.add(row['id'])
                rows.append(row)
                if limit is not None and len(rows) >= limit:
                    break
    return rows, dict(skipped)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--controller-out', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--origin', required=True)
    parser.add_argument('--verification', required=True, type=Path,
                        help='the receipt that verifies the source (bound by path and sha256)')
    parser.add_argument('--selection', required=True, help='how the rows were chosen, in words')
    parser.add_argument('--geometry', required=True, type=Path, help="the live SFT stage's packed-geometry.json")
    parser.add_argument('--ids', type=Path, help='one row id per line: keep only these rows')
    parser.add_argument('--exclude-ids-from', type=Path, help='logical-family map of the current joint SFT source')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--source-name', default='prepared-source-graphs.jsonl')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)

    from natlang_neuralese.common.custody import ProtectedWorlds
    from natlang_neuralese.rl.memetic_admission import STATIC_FAMILY, queue_package
    geometry = json.loads(args.geometry.read_text())['identity']
    protected = ProtectedWorlds.load()
    rows, skipped = select(
        args.source, geometry=geometry, ids=None if args.ids is None else read_ids(args.ids),
        exclude=family_map_ids(args.exclude_ids_from) if args.exclude_ids_from else frozenset(),
        protected=protected, limit=args.limit)
    summary = {'source': str(args.source.resolve()), 'rows': len(rows), 'skipped': skipped}
    if not rows or args.dry_run:
        print(json.dumps({**summary, 'queued': False}))
        return 0 if rows or args.dry_run else 1
    provenance = {
        'origin': args.origin, 'selection': args.selection,
        'verification': {'path': str(args.verification.resolve()),
                         'sha256': sha256_file_hex_cached(args.verification)},
        'source': {'path': str(args.source.resolve()), 'sha256': sha256_file_hex_cached(args.source)},
        'ids_sha256': None if args.ids is None else sha256_file_hex_cached(args.ids),
        'skipped': skipped, 'protection_digest': protected.digest}
    receipt = queue_package(
        args.controller_out / 'memetic', rows, family=STATIC_FAMILY, tokenizer=geometry['tokenizer'],
        roles=geometry['roles'], protected=protected, manifests=[args.source.resolve()], actors=[],
        scope=f'prepared static source: {args.origin}; admission is a recorded source append at the next mandatory SFT '
              'window, after the lineage\'s own rows, under the shared family share cap',
        source_name=args.source_name, static_source=provenance)
    print(json.dumps({**summary, 'queued': True, 'package': receipt['source']['path']}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
