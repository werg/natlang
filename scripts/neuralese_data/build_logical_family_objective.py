#!/usr/bin/env python3
"""Logical-family CE objective for a built LFM3 joint-SFT mixture (2026-10-31).

The mixture policy declares target own-output loss shares per logical family (``objective.logical_family_targets``).
Source-cohort balance weighted rows by physical cohort files, which changed objective mass when identical rows were
rechunked (AGENTS.md, 2026-10-29). This replaces it with the exact logical-family objective:

  1. map:     every TRAIN row -> its family, from the cohort leaf it was written to (t1-communication-* ->
              native-communication, t5-* -> coding, t6-* -> external-agent, every other tier -> natlang), as a
              natlang.logical-family-map/1 bound to the manifest SHA and each row's exact binding;
  2. census:  the exact unscaled own-output loss mass of every row under the CE policy's objective (graph_weights),
              each leaf's SHA verified before reading;
  3. policy:  declared/1 scales = target * M / mass(family) (true cap max_scale: refuses a scale above it), written
              into a copy of the CE policy with family_balance {key: logical-family, map, scales}.

  build_logical_family_objective.py --train MANIFEST --ce-policy CE --targets MIXPOLICY --out DIR [--workers N]
  -> DIR/{families.json, census.json, ce-policy.json, receipt.json}
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
sys.path.insert(0, str(REPO))
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402

FAMILY_RULES = [(re.compile(r'^t\d+-communication-'), 'native-communication'), (re.compile(r'^t5-'), 'coding'),
                (re.compile(r'^t6-'), 'external-agent'), (re.compile(r'^t\d+-'), 'natlang')]


def family_of(leaf: Path) -> str:
    for pattern, family in FAMILY_RULES:
        if pattern.match(leaf.name):
            return family
    raise ValueError(f'no logical family for cohort leaf {leaf}')


def feature_family(row):
    """Classify actual coding-target port rows without renaming their shared interface family."""
    features = row.get('features') or {}
    if features.get('family') == 'port_record_calls' and features.get('domain') == 'trajectory_continuation_coding':
        return 'coding'
    if row.get('interface_family') == 'port_record_calls':
        family = row.get('logical_family')
        if family not in {'coding', 'natlang'}:
            raise ValueError('port target feature lacks an exact logical family')
        return family
    return None


def feature_overrides(manifest_path):
    """Bind recorded row features to coding objective mass, never a new semantic task family."""
    from natlang_neuralese.common.jsonio import iter_jsonl_rows
    manifest = json.loads(manifest_path.read_text())
    refs = list(manifest.get('logical_family_features') or [])
    default = manifest_path.parent / 'sample-features.jsonl.zst'
    if default.exists() and not refs:
        refs = [{'path': str(default), 'sha256': sha256_file_hex(default)}]
    result, bindings = {}, []
    for ref in refs:
        path = Path(ref['path']); path = path if path.is_absolute() else manifest_path.parent / path
        if sha256_file_hex(path) != ref['sha256']:
            raise ValueError('logical family feature input changed')
        bindings.append(bind(path))
        for _, row in iter_jsonl_rows(path):
            family = feature_family(row)
            if family is not None:
                if row['id'] in result:
                    raise ValueError('logical family features repeat a classified row')
                result[row['id']] = family
    return result, bindings


def leaf_paths(source) -> list[Path]:
    """Each TRAIN row's graph-file leaf (as native_cohort_schedule.source_cohort_ids walks the validated tree)."""
    def walk(node):
        if node.kind != 'manifest':
            return [Path(node.path)] * len(node)
        out = []
        for child in node.children:
            out.extend(walk(child))
        return out
    paths = walk(source)
    if len(paths) != len(source.entries):
        raise ValueError('leaf paths do not cover source rows')
    return paths


def capped_scales(targets: dict, mass: dict, cap: float) -> tuple[dict, list]:
    """Declared scales giving each family its target share of the total own-output mass M, with ``cap`` a true cap: a
    family whose target needs a scale above the cap gets the cap (and less than its target share), and the remaining
    share goes to the other families in proportion to their targets; total scaled mass stays M (mean one)."""
    total = sum(mass.values())
    capped = {}
    while True:
        free = {f: t for f, t in targets.items() if f not in capped}
        share_left = 1 - sum(capped[f] * mass[f] / total for f in capped)
        norm = sum(free.values())
        scales = {f: free[f] / norm * share_left * total / mass[f] for f in free}
        over = [f for f in scales if scales[f] > cap]
        if not over:
            return {**scales, **capped}, sorted(capped)
        for f in over:
            capped[f] = cap


def bind(path: Path) -> dict:
    return {'path': str(path), 'sha256': sha256_file_hex(path)}


_OBJECTIVE = None


def _init(policy_path):
    global _OBJECTIVE
    from natlang_neuralese.train.own_output_supervision import policy_objective
    _OBJECTIVE = policy_objective(json.loads(Path(policy_path).read_text()))


def _leaf_mass(item):
    import zstandard
    from natlang_neuralese.train.own_output_supervision import graph_weights
    index, source = item
    if sha256_file_hex(Path(source['path'])) != source['sha256']:
        raise ValueError(f"leaf changed: {source['path']}")
    out = []
    with open(source['path'], 'rb') as fh:
        raw = zstandard.ZstdDecompressor().stream_reader(fh) if source['path'].endswith('.zst') else fh
        for line in io.TextIOWrapper(raw, encoding='utf8'):
            if line.strip():
                row = json.loads(line)
                weights, _ = graph_weights(row, _OBJECTIVE)
                out.append((row['id'], float(sum(weights))))
    return index, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--train', required=True, help='mixture train-sources.json')
    ap.add_argument('--ce-policy', required=True, help='CE policy built for this manifest (its objective is used '
                    'without family scales for the census)')
    ap.add_argument('--targets', required=True, help='mixture policy declaring objective.logical_family_targets')
    ap.add_argument('--fingerprint', required=True)
    ap.add_argument('--role-ids', type=int, nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    args = ap.parse_args()
    from natlang_neuralese.data.native_sources import NativeGraphSource
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(args.train).resolve()
    targets = json.loads(Path(args.targets).read_text())['objective']['logical_family_targets']
    cap = json.loads(Path(args.targets).read_text())['objective']['family_balance']['max_scale']
    if abs(sum(targets.values()) - 1) > 1e-9:
        raise ValueError(f'logical family targets do not sum to 1: {targets}')
    train = NativeGraphSource(manifest_path, args.fingerprint, required_split='train', role_ids=args.role_ids)
    overrides, feature_bindings = feature_overrides(manifest_path)
    train_ids = {entry['id'] for entry in train.entries}
    if set(overrides) - train_ids:
        raise ValueError('logical family features name rows absent from the exact train source')
    families = [overrides.get(entry['id'], family_of(path))
                for entry, path in zip(train.entries, leaf_paths(train))]
    rows = [{'id': e['id'], 'binding': e['binding'], 'family': f} for e, f in zip(train.entries, families)]
    (out / 'families.json').write_bytes(json.dumps({'schema': 'natlang.logical-family-map/1',
                                                    'source_sha256': train.file_sha256, 'rows': rows},
                                                   separators=(',', ':')).encode())
    # Census: the CE objective without family scales (the scales are what this computes).
    ce = json.loads(Path(args.ce_policy).read_text())
    census_policy = json.loads(json.dumps(ce))
    census_policy['objective'].pop('family_balance', None)
    (out / 'census-policy.json').write_text(json.dumps(census_policy, indent=1, sort_keys=True) + '\n')
    # Leaf paths are relative to the manifest's directory unless absolute.
    leaves = [{**s, 'path': str(manifest_path.parent / s['path'])}
              for s in json.loads(manifest_path.read_text())['sources']]
    if any(s.get('kind') != 'native-graphs' for s in leaves):
        raise ValueError('census reads flat native-graph manifests only')
    results = {}
    with Pool(args.workers, initializer=_init, initargs=(str(out / 'census-policy.json'),)) as pool:
        for index, masses in pool.imap_unordered(_leaf_mass, list(enumerate(leaves))):
            results[index] = masses
    masses = [m for i in range(len(leaves)) for m in results[i]]
    if [m[0] for m in masses] != [r['id'] for r in rows]:
        raise ValueError('census row order differs from the family map')
    mass, count, zero = collections.Counter(), collections.Counter(), collections.Counter()
    for (_, m), f in zip(masses, families):
        mass[f] += m
        count[f] += 1
        zero[f] += m == 0
    total = sum(mass.values())
    missing = sorted(set(targets) - set(mass))
    extra = sorted(set(mass) - set(targets))
    if extra:
        raise ValueError(f'families without targets: {extra}')
    # A declared family absent from this manifest gets no mass; the remaining targets are renormalized.
    present = {f: t for f, t in targets.items() if mass.get(f)}
    scales, capped = capped_scales(present, {f: mass[f] for f in present}, cap)
    census = {'schema': 'natlang.own-output-mass-census/1', 'manifest': bind(manifest_path), 'rows': len(rows),
              'policy': bind(out / 'census-policy.json'), 'objective_without_family_scales': True,
              'families': {f: {'rows': count[f], 'zero_mass_rows': zero[f], 'unscaled_mass': mass[f],
                               'unscaled_share': mass[f] / total} for f in sorted(mass)},
              'unscaled_mass_total': total}
    (out / 'census.json').write_text(json.dumps(census, indent=1) + '\n')
    policy = json.loads(json.dumps(ce))
    policy['objective']['family_balance'] = {'key': 'logical-family', 'max_scale': cap, 'normalization': 'declared/1',
                                             'map': bind(out / 'families.json'),
                                             'scales': {f: round(v, 6) for f, v in sorted(scales.items())}}
    (out / 'ce-policy.json').write_text(json.dumps(policy, indent=1, sort_keys=True) + '\n')
    receipt = {'schema': 'natlang.logical-family-objective/1', 'train_sources': bind(manifest_path),
               'base_ce_policy': bind(Path(args.ce_policy)), 'targets': bind(Path(args.targets)),
               'target_shares': targets, 'absent_families': missing, 'capped_families': capped, 'max_scale': cap, 'map': bind(out / 'families.json'),
               'census': bind(out / 'census.json'), 'ce_policy': bind(out / 'ce-policy.json'), 'scales': scales,
               'effective_shares': {f: scales[f] * mass[f] / total for f in scales},
               'rule': [[p.pattern, f] for p, f in FAMILY_RULES],
               'feature_inputs': feature_bindings, 'feature_classified_rows': len(overrides)}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({k: receipt[k] for k in ('scales', 'effective_shares', 'absent_families', 'capped_families')},
                     indent=1))


if __name__ == '__main__':
    main()
