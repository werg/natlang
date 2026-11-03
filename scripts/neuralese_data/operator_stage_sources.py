#!/usr/bin/env python3
"""Rendered learned-operator graphs -> `natlang.native-graph-sources/1` train/held lists for the operator component.

    operator_stage_sources.py [--graphs-root DIR] --out DIR

Reads every `<operator>/<source>/receipt.json` under the graphs root (build_operator_graphs.py), and writes
`operator-train-sources.json` (graphs-train files) and `operator-held-sources.json` (graphs-test files), each source
bound by path and sha256, plus `operator-stage.json`: per-operator train/held row counts against the operator's
dedicated data target. The joint SFT continuation appends these sources through its recorded source transition and
sets the stage parameter `learned_operators: all` (plans/neuralese/LEARNED_COMPACTION_OPERATORS.md). Text-cut files
are ablation baselines and are never training sources.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training' / 'neuralese'), str(ROOT)]
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402
from natlang_neuralese.model.learned_operators import OPERATORS, registry_sha256  # noqa: E402

GRAPHS_ROOT = data_path('data_hdd', 'natlang-compaction-operators-20261103', 'graphs')


def collect(graphs_root: Path):
    train, held, counts = [], [], {name: {'train': 0, 'test': 0, 'sources': []} for name in OPERATORS}
    for receipt_path in sorted(graphs_root.glob('*/*/receipt.json')):
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('limit') is not None:
            continue  # smoke renders are never sources
        operator = receipt['operator']
        for split, target in (('train', train), ('test', held)):
            path = receipt_path.parent / f'graphs-{split}.jsonl.zst'
            output = receipt['outputs'][f'graphs-{split}']
            if not output['rows']:
                continue
            if sha256_file_hex(path) != output['sha256']:
                raise SystemExit(f'{path} differs from its build receipt')
            target.append({'kind': 'native-graphs', 'path': str(path), 'sha256': output['sha256'],
                           'training_priority': 1, 'learned_operator': operator,
                           'operator_source': receipt['source'], 'rows': output['rows']})
            counts[operator][split] += output['rows']
        counts[operator]['sources'].append(receipt['source'])
    return train, held, counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--graphs-root', type=Path, default=GRAPHS_ROOT)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    train, held, counts = collect(args.graphs_root)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, sources in (('operator-train-sources.json', train), ('operator-held-sources.json', held)):
        (args.out / name).write_text(json.dumps({'schema': 'natlang.native-graph-sources/1', 'sources': sources},
                                                indent=1) + '\n')
    stage = {'schema': 'natlang.learned-operator-stage-sources/1', 'registry_sha256': registry_sha256(),
             'stage_parameters': {'learned_operators': 'all'},
             'operators': {name: {**value, 'data_target': OPERATORS[name].data_target,
                                  'target_met': value['train'] >= OPERATORS[name].data_target}
                           for name, value in counts.items()},
             'train_sources': {'path': str(args.out / 'operator-train-sources.json'),
                               'sha256': sha256_file_hex(args.out / 'operator-train-sources.json')},
             'held_sources': {'path': str(args.out / 'operator-held-sources.json'),
                              'sha256': sha256_file_hex(args.out / 'operator-held-sources.json')}}
    (args.out / 'operator-stage.json').write_text(json.dumps(stage, indent=1) + '\n')
    print(json.dumps(stage['operators'], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
