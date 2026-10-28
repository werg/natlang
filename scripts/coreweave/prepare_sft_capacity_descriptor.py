#!/usr/bin/env python3
"""Bind the immutable seed training source and its freshly prepared packed-geometry sidecar.

This is metadata only: the actual learner validates parser/row custody and measures hardware capacity.
Output the printed path/SHA binding as the stage's memory_qualification recipe parameter.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))
from natlang_neuralese.train.native_resident import binding, bound
from natlang_neuralese.common.jsonio import write_json_durable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--geometry', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = binding(args.source)
    geometry = binding(args.geometry)
    value = json.loads(bound(geometry).read_text())
    if value.get('schema') != 'natlang.native-packed-geometry/1' or value.get('source') != source:
        raise ValueError('geometry is not bound to the requested exact seed source')
    descriptor = {'schema': 'natlang.native-sft-capacity-cases/1', 'source': source, 'geometry': geometry}
    if args.out.exists():
        if json.loads(args.out.read_text()) != descriptor:
            raise ValueError('existing immutable descriptor differs')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_durable(args.out, descriptor)
    print(json.dumps(binding(args.out), sort_keys=True))


if __name__ == '__main__':
    main()
