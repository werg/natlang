#!/usr/bin/env python3
"""Run a bound CPU preparer against the learner's unchanged package and caches.

The generated request uses the existing resident transaction protocol. This tool
never imports a newer model/trainer into the learner or changes a running owner.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--preparer', type=Path, required=True,
                        help='JSON binding of the independently reviewed native_resident.py CPU helper')
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--stage-id', required=True)
    parser.add_argument('--tokenizer-fingerprint', required=True)
    parser.add_argument('--role-ids', type=int, nargs=4, required=True)
    parser.add_argument('--geometry-evidence', type=Path,
                        help='JSON map of train/test to exact completed geometry-report bindings')
    args = parser.parse_args(argv)
    package = args.package.resolve(strict=True)
    declared = json.loads(args.preparer.read_text())
    source = Path(declared['path']).resolve(strict=True)
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != declared['sha256']:
        raise ValueError('external resident preparer changed')
    if 'natlang_neuralese' in sys.modules:
        raise ValueError('CPU preparer must start with the declared package, not a previously imported runtime')
    sys.path.insert(0, str(package.parent))
    helper = runpy.run_path(str(source), run_name='natlang_neuralese.train.native_resident_external')
    request = helper['prepare'](args.directory, args.recipe, args.stage_id,
        tokenizer_fingerprint=args.tokenizer_fingerprint, role_ids=args.role_ids,
        runtime_package=package,
        geometry_evidence=json.loads(args.geometry_evidence.read_text()) if args.geometry_evidence else None)
    print(json.dumps({'request': request, 'preparer': declared,
                      'runtime_code_sha256': helper['digest'](helper['code_inventory'](package))}))


if __name__ == '__main__':
    main()
