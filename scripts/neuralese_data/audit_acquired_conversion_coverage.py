#!/usr/bin/env python3
"""Inventory committed acquired-session conversions while other exports continue.

This binds receipt bytes and checks declared output presence/size. It does not
admit training data or replace output hashing, protection screening or graph audit.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from scripts.neuralese_data.convert_acquired_sessions_student import output_dir
from natlang_neuralese.common.jsonio import write_json_durable


def inventory(exports_path: Path, root: Path) -> dict:
    raw = exports_path.read_bytes()
    exports = json.loads(raw)['exports']
    if len(exports) != len(set(exports)):
        raise ValueError('duplicate exports in acquisition manifest')
    rows, errors, pending = [], [], 0
    buckets: dict[str, Counter] = {}
    for export in exports:
        directory = output_dir(root, export)
        source = directory.relative_to(root).parts[0]
        bucket = buckets.setdefault(source, Counter())
        path = directory / 'receipt.json'
        if not path.exists():
            pending += 1
            bucket['pending'] += 1
            continue
        try:
            before = path.stat()
            payload = path.read_bytes()
            after = path.stat()
            identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if identity(before) != identity(after):
                raise ValueError('receipt changed during inventory')
            receipt = json.loads(payload)
            if receipt.get('training_admission') is not False:
                raise ValueError('conversion receipt claims training admission')
            if Path(receipt['export_receipt']['path']).resolve() != Path(export).resolve():
                raise ValueError('receipt belongs to a different acquired export')
            outputs = receipt['outputs']
            for output in outputs:
                destination = Path(output['path'])
                if not destination.is_relative_to(directory) or destination.stat().st_size != output['bytes']:
                    raise ValueError('declared output is missing, relocated or has different size')
                if type(output['rows']) is not int or output['rows'] < 0:
                    raise ValueError('invalid output row count')
            counts = receipt.get('counts') or {}
            for split in ('train', 'test'):
                actual = sum(o['rows'] for o in outputs if Path(o['path']).parent.name == split)
                if actual != counts.get('graphs:' + split, 0):
                    raise ValueError('declared graph count differs from output rows')
            bucket['receipt_present'] += 1
            bucket['complete_receipts'] += 1
            bucket['source_sessions'] += receipt['source_record_count']
            for key, count in counts.items():
                bucket['reason:' + key] += count
            rows.append({'receipt': str(path), 'receipt_sha256': hashlib.sha256(payload).hexdigest(),
                         'export': export, 'export_sha256': receipt['export_receipt']['sha256'],
                         'source_record_count': receipt['source_record_count'],
                         'builder_sha256': receipt['builder_sha256'], 'counts': counts,
                         'outputs': outputs})
        except (OSError, KeyError, TypeError, ValueError) as error:
            errors.append({'export': export, 'receipt': str(path), 'error': str(error)})
    return {'schema': 'natlang.acquired-session-conversion-coverage-audit/1',
            'verification_scope': 'Receipt bytes and declared output presence/size; not output hash verification or training admission. Point-in-time inventory of completed exports.',
            'exports_manifest': str(exports_path), 'exports_manifest_sha256': hashlib.sha256(raw).hexdigest(),
            'current_output_root': str(root), 'expected_exports': len(exports),
            'complete_receipts': len(rows), 'pending': pending, 'errors': errors,
            'per_source': {k: dict(v) for k, v in buckets.items()}, 'receipts': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exports', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('refusing to overwrite an earlier inventory')
    result = inventory(args.exports.resolve(strict=True), args.root.resolve(strict=True))
    write_json_durable(args.out, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'receipts'}), flush=True)
    if result['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
