#!/usr/bin/env python3
"""Advise Linux to release unused cache for explicitly selected immutable datasets/weights.

This does not delete files, change tensor weights, or flush the whole host's cache.
CUDA on a unified-memory machine may see little allocatable free memory even when
Linux reports substantial MemAvailable. Record both independently around this tool.
"""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def memory():
    return {key: int(value.split()[0]) * 1024
            for line in Path('/proc/meminfo').read_text().splitlines()
            for key, value in [line.split(':', 1)]
            if key in ('MemFree', 'MemAvailable', 'Cached', 'Buffers', 'Dirty', 'Writeback')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', required=True)
    receipts = parser.add_mutually_exclusive_group(required=True)
    receipts.add_argument('--receipt')
    receipts.add_argument('--receipt-dir')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--min-free-gib', type=float, default=0,
                        help='Skip advisory reclamation when Linux MemFree already exceeds this level')
    args = parser.parse_args()
    if args.receipt_dir:
        directory = Path(args.receipt_dir)
        directory.mkdir(parents=True, exist_ok=True)
        args.receipt = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    # Reserve the receipt before taking action; never overwrite a previous diagnosis.
    with open(args.receipt, 'x') as receipt:
        report = {'schema': 'natlang.file-cache-reclaim/1', 'started': datetime.now(timezone.utc).isoformat(),
                  'apply': args.apply, 'before': memory(), 'files': 0, 'file_bytes': 0, 'errors': [],
                  'roots': [str(Path(p).resolve(strict=True)) for p in args.root]}
        seen = set()
        needed = report['before']['MemFree'] < args.min_free_gib * 1024**3 if args.min_free_gib else True
        report['status'] = 'advised' if needed and args.apply else 'not_needed' if not needed else 'dry_run'
        for raw in report['roots'] if needed else []:
            root = Path(raw)
            if root in (Path('/'), Path('/home'), Path('/mnt'), Path('/mnt/external')):
                raise ValueError('Select specific dataset/weight directories, not an entire host or volume')
            paths = (p for p in root.rglob('*') if p.is_file()) if root.is_dir() else [root]
            for path in paths:
                if path.suffix not in ('.parquet', '.safetensors', '.gguf', '.bin', '.pt', '.jsonl') and root.is_dir():
                    continue
                try:
                    with path.open('rb') as stream:
                        stat = os.fstat(stream.fileno())
                        identity = stat.st_dev, stat.st_ino
                        if identity in seen:
                            continue
                        seen.add(identity)
                        if args.apply:
                            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        report['files'] += 1
                        report['file_bytes'] += stat.st_size
                except OSError as error:
                    report['errors'].append({'path': str(path), 'error': str(error)})
        report['after'] = memory()
        report['finished'] = datetime.now(timezone.utc).isoformat()
        json.dump(report, receipt, indent=2)
        receipt.write('\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
