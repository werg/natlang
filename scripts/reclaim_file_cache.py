#!/usr/bin/env python3
"""Advise Linux to release unused cache for explicitly selected immutable datasets/weights.

This does not delete files, change tensor weights, or flush the whole host's cache.
CUDA on a unified-memory machine may see little allocatable free memory even when
Linux reports substantial MemAvailable. Record both independently around this tool.
"""
import argparse
import json
import os
import time
import urllib.request
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
    parser.add_argument('--min-file-age-seconds', type=float, default=300,
                        help='Skip recently changed files that may still be active outputs or dirty cache')
    parser.add_argument('--min-free-gib', type=float, default=0,
                        help='Skip advisory reclamation when Linux MemFree already exceeds this level')
    parser.add_argument('--ready-url',
                        help='Skip these roots until this model readiness endpoint returns HTTP 200')
    args = parser.parse_args()
    if args.min_file_age_seconds < 0 or args.min_free_gib < 0:
        parser.error('Age and free-memory thresholds must be nonnegative')
    roots = [str(Path(p).resolve(strict=True)) for p in args.root]
    for root in roots:
        if Path(root) in (Path('/'), Path('/home'), Path('/mnt'), Path('/mnt/external')):
            parser.error('Select specific dataset/weight directories, not an entire host or volume')
    if args.receipt_dir:
        directory = Path(args.receipt_dir)
        directory.mkdir(parents=True, exist_ok=True)
        args.receipt = directory / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + '.json')
    # Reserve the receipt before taking action; never overwrite a previous diagnosis.
    with open(args.receipt, 'x') as receipt:
        report = {'schema': 'natlang.file-cache-reclaim/1', 'started': datetime.now(timezone.utc).isoformat(),
                  'apply': args.apply, 'before': memory(), 'files': 0, 'file_bytes': 0, 'errors': [],
                  'roots': roots,
                  'skipped_recent': 0, 'min_file_age_seconds': args.min_file_age_seconds}
        seen = set()
        needed = report['before']['MemFree'] < args.min_free_gib * 1024**3 if args.min_free_gib else True
        final_status = 'advised' if needed and args.apply else 'not_needed' if not needed else 'dry_run'
        report['status'] = 'in_progress'
        # Preserve an honest receipt even if an advisory call blocks or is interrupted.
        def save_report():
            receipt.seek(0)
            json.dump(report, receipt, indent=2)
            receipt.write('\n')
            receipt.truncate()
            receipt.flush()
            os.fsync(receipt.fileno())
        save_report()
        if needed and args.ready_url:
            report['ready_url'] = args.ready_url
            try:
                with urllib.request.urlopen(args.ready_url, timeout=5) as response:
                    ready = response.status == 200
            except (OSError, ValueError) as error:
                ready = False
                report['readiness_error'] = str(error)
            if not ready:
                needed = False
                final_status = 'model_not_ready'
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
                        if time.time() - stat.st_mtime < args.min_file_age_seconds:
                            report['skipped_recent'] += 1
                            continue
                        identity = stat.st_dev, stat.st_ino
                        if identity in seen:
                            continue
                        seen.add(identity)
                        if args.apply:
                            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
                        report['files'] += 1
                        report['file_bytes'] += stat.st_size
                        if report['files'] % 1000 == 0:
                            save_report()
                except OSError as error:
                    report['errors'].append({'path': str(path), 'error': str(error)})
        report['after'] = memory()
        report['status'] = final_status
        report['finished'] = datetime.now(timezone.utc).isoformat()
        save_report()
    print(json.dumps(report))


if __name__ == '__main__':
    main()
