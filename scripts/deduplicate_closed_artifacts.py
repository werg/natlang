#!/usr/bin/env python3
"""Deduplicate a hash-pinned review of closed artifacts, preserving both paths.

Only use on finished, immutable runs. Hardlinked files become read-only: future
changes must create a new file rather than modifying a shared inode in place.
"""
import argparse
import json
import os
from pathlib import Path
import stat

from sync_training_corpora import digest, local_references


def identity(path):
    value = path.stat()
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if digest(args.plan)[0] != args.sha256:
        raise ValueError('Reviewed plan changed')
    plan = json.loads(args.plan.read_text())
    if plan.get('schema') != 'natlang.storage.dedup-proposal/1':
        raise ValueError('Unsupported review schema')
    base = Path(plan['base']).resolve(strict=True)
    pairs, selected = [], set()
    for entry in plan['exact_duplicate_payload_pairs']:
        keep, duplicate = (Path(entry[key]) for key in ('canonical_keep', 'deduplicate_path'))
        for path in (keep, duplicate):
            if path.is_symlink() or not path.is_file() or base not in path.resolve().parents:
                raise ValueError(f'Expected a regular file inside reviewed base: {path}')
            if path in selected:
                raise ValueError(f'Overlapping file pairs: {path}')
            selected.add(path)
        before = [identity(path) for path in (keep, duplicate)]
        if before[0][0] != before[1][0]:
            raise ValueError('Hardlinks require the same filesystem')
        for path in (keep, duplicate):
            if path.stat().st_size != entry['bytes'] or digest(path)[0] != entry['sha256']:
                raise ValueError(f'Reviewed bytes changed: {path}')
        if before != [identity(path) for path in (keep, duplicate)]:
            raise ValueError('Files changed during hash verification')
        pairs.append((entry, keep, duplicate, before))
    references = local_references(list(selected))
    if references['open_fds'] or references['live_job_references']:
        raise ValueError('Files are referenced by live work: ' + json.dumps(references))
    result = {'schema': 'natlang.closed-artifact-dedup/1', 'plan': str(args.plan),
              'plan_sha256': args.sha256, 'execute': args.execute,
              'policy': 'Finished immutable artifacts only; shared inodes read-only; future changes require a new file.',
              'local_references': references, 'pairs': [], 'reclaimed_allocated_bytes': 0}
    if not args.execute:
        print(json.dumps({'status': 'verified-preflight', 'pairs': len(pairs),
                          'bytes': sum(entry['bytes'] for entry, *_ in pairs)}))
        return
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    # Every completed pair is fsynced before moving to the next one. A stopped
    # operation leaves a readable journal and never an absent original path.
    with args.receipt.open('x') as journal:
        for entry, keep, duplicate, before in pairs:
            if before != [identity(path) for path in (keep, duplicate)]:
                raise ValueError('Files changed after review')
            if before[0][:2] == before[1][:2]:
                continue
            duplicate_stat = duplicate.stat()
            reclaimed = duplicate_stat.st_blocks * 512 if duplicate_stat.st_nlink == 1 else 0
            original_mode = stat.S_IMODE(keep.stat().st_mode)
            temporary = duplicate.with_name(f'.{duplicate.name}.dedup-{os.getpid()}')
            os.link(keep, temporary)
            try:
                # Removing write permission applies to both preserved paths.
                keep.chmod(original_mode & ~0o222)
                os.replace(temporary, duplicate)
                directory_fd = os.open(duplicate.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            finally:
                if temporary.exists():
                    temporary.unlink()
            if identity(keep)[:2] != identity(duplicate)[:2]:
                raise ValueError('Atomic replacement did not preserve the shared inode')
            record = {**entry, 'original_keep_mode': oct(original_mode),
                      'shared_inode': identity(keep)[:2], 'reclaimed_allocated_bytes': reclaimed}
            result['pairs'].append(record)
            result['reclaimed_allocated_bytes'] += reclaimed
            journal.write(json.dumps(record) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
    summary = args.receipt.with_suffix(args.receipt.suffix + '.summary.json')
    with summary.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'status': 'deduplicated', 'pairs': len(result['pairs']),
                      'reclaimed_allocated_bytes': result['reclaimed_allocated_bytes'],
                      'receipt': str(args.receipt)}))


if __name__ == '__main__':
    main()
