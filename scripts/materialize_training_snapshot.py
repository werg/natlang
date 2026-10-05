#!/usr/bin/env python3
"""Freeze a manifest-selected view of an archived/live run as independent bytes.

Replacing a storage alias never alters the source run. No hardlinks to mutable
source files. Use for a newly published run selection before future collection.
"""
import argparse
import json
from pathlib import Path
import shutil
import tempfile
from sync_training_corpora import digest, relative, verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--id', required=True)
    parser.add_argument('--storage', type=Path, help='Optional immutable storage directory behind canonical alias')
    args = parser.parse_args()
    manifest = json.loads((args.repo / 'training/corpus-manifests' / (relative(args.id) + '.json')).read_text())
    canonical = args.repo / relative(manifest['path'])
    if not canonical.is_symlink():
        verify(args.repo, manifest)
        return
    target = args.storage or canonical
    if args.storage and target.exists():
        raise ValueError('storage already exists; verify it rather than replace it')
    target.parent.mkdir(parents=True, exist_ok=True)
    needed = sum(item['bytes'] for item in manifest['files'])
    if shutil.disk_usage(target.parent).free < needed + 8 * 2**30:
        raise ValueError('insufficient snapshot storage')
    temporary = Path(tempfile.mkdtemp(prefix='snapshot-', dir=target.parent))
    for item in manifest['files']:
        source = canonical / relative(item['path'])
        destination = temporary / relative(item['path'])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        sha, _ = digest(destination)
        if destination.stat().st_size != item['bytes'] or sha != item['sha256']:
            raise ValueError(f'changed source or bad copy; retained diagnostic directory {temporary}')
    # All bytes verified before changing the projection. Existing source is kept.
    if args.storage:
        temporary.rename(target)
        replacement = canonical.with_name(canonical.name + '.new-link')
        replacement.symlink_to(target.resolve(), target_is_directory=True)
        replacement.replace(canonical)
    else:
        canonical.unlink()
        temporary.rename(canonical)
    verify(args.repo, manifest)


if __name__ == '__main__':
    main()
