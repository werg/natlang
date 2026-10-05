#!/usr/bin/env python3
"""Retire the former DGX mirror beneath the canonical checkout, retaining aliases.

This is a one-time archival move, not a synchronization mechanism. No artifacts,
tracked modifications, or historical source commits are deleted.
"""
import argparse
import datetime
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path('/home/werg/natlang'))
    parser.add_argument('--old-root', type=Path, default=Path('/home/werg/natlang-remote'))
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    archive = args.repo / 'runs/dgx-legacy-imports'
    entries = []
    for path in sorted(args.old_root.iterdir()):
        destination = archive / path.name
        if path.is_symlink() and destination.exists() and path.resolve() == destination.resolve():
            continue
        if destination.exists():
            raise ValueError(f'archive collision: {destination}')
        item = {'old': str(path), 'archive': str(destination), 'worktree': (path / '.git').is_file() if path.is_dir() else False}
        if item['worktree']:
            item['commit'] = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
            item['status'] = subprocess.check_output(['git', '-C', str(path), 'status', '--porcelain'], text=True)
            item['unmerged_commits'] = int(subprocess.check_output(['git', '-C', str(path), 'rev-list', '--count', 'HEAD', '--not', 'origin/main'], text=True))
        entries.append(item)
    print(json.dumps(entries, indent=2))
    if not args.apply:
        return
    archive.mkdir(parents=True, exist_ok=True)
    for item in entries:
        source, target = Path(item['old']), Path(item['archive'])
        if item['worktree']:
            subprocess.run(['git', '-C', str(args.repo), 'worktree', 'move', str(source), str(target)], check=True)
        else:
            try:
                source.rename(target)
            except PermissionError:
                # Root-owned download folders cannot change their '..' as this
                # user. Move their enclosing owned directory below instead.
                item['archive'] = str(archive / 'retired-root' / source.name)
                item['parent_move'] = True
                continue
        source.symlink_to(target, target_is_directory=target.is_dir())
    retired = archive / 'retired-root'
    if not args.old_root.is_symlink():
        args.old_root.rename(retired)
        args.old_root.symlink_to(retired, target_is_directory=True)
    receipt = {'time': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'canonical_repo': str(args.repo),
               'policy': 'Archives only; do not launch development/generation here. Old paths are provenance aliases.', 'entries': entries}
    (archive / 'CONSOLIDATION.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (args.old_root / 'RETIRED.md').write_text('This mirror is retired. Work in /home/werg/natlang. Historical aliases point into runs/dgx-legacy-imports; do not run new work here.\n')


if __name__ == '__main__':
    main()
