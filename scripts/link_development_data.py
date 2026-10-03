#!/usr/bin/env python3
"""Expose external pipeline artifacts without symlinking Git-tracked parents."""
import argparse
import json
from pathlib import Path
import subprocess


def link_data(repo, storage):
    repo = Path(repo).resolve(strict=True)
    storage = Path(storage).resolve(strict=True)
    tracked = set(subprocess.check_output(
        ['git', '-C', str(repo), 'ls-files', '-z']).decode().split('\0'))
    ancestors = set()
    for name in tracked:
        p = Path(name)
        ancestors.update(str(parent) for parent in p.parents if str(parent) != '.')
    linked = 0
    conflicts = []

    def expose(relative):
        nonlocal linked
        source = storage / relative
        target = repo / relative
        if not source.resolve().is_relative_to(storage):
            conflicts.append({'path': relative, 'reason': 'source_reference_outside_external_mirror'})
            return
        if target.is_symlink():
            if target.resolve() != source.resolve():
                raise ValueError(f'unexpected development link: {target}')
            if relative in ancestors or relative in tracked:
                raise ValueError(f'tracked Git path has a symlink parent: {relative}')
            return
        if relative in tracked:
            if not target.is_file():
                raise ValueError(f'tracked checkout file is missing: {relative}')
            # Git, rather than a data mirror, owns tracked source/examples.
            return
        if source.is_dir() and relative in ancestors:
            target.mkdir(parents=True, exist_ok=True)
            for item in source.iterdir():
                if item.name in {'.~tmp~', '.natlang-sync-partial', '.natlang-priority-partial'}:
                    continue
                expose(relative + '/' + item.name)
        elif target.exists():
            if target.is_dir() and source.is_dir():
                for item in source.iterdir():
                    if item.name not in {'.~tmp~', '.natlang-sync-partial', '.natlang-priority-partial'}:
                        expose(relative + '/' + item.name)
            else:
                conflicts.append({'path': relative, 'reason': 'preexisting_development_path_preserved'})
            # Preserve any pre-existing development file; never overwrite it.
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(source, target_is_directory=source.is_dir())
            linked += 1

    for relative in ['data', 'runs', 'vendor/datasets', 'vendor/directory-sources',
                     'vendor/directory-sources-expansion-20260929']:
        if (storage / relative).exists():
            expose(relative)
    report = {'linked_new_branches': linked, 'conflicts': conflicts,
              'tracked_paths_owned_by_git': True}
    temporary = storage / '.projection-report.json.tmp'
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(storage / '.projection-report.json')
    print(f'Linked {linked} new pipeline branches; tracked checkout paths preserved; {len(conflicts)} conflicts recorded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True)
    parser.add_argument('--storage', required=True)
    args = parser.parse_args()
    link_data(args.repo, args.storage)
