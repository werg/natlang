#!/usr/bin/env python3
"""Derive an isolated skill-authoring runtime from an already sealed dependency tree."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat

from clone_runtime_tree import clone_runtime_tree


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def inventory(root):
    files, links = {}, []
    for path in sorted(root.rglob('*')):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError('runtime symlink escapes snapshot: ' + relative)
            links.append({'path': relative, 'target': os.readlink(path)})
        elif path.is_file() and relative != 'frozen-runtime.json':
            files[relative] = digest(path)
    return files, links


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent, source = args.parent.resolve(strict=True), args.source.resolve(strict=True)
    output = args.output.resolve()
    parent_manifest = json.loads((parent / 'frozen-runtime.json').read_text())
    parent_files, parent_links = inventory(parent)
    if parent_files != parent_manifest['files'] or parent_links != parent_manifest['symlinks']:
        raise ValueError('parent runtime seal changed')
    selected = {}
    for directory in ['src', 'dist', 'scripts/skills']:
        for path in sorted((source / directory).rglob('*')):
            if path.is_file():
                if path.is_symlink():
                    raise ValueError('source code must be physical files')
                selected[path.relative_to(source).as_posix()] = digest(path)
    if 'dist/improvement/skill-authoring.js' not in selected or 'scripts/skills/collect-episodes.mjs' not in selected:
        raise ValueError('build the new authoring runtime before freezing')
    clone_runtime_tree(parent, output)
    # Only the fresh copy becomes writable; no shared inodes or dependency installs.
    for path in [output, *output.rglob('*')]:
        if path.is_dir() and not path.is_symlink():
            os.chmod(path, stat.S_IMODE(path.stat().st_mode) | 0o200)
    for directory in ['src', 'dist', 'scripts/skills']:
        target = output / directory
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source / directory, target)
    (output / 'frozen-runtime.json').unlink()
    copied = {relative: digest(output / relative) for relative in selected}
    current = {relative: digest(source / relative) for relative in selected}
    actual_paths = {path.relative_to(source).as_posix() for directory in ['src','dist','scripts/skills']
                    for path in (source / directory).rglob('*') if path.is_file()}
    if set(selected) != actual_paths or selected != copied or selected != current:
        raise ValueError('source changed during runtime preparation; keep failed candidate and use a fresh output')
    files, links = inventory(output)
    for relative in files:
        path = output / relative
        if path.stat().st_nlink != 1:
            raise ValueError('shared writable inode in new snapshot: ' + relative)
        os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~0o222)
    manifest = {'schema': 'natlang.skill-authoring-runtime/1', 'files': files, 'symlinks': links,
        'parent_runtime_manifest_sha256': digest(parent / 'frozen-runtime.json'),
        'derivation': 'Physical sealed parent dependency copy; exact current source/compiled/skill scripts copied without installing dependencies',
        'source_files': selected, 'package_policy': 'Parent dependency versions retained; direct relative authoring entrypoint'}
    manifest_path = output / 'frozen-runtime.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    os.chmod(manifest_path, 0o444)
    for path in [*output.rglob('*'), output]:
        if path.is_dir() and not path.is_symlink():
            os.chmod(path, stat.S_IMODE(path.stat().st_mode) & ~0o222)
    print(json.dumps({'runtime': str(output), 'manifest_sha256': digest(manifest_path), 'files': len(files)}))


if __name__ == '__main__':
    main()
