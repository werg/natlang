#!/usr/bin/env python3
"""A pinned copy of the checkout for one experiment (AGENTS.md "Improvements enter at the next experiment").

Creates a detached worktree of COMMIT at DEST (no commits are ever made there), copies the checkout's ignored build
outputs into it (ts-host/dist and node_modules, applications/dist, generated .d.nl.ts, .natlang), and links the large
ignored data roots (runs, data, models, the venvs) back to the checkout. Everything an experiment executes (learner,
actor, collector, harness, scorer, held eval) then runs this one commit, and `code_provenance` reports it.

    pin_checkout.py create COMMIT DEST      # prints DEST; reuses an existing pin of the same commit
    pin_checkout.py remove DEST             # git worktree remove (when the experiment ends)

Ignored build outputs are copied from the checkout, then ts-host is rebuilt in the pin (``npm run build:node``) so the
harness the episodes run is built from the pinned commit's sources.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINKED = ('runs', 'data', 'models', '.venv', '.venv-neuralese', '.venv-neuralese-audit', '.venv-decider-systempython')
# Content-addressed model packages (14 GB of checkpoints keyed by sha256): linked like the data roots, never copied.
SHARED = ('.natlang/packages/native-models',)
SKIPPED = ('.local/', '.coordination/', '.development-source-snapshots/', '.pytest_cache/', '.ruff_cache/', '.claude/')


def git(*args, cwd=None, pass_fds=()):
    return subprocess.run(['git', *args], cwd=cwd or REPO, check=True, capture_output=True,
                          text=True, pass_fds=pass_fds).stdout


def destination(dest):
    path = Path(dest).expanduser().absolute()
    return path.parent.resolve() / path.name


@contextlib.contextmanager
def destination_lock(dest):
    # Shared across every checkout of this repository; never unlink a lock inode
    # while a waiter may still hold it. The pin itself can be removed safely.
    common = Path(git('rev-parse', '--path-format=absolute', '--git-common-dir').strip())
    directory = common / 'natlang-pin-locks'
    directory.mkdir(exist_ok=True)
    key = hashlib.sha256(os.fsencode(dest)).hexdigest()
    with (directory / key).open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield lock.fileno()


def registered_checkout(dest, commit):
    records = git('worktree', 'list', '--porcelain', '-z').split('\0\0')
    record = next((dict(field.split(' ', 1) if ' ' in field else (field, '')
                        for field in row.split('\0') if field)
                   for row in records if row.startswith(f'worktree {dest}\0')), None)
    if record is None or dest.is_symlink() or not dest.is_dir():
        raise SystemExit(f'{dest} is not an existing registered pin checkout')
    actual = git('rev-parse', 'HEAD', cwd=dest).strip()
    if record.get('HEAD') != commit or actual != commit or 'detached' not in record:
        raise SystemExit(f'{dest} must be a detached checkout of {commit[:12]}, found {actual[:12]}')
    if git('status', '--porcelain', '--untracked-files=no', cwd=dest).strip():
        raise SystemExit(f'{dest} has modified tracked sources; refusing pin reuse or removal')


def publish_marker(marker, recorded):
    # A failed/killed write leaves no completion claim. Leftover temporary files
    # are untracked and harmless on the next exact-checkout retry.
    fd, temporary = tempfile.mkstemp(prefix='.natlang-pin-', suffix='.tmp', dir=marker.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(json.dumps(recorded) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, marker)
        directory = os.open(marker.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)



def ensure_link(source, target):
    """Preserve the intended link, including when its source is dangling."""
    if target.is_symlink():
        # Compare link destinations lexically, without dereferencing either side:
        # canonical shared roots may themselves intentionally be dangling links.
        actual = os.path.abspath(target.parent / os.readlink(target))
        if actual != os.path.abspath(source):
            raise SystemExit(f'{target} points to {actual}, not the intended shared path {source}')
        return
    if os.path.lexists(target):
        raise SystemExit(f'{target} conflicts with the intended shared link to {source}')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(source)


def ensure_shared_view(source, target):
    """Accept helper scaffolding only when every existing leaf is its exact shared link."""
    if target.is_symlink() or not os.path.lexists(target):
        ensure_link(source, target)
        return
    if not target.is_dir() or not source.is_dir():
        raise SystemExit(f'{target} conflicts with the intended shared view of {source}')
    for child in target.iterdir():
        original = source / child.name
        if child.is_symlink():
            ensure_link(original, child)
        elif child.is_dir() and original.is_dir():
            ensure_shared_view(original, child)
        else:
            raise SystemExit(f'{child} contains unique data, not an exact shared link')
    # A previous preparation may have stopped halfway through the container.
    # Fill missing aliases; never replace or remove an existing unique path.
    for original in source.iterdir():
        if not os.path.lexists(target / original.name):
            ensure_link(original, target / original.name)


def create(commit, dest):
    dest = destination(dest)
    commit = git('rev-parse', '--verify', f'{commit}^{{commit}}').strip()
    with destination_lock(dest) as lock_fd:
        return create_locked(commit, dest, lock_fd)


def create_locked(commit, dest, lock_fd):
    # Creating from an existing pin must not make its data/env lifetime an ancestor.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/neuralese'))
    from natlang_neuralese.common.data_registry import canonical_checkout_repo
    owner = canonical_checkout_repo(REPO)
    marker = dest / '.natlang-pin.json'
    if dest.exists() or dest.is_symlink():
        registered_checkout(dest, commit)
        if marker.exists():
            recorded = json.loads(marker.read_text())
            if recorded['commit'] != commit:
                raise SystemExit(f'{dest} pins {recorded["commit"][:12]}, not {commit[:12]}')
            return dest
    else:
        git('worktree', 'add', '--detach', str(dest), commit, pass_fds=(lock_fd,))
        registered_checkout(dest, commit)
    # Establish shared ownership before processing ignored listings. Nested
    # .gitignore rules can list both a directory and its descendants.
    covered = set(SHARED)
    for name in LINKED:
        source, target = owner / name, dest / name
        if not os.path.lexists(source):
            continue
        if git('ls-files', '--', name, cwd=dest).strip():
            continue
        ensure_shared_view(source, target)
        covered.add(name)
    for name in SHARED:
        if os.path.lexists(owner / name):
            ensure_link(owner / name, dest / name)
    ignored = git('ls-files', '-o', '-i', '--exclude-standard', '--directory').splitlines()
    for path in sorted(ignored, key=lambda name: (name.rstrip('/').count('/'), name)):
        relative = path.rstrip('/')
        if path.startswith(SKIPPED) or any(relative == root or relative.startswith(root + '/') for root in covered):
            continue
        top = relative.split('/', 1)[0]
        source, target = owner / relative, dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if top in LINKED:
            if relative != top:  # tracked roots remain real directories
                ensure_shared_view(source, target)
                covered.add(relative)
            continue
        if source.is_symlink():
            ensure_link(source, target)
            covered.add(relative)
            continue
        if target.is_symlink():
            raise SystemExit(f'{target} is a shared link; refusing to copy through it')
        # Every ancestor copy excludes the shared store, even when its source
        # is missing. No rsync operation may write through its destination link.
        inner = [root for root in SHARED if root.startswith(relative + '/')]
        subprocess.run(['rsync', '-a', *[f'--exclude=/{root[len(relative) + 1:]}' for root in inner],
                        str(source).rstrip('/') + ('/' if source.is_dir() else ''), str(target)],
                       check=True, pass_fds=(lock_fd,))
        covered.add(relative)
    node = sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin'))
    env = {**os.environ, 'PATH': f'{node[-1]}:{os.environ["PATH"]}'} if node else None
    subprocess.run(['npm', 'run', '--silent', 'build:node'], cwd=dest / 'ts-host', env=env, check=True,
                   stdout=sys.stderr, pass_fds=(lock_fd,))
    registered_checkout(dest, commit)
    publish_marker(marker, {'commit': commit, 'created': time.time(), 'source': str(owner)})
    return dest


def remove(dest):
    dest = destination(dest)
    with destination_lock(dest) as lock_fd:
        commit = git('rev-parse', 'HEAD', cwd=dest).strip()
        registered_checkout(dest, commit)
        git('worktree', 'remove', '--force', str(dest), pass_fds=(lock_fd,))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('create')
    c.add_argument('commit')
    c.add_argument('dest')
    r = sub.add_parser('remove')
    r.add_argument('dest')
    args = parser.parse_args()
    if args.command == 'create':
        print(create(args.commit, args.dest))
    else:
        remove(args.dest)


if __name__ == '__main__':
    main()
