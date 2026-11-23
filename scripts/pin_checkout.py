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



def create(commit, dest):
    dest = destination(dest)
    commit = git('rev-parse', '--verify', f'{commit}^{{commit}}').strip()
    with destination_lock(dest) as lock_fd:
        return create_locked(commit, dest, lock_fd)


def create_locked(commit, dest, lock_fd):
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
    ignored = git('ls-files', '-o', '-i', '--exclude-standard', '--directory').splitlines()
    for path in ignored:
        top = path.split('/', 1)[0].rstrip('/')
        if path.startswith(SKIPPED):
            continue
        source, target = REPO / path, dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if top in LINKED:                  # data under a tracked data root (runs/, data/, models/): linked, never copied
            if '/' in path.rstrip('/') and not target.exists():
                target.symlink_to(source)
            continue
        inner = [s for s in SHARED if s.startswith(path.rstrip('/') + '/')]
        subprocess.run(['rsync', '-a', *[f'--exclude=/{s[len(path.rstrip("/")) + 1:]}' for s in inner],
                        str(source).rstrip('/') + ('/' if source.is_dir() else ''), str(target).rstrip('/')], check=True, pass_fds=(lock_fd,))
        for s in inner:
            if (REPO / s).exists():
                (dest / s).parent.mkdir(parents=True, exist_ok=True)
                if not (dest / s).is_symlink():
                    (dest / s).symlink_to(REPO / s)
                elif (dest / s).resolve() != (REPO / s).resolve():
                    raise SystemExit(f'{dest / s} points outside the shared model store')
    for name in LINKED:
        if (REPO / name).exists() and not (dest / name).exists():
            (dest / name).symlink_to(REPO / name)
    node = sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin'))
    env = {**os.environ, 'PATH': f'{node[-1]}:{os.environ["PATH"]}'} if node else None
    subprocess.run(['npm', 'run', '--silent', 'build:node'], cwd=dest / 'ts-host', env=env, check=True,
                   stdout=sys.stderr, pass_fds=(lock_fd,))
    registered_checkout(dest, commit)
    publish_marker(marker, {'commit': commit, 'created': time.time(), 'source': str(REPO)})
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
