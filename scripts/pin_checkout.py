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
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINKED = ('runs', 'data', 'models', '.venv', '.venv-neuralese', '.venv-neuralese-audit', '.venv-decider-systempython')
SKIPPED = ('.local/', '.coordination/', '.development-source-snapshots/', '.pytest_cache/', '.ruff_cache/', '.claude/')


def git(*args, cwd=REPO):
    return subprocess.run(['git', *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def create(commit, dest):
    dest = Path(dest)
    commit = git('rev-parse', commit).strip()
    marker = dest / '.natlang-pin.json'
    if marker.exists():
        recorded = json.loads(marker.read_text())
        if recorded['commit'] != commit:
            raise SystemExit(f'{dest} pins {recorded["commit"][:12]}, not {commit[:12]}')
        return dest
    git('worktree', 'add', '--detach', str(dest), commit)
    ignored = git('ls-files', '-o', '-i', '--exclude-standard', '--directory').splitlines()
    for path in ignored:
        top = path.split('/', 1)[0].rstrip('/')
        if top in LINKED or path.startswith(SKIPPED):
            continue
        source, target = REPO / path, dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['cp', '-a', '--no-target-directory', str(source).rstrip('/'), str(target).rstrip('/')], check=True)
    for name in LINKED:
        if (REPO / name).exists() and not (dest / name).exists():
            (dest / name).symlink_to(REPO / name)
    node = sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin'))
    env = {**os.environ, 'PATH': f'{node[-1]}:{os.environ["PATH"]}'} if node else None
    subprocess.run(['npm', 'run', '--silent', 'build:node'], cwd=dest / 'ts-host', env=env, check=True,
                   stdout=sys.stderr)
    marker.write_text(json.dumps({'commit': commit, 'created': time.time(), 'source': str(REPO)}) + '\n')
    return dest


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
        git('worktree', 'remove', '--force', str(args.dest))


if __name__ == '__main__':
    main()
