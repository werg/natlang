#!/usr/bin/env python3
"""Prune completed pytest runs independently, preserving live and recent runs."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.archive_to_external import open_paths, process_cwds


def live_lock(directory: Path) -> bool:
    try:
        pid = int((directory / '.lock').read_text().strip())
    except FileNotFoundError:
        return False
    except (OSError, ValueError):
        return True  # An unreadable/partially written lock is not a dead owner.
    if pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def in_use(directory: Path, held: set[str]) -> bool:
    prefix = str(directory) + '/'
    return live_lock(directory) or any(path == str(directory) or path.startswith(prefix) for path in held)


def candidates(root: Path, held: set[str], *, age_seconds: float, keep_recent: int, now: float):
    if root.is_symlink() or not root.is_dir() or root.stat().st_uid != os.getuid():
        return []
    runs = [path for path in root.iterdir() if re.fullmatch(r'pytest-\d+', path.name)
            and not path.is_symlink() and path.is_dir() and path.stat().st_uid == os.getuid()]
    protected = set(sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)[:keep_recent])
    current = root / 'pytest-current'
    if current.is_symlink():
        protected.add(current.resolve())
    result = []
    cutoff = now - age_seconds
    for directory in runs:
        if directory in protected or in_use(directory, held):
            continue
        # A parent directory's old mtime does not establish that its contents are cold.
        try:
            recent = directory.stat().st_mtime > cutoff
            for parent, dirs, files in os.walk(directory, followlinks=False):
                if recent:
                    break
                recent = any((Path(parent) / name).lstat().st_mtime > cutoff for name in dirs + files)
        except OSError:
            continue
        if not recent:
            result.append(directory)
    return sorted(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--age-minutes', type=float, default=180)
    parser.add_argument('--keep-recent', type=int, default=3)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.age_minutes < 0 or args.keep_recent < 1:
        parser.error('age must be nonnegative and at least one recent run must be retained')
    root = args.root.absolute()
    held = open_paths() | process_cwds()
    for directory in candidates(root, held, age_seconds=args.age_minutes * 60,
                                keep_recent=args.keep_recent, now=time.time()):
        # Refresh process custody before each deletion, including mapped files.
        if args.apply:
            if directory.is_symlink() or in_use(directory, open_paths() | process_cwds()):
                continue
            shutil.rmtree(directory)
        print(json.dumps({'at':time.time(), 'pytest_scratch':str(directory), 'removed':args.apply}), flush=True)


if __name__ == '__main__':
    main()
