#!/usr/bin/env python3
"""Remove experiment pins and code worktrees on the DGX root disk that nothing uses any more.

Every experiment makes a ~4 GB pin (``pin_checkout.py create``); nothing removed them when the experiment ended, so
the root disk kept falling under its floor (DECISIONS 2026-11-23 storage). Run by ``storage_sweep.sh`` (the
``natlang-storage-sweep`` timer, under the memory ledger). Dry run unless ``--apply``.

A candidate is a git worktree of the shared checkout, or a directory holding ``.natlang-pin.json`` under the ops and
runs roots, that lies on the root filesystem. It is removed only when all of these hold:

- it is older than ``--min-age-hours`` (pin creation time, else the newest mtime of its top-level entries);
- no process references it: cwd, executable, open files, memory maps, command line or environment (any readable
  process, so container processes and ledger units count);
- no active or timer-triggered user unit names it in its command line;
- no watch or state file names it: files of at most 2 MB modified within ``--state-days`` days directly in the ops
  tree's first three levels (drivers' state, launch scripts, ``watch`` directories);
- it carries no ``.keep-on-nvme`` / ``.natlang-keep`` marker and is not listed in
  ``~/.config/natlang/archive-exclude.txt`` (put a pin there to keep it, e.g. a driver awaiting its verdict);
- a worktree on a branch is clean and its branch tip is contained in origin/main (agent work is never discarded);
  a detached worktree without a pin marker is clean in tracked files.

Pins are removed with ``git worktree remove --force`` (they never hold commits), stray pin directories that are not
registered worktrees with ``rm -rf`` (symlinks into the data roots are unlinked, never followed). Each decision is
logged as one JSON line.

    python3 scripts/prune_stale_checkouts.py                 # dry run
    python3 scripts/prune_stale_checkouts.py --apply
    python3 scripts/prune_stale_checkouts.py --check PATH... # exit 1 if any PATH is referenced (for cleanup scripts)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[1]
OPS = Path(os.environ.get('NATLANG_OPS_ROOT', HOME / 'data' / 'lfm3-rl-dgx-20261029' / 'ops'))
PIN_ROOTS = (OPS, REPO / 'runs', HOME / 'data')
EXCLUDE_FILE = HOME / '.config' / 'natlang' / 'archive-exclude.txt'
MARKERS = ('.keep-on-nvme', '.natlang-keep')
STATE_MAX_BYTES = 2 * 2**20


def _under(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip('/') + '/')


def process_references() -> list[tuple[str, str, str]]:
    """(pid, kind, text) for every path-bearing fact of every readable process: cwd, exe, fds, maps (paths), and
    cmdline / environ (whole text, matched by substring)."""
    found = []
    me = str(os.getpid())
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or pid == me:
            continue
        base = f'/proc/{pid}'
        for kind in ('cwd', 'exe'):
            try:
                found.append((pid, kind, os.readlink(f'{base}/{kind}')))
            except OSError:
                pass
        try:
            for fd in os.listdir(f'{base}/fd'):
                try:
                    found.append((pid, 'fd', os.readlink(f'{base}/fd/{fd}')))
                except OSError:
                    pass
        except OSError:
            pass
        try:
            with open(f'{base}/maps') as maps:
                for line in maps:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].startswith('/'):
                        found.append((pid, 'map', parts[5].strip().removesuffix(' (deleted)')))
        except OSError:
            pass
        for kind in ('cmdline', 'environ'):
            try:
                with open(f'{base}/{kind}', 'rb') as stream:
                    found.append((pid, kind, stream.read().replace(b'\0', b' ').decode('utf-8', 'replace')))
            except OSError:
                pass
    return found


def unit_commands() -> list[tuple[str, str]]:
    """(unit, ExecStart text) of active user units and of the services user timers trigger."""
    def show(units, prop):
        if not units:
            return []
        out = subprocess.run(['systemctl', '--user', 'show', '-p', 'Id', '-p', prop, *units], capture_output=True,
                             text=True).stdout
        rows, current = [], {}
        for line in out.splitlines() + ['']:
            if not line:
                if current:
                    rows.append(current)
                current = {}
            elif '=' in line:
                key, value = line.split('=', 1)
                current[key] = value
        return rows
    try:
        listed = subprocess.run(['systemctl', '--user', 'list-units', '--all', '--no-legend', '--plain',
                                 '--type=service,timer'], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    services, timers = [], []
    for line in listed.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        unit, active = parts[0], parts[2]
        if unit.endswith('.timer'):
            timers.append(unit)
        elif active in ('active', 'activating', 'reloading', 'deactivating'):
            services.append(unit)
    services += [row.get('Triggers', '') for row in show(timers, 'Triggers') if row.get('Triggers')]
    return [(row.get('Id', ''), row.get('ExecStart', '')) for row in show(sorted(set(services)), 'ExecStart')]


def state_files(days: float, now: float) -> list[Path]:
    """Small recent files in the ops tree's first two levels and in its watch directories."""
    found = []
    if not OPS.is_dir():
        return found
    def subdirs(directory):
        try:
            return [d for d in directory.iterdir() if d.is_dir() and not d.is_symlink()]
        except OSError:
            return []
    first = subdirs(OPS)
    second = [d for f in first for d in subdirs(f)]
    dirs = [OPS, *first, *second]
    # watch directories three levels down, found without following symlinks into the HDD
    dirs += [w for s in second for t in subdirs(s) for w in subdirs(t) if w.name == 'watch']
    for directory in dirs:
        try:
            entries = list(directory.iterdir())
        except OSError:
            continue
        for path in entries:
            try:
                st = path.lstat()
            except OSError:
                continue
            if path.is_file() and not path.is_symlink() and st.st_size <= STATE_MAX_BYTES \
                    and now - st.st_mtime <= days * 86400:
                found.append(path)
    return found


def user_excludes(path: Path = EXCLUDE_FILE) -> list[str]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    return [os.path.expanduser(line.strip()) for line in lines if line.strip() and not line.startswith('#')]


def git(*args, cwd=REPO, check=True):
    return subprocess.run(['git', *args], cwd=cwd, check=check, capture_output=True, text=True).stdout


def worktrees() -> dict[str, dict]:
    """Registered worktrees of the shared checkout (except the checkout itself): path -> {head, branch}."""
    found, current = {}, {}
    for line in git('worktree', 'list', '--porcelain').splitlines() + ['']:
        if not line:
            if current.get('worktree') and Path(current['worktree']) != REPO:
                found[current['worktree']] = current
            current = {}
        else:
            key, _, value = line.partition(' ')
            current[key] = value or True
    return found


def pin_dirs(roots=None, max_depth: int = 3) -> list[str]:
    """Directories holding ``.natlang-pin.json`` at most MAX_DEPTH levels under the roots, on the root filesystem.

    The walk never follows symlinks and never leaves the root filesystem: ``~/data`` holds hundreds of symlinks into
    the external HDD, and ``Path.glob`` followed them into directories of tens of thousands of entries, which kept a
    sweep in disk sleep past the next timer start (2026-11-23)."""
    found = []
    try:
        root_dev = os.stat('/').st_dev
    except OSError:
        return found
    for root in PIN_ROOTS if roots is None else roots:
        try:
            if root.is_symlink() or not root.is_dir() or os.stat(root).st_dev != root_dev:
                continue
        except OSError:
            continue
        level = [str(root)]
        for _ in range(max_depth):
            following = []
            for directory in level:
                try:
                    with os.scandir(directory) as entries:
                        for entry in entries:
                            try:
                                if not entry.is_dir(follow_symlinks=False) \
                                        or entry.stat(follow_symlinks=False).st_dev != root_dev:
                                    continue
                            except OSError:
                                continue
                            if os.path.lexists(os.path.join(entry.path, '.natlang-pin.json')):
                                found.append(entry.path)
                            following.append(entry.path)
                except OSError:
                    continue
            level = following
    return sorted(set(found))


def age_hours(path: Path, now: float) -> float:
    marker = path / '.natlang-pin.json'
    try:
        created = json.loads(marker.read_text()).get('created')
        if created:
            return (now - float(created)) / 3600
    except (OSError, ValueError):
        pass
    newest = 0.0
    try:
        newest = path.lstat().st_mtime
        for entry in path.iterdir():
            try:
                newest = max(newest, entry.lstat().st_mtime)
            except OSError:
                pass
    except OSError:
        pass
    return (now - newest) / 3600


def references(path: str, procs, units, states) -> list[str]:
    """Why ``path`` is in use (empty: unused)."""
    why = []
    real = os.path.realpath(path)
    names = {path, real}
    for pid, kind, text in procs:
        if kind in ('cmdline', 'environ'):
            if any(name in text for name in names):
                why.append(f'pid {pid} {kind}')
        elif any(_under(text, name) for name in names):
            why.append(f'pid {pid} {kind} {text}')
        if len(why) >= 5:
            return why
    for unit, command in units:
        if any(name in command for name in names):
            why.append(f'unit {unit}')
    base = os.path.basename(path.rstrip('/'))
    for state in states:
        # the pin's own creation log (ops/ei/pin-<c12>.log, <run>/pin.log) names it without using it
        if _under(str(state), path) or state.name == 'pin.log' or state.name.startswith(base + '.'):
            continue
        try:
            text = state.read_text(errors='replace')
        except OSError:
            continue
        if path in text or real in text or (base.startswith('pin-') and base in text):
            why.append(f'state {state}')
    return why


def keep_reason(path: str, excludes) -> str | None:
    for marker in MARKERS:
        if (Path(path) / marker).exists():
            return f'marker {marker}'
    for ex in excludes:
        if _under(path, ex):
            return f'archive-exclude {ex}'
    return None


def on_root_fs(path: str) -> bool:
    try:
        return os.stat(path).st_dev == os.stat('/').st_dev
    except OSError:
        return False


def candidates():
    trees = worktrees()
    paths = set(trees) | set(pin_dirs())
    return [(path, trees.get(path)) for path in sorted(paths)]


def tree_verdict(path: str, tree: dict | None) -> str | None:
    """None if the checkout may go; else why it stays (agent work, local changes)."""
    pin = (Path(path) / '.natlang-pin.json').exists()
    if tree is None:
        return None if pin else 'not a worktree or pin'
    if tree.get('prunable'):
        return None
    branch = tree.get('branch')
    try:
        dirty = git('status', '--porcelain', '--untracked-files=no', cwd=path, check=False).strip()
    except OSError:
        return 'status failed'
    if branch:
        if dirty:
            return f'branch {branch} has local changes'
        merged = subprocess.run(['git', 'merge-base', '--is-ancestor', tree.get('HEAD', ''), 'origin/main'], cwd=REPO,
                                capture_output=True).returncode == 0
        return None if merged else f'branch {branch} not in origin/main'
    if not pin and dirty:
        return 'detached worktree with local changes'
    return None


def remove(path: str, tree: dict | None):
    if tree is not None:
        subprocess.run(['git', 'worktree', 'remove', '--force', '--force', path], cwd=REPO, check=True,
                       capture_output=True)
    if os.path.lexists(path):
        shutil.rmtree(path)
    if tree is not None:
        git('worktree', 'prune', check=False)


def du_bytes(path: str) -> int:
    try:
        out = subprocess.run(['du', '-sxb', path], capture_output=True, text=True, timeout=600).stdout
        return int(out.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return 0


def log(record, stream):
    record = {'at': time.strftime('%Y-%m-%dT%H:%M:%S'), **record}
    print(json.dumps(record), file=stream, flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--min-age-hours', type=float, default=6.0)
    parser.add_argument('--state-days', type=float, default=3.0)
    parser.add_argument('--check', nargs='+', metavar='PATH', help='report references of PATHs; exit 1 if any')
    args = parser.parse_args(argv)
    now = time.time()
    procs, units, states = process_references(), unit_commands(), state_files(args.state_days, now)
    if args.check:
        busy = False
        for path in args.check:
            path = os.path.abspath(path)
            why = references(path, procs, units, [])
            if why:
                busy = True
                log({'path': path, 'referenced': why}, sys.stdout)
        return 1 if busy else 0
    excludes = user_excludes()
    freed = 0
    for path, tree in candidates():
        if not os.path.isdir(path) or not on_root_fs(path):
            continue
        record = {'path': path, 'kind': 'pin' if (Path(path) / '.natlang-pin.json').exists() else 'worktree'}
        age = age_hours(Path(path), now)
        record['age_hours'] = round(age, 1)
        why = keep_reason(path, excludes)
        if why is None and age < args.min_age_hours:
            why = 'young'
        if why is None:
            why = tree_verdict(path, tree)
        if why is None:
            refs = references(path, procs, units, states)
            why = '; '.join(refs[:3]) if refs else None
        if why:
            log({**record, 'kept': why}, sys.stdout)
            continue
        size = du_bytes(path)
        if args.apply:
            try:
                remove(path, tree)
            except (OSError, subprocess.CalledProcessError) as error:
                log({**record, 'error': str(error)}, sys.stdout)
                continue
            freed += size
        log({**record, 'removed' if args.apply else 'would_remove': True, 'bytes': size}, sys.stdout)
    log({'summary': True, 'applied': args.apply, 'freed_bytes': freed}, sys.stdout)
    return 0


if __name__ == '__main__':
    sys.exit(main())
