#!/usr/bin/env python3
"""Move finished large files (checkpoints, dumps, exports) from the DGX NVMe to the external HDD.

The NVMe holds latency-critical working data; finished artifacts belong on /mnt/external. A candidate is a regular
file of at least --min-gb under one of the roots, neither modified nor read for --min-age-hours, not open by any process (host
/proc fds and memory maps, which include container processes), and not excluded (built-in paths, files and out dirs
named by active memory-ledger claims, the user exclude file, a `.keep-on-nvme` marker in any parent directory).

Each move copies to ARCHIVE/<root name>/<path relative to the root> preserving mtime, fsyncs, verifies size and
SHA-256 against the source, checks the source did not change meanwhile, then atomically replaces the source with a
symlink to the copy, so readers keep working. Every move is appended to the manifest (JSONL). Nothing is deleted
without a verified copy; symlinks are never followed. Dry run by default.

    python3 scripts/archive_to_external.py                 # dry run: what would move, plus a report
    python3 scripts/archive_to_external.py --apply --budget-gb 150
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

GIB = 2**30
HOME = Path.home()
ROOTS = (HOME / 'natlang' / 'runs', HOME / 'data')
ARCHIVE = Path('/mnt/external/natlang-development-data/archive')
MANIFEST_NAME = 'manifest.jsonl'
EXCLUDE_FILE = HOME / '.config' / 'natlang' / 'archive-exclude.txt'
LEDGER = Path(os.environ.get('NATLANG_MEMORY_LEDGER', HOME / '.local/state/natlang/memory-ledger.json'))
# Latency-critical working copies (memory: nvme-for-latency-critical-data) and data in active use.
BUILTIN_EXCLUDES = (
    HOME / 'data' / 'models',
    HOME / 'data' / 'bird-sqlite',
    HOME / 'natlang-data-nvme',
    HOME / 'data' / 'mellum-qat' / 'convert-v2',
)
BUILTIN_EXCLUDE_GLOBS = (str(HOME / 'data' / 'mellum-qat' / 'teacher-*'),)
MARKER = '.keep-on-nvme'


def process_cwds() -> set[str]:
    cwds = set()
    for pid in os.listdir('/proc'):
        if pid.isdigit():
            try:
                cwds.add(os.readlink(f'/proc/{pid}/cwd'))
            except OSError:
                pass
    return cwds


def run_dir(path: Path, roots) -> Path | None:
    """The run (first-level directory under its root) a file belongs to."""
    for root in roots:
        if root in path.parents:
            rel = path.relative_to(root).parts
            return root / rel[0] if len(rel) > 1 else None
    return None


def busy_run_dirs(roots, held, cwds) -> set[Path]:
    """Runs some process works in or holds a file of: not finished, whatever their files' ages."""
    busy = set()
    for item in set(held) | set(cwds):
        if item.startswith('/'):
            found = run_dir(Path(item), roots)
            if found:
                busy.add(found)
    return busy


# Package and kernel caches that rebuild themselves: cleared, not archived.
REGENERABLE_CACHES = ('pip', 'uv', 'node-gyp', 'yarn', 'ccache', 'triton', 'torch/inductor', 'torch_extensions')


def clear_regenerable(apply: bool, cache=HOME / '.cache') -> list[dict]:
    cleared = []
    for name in REGENERABLE_CACHES:
        path = cache / name
        if path.is_dir() and not path.is_symlink():
            size = int(subprocess.run(['du', '-sb', str(path)], capture_output=True, text=True).stdout.split()[0] or 0)
            if apply:
                shutil.rmtree(path, ignore_errors=True)
            cleared.append({'path': str(path), 'gb': round(size / GIB, 2), 'cleared': apply})
    return cleared


def open_paths() -> set[str]:
    """Every path any readable process holds open (fds) or mapped."""
    held = set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit():
            continue
        fd_dir = f'/proc/{pid}/fd'
        try:
            for fd in os.listdir(fd_dir):
                try:
                    held.add(os.readlink(f'{fd_dir}/{fd}'))
                except OSError:
                    pass
        except OSError:
            pass
        try:
            with open(f'/proc/{pid}/maps') as maps:
                for line in maps:
                    parts = line.split(None, 5)
                    if len(parts) == 6 and parts[5].startswith('/'):
                        held.add(parts[5].strip().removesuffix(' (deleted)'))
        except OSError:
            pass
    return held


def ledger_excludes(path: Path = LEDGER) -> list[Path]:
    """Absolute paths named in active ledger claims' commands (their inputs, outputs and out dirs)."""
    try:
        claims = json.loads(path.read_text()).get('claims', {})
    except (OSError, ValueError):
        return []
    found = []
    for claim in claims.values():
        command = claim.get('command') or []
        words = command if isinstance(command, list) else str(command).split()
        for word in words:
            for part in str(word).replace('=', ' ').split():
                if part.startswith('/home/') or part.startswith('/mnt/'):
                    found.append(Path(part.strip('\'"')))
        container = None
        text = ' '.join(map(str, words))
        if 'docker start' in text:
            container = text.split()[-1]
        if container:  # the container's own arguments name its paths
            try:
                args = subprocess.run(['docker', 'inspect', '-f', '{{json .Args}}', container], capture_output=True,
                                      text=True, timeout=10).stdout
                for part in json.loads(args or '[]'):
                    for piece in str(part).replace('=', ' ').split():
                        if piece.startswith('/home/') or piece.startswith('/mnt/'):
                            found.append(Path(piece))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
    return found


def user_excludes(path: Path = EXCLUDE_FILE) -> list[Path]:
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return []
    return [Path(os.path.expanduser(line.strip())) for line in lines if line.strip() and not line.startswith('#')]


def excluded(file: Path, excludes: list[Path], globs=BUILTIN_EXCLUDE_GLOBS, roots=None) -> str | None:
    """Why ``file`` stays, or None: it is an excluded path or lies under one. Excludes that are a root, or above
    one (a claim's working directory, a whole-home mount), protect nothing on their own."""
    roots = tuple(roots or ROOTS)
    for ex in excludes:
        if any(ex == root or ex in root.parents for root in roots):
            continue
        if file == ex or ex in file.parents:
            return f'excluded by {ex}'
    for pattern in globs:
        for parent in file.parents:
            if parent.match(pattern):
                return f'excluded by {pattern}'
    for parent in file.parents:
        if (parent / MARKER).exists():
            return f'{MARKER} in {parent}'
        if parent in roots or parent == HOME:
            break
    return None


def candidates(roots, *, min_bytes, min_age, excludes, held, now=None, busy=(), idle_min_age=None):
    """(path, size, reason-or-None) for every large file under the roots; reason None means it may move. Files of
    runs no process uses (not in ``busy``) may move after ``idle_min_age`` instead of ``min_age``."""
    now = now or time.time()
    for root in roots:
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in filenames:
                path = Path(dirpath) / name
                try:
                    st = path.lstat()
                except OSError:
                    continue
                if not path.is_file() or path.is_symlink() or st.st_size < min_bytes:
                    continue
                reason = excluded(path, excludes, roots=roots)
                owner = run_dir(path, roots)
                age = min_age
                if idle_min_age is not None and owner is not None and owner not in busy:
                    age = idle_min_age
                elif idle_min_age is not None and owner in busy:
                    reason = reason or f'run directory {owner.name} in use'
                min_age_here = age
                if reason is None and now - st.st_mtime < min_age_here:
                    reason = f'modified {(now - st.st_mtime) / 3600:.1f} h ago'
                # A training input is read for days without being modified; relatime still updates atime at least
                # daily, so a recent access keeps a hot input on the NVMe.
                if reason is None and now - st.st_atime < min_age_here:
                    reason = f'read {(now - st.st_atime) / 3600:.1f} h ago'
                if reason is None and str(path) in held:
                    reason = 'open by a process'
                yield path, st.st_size, reason


def sha256(path: Path, chunk=8 << 20, from_disk=False) -> str:
    """``from_disk``: drop the file's cached pages first, so the hash reads what reached the disk."""
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        if from_disk and hasattr(os, 'posix_fadvise'):
            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def destination(path: Path, roots, archive: Path) -> Path:
    for root in roots:
        if root in path.parents:
            return archive / root.name / path.relative_to(root)
    raise ValueError(f'{path} is under no root')


def archive_file(path: Path, dest: Path, manifest: Path) -> dict:
    """Copy, verify, then replace the source with a symlink. Raises (leaving the source untouched) on any doubt."""
    before = path.lstat()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + '.archiving')
    source_hash = hashlib.sha256()
    with open(path, 'rb') as src, open(partial, 'wb') as out:
        while block := src.read(8 << 20):
            source_hash.update(block)
            out.write(block)
        out.flush()
        os.fsync(out.fileno())
    os.utime(partial, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = path.lstat()
    if (after.st_size, after.st_mtime_ns, after.st_ino) != (before.st_size, before.st_mtime_ns, before.st_ino):
        partial.unlink()
        raise RuntimeError(f'{path} changed during the copy')
    digest = source_hash.hexdigest()
    if partial.stat().st_size != before.st_size or sha256(partial, from_disk=True) != digest:
        partial.unlink()
        raise RuntimeError(f'verification failed for {path}')
    os.replace(partial, dest)
    link = path.with_name(path.name + '.archive-link')
    os.symlink(dest, link)
    os.replace(link, path)  # atomic: readers see the old file or the link, never nothing
    record = {'source': str(path), 'dest': str(dest), 'bytes': before.st_size, 'sha256': digest,
              'mtime': before.st_mtime, 'time': time.time()}
    with open(manifest, 'a') as handle:
        handle.write(json.dumps(record) + '\n')
    return record


def report_extras(min_bytes):
    """Large items the owner may want to clean up by hand (never moved automatically)."""
    out = {}
    cache = HOME / '.cache'
    if cache.is_dir():
        sizes = []
        for child in cache.iterdir():
            try:
                used = int(subprocess.run(['du', '-sb', str(child)], capture_output=True, text=True,
                                          timeout=300).stdout.split()[0])
            except (OSError, ValueError, IndexError, subprocess.SubprocessError):
                continue
            if used >= min_bytes:
                sizes.append((used, str(child)))
        out['cache'] = [{'path': p, 'gb': round(b / GIB, 1)} for b, p in sorted(sizes, reverse=True)]
    repo = HOME / 'natlang'
    try:
        ignored = subprocess.run(['git', '-C', str(repo), 'ls-files', '--others', '--ignored', '--exclude-standard',
                                  '--directory'], capture_output=True, text=True, timeout=300).stdout.split('\n')
    except (OSError, subprocess.SubprocessError):
        ignored = []
    big = []
    for rel in ignored:
        if not rel or rel.startswith('runs/'):
            continue
        path = repo / rel
        try:
            used = int(subprocess.run(['du', '-sb', str(path)], capture_output=True, text=True,
                                      timeout=300).stdout.split()[0])
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            continue
        if used >= min_bytes:
            big.append((used, rel))
    out['repo_ignored'] = [{'path': p, 'gb': round(b / GIB, 1)} for b, p in sorted(big, reverse=True)[:30]]
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--apply', action='store_true', help='move files (default: dry run)')
    p.add_argument('--min-gb', type=float, default=1.0)
    p.add_argument('--min-age-hours', type=float, default=24.0)
    p.add_argument('--budget-gb', type=float, default=200.0, help='bytes moved per run at most')
    p.add_argument('--hdd-floor-gb', type=float, default=100.0, help='free space to keep on the external disk')
    p.add_argument('--archive', type=Path, default=ARCHIVE)
    p.add_argument('--root', type=Path, action='append', help='override the roots')
    p.add_argument('--report', action='store_true', help='also report ~/.cache and git-ignored large items')
    p.add_argument('--list-limit', type=int, default=40)
    p.add_argument('--idle-run-min-age-hours', type=float, default=None,
                   help='shorter age for files of runs no process uses (a cleanout); default: --min-age-hours')
    p.add_argument('--clear-regenerable-caches', action='store_true',
                   help='delete package/kernel caches under ~/.cache that rebuild themselves ' + str(REGENERABLE_CACHES))
    p.add_argument('--until-done', action='store_true', help='repeat --budget-gb batches until nothing is movable')
    a = p.parse_args(argv)
    roots = tuple(a.root) if a.root else ROOTS
    excludes = list(BUILTIN_EXCLUDES) + ledger_excludes() + user_excludes()
    held = open_paths()
    busy = busy_run_dirs(roots, held, process_cwds())
    idle_age = None if a.idle_run_min_age_hours is None else a.idle_run_min_age_hours * 3600
    movable, kept = [], []
    for path, size, reason in candidates(roots, min_bytes=int(a.min_gb * GIB), min_age=a.min_age_hours * 3600,
                                         excludes=excludes, held=held, busy=busy, idle_min_age=idle_age):
        (kept if reason else movable).append((size, str(path), reason))
    movable.sort(reverse=True)
    summary = {'apply': a.apply, 'movable_files': len(movable), 'movable_gb': round(sum(s for s, *_ in movable) / GIB, 1),
               'kept_files': len(kept), 'kept_gb': round(sum(s for s, *_ in kept) / GIB, 1),
               'nvme_free_gb': round(shutil.disk_usage(roots[0] if roots[0].exists() else '/').free / GIB, 1)}
    moved, failed, budget = [], [], int(a.budget_gb * GIB)
    if a.apply:
        a.archive.mkdir(parents=True, exist_ok=True)
        manifest = a.archive / MANIFEST_NAME
        for size, path, _ in movable:
            if sum(r['bytes'] for r in moved) + size > budget and not a.until_done:
                continue
            if a.until_done and sum(r['bytes'] for r in moved) + size > budget:
                budget += int(a.budget_gb * GIB)  # next batch: re-check the disk floor and that it is still closed
                if str(path) in open_paths():
                    failed.append({'path': path, 'error': 'opened since selection'})
                    continue
            if shutil.disk_usage(a.archive).free - size < a.hdd_floor_gb * GIB:
                failed.append({'path': path, 'error': 'external disk floor'})
                continue
            try:
                moved.append(archive_file(Path(path), destination(Path(path), roots, a.archive), manifest))
                print(json.dumps({'archived': path, 'gb': round(size / GIB, 2)}), flush=True)
            except (OSError, RuntimeError, ValueError) as error:
                failed.append({'path': path, 'error': str(error)[:300]})
        summary['nvme_free_after_gb'] = round(shutil.disk_usage(roots[0]).free / GIB, 1)
    summary.update(moved_files=len(moved), moved_gb=round(sum(r['bytes'] for r in moved) / GIB, 1), failed=failed,
                   would_move=[{'path': p, 'gb': round(s / GIB, 1)} for s, p, _ in movable[:a.list_limit]],
                   kept_largest=[{'path': p, 'gb': round(s / GIB, 1), 'why': r}
                                 for s, p, r in sorted(kept, reverse=True)[:a.list_limit]])
    if a.clear_regenerable_caches:
        summary['regenerable_caches'] = clear_regenerable(a.apply)
    if a.report:
        summary['report'] = report_extras(int(a.min_gb * GIB))
    print(json.dumps(summary, indent=1))
    return 1 if failed and a.apply else 0


if __name__ == '__main__':
    sys.exit(main())
