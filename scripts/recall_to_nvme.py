#!/usr/bin/env python3
"""Bring the hot files of an archived tree back onto the NVMe behind the same path (the inverse of an archive move).

A directory moved to the external HDD leaves a symlink at its NVMe path (``archive_to_external.py`` per file, ad-hoc
moves per directory). When it turns out to be a hot input again (a round build reading 27k per-group files of the
teacher pools spent hours in HDD seeks, 2026-11-23), this rebuilds the tree on the NVMe at the same path: directories
are real, every file is copied and verified (size, and SHA-256 of the copy against the bytes read from the HDD), and the paths matching ``--cold``
patterns (big files or directories no hot reader needs) stay on the HDD as symlinks to their archived copy, the same
per-path symlink the archiver leaves. Readers keep their paths; the HDD copy stays complete (it is the archive).

Steps: copy into ``LINK.recall-staging`` on the NVMe and verify; wait until no process holds a file or cwd under the
link; swap (rename the symlink aside, rename the staging tree in, remove the old symlink); write ``.keep-on-nvme`` so
the archiver leaves it; append registry records with the NVMe path for every dataset whose latest record lies under
the archived target. Dry run (sizes only) unless ``--apply``.

    python3 scripts/recall_to_nvme.py ~/data/lfm3-rl-dgx-20261029/ops/teacher-gen-20261115 \\
        --cold '*/claims' --cold '*/episodes/*/row-*' --cold '*/episodes/*/rollouts.jsonl' --apply
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
GIB = 2**30


def log(**fields):
    print(json.dumps({'at': time.strftime('%Y-%m-%dT%H:%M:%S'), **fields}), flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def is_cold(rel: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(rel, pattern) for pattern in patterns)


def plan(target: Path, patterns):
    """(directories, hot files, cold paths) relative to TARGET, without following symlinks."""
    dirs, hot, cold = [], [], []
    for directory, children, names in os.walk(target):
        rel_dir = os.path.relpath(directory, target)
        rel_dir = '' if rel_dir == '.' else rel_dir
        kept = []
        for child in sorted(children):
            rel = os.path.join(rel_dir, child)
            if is_cold(rel, patterns) or os.path.islink(os.path.join(directory, child)):
                cold.append(rel)
            else:
                kept.append(child)
                dirs.append(rel)
        children[:] = kept
        for name in sorted(names):
            rel = os.path.join(rel_dir, name)
            (cold if is_cold(rel, patterns) or os.path.islink(os.path.join(directory, name)) else hot).append(rel)
    return dirs, hot, cold


def copy_hashed(src: Path, dst: Path) -> str:
    """Copy SRC to DST (data and times) and return the SHA-256 of the bytes read, so the HDD is read once."""
    digest = hashlib.sha256()
    with open(src, 'rb') as reader, open(dst, 'wb') as writer:
        for block in iter(lambda: reader.read(8 << 20), b''):
            digest.update(block)
            writer.write(block)
    shutil.copystat(src, dst)
    return digest.hexdigest()


def build(target: Path, staging: Path, dirs, hot, cold) -> dict[str, str]:
    """Copy the hot files and link the cold paths; return the source SHA-256 of every file copied."""
    staging.mkdir(parents=True, exist_ok=True)
    for rel in dirs:
        (staging / rel).mkdir(exist_ok=True)
    copied, digests = 0, {}
    for index, rel in enumerate(hot):
        src, dst = target / rel, staging / rel
        if dst.exists() and dst.stat().st_size == src.stat().st_size \
                and dst.stat().st_mtime_ns == src.stat().st_mtime_ns:
            continue
        digests[rel] = copy_hashed(src, dst)
        copied += src.stat().st_size
        if index % 5000 == 0:
            log(event='copying', files=index, of=len(hot), gb=round(copied / GIB, 2))
    for rel in cold:
        link = staging / rel
        source = target / rel
        # an archived symlink keeps its own target; anything else points at its archived copy
        want = os.readlink(source) if os.path.islink(source) else str(source)
        if os.path.lexists(link):
            if os.path.islink(link) and os.readlink(link) == want:
                continue
            raise SystemExit(f'staging holds an unexpected entry: {link}')
        os.symlink(want, link)
    return digests


def verify(target: Path, staging: Path, hot, cold, digests=None) -> list[str]:
    """Paths whose staging copy differs from the archive. DIGESTS (from ``build``) spares re-reading the HDD for files
    hashed while they were copied; a file copied by an earlier, interrupted run is hashed on both sides."""
    bad, digests = [], digests or {}
    for rel in hot:
        src, dst = target / rel, staging / rel
        if not dst.is_file() or dst.is_symlink() or dst.stat().st_size != src.stat().st_size \
                or sha256(dst) != (digests.get(rel) or sha256(src)):
            bad.append(rel)
    for rel in cold:
        link = staging / rel
        if not link.is_symlink() or os.path.realpath(link) != os.path.realpath(target / rel):
            bad.append(rel)
    return bad


def holders(paths) -> list[int]:
    roots = [str(p) for p in paths]
    held = set()
    for pid in os.listdir('/proc'):
        if not pid.isdigit() or int(pid) == os.getpid():
            continue
        found = []
        try:
            found.append(os.readlink(f'/proc/{pid}/cwd'))
            found += [os.readlink(f'/proc/{pid}/fd/{fd}') for fd in os.listdir(f'/proc/{pid}/fd')]
        except OSError:
            pass
        if any(f == r or f.startswith(r + '/') for f in found for r in roots):
            held.add(int(pid))
    return sorted(held)


def swap(link: Path, staging: Path, quiet_checks: int = 3, wait_seconds: float = 6 * 3600, pause: float = 10):
    target = Path(os.readlink(link))
    deadline, quiet = time.time() + wait_seconds, 0
    while quiet < quiet_checks:   # no holder in several checks, so a short-lived reader is not raced
        held = holders([link, target.resolve()])
        quiet = 0 if held else quiet + 1
        if held:
            log(event='waiting for readers', pids=held[:10])
            if time.time() > deadline:
                raise SystemExit('readers still hold files under the tree')
        if quiet < quiet_checks:
            time.sleep(pause)
    aside = link.with_name(link.name + '.recall-old')
    os.rename(link, aside)
    try:
        os.rename(staging, link)
    except OSError:
        os.rename(aside, link)
        raise
    os.unlink(aside)
    (link / '.keep-on-nvme').write_text('recalled from the HDD archive: hot input (scripts/recall_to_nvme.py)\n')


def update_registry(link: Path, target: Path) -> list[str]:
    sys.path[:0] = [str(REPO / 'training' / 'neuralese'), str(REPO)]
    from natlang_neuralese.common import data_registry as reg
    archived = reg.portable(target)
    latest = {}
    for line in reg.journal_path().open():
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get('op', 'dataset') == 'dataset' and record.get('id'):
            latest[record['id']] = record
    moved = []
    for ident, record in latest.items():
        path = str(record.get('path', ''))
        if path != archived and not path.startswith(archived + '/'):
            continue
        sub = path[len(archived):].lstrip('/')
        new = {k: v for k, v in record.items() if k not in ('schema', 'time', 'machine')}
        new.update(path=reg.portable(link / sub) if sub else reg.portable(link), recalled_from=path,
                   note=(record.get('note') or '') + f' Recalled to the NVMe {time.strftime("%Y-%m-%d")} '
                        '(scripts/recall_to_nvme.py: hot files copied and SHA-256 verified, cold paths symlinked '
                        'to the HDD copy).')
        reg._append(new)
        moved.append(ident)
    return moved


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('link', type=Path, help='the NVMe path, now a symlink to the archived directory')
    parser.add_argument('--cold', action='append', default=[],
                        help='relative path glob left on the HDD as a symlink (repeatable; fnmatch, * crosses /)')
    parser.add_argument('--min-free-gb', type=float, default=20.0, help='NVMe free space to keep after the copy')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    link = args.link.absolute()
    if not link.is_symlink() or not link.resolve().is_dir():
        raise SystemExit(f'{link} is not a symlink to an archived directory')
    target = Path(os.readlink(link))
    target = target if target.is_absolute() else (link.parent / target)
    dirs, hot, cold = plan(target, args.cold)
    hot_bytes = sum((target / rel).stat().st_size for rel in hot)
    free = shutil.disk_usage(link.parent).free
    log(event='plan', link=str(link), target=str(target), dirs=len(dirs), hot_files=len(hot),
        hot_gb=round(hot_bytes / GIB, 2), cold_paths=len(cold), nvme_free_gb=round(free / GIB, 1))
    if free - hot_bytes < args.min_free_gb * GIB:
        raise SystemExit(f'not enough NVMe space: {free / GIB:.1f} GB free, {hot_bytes / GIB:.1f} GB to copy, '
                         f'{args.min_free_gb} GB to keep')
    if not args.apply:
        return 0
    staging = link.with_name(link.name + '.recall-staging')
    digests = build(target, staging, dirs, hot, cold)
    bad = verify(target, staging, hot, cold, digests)
    log(event='verify', bad=len(bad), sample=bad[:5])
    if bad:
        raise SystemExit('staging copy does not verify; the symlink is unchanged')
    swap(link, staging)
    log(event='swapped', path=str(link))
    log(event='registry', recalled=update_registry(link, target))
    return 0


if __name__ == '__main__':
    sys.exit(main())
