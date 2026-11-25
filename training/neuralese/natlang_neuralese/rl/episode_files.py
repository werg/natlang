"""An episode's job files once nothing reads them raw any more (DECISIONS 2026-11-25 "episode files compacted").

The ts-host collector (``ts-host/src/teacher/collector.ts``) writes, per job of an episode, ``jobs/<key>.partial.json``
(every turn with its raw response, RL capture included) and ``jobs/<key>.partial.json.evidence-<attempt>.jsonl`` (the
provider exchanges and action frames, each again with the raw response), plus ``jobs/<key>.result.json`` (a byte copy
of the sample's ``results.jsonl``). It removes the partial and its evidence when the job ends normally; a job that ends
otherwise (a call-budget stop: ``task_rollouts._budget_terminated_prefix`` reads both while scoring, a signal: the
collection stopped under it) leaves them, 10-60 MB each, and the result copy stays always. aei1's collections grew ~10
GB/hour on the DGX that way (2026-11-25).

``compact_sample`` runs once a sample's episode process has exited and its group is scored for the last time or will
never be scored (abandoned): it keeps every byte that was recorded, zstd-compressed (~30x: the raw responses repeat the
prompt token ids of every call), and drops only exact copies and dead writers' temp files:

1. ``jobs/*.result.json`` that equals ``results.jsonl`` byte for byte: removed (the copy);
2. ``jobs/*.tmp-<pid>-<uuid>`` whose writer process is gone: removed (an atomic write that never completed);
3. ``jobs/*.partial.json`` and ``jobs/*.partial.json.evidence-*.jsonl`` of at least MIN_BYTES: replaced by
   ``<name>.zst`` (content size and checksum in the frame; the bytes are verified before the original goes, its mtime
   is kept).

``read_bytes`` reads such a file raw or compacted; ``evidence_files`` lists a jobs directory's evidence either way.
``NATLANG_EPISODE_COMPACT=0`` opts out (nothing is touched).

    python -m natlang_neuralese.rl.episode_files compact DIR [--apply] [--quiet-minutes 30]

compacts every sample under DIR (a collection, a group, a teacher campaign, a held measurement) whose files are all
quiet that long and that no process holds open; without ``--apply`` it only reports what it would do.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ENV = 'NATLANG_EPISODE_COMPACT'
SUFFIX = '.zst'
RAW = re.compile(r'\.partial\.json(\.evidence-[0-9a-f-]{36}\.jsonl)?$')
TMP = re.compile(r'\.tmp-(\d+)-[0-9a-f-]{36}$')
LEVEL = 3
MIN_BYTES = 4096    # smaller files stay raw: compression would save nothing


def enabled():
    return os.environ.get(ENV, '1') != '0'


def _pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError, ValueError):
        return True     # in doubt the writer lives: its file stays


def _compress(data):
    try:
        import zstandard
        return zstandard.ZstdCompressor(level=LEVEL, write_checksum=True, write_content_size=True).compress(data)
    except ImportError:
        return subprocess.run(['zstd', '-q', f'-{LEVEL}', '-c'], input=data, capture_output=True, check=True).stdout


def _decompress(data):
    try:
        import zstandard
        return zstandard.ZstdDecompressor().decompress(data, max_output_size=1 << 34)
    except ImportError:
        return subprocess.run(['zstd', '-q', '-d', '-c'], input=data, capture_output=True, check=True).stdout


def read_bytes(path):
    """The bytes of ``path``, or of its compacted ``<path>.zst`` (``path`` may name either)."""
    path = Path(path)
    if path.name.endswith(SUFFIX):
        return _decompress(path.read_bytes())
    try:
        return path.read_bytes()
    except FileNotFoundError:
        packed = path.with_name(path.name + SUFFIX)
        if packed.is_file():
            return _decompress(packed.read_bytes())
        raise


def evidence_files(jobs):
    """A jobs directory's evidence sidecars, raw or compacted."""
    jobs = Path(jobs)
    return sorted([*jobs.glob('*.evidence-*.jsonl'), *jobs.glob('*.evidence-*.jsonl' + SUFFIX)])


def _same_bytes(a, b):
    if a.stat().st_size != b.stat().st_size:
        return False
    with open(a, 'rb') as x, open(b, 'rb') as y:
        while True:
            u, v = x.read(8 << 20), y.read(8 << 20)
            if u != v:
                return False
            if not u:
                return True


def compress_file(path):
    """``path`` replaced by ``<path>.zst`` holding the same bytes (verified); (raw bytes, compressed bytes)."""
    data = path.read_bytes()
    stat = path.stat()
    packed = path.with_name(path.name + SUFFIX)
    if packed.exists():     # an earlier pass stopped between its rename and the unlink
        if _decompress(packed.read_bytes()) != data:
            raise RuntimeError(f'{packed} exists with other content')
        path.unlink()
        return len(data), packed.stat().st_size
    blob = _compress(data)
    if _decompress(blob) != data:
        raise RuntimeError(f'{path}: compression did not round-trip')
    tmp = packed.with_name(f'{packed.name}.tmp-{os.getpid()}')
    tmp.write_bytes(blob)
    os.utime(tmp, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    os.replace(tmp, packed)
    path.unlink()
    return len(data), len(blob)


def compact_sample(sample):
    """Compact one finished sample directory (``row-*/sample-*``; see the module doc). Returns a Counter of
    files and bytes per action. Never raises for one file: a file that fails stays as it was."""
    stats = collections.Counter()
    jobs = Path(sample) / 'jobs'
    if not enabled() or not jobs.is_dir() or jobs.is_symlink():
        return stats
    results = Path(sample) / 'results.jsonl'
    for path in sorted(jobs.iterdir()):
        name = path.name
        try:
            if path.is_symlink() or not path.is_file():
                continue
            if name.endswith('.result.json'):
                if results.is_file() and not results.is_symlink() and _same_bytes(path, results):
                    stats['duplicate_result_bytes'] += path.stat().st_size
                    path.unlink()
                    stats['duplicate_result_files'] += 1
            elif TMP.search(name):
                if not _pid_alive(TMP.search(name).group(1)):
                    stats['tmp_bytes'] += path.stat().st_size
                    path.unlink()
                    stats['tmp_files'] += 1
            elif RAW.search(name) and path.stat().st_size >= MIN_BYTES:
                raw, packed = compress_file(path)
                stats['compressed_files'] += 1
                stats['compressed_raw_bytes'] += raw
                stats['compressed_bytes'] += packed
        except Exception as error:      # one odd file never loses the rest
            stats['errors'] += 1
            print(json.dumps({'event': 'episode compaction error', 'path': str(path), 'error': repr(error)[:300]}),
                  file=sys.stderr, flush=True)
    return stats


def samples_of(directory):
    """The sample directories under ``directory`` (itself one, a group, a collection, a campaign, a measurement)."""
    directory = Path(directory)
    if (directory / 'jobs').is_dir():
        return [directory]
    return sorted(p.parent for p in directory.glob('**/jobs') if p.is_dir() and not p.is_symlink()
                  and p.parent.name.startswith('sample-'))


def compact_group(group):
    """``compact_sample`` over every sample of ``group`` (lost attempts set aside included)."""
    total = collections.Counter()
    for sample in samples_of(group):
        total.update(compact_sample(sample))
    return total


def freed(stats):
    return (stats.get('duplicate_result_bytes', 0) + stats.get('tmp_bytes', 0)
            + stats.get('compressed_raw_bytes', 0) - stats.get('compressed_bytes', 0))


def open_paths(root):
    """Every path under ``root`` some process holds open (``/proc/*/fd``; unreadable processes are skipped)."""
    root = str(Path(root).resolve())
    held = set()
    for fd_dir in Path('/proc').glob('[0-9]*/fd'):
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith(root):
                    held.add(target)
        except OSError:
            continue
    return held


def quiet_sample(sample, now, quiet_seconds, held):
    """No file of the sample changed in ``quiet_seconds`` and none is open (its episode is over)."""
    for path in [sample, *sample.rglob('*')]:
        try:
            if now - path.lstat().st_mtime < quiet_seconds:
                return False
        except FileNotFoundError:
            continue
        if str(path.resolve()) in held:
            return False
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('compact', help='compact the quiet, unheld samples under each DIR')
    c.add_argument('dirs', nargs='+')
    c.add_argument('--apply', action='store_true', help='compact (default: report only)')
    c.add_argument('--quiet-minutes', type=float, default=30.)
    c.add_argument('--skip-group', action='append', default=[], metavar='NAME',
                   help='a group directory name (gNNNNNN) left untouched (a live collector\'s in-flight group)')
    args = p.parse_args(argv)
    total, now = collections.Counter(), time.time()
    for directory in map(Path, args.dirs):
        held = open_paths(directory)
        kept = 0
        for sample in samples_of(directory):
            if any(part in args.skip_group for part in sample.parts) or not quiet_sample(
                    sample, now, args.quiet_minutes * 60, held):
                kept += 1
                continue
            if args.apply:
                stats = compact_sample(sample)
            else:
                stats = collections.Counter()
                for path in (sample / 'jobs').iterdir():
                    if RAW.search(path.name):
                        stats['compressible_raw_bytes'] += path.stat().st_size
                        stats['compressible_files'] += 1
            total.update(stats)
        total['samples_kept_live'] += kept
        print(json.dumps({'event': 'compacted' if args.apply else 'would compact', 'dir': str(directory),
                          'kept_live': kept}), flush=True)
    print(json.dumps({'event': 'total', 'applied': args.apply, 'freed_gb': round(freed(total) / 1e9, 2),
                      **dict(total)}), flush=True)


if __name__ == '__main__':
    main()
