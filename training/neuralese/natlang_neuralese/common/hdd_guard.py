"""Keep training-step I/O off the slow external HDD (DGX).

Data inputs (``STAGED_ROLES``: the corpora a stage reads, again at every resume) that resolve (realpath) under the
HDD prefix are copied once to an NVMe cache keyed by SHA-256 and the staged path is passed to the stage; the declared
hash is checked on the copy. Load-once state (heads, ``continue_from`` and its sidecars, student/warm-up checkpoints:
read once at startup, so a copy buys no step speed, only disk churn; the 44 GB v3 warm-up state was copied back from
the archive) is read directly from the HDD and recorded as ``read_from_hdd``. The cache is bounded
(NATLANG_STAGED_INPUTS_MAX_GB, default 64): least recently used entries are evicted before a new copy, and the
archiver or storage retention may delete any entry (a stage re-stages what it needs). The out dir (checkpoints, logs)
must not resolve under the HDD. A no-op where the prefix is absent or its device is known to be
nonrotating (Pop's `/srv/storage` is an NVMe partition). The prefix comes from
NATLANG_HDD_PREFIX (default: the data_hdd root of common.paths); the cache from NATLANG_STAGED_INPUTS (default
~/data/staged-inputs).
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

from .paths import resolve

RECORDS: list[dict] = []
# Native graph/rollout corpora are streamed during a run and on resume, so stage
# them once to local storage when their source lives on the external HDD. Other
# roles are small configuration, proofs, or load-once state and stay in place.
STAGED_ROLES = frozenset({'native_graphs', 'held_native_graphs', 'continued_native_graphs',
                          'continued_held_native_graphs', 'foundation_native_graphs', 'rollouts',
                          'rl_block_bundle', 'pending_native_sources'})


def hdd_prefix() -> Path | None:
    explicit = os.environ.get('NATLANG_HDD_PREFIX')
    prefix = Path(explicit or resolve('data_hdd'))
    if not prefix.exists():
        return None
    prefix = prefix.resolve()
    # A machine may use its logical data_hdd root for a second NVMe partition.
    # Keep an explicit prefix authoritative, but avoid staging from that NVMe
    # back onto itself when the logical root is only a storage location name.
    if not explicit and _rotational(prefix) is False:
        return None
    return prefix


def _rotational(path: Path) -> bool | None:
    """Read the mounted block device's rotation flag; unknown keeps old behavior."""
    try:
        device = path.stat().st_dev
        sys_device = Path(f'/sys/dev/block/{os.major(device)}:{os.minor(device)}').resolve()
        for candidate in (sys_device, *sys_device.parents):
            flag = candidate / 'queue' / 'rotational'
            if flag.is_file():
                return flag.read_text().strip() == '1'
    except OSError:
        pass
    return None


def on_hdd(path) -> bool:
    prefix = hdd_prefix()
    if prefix is None:
        return False
    real = Path(os.path.realpath(path))
    return real == prefix or prefix in real.parents


def require_nvme(path, what='output directory') -> None:
    if on_hdd(path):
        raise ValueError(f'the {what} {path} resolves to the external HDD; use a directory on the NVMe')


def cache_dir() -> Path:
    return Path(os.environ.get('NATLANG_STAGED_INPUTS', Path.home() / 'data' / 'staged-inputs'))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def input_path(role: str, path, sha256: str | None = None) -> str:
    """The path a stage reads for input ``role``: data roles staged (``stage_input``), load-once state in place."""
    if role in STAGED_ROLES:
        return stage_input(path, sha256)
    if on_hdd(path):
        RECORDS.append({'role': role, 'source': os.path.realpath(path), 'read_from_hdd': True,
                        'reason': 'load-once state: read once at startup, staging buys no step speed'})
    return str(path)


def _evict(cache: Path, incoming: int) -> None:
    """Least recently used cache entries out until ``incoming`` more bytes fit the bound."""
    bound = int(float(os.environ.get('NATLANG_STAGED_INPUTS_MAX_GB', '64')) * 2 ** 30)
    if not cache.is_dir():
        return
    entries = [(f.stat().st_mtime, f.stat().st_size, f) for f in cache.glob('*/*') if f.is_file()]
    used = sum(size for _, size, _ in entries)
    for _, size, path in sorted(entries):
        if used + incoming <= bound:
            break
        path.unlink(missing_ok=True)
        try:
            path.parent.rmdir()
        except OSError:
            pass
        used -= size


def stage_input(path, sha256: str | None = None) -> str:
    """The path to read: ``path`` itself if it is on the NVMe, else its verified NVMe copy."""
    if not on_hdd(path):
        return str(path)
    source = Path(os.path.realpath(path))
    digest = sha256 or _sha(source)
    target = cache_dir() / digest / source.name
    if target.is_file() and target.stat().st_size == source.stat().st_size:
        os.utime(target)  # recently used
    else:
        _evict(cache_dir(), source.stat().st_size)
        target.parent.mkdir(parents=True, exist_ok=True)
        part = target.with_name(target.name + f'.part{os.getpid()}')
        shutil.copyfile(source, part)
        if _sha(part) != digest:
            part.unlink()
            raise ValueError(f'staged copy of {source} does not match its declared SHA-256')
        os.replace(part, target)
    RECORDS.append({'source': str(source), 'staged': str(target), 'sha256': digest,
                    'declared_hash': sha256 is not None})
    return str(target)
