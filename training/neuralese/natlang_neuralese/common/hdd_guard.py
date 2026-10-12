"""Keep training-step I/O off the slow external HDD (DGX: /mnt/external).

Inputs that resolve (realpath) under the HDD prefix are copied once to an NVMe cache keyed by SHA-256 and the staged
path is passed to the stage; the declared hash is checked on the copy. The out dir (checkpoints, logs) must not
resolve under the HDD. A no-op where the prefix does not exist (Pop). The prefix comes from NATLANG_HDD_PREFIX
(default /mnt/external); the cache from NATLANG_STAGED_INPUTS (default ~/data/staged-inputs).
"""
from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path

RECORDS: list[dict] = []


def hdd_prefix() -> Path | None:
    prefix = Path(os.environ.get('NATLANG_HDD_PREFIX', '/mnt/external'))
    return prefix.resolve() if prefix.exists() else None


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


def stage_input(path, sha256: str | None = None) -> str:
    """The path to read: ``path`` itself if it is on the NVMe, else its verified NVMe copy."""
    if not on_hdd(path):
        return str(path)
    source = Path(os.path.realpath(path))
    digest = sha256 or _sha(source)
    target = cache_dir() / digest / source.name
    if not (target.is_file() and target.stat().st_size == source.stat().st_size):
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
