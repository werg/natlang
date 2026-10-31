"""Create an immutable, independently owned snapshot of a Natlang codebase version."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable


def version_codebase(source: Path, destination: Path, include: Iterable[str] = (
        'metadata.json', 'package', 'worlds', 'source')) -> Path:
    include = tuple(include)
    source = source.resolve(strict=True)
    destination = destination.absolute()
    if destination.exists():
        raise FileExistsError(f'version destination already exists: {destination}')
    if destination.parent.resolve(strict=True) != source.parent:
        raise ValueError('version destination must be a sibling of the source codebase')
    stage = Path(tempfile.mkdtemp(prefix=f'.{destination.name}.stage-', dir=destination.parent))
    try:
        for relative in include:
            rel = Path(relative)
            if rel.is_absolute() or not rel.parts or any(part in {'', '.', '..'} for part in rel.parts):
                raise ValueError(f'unsafe version path: {relative!r}')
            src = source / rel
            if src.is_symlink() or not src.exists():
                raise ValueError(f'version input must exist and cannot be a symlink: {relative}')
            if src.is_dir() and any(path.is_symlink() for path in src.rglob('*')):
                raise ValueError(f'version input tree contains a symlink: {relative}')
            out = stage / rel
            if src.is_dir():
                shutil.copytree(src, out, symlinks=False, copy_function=shutil.copy2,
                    ignore=shutil.ignore_patterns('journals', '__pycache__', '*.pyc'))
            elif src.is_file():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out)
            else:
                raise ValueError(f'version input is not a regular file or directory: {relative}')
        _assert_independent(source, stage, include)
        os.replace(stage, destination)
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _assert_independent(source: Path, destination: Path, include: Iterable[str]) -> None:
    ignored_dirs = {'journals', '__pycache__'}
    for relative in include:
        src, dst = source / relative, destination / relative
        if src.is_file():
            pairs = [(src, dst)]
        else:
            pairs = []
            for path in src.rglob('*'):
                rel = path.relative_to(src)
                if any(part in ignored_dirs for part in rel.parts) or path.name.endswith('.pyc'):
                    continue
                if path.is_file():
                    pairs.append((path, dst / rel))
        for original, copy in pairs:
            if not copy.is_file() or original.stat().st_ino == copy.stat().st_ino and original.stat().st_dev == copy.stat().st_dev:
                raise RuntimeError(f'version snapshot shares or lacks regular-file bytes: {original}')
