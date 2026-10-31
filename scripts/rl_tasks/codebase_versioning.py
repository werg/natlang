"""Create an immutable, independently owned snapshot of a Natlang codebase version."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
from typing import Iterable

IGNORED_DIRS = {'.git', 'node_modules', '.venv', '.natlang', 'build', 'dist', 'cache', 'caches',
                '__pycache__', 'journals'}


def version_codebase(source: Path, destination: Path, include: Iterable[str] = (
        'metadata.json', 'package', 'worlds', 'source'), *,
        exclude_paths: Iterable[str] = ()) -> Path:
    include = tuple(include)
    exclude_paths = tuple(Path(value).as_posix().strip('/') for value in exclude_paths)
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
            if src.is_dir():
                for directory, dirs, files in os.walk(src, topdown=True, followlinks=False):
                    current = Path(directory)
                    for name in tuple(dirs):
                        item = current / name
                        if _ignored(source, item, exclude_paths):
                            dirs.remove(name)
                        elif item.is_symlink():
                            raise ValueError(f'version input tree contains a symlink: {item}')
                    for name in files:
                        item = current / name
                        if not _ignored(source, item, exclude_paths) and item.is_symlink():
                            raise ValueError(f'version input tree contains a symlink: {item}')
            out = stage / rel
            if src.is_dir():
                shutil.copytree(src, out, symlinks=False, copy_function=shutil.copy2,
                    ignore=_ignore_callback(source, exclude_paths))
            elif src.is_file():
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, out)
            else:
                raise ValueError(f'version input is not a regular file or directory: {relative}')
        _assert_independent(source, stage, include, exclude_paths)
        os.replace(stage, destination)
        return destination
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise


def _ignored(source: Path, path: Path, exclude_paths: tuple[str, ...]) -> bool:
    relative = path.relative_to(source).as_posix()
    name = path.name
    if name in IGNORED_DIRS or name.endswith('.pyc') or name.startswith('.venv-'):
        return True
    for excluded in exclude_paths:
        if relative == excluded or relative.startswith(excluded + '/'):
            return True
    return False


def _ignore_callback(source: Path, exclude_paths: tuple[str, ...]):
    def ignored(directory: str, names: list[str]) -> set[str]:
        parent = Path(directory)
        return {name for name in names if _ignored(source, parent / name, exclude_paths)}
    return ignored


def _assert_independent(source: Path, destination: Path, include: Iterable[str],
                        exclude_paths: tuple[str, ...]) -> None:
    for relative in include:
        src, dst = source / relative, destination / relative
        if src.is_file():
            pairs = [(src, dst)]
        else:
            pairs = []
            for directory, dirs, files in os.walk(src, topdown=True, followlinks=False):
                parent = Path(directory)
                dirs[:] = [name for name in dirs if not _ignored(source, parent / name, exclude_paths)]
                for name in files:
                    path = parent / name
                    if not _ignored(source, path, exclude_paths):
                        pairs.append((path, dst / path.relative_to(src)))
        for original, copy in pairs:
            if not copy.is_file() or original.stat().st_ino == copy.stat().st_ino and original.stat().st_dev == copy.stat().st_dev:
                raise RuntimeError(f'version snapshot shares or lacks regular-file bytes: {original}')
