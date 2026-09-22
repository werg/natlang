"""Filesystem adapters for host-backed lazy ``Dict<File>`` inputs."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

from .host_tree import LazyDict, TreeEntry, lazy_dict

FILE_TREE_LEAF_TYPE = '{ kind: "text", text: Text, bytes: Num } | { kind: "binary", bytes: Num }'
FILE_WRITE_TYPE = '{ path: Text, text: Text }'


class _FilesystemProvider:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)

    def _target(self, path: tuple[str, ...]) -> Path:
        if any(not part or part in (".", "..") or "\0" in part for part in path):
            raise ValueError(f"file path escapes its root: {'/'.join(path)}")
        try:
            target = self.root.joinpath(*path).resolve(strict=True)
            target.relative_to(self.root)
        except FileNotFoundError:
            raise ValueError(f"no such file or directory: {'/'.join(path)}") from None
        except ValueError:
            raise ValueError(f"file path escapes its root: {'/'.join(path)}") from None
        return target

    def list(self, path: tuple[str, ...]):
        directory = self._target(path)
        if not directory.is_dir():
            raise ValueError(f"not a directory: {'/'.join(path)}")
        entries = []
        for child in directory.iterdir():
            try:
                kind = "branch" if self._target(path + (child.name,)).is_dir() else "leaf"
            except ValueError:
                kind = "leaf"
            entries.append(TreeEntry(child.name, kind))
        return entries

    def read(self, path: tuple[str, ...]):
        file = self._target(path)
        if not file.is_file():
            raise ValueError(f"not a regular file: {'/'.join(path)}")
        content = file.read_bytes()
        if b"\0" in content:
            return {"kind": "binary", "bytes": len(content)}
        return {"kind": "text", "text": content.decode("utf-8", errors="replace"), "bytes": len(content)}


class FilesystemFileTree(LazyDict):
    """A filesystem-backed lazy ``Dict<File>`` for an ordinary natlang input."""

    def __init__(self, root: str | Path, *, label: str = "project files"):
        super().__init__(_FilesystemProvider(root), label=label)


def text_file_tree(files: dict[str, str], *, label: str = "project files") -> LazyDict:
    """Turn an in-memory text project into a lazy ``Dict<File>``."""
    return lazy_dict({path.lstrip("/\\"): {"kind": "text", "text": text,
                                             "bytes": len(text.encode("utf-8"))}
                      for path, text in files.items()}, label=label)


def validate_file_writes(writes: Iterable[Any]) -> list[dict[str, str]]:
    """Validate and normalize a portable ``{path, text}[]`` change plan."""
    if isinstance(writes, (str, bytes, dict)):
        raise TypeError("file writes must be a list of {path, text} records")
    checked, seen = [], set()
    for index, item in enumerate(writes):
        if not isinstance(item, dict) or set(item) != {"path", "text"}:
            raise TypeError(f"file write {index} must contain exactly path and text")
        raw, text = item["path"], item["text"]
        if not isinstance(raw, str) or not isinstance(text, str):
            raise TypeError(f"file write {index} path and text must be strings")
        if not raw or raw.startswith(("/", "\\")) or "\\" in raw or "\0" in raw:
            raise ValueError(f"invalid relative file path: {raw!r}")
        parts = raw.split("/")
        if any(not part or part in (".", "..") for part in parts):
            raise ValueError(f"invalid relative file path: {raw!r}")
        path = "/".join(parts)
        if path in seen:
            raise ValueError(f"duplicate file write path: {path}")
        seen.add(path); checked.append({"path": path, "text": text})
    return checked


def commit_file_writes(root: str | Path, writes: Iterable[Any], *, overwrite: bool = True) -> list[dict]:
    """Commit a validated change plan beneath root, using same-directory atomic replacement."""
    base = Path(root).resolve(strict=True)
    if not base.is_dir():
        raise ValueError(f"file-write root is not a directory: {base}")
    planned = []
    for item in validate_file_writes(writes):
        target = base.joinpath(*item["path"].split("/")).resolve(strict=False)
        try:
            target.relative_to(base)
        except ValueError:
            raise ValueError(f"file write escapes its root: {item['path']}") from None
        if target.exists() and not target.is_file():
            raise ValueError(f"file write target is not a regular file: {item['path']}")
        if target.exists() and not overwrite:
            raise FileExistsError(item["path"])
        planned.append((item, target, target.exists()))

    receipts = []
    for item, target, overwritten in planned:
        target.parent.mkdir(parents=True, exist_ok=True)
        resolved_parent = target.parent.resolve(strict=True)
        try:
            resolved_parent.relative_to(base)
        except ValueError:
            raise ValueError(f"file write escapes its root: {item['path']}") from None
        encoded = item["text"].encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(prefix=".natlang-write-", dir=resolved_parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(encoded)
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
        receipts.append({"path": item["path"], "bytes": len(encoded), "overwritten": overwritten})
    return receipts
