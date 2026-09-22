"""Read-only host file trees exposed lazily through the ``files/...`` namespace."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Union

FileValue = Union[str, bytes, bytearray, memoryview]


class ReadonlyFileTree(Protocol):
    def read(self, path: str, start: int | None = None, end: int | None = None) -> dict: ...


def _clean(path: str) -> str:
    normalized = path.replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    if normalized.startswith("/") or any(part in (".", "..") or "\0" in part for part in parts):
        raise ValueError(f"file path escapes its root: {path}")
    return "/".join(parts)


def _range(start: int | None, end: int | None, line_count: int) -> tuple[int, int]:
    first = 1 if start is None else start
    last = (min(200, line_count) if start is None else first) if end is None else end
    if (isinstance(first, bool) or isinstance(last, bool) or not isinstance(first, int) or
            not isinstance(last, int) or first < 1 or last < first):
        raise ValueError("file line range must satisfy 1 <= start <= end")
    return first, last


class MemoryFileTree:
    """Portable provider for browser-like and already-memory-backed embeddings."""

    def __init__(self, files: Mapping[str, FileValue]):
        self.files = {_clean(path): value for path, value in files.items()}

    def read(self, path: str, start: int | None = None, end: int | None = None) -> dict:
        key = _clean(path)
        if key in self.files:
            value = self.files[key]
            if not isinstance(value, str):
                return {"kind": "binary", "path": key, "bytes": len(bytes(value))}
            lines = value.replace("\r\n", "\n").split("\n")
            first, last = _range(start, end, len(lines))
            return {"kind": "text", "path": key, "text": "\n".join(lines[first - 1:last]),
                    "start": first, "end": min(last, len(lines)), "truncated": last < len(lines),
                    "bytes": len(value.encode("utf-8"))}
        prefix = key + "/" if key else ""
        children = {}
        for name, value in self.files.items():
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            child = rest.split("/", 1)[0]
            if not child:
                continue
            nested = "/" in rest
            entry = {"name": child, "kind": "directory" if nested else "file"}
            if not nested:
                entry["bytes"] = len(value.encode("utf-8")) if isinstance(value, str) else len(bytes(value))
            children[child] = entry
        if not children and key:
            raise ValueError(f"no such file or directory: {key}")
        return {"kind": "directory", "path": key,
                "entries": [children[name] for name in sorted(children)]}


class FilesystemFileTree:
    """Filesystem provider which reads directories and bytes only when requested."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve(strict=True)

    def _target(self, path: str) -> tuple[str, Path]:
        key = _clean(path)
        try:
            target = (self.root / key).resolve(strict=True)
        except (FileNotFoundError, OSError):
            raise ValueError(f"no such file or directory: {key}") from None
        try:
            target.relative_to(self.root)
        except ValueError:
            raise ValueError(f"file path escapes its root: {path}") from None
        return key, target

    def read(self, path: str, start: int | None = None, end: int | None = None) -> dict:
        key, target = self._target(path)
        if target.is_dir():
            entries = []
            for child in sorted(target.iterdir(), key=lambda item: item.name):
                if child.is_symlink():
                    entries.append({"name": child.name, "kind": "symlink"})
                elif child.is_dir():
                    entries.append({"name": child.name, "kind": "directory"})
                elif child.is_file():
                    entries.append({"name": child.name, "kind": "file", "bytes": child.stat().st_size})
            return {"kind": "directory", "path": key, "entries": entries}
        if not target.is_file():
            raise ValueError(f"not a regular file: {key}")
        size = target.stat().st_size
        with target.open("rb") as handle:
            probe = handle.read(65536)
            if b"\0" in probe:
                return {"kind": "binary", "path": key, "bytes": size}
        first = 1 if start is None else start
        last = (first + 199 if start is None else first) if end is None else end
        _range(first, last, last)
        selected, seen, truncated = [], 0, False
        with target.open("r", encoding="utf-8", errors="replace", newline=None) as handle:
            for seen, line in enumerate(handle, 1):
                if seen > last:
                    truncated = True
                    break
                if seen >= first:
                    selected.append(line.rstrip("\n"))
            # Match the portable memory provider: a final newline creates a
            # final empty line, without loading the file body into memory.
            if not truncated and size:
                with target.open("rb") as tail:
                    tail.seek(-1, 2)
                    ends_in_newline = tail.read(1) == b"\n"
                if ends_in_newline:
                    seen += 1
                    if first <= seen <= last:
                        selected.append("")
                    elif seen > last:
                        truncated = True
        if size == 0:
            seen = 1
            if first == 1:
                selected.append("")
        return {"kind": "text", "path": key, "text": "\n".join(selected),
                "start": first, "end": min(last, seen), "truncated": truncated, "bytes": size}


def format_file_tree_read(result: dict) -> str:
    if result["kind"] == "directory":
        if not result["entries"]:
            return "(empty directory)"
        lines = []
        for entry in result["entries"]:
            label = "dir " if entry["kind"] == "directory" else "link" if entry["kind"] == "symlink" else "file"
            size = "" if "bytes" not in entry else f"  {entry['bytes']} bytes"
            lines.append(f"{label}  {entry['name']}{size}")
        return "\n".join(lines)
    if result["kind"] == "binary":
        return f"{result['path']}: binary file, {result['bytes']} bytes; content requires a host binary capability"
    suffix = (f"\n… more content; read another line range from files/{result['path']}"
              if result["truncated"] else "")
    return result["text"] + suffix
