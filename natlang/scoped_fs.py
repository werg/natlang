"""Small copy-on-write folder handles for scoped lambda filesystems.

This module intentionally has no runtime or tool-surface dependencies.  A
``Folder`` is an immutable captured file view plus a private overlay.  Forks
can be edited independently and an atomic selected delta can be installed into
their parent while holding the parent's writer lock.
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping, Iterator


_TOMBSTONE = object()


def _path(path: str, *, allow_root: bool = True) -> str:
    if not isinstance(path, str):
        raise TypeError("path must be a string")
    if allow_root and path in ("", "."):
        return ""
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        raise ValueError(f"invalid relative POSIX path: {path!r}")
    parts = path.split("/")
    if any(not part or part in (".", "..") for part in parts):
        raise ValueError(f"invalid relative POSIX path: {path!r}")
    return "/".join(parts)


def _bytes(value: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    raise TypeError("file contents must be text or bytes")


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _under(path: str, parent: str) -> bool:
    return not parent or path == parent or path.startswith(parent + "/")


def _matches(path: str, pattern: str | None) -> bool:
    if pattern is None:
        return True
    # fnmatch's ** is permissive enough for a portable discovery pattern and
    # matching against the full relative POSIX name is deterministic.
    return fnmatch.fnmatchcase(path, pattern)


@dataclass(frozen=True)
class EntryStat:
    path: str
    kind: str
    bytes: int
    digest: str | None


@dataclass(frozen=True)
class SearchMatch:
    path: str
    line: int
    text: str


@dataclass(frozen=True)
class Change:
    path: str
    kind: str  # added, modified, deleted
    before: bytes | None
    after: bytes | None

    @property
    def before_digest(self) -> str | None:
        return None if self.before is None else _digest(self.before)

    @property
    def after_digest(self) -> str | None:
        return None if self.after is None else _digest(self.after)


@dataclass(frozen=True)
class ChangeSet:
    changes: tuple[Change, ...]
    moves: tuple[tuple[str, str], ...] = ()

    @property
    def empty(self) -> bool:
        return not self.changes

    def selected(self, include: Iterable[str] | None = None,
                 exclude: Iterable[str] | None = None) -> "ChangeSet":
        includes = tuple(_path_pattern(item) for item in (include or ()))
        excludes = tuple(_path_pattern(item) for item in (exclude or ()))
        def keep(change: Change) -> bool:
            return ((not includes or any(pattern(change.path) for pattern in includes)) and
                    not any(pattern(change.path) for pattern in excludes))
        changes = tuple(change for change in self.changes if keep(change))
        moves = tuple((source, target) for source, target in self.moves
                      if any(change.path in (source, target) for change in changes))
        return ChangeSet(changes, moves)


def _path_pattern(pattern: str) -> Callable[[str], bool]:
    if not isinstance(pattern, str) or not pattern:
        raise ValueError("file selectors must be nonempty relative POSIX patterns")
    _path(pattern.replace("**", "x"))
    return lambda path: fnmatch.fnmatchcase(path, pattern)


class FolderBusyError(RuntimeError):
    """Raised when a writer lock cannot be acquired immediately."""


class FolderConflictError(RuntimeError):
    """Raised when a fork was based on a file changed by its parent."""


class WriterLock:
    """A small per-root writer lock with blocking and nonblocking modes."""

    def __init__(self):
        self._lock = threading.Lock()

    def acquire(self, *, blocking: bool = True, timeout: float | None = None):
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        if not blocking:
            acquired = self._lock.acquire(False)
        elif timeout is None:
            acquired = self._lock.acquire(True)
        else:
            acquired = self._lock.acquire(True, timeout)
        if not acquired:
            raise FolderBusyError("folder writer is busy")
        return self

    def release(self) -> None:
        self._lock.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *_):
        self.release()


class _Entry:
    def __init__(self, folder: "Folder", path: str):
        self.folder, self.path = folder, _path(path)

    def __deepcopy__(self, _memo):
        return self

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1] if self.path else ""

    @property
    def relative_path(self) -> str:
        return self.path

    @property
    def parent(self) -> "FolderHandle | None":
        if not self.path:
            return None
        parent = self.path.rsplit("/", 1)[0] if "/" in self.path else ""
        return FolderHandle(self.folder, parent)

    def exists(self) -> bool:
        return self.folder.exists(self.path)

    def stat(self) -> EntryStat:
        return self.folder.stat(self.path)

    def remove(self) -> None:
        self.folder.remove(self.path)

    def move_to(self, destination: "FolderHandle | FileHandle | str") -> None:
        if isinstance(destination, FolderHandle):
            target = self.folder.join(destination.path, self.name)
        else:
            target = destination.path if isinstance(destination, _Entry) else _path(destination)
        self.folder.move(self.path, target)


class FileHandle(_Entry):
    def read_text(self, start_line: int | None = None, end_line: int | None = None) -> str:
        return self.folder.read_text(self.path, start_line, end_line)

    def read_bytes(self) -> bytes:
        return self.folder.read_bytes(self.path)

    def read_json(self):
        return json.loads(self.read_text())

    def write_text(self, content: str) -> None:
        self.folder.write_text(self.path, content)

    def write_bytes(self, content: bytes | bytearray | memoryview) -> None:
        self.folder.write_bytes(self.path, content)

    def write_json(self, value) -> None:
        self.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def edit_text(self, find: str, replace_with: str, *, fuzzy: bool = False):
        return self.folder.edit_text(self.path, find, replace_with, fuzzy=fuzzy)


class FolderHandle(_Entry):
    def dir(self, path: str) -> "FolderHandle":
        return FolderHandle(self.folder, self.folder.join(self.path, path))

    def file(self, path: str) -> FileHandle:
        return FileHandle(self.folder, self.folder.join(self.path, path))

    def entry(self, path: str) -> _Entry:
        joined = self.folder.join(self.path, path)
        return FileHandle(self.folder, joined) if self.folder.is_file(joined) else FolderHandle(self.folder, joined)

    def entries(self, pattern: str | None = None) -> list[_Entry]:
        return [self.folder.entry(item.path) for item in self.folder.list(self.path, pattern=pattern)]

    def files(self, pattern: str | None = None) -> list[FileHandle]:
        return [FileHandle(self.folder, item.path) for item in self.folder.list_files(self.path, pattern=pattern)]

    def folders(self, pattern: str | None = None) -> list["FolderHandle"]:
        return [FolderHandle(self.folder, item.path) for item in self.folder.list_folders(self.path, pattern=pattern)]

    def walk(self, pattern: str | None = None) -> Iterator[_Entry]:
        entries = [*self.folder.list_files(self.path, pattern=pattern),
                   *self.folder.list_folders(self.path, pattern=pattern)]
        for item in sorted(entries, key=lambda value: value.path):
            yield self.folder.entry(item.path)

    def diff(self) -> ChangeSet:
        return self.folder.diff(self.path)

    def begin_transaction(self, *, blocking: bool = True) -> "FolderTransaction":
        return self.folder.begin_transaction(blocking=blocking, path=self.path)


class FolderTransaction:
    """An exclusive reducer transaction over a private child folder.

    The parent writer is acquired before the child is exposed and remains held
    until ``commit`` or ``abort``.  Commit installs the selected delta through
    the already-held lock, so it never reacquires the semaphore.
    """

    def __init__(self, parent: "Folder", child: "Folder", prefix: str = ""):
        self.parent, self.folder, self.prefix = parent, child, prefix
        self._closed = False

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("folder transaction is closed")

    @property
    def open(self) -> bool:
        return not self._closed

    def commit(self, *, include: Iterable[str] | None = None,
               exclude: Iterable[str] | None = None) -> ChangeSet:
        self._ensure_open()
        try:
            selected = self.folder.diff().selected(include, exclude)
            def rooted(path: str) -> str:
                return self.parent.join(self.prefix, path) if self.prefix else path
            mapped = ChangeSet(tuple(Change(rooted(change.path), change.kind,
                                            change.before, change.after)
                                     for change in selected.changes),
                               tuple((rooted(source), rooted(target))
                                     for source, target in selected.moves))
            self.parent._install_locked(mapped)
            # A reducer observes and audits paths relative to its project root.
            result = selected
        except BaseException:
            self.abort()
            raise
        self._close()
        return result

    def abort(self) -> None:
        self._ensure_open()
        self._close()

    def _close(self) -> None:
        if not self._closed:
            self._closed = True
            self.parent.writer().release()

    def __enter__(self) -> "FolderTransaction":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, *_):
        if not self._closed:
            self.abort()


class Folder:
    """An immutable base snapshot plus a mutable copy-on-write overlay."""

    def __init__(self, files: Mapping[str, str | bytes] | None = None, *, access: str = "write",
                 _base: Mapping[str, bytes] | None = None, _overlay: Mapping[str, object] | None = None,
                 _writer: WriterLock | None = None, _moves: Iterable[tuple[str, str]] = ()):
        if access not in ("read", "write", "overlay"):
            raise ValueError("access must be read, write, or overlay")
        self.access = access
        source = _base if _base is not None else (files or {})
        self._base = { _path(name, allow_root=False): _bytes(value) for name, value in source.items() }
        self._overlay = dict(_overlay or {})
        self._writer = _writer or WriterLock()
        self._moves = list(_moves)

    def __deepcopy__(self, _memo):
        # Folder values are opaque capabilities. Function boundaries preserve
        # the handle; reducer invocation explicitly creates the isolated fork.
        return self

    @classmethod
    def from_files(cls, files: Mapping[str, str | bytes], *, access: str = "write") -> "Folder":
        return cls(files, access=access)

    @classmethod
    def from_directory(cls, root: str | Path, *, access: str = "write") -> "Folder":
        base = Path(root).resolve(strict=True)
        if not base.is_dir():
            raise ValueError(f"not a directory: {root}")
        files = {path.relative_to(base).as_posix(): path.read_bytes()
                 for path in base.rglob("*") if path.is_file()}
        return cls(files, access=access)

    def root(self) -> FolderHandle:
        return FolderHandle(self, "")

    def dir(self, path: str = "") -> FolderHandle:
        return FolderHandle(self, _path(path))

    def file(self, path: str) -> FileHandle:
        return FileHandle(self, _path(path, allow_root=False))

    def entry(self, path: str) -> _Entry:
        clean = _path(path)
        return FileHandle(self, clean) if self.is_file(clean) else FolderHandle(self, clean)

    def join(self, parent: str, child: str) -> str:
        child = _path(child)
        if not parent:
            return child
        if not child:
            return parent
        return _path(parent + "/" + child)

    def _effective(self) -> dict[str, bytes]:
        result = dict(self._base)
        for path, value in self._overlay.items():
            if value is _TOMBSTONE:
                result.pop(path, None)
            else:
                result[path] = value  # type: ignore[assignment]
        return result

    def _check_write(self) -> None:
        if self.access == "read":
            raise PermissionError("folder is read-only")

    def _read(self, path: str) -> bytes:
        clean = _path(path, allow_root=False)
        value = self._effective().get(clean)
        if value is None:
            raise FileNotFoundError(clean)
        return value

    def exists(self, path: str) -> bool:
        clean = _path(path)
        if clean in self._effective():
            return True
        return any(item.startswith(clean + "/") for item in self._effective()) if clean else bool(self._effective())

    def is_file(self, path: str) -> bool:
        return _path(path) in self._effective()

    def is_dir(self, path: str) -> bool:
        clean = _path(path)
        return not clean or any(item.startswith(clean + "/") for item in self._effective())

    def stat(self, path: str) -> EntryStat:
        clean = _path(path)
        if self.is_file(clean):
            value = self._effective()[clean]
            return EntryStat(clean, "file", len(value), _digest(value))
        if self.is_dir(clean):
            return EntryStat(clean, "folder", 0, None)
        raise FileNotFoundError(clean)

    def _paths(self, path: str = "") -> list[str]:
        clean = _path(path)
        return sorted(item for item in self._effective() if _under(item, clean))

    def list(self, path: str = "", *, pattern: str | None = None) -> list[EntryStat]:
        clean = _path(path)
        children: dict[str, str] = {}
        for item in self._paths(clean):
            rest = item[len(clean) + 1:] if clean else item
            first = rest.split("/", 1)[0]
            child = self.join(clean, first)
            kind = "folder" if "/" in rest else "file"
            children[child] = "folder" if children.get(child) == "folder" or kind == "folder" else "file"
        return [self.stat(item) for item in sorted(children)
                if _matches(item, pattern)]

    def list_files(self, path: str = "", *, pattern: str | None = None) -> list[EntryStat]:
        return [self.stat(item) for item in self._paths(path)
                if _matches(item, pattern) and self.is_file(item)]

    def list_folders(self, path: str = "", *, pattern: str | None = None) -> list[EntryStat]:
        clean = _path(path)
        candidates = set()
        for item in self._paths(clean):
            parts = item.split("/")
            for index in range(1, len(parts)):
                candidates.add("/".join(parts[:index]))
        return [self.stat(item) for item in sorted(candidates)
                if item != clean and _under(item, clean) and _matches(item, pattern)]

    def read_bytes(self, path: str) -> bytes:
        return self._read(path)

    def read_text(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        value = self._read(path).decode("utf-8")
        if start_line is None and end_line is None:
            return value
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = start_line
        if start_line < 1 or end_line < start_line:
            raise ValueError("line range must be one-based and ordered")
        lines = value.splitlines(keepends=True)
        return "".join(lines[start_line - 1:end_line])

    def read_json(self, path: str):
        return json.loads(self.read_text(path))

    def write_bytes(self, path: str, content: bytes | bytearray | memoryview) -> None:
        self._check_write()
        self._overlay[_path(path, allow_root=False)] = _bytes(content)

    def write_text(self, path: str, content: str) -> None:
        self.write_bytes(path, content)

    def write_json(self, path: str, value) -> None:
        self.write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")

    def remove(self, path: str) -> None:
        self._check_write()
        clean = _path(path, allow_root=False)
        if not self.exists(clean):
            raise FileNotFoundError(clean)
        for item in list(self._effective()):
            if item == clean or item.startswith(clean + "/"):
                self._overlay[item] = _TOMBSTONE

    def move(self, source: str, destination: str) -> None:
        self._check_write()
        source, destination = _path(source, allow_root=False), _path(destination, allow_root=False)
        if source == destination or destination.startswith(source + "/"):
            raise ValueError("cannot move an entry into itself")
        values = self._effective()
        selected = {item: value for item, value in values.items() if item == source or item.startswith(source + "/")}
        if not selected:
            raise FileNotFoundError(source)
        if self.exists(destination):
            raise FileExistsError(destination)
        for item, value in selected.items():
            suffix = item[len(source):].lstrip("/")
            target = destination + (("/" + suffix) if suffix else "")
            self._overlay[target] = value
            self._overlay[item] = _TOMBSTONE
        self._moves.append((source, destination))

    def edit_text(self, path: str, find: str, replace_with: str, *, fuzzy: bool = False):
        if not isinstance(find, str) or not isinstance(replace_with, str):
            raise TypeError("find and replace_with must be strings")
        original = self.read_text(path)
        count = original.count(find)
        if count != 1 and fuzzy:
            original, count, find = self._fuzzy_edit(original, find, replace_with)
            if count == 1:
                self.write_text(path, original)
                return {"path": _path(path), "changed": True, "digest": _digest(original.encode())}
        if count != 1:
            raise ValueError("edit requires exactly one matching span")
        updated = original.replace(find, replace_with, 1)
        self.write_text(path, updated)
        return {"path": _path(path), "changed": True, "digest": _digest(updated.encode())}

    @staticmethod
    def _fuzzy_edit(original: str, find: str, replacement: str):
        normalize = lambda text: re.sub(r"\s+", " ", text).strip()
        wanted = normalize(find)
        if not wanted:
            return original, 0, find
        matches = []
        for match in re.finditer(r"[^\n]*(?:\n|$)", original):
            line = match.group(0)
            if normalize(line) == wanted:
                matches.append((match.start(), match.end(), line))
        if len(matches) != 1:
            return original, len(matches), find
        start, end, line = matches[0]
        newline = "\n" if line.endswith("\n") else ""
        return original[:start] + replacement + newline + original[end:], 1, ""

    def search(self, query: str, path: str = "", *, pattern: str | None = None,
               regex: bool = False) -> list[SearchMatch]:
        if not isinstance(query, str) or not query:
            raise ValueError("search query must be nonempty")
        clean = _path(path)
        expression = re.compile(query) if regex else None
        results = []
        for item in self.list_files(clean, pattern=pattern):
            try:
                text = self.read_text(item.path)
            except UnicodeDecodeError:
                continue
            for line_no, line in enumerate(text.splitlines(), 1):
                if (expression.search(line) if expression else query in line):
                    results.append(SearchMatch(item.path, line_no, line))
        return results

    def diff(self, path: str = "") -> ChangeSet:
        clean = _path(path)
        current, before = self._effective(), self._base
        paths = sorted({item for item in set(current) | set(before) if _under(item, clean)})
        changes = []
        for item in paths:
            old, new = before.get(item), current.get(item)
            if old == new:
                continue
            kind = "added" if old is None else "deleted" if new is None else "modified"
            changes.append(Change(item, kind, old, new))
        return ChangeSet(tuple(changes), tuple(self._moves))

    def fork(self, *, access: str = "overlay") -> "Folder":
        if access not in ("read", "write", "overlay"):
            raise ValueError("access must be read, write, or overlay")
        return Folder(_base=self._effective(), access=access, _writer=self._writer)

    def writer(self) -> WriterLock:
        return self._writer

    def begin_transaction(self, *, blocking: bool = True, path: str = "") -> FolderTransaction:
        self._check_write()
        prefix = _path(path)
        self._writer.acquire(blocking=blocking)
        effective = self._effective()
        base = {name[len(prefix) + 1:] if prefix else name: value
                for name, value in effective.items()
                if _under(name, prefix) and name != prefix}
        # The private overlay has its own writer semaphore.  The parent lock is
        # already held, so nested applies serialize within the overlay without
        # trying to reacquire the parent lock.
        return FolderTransaction(self, Folder(_base=base, access="overlay"), prefix)

    def _install_locked(self, changes: ChangeSet, *, include: Iterable[str] | None = None,
                        exclude: Iterable[str] | None = None) -> ChangeSet:
        selected = changes.selected(include, exclude)
        current = self._effective()
        for change in selected.changes:
            if current.get(change.path) != change.before:
                raise FolderConflictError(change.path)
        updates = {change.path: (_TOMBSTONE if change.after is None else change.after)
                   for change in selected.changes}
        self._overlay.update(updates)
        return selected

    def install(self, changes: ChangeSet, *, include: Iterable[str] | None = None,
                exclude: Iterable[str] | None = None, blocking: bool = True) -> ChangeSet:
        self._check_write()
        self._writer.acquire(blocking=blocking)
        try:
            selected = self._install_locked(changes, include=include, exclude=exclude)
        finally:
            self._writer.release()
        return selected

    def install_from(self, child: "Folder", *, include: Iterable[str] | None = None,
                     exclude: Iterable[str] | None = None, blocking: bool = True) -> ChangeSet:
        return self.install(child.diff(), include=include, exclude=exclude, blocking=blocking)
