"""General read-only, lazily observed trees supplied by an application host."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Protocol


@dataclass(frozen=True)
class TreeEntry:
    name: str
    kind: str  # branch | leaf


class TreeProvider(Protocol):
    def list(self, path: tuple[str, ...]) -> Iterable[TreeEntry]: ...
    def read(self, path: tuple[str, ...]) -> Any: ...


class _TreeCache:
    def __init__(self, provider: TreeProvider):
        self.provider = provider
        self.branches: dict[tuple[str, ...], tuple[TreeEntry, ...]] = {}
        self.leaves: dict[tuple[str, ...], Any] = {}


class LazyDict:
    """Host representation of Dict<T>; each observed branch or leaf is fetched once."""

    def __init__(self, provider: TreeProvider, *, label: str = "host tree",
                 _cache: _TreeCache | None = None, _path: tuple[str, ...] = ()):
        self._cache = _cache or _TreeCache(provider)
        self._path = _path
        self.label = label

    @property
    def path(self) -> tuple[str, ...]:
        return self._path

    def entries(self) -> tuple[TreeEntry, ...]:
        if self._path not in self._cache.branches:
            entries, names = [], set()
            for raw in self._cache.provider.list(self._path):
                entry = raw if isinstance(raw, TreeEntry) else TreeEntry(str(raw["name"]), str(raw["kind"]))
                if (not entry.name or "/" in entry.name or "\\" in entry.name or entry.name in (".", "..") or
                        entry.name.startswith("$") or entry.kind not in ("branch", "leaf")):
                    raise ValueError(f"invalid host-tree entry: {entry!r}")
                if entry.name in names:
                    raise ValueError(f"duplicate host-tree entry: {entry.name}")
                names.add(entry.name); entries.append(entry)
            self._cache.branches[self._path] = tuple(sorted(entries, key=lambda item: item.name))
        return self._cache.branches[self._path]

    def child(self, name: str):
        entry = next((item for item in self.entries() if item.name == name), None)
        if entry is None:
            raise KeyError(name)
        path = self._path + (name,)
        if entry.kind == "branch":
            return LazyDict(self._cache.provider, label=self.label, _cache=self._cache, _path=path)
        if path not in self._cache.leaves:
            self._cache.leaves[path] = self._cache.provider.read(path)
        return self._cache.leaves[path]

    def __deepcopy__(self, memo):
        return self

    def __repr__(self):
        at = "/".join(self._path)
        suffix = f", {at!r}" if at else ""
        return f"LazyDict({self.label!r}{suffix})"


class MemoryTreeProvider:
    """Portable provider over slash-separated leaf paths."""

    def __init__(self, leaves: Mapping[str, Any]):
        self.leaves = {}
        for raw, value in leaves.items():
            path = tuple(part for part in str(raw).replace("\\", "/").split("/") if part)
            if not path or str(raw).startswith("/") or any(part in (".", "..") for part in path):
                raise ValueError(f"invalid host-tree path: {raw}")
            self.leaves[path] = value
        for path in self.leaves:
            if any(path[:end] in self.leaves for end in range(1, len(path))):
                raise ValueError(f"host-tree leaf is also a branch: {'/'.join(path)}")

    def list(self, path: tuple[str, ...]):
        children = {}
        for leaf in self.leaves:
            if leaf[:len(path)] != path or len(leaf) == len(path):
                continue
            name = leaf[len(path)]
            kind = "leaf" if len(leaf) == len(path) + 1 else "branch"
            if children.get(name) == "branch":
                continue
            children[name] = kind
        return [TreeEntry(name, kind) for name, kind in children.items()]

    def read(self, path: tuple[str, ...]):
        if path not in self.leaves:
            raise KeyError("/".join(path))
        return self.leaves[path]


def lazy_dict(leaves: Mapping[str, Any], *, label: str = "lazy dictionary") -> LazyDict:
    return LazyDict(MemoryTreeProvider(leaves), label=label)
