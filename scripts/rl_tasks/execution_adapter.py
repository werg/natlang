"""Hash-bound execution adapter and external runtime dependency validation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def verify_runtime_tree(*, root: Path, manifest_path: Path, manifest_sha256: str) -> dict:
    """Verify an exact external directory against a codebase-owned manifest."""
    root = root.expanduser().resolve(strict=True)
    manifest_path = manifest_path.expanduser().resolve(strict=True)
    if not root.is_dir() or not manifest_path.is_file() or sha256_file(manifest_path) != manifest_sha256:
        raise ValueError('runtime dependency tree or manifest is missing or changed')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('schema') != 'natlang.telecom-python-dependency-tree/1' or Path(manifest.get('root', '')).resolve() != root:
        raise ValueError('runtime dependency tree manifest has the wrong schema or root')
    expected = manifest.get('files')
    if not isinstance(expected, dict) or not expected:
        raise ValueError('runtime dependency tree manifest must list files')
    for relative, digest in expected.items():
        rel = Path(relative)
        if (not isinstance(relative, str) or not relative or rel.is_absolute() or
                any(part in {'', '.', '..'} for part in rel.parts) or
                not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest)):
            raise ValueError(f'invalid runtime dependency manifest entry: {relative!r}')
        candidate = root / rel
        if candidate.is_symlink() or not candidate.is_file() or sha256_file(candidate) != digest:
            raise ValueError(f'runtime dependency file is missing or changed: {candidate}')
    actual = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError(f'runtime dependency tree contains symlink: {path}')
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(expected):
        raise ValueError('runtime dependency tree file set differs from its manifest')
    tree_digest = hashlib.sha256(json.dumps(sorted(expected.items()), separators=(',', ':')).encode()).hexdigest()
    return {'root': str(root), 'manifest_path': str(manifest_path), 'manifest_sha256': manifest_sha256,
            'tree_sha256': tree_digest, 'file_count': len(expected),
            'total_bytes': sum((root / relative).stat().st_size for relative in expected),
            'files': expected}


def verify_execution_adapter_paths(metadata: dict, codebase_root: Path) -> list[dict[str, str]]:
    import re

    adapter = metadata.get('execution_adapter')
    if adapter is None:
        return []
    if not isinstance(adapter, dict):
        raise ValueError('metadata.execution_adapter must be an object')
    paths = adapter.get('paths', {})
    if not isinstance(paths, dict):
        raise ValueError('metadata.execution_adapter.paths must be an object')
    bindings = []
    root = codebase_root.resolve(strict=True)
    for relative, expected in sorted(paths.items()):
        rel = Path(relative) if isinstance(relative, str) else Path('.')
        if (not isinstance(relative, str) or not relative or rel.is_absolute() or
                any(part in {'', '.', '..'} for part in rel.parts) or '\\' in relative):
            raise ValueError(f'unsafe execution_adapter path: {relative!r}')
        if not isinstance(expected, str) or not re.fullmatch(r'[a-f0-9]{64}', expected):
            raise ValueError(f'execution_adapter path requires a lowercase SHA-256: {relative}')
        target = root / rel
        current = root
        for part in rel.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f'execution_adapter path cannot traverse a symlink: {relative}')
        try:
            target.resolve(strict=True).relative_to(root)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f'execution_adapter path is missing or escapes codebase root: {relative}') from exc
        if not target.is_file():
            raise ValueError(f'execution_adapter path is not a regular file: {relative}')
        actual = sha256_file(target)
        if actual != expected:
            raise ValueError(f'execution_adapter path hash mismatch for {relative}: expected {expected}, got {actual}')
        bindings.append({'path': str(target.resolve(strict=True)), 'relative_path': relative, 'sha256': actual})
    return bindings


def verify_execution_adapter_trees(metadata: dict, codebase_root: Path) -> list[dict]:
    adapter = metadata.get('execution_adapter') or {}
    trees = adapter.get('runtime_trees', [])
    if not isinstance(trees, list):
        raise ValueError('metadata.execution_adapter.runtime_trees must be a list')
    root = codebase_root.resolve(strict=True)
    checked = []
    names = set()
    for record in trees:
        if not isinstance(record, dict) or not isinstance(record.get('name'), str) or record['name'] in names:
            raise ValueError('runtime tree declarations must have unique names')
        names.add(record['name'])
        relative = record.get('manifest_path')
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or any(p in {'', '.', '..'} for p in Path(relative).parts):
            raise ValueError('runtime tree manifest_path must be codebase-relative')
        manifest_path = root / relative
        if manifest_path.is_symlink() or not manifest_path.resolve(strict=True).is_relative_to(root):
            raise ValueError('runtime tree manifest path is unsafe')
        checked.append({'name': record['name'], **verify_runtime_tree(
            root=Path(record['root']), manifest_path=manifest_path,
            manifest_sha256=record.get('manifest_sha256', ''))})
    return checked
