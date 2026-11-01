"""Hash-bound execution adapter and external runtime dependency validation."""
from __future__ import annotations

import json
import re
from pathlib import Path
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file, sha256_hex


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
    tree_digest = sha256_hex(json.dumps(sorted(expected.items()), separators=(',', ':')).encode())
    return {'root': str(root), 'manifest_path': str(manifest_path), 'manifest_sha256': manifest_sha256,
            'tree_sha256': tree_digest, 'file_count': len(expected),
            'total_bytes': sum((root / relative).stat().st_size for relative in expected),
            'files': expected}


def reject_uninvoked_execution_adapter(metadata: dict) -> None:
    if metadata.get('execution_adapter') is not None:
        raise ValueError('metadata.execution_adapter is not invoked by run_bonsai_queue')


def verify_runtime_trees(metadata: dict, codebase_root: Path) -> list[dict]:
    runtime = metadata.get('collector_runtime') or {}
    if not isinstance(runtime, dict):
        raise ValueError('metadata.collector_runtime must be an object')
    trees = runtime.get('trees', [])
    if not isinstance(trees, list):
        raise ValueError('metadata.collector_runtime.trees must be a list')
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
