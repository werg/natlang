"""Load hash-bound external service modules for authored Natlang codebases.

Service source is an execution dependency, not part of the model-editable package.
The collector receives the text through semantics.services and derives the exact
model-visible namespace with ts-host/native/external.ts.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

_ALIAS = re.compile(r'^[A-Za-z_$][A-Za-z0-9_$]*$')
_SHA256 = re.compile(r'^[0-9a-f]{64}$')


def load_external_services(package_root: Path, metadata: dict[str, Any]) -> tuple[dict[str, str], dict[str, dict[str, str]]]:
    package_root = package_root.resolve(strict=True)
    if package_root.name != 'package':
        raise ValueError('package root must be named package/')
    codebase_root = package_root.parent.resolve(strict=True)
    declarations = metadata.get('external_services', {})
    if not isinstance(declarations, dict):
        raise ValueError('metadata.external_services must map service aliases to bound source records')

    services: dict[str, str] = {}
    bindings: dict[str, dict[str, str]] = {}
    for alias, binding in sorted(declarations.items()):
        if not isinstance(alias, str) or not _ALIAS.fullmatch(alias):
            raise ValueError(f'invalid external service alias: {alias!r}')
        if not isinstance(binding, dict):
            raise ValueError(f'external service {alias!r} must be an object')
        source_path = binding.get('source_path')
        expected_sha = binding.get('source_sha256')
        if not isinstance(source_path, str) or not source_path or source_path.startswith('/') or '\\' in source_path:
            raise ValueError(f'external service {alias!r} needs a safe codebase-relative source_path')
        parts = Path(source_path).parts
        if any(part in {'', '.', '..'} for part in parts) or parts[0] == 'package':
            raise ValueError(f'external service {alias!r} must stay outside the model-editable package')
        if Path(source_path).suffix != '.ts':
            raise ValueError(f'external service {alias!r} source must be TypeScript for the native external-module contract')
        if not isinstance(expected_sha, str) or not _SHA256.fullmatch(expected_sha):
            raise ValueError(f'external service {alias!r} needs a full lowercase source_sha256')
        source_file = codebase_root / source_path
        if source_file.is_symlink() or not source_file.is_file():
            raise ValueError(f'external service {alias!r} source is not a regular file')
        resolved = source_file.resolve(strict=True)
        if not resolved.is_relative_to(codebase_root) or resolved.is_relative_to(package_root):
            raise ValueError(f'external service {alias!r} source escapes its codebase service boundary')
        source_bytes = source_file.read_bytes()
        actual_sha = hashlib.sha256(source_bytes).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f'external service {alias!r} source hash mismatch')
        try:
            source = source_bytes.decode('utf-8')
        except UnicodeDecodeError as error:
            raise ValueError(f'external service {alias!r} source must be UTF-8 TypeScript') from error
        if not source.strip():
            raise ValueError(f'external service {alias!r} source is empty')
        services[alias] = source
        bindings[alias] = {'source_path': source_path, 'source_sha256': actual_sha}
    return services, bindings


def bind_world_services(package_root: Path, metadata: dict[str, Any], services: dict[str, str],
                        bindings: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Bind hidden JSON fixtures into trusted host source, keyed by exact world ID."""
    root = package_root.parent.resolve(strict=True)
    result: dict[str, dict[str, str]] = {}
    for alias, declaration in metadata.get('external_services', {}).items():
        if 'fixtures_path' not in declaration:
            continue
        path = declaration['fixtures_path']
        digest = declaration.get('fixtures_sha256')
        if (not isinstance(path, str) or not path or path.startswith('/') or '\\' in path
                or any(part in {'', '.', '..'} for part in path.split('/')) or path.startswith('package/')
                or not isinstance(digest, str) or not _SHA256.fullmatch(digest)):
            raise ValueError(f'invalid fixture binding for {alias}')
        file = root / path
        if file.is_symlink() or not file.is_file() or not file.resolve().is_relative_to(root):
            raise ValueError(f'unsafe fixture binding for {alias}')
        raw = file.read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f'fixture hash mismatch for {alias}')
        marker = '/*__NATLANG_WORLD_FIXTURE__*/'
        if services[alias].count(marker) != 1:
            raise ValueError(f'{alias} must contain exactly one fixture marker')
        bindings[alias].update(fixtures_path=path, fixtures_sha256=digest)
        seen = set()
        for line in raw.decode('utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'] or 'fixture' not in row or row['id'] in seen:
                raise ValueError(f'invalid or duplicate fixture world for {alias}')
            seen.add(row['id'])
            source = services[alias].replace(marker, json.dumps(row['fixture'], ensure_ascii=True, separators=(',', ':')))
            result.setdefault(row['id'], {})[alias] = source
    return result
