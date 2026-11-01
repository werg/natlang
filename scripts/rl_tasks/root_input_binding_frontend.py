"""Validate task input keys with the exact root binder used by Luna collection."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / 'scripts/rl_tasks/validate_root_input_bindings.mjs'


def validate_root_input_bindings(records: list[dict], *, node: str | None = None,
                                 runtime_dist: Path | None = None) -> dict:
    if not records:
        raise ValueError('root input binding preflight requires at least one case')
    groups: dict[tuple[str, str], dict] = {}
    for record in records:
        semantics = record.get('semantics') or {}
        root, files, inputs = semantics.get('root'), semantics.get('files'), semantics.get('inputs')
        if not isinstance(root, str) or not isinstance(files, dict) or not isinstance(inputs, dict):
            raise ValueError(f"{record.get('id')}: missing root/files/inputs for collector binding")
        key = (root, json.dumps(files, sort_keys=True, ensure_ascii=False, separators=(',', ':')))
        group = groups.setdefault(key, {'root': root, 'files': files, 'worlds': []})
        group['worlds'].append({'id': record.get('id'), 'inputs': inputs})
    request = {'roots': list(groups.values())}
    if runtime_dist is not None:
        request['runtime_dist'] = str(runtime_dist.resolve(strict=True))
    executable = node or shutil.which('node')
    if not executable:
        raise RuntimeError('Node is required to preflight root argument binding with the collector')
    result = subprocess.run([executable, str(FRONTEND)], input=json.dumps(request, ensure_ascii=False), text=True,
                            capture_output=True, cwd=REPO, timeout=120, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f'collector root argument binding rejected case inputs: {detail}')
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError('root input frontend returned invalid JSON') from error
    if receipt.get('schema') != 'natlang.root-input-binding-preflight/1' or receipt.get('loaded_by') != 'teacher.prepareProgramNode':
        raise RuntimeError('root input frontend did not return an authenticated collector binding receipt')
    return receipt
