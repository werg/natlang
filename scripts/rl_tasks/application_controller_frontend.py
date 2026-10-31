"""Run controller source through the exact Natlang loader used by collection."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / 'scripts/rl_tasks/validate_controller_frontend.mjs'


def validate_controller_frontend(files: dict[str, str], root: str, controller: dict[str, object], *,
                                 node: str | None = None) -> dict[str, object]:
    """Compile and attach the actual declared controller through the collector's Natlang loader."""
    executable = node or shutil.which('node')
    if not executable:
        raise RuntimeError('Node is required to preflight an application controller with the collector loader')
    request = json.dumps({'files': files, 'root': root, 'controller': controller}, ensure_ascii=False)
    result = subprocess.run([executable, str(FRONTEND)], input=request, text=True,
                            capture_output=True, cwd=REPO, timeout=120, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise ValueError(f'collector Natlang loader rejected application controller: {detail}')
    try:
        receipt = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError('controller frontend returned invalid JSON') from error
    if (receipt.get('schema') != 'natlang.application-controller-frontend-check/1' or
            receipt.get('loaded_by') != 'natlang.runtime.loadVirtualNatlang' or
            receipt.get('accepted') is not True):
        raise RuntimeError('controller frontend did not return a qualified loader receipt')
    return receipt
