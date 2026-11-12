"""ts-host's own prompt composition for training-data rendering: one implementation for serving and training rows.

Training rows rebuilt from recordings (serving_prompt.serving_messages) and program-header rendering
(prompt_rendering.render_system_prompt) are computed by the runtime's code, through a persistent
``ts-host/scripts/serving-request.mjs`` child per process, never by a Python port of it. ``NATLANG_TS_DIST`` names the
ts-host build (default ts-host/dist), ``NATLANG_NODE`` the node binary (default: node on PATH).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
_lock = threading.Lock()
_child: subprocess.Popen | None = None
_child_pid: int | None = None


def _process() -> subprocess.Popen:
    global _child, _child_pid
    if _child is None or _child.poll() is not None or _child_pid != os.getpid():
        node = os.environ.get('NATLANG_NODE') or shutil.which('node')
        if not node:
            raise RuntimeError('ts-host prompt composition needs node (NATLANG_NODE or PATH)')
        dist = os.environ.get('NATLANG_TS_DIST') or str(REPO / 'ts-host' / 'dist')
        _child = subprocess.Popen([node, str(REPO / 'ts-host' / 'scripts' / 'serving-request.mjs'), dist],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        _child_pid = os.getpid()
    return _child


def request(payload: dict) -> dict:
    """One request to ts-host (``serving-request.mjs`` ops); a ts-host error raises ValueError."""
    with _lock:
        child = _process()
        child.stdin.write(json.dumps(payload) + '\n')
        child.stdin.flush()
        line = child.stdout.readline()
    if not line:
        raise RuntimeError('ts-host serving-request child exited')
    answer = json.loads(line)
    if isinstance(answer, dict) and 'error' in answer:
        raise ValueError(f"ts-host: {answer['error']}")
    return answer


@lru_cache(maxsize=1)
def identity() -> dict:
    """The ts-host code that composes the prompts (commit, dirty flag, build directory, piece-history sha)."""
    return request({'op': 'identity'})
