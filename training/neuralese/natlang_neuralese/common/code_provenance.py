"""Which code produced a record: the git commit (and dirty flag) of the checkout the process runs from.

Production RL, the actors and the learner run directly from the canonical checkout (owner, 2026-11-09). A process keeps
the code it started with; each newly started worker, learner or actor gets current main. A code change is therefore
never a reason to refuse, resume-block or fail an attestation: every round, update, checkpoint and actor records the
commit that produced it instead (``code_provenance``), and the readers compare nothing but the schema.

Provenance is read once per process (``code_provenance()`` caches the first call), because a running process keeps
the modules it imported. ``dirty`` means tracked files differ from ``HEAD`` at that moment.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

SCHEMA = 'natlang.code-provenance/1'
_CACHE = {}


def _git(root, *args):
    try:
        done = subprocess.run(['git', '-c', 'safe.directory=*', '-C', str(root), *args], capture_output=True,
                              text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def repository_root(package=None):
    """The checkout that holds this package (``natlang_neuralese`` lives at ``<repo>/training/neuralese``)."""
    return Path(package).resolve() if package is not None else Path(__file__).resolve().parents[3]


def compute(root):
    root = Path(root)
    declared = os.environ.get('NATLANG_CODE_COMMIT')   # a container that mounts the checkout without git (Pop actor)
    if declared:
        return {'schema': SCHEMA, 'git_commit': declared, 'dirty': os.environ.get('NATLANG_CODE_DIRTY') == '1'}
    commit = _git(root, 'rev-parse', 'HEAD')
    if commit is not None:
        dirty = bool(_git(root, 'status', '--porcelain', '--untracked-files=no'))
        return {'schema': SCHEMA, 'git_commit': commit, 'dirty': dirty}
    # An archive-style tree (a retired frozen runtime, a cloud code bundle) names its commit in a receipt file.
    for name, read in (('runtime-input-closure.json', lambda p: json.loads(p.read_text()).get('commit')),
                       ('.coreweave-code-commit', lambda p: p.read_text().strip())):
        path = root / name
        if path.is_file():
            try:
                return {'schema': SCHEMA, 'git_commit': read(path) or None, 'dirty': None}
            except (OSError, ValueError):
                break
    return {'schema': SCHEMA, 'git_commit': None, 'dirty': None}


def code_provenance(root=None, *, fresh=False):
    """``{schema, git_commit, dirty}`` of the checkout this process runs from (cached for the life of the process).
    ``fresh`` reads the checkout now: the code a child process started at this moment will run."""
    root = repository_root(root)
    if fresh:
        return compute(root)
    if root not in _CACHE:
        _CACHE[root] = compute(root)
    return dict(_CACHE[root])


def code_changes(recorded, current=None):
    """``{'git_commit': [recorded, current]}`` when the commit differs (informational: never a reason to refuse)."""
    current = current if current is not None else code_provenance()
    old = (recorded or {}).get('git_commit') if isinstance(recorded, dict) else None
    new = current.get('git_commit')
    return {'git_commit': [old, new]} if old != new else {}
