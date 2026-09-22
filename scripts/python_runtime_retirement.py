"""Guards for Python execution paths retired during the Node migration."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
NODE_TEACHER_ENTRYPOINT = ROOT / "ts-host/scripts/teacher-collector.mjs"


def refuse_legacy_scope_eval(entrypoint: str) -> None:
    """Prevent a Python CLI from collecting new scope-eval-v1 trajectories."""
    if NODE_TEACHER_ENTRYPOINT.exists():
        raise SystemExit(
            f"{entrypoint} is retired for scope-eval-v1; run "
            "node ts-host/scripts/teacher-collector.mjs instead"
        )
