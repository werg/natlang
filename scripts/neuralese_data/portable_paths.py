"""Portable host-tool path resolution for historical conversion utilities."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import Mapping


def node_executable(*, environ: Mapping[str, str] | None = None,
                    which=shutil.which) -> str:
    """Return the explicitly configured Node binary or the active PATH binary.

    Callers that require a specific Node release should provide it in
    ``NATLANG_NODE`` or their launched PATH. No workstation-specific fallback
    is embedded in the conversion code.
    """
    env = os.environ if environ is None else environ
    configured = env.get("NATLANG_NODE")
    candidate = configured if configured else which("node", path=env.get("PATH"))
    if not candidate:
        raise RuntimeError("Node is unavailable; set NATLANG_NODE or add the qualified Node binary to PATH")
    path = Path(candidate).expanduser().resolve(strict=True)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"configured Node executable is not an executable file: {path}")
    return str(path)
