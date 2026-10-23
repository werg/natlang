"""Location binding for the external llama.cpp GGUF tooling checkout.

This is used by ordinary model precision conversion. Neuralese projected-head
export is retired; the fork pin remains only as a toolchain location for the
independent GGUF readers and packers.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


PIN_FILE = Path(__file__).resolve().parents[2] / "llama-cpp-fork.json"


def llama_cpp_root() -> Path:
    configured = os.environ.get("NATLANG_LLAMA_NEURALESE")
    if configured:
        return Path(configured).expanduser().resolve()
    pinned = Path(json.loads(PIN_FILE.read_text())["path"])
    return pinned if pinned.is_absolute() else (PIN_FILE.parents[2] / pinned).resolve()
