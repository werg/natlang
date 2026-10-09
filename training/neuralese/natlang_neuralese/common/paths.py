"""Logical machine roots (plans/ARCHITECTURE_IMPROVEMENT.md C4).

Resolution order per root: environment variable ``NATLANG_<ROOT>`` (upper case),
then ``~/.config/natlang/machine.toml`` (top-level keys, or a ``[roots]`` table),
then the built-in default, which equals the literal path used before this module
existed. Stdlib-only.
"""
from __future__ import annotations

import os
import tomllib
from pathlib import Path

DEFAULTS: dict[str, str] = {
    "repo": "/home/werg/natlang",
    "models": "/home/werg/data/models",
    "data_nvme": "/home/werg/data",
    "data_hdd": "/mnt/external",
    "archive": "/mnt/external/sdkb-archive",
    "llama_cpp": "/home/werg/llama.cpp-neuralese",
}

PROFILE_ENV = "NATLANG_MACHINE_TOML"


def profile_path() -> Path:
    override = os.environ.get(PROFILE_ENV)
    return Path(override) if override else Path.home() / ".config" / "natlang" / "machine.toml"


def _profile() -> dict[str, str]:
    path = profile_path()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return {}
    roots = data.get("roots", data)
    if not isinstance(roots, dict):
        return {}
    return {k: v for k, v in roots.items() if isinstance(v, str)}


def root(name: str) -> Path:
    """Directory for a logical root."""
    if name not in DEFAULTS:
        raise KeyError(f"unknown logical root {name!r}; known: {sorted(DEFAULTS)}")
    value = os.environ.get(f"NATLANG_{name.upper()}") or _profile().get(name) or DEFAULTS[name]
    return Path(value).expanduser()


def resolve(name: str, *relative: str) -> Path:
    """``root(name) / relative...``."""
    return root(name).joinpath(*relative)


def resolve_str(name: str, *relative: str) -> str:
    return str(resolve(name, *relative))
