"""Stable invocation identity and model sampling policy.

The logical path is reproducible; the run id distinguishes independent executions.
Seed derivation v1 uses UTF-8 JSON with sorted keys and SHA-256, reduced to the
signed 31-bit nonnegative range accepted by the supported llama.cpp backends.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class SeedPolicy:
    mode: str = "compatibility"  # compatibility | derived | backend
    root: Optional[int] = None
    version: str = "sha256-json-v1"

    def __post_init__(self):
        if self.mode not in ("compatibility", "derived", "backend"):
            raise ValueError("unknown seed policy")
        if self.mode == "derived" and (not isinstance(self.root, int) or isinstance(self.root, bool)):
            raise ValueError("derived seed policy requires an integer root")

    def seed(self, call_path: str, attempt: int, purpose: str, ordinal: int = 0) -> Optional[int]:
        if self.mode == "compatibility":
            return 0
        if self.mode == "backend":
            return None
        payload = {"version": self.version, "root": self.root, "path": call_path,
                   "attempt": attempt, "purpose": purpose, "ordinal": ordinal}
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") % (2**31)


@dataclass(frozen=True)
class RunOptions:
    seed: SeedPolicy = field(default_factory=SeedPolicy)
    max_episodes: int = 256
    max_depth: int = 8
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    @classmethod
    def compatibility(cls, **kwargs) -> "RunOptions":
        return cls(**kwargs)


@dataclass(frozen=True)
class Invocation:
    run_id: str
    path: str
    attempt: int
    parent_path: Optional[str] = None

    @property
    def call_id(self) -> str:
        return f"{self.path or '$root'}@{self.attempt}"
