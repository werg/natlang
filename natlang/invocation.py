"""Stable invocation identity and model sampling policy.

The logical path is reproducible; the run id distinguishes independent executions.
Seed derivation v1 uses UTF-8 JSON with sorted keys and SHA-256, reduced to the
signed 31-bit nonnegative range accepted by the supported llama.cpp backends.
"""
from __future__ import annotations

import hashlib
import json
import random
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

    @property
    def backend_range(self) -> str:
        return "llama.cpp signed nonnegative 31-bit seed (0..2147483647)"


@dataclass(frozen=True)
class ModelSettings:
    temperature: float = 0.2
    max_turns: Optional[int] = None
    max_tokens: Optional[int] = None
    max_seconds: Optional[float] = None
    turn_tokens: Optional[int] = None

    def __post_init__(self):
        if ((self.max_turns is not None and self.max_turns < 1) or
                (self.max_tokens is not None and self.max_tokens < 1) or
                (self.max_seconds is not None and self.max_seconds <= 0)):
            raise ValueError("explicit model budgets must be positive")
        if self.turn_tokens is not None and self.turn_tokens < 1:
            raise ValueError("turn token budget must be positive")


@dataclass(frozen=True)
class RunOptions:
    seed: SeedPolicy = field(default_factory=SeedPolicy)
    model: Optional[ModelSettings] = None  # None keeps per-agent compatibility settings
    world_seed: Optional[int] = None
    max_episodes: Optional[int] = None
    max_depth: Optional[int] = None
    max_actions: Optional[int] = None
    max_tool_calls: Optional[int] = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self):
        if self.max_episodes is not None and self.max_episodes < 1:
            raise ValueError("max_episodes must be positive or None")
        if self.max_depth is not None and self.max_depth < 1:
            raise ValueError("max_depth must be positive or None")
        if self.max_actions is not None and self.max_actions < 1:
            raise ValueError("max_actions must be positive or None")
        if self.max_tool_calls is not None and self.max_tool_calls < 1:
            raise ValueError("max_tool_calls must be positive or None")

    @classmethod
    def compatibility(cls, **kwargs) -> "RunOptions":
        return cls(**kwargs)

    def world_rng(self, purpose: str, path: str = "") -> random.Random:
        if self.world_seed is None:
            raise ValueError("world RNG needs a separately supplied world seed")
        payload = json.dumps({"version": "world-sha256-json-v1", "seed": self.world_seed,
                              "purpose": purpose, "path": path}, sort_keys=True,
                             separators=(",", ":"), ensure_ascii=False).encode()
        return random.Random(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big"))


@dataclass(frozen=True)
class Invocation:
    run_id: str
    path: str
    attempt: int
    parent_path: Optional[str] = None

    @property
    def call_id(self) -> str:
        return f"{self.path or '$root'}@{self.attempt}"
