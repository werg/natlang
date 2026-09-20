"""Explicit, measured model assistance for optional host libraries."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HelpRequest:
    model: str
    prompt: str
    temperature: float = 0
    max_tokens: int = 256
    seed: int | None = None


@dataclass(frozen=True)
class HelpResult:
    status: str
    observation: str | None
    model: str
    usage: dict
    error: str | None = None


class ModelHelp:
    def __init__(self, driver):
        self.driver = driver

    def request(self, req: HelpRequest) -> HelpResult:
        if not req.model or not req.prompt or req.max_tokens < 1:
            raise ValueError("model, prompt and positive token budget are required")
        try:
            reply = self.driver(req)
            if not isinstance(reply.get("observation"), str):
                raise ValueError("model help returned no text observation")
            return HelpResult("ok", reply["observation"], req.model,
                              dict(reply.get("usage") or {}))
        except Exception as exc:
            return HelpResult("failed", None, req.model, {}, str(exc))
