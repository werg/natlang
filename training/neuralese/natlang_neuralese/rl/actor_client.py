"""Bounded HTTP reads from a caller-verified native actor endpoint."""
from __future__ import annotations

import urllib.request


class NativeActorClient:
    """Read exact artifacts from a separately identity-verified native actor."""
    def __init__(self, base: str, timeout: float = 3600):
        self.base, self.timeout = base.rstrip("/"), timeout

    def get_bytes(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as response:
            return response.read()
