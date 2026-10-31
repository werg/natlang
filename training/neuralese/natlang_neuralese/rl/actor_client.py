"""Bounded HTTP reads and owner-held artifact uploads for a caller-verified native actor endpoint."""
from __future__ import annotations

import json
import urllib.request

OWNER_HEADER = "x-natlang-owner"  # serve/http.py OWNER_HEADER


class NativeActorClient:
    """Read exact artifacts from a separately identity-verified native actor."""
    def __init__(self, base: str, timeout: float = 3600):
        self.base, self.timeout = base.rstrip("/"), timeout

    def get_bytes(self, path: str) -> bytes:
        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as response:
            return response.read()

    def put_bytes(self, path: str, data: bytes, *, owner: str | None = None) -> dict:
        """Upload an exact artifact; ``owner`` holds it on the server (resume restores retained partial blocks)."""
        request = urllib.request.Request(self.base + path, data=data, method="PUT",
                                         headers={"content-type": "application/octet-stream",
                                                  **({OWNER_HEADER: owner} if owner else {})})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())
