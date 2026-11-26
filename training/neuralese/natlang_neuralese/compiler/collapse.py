"""Release identity of compiled sites.

A compiled site needs no separate collapse: the compiler emits the program tuner's own stack (``compiled_site``): an
``xs`` adapter core in the canonical basis, which ``tune.build.export_release_nodes`` exports as a standard GGUF LoRA,
and the header's native state (``.nzs``), which every engine splices as stored. ``compiled_site.compile_release``
writes both in the ``tune.build`` release layout. This module keeps only the cache key that binds a compiled entry to
its exact site, header, compiler, policy and base.
"""
from __future__ import annotations

import hashlib
import json

COLLAPSE_SCHEMA = "natlang.compiler.collapsed-release/2"


def provenance_key(*, site_id: str, header_sha: str, compiler_sha: str, policy_sha: str,
                   base_gguf_sha256: str = "") -> str:
    """Deterministic cache key binding exact site, header, compiler, policy, and base GGUF identity."""
    payload = json.dumps({"schema": COLLAPSE_SCHEMA, "site_id": str(site_id), "header_sha": str(header_sha),
                          "compiler_sha": str(compiler_sha), "policy_sha": str(policy_sha),
                          "base_gguf_sha256": str(base_gguf_sha256)},
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
