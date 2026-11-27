"""Per-call compilation with retrieved experience: ``[z_site, z_kb]`` -> a compiled release (``.nzs`` + ``xs`` cores).

Bridges ``ScopedExperienceKB`` (retrieval, Stage C slots), the Live-KB writer (``PplxWarmstartWriter`` in production; its lookup query),
``CompiledSiteHead`` and an optional ``LearnedBankBasis``: one call site, its retrieved experience and its enclosing
scopes' cores (``C = C_site + C_scope ...``) compile into a ``compiled_site.compile_release`` release, cached by
provenance. The release is the ordinary compiled-site release (header state + GGUF-exportable cores), so serving it is
the compiled-site row of ``docs/ENGINE_PARITY.md``; nothing here adds a serving feature.

The KB's records must have been written by the same writer whose query is used here (one key/value space); the
caller supplies ``exclude_tasks`` (held closures, the call's own task) and the exact compiler and policy identities.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .collapse import provenance_key
from .compiled_site import (
    CompiledSiteHead,
    LearnedBankBasis,
    compile_release,
    compose_site_cores,
    init_block_values,
    init_header_state,
)


@dataclass(frozen=True)
class PerCallCompileRequest:
    site_id: str
    header: str
    messages: list[dict]
    tools: list[dict]
    compiler_sha: str                       # the compiler head's checkpoint identity (no default: provenance)
    policy_sha: str                         # the served policy's checkpoint identity
    scopes: tuple[str, ...] = ('natlang:global',)
    exclude_tasks: tuple[str, ...] = ()
    scope_headers: tuple[str, ...] = ()     # enclosing repo/folder/partial headers composed in C-space
    query_text: str | None = None
    max_kb_slots: int = 8


class PerCallNeuralHKBCompiler:
    """One-pass call-entry compiler: header + retrieved experience + scope cores, cached by provenance key."""

    def __init__(self, engine, head: CompiledSiteHead, *, adapter_id: str, release_root: str | Path,
                 writer=None, scoped_kb=None, basis_module: LearnedBankBasis | None = None):
        if (scoped_kb is None) != (writer is None):
            raise ValueError('retrieval needs both the experience KB and the writer whose space its records are in')
        self.engine = engine
        self.head = head.eval()
        self.adapter_id = adapter_id
        self.release_root = Path(release_root)
        self.release_root.mkdir(parents=True, exist_ok=True)
        self.writer = writer.eval() if writer is not None else None
        self.scoped_kb = scoped_kb
        self.basis_module = basis_module
        self._cache: dict[str, dict] = {}

    def _kb_generation(self):
        return None if self.scoped_kb is None else self.scoped_kb.provenance()['generation']

    @torch.no_grad()
    def compile_call(self, req: PerCallCompileRequest, *, owner=None) -> dict:
        """Compile one call site with its retrieved experience and scope cores into a cached release."""
        identity = {'header': req.header, 'scope_headers': list(req.scope_headers), 'scopes': list(req.scopes),
                    'exclude_tasks': sorted(req.exclude_tasks), 'query': req.query_text, 'max_kb_slots': req.max_kb_slots,
                    'kb_generation': self._kb_generation(), 'base': self.engine.native_identity}
        header_sha = hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()
        key = provenance_key(site_id=req.site_id, header_sha=header_sha, compiler_sha=req.compiler_sha,
                             policy_sha=req.policy_sha)
        if key in self._cache:
            return self._cache[key]
        state_id = init_header_state(self.engine, req.messages, req.tools, req.header)
        init_block = self.engine.store.get_native_state(state_id).block
        z_kb = k_kb = w_kb = kb_valid = None
        kb_hits = 0
        if self.scoped_kb is not None and req.scopes:
            q_vec = self.writer.encode_query_texts(self.engine, [req.query_text or req.header])[0].detach().cpu()
            z_kb, k_kb, w_kb, kb_valid = self.scoped_kb.retrieve_unified_slots(
                q_vec, scopes=list(req.scopes), exclude_tasks=set(req.exclude_tasks), max_slots=req.max_kb_slots,
                normalize=False)
            kb_hits = int(kb_valid.sum().item())
        extra_cores = None
        scope_writes: list[str] = []
        if req.scope_headers:
            # A scope text that occurs once in the call is written in the call, as training writes it
            # (run_t2.Sites.part_block); an enclosing scope that is not in the call (a repo or folder header) is
            # written alone, a form the compositional step does not train: the release records which each was.
            scope_cores = []
            for text in req.scope_headers:
                try:
                    sid = init_header_state(self.engine, req.messages, req.tools, text)
                    scope_writes.append('in-call')
                except ValueError:
                    sid = self.engine.native_write_text(text)['id']
                    scope_writes.append('alone')
                block = self.engine.store.get_native_state(sid).block
                scope_cores.append(self.head(init_block_values(block), kv=False, apply_dropout=False).cores)
            extra_cores = compose_site_cores(*scope_cores)
        out_dir = self.release_root / key[:16]
        result = compile_release(
            self.engine, self.head, adapter_id=self.adapter_id, init_block=init_block, basis_module=self.basis_module,
            z_kb=z_kb, k_kb=k_kb, w_kb=w_kb, kb_valid=kb_valid, extra_cores=extra_cores, out=out_dir,
            site_id=req.site_id, owner=owner,
            provenance={'cache_key': key, 'compiler_sha': req.compiler_sha, 'policy_sha': req.policy_sha,
                        'scopes': list(req.scopes), 'kb_hits': kb_hits, 'kb_generation': self._kb_generation(),
                        'compositional_scopes': len(req.scope_headers), 'scope_writes': scope_writes})
        result['cacheKey'] = key
        result['kbHits'] = kb_hits
        result['releaseDir'] = str(out_dir)
        self._cache[key] = result
        return result
