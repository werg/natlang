"""Compile-Time Collapse: fold (B, C, A) -> standard rank-r LoRA factors and Mode-P memory -> .nzs payload.

Gives immediate 4-engine parity (managed llama-server, in-page wllama, SGLang, Python native)
for static compiled bundles (Tier 1, Tier 1.5, codebase scopes, runtime closures) without
requiring custom dynamic-HKB kernels on serving engines:
- Adapter half: B'_l = B_l @ C_l, A'_l = A_l -> exact standard rank-r LoRA keyed by HF LFM3 module path.
- Memory half (Mode P): pre-RoPE K/V tensors [M, L_attn, 2, H_kv, D_head] + per-layer gate -> positioned
  .nzs native-state payload (placed at the site's native-state import span under existing RoPE/SWA rules).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import torch

COLLAPSE_SCHEMA = "natlang.compiler.collapsed-release/1"


def provenance_key(
    *,
    site_id: str,
    header_sha: str,
    compiler_sha: str,
    policy_sha: str,
    base_gguf_sha256: str = "",
) -> str:
    """Deterministic cache key binding exact site, header, compiler, policy, and base GGUF identity."""
    payload = json.dumps(
        {
            "schema": COLLAPSE_SCHEMA,
            "site_id": str(site_id),
            "header_sha": str(header_sha),
            "compiler_sha": str(compiler_sha),
            "policy_sha": str(policy_sha),
            "base_gguf_sha256": str(base_gguf_sha256),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CollapsedMemoryBundle:
    """Mode-P positioned pre-RoPE KV tensors ready for .nzs native-state serialization."""

    memory_mode: str                   # "positioned" (Mode P, .nzs-exact)
    pre_rope_k: torch.Tensor           # [M_valid, L_attn, H_kv, D_head] FP32
    values: torch.Tensor               # [M_valid, L_attn, H_kv, D_head] FP32
    layer_gate: torch.Tensor           # [L_attn] FP32 effective tanh gate
    slot_count: int


@dataclass(frozen=True)
class CollapsedRelease:
    """Complete collapsed release artifact for one compiled site / scope stack."""

    schema: str
    cache_key: str
    site_id: str
    header_sha: str
    compiler_sha: str
    policy_sha: str
    base_gguf_sha256: str
    rank: int
    lora_factors: dict[str, tuple[torch.Tensor, torch.Tensor]]  # hf_module -> (A [r, d_in], B_eff [d_out, r])
    memory: CollapsedMemoryBundle | None


@torch.no_grad()
def collapse_to_lora_factors(
    decoder,
    c_cores: torch.Tensor,
    *,
    scale: float = 1.0,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Collapse resident (A_l, B_l) + compiled C_l into standard rank-r LoRA (A_l, B'_l = scale * B_l @ C_l).

    Args:
        decoder: NativeLFMDecoder instance wrapping LFM3 AdapterLinear layers (with adapter_names).
        c_cores: [M_modules, r, r] or [1, M_modules, r, r] middle matrices.
        scale: optional scalar multiplier applied to B'_l.
    Returns:
        dict mapping canonical HF module path (e.g. ``model.layers.0.conv.in_proj``) -> (A_fp32, B_eff_fp32).
    """
    if c_cores.ndim == 4:
        if c_cores.shape[0] != 1:
            raise ValueError(f"collapse_to_lora_factors expects batch size 1, got {c_cores.shape[0]}")
        c_cores = c_cores[0]
    if c_cores.ndim != 3:
        raise ValueError(f"Expected c_cores [M, r, r], got {tuple(c_cores.shape)}")

    adapter_names = getattr(decoder, "adapter_names", None)
    if adapter_names is None or len(adapter_names) != c_cores.shape[0]:
        raise ValueError(
            f"Decoder adapter_names ({0 if adapter_names is None else len(adapter_names)}) "
            f"must match c_cores count ({c_cores.shape[0]})"
        )

    from hindsight_kb.models.adapters import AdapterLinear

    by_id: dict[int, AdapterLinear] = {}
    for module in decoder.modules():
        if isinstance(module, AdapterLinear):
            by_id[int(module.module_id)] = module

    factors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for idx, name in enumerate(adapter_names):
        adapter = by_id[idx]
        a_fp32 = adapter.A.detach().float().contiguous()           # [r, d_in]
        b_fp32 = adapter.B.detach().float()                        # [d_out, r]
        c_fp32 = c_cores[idx].detach().float().to(b_fp32.device)   # [r, r]
        b_eff = (b_fp32 @ c_fp32) * float(scale)                   # [d_out, r]
        factors[name] = (a_fp32.cpu(), b_eff.contiguous().cpu())
    return factors


@torch.no_grad()
def collapse_mode_p_memory(
    mem_kv: torch.Tensor,
    mem_valid: torch.Tensor,
    mem_gate: torch.Tensor,
) -> CollapsedMemoryBundle:
    """Extract active Mode-P positioned pre-RoPE K/V tensors for .nzs serialization."""
    if mem_kv.ndim == 6:
        if mem_kv.shape[0] != 1:
            raise ValueError(f"collapse_mode_p_memory expects batch size 1, got {mem_kv.shape[0]}")
        mem_kv = mem_kv[0]
    if mem_valid.ndim == 2:
        mem_valid = mem_valid[0]
    valid_idx = mem_valid.to(torch.bool)
    active = mem_kv[valid_idx].detach().float().cpu()  # [M_valid, L_attn, 2, H_kv, D_head]
    gate = mem_gate.detach().float().cpu()
    return CollapsedMemoryBundle(
        memory_mode="positioned",
        pre_rope_k=active[:, :, 0].contiguous(),
        values=active[:, :, 1].contiguous(),
        layer_gate=gate.contiguous(),
        slot_count=int(active.shape[0]),
    )
