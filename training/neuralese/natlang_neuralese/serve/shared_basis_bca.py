"""Shared-basis ``B C A`` per-request adapter dispatcher for concurrent multi-site serving.

All Mode-P compiled sites (``CompiledSiteHead``) and low-DOF adaptations (``tune.low_dof``) share ONE frozen
``(B, A)`` basis per target matrix and differ only by a tiny ``[r, r]`` core matrix ``C`` per request/site.
Keeping ``(B, A)`` resident once and dispatching per-request ``C`` eliminates dynamic multi-slot LoRA swapping
(which crashes SGLang above 2 concurrent adapted requests) and scales to hundreds of concurrent compiled sites
at negligible VRAM overhead.
"""
from __future__ import annotations

import torch


class SharedBasisBCADispatcher:
    """Resident shared ``(left_u, right_v)`` basis applying per-request ``[r, r]`` cores ``C`` to batched activations."""

    def __init__(self, left_u: dict[str, torch.Tensor], right_v: dict[str, torch.Tensor], *, device='cpu'):
        if set(left_u.keys()) != set(right_v.keys()) or not left_u:
            raise ValueError('left_u and right_v must share a non-empty set of target names')
        self.device = torch.device(device)
        self.left_u = {k: v.detach().to(device=self.device, dtype=torch.float32).contiguous() for k, v in left_u.items()}
        self.right_v = {k: v.detach().to(device=self.device, dtype=torch.float32).contiguous() for k, v in right_v.items()}
        self._cores: dict[str, dict[str, torch.Tensor]] = {}

    def register_site(self, site_id: str, cores: dict[str, torch.Tensor]) -> None:
        """Register (or replace) a site's ``[r_l, r_r]`` cores keyed by target name."""
        stored: dict[str, torch.Tensor] = {}
        for name, core in cores.items():
            if name not in self.left_u:
                continue
            r_l = self.left_u[name].shape[1]
            r_r = self.right_v[name].shape[1]
            stored[name] = core.detach().to(device=self.device, dtype=torch.float32).reshape(r_l, r_r).contiguous()
        self._cores[str(site_id)] = stored

    def unregister_site(self, site_id: str) -> None:
        self._cores.pop(str(site_id), None)

    @property
    def registered_sites(self) -> tuple[str, ...]:
        return tuple(sorted(self._cores.keys()))

    def apply_delta(
        self,
        target_name: str,
        x: torch.Tensor,
        site_ids: list[str | None],
        *,
        scales: list[float] | None = None,
    ) -> torch.Tensor:
        """Compute batched per-request delta ``y_b = scale_b * x_b @ V @ C_b^T @ U^T`` for ``x`` of shape ``[B, ..., in]``.

        Requests whose ``site_id`` is ``None`` or unregistered receive exact zero delta (base model).
        Uses ``hindsight_kb.kernels.fused_bca_linear`` on CUDA when available, otherwise exact FP32 PyTorch matmuls.
        """
        if target_name not in self.left_u:
            return torch.zeros((*x.shape[:-1], x.shape[-1]), dtype=x.dtype, device=x.device)
        b = x.shape[0]
        if len(site_ids) != b:
            raise ValueError(f'batch size {b} != len(site_ids) {len(site_ids)}')
        u = self.left_u[target_name].to(device=x.device)   # [out, r_l]
        v = self.right_v[target_name].to(device=x.device)  # [in, r_r]
        r_l, r_r = u.shape[1], v.shape[1]
        out_dim = u.shape[0]

        c_batch = torch.zeros((b, r_l, r_r), dtype=torch.float32, device=x.device)
        any_active = False
        for i, sid in enumerate(site_ids):
            if sid is None:
                continue
            site_map = self._cores.get(str(sid))
            if site_map is None or target_name not in site_map:
                continue
            sc = 1.0 if scales is None else float(scales[i])
            if sc != 0.0:
                c_batch[i] = site_map[target_name].to(device=x.device) * sc
                any_active = True

        if not any_active:
            return torch.zeros((*x.shape[:-1], out_dim), dtype=x.dtype, device=x.device)

        # Exact factored contraction: h = x @ V ([B, ..., r_r]), z = h @ C^T ([B, ..., r_l]), delta = z @ U^T ([B, ..., out])
        orig_shape = x.shape[:-1]
        xf = x.reshape(b, -1, x.shape[-1]).float()  # [B, T, in]
        h = torch.matmul(xf, v)                     # [B, T, r_r]
        z = torch.bmm(h, c_batch.transpose(1, 2))   # [B, T, r_l]
        delta = torch.matmul(z, u.transpose(0, 1))  # [B, T, out]
        return delta.reshape(*orig_shape, out_dim).to(dtype=x.dtype)
