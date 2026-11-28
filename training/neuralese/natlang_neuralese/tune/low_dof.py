"""Low-DOF execution-grounded local adaptation over a shared Mode-P LoRA subspace.

Solves both sample efficiency AND non-overfitting for per-program / per-function adaptation:
1. Extract a shared low-rank orthonormal basis ``(U_r, V_r)`` and ``K`` global behavioral component directions
   from one or more verified execution-winning Mode-P LoRA factor exports (``adapter_factors_fp32.safetensors``).
2. Adapt a new function/program by optimizing ONLY ``K`` scalar coefficients ``c in R^K`` (e.g. 8-128 floats across
   the entire 1.7B network) using an on-policy KL-anchored advantage-weighted execution objective (NOT unconstrained
   SFT NLL), regularized by proximal L2 pull toward ``c_0`` and gated by a mandatory held-out query split.
3. Collapse ``c`` directly into standard Mode-P LoRA factors (``adapter_factors_fp32.safetensors`` +
   ``policy_manifest.json``) that SGLang, llama.cpp, wllama, and native Python already serve without engine changes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from .latent_ttt import split_support_and_query

BASIS_SCHEMA = 'natlang.low-dof-subspace-basis/1'


@dataclass
class FactorSubspaceBasis:
    """Shared per-matrix ``(U_r, V_r)`` basis and ``K`` unit-norm global behavioral components."""

    target_names: tuple[str, ...]
    left_u: dict[str, torch.Tensor]          # param_name -> [out_features, r_eff] FP32
    right_v: dict[str, torch.Tensor]         # param_name -> [in_features, r_eff] FP32
    components: dict[str, torch.Tensor]      # param_name -> [K, r_eff, r_eff] FP32
    singular_values: torch.Tensor            # [K] FP32
    donors: tuple[str, ...]

    @property
    def num_components(self) -> int:
        return int(self.singular_values.numel())

    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        tensors: dict[str, torch.Tensor] = {'singular_values': self.singular_values.detach().float().cpu().contiguous()}
        ranks: dict[str, int] = {}
        for name in self.target_names:
            u = self.left_u[name].detach().float().cpu().contiguous()
            v = self.right_v[name].detach().float().cpu().contiguous()
            c = self.components[name].detach().float().cpu().contiguous()
            tensors[f'u.{name}'] = u
            tensors[f'v.{name}'] = v
            tensors[f'comp.{name}'] = c
            ranks[name] = int(u.shape[1])
        save_file(tensors, str(out / 'subspace_basis_fp32.safetensors'))
        meta = {
            'schema': BASIS_SCHEMA,
            'num_components': self.num_components,
            'target_names': list(self.target_names),
            'ranks': ranks,
            'donors': list(self.donors),
        }
        (out / 'subspace_basis.json').write_text(json.dumps(meta, indent=1) + '\n', encoding='utf-8')
        return out

    @classmethod
    def load(cls, basis_dir: str | Path) -> 'FactorSubspaceBasis':
        path = Path(basis_dir)
        meta = json.loads((path / 'subspace_basis.json').read_text(encoding='utf-8'))
        if meta.get('schema') != BASIS_SCHEMA:
            raise ValueError(f'unexpected subspace basis schema {meta.get("schema")!r}')
        tensors = load_file(str(path / 'subspace_basis_fp32.safetensors'))
        names = tuple(meta['target_names'])
        return cls(
            target_names=names,
            left_u={n: tensors[f'u.{n}'].float() for n in names},
            right_v={n: tensors[f'v.{n}'].float() for n in names},
            components={n: tensors[f'comp.{n}'].float() for n in names},
            singular_values=tensors['singular_values'].float(),
            donors=tuple(meta.get('donors') or ()),
        )


def _load_factor_dir(factor_dir: str | Path) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor, float]], dict]:
    path = Path(factor_dir)
    manifest = json.loads((path / 'policy_manifest.json').read_text(encoding='utf-8'))
    raw = load_file(str(path / 'adapter_factors_fp32.safetensors'))
    scales = manifest.get('scales') or {}
    default_scale = float(manifest.get('adapter_spec', {}).get('alpha', 1.0)) / max(
        1.0, float(manifest.get('adapter_spec', {}).get('rank', 1.0))
    )
    entries: dict[str, tuple[torch.Tensor, torch.Tensor, float]] = {}
    for key in sorted(raw):
        if not key.startswith('lora_B.'):
            continue
        name = key.removeprefix('lora_B.')
        a_key = f'lora_A.{name}'
        if a_key not in raw:
            raise ValueError(f'{path}: missing {a_key} matching {key}')
        b = raw[key].detach().float().cpu()
        a = raw[a_key].detach().float().cpu()
        if not (torch.isfinite(b).all() and torch.isfinite(a).all()):
            raise ValueError(f'{path}: non-finite factor tensor in {name}')
        scale = float(scales.get(name, default_scale))
        entries[name] = (b, a, scale)
    if not entries:
        raise ValueError(f'{path}: no lora_B.*/lora_A.* tensors found')
    return entries, manifest


def _orthonormal_span(matrices: list[torch.Tensor], max_rank: int) -> torch.Tensor:
    """Compact orthonormal column basis (``[dim, r_eff]``) spanning the columns of ``matrices`` via thin QR + SVD."""
    stacked = torch.cat(matrices, dim=1)
    q, r = torch.linalg.qr(stacked, mode='reduced')
    u_r, s, _ = torch.linalg.svd(r, full_matrices=False)
    keep = int((s > 1e-7 * max(1.0, float(s[0].item()) if s.numel() else 0.0)).sum().item())
    k = max(1, min(int(max_rank), keep, q.shape[1]))
    return (q @ u_r[:, :k]).contiguous()


def extract_factor_basis(
    factor_dirs: list[str | Path],
    *,
    max_rank: int = 16,
    max_components: int = 32,
) -> FactorSubspaceBasis:
    """Build a shared low-DOF subspace basis from one or more verified Mode-P LoRA factor exports."""
    if not factor_dirs:
        raise ValueError('extract_factor_basis requires at least one factor directory')
    if max_rank < 1 or max_components < 1:
        raise ValueError('max_rank and max_components must be >= 1')

    loaded = [_load_factor_dir(d) for d in factor_dirs]
    target_names = tuple(sorted(set.intersection(*(set(entries.keys()) for entries, _ in loaded))))
    if not target_names:
        raise ValueError('donor factor directories share no common target parameter names')

    left_u: dict[str, torch.Tensor] = {}
    right_v: dict[str, torch.Tensor] = {}
    donor_cores: list[dict[str, torch.Tensor]] = [{} for _ in loaded]

    for name in target_names:
        b_list: list[torch.Tensor] = []
        at_list: list[torch.Tensor] = []
        ref_shape = None
        for entries, _ in loaded:
            b, a, scale = entries[name]
            shape = (b.shape[0], a.shape[1])
            if ref_shape is None:
                ref_shape = shape
            elif shape != ref_shape:
                raise ValueError(f'target {name} shape mismatch across donors: {shape} vs {ref_shape}')
            w = math.sqrt(abs(scale))
            b_list.append(b * w)
            at_list.append(a.transpose(0, 1) * w)
        u = _orthonormal_span(b_list, max_rank)
        v = _orthonormal_span(at_list, max_rank)
        left_u[name] = u
        right_v[name] = v
        for idx, (entries, _) in enumerate(loaded):
            b, a, scale = entries[name]
            # Exact projection C_d = U^T (scale * B A) V = scale * (U^T B) @ (A V)
            core = (float(scale) * ((u.transpose(0, 1) @ b) @ (a @ v))).contiguous()
            donor_cores[idx][name] = core

    # Flatten each donor's cores across all target matrices -> [D, total_core_floats]
    flat_rows = []
    slices: dict[str, tuple[int, int, tuple[int, int]]] = {}
    offset = 0
    for name in target_names:
        r_l = left_u[name].shape[1]
        r_r = right_v[name].shape[1]
        size = r_l * r_r
        slices[name] = (offset, offset + size, (r_l, r_r))
        offset += size
    for idx in range(len(loaded)):
        flat_rows.append(torch.cat([donor_cores[idx][n].reshape(-1) for n in target_names], dim=0))
    mat = torch.stack(flat_rows, dim=0)  # [D, total]
    _, s_vals, vh = torch.linalg.svd(mat, full_matrices=False)
    k = max(1, min(int(max_components), int(vh.shape[0])))
    s_keep = s_vals[:k].contiguous()
    vh_keep = vh[:k]  # [K, total], each row unit L2 norm across all layers

    components: dict[str, torch.Tensor] = {}
    for name in target_names:
        lo, hi, shape2 = slices[name]
        components[name] = vh_keep[:, lo:hi].reshape(k, *shape2).contiguous()

    return FactorSubspaceBasis(
        target_names=target_names,
        left_u=left_u,
        right_v=right_v,
        components=components,
        singular_values=s_keep,
        donors=tuple(str(Path(d).name) for d in factor_dirs),
    )


def compose_subspace_factors(
    basis: FactorSubspaceBasis,
    coeffs: torch.Tensor,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Collapse ``coeffs`` (``[K]``) into standard Mode-P ``{param_name: (lora_B, lora_A)}`` with unit scale."""
    c = coeffs.float().reshape(-1)
    if c.numel() != basis.num_components:
        raise ValueError(f'expected {basis.num_components} coefficients, got {c.numel()}')
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name in basis.target_names:
        comp = basis.components[name].to(device=c.device, dtype=torch.float32)  # [K, r_l, r_r]
        core = torch.tensordot(c, comp, dims=([0], [0]))  # [r_l, r_r]
        u = basis.left_u[name].to(device=c.device, dtype=torch.float32)
        v = basis.right_v[name].to(device=c.device, dtype=torch.float32)
        lora_b = u @ core                 # [out, r_r]
        lora_a = v.transpose(0, 1)        # [r_r, in]
        out[name] = (lora_b, lora_a)
    return out


def export_subspace_factors(
    basis: FactorSubspaceBasis,
    coeffs: torch.Tensor,
    out_dir: str | Path,
    *,
    base_manifest: dict | None = None,
) -> Path:
    """Write collapsed Mode-P LoRA factors (``adapter_factors_fp32.safetensors`` + ``policy_manifest.json``)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    factors = compose_subspace_factors(basis, coeffs.detach().cpu())
    tensors: dict[str, torch.Tensor] = {}
    scales: dict[str, float] = {}
    for name, (b, a) in factors.items():
        tensors[f'lora_B.{name}'] = b.detach().float().cpu().contiguous()
        tensors[f'lora_A.{name}'] = a.detach().float().cpu().contiguous()
        scales[name] = 1.0
    save_file(tensors, str(out / 'adapter_factors_fp32.safetensors'))
    manifest = dict(base_manifest or {})
    manifest.update({
        'schema': manifest.get('schema') or 'natlang.lfm3-lora-factors/1',
        'subspace_basis_schema': BASIS_SCHEMA,
        'subspace_coeffs': [round(float(x), 8) for x in coeffs.detach().cpu().tolist()],
        'scales': scales,
    })
    (out / 'policy_manifest.json').write_text(json.dumps(manifest, indent=1) + '\n', encoding='utf-8')
    return out


@dataclass(frozen=True)
class LowDofExecConfig:
    steps: int = 24
    lr: float = 0.05
    kl_beta: float = 0.01
    prox_lambda: float = 0.05
    clip_norm: float = 1.0
    eval_every: int = 4
    min_query_gain: float = 1e-4
    query_fraction: float = 0.25


@dataclass
class LowDofExecResult:
    accepted: bool
    best_step: int
    base_query_score: float
    best_query_score: float
    final_support_loss: float
    coeffs_best: torch.Tensor
    history: list[dict[str, float]]


def _group_advantages(examples: list[dict], device: torch.device) -> torch.Tensor:
    """Group-centered normalized advantages from each example's ``reward`` (falls back to batch centering)."""
    rewards = torch.tensor([float(ex['reward']) for ex in examples], dtype=torch.float32, device=device)
    groups: dict[str, list[int]] = {}
    for idx, ex in enumerate(examples):
        groups.setdefault(str(ex.get('group') or f'item-{idx}'), []).append(idx)
    adv = torch.zeros_like(rewards)
    multi = any(len(idxs) >= 2 for idxs in groups.values())
    if multi:
        for idxs in groups.values():
            if len(idxs) < 2:
                continue
            r = rewards[idxs]
            std = float(r.std(unbiased=False).item())
            if std > 1e-6:
                adv[idxs] = (r - r.mean()) / (std + 1e-6)
    if float(adv.abs().sum().item()) == 0.0:
        std = float(rewards.std(unbiased=False).item())
        if std > 1e-6:
            adv = (rewards - rewards.mean()) / (std + 1e-6)
    return adv


def fit_subspace_execution(
    score_fn,
    examples: list[dict],
    *,
    num_components: int,
    init_coeffs: torch.Tensor | None = None,
    cfg: LowDofExecConfig = LowDofExecConfig(),
) -> LowDofExecResult:
    """Optimize ONLY ``K`` subspace coefficients using KL-anchored execution-grounded advantages + held-query gate.

    ``score_fn(coeffs, batch) -> (policy_logps, ref_logps)`` returns length-``len(batch)`` FP32 per-episode (or
    mean-token) log-probabilities under ``coeffs`` and under the frozen reference policy.
    Every example in ``examples`` must carry ``reward`` (float) and optional ``group`` (task id).
    Returns ``init_coeffs`` unchanged (``accepted=False``) if no step improves the held-out query score by at least
    ``cfg.min_query_gain``.
    """
    support, query = split_support_and_query(examples, query_fraction=cfg.query_fraction)
    c0 = (
        torch.zeros((int(num_components),), dtype=torch.float32)
        if init_coeffs is None
        else init_coeffs.detach().float().reshape(-1).clone()
    )
    if c0.numel() != int(num_components):
        raise ValueError(f'init_coeffs length {c0.numel()} != num_components {num_components}')
    c = c0.clone().requires_grad_(True)
    opt = torch.optim.AdamW([c], lr=cfg.lr, weight_decay=0.0)

    def _eval_objective(c_val: torch.Tensor, batch: list[dict]) -> tuple[torch.Tensor, float]:
        pol_lp, ref_lp = score_fn(c_val, batch)
        adv = _group_advantages(batch, pol_lp.device)
        dlogp = pol_lp - ref_lp.detach()
        kl_k2 = 0.5 * dlogp.pow(2).mean()
        # Higher is better: positive advantages increase log-prob ratio, penalized by k2 KL drift.
        pg_score = (adv * dlogp).mean() - float(cfg.kl_beta) * kl_k2
        return pg_score, float(kl_k2.detach().item())

    with torch.no_grad():
        base_q_tensor, base_q_kl = _eval_objective(c0, query)
        base_q = float(base_q_tensor.item())

    best_q = base_q
    best_step = 0
    best_c = c0.clone()
    history: list[dict[str, float]] = [{'step': 0.0, 'query_score': base_q, 'query_kl': base_q_kl, 'prox_dist': 0.0}]
    last_sup_loss = float('nan')

    for step in range(1, cfg.steps + 1):
        opt.zero_grad(set_to_none=True)
        sup_score, sup_kl = _eval_objective(c, support)
        prox = 0.5 * float(cfg.prox_lambda) * (c - c0).pow(2).sum()
        loss = -sup_score + prox
        loss.backward()
        if cfg.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_([c], cfg.clip_norm)
        opt.step()
        last_sup_loss = float(loss.detach().item())

        if step % cfg.eval_every == 0 or step == cfg.steps:
            with torch.no_grad():
                q_tensor, q_kl = _eval_objective(c, query)
                q_score = float(q_tensor.item())
                dist = float((c - c0).norm().item())
            history.append({
                'step': float(step),
                'support_loss': last_sup_loss,
                'support_kl': sup_kl,
                'query_score': q_score,
                'query_kl': q_kl,
                'prox_dist': dist,
            })
            if q_score > best_q + cfg.min_query_gain:
                best_q = q_score
                best_step = step
                best_c = c.detach().clone()

    accepted = best_step > 0 and best_q > base_q + cfg.min_query_gain
    return LowDofExecResult(
        accepted=accepted,
        best_step=best_step,
        base_query_score=base_q,
        best_query_score=best_q,
        final_support_loss=last_sup_loss,
        coeffs_best=best_c if accepted else c0.clone(),
        history=history,
    )
