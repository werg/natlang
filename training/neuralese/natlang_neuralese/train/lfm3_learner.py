"""Shared LFM3 LoRA + trainable Neuralese rows learner core (`lfm3opti.md` §5, v2 §3 & §5).

Used by:
- `exp0_sft.py` (SFT: default r=128, alpha=128, lr=5e-5)
- `grpo.py` (RL: default r=32, alpha=32, lr=1e-4)
- Probes (`behaviour_probe.py`, `sanity_logprob_paths.py`, `operator_ablation.py`, `passthrough_check.py`)
- `operator_soft_states` fits and learner bench

Parameterization (`lora+rows`):
- Frozen BF16 base weights (no dense merge during training; FP8 frozen-base GEMMs on Blackwell by default).
- Canonical unscaled FP32 LoRA leaves (A ~ N(0, 1/sqrt(d_in)), B = 0) across the 9 LFM3 projection families.
- Trainable FP32 embedding/head rows for the Neuralese marker and token-override ids only.
- Frozen head, all other embeddings, norms, conv kernels, and attention gate (unless `gate_rank` <= 16).
- Resident FP32 AdamW state strictly over trainable leaves (zero frozen tensors in optimizer param groups).
"""
from __future__ import annotations

import contextlib
import dataclasses
import gc
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ..common.hashing import canonical_json_sha256_hex
from ..tune.targets import LFM3_LORA_FAMILIES, resolve_lfm3_lora_targets
from . import lfm3_fast


@dataclass(frozen=True)
class AdapterSpec:
    """Immutable LoRA + trainable Neuralese rows specification."""
    target_families: tuple[str, ...] = LFM3_LORA_FAMILIES
    rank: int = 128
    alpha: float = 128.0
    dropout: float = 0.0              # fixed 0.0 for deterministic prefix & RL parity
    gate_rank: int | None = None      # None = frozen (max <= 16 if enabled)
    dtype: str = 'float32'            # canonical trainable leaf storage dtype
    trainable_rows: bool = True       # trainable FP32 Neuralese marker & override rows

    def __post_init__(self):
        if self.rank < 1:
            raise ValueError(f'AdapterSpec.rank must be >= 1, got {self.rank}')
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError(f'AdapterSpec.alpha must be positive and finite, got {self.alpha}')
        if self.dropout != 0.0:
            raise ValueError(f'AdapterSpec.dropout must be 0.0 for prefix/RL parity, got {self.dropout}')
        if self.gate_rank is not None and not 1 <= self.gate_rank <= 16:
            raise ValueError(f'AdapterSpec.gate_rank must be in [1, 16], got {self.gate_rank}')

    @property
    def scale(self) -> float:
        return float(self.alpha) / float(self.rank)

    def to_dict(self) -> dict:
        return {
            'target_families': list(self.target_families),
            'rank': int(self.rank),
            'alpha': float(self.alpha),
            'dropout': float(self.dropout),
            'gate_rank': self.gate_rank,
            'dtype': self.dtype,
            'trainable_rows': bool(self.trainable_rows),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AdapterSpec':
        return cls(
            target_families=tuple(data.get('target_families', LFM3_LORA_FAMILIES)),
            rank=int(data['rank']),
            alpha=float(data['alpha']),
            dropout=float(data.get('dropout', 0.0)),
            gate_rank=data.get('gate_rank'),
            dtype=str(data.get('dtype', 'float32')),
            trainable_rows=bool(data.get('trainable_rows', True)),
        )

    @classmethod
    def for_sft(cls, *, rank: int = 128, alpha: float | None = None, trainable_rows: bool = True,
                gate_rank: int | None = None) -> 'AdapterSpec':
        """Default SFT recipe (lfm3opti.md §5): r=128, alpha=r, all 9 projection families + trainable Neuralese rows."""
        return cls(target_families=LFM3_LORA_FAMILIES, rank=rank, alpha=float(rank if alpha is None else alpha),
                   trainable_rows=trainable_rows, gate_rank=gate_rank)

    @classmethod
    def for_rl(cls, *, rank: int = 32, alpha: float | None = None, trainable_rows: bool = True,
               gate_rank: int | None = None) -> 'AdapterSpec':
        """Default RL recipe (lfm3opti.md §5): r=32, alpha=r, all 9 projection families + trainable Neuralese rows."""
        return cls(target_families=LFM3_LORA_FAMILIES, rank=rank, alpha=float(rank if alpha is None else alpha),
                   trainable_rows=trainable_rows, gate_rank=gate_rank)


@dataclass(frozen=True)
class LinearFactorLeaf:
    """Canonical FP32 leaf tensors. Never cache derived scaled/cast tensors here."""
    a: torch.Tensor                   # [rank, in_features], FP32 leaf
    b: torch.Tensor                   # [out_features, rank], FP32 leaf
    scale: float                      # alpha / rank (Python float, applied once)

    def detach_clone(self) -> 'LinearFactorLeaf':
        return LinearFactorLeaf(
            a=self.a.detach().clone().requires_grad_(False),
            b=self.b.detach().clone().requires_grad_(False),
            scale=float(self.scale),
        )


@dataclass(frozen=True)
class LayerAdapterBundle:
    """Flat, fixed-topology per-layer factor container passed into compiled kernels."""
    layer_idx: int
    layer_kind: Literal['attn', 'conv']
    q_proj: LinearFactorLeaf | None = None
    k_proj: LinearFactorLeaf | None = None
    v_proj: LinearFactorLeaf | None = None
    o_proj: LinearFactorLeaf | None = None
    gate_proj: LinearFactorLeaf | None = None
    in_proj: LinearFactorLeaf | None = None
    out_proj: LinearFactorLeaf | None = None
    w1: LinearFactorLeaf | None = None
    w2: LinearFactorLeaf | None = None
    w3: LinearFactorLeaf | None = None

    def iter_leaves(self):
        for field in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'in_proj', 'out_proj', 'w1', 'w2', 'w3'):
            leaf = getattr(self, field)
            if leaf is not None:
                yield field, leaf

    def detach_clone(self) -> 'LayerAdapterBundle':
        kwargs = {field: (None if getattr(self, field) is None else getattr(self, field).detach_clone())
                  for field in ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'in_proj', 'out_proj', 'w1', 'w2', 'w3')}
        return LayerAdapterBundle(layer_idx=self.layer_idx, layer_kind=self.layer_kind, **kwargs)


@dataclass(frozen=True)
class TrainableRows:
    """Trainable FP32 embedding/head rows for the Neuralese marker and token-override ids only."""
    token_ids: torch.Tensor           # [K] int64
    rows: torch.Tensor                # [K, hidden_size] FP32 leaf
    marker_ids: tuple[int, ...] = ()
    override_ids: tuple[int, ...] = ()

    def detach_clone(self) -> 'TrainableRows':
        return TrainableRows(
            token_ids=self.token_ids.detach().clone(),
            rows=self.rows.detach().clone().requires_grad_(False),
            marker_ids=self.marker_ids,
            override_ids=self.override_ids,
        )


@dataclass(frozen=True)
class EffectivePolicySpec:
    base_checkpoint_id: str
    base_weights_sha256: str
    tokenizer_sha256: str
    native_interface_sha256: str
    parameterization: Literal['lora+rows']
    adapter_spec: AdapterSpec
    policy_version: int
    effective_policy_sha256: str


class ReferenceView(tuple):
    """``(adapters, trainable_rows)`` of an active ``Lfm3Learner.reference_view``: ``adapters`` is ``()`` (bare base) or
    a snapshot bundle tuple, ``trainable_rows`` is ``False`` (start rows) or a ``TrainableRows`` snapshot."""

    def __new__(cls, adapters, trainable_rows):
        return super().__new__(cls, (adapters, trainable_rows))

    @property
    def adapters(self):
        return self[0]

    @property
    def trainable_rows(self):
        return self[1]

    @property
    def kwargs(self) -> dict:
        return {'adapters': self[0], 'trainable_rows': self[1]}


DEFAULT_SFT_LR = 5e-5
DEFAULT_RL_LR = 1e-4


def load_base_model(checkpoint, base, *, device='cuda', dtype=torch.bfloat16):
    """Load an HF LFM3 backbone from a native checkpoint with its sparse interface rows merged into the embedding.

    Drops the checkpoint's legacy ``training`` dict immediately. Returns
    ``(model, state, special_ids, marker_ids, override_ids)``.
    """
    from transformers import AutoConfig, AutoModelForCausalLM
    from ..third_party import lfm3
    from .checkpoint_policy import CheckpointPolicy

    lfm3.register()
    state = CheckpointPolicy.load(Path(checkpoint))
    config = AutoConfig.from_pretrained(base, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(base, config=config, dtype=dtype, trust_remote_code=False,
                                                 attn_implementation='sdpa')
    backbone = dict(state['backbone_state'])
    embedding = backbone['model.embed_tokens.weight']
    head = backbone.pop('lm_head.weight', None)
    if head is not None and not torch.equal(head, embedding):
        raise ValueError('untied lm_head: Lfm3Learner serves tied-embedding checkpoints only')
    if not config.tie_word_embeddings:
        raise ValueError('tied embeddings expected')
    interface = state['interface_state']
    merged = embedding.clone()
    marker_ids = tuple(int(i) for i in state['model_config']['marker_ids'])
    override_ids = tuple(int(i) for i in state['model_config'].get('token_override_ids') or ())
    sparse = [(marker_ids, interface['input_rows'])]
    if override_ids:
        sparse.append((override_ids, interface['extra_input_rows']))
    special = []
    for ids, rows in sparse:
        merged[torch.tensor(list(ids), dtype=torch.long)] = rows.to(merged.dtype)
        special.extend(ids)
    backbone['model.embed_tokens.weight'] = merged
    missing = model.load_state_dict({k: v.to(dtype) for k, v in backbone.items()}, strict=False)
    if missing.unexpected_keys or [k for k in missing.missing_keys if k != 'lm_head.weight']:
        raise ValueError(f'state mismatch {missing}')
    # Drop unused legacy optimizer state from source checkpoint to keep host RAM lean.
    trimmed_state = {k: v for k, v in state.items() if k != 'training'}
    model.to(device=device, dtype=dtype)
    return model, trimmed_state, special, marker_ids, override_ids


def _tensor_digest(hasher, name: str, tensor: torch.Tensor) -> None:
    t = tensor.detach().contiguous().cpu()
    hasher.update(name.encode('utf-8'))
    hasher.update(b'\0')
    hasher.update(str(t.dtype).encode('utf-8'))
    hasher.update(str(tuple(t.shape)).encode('utf-8'))
    hasher.update(b'\0')
    hasher.update(t.view(torch.uint8).numpy().tobytes())


class Lfm3Learner:
    """Unified `lora+rows` learner core shared across SFT, GRPO, probes and learner bench."""

    def __init__(self, model, spec: AdapterSpec, bundles: tuple[LayerAdapterBundle, ...],
                 target_map: dict[str, tuple[int, str, str]], *,
                 trainable_rows: TrainableRows | None = None,
                 state: dict | None = None,
                 base_checkpoint_path: str | None = None,
                 base_weights_sha256: str = '',
                 compile_layers: bool = True,
                 fp8_base: bool | None = None):
        self.model = model
        self.spec = spec
        self.bundles = bundles
        self.target_map = target_map            # param_name -> (layer_idx, layer_kind, short_name)
        self.trainable_rows = trainable_rows
        self.state = state
        self.base_checkpoint_path = base_checkpoint_path
        self.base_weights_sha256 = base_weights_sha256
        self.model._lfm3_adapters = self.bundles
        self.model._lfm3_trainable_rows = self.trainable_rows
        self._special_ids: tuple[int, ...] = ()
        lfm3_fast.prepare(self.model, compile_layers=compile_layers, fp8_base=fp8_base, parameterization='lora+rows')
        self.model.config.use_cache = False

    @classmethod
    def from_model(cls, model, spec: AdapterSpec, *, special_ids=(), marker_ids=(), override_ids=(),
                   state: dict | None = None, base_checkpoint_path: str | None = None,
                   base_weights_sha256: str = '', compile_layers: bool = True,
                   fp8_base: bool | None = None, seed: int = 0) -> 'Lfm3Learner':
        """Attach frozen BF16 base + FP32 LoRA leaves + optional FP32 trainable Neuralese rows onto ``model``."""
        for p in model.parameters():
            p.requires_grad_(False)
        device = model.model.embed_tokens.weight.device
        resolved = resolve_lfm3_lora_targets(model, spec.target_families, gate_rank=spec.gate_rank)
        gen = torch.Generator(device='cpu').manual_seed(int(seed))
        per_layer: dict[int, dict[str, LinearFactorLeaf]] = {i: {} for i in range(len(model.model.layers))}
        target_map: dict[str, tuple[int, str, str]] = {}
        for param_name, layer_idx, layer_kind, short_name, shape in resolved:
            out_features, in_features = shape
            rank = spec.gate_rank if short_name == 'gate_proj' and spec.gate_rank is not None else spec.rank
            scale = float(spec.alpha) / float(rank)
            if rank > min(out_features, in_features):
                raise ValueError(f'LoRA rank {rank} exceeds matrix {param_name} shape {shape}')
            a = (torch.randn((rank, in_features), generator=gen, dtype=torch.float32) / math.sqrt(in_features)).to(
                device=device).detach().requires_grad_(True)
            b = torch.zeros((out_features, rank), dtype=torch.float32, device=device, requires_grad=True)
            per_layer[layer_idx][short_name] = LinearFactorLeaf(a=a, b=b, scale=scale)
            target_map[param_name] = (layer_idx, layer_kind, short_name)
        bundles = []
        for idx, layer in enumerate(model.model.layers):
            kind: Literal['attn', 'conv'] = 'attn' if layer.is_attention_layer else 'conv'
            bundles.append(LayerAdapterBundle(layer_idx=idx, layer_kind=kind, **per_layer[idx]))
        t_rows = None
        if spec.trainable_rows and len(special_ids):
            unique_ids = list(dict.fromkeys(int(i) for i in special_ids))
            id_tensor = torch.tensor(unique_ids, dtype=torch.long, device=device)
            init_rows = model.model.embed_tokens.weight.detach().index_select(0, id_tensor).float().clone()
            init_rows.requires_grad_(True)
            t_rows = TrainableRows(
                token_ids=id_tensor,
                rows=init_rows,
                marker_ids=tuple(int(i) for i in marker_ids),
                override_ids=tuple(int(i) for i in override_ids),
            )
        learner = cls(model, spec, tuple(bundles), target_map, trainable_rows=t_rows, state=state,
                      base_checkpoint_path=base_checkpoint_path, base_weights_sha256=base_weights_sha256,
                      compile_layers=compile_layers, fp8_base=fp8_base)
        learner._special_ids = tuple(dict.fromkeys(int(i) for i in special_ids))
        return learner

    @classmethod
    def from_checkpoint(cls, checkpoint, base, spec: AdapterSpec, *, device='cuda', dtype=torch.bfloat16,
                        compile_layers: bool = True, fp8_base: bool | None = None, seed: int = 0,
                        base_weights_sha256: str = '') -> 'Lfm3Learner':
        model, state, special, marker_ids, override_ids = load_base_model(checkpoint, base, device=device, dtype=dtype)
        return cls.from_model(
            model, spec, special_ids=special, marker_ids=marker_ids, override_ids=override_ids,
            state=state, base_checkpoint_path=str(checkpoint), base_weights_sha256=base_weights_sha256,
            compile_layers=compile_layers, fp8_base=fp8_base, seed=seed,
        )

    @property
    def device(self) -> torch.device:
        return self.model.model.embed_tokens.weight.device

    @property
    def fp8_enabled(self) -> bool:
        return bool((getattr(self.model, '_lfm3_fast', None) or {}).get('fp8_base', False))

    @property
    def fp8_base(self) -> bool:
        """Whether the frozen base GEMMs run in FP8 (the value to pass as ``fp8_base=`` to ``lfm3_fast``)."""
        return self.fp8_enabled

    @property
    def checkpoint_state(self) -> dict | None:
        """The source native checkpoint (``CheckpointPolicy.load`` layout, legacy ``training`` dict dropped)."""
        return self.state

    @property
    def special_ids(self) -> list[int]:
        """The Neuralese marker then token-override ids (the source checkpoint's interface rows), in checkpoint order."""
        return list(self._special_ids)

    @property
    def native_rows(self) -> dict:
        """What a pruned native checkpoint needs from the source (``grpo.write_pruned_checkpoint``): the source backbone's
        own embedding rows at ``special_ids`` (the interface rows live in ``interface_state``), whether the head is
        tied, and the backbone tensor names."""
        original = (self.state or {}).get('backbone_state') or {}
        if 'model.embed_tokens.weight' not in original:
            raise ValueError('native_rows: the learner has no source backbone_state')
        ids = torch.tensor(self.special_ids, dtype=torch.long)
        return {'embedding_rows': original['model.embed_tokens.weight'][ids].clone(),
                'tied_head': 'lm_head.weight' in original,
                'names': [n for n in original if n != 'lm_head.weight']}

    def named_trainable_tensors(self) -> dict[str, torch.Tensor]:
        """Deterministic mapping of every FP32 trainable leaf tensor (factors + trainable Neuralese rows)."""
        out: dict[str, torch.Tensor] = {}
        for param_name in sorted(self.target_map):
            layer_idx, _kind, short_name = self.target_map[param_name]
            leaf: LinearFactorLeaf = getattr(self.bundles[layer_idx], short_name)
            out[f'lora_A.{param_name}'] = leaf.a
            out[f'lora_B.{param_name}'] = leaf.b
        if self.trainable_rows is not None:
            out['trainable_rows.rows'] = self.trainable_rows.rows
        return out

    def trainable_parameters(self) -> list[torch.Tensor]:
        """Flat list of FP32 trainable leaves in deterministic order."""
        return list(self.named_trainable_tensors().values())

    def parameter_summary(self) -> dict:
        """Exact trainable vs frozen parameter counts and resident memory footprint (GiB)."""
        adapter_params = sum(leaf.a.numel() + leaf.b.numel() for b in self.bundles for _, leaf in b.iter_leaves())
        row_params = self.trainable_rows.rows.numel() if self.trainable_rows is not None else 0
        trainable = adapter_params + row_params
        frozen = sum(p.numel() for p in self.model.parameters())
        by_family: dict[str, int] = {}
        for param_name, (layer_idx, _kind, short_name) in self.target_map.items():
            family = '.'.join(param_name.split('.')[3:5])
            leaf = getattr(self.bundles[layer_idx], short_name)
            by_family[family] = by_family.get(family, 0) + leaf.a.numel() + leaf.b.numel()
        return {
            'parameterization': 'lora+rows',
            'rank': self.spec.rank,
            'alpha': self.spec.alpha,
            'target_count': len(self.target_map),
            'adapter_params': adapter_params,
            'trainable_row_params': row_params,
            'trainable_params': trainable,
            'frozen_base_params': frozen,
            'trainable_fraction': round(trainable / max(1, frozen), 6),
            'trainable_state_gib': round(16 * trainable / 2**30, 4),  # params + grad + 2 AdamW moments in FP32
            'fp8_base': self.fp8_enabled,
            'by_family': dict(sorted(by_family.items())),
        }

    def build_optimizer(self, lr: float, *, betas=(0.9, 0.95), eps: float = 1e-8,
                        weight_decay: float = 0.0, fused: bool | None = None) -> torch.optim.AdamW:
        """Resident FP32 AdamW strictly over trainable leaves; verifies zero frozen tensors in param groups."""
        params = self.trainable_parameters()
        if not params:
            raise RuntimeError('Lfm3Learner has no trainable parameters')
        for p in params:
            if not p.requires_grad or p.dtype != torch.float32:
                raise ValueError(f'trainable leaf must be FP32 with requires_grad=True, got {p.dtype}, requires_grad={p.requires_grad}')
        frozen_ids = {id(p) for p in self.model.parameters()}
        if any(id(p) in frozen_ids for p in params):
            raise RuntimeError('frozen base parameter leaked into trainable optimizer set')
        use_fused = (self.device.type == 'cuda') if fused is None else bool(fused)
        return torch.optim.AdamW(
            [{'params': params, 'weight_decay': float(weight_decay)}],
            lr=float(lr), betas=betas, eps=float(eps), fused=use_fused,
        )

    def zero_grad(self, *, set_to_none: bool = True) -> None:
        for p in self.trainable_parameters():
            p.grad = None if set_to_none else torch.zeros_like(p)

    def family_grad_norms(self) -> dict[str, float]:
        """Per-projection-family L2 gradient norm (plus trainable_rows) for telemetry."""
        sq: dict[str, float] = {}
        for param_name, (layer_idx, _kind, short_name) in self.target_map.items():
            family = '.'.join(param_name.split('.')[3:5])
            leaf = getattr(self.bundles[layer_idx], short_name)
            acc = sq.get(family, 0.0)
            if leaf.a.grad is not None:
                acc += float(leaf.a.grad.detach().float().pow(2).sum().item())
            if leaf.b.grad is not None:
                acc += float(leaf.b.grad.detach().float().pow(2).sum().item())
            sq[family] = acc
        if self.trainable_rows is not None and self.trainable_rows.rows.grad is not None:
            sq['trainable_rows'] = float(self.trainable_rows.rows.grad.detach().float().pow(2).sum().item())
        return {k: round(math.sqrt(v), 6) for k, v in sorted(sq.items())}

    def snapshot_reference(self) -> tuple[tuple[LayerAdapterBundle, ...], TrainableRows | None]:
        """Immutable frozen FP32 snapshot of the current effective policy for KL / Ambient reference passes.

        When every B factor is zero (fresh run from base), returns an empty adapter tuple ``()`` (zero extra memory)
        alongside a detached snapshot of the starting special rows.
        """
        all_zero_b = all(
            bool((leaf.b.detach() == 0).all().item())
            for bundle in self.bundles for _, leaf in bundle.iter_leaves()
        )
        ref_bundles: tuple[LayerAdapterBundle, ...] = () if all_zero_b else tuple(b.detach_clone() for b in self.bundles)
        ref_rows = None if self.trainable_rows is None else self.trainable_rows.detach_clone()
        return ref_bundles, ref_rows

    @contextlib.contextmanager
    def reference_view(self, ref_bundles: tuple[LayerAdapterBundle, ...] | None = (),
                       ref_rows: TrainableRows | None = None):
        """Make the model compute the reference policy for the duration of the block, without touching live leaves.

        ``ref_bundles``: a ``snapshot_reference`` adapter tuple; ``()`` or ``None`` both mean the bare base (B = 0).
        ``ref_rows``: a ``TrainableRows`` snapshot; ``None`` means the start rows (the ones merged into the frozen
        embedding at load). ``None`` never means the live policy.

        While active, ``model._lfm3_adapters`` / ``model._lfm3_trainable_rows`` are the reference's, so every
        ``lfm3_fast`` call that omits ``adapters=`` / ``trainable_rows=`` (``hidden``, ``run_prefix_tree``,
        ``linear_nll``, ``token_logprobs``, probes) runs the reference. Yields a ``ReferenceView`` named tuple
        ``(adapters, trainable_rows)`` holding the explicit values (``()`` for the bare base, ``False`` for no rows)
        that may also be passed explicitly; ``view.kwargs`` is the same as a dict. Restores the live policy on exit,
        also on error."""
        adapters = () if not ref_bundles else tuple(ref_bundles)
        rows = False if ref_rows is None else ref_rows
        saved = (self.model._lfm3_adapters, self.model._lfm3_trainable_rows)
        self.model._lfm3_adapters = adapters
        self.model._lfm3_trainable_rows = None if rows is False else rows
        try:
            yield ReferenceView(adapters, rows)
        finally:
            self.model._lfm3_adapters, self.model._lfm3_trainable_rows = saved

    def effective_policy_sha256(self, base_sha256: str | None = None) -> str:
        """Deterministic SHA-256 of base digest + adapter spec + sorted FP32 factor tensors + trainable rows."""
        h = hashlib.sha256()
        h.update(canonical_json_sha256_hex({
            'base_weights_sha256': base_sha256 if base_sha256 is not None else self.base_weights_sha256,
            'adapter_spec': self.spec.to_dict(),
        }).encode('utf-8'))
        for name, tensor in sorted(self.named_trainable_tensors().items()):
            _tensor_digest(h, name, tensor)
        return h.hexdigest()

    @torch.no_grad()
    def merged_backbone_and_interface(self, *, device='cpu', dtype=torch.bfloat16) -> tuple[dict[str, torch.Tensor], dict]:
        """Construct immutable merged BF16 backbone weights ``bf16(fp32(W0) + s * B_fp32 @ A_fp32)`` and updated
        ``interface_state`` without mutating the live training model or its factors."""
        backbone: dict[str, torch.Tensor] = {}
        for name, param in self.model.state_dict().items():
            if name == 'lm_head.weight':
                continue
            if name in self.target_map:
                layer_idx, _kind, short_name = self.target_map[name]
                leaf: LinearFactorLeaf = getattr(self.bundles[layer_idx], short_name)
                merged = param.detach().float() + float(leaf.scale) * (leaf.b.detach().float() @ leaf.a.detach().float())
                backbone[name] = merged.to(device=device, dtype=dtype).contiguous()
            else:
                backbone[name] = param.detach().to(device=device, dtype=dtype).contiguous()
        interface = dict((self.state or {}).get('interface_state') or {})
        if self.trainable_rows is not None and self.trainable_rows.token_ids.numel():
            rows_cpu = self.trainable_rows.rows.detach().to(device=device, dtype=dtype)
            id_to_pos = {int(t): idx for idx, t in enumerate(self.trainable_rows.token_ids.tolist())}
            embed = backbone['model.embed_tokens.weight'].clone()
            embed[self.trainable_rows.token_ids.to(device=device)] = rows_cpu
            backbone['model.embed_tokens.weight'] = embed.contiguous()
            # Tied checkpoints hold ``output_rows = None`` / ``extra_output_rows = None`` (the head reuses the input
            # rows): only present tensors are updated, and None stays None.
            for ids, field, out_field in ((self.trainable_rows.marker_ids, 'input_rows', 'output_rows'),
                                          (self.trainable_rows.override_ids, 'extra_input_rows', 'extra_output_rows')):
                if not ids or interface.get(field) is None:
                    continue
                index = torch.tensor([id_to_pos[i] for i in ids], dtype=torch.long)
                updated = rows_cpu.index_select(0, index).contiguous()
                interface[field] = updated.to(interface[field].dtype)
                if interface.get(out_field) is not None:
                    interface[out_field] = updated.to(interface[out_field].dtype)
        return backbone, interface


def capture_rng_state() -> dict:
    """Snapshot Python, NumPy, CPU PyTorch, and CUDA RNG states for deterministic OOM retry & resume."""
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if 'python' in state:
        random.setstate(state['python'])
    if 'numpy' in state:
        np.random.set_state(state['numpy'])
    if 'torch_cpu' in state:
        torch.set_rng_state(state['torch_cpu'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['torch_cuda'])


@dataclass
class OOMRetryPlan:
    """Mutable physical execution plan reduced deterministically on ``torch.OutOfMemoryError`` (v2 §6.2).

    The plan is persistent: once shrunk it stays shrunk for the rest of the run, and trainers save ``to_dict()`` in
    their learner state and restore it with ``from_dict`` so a resume runs the same physical plan (no second OOM)."""
    pack_tokens: int
    activation_gib: float
    min_pack_tokens: int = 1
    retries: int = 0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'OOMRetryPlan':
        return cls(pack_tokens=int(data['pack_tokens']), activation_gib=float(data['activation_gib']),
                   min_pack_tokens=int(data.get('min_pack_tokens', 1)), retries=int(data.get('retries', 0)))


def run_step_with_oom_retry(step_fn, learner: Lfm3Learner, plan: OOMRetryPlan, *, max_retries: int = 8,
                            on_retry=None):
    """Execute one optimizer step body ``step_fn(pack_tokens, activation_gib)`` with deterministic OOM recovery:
    1. Snapshot RNG state.
    2. On OOM: leave the ``except`` block first (its traceback holds the failed graph's frames and tensors), then clear
       the trainable gradients, collect, empty the CUDA cache and restore the RNG; first lower ``activation_gib``
       (halve, then 0 = full checkpointing), then halve ``pack_tokens`` down to ``min_pack_tokens`` (single chain), and
       retry the exact same batch. ``on_retry(plan, attempt, error_text)`` is called before each retry.
    3. Hard abort with structured diagnostics if a single chain OOMs at full checkpointing.
    """
    entry_rng = capture_rng_state()
    attempt = 0
    while True:
        error_text = None
        try:
            return step_fn(plan.pack_tokens, plan.activation_gib)
        except torch.OutOfMemoryError as error:
            error_text = str(error)[:400]
        # outside the except clause: the exception, its traceback and the frames holding the graph are released here
        attempt += 1
        plan.retries += 1
        learner.zero_grad(set_to_none=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        restore_rng_state(entry_rng)
        if attempt > max_retries:
            raise RuntimeError(json.dumps({
                'event': 'oom_hard_abort', 'attempts': attempt, 'pack_tokens': plan.pack_tokens,
                'activation_gib': plan.activation_gib, 'error': error_text}))
        if plan.activation_gib > 0.0:
            plan.activation_gib = 0.0 if plan.activation_gib <= 4.0 else plan.activation_gib * 0.5
        elif plan.pack_tokens > plan.min_pack_tokens:
            plan.pack_tokens = max(plan.min_pack_tokens, plan.pack_tokens // 2)
        else:
            raise RuntimeError(json.dumps({
                'event': 'oom_single_chain_full_checkpoint_abort', 'attempts': attempt,
                'pack_tokens': plan.pack_tokens, 'activation_gib': plan.activation_gib, 'error': error_text}))
        if on_retry is not None:
            on_retry(plan, attempt, error_text)


def load_effective_policy(path, base, *, device='cuda', dtype=torch.bfloat16, compile_layers: bool = True,
                          fp8_base: bool = False):
    """Unified policy loader for all probes (`behaviour_probe`, `sanity_logprob_paths`, `operator_ablation`,
    `passthrough_check`). Loads either a merged native checkpoint `.pt` or an unmerged `learner_state` directory
    (attaching its FP32 LoRA factors + trainable rows so probes never evaluate the bare base)."""
    target = Path(path)
    if target.is_dir() and (target / 'learner_manifest.json').is_file():
        from .learner_state import load_learner_state
        manifest = json.loads((target / 'learner_manifest.json').read_text())
        spec = AdapterSpec.from_dict(manifest['adapter_spec'])
        base_ckpt = manifest['source_checkpoint_id']
        learner = Lfm3Learner.from_checkpoint(
            base_ckpt, base, spec, device=device, dtype=dtype, compile_layers=compile_layers,
            fp8_base=fp8_base, base_weights_sha256=manifest.get('base_weights_sha256', ''),
        )
        load_learner_state(target, learner, optimizer=None, restore_rng=False)
        learner.model.eval()
        return learner.model, learner.state, learner
    model, state, _special, _markers, _overrides = load_base_model(target, base, device=device, dtype=dtype)
    model.eval()
    lfm3_fast.prepare(model, compile_layers=compile_layers, fp8_base=fp8_base, parameterization='lora+rows')
    return model, state, None
