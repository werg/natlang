"""Shared LFM3 LoRA + trainable Neuralese rows learner core (`lfm3opti.md` §5, v2 §3 & §5).

Used by:
- `exp0_sft.py` (SFT: default r=128, alpha=128, lr=5e-5)
- `grpo.py` (RL: default r=32, alpha=32, lr=1e-4)
- Probes (`behaviour_probe.py`, `sanity_logprob_paths.py`, `operator_ablation.py`, `passthrough_check.py`)
- `operator_soft_states` fits and learner bench

Parameterization (`lora+rows`):
- Frozen BF16 base weights (no dense merge during training; FP8 frozen-base GEMMs on Blackwell by default).
- Canonical unscaled FP32 LoRA leaves (A ~ N(0, 1/sqrt(d_in)), B = 0) across the 9 LFM3 projection families.
- Trainable FP32 embedding/head rows for the Neuralese marker and token-override ids only, in their own optimizer group
  at ``build_optimizer(row_lr_scale=)`` x the LR, the four role markers initialised from text when they are still the
  base's generic reserved row (``from_model(marker_init=)``).
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


# The Neuralese role-marker rows start from text (AGENTS.md: every span and operator starts from text). Through sft-v9 the
# four rows were the base model's generic reserved-token row, bit-identical to each other (the pre-c9932d915 trainer
# froze them), so nothing marked where a value starts or ends, and a reader echoed them as stray tokens (DECISIONS
# 2026-11-23 "Neuralese read-back"). Each row becomes the mean embedding of a short phrase saying what it marks, its
# deviation from the mean ordinary row rescaled to an ordinary row's deviation (an ordinary token in norm and in its
# shared component, so the tied head scores it like one).
MARKER_INIT = 'text-v1'
MARKER_PHRASES = {
    '<|neuralese_produce|>': ' Write the value:',
    '<|/neuralese_produce|>': ' (value written)',
    '<|neuralese_consume|>': ' Quoted value:',
    '<|/neuralese_consume|>': ' (quote ends)',
}
# Rows that are all this close (pairwise cosine) are still the untrained reserved row; text initialisation applies to
# them only, so trained rows are never reset.
MARKER_DEGENERATE_COSINE = 0.999
# The trainable rows' learning rate over the factors': AdamW steps every coordinate by about the LR, and an embedding
# row's coordinates are ~0.02 (norm ~1 over 2048), so at the SFT LR 5e-5 a row needs ~500 consistent steps to move by
# its own size (a round is ~190); x10 makes that ~50.
ROW_LR_SCALE = 10.0


def marker_rows_degenerate(rows: torch.Tensor) -> bool:
    """Whether the marker rows are indistinguishable (every pairwise cosine above ``MARKER_DEGENERATE_COSINE``)."""
    if rows.shape[0] < 2:
        return False
    unit = torch.nn.functional.normalize(rows.detach().float(), dim=-1)
    cosine = unit @ unit.T
    off = cosine[~torch.eye(rows.shape[0], dtype=torch.bool, device=rows.device)]
    return bool((off > MARKER_DEGENERATE_COSINE).all())


@torch.no_grad()
def text_marker_rows(embedding: torch.Tensor, tokenizer, names, *, ordinary_stop: int) -> torch.Tensor:
    """FP32 rows [len(names), D] for the role markers ``names`` (``MARKER_PHRASES``): the mean embedding of each phrase's
    tokens, re-centred on the mean ordinary row (ids below ``ordinary_stop``) with its deviation scaled to the mean
    ordinary deviation."""
    total = torch.zeros(embedding.shape[1], dtype=torch.float64, device=embedding.device)
    for start in range(0, ordinary_stop, 16384):
        total += embedding[start:min(ordinary_stop, start + 16384)].double().sum(0)
    mean = (total / ordinary_stop).float()
    spread = 0.0
    for start in range(0, ordinary_stop, 16384):
        spread += float((embedding[start:min(ordinary_stop, start + 16384)].float() - mean).norm(dim=1).double().sum())
    spread /= ordinary_stop
    rows = []
    for name in names:
        if name not in MARKER_PHRASES:
            raise ValueError(f'no initialisation phrase for role marker {name!r}')
        ids = tokenizer(MARKER_PHRASES[name], add_special_tokens=False)['input_ids']
        if not ids or max(ids) >= ordinary_stop:
            raise ValueError(f'the phrase of {name!r} does not tokenize into ordinary tokens')
        deviation = embedding[torch.tensor(ids, device=embedding.device)].float().mean(0) - mean
        rows.append(mean + deviation * (spread / float(deviation.norm())))
    return torch.stack(rows)


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
    special, exact_rows = [], {}
    for ids, rows in sparse:
        merged[torch.tensor(list(ids), dtype=torch.long)] = rows.to(merged.dtype)
        special.extend(ids)
        exact_rows.update({int(i): row.detach().clone() for i, row in zip(ids, rows)})
    # the FP32 master rows a merged checkpoint carries (``master_rows``), where they still round to its interface rows
    master = state.get('trainable_rows_fp32')
    if master is not None:
        for token, row in zip(master['token_ids'].tolist(), master['rows']):
            stored = exact_rows.get(int(token))
            if stored is not None and torch.equal(row.to(stored.dtype), stored):
                exact_rows[int(token)] = row.detach().float().clone()
    exact_rows = {token: row.float() for token, row in exact_rows.items()}
    backbone['model.embed_tokens.weight'] = merged
    missing = model.load_state_dict({k: v.to(dtype) for k, v in backbone.items()}, strict=False)
    if missing.unexpected_keys or [k for k in missing.missing_keys if k != 'lm_head.weight']:
        raise ValueError(f'state mismatch {missing}')
    # Drop unused legacy optimizer state from source checkpoint to keep host RAM lean.
    trimmed_state = {k: v for k, v in state.items() if k != 'training'}
    model.to(device=device, dtype=dtype)
    # the interface rows at the checkpoint's own precision (FP32 in merged checkpoints): the learner's trainable rows
    # start from these, not from their cast into the BF16 embedding
    model._natlang_interface_rows = exact_rows
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
                   fp8_base: bool | None = None, seed: int = 0, tokenizer=None,
                   marker_names=None, marker_init: str = MARKER_INIT) -> 'Lfm3Learner':
        """Attach frozen BF16 base + FP32 LoRA leaves + optional FP32 trainable Neuralese rows onto ``model``.

        The trainable rows start from the checkpoint's interface rows at their own precision; with ``marker_init``
        'text-v1' (default; 'none' keeps them), role-marker rows that are still the base's indistinguishable reserved row start from text
        (``text_marker_rows``; ``tokenizer`` required then; ``marker_names`` in ``marker_ids`` order, default
        ``shared_state.ROLE_MARKERS``). The frozen embedding keeps the source rows, so a reference
        view of the start rows (``ref_rows=None``) is the policy that was served."""
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
            # Keep alpha/r ratio invariant when gate_rank overrides rank (otherwise alpha=128/gate_rank=8 gives 16x over-scaling).
            scale = float(spec.alpha) / float(spec.rank)
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
            exact = getattr(model, '_natlang_interface_rows', None) or {}
            for index, token in enumerate(unique_ids):
                if token in exact:
                    init_rows[index] = exact[token].to(device=device)
            markers = [unique_ids.index(int(i)) for i in marker_ids]
            if marker_init not in ('text-v1', 'none'):
                raise ValueError(f"marker_init must be 'text-v1' or 'none', got {marker_init!r}")
            if marker_init == 'text-v1' and markers and marker_rows_degenerate(init_rows[markers]):
                if tokenizer is None:
                    raise ValueError('text initialisation of the role-marker rows needs the tokenizer')
                from ..model.shared_state import ROLE_MARKERS
                names = list(marker_names or ROLE_MARKERS)
                if len(names) != len(markers):
                    raise ValueError(f'{len(markers)} role-marker ids but marker names {names}')
                init_rows[markers] = text_marker_rows(model.model.embed_tokens.weight.detach(), tokenizer, names,
                                                      ordinary_stop=min(unique_ids)).to(init_rows.device)
                model._natlang_marker_init = {'init': 'text-v1', 'markers': dict(zip(names, map(int, marker_ids)))}
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
                        base_weights_sha256: str = '', marker_init: str = MARKER_INIT) -> 'Lfm3Learner':
        model, state, special, marker_ids, override_ids = load_base_model(checkpoint, base, device=device, dtype=dtype)
        from transformers import AutoTokenizer
        return cls.from_model(
            model, spec, special_ids=special, marker_ids=marker_ids, override_ids=override_ids,
            state=state, base_checkpoint_path=str(checkpoint), base_weights_sha256=base_weights_sha256,
            compile_layers=compile_layers, fp8_base=fp8_base, seed=seed,
            tokenizer=AutoTokenizer.from_pretrained(base, trust_remote_code=False) if marker_init == 'text-v1' else None,
            marker_names=(state.get('interface_state') or {}).get('marker_names'), marker_init=marker_init,
        )

    @property
    def marker_init(self) -> dict | None:
        """{'init', 'markers'} when this learner's role-marker rows started from text (``from_model``), else None."""
        return getattr(self.model, '_natlang_marker_init', None)

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
                        weight_decay: float = 0.0, fused: bool | None = None,
                        row_lr_scale: float = ROW_LR_SCALE) -> torch.optim.AdamW:
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
        # Two groups when rows train: the factors, then the rows at ``row_lr_scale`` x the LR. ``lr_scale`` stays
        # in each group so a schedule sets ``group['lr'] = lr * group['lr_scale']``.
        rows = [] if self.trainable_rows is None else [self.trainable_rows.rows]
        factors = [p for p in params if all(p is not r for r in rows)]
        groups = [{'params': factors, 'weight_decay': float(weight_decay), 'lr': float(lr), 'lr_scale': 1.0}]
        if rows:
            scale = float(row_lr_scale)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f'row_lr_scale must be positive and finite, got {scale}')
            groups.append({'params': rows, 'weight_decay': float(weight_decay), 'lr': float(lr) * scale,
                           'lr_scale': scale})
        return torch.optim.AdamW(groups, lr=float(lr), betas=betas, eps=float(eps), fused=use_fused)

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
    def factor_tensors(self) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        """(CPU FP32 tensors in the shipped factor layout, per-matrix scale): ``lora_A.<param>``/``lora_B.<param>`` for
        every target, ``trainable_rows.rows``/``.token_ids`` when rows train. What ``merge_policy_state`` merges, the
        learner's own export and a shipped factor directory alike."""
        tensors: dict[str, torch.Tensor] = {}
        scales: dict[str, float] = {}
        for param_name in sorted(self.target_map):
            layer_idx, _kind, short_name = self.target_map[param_name]
            leaf: LinearFactorLeaf | None = getattr(self.bundles[layer_idx], short_name)
            if leaf is None:
                continue
            tensors[f'lora_A.{param_name}'] = leaf.a.detach().to(device='cpu', dtype=torch.float32).contiguous()
            tensors[f'lora_B.{param_name}'] = leaf.b.detach().to(device='cpu', dtype=torch.float32).contiguous()
            scales[param_name] = float(leaf.scale)
        if self.trainable_rows is not None:
            tensors['trainable_rows.rows'] = self.trainable_rows.rows.detach().to(device='cpu', dtype=torch.float32).contiguous()
            tensors['trainable_rows.token_ids'] = self.trainable_rows.token_ids.detach().to(device='cpu', dtype=torch.int64).contiguous()
        return tensors, scales

    def merged_backbone_and_interface(self, *, dtype=torch.bfloat16) -> tuple[dict[str, torch.Tensor], dict]:
        """The merged policy of this learner over its source checkpoint state (``merge_policy_state``)."""
        if not (self.state or {}).get('backbone_state'):
            raise ValueError('merged_backbone_and_interface: the learner has no source checkpoint state')
        tensors, scales = self.factor_tensors()
        backbone, interface, _changed = merge_policy_state(self.state, tensors, scales, dtype=dtype)
        return backbone, interface


def master_rows(tensors: dict) -> dict | None:
    """The FP32 master copy of the trainable rows a merged checkpoint carries beside its interface rows
    (``trainable_rows_fp32``: {'token_ids': int64 [K], 'rows': FP32 [K, D]}), or None without trainable rows or with
    NATLANG_MERGED_ROW_DTYPE=source. Engines load the interface rows at the checkpoint's row dtype (BF16 for LFM3: the
    native engine's bf16 profile, the SGLang and GGUF exports cast them anyway); the next learner starts its rows from
    this copy (``load_base_model``), so a round's row update is no longer rounded to BF16 between rounds."""
    import os
    choice = os.environ.get('NATLANG_MERGED_ROW_DTYPE', 'float32')
    if choice not in ('float32', 'source'):
        raise ValueError(f"NATLANG_MERGED_ROW_DTYPE must be 'float32' or 'source', not {choice!r}")
    if choice == 'source' or 'trainable_rows.rows' not in tensors:
        return None
    return {'token_ids': tensors['trainable_rows.token_ids'].detach().to('cpu', torch.int64).clone(),
            'rows': tensors['trainable_rows.rows'].detach().to('cpu', torch.float32).clone()}


def merge_policy_state(state: dict, tensors: dict, scales: dict, *, dtype=torch.bfloat16) -> tuple[dict, dict, int]:
    """The one merge of a ``lora+rows`` policy into its source native checkpoint ``state`` (async-train §2.5): the
    learner's own export (``learner_state.export_merged_checkpoint``) and a shipped factor directory
    (``learner_state.merge_factors_to_checkpoint``) both call this, so a DGX merge of shipped factors equals the
    learner's export bit for bit. On CPU in FP32: ``W = dtype(fp32(dtype(W0)) + s * fp32(B) @ fp32(A))`` per target, every
    other tensor ``dtype(W0)``; trainable rows replace the interface rows (``input_rows``/``output_rows`` at the marker
    ids, ``extra_input_rows``/``extra_output_rows`` at the override ids; a tied layout's ``*output_rows`` = None stays
    None) while the embedding keeps the source rows at those ids (the native loader merges the interface rows in);
    the head stays tied to the embedding. ``tensors``: ``lora_A.<param>``/``lora_B.<param>`` and optionally
    ``trainable_rows.rows``/``trainable_rows.token_ids``; ``scales``: ``<param>`` -> alpha / rank, required for every
    target. Returns ``(backbone, interface, changed elements)``."""
    original = state['backbone_state']
    targets = {k[len('lora_A.'):] for k in tensors if k.startswith('lora_A.')}
    missing = sorted(t for t in targets if t not in original or f'lora_B.{t}' not in tensors or t not in scales)
    if missing:
        raise ValueError(f'factor targets without a base tensor, a B factor or a scale: {missing[:4]}')
    backbone: dict[str, torch.Tensor] = {}
    changed = 0
    for name, w in original.items():
        if name == 'lm_head.weight':
            continue
        base = w.detach().to(device='cpu', dtype=dtype)
        if name in targets:
            a = tensors[f'lora_A.{name}'].to(device='cpu', dtype=torch.float32)
            b = tensors[f'lora_B.{name}'].to(device='cpu', dtype=torch.float32)
            merged = (base.float() + float(scales[name]) * (b @ a)).to(dtype).contiguous()
            changed += int((merged != base).sum())
            backbone[name] = merged
        else:
            backbone[name] = base.contiguous()
    interface = dict(state.get('interface_state') or {})
    if 'trainable_rows.rows' in tensors:
        ids = [int(i) for i in tensors['trainable_rows.token_ids'].tolist()]
        rows = tensors['trainable_rows.rows'].to(device='cpu', dtype=torch.float32)
        position = {tid: idx for idx, tid in enumerate(ids)}
        config = state.get('model_config') or {}
        groups = ((tuple(int(i) for i in config.get('marker_ids') or ()), 'input_rows', 'output_rows'),
                  (tuple(int(i) for i in config.get('token_override_ids') or ()), 'extra_input_rows', 'extra_output_rows'))
        covered = set()
        for group_ids, field, out_field in groups:
            if not group_ids or interface.get(field) is None:
                continue
            absent = [i for i in group_ids if i not in position]
            if absent:
                raise ValueError(f'trainable rows lack {field} ids {absent}')
            covered.update(group_ids)
            updated = rows.index_select(0, torch.tensor([position[i] for i in group_ids], dtype=torch.long)).contiguous()
            for key in (field, out_field):
                if interface.get(key) is not None:
                    new = updated.to(interface[key].dtype)
                    changed += int((new != interface[key]).sum())
                    interface[key] = new
        stray = sorted(set(ids) - covered)
        if stray:
            raise ValueError(f'trainable rows {stray[:4]} are no interface row of the source checkpoint')
    if 'lm_head.weight' in original or 'model.embed_tokens.weight' in backbone:
        backbone['lm_head.weight'] = backbone['model.embed_tokens.weight']
    return backbone, interface, changed


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
