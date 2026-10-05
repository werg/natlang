"""Tiny weight adapters as values (LEARNING_CONTINUUM.md §6; decision 37).

An adapter is a small coefficient tensor that changes some linear maps of the frozen backbone inside the calls
that bind it. Its structure is fully described by its **spec**, which is also its store dialect, so gradients
and optimiser states (which keep their source's dialect) stay adapters of the same shape:

    adapter/2;base=<hash12>;kind=xs;r=8;u=0;layers=6-15;targets=out,ffn_down;seed=0

Parameterisation (defaults of §14; overridable per spec). For each adapted matrix W (out × in) with top-r singular
vectors U_r (out × r) and V_r (in × r), computed once from the frozen weights:

- `xs` (LoRA-XS): ΔW = U_r R V_rᵀ with a trainable r × r matrix R per matrix; r² coefficients each.
- `tiny` (TinyLoRA): R = Σ_j v_j P_j with fixed random r × r matrices P_j (seeded) and a trainable vector v of u
  coefficients per matrix.

Singular vectors are defined only up to sign, and SVD implementations disagree (CPU and CUDA flip about half of
the top columns of these weights), while ΔW = U_r R V_rᵀ changes under a flip unless R is diagonal. `adapter/2`
fixes the signs (each column of U_r has its largest-magnitude entry positive, V_r flipped with it), so an adapter
means the same on every device and in every exported LoRA. `adapter/1` blocks keep the bases as the device's SVD
returns them (their meaning was set on the device that trained them).

Both stay in the top-r subspace, so a decoded adapter is exactly a rank-r LoRA (A = R V_rᵀ, B = U_r) that llama.cpp
and vLLM serve natively. Zero coefficients are the zero delta. `targets` name abstract roles: `out` is the
mixer's output projection (attention `out_proj` or convolution `out_proj`), `ffn_down` the feed-forward `w2`,
`ffn_up` `w1` and `w3`. Default layers are those after the sketch cutoff, so the writer's sketch path is the base
model's.

Application: forward hooks on the target modules add each active adapter's delta. Which adapters are active is a
per-row context (`active`), so a batch may mix requests with different adapters, and coefficient tensors may be
autograd leaves (gradient sessions differentiate through them).
"""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
from dataclasses import dataclass

import torch

SCHEMA = "adapter/2"
SCHEMAS = ("adapter/1", SCHEMA)  # adapter/1: legacy, device-dependent singular-vector signs
ROLES = {"out": ("self_attn.out_proj", "conv.out_proj"), "ffn_down": ("feed_forward.w2",),
         "ffn_up": ("feed_forward.w1", "feed_forward.w3")}


@dataclass(frozen=True)
class AdapterSpec:
    base: str
    kind: str = "xs"
    rank: int = 8
    dim: int = 0          # u, the coefficients per matrix of a `tiny` adapter
    layers: tuple[int, ...] = ()
    targets: tuple[str, ...] = ("out", "ffn_down")
    seed: int = 0
    version: int = 2  # 1: legacy signs (as the device's SVD returns them)

    def __post_init__(self):
        if self.kind not in ("xs", "tiny"):
            raise ValueError(f"unknown adapter kind {self.kind!r}")
        if self.kind == "tiny" and self.dim < 1:
            raise ValueError("a tiny adapter needs u >= 1 coefficients per matrix")
        if self.rank < 1 or not self.layers or not self.targets:
            raise ValueError("an adapter needs a rank, layers and targets")
        unknown = [t for t in self.targets if t not in ROLES]
        if unknown:
            raise ValueError(f"unknown adapter targets {unknown}; known: {sorted(ROLES)}")

    @property
    def width(self) -> int:
        """Coefficients per adapted matrix: one store row per matrix."""
        return self.rank * self.rank if self.kind == "xs" else self.dim

    def dialect(self) -> str:
        layers = self.layers
        span = f"{layers[0]}-{layers[-1]}" if list(layers) == list(range(layers[0], layers[-1] + 1)) else ",".join(map(str, layers))
        return (f"adapter/{self.version};base={self.base};kind={self.kind};r={self.rank};u={self.dim};layers={span};"
                f"targets={','.join(self.targets)};seed={self.seed}")

    @staticmethod
    def parse(dialect: str) -> "AdapterSpec":
        head, *fields = dialect.split("#")[0].split(";")
        if head not in SCHEMAS:
            raise ValueError(f"not an adapter dialect: {dialect!r}")
        values = dict(field.split("=", 1) for field in fields)
        span = values["layers"]
        layers = tuple(range(int(span.split("-")[0]), int(span.split("-")[1]) + 1)) if "-" in span else \
            tuple(int(x) for x in span.split(","))
        return AdapterSpec(base=values["base"], kind=values["kind"], rank=int(values["r"]), dim=int(values["u"]),
                           layers=layers, targets=tuple(values["targets"].split(",")), seed=int(values["seed"]),
                           version=int(head.split("/")[1]))


def is_adapter_dialect(dialect: str) -> bool:
    return any(dialect.split("#")[0].startswith(schema + ";") for schema in SCHEMAS)


# Active adapters: one entry per batch row, each a list of (AdapterSpec, coefficients [n_matrices, width], scale).
_ACTIVE: contextvars.ContextVar = contextvars.ContextVar("natlang_active_adapters", default=None)


@contextlib.contextmanager
def active(rows):
    """Make adapters active for the forwards inside: `rows` has one entry per batch row (a list of
    (spec, coefficients, scale)); a single entry applies to every row."""
    token = _ACTIVE.set(rows if rows and any(rows) else None)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


class AdapterBank:
    """Frozen bases of one backbone: top-r singular vectors per adapted matrix, fixed random matrices for `tiny`,
    and the forward hooks that apply active adapters."""

    @classmethod
    def of(cls, backbone) -> "AdapterBank":
        """The backbone's one bank: hooks on a module apply every active adapter, so two banks on one backbone
        would apply each adapter twice."""
        bank = getattr(backbone, "_natlang_adapter_bank", None)
        if bank is None:
            bank = cls(backbone)
            backbone._natlang_adapter_bank = bank
        return bank

    def __init__(self, backbone):
        self.backbone = backbone
        self._bases: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}
        self._projections: dict[tuple, torch.Tensor] = {}
        self._base_hash: str | None = None
        self._hooked: set[int] = set()

    # Structure --------------------------------------------------------------------------------
    def matrices(self, spec: AdapterSpec) -> list[tuple[int, str, torch.nn.Module]]:
        """(layer, module name, module) for every adapted matrix, in coefficient-row order."""
        out = []
        for layer in spec.layers:
            if not 0 <= layer < self.backbone.num_layers:
                raise ValueError(f"adapter layer {layer} outside the model")
            block = self.backbone.layers[layer]
            for role in spec.targets:
                for name in ROLES[role]:
                    module = block.get_submodule(name) if _has(block, name) else None
                    if module is not None:
                        out.append((layer, name, module))
        return out

    def base_hash(self) -> str:
        """Content hash of the frozen weights adapters can touch (every role's modules, all layers)."""
        if self._base_hash is None:
            digest = hashlib.sha256()
            for layer in range(self.backbone.num_layers):
                block = self.backbone.layers[layer]
                for names in ROLES.values():
                    for name in names:
                        if _has(block, name):
                            weight = _weight(block.get_submodule(name))
                            digest.update(name.encode() + str(layer).encode())
                            digest.update(weight.detach().float().cpu().numpy().tobytes())
            self._base_hash = digest.hexdigest()[:12]
        return self._base_hash

    def spec(self, kind: str = "xs", rank: int = 8, dim: int = 0, layers=None, targets=("out", "ffn_down"),
             seed: int = 0, cutoff: int | None = None) -> AdapterSpec:
        """A spec for this backbone; default layers are those from `cutoff` (the sketch cutoff) to the top."""
        if layers is None:
            layers = range(cutoff or 0, self.backbone.num_layers)
        return AdapterSpec(base=self.base_hash(), kind=kind, rank=rank, dim=dim, layers=tuple(layers),
                           targets=tuple(targets), seed=seed)

    def zeros(self, spec: AdapterSpec) -> torch.Tensor:
        return torch.zeros(len(self.matrices(spec)), spec.width)

    def check(self, spec: AdapterSpec, coefficients: torch.Tensor):
        if spec.base != self.base_hash():
            raise ValueError(f"adapter for base {spec.base} cannot run on base {self.base_hash()}")
        expected = (len(self.matrices(spec)), spec.width)
        if tuple(coefficients.shape) != expected:
            raise ValueError(f"adapter coefficients have shape {tuple(coefficients.shape)}, expected {expected}")

    # Bases ------------------------------------------------------------------------------------
    def bases(self, layer: int, name: str, module, rank: int, version: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        key = (layer, name, rank, version)
        if key not in self._bases:
            # Normal tensors even when first needed under inference mode: gradient sessions save them for backward.
            with torch.inference_mode(False), torch.no_grad():
                weight = _weight(module).detach().float().clone()
                u, _, vh = torch.linalg.svd(weight, full_matrices=False)
                u_r, v_r = u[:, :rank], vh[:rank].t()  # U_r [out, r], V_r [in, r]
                if version >= 2:
                    signs = torch.sign(u_r.gather(0, u_r.abs().argmax(0, keepdim=True))).clamp_min(0) * 2 - 1
                    u_r, v_r = u_r * signs, v_r * signs
                self._bases[key] = (u_r.contiguous(), v_r.contiguous())
        return self._bases[key]

    def projections(self, spec: AdapterSpec, row: int, device) -> torch.Tensor:
        """[u, r, r] fixed random matrices of a `tiny` adapter's row, scaled so R has unit-scale entries for unit v."""
        key = (spec.seed, spec.dim, spec.rank, row, str(device))
        if key not in self._projections:
            seed = int.from_bytes(hashlib.sha256(repr(key[:4]).encode()).digest()[:8], "little") % (2**31)
            with torch.inference_mode(False):
                generator = torch.Generator().manual_seed(seed)
                matrices = torch.randn(spec.dim, spec.rank, spec.rank, generator=generator) / spec.dim ** 0.5
                self._projections[key] = matrices.to(device)
        return self._projections[key]

    def core(self, spec: AdapterSpec, coefficients: torch.Tensor, row: int) -> torch.Tensor:
        """R (r × r) of one adapted matrix."""
        c = coefficients[row].float()
        if spec.kind == "xs":
            return c.view(spec.rank, spec.rank)
        return torch.einsum("j,jab->ab", c, self.projections(spec, row, c.device))

    def lora(self, spec: AdapterSpec, coefficients: torch.Tensor) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        """The adapter as rank-r LoRA factors per module (`model.layers.<l>.<name>` → (A [r, in], B [out, r]))."""
        out = {}
        for row, (layer, name, module) in enumerate(self.matrices(spec)):
            u_r, v_r = self.bases(layer, name, module, spec.rank, spec.version)
            core = self.core(spec, coefficients, row).to(u_r.device)
            out[f"model.layers.{layer}.{name}"] = (core @ v_r.t(), u_r)
        return out

    # Application ------------------------------------------------------------------------------
    def install(self, spec: AdapterSpec):
        """Register the hooks for a spec's modules (idempotent)."""
        for layer, name, module in self.matrices(spec):
            if id(module) in self._hooked:
                continue
            module.register_forward_hook(self._hook(layer, name))
            self._hooked.add(id(module))

    def _hook(self, layer: int, name: str):
        def hook(module, inputs, output):
            rows = _ACTIVE.get()
            if rows is None:
                return None
            x = inputs[0]
            delta = None
            per_row = len(rows) > 1
            if per_row and len(rows) != x.shape[0]:
                raise RuntimeError(f"{len(rows)} adapter rows for a batch of {x.shape[0]}")
            groups: dict[int, list[int]] = {}
            for index, entry in enumerate(rows if per_row else rows[:1]):
                if entry:
                    groups.setdefault(id(entry), []).append(index)
            for indices in groups.values():
                entry = rows[indices[0]] if per_row else rows[0]
                for spec, coefficients, scale in entry:
                    row = self._row(spec, layer, name)
                    if row is None:
                        continue
                    u_r, v_r = self.bases(layer, name, module, spec.rank, spec.version)
                    core = self.core(spec, coefficients.to(x.device), row)
                    xs = x[indices] if per_row else x
                    change = (((xs.float() @ v_r) @ core.t()) @ u_r.t()) * scale
                    if per_row:
                        full = torch.zeros(output.shape, dtype=change.dtype, device=x.device)
                        full[indices] = change
                        change = full
                    delta = change if delta is None else delta + change
            return output if delta is None else output + delta.to(output.dtype)
        return hook

    def _row(self, spec: AdapterSpec, layer: int, name: str) -> int | None:
        key = ("row", spec.dialect(), layer, name)
        if key not in self._bases:
            rows = {(l, n): i for i, (l, n, _) in enumerate(self.matrices(spec))}
            self._bases[key] = rows.get((layer, name))  # type: ignore[assignment]
        return self._bases[key]  # type: ignore[return-value]


def _has(block, name: str) -> bool:
    try:
        block.get_submodule(name)
        return True
    except AttributeError:
        return False


def _weight(module) -> torch.Tensor:
    """The frozen weight of a target, through a PEFT LoRA wrapper if one was injected."""
    base = getattr(module, "base_layer", module)
    return base.weight
