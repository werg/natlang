"""Optimiser policies for the port trainer.

`adamw` is the A–F pilot lineage: one torch AdamW, its state format unchanged. `muon` follows the project's
crisp-student optimiser (MuonWithAdamW, below; scripts/training_optimizers.py re-exports it): Muon for hidden matrices, AdamW for
everything else, both children persisted with a parameter schema that is checked on load. The port version
also keeps vocabulary-sized readouts, embedding tables, control rows and single-row heads out of Muon, and
routes parameter groups added during a run (phase-F LoRA layers) to the AdamW child. A checkpoint records
its policy and is never resumed under another one.
"""

from __future__ import annotations

import math

import torch
from torch import nn

POLICIES = ("adamw", "muon")
FORMAT = "natlang.port-muon-adamw/1"
# latent optimizer -> its state key in PortMuonAdamW's checkpoint (and its schema label)
LATENT_KEYS = {"lionsr": "lion", "muonsr": "muonsr", "adaminisr": "adamini"}


def port_named_parameters(backbone, heads) -> list[tuple[str, torch.nn.Parameter]]:
    """The trainable port parameters in the trainer's order (control rows first)."""
    return [("backbone.control_rows", backbone.control_rows),
            *((f"heads.{name}", p) for name, p in heads.named_parameters() if p.requires_grad)]


def muon_eligible(name: str, parameter: torch.Tensor, vocab_size: int, embedding_ids: set[int]) -> bool:
    return (parameter.ndim == 2 and min(parameter.shape) > 1 and vocab_size not in parameter.shape
            and id(parameter) not in embedding_ids and name != "backbone.control_rows" and "lora_" not in name
            # Quantizer scales and MoE routers are not hidden-state matrices: AdamW.
            and "learned_scale" not in name and not name.endswith("_blocks") and ".mlp.gate." not in name)


class PortMuonAdamW(torch.optim.Optimizer):
    def __init__(self, named_parameters, *, lr: float, vocab_size: int, embedding_ids=(), momentum: float = 0.95,
                 ns_steps: int = 5, latent=(), latent_optimizer: str = "lionsr", latent_options=None):
        """``latent``: (name, parameter, lr, row_scale) of BF16 latents stepped in backward with one BF16 momentum and
        stochastic-rounding writes, for the latent backbone policy. ``latent_optimizer``: ``lionsr`` (per-row step
        ``lr x row_scale``, the memory-lean path of Mellum's conversion v3; ``lr`` in units of each row's ternary
        scale), ``muonsr`` (Muon per matrix, ``lr`` on AdamW's scale; the fourth element is the latent's
        ``split_rows``) or ``adaminisr`` (Adam-mini,
        ``lr`` on AdamW's scale; the fourth element is the latent's ``block_rows``, ``latent_block_rows``).
        ``latent_options``: constructor options of the latent optimizer (LionSR/AdamMiniSR: ``skip_idle``,
        ``cautious``; AdamMiniSR: ``betas``; MuonSR: ``ns``) plus ``weight_decay`` of every latent group."""
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("the muon policy needs a PyTorch build with torch.optim.Muon")
        embedding_ids = set(embedding_ids)
        self.schema, muon, auxiliary, seen = [], [], [], set()
        for name, parameter in named_parameters:
            if id(parameter) in seen:
                raise ValueError(f"duplicate trainable parameter {name}")
            seen.add(id(parameter))
            use_muon = muon_eligible(name, parameter, vocab_size, embedding_ids)
            (muon if use_muon else auxiliary).append(parameter)
            self.schema.append({"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                                "optimizer": "muon" if use_muon else "adamw"})
        latent = list(latent)
        for name, parameter, _, _ in latent:
            if id(parameter) in seen:
                raise ValueError(f"duplicate trainable parameter {name}")
            seen.add(id(parameter))
            self.schema.append({"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                                "optimizer": LATENT_KEYS.get(latent_optimizer, latent_optimizer)})
        if latent_optimizer not in LATENT_KEYS:
            raise ValueError(f"unknown latent optimizer {latent_optimizer!r}")
        self.latent_key = LATENT_KEYS[latent_optimizer]
        options = dict(latent_options or {})
        if not latent:
            self.latent = None
        elif latent_optimizer == "lionsr":
            decay = options.pop("weight_decay", 0.0)
            self.latent = LionSR([{"params": [parameter], "lr": rate, "row_scale": scale, "qat_latent_lr": rate,
                                   "weight_decay": decay, "name": name} for name, parameter, rate, scale in latent],
                                 lr=latent[0][2], fused=True, **options)
        elif latent_optimizer == "adaminisr":
            decay = options.pop("weight_decay", 0.0)
            self.latent = AdamMiniSR([{"params": [parameter], "lr": rate, "block_rows": int(rows or 1),
                                       "qat_latent_lr": rate, "weight_decay": decay, "name": name}
                                      for name, parameter, rate, rows in latent], lr=latent[0][2], **options)
        else:
            decay = options.pop("weight_decay", 0.0)
            self.latent = MuonSR([{"params": [parameter], "lr": rate, "qat_latent_lr": rate, "weight_decay": decay,
                                   "split_rows": int(split or 1), "name": name}
                                  for name, parameter, rate, split in latent], lr=latent[0][2],
                                 momentum=momentum, ns_steps=ns_steps, **options)
        if not muon:
            raise ValueError("the muon policy found no eligible hidden matrices")
        self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=0.0, momentum=momentum, ns_steps=ns_steps,
                                     adjust_lr_fn="match_rms_adamw")
        self.auxiliary = torch.optim.AdamW(auxiliary, lr=lr, weight_decay=0.0) if auxiliary else None
        self._constructed = False
        super().__init__(self._groups(), {})
        self._constructed = True

    def _groups(self):
        return (self.muon.param_groups + (self.auxiliary.param_groups if self.auxiliary else [])
                + (self.latent.param_groups if self.latent else []))

    def add_param_group(self, group: dict) -> None:
        """Groups added during a run (LoRA layers) are AdamW groups."""
        if not self._constructed:  # the base constructor registering the initial groups
            return super().add_param_group(group)
        params = list(group["params"])
        for parameter in params:
            self.schema.append({"name": None, "shape": list(parameter.shape), "dtype": str(parameter.dtype),
                                "optimizer": "adamw"})
        if self.auxiliary is None:
            self.auxiliary = torch.optim.AdamW(params, lr=group.get("lr", 1e-3), weight_decay=group.get("weight_decay", 0.0))
            self.auxiliary.param_groups[0].update({k: v for k, v in group.items() if k != "params"})
        else:
            self.auxiliary.add_param_group({**group, "params": params})
        self.param_groups = self._groups()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("PortMuonAdamW does not support closures")
        self.muon.step()
        if self.auxiliary is not None:
            self.auxiliary.step()
        if self.latent is not None:
            self.latent.step()

    def state_dict(self):
        state = {"format": FORMAT, "schema": self.schema, "muon": self.muon.state_dict(),
                 "adamw": self.auxiliary.state_dict() if self.auxiliary else None}
        if self.latent is not None:
            state[self.latent_key] = self.latent.state_dict()
        return state

    def load_state_dict(self, state_dict):
        if state_dict.get("format") != FORMAT or state_dict.get("schema") != self.schema:
            raise ValueError("port optimiser checkpoint parameter names/shapes/dtypes/partition differ")
        if (state_dict.get("adamw") is None) != (self.auxiliary is None):
            raise ValueError("port optimiser checkpoint AdamW partition differs")
        if (state_dict.get(getattr(self, "latent_key", "lion")) is None) != (self.latent is None):
            raise ValueError("port optimiser checkpoint latent optimizer partition differs")
        self.muon.load_state_dict(state_dict["muon"])
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(state_dict["adamw"])
        if self.latent is not None:
            self.latent.load_state_dict(state_dict[self.latent_key])
        # Child loads replace group dictionaries; reconnect the trainer's view of them.
        self.param_groups = self._groups()


def make_port_optimizer(policy: str, backbone, heads, lr: float, momentum: float = 0.95):
    if policy not in POLICIES:
        raise ValueError(f"unknown optimiser policy {policy!r}; expected one of {POLICIES}")
    named = port_named_parameters(backbone, heads)
    if policy == "adamw":
        return torch.optim.AdamW([p for _, p in named], lr=lr, weight_decay=0.0)
    embedding_ids = {id(p) for module in heads.modules() if isinstance(module, nn.Embedding) for p in module.parameters()}
    return PortMuonAdamW(named, lr=lr, vocab_size=int(backbone.embedding_weight.shape[0]), embedding_ids=embedding_ids,
                         momentum=momentum)


def stochastic_round_(target: torch.Tensor, value: torch.Tensor) -> None:
    """Write FP32 ``value`` into BF16 ``target`` with unbiased stochastic rounding: random low mantissa bits are added
    before truncation, so updates below BF16 resolution survive in expectation."""
    bits = value.contiguous().view(torch.int32)
    noise = torch.randint_like(bits, 0, 1 << 16)
    target.copy_(((bits + noise) & -65536).view(torch.float32))


def _lion_kernel(p, g, m, scale, active, lr, beta1, beta2, decay, cautious: bool = False, rounding: bool = True):
    """One Lion update (FP32 math) of a BF16 latent; returns (new p, new momentum). ``scale``: per-row step multiplier
    (``lr x row_scale``) or None. ``active``: a bool mask broadcastable to ``p`` (False for an idle expert matrix: no
    update, no weight decay, momentum unchanged) or None. ``cautious`` (Liang et al., arXiv:2411.16085, without the
    paper's lr renormalization): a coordinate moves only where the update's sign agrees with this gradient's sign.
    ``rounding``: stochastic-rounding write (BF16 targets); rounding an unchanged BF16 value is exact."""
    gf, mf, value = g.float(), m.float(), p.float()
    update = torch.sign(beta1 * mf + (1 - beta1) * gf)
    if cautious:
        update = update * (update * gf > 0).to(update.dtype)
    stepped = value * (1 - lr * decay) - (lr if scale is None else lr * scale) * update
    momentum = beta2 * mf + (1 - beta2) * gf
    if active is not None:
        stepped = torch.where(active, stepped, value)
        momentum = torch.where(active, momentum, mf)
    if rounding:
        bits = stepped.view(torch.int32)
        stepped = ((bits + torch.randint_like(bits, 0, 1 << 16)) & -65536).view(torch.float32)
    return stepped.to(p.dtype), momentum.to(m.dtype)


_fused_lion_chunk = torch.compile(_lion_kernel, dynamic=True)
_fused_lion_rows = torch.compile(_lion_kernel, dynamic=False)


def stack_activity(grad: torch.Tensor) -> torch.Tensor:
    """Per matrix of a stacked (ndim > 2) gradient, whether any entry is nonzero (an expert routed no tokens in this
    update has an exactly zero slice). Shape ``grad.shape[:-2]``; no host sync, no full-size temporary."""
    flat = grad.reshape(-1, grad.shape[-2] * grad.shape[-1])
    return ((flat.amax(1) != 0) | (flat.amin(1) != 0)).view(grad.shape[:-2])


class LionSR(torch.optim.Optimizer):
    """Lion (sign of interpolated momentum; one BF16 momentum buffer) for BF16 latents with stochastic-rounding
    writes: the memory-lean optimizer of Mellum's full-latent QAT (2 copies of the weights instead of 4-6). The update
    magnitude is ``lr`` per element, so for ternary latents ``lr`` is set in units of the codes' scale.

    Idle matrices of a stack (``skip_idle``, default on; 2026-10-11 optimizer review): Lion moves every coordinate by
    the full ``lr`` even where the gradient is zero (the sign of a decaying momentum stays +-1), so an expert routed no
    tokens in an update drifted at full rate. A matrix of a stacked (ndim > 2) latent whose gradient slice is exactly
    zero gets no update and no weight decay, and its momentum is left unchanged (lazy semantics: an expert's momentum
    is the history of its own gradients, so an update it took no part in neither steps nor ages it; MuonSR instead
    decays an idle expert's momentum). ``cautious``: Liang et al.'s mask (arXiv:2411.16085) without the paper's
    ``d/(nnz+1)`` lr renormalization (it about doubles the step at batch-1 noise). Both are configuration, not state:
    they are constructor arguments kept across ``load_state_dict`` (the saved state format is unchanged).
    ``idle_updates``: per stacked latent, the count of updates in which each matrix was idle (diagnostic only)."""

    def __init__(self, params, lr: float, betas=(0.9, 0.99), weight_decay: float = 0.0, chunk: int = 1 << 26,
                 fused: bool = False, skip_idle: bool = True, cautious: bool = False):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
        self.chunk, self.fused = chunk, fused
        self.skip_idle, self.cautious = skip_idle, cautious
        self.idle_updates, self.updates = {}, {}
        self._hooks, self._gated, self._armed = [], False, None

    def _activity(self, p):
        """The stacked latent's per-matrix activity mask (shape ``p.shape[:-2]``) when idle skipping applies, else None;
        counts idle matrices either way."""
        if p.ndim <= 2:
            return None
        active = stack_activity(p.grad)
        key = id(p)
        if key not in self.idle_updates:
            self.idle_updates[key] = torch.zeros(active.shape, dtype=torch.int32, device=active.device)
        self.idle_updates[key] += (~active).to(torch.int32)
        self.updates[key] = self.updates.get(key, 0) + 1
        return active if self.skip_idle else None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("LionSR does not support closures")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, group)

    def step_in_backward(self, *, gated: bool = False, spike=None):
        """Update each parameter as soon as its gradient is complete and free that gradient (each parameter must
        be used once per backward): the full-latent conversion then never holds a gradient copy of the model.

        ``spike`` (a ``SpikeClip``, with ``gated``): the stage's spike-only global clip for latents that step before
        the global norm is known. Inside ``in_backward()`` the norm accumulated so far in this update is a lower
        bound of the global norm; once it passes the clip's threshold (a multiple of the running median of recent
        global norms), each further latent's gradient is scaled by ``threshold / that bound``. An update whose global
        norm stays under the threshold (every non-spike update) is therefore never clipped, as with exact clipping;
        on a spike, latents stepped before the bound crossed the threshold step unclipped and the later ones get a
        factor at least the exact one. ``squared_norm`` is always the unclipped norm.

        ``gated``: the hooks act only inside ``with optimizer.in_backward():`` (one update's single backward); any
        other backward accumulates ``.grad`` as usual for ``step()``. A trainer whose update runs several objective
        graphs backpropagates them together under ``model.layer_staging.LayerStaging`` (layer by layer across the
        graphs; a plain summed backward would buffer one graph's gradient for the whole model), so each latent gets
        exactly one step per update with its whole gradient.
        The group is looked up when the hook fires, so learning-rate staging and ``load_state_dict`` (which
        replaces the group dictionaries) apply."""
        if self._hooks:
            raise RuntimeError("LionSR step-in-backward hooks are already registered")
        self._gated = gated
        self.spike = spike
        for index, group in enumerate(self.param_groups):
            for p in group["params"]:
                def hook(param, index=index):
                    armed = self._armed
                    if self._gated and armed is None:
                        return
                    if armed is not None:
                        if id(param) in armed.stepped:
                            name = self.param_groups[index].get("name", "?")
                            raise RuntimeError(f"latent {name} received a second gradient in one in-backward update: "
                                               "a forward ran its layer outside the layer staging "
                                               "(model/layer_staging.py tape_for_grad)")
                        # No host sync per latent: the trainer checks the total after the backward. Each tensor's
                        # norm reduces in FP32 (FP64 is ~9x slower on the GB10: ~1 s per Mellum update), the
                        # update's total accumulates in FP64.
                        squared = torch.linalg.vector_norm(param.grad, dtype=torch.float32).double().square()
                        armed.squared_norm = armed.squared_norm.to(param.device) + squared
                        armed.stepped.add(id(param))
                        if armed.threshold is not None:  # spike clip by the lower bound of the global norm so far
                            factor = (armed.threshold / armed.squared_norm.sqrt().clamp_min(1e-30)).clamp(max=1.0)
                            param.grad.mul_(factor.to(param.grad.dtype))
                            squared = squared * factor.square()
                        armed.applied_squared = armed.applied_squared.to(param.device) + squared
                    self._update(param, self.param_groups[index])
                    param.grad = None
                self._hooks.append(p.register_post_accumulate_grad_hook(hook))

    def load_state_dict(self, state_dict):
        """Also moves each group's per-row scale onto its parameter's device as an owned copy. torch's loader keeps the
        saved group values, and a scale still viewing an ``mmap=True`` checkpoint keeps the whole file mapping (and the
        pages the device copies dirtied: ~46 GB of host memory on Mellum's resume) alive for the rest of the run."""
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            scale = group.get("row_scale")
            if isinstance(scale, torch.Tensor):
                group["row_scale"] = scale.to(group["params"][0].device, copy=True)

    def in_backward(self):
        """Context of one update's single backward under ``step_in_backward(gated=True)``: inside it each latent
        steps once as its gradient completes. The value records ``stepped`` (parameter ids) and ``squared_norm``
        (FP64 squared norm of the stepped gradients, for the update's global gradient norm)."""
        if not self._gated:
            raise RuntimeError("in_backward needs step_in_backward(gated=True)")
        return _InBackward(self)

    @torch.no_grad()
    def _update(self, p, group):
        beta1, beta2 = group["betas"]
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16)
        active = self._activity(p)
        options = dict(lr=group["lr"], beta1=beta1, beta2=beta2, decay=group["weight_decay"], cautious=self.cautious,
                       rounding=p.dtype == torch.bfloat16)
        if group.get("row_scale") is not None:  # per-row step size (lr x row scale), whole tensor at once
            value, momentum = _fused_lion_rows(p, p.grad, state["momentum"], group["row_scale"].to(p.device),
                                               None if active is None else active.view(*active.shape, 1, 1), **options)
            p.copy_(value)
            state["momentum"].copy_(momentum)
            return
        kernel = _fused_lion_chunk if self.fused and p.is_cuda and p.dtype == torch.bfloat16 else _lion_kernel
        if active is None:  # flat chunks
            pieces, step = (p.view(-1), p.grad.view(-1), state["momentum"].view(-1)), self.chunk
        else:  # whole matrices per chunk, each with its activity
            width = p.shape[-2] * p.shape[-1]
            pieces = (p.view(-1, width), p.grad.view(-1, width), state["momentum"].view(-1, width))
            step, flags = max(1, self.chunk // width), active.reshape(-1, 1)
        flat_p, flat_g, flat_m = pieces
        for start in range(0, flat_p.shape[0], step):
            end = start + step
            value, momentum = kernel(flat_p[start:end], flat_g[start:end], flat_m[start:end], None,
                                     None if active is None else flags[start:end], **options)
            flat_p[start:end].copy_(value)
            flat_m[start:end].copy_(momentum)


NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)  # torch.optim.Muon's quintic (Keller Jordan's Muon)


# "cubic5" (arXiv:2606.00371): five cubic steps X <- a X + b (X X^T) X with per-step coefficients, 10 dominant
# matmuls instead of the quintic's 15, from the same Frobenius-normalized BF16 start.
CUBIC5_COEFFICIENTS = ((3.3656576, -3.3420992), (2.5744352, -1.4957376), (2.5368962, -1.4312570),
                       (2.4418906, -1.2764040), (2.2230472, -0.9630650))


def newton_schulz(update: torch.Tensor, steps: int = 5, eps: float = 1e-7, variant: str = "quintic") -> torch.Tensor:
    """torch.optim.Muon's orthogonalization (5 quintic Newton-Schulz steps in BF16) over the last two dims, batched over
    any leading dims, so each expert of a stacked [E, rows, cols] tensor is its own matrix. ``variant="cubic5"``: the
    five-step cubic iteration (``steps`` is then fixed at 5)."""
    a, b, c = NS_COEFFICIENTS
    if variant not in ("quintic", "cubic5"):
        raise ValueError(f"unknown Newton-Schulz variant {variant!r}")
    x = update.bfloat16()
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    x = x / (torch.linalg.vector_norm(x.float(), dim=(-2, -1), keepdim=True).to(x.dtype) + eps)
    if variant == "cubic5":
        for a, b in CUBIC5_COEFFICIENTS:
            x = a * x + b * ((x @ x.transpose(-2, -1)) @ x)
        return x.transpose(-2, -1) if transposed else x
    for _ in range(steps):
        gram = x @ x.transpose(-2, -1)
        x = a * x + (b * gram + c * gram @ gram) @ x
    return x.transpose(-2, -1) if transposed else x


class MuonSR(LionSR):
    """Muon for BF16 latents with LionSR's mechanics (owner 2026-10-11: the shared default optimizer for the backbone):
    one BF16 momentum, Nesterov, the update orthogonalized per matrix (``newton_schulz``; each expert of a stacked
    expert tensor separately), scaled like the shared PortMuonAdamW (``match_rms_adamw``: 0.2·sqrt(max(rows, cols)),
    so ``lr`` is on AdamW's scale), stochastic-rounding writes, stepping in backward (``step_in_backward``). A latent
    with fewer than two dims has no matrix to orthogonalize and is refused.

    Idle matrices of a stack: an expert routed no tokens in this update has a zero gradient, yet its decaying
    momentum would still get a full-size orthogonal update (noise amplified on rarely routed experts). Such a matrix
    (gradient norm at most ``idle_fraction`` of its running gradient norm; 0: exactly zero) is not stepped; its
    momentum still decays, and it gets no weight decay. ``ns``: the Newton-Schulz variant (``quintic``, or ``cubic5``:
    a third fewer matmuls), configuration kept across ``load_state_dict``. Mellum2 was pretrained and SFT'd with this
    Muon (Nesterov 0.95, 5 NS steps, 0.2 sqrt(max(m, n)) scale, wd 0.1; Mellum2 report arXiv:2605.31268)."""

    def __init__(self, params, lr: float, momentum: float = 0.95, weight_decay: float = 0.0, ns_steps: int = 5,
                 idle_fraction: float = 0.0, ns: str = "quintic"):
        torch.optim.Optimizer.__init__(self, params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                                          ns_steps=ns_steps, idle_fraction=idle_fraction))
        self.chunk, self.fused, self.ns = None, False, ns
        self._hooks, self._gated, self._armed = [], False, None
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2 or p.shape[-2] % (group.get("split_rows", 1) or 1):
                    raise ValueError("MuonSR latents are matrices (or stacks of matrices) whose rows split evenly")

    @torch.no_grad()
    def _update(self, p, group):
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16)
        beta = group["momentum"]
        g = p.grad.float()
        active = None
        if p.ndim > 2:  # a stack of matrices (experts): leave idle ones alone
            norms = torch.linalg.vector_norm(g, dim=(-2, -1))
            running = state.get("grad_norm")
            threshold = group.get("idle_fraction", 0.0) * (running if running is not None else norms)
            active = (norms > threshold).to(g.dtype).view(*norms.shape, 1, 1)
            state["grad_norm"] = norms if running is None else torch.where(norms > 0, 0.9 * running + 0.1 * norms,
                                                                             running)
        m = state["momentum"].float().mul_(beta).add_(g)
        state["momentum"].copy_(m)
        # ``split_rows``: the matrix is that many row blocks orthogonalized separately (Mellum's fused expert gate_up
        # is the gate and up projections its pretraining Muon saw as two matrices; half the Newton-Schulz cost).
        split = group.get("split_rows", 1) or 1
        rows, cols = p.shape[-2] // split, p.shape[-1]
        nesterov = g.add_(m, alpha=beta).view(*p.shape[:-2], split, rows, cols)
        direction = newton_schulz(nesterov, group["ns_steps"], variant=getattr(self, "ns", "quintic")).view(p.shape)
        if active is not None:
            direction = direction * active.to(direction.dtype)
        scale = 0.2 * math.sqrt(max(rows, cols))
        value = p.float()
        if group["weight_decay"]:
            decay = 1 - group["lr"] * group["weight_decay"]
            value.mul_(decay if active is None else 1 - (1 - decay) * active)
        value.add_(direction.float(), alpha=-group["lr"] * scale)
        if p.dtype == torch.bfloat16:
            stochastic_round_(p, value)
        else:
            p.copy_(value)


def _adamini_kernel(p, g, m, v, count, active, block_rows: int, lr, beta1, beta2, eps, decay, cautious: bool = False,
                    rounding: bool = True):
    """One Adam-mini update (FP32 math) of a latent ``p`` [..., rows, cols]: full first moment ``m`` (BF16), one
    second moment per block of ``block_rows`` consecutive rows (``v`` FP32 [..., rows // block_rows, 1]), per-matrix
    step counts ``count`` [..., 1, 1] for the bias corrections. ``active``/``cautious``/``rounding`` as in
    ``_lion_kernel``. Returns (new p, new m, new v, new count)."""
    gf, mf, value = g.float(), m.float(), p.float()
    lead, rows, cols = p.shape[:-2], p.shape[-2], p.shape[-1]
    blocks = (*lead, rows // block_rows, block_rows * cols)
    m_new = beta1 * mf + (1 - beta1) * gf
    v_new = beta2 * v + (1 - beta2) * gf.square().reshape(blocks).mean(-1, keepdim=True)
    count_new = count + 1
    if active is not None:
        count_new = torch.where(active, count_new, count)
    first = 1 - torch.pow(beta1, count_new)
    second = 1 - torch.pow(beta2, count_new)
    update = ((m_new / first).reshape(blocks) / ((v_new / second).sqrt() + eps)).reshape(p.shape)
    if cautious:
        update = update * (update * gf > 0).to(update.dtype)
    stepped = value * (1 - lr * decay) - lr * update
    if active is not None:
        stepped = torch.where(active, stepped, value)
        m_new = torch.where(active, m_new, mf)
        v_new = torch.where(active, v_new, v)
    if rounding:
        bits = stepped.view(torch.int32)
        stepped = ((bits + torch.randint_like(bits, 0, 1 << 16)) & -65536).view(torch.float32)
    return stepped.to(p.dtype), m_new.to(m.dtype), v_new, count_new


_fused_adamini = torch.compile(_adamini_kernel, dynamic=False)


def beta_for_half_life(tokens_per_update: float, half_life_tokens: float) -> float:
    """The EMA coefficient whose half-life is ``half_life_tokens`` at ``tokens_per_update`` (Marek et al.,
    arXiv:2507.07101: at small batch keep the second moment's half-life fixed in tokens, not in updates)."""
    return 0.5 ** (tokens_per_update / half_life_tokens)


class AdamMiniSR(LionSR):
    """Adam-mini (Zhang et al., arXiv:2406.16793) for BF16 latents with LionSR's mechanics: one BF16 first moment like
    LionSR's momentum, one FP32 second moment per parameter block (a group's ``block_rows`` consecutive rows: a head
    for attention q/k, one output row otherwise; a stacked expert tensor's blocks never cross experts), Adam's bias
    corrections, stochastic-rounding writes, stepping in backward. Unlike Lion's sign, the step scales with the
    gradient relative to its block's history, so weights with small or noisy gradients move little; ``lr`` is on
    AdamW's scale. Idle matrices of a stack (``skip_idle``) and ``cautious`` as in LionSR (an idle expert's moments
    and step count stay unchanged). ``betas[1]`` is usually derived from a token half-life (``beta_for_half_life``)."""

    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 skip_idle: bool = True, cautious: bool = False):
        torch.optim.Optimizer.__init__(self, params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                                                          block_rows=1))
        self.chunk, self.fused = None, False
        self.skip_idle, self.cautious = skip_idle, cautious
        self.idle_updates, self.updates = {}, {}
        self._hooks, self._gated, self._armed = [], False, None
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim < 2 or p.shape[-2] % group["block_rows"]:
                    raise ValueError("AdamMiniSR latents are matrices (or stacks) whose rows divide into blocks")

    def load_state_dict(self, state_dict):
        """torch casts floating state to the parameter's dtype on load; the FP32 second moments are put back from the
        saved values (``step`` is never cast)."""
        saved = {index: entry["v"] for index, entry in state_dict["state"].items() if "v" in entry}
        super().load_state_dict(state_dict)
        params = [p for group in self.param_groups for p in group["params"]]
        for index, value in saved.items():
            self.state[params[index]]["v"] = value.to(params[index].device, torch.float32, copy=True)

    @torch.no_grad()
    def _update(self, p, group):
        beta1, beta2 = group["betas"]
        block_rows = group.get("block_rows", 1)
        state = self.state[p]
        if "momentum" not in state:
            state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16)
            state["v"] = torch.zeros(*p.shape[:-2], p.shape[-2] // block_rows, 1, dtype=torch.float32, device=p.device)
            state["step"] = torch.zeros(*p.shape[:-2], 1, 1, dtype=torch.float32, device=p.device)
        active = self._activity(p)
        kernel = _fused_adamini if p.is_cuda else _adamini_kernel
        value, momentum, v, count = kernel(
            p, p.grad, state["momentum"], state["v"], state["step"],
            None if active is None else active.view(*active.shape, 1, 1), block_rows, lr=group["lr"], beta1=beta1,
            beta2=beta2, eps=group["eps"], decay=group["weight_decay"], cautious=self.cautious,
            rounding=p.dtype == torch.bfloat16)
        p.copy_(value)
        state["momentum"].copy_(momentum)
        state["v"].copy_(v)
        state["step"].copy_(count)


def latent_block_rows(name: str, parameter: torch.Tensor, head_dim: int | None) -> int:
    """Adam-mini's partition for one latent: a block per attention head for q/k projections (Hessian blocks per head),
    one block per output row otherwise (value, output, MLP and each expert's rows)."""
    if head_dim and any(part in name for part in (".q_proj.", ".k_proj.")) and parameter.shape[-2] % head_dim == 0:
        return head_dim
    return 1


class _InBackward:
    def __init__(self, optimizer):
        self.optimizer, self.stepped, self.squared_norm = optimizer, set(), torch.zeros((), dtype=torch.float64)
        spike = getattr(optimizer, "spike", None)
        # The spike clip's threshold for this update (from earlier updates' norms only), the same one the trainer's
        # exact clip of the other trainables uses (``SpikeClip.coefficient`` before ``observe``).
        self.threshold = spike.threshold() if spike is not None else None
        self.applied_squared = torch.zeros((), dtype=torch.float64)

    def __enter__(self):
        if self.optimizer._armed is not None:
            raise RuntimeError("in_backward contexts do not nest")
        self.optimizer._armed = self
        return self

    def __exit__(self, *exc):
        self.optimizer._armed = None
        return False

    def norm(self) -> torch.Tensor:
        """The stepped gradients' FP64 norm; raises when it is nonfinite (the latents already stepped with it)."""
        value = self.squared_norm.sqrt()
        if not torch.isfinite(value):
            raise RuntimeError("nonfinite latent gradient in the in-backward LionSR update")
        return value

    def clip_record(self, global_norm) -> dict:
        """After the update's backward: ``global_norm`` is the update's exact, unclipped global gradient norm (these
        latents' ``norm()`` with every other trainable's). The record measures the deviation from exact clipping:
        the latents' norm as applied (``latent_applied_norm``) against exact clipping's (``latent_exact_norm``)."""
        total, latent = float(global_norm), float(self.squared_norm.sqrt())
        exact = 1.0 if self.threshold is None or not total > self.threshold else self.threshold / total
        return {"threshold": self.threshold, "global_norm": total, "latent_norm": latent,
                "exact_coefficient": exact, "latent_applied_norm": float(self.applied_squared.sqrt()),
                "latent_exact_norm": latent * exact}

    @property
    def started(self) -> bool:
        """Whether any latent stepped: after that the live weights are mid-update and must not be checkpointed."""
        return bool(self.stepped)


class SpikeClip:
    """Spike-only global gradient clipping, shared by the direct and the end-to-end stage (coordinator decision
    2026-10-11): a fixed clip of 1.0 (Mellum2's SFT value) is active on every update at our gradient norms (17-915
    for batch-1 windows) and measured harmful, so an update is clipped only when its global norm exceeds ``factor``
    times the running median of the last ``WINDOW`` updates' global norms (none before ``WARMUP`` norms). The
    threshold comes from earlier updates only, so the exact clip (``coefficient``) and the in-backward latents'
    clip (``LionSR.step_in_backward(spike=...)``) use the same one. Its history is run state (``state_dict``)."""

    WINDOW, WARMUP = 64, 8

    def __init__(self, factor: float):
        if not factor > 1:
            raise ValueError("the spike clip factor is a multiple of the median norm above 1")
        self.factor, self.history = float(factor), []

    def threshold(self):
        if len(self.history) < self.WARMUP:
            return None
        ordered = sorted(self.history)
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        return self.factor * median

    def coefficient(self, norm: float) -> float:
        threshold = self.threshold()
        return 1.0 if threshold is None or not norm > threshold else threshold / norm

    def observe(self, norm: float) -> None:
        if math.isfinite(norm):
            self.history = (self.history + [float(norm)])[-self.WINDOW:]

    def state_dict(self) -> dict:
        return {"factor": self.factor, "history": list(self.history)}

    def load_state_dict(self, state) -> None:
        if state:
            self.history = [float(x) for x in state.get("history", [])][-self.WINDOW:]


MUON_MIN_SIDE = 16  # MuonSR latents: both matrix sides at least this (smaller ones: AdamW)


def latent_split(named, latent_optimizer: str):
    """The latent backbone policy's split of ``named`` backbone tensors into (latents, others): latents are weight
    matrices (and stacks of them) stepped by the latent optimizer; MoE routers (``.mlp.gate.``), vectors and, under
    MuonSR, matrices with a side under ``MUON_MIN_SIDE`` (LFM's short-conv kernels [d, 1, k]: no meaningful
    orthogonalization; Moonlight keeps non-matrix weights on Adam) are others, on AdamW at the backbone rate.
    Shared by the direct and the end-to-end stage."""
    def is_latent(n, q):
        narrow = latent_optimizer == 'muonsr' and q.ndim >= 2 and min(q.shape[-2:]) < MUON_MIN_SIDE
        return q.ndim >= 2 and '.mlp.gate.' not in n and not narrow
    named = list(named)
    return [(n, q) for n, q in named if is_latent(n, q)], [(n, q) for n, q in named if not is_latent(n, q)]


def latent_entries(latent_named, latent_optimizer: str, lr: float, head_dim: int | None = None):
    """``PortMuonAdamW``'s ``latent`` argument: (name, parameter, lr, fourth) with LionSR's per-row ternary scale,
    AdamMiniSR's ``block_rows`` or MuonSR's ``split_rows`` (Mellum's fused expert gate_up orthogonalized as its two
    matrices, gate and up, as its pretraining Muon saw them)."""
    from ..maple.qat_convert import row_scale
    return [(n, q, lr, row_scale(q) if latent_optimizer == 'lionsr' else
             latent_block_rows(n, q, head_dim) if latent_optimizer == 'adaminisr' else
             (2 if '.gate_up' in n else 1)) for n, q in latent_named]


def latent_options(latent_optimizer: str, *, weight_decay: float = 0.0, skip_idle: bool = True,
                   cautious: bool = False, ns: str = 'quintic', betas=None) -> dict:
    """``PortMuonAdamW``'s ``latent_options`` for one latent optimizer (only the options it takes)."""
    options = dict(weight_decay=weight_decay)
    if latent_optimizer in ('lionsr', 'adaminisr'):
        options.update(skip_idle=skip_idle, cautious=cautious)
    if latent_optimizer == 'muonsr':
        options.update(ns=ns)
    if latent_optimizer == 'adaminisr' and betas is not None:
        options['betas'] = tuple(betas)
    return options


def latent_partition(named, lr: float):
    """The latent policy's LionSR partition (conversion v3): every backbone weight matrix as (name, parameter,
    lr, per-row ternary scale). Its ``lr`` is in units of each row's ternary scale."""
    from ..maple.qat_convert import row_scale
    return [(n, q, lr, row_scale(q)) for n, q in named if n.startswith('backbone.') and q.ndim >= 2]


# ----------------------------------------------------------------------------------------------------------------
# Crisp-student Muon+AdamW (moved verbatim from scripts/training_optimizers.py; that module re-exports these names).
#
# Newton-Schulz: neither class implements its own; both delegate to torch.optim.Muon (same momentum/ns_steps/
# adjust_lr_fn="match_rms_adamw"), so there is nothing to deduplicate. What differs is routing, deliberately:
#   MuonWithAdamW: any requires_grad 2-D tensor whose name has no vocabulary part (lm_head/embed_tokens/embeddings/
#                  word_embeddings/wte) and whose id is not excluded. Non-trainable parameters are skipped.
#   PortMuonAdamW (muon_eligible): also excludes vocab-sized dims, single-row tensors, control rows, LoRA, quantizer
#                  scales, MoE routers, and routes groups added mid-run to AdamW. Checkpoint formats differ
#                  ("natlang.muon-adamw/1" vs FORMAT), so neither resumes under the other.
# Both routing rules stay under their own names because the golden tests pin each.


class MuonWithAdamW(torch.optim.Optimizer):
    """One scheduler-facing optimizer with both child optimizers fully persisted.

    Parameter names/shapes are checked before loading: PyTorch's native optimizer
    loaders match parameters by position and cannot detect an accidental reorder.
    """

    def __init__(self, named_parameters, *, lr, excluded_ids=(), momentum=0.95,
                 ns_steps=5):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("--optimizer muon requires a PyTorch build with torch.optim.Muon")
        excluded_ids = set(excluded_ids)
        self.schema = []
        muon, auxiliary = [], []
        seen = set()
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            if id(parameter) in seen:
                raise ValueError("Duplicate trainable parameter in optimizer")
            seen.add(id(parameter))
            vocabulary = any(part in name.split(".") for part in
                             ("lm_head", "embed_tokens", "embeddings", "word_embeddings", "wte"))
            use_muon = parameter.ndim == 2 and id(parameter) not in excluded_ids and not vocabulary
            (muon if use_muon else auxiliary).append(parameter)
            self.schema.append({"name": name, "shape": list(parameter.shape),
                                "dtype": str(parameter.dtype),
                                "optimizer": "muon" if use_muon else "adamw"})
        if not muon:
            raise ValueError("Muon requested but no eligible trainable hidden matrices were found")
        self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=0.0,
                                     momentum=momentum, ns_steps=ns_steps,
                                     adjust_lr_fn="match_rms_adamw")
        self.auxiliary = (torch.optim.AdamW(auxiliary, lr=lr, weight_decay=0.0)
                          if auxiliary else None)
        groups = self.muon.param_groups + (self.auxiliary.param_groups if self.auxiliary else [])
        super().__init__(groups, {})

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MuonWithAdamW does not support optimizer closures")
        self.muon.step()
        if self.auxiliary is not None:
            self.auxiliary.step()

    def state_dict(self):
        return {"format": "natlang.muon-adamw/1", "schema": self.schema,
                "muon": self.muon.state_dict(),
                "adamw": self.auxiliary.state_dict() if self.auxiliary else None}

    def load_state_dict(self, state_dict):
        if state_dict.get("format") != "natlang.muon-adamw/1" or state_dict.get("schema") != self.schema:
            raise ValueError("Muon checkpoint parameter names/shapes/dtypes/partition differ")
        if (state_dict.get("adamw") is None) != (self.auxiliary is None):
            raise ValueError("Muon checkpoint auxiliary optimizer partition differs")
        self.muon.load_state_dict(state_dict["muon"])
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(state_dict["adamw"])
        # Child load_state_dict replaces group dictionaries; reconnect the scheduler.
        self.param_groups = self.muon.param_groups + (self.auxiliary.param_groups if self.auxiliary else [])


def make_muon_optimizer(model, *, lr, momentum=0.95, ns_steps=5):
    vocabulary_ids = set()
    for accessor in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(model, accessor, None)
        module = getter() if callable(getter) else None
        if module is not None:
            vocabulary_ids.update(id(parameter) for parameter in module.parameters())
    return MuonWithAdamW(model.named_parameters(), lr=lr, excluded_ids=vocabulary_ids,
                         momentum=momentum, ns_steps=ns_steps)
