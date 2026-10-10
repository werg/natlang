"""Optimiser policies for the port trainer.

`adamw` is the A–F pilot lineage: one torch AdamW, its state format unchanged. `muon` follows the project's
crisp-student optimiser (MuonWithAdamW, below; scripts/training_optimizers.py re-exports it): Muon for hidden matrices, AdamW for
everything else, both children persisted with a parameter schema that is checked on load. The port version
also keeps vocabulary-sized readouts, embedding tables, control rows and single-row heads out of Muon, and
routes parameter groups added during a run (phase-F LoRA layers) to the AdamW child. A checkpoint records
its policy and is never resumed under another one.
"""

from __future__ import annotations

import torch
from torch import nn

POLICIES = ("adamw", "muon")
FORMAT = "natlang.port-muon-adamw/1"


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
                 ns_steps: int = 5, latent=()):
        """``latent``: (name, parameter, lr, row_scale) of BF16 latents stepped by ``LionSR`` (one BF16 momentum,
        stochastic rounding, per-row step ``lr x row_scale``): the memory-lean path of Mellum's conversion v3, for
        the latent backbone policy. Their ``lr`` is in units of each row's ternary scale."""
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
                                "optimizer": "lion"})
        self.latent = (LionSR([{"params": [parameter], "lr": rate, "row_scale": scale, "qat_latent_lr": rate,
                                "weight_decay": 0.0, "name": name} for name, parameter, rate, scale in latent],
                              lr=latent[0][2], fused=True) if latent else None)
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
            state["lion"] = self.latent.state_dict()
        return state

    def load_state_dict(self, state_dict):
        if state_dict.get("format") != FORMAT or state_dict.get("schema") != self.schema:
            raise ValueError("port optimiser checkpoint parameter names/shapes/dtypes/partition differ")
        if (state_dict.get("adamw") is None) != (self.auxiliary is None):
            raise ValueError("port optimiser checkpoint AdamW partition differs")
        if (state_dict.get("lion") is None) != (self.latent is None):
            raise ValueError("port optimiser checkpoint LionSR latent partition differs")
        self.muon.load_state_dict(state_dict["muon"])
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(state_dict["adamw"])
        if self.latent is not None:
            self.latent.load_state_dict(state_dict["lion"])
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


def _lion_chunk(p, g, m, lr, beta1, beta2, decay):
    """One fused Lion update of a BF16 chunk with stochastic rounding; returns (new p, new momentum)."""
    gf, mf, value = g.float(), m.float(), p.float()
    value = value * (1 - lr * decay) - lr * torch.sign(beta1 * mf + (1 - beta1) * gf)
    bits = value.view(torch.int32)
    rounded = ((bits + torch.randint_like(bits, 0, 1 << 16)) & -65536).view(torch.float32)
    return rounded.to(p.dtype), (beta2 * mf + (1 - beta2) * gf).to(m.dtype)


_fused_lion_chunk = torch.compile(_lion_chunk, dynamic=True)


def _lion_rows(p, g, m, scale, lr, beta1, beta2, decay):
    gf, mf, value = g.float(), m.float(), p.float()
    value = value * (1 - lr * decay) - (lr * scale) * torch.sign(beta1 * mf + (1 - beta1) * gf)
    if p.dtype != torch.bfloat16:  # stochastic rounding only for BF16 latents
        return value.to(p.dtype), (beta2 * mf + (1 - beta2) * gf).to(m.dtype)
    bits = value.view(torch.int32)
    rounded = ((bits + torch.randint_like(bits, 0, 1 << 16)) & -65536).view(torch.float32)
    return rounded.to(p.dtype), (beta2 * mf + (1 - beta2) * gf).to(m.dtype)


_fused_lion_rows = torch.compile(_lion_rows, dynamic=False)


class LionSR(torch.optim.Optimizer):
    """Lion (sign of interpolated momentum; one BF16 momentum buffer) for BF16 latents with stochastic-rounding
    writes: the memory-lean optimizer of Mellum's full-latent QAT (2 copies of the weights instead of 4-6). The update
    magnitude is ``lr`` per element, so for ternary latents ``lr`` is set in units of the codes' scale."""

    def __init__(self, params, lr: float, betas=(0.9, 0.99), weight_decay: float = 0.0, chunk: int = 1 << 26,
                 fused: bool = False):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
        self.chunk, self.fused = chunk, fused
        self._hooks, self._gated, self._armed = [], False, None

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("LionSR does not support closures")
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    self._update(p, group)

    def step_in_backward(self, *, gated: bool = False):
        """Update each parameter as soon as its gradient is complete and free that gradient (each parameter must
        be used once per backward): the full-latent conversion then never holds a gradient copy of the model.

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
        for index, group in enumerate(self.param_groups):
            for p in group["params"]:
                def hook(param, index=index):
                    armed = self._armed
                    if self._gated and armed is None:
                        return
                    if armed is not None:
                        if id(param) in armed.stepped:
                            raise RuntimeError("a LionSR latent received a second gradient in one in-backward update; "
                                               "run the update's objectives as one backward")
                        # No host sync per latent: the trainer checks the total after the backward. Each tensor's
                        # norm reduces in FP32 (FP64 is ~9x slower on the GB10: ~1 s per Mellum update), the
                        # update's total accumulates in FP64.
                        armed.squared_norm = (armed.squared_norm.to(param.device) + torch.linalg.vector_norm(
                            param.grad, dtype=torch.float32).double().square())
                        armed.stepped.add(id(param))
                    self._update(param, self.param_groups[index])
                    param.grad = None
                self._hooks.append(p.register_post_accumulate_grad_hook(hook))

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
        if group.get("row_scale") is not None:  # per-row step size (lr x row scale), whole tensor at once
            value, momentum = _fused_lion_rows(p, p.grad, state["momentum"], group["row_scale"].to(p.device),
                                               group["lr"], beta1, beta2, group["weight_decay"])
            p.copy_(value)
            state["momentum"].copy_(momentum)
            return
        flat_p, flat_g, flat_m = p.view(-1), p.grad.view(-1), state["momentum"].view(-1)
        if self.fused and p.is_cuda and p.dtype == torch.bfloat16:
            for start in range(0, flat_p.numel(), self.chunk):
                end = start + self.chunk
                value, momentum = _fused_lion_chunk(flat_p[start:end], flat_g[start:end], flat_m[start:end],
                                                    group["lr"], beta1, beta2, group["weight_decay"])
                flat_p[start:end].copy_(value)
                flat_m[start:end].copy_(momentum)
            return
        for start in range(0, flat_p.numel(), self.chunk):
            end = start + self.chunk
            g = flat_g[start:end].float()
            m = flat_m[start:end].float()
            update = (beta1 * m + (1 - beta1) * g).sign_()
            value = flat_p[start:end].float()
            if group["weight_decay"]:
                value.mul_(1 - group["lr"] * group["weight_decay"])
            value.add_(update, alpha=-group["lr"])
            if flat_p.dtype == torch.bfloat16:
                stochastic_round_(flat_p[start:end], value)
            else:
                flat_p[start:end].copy_(value)
            flat_m[start:end].copy_(m.mul_(beta2).add_(g, alpha=1 - beta2))


class _InBackward:
    def __init__(self, optimizer):
        self.optimizer, self.stepped, self.squared_norm = optimizer, set(), torch.zeros((), dtype=torch.float64)

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

    @property
    def started(self) -> bool:
        """Whether any latent stepped: after that the live weights are mid-update and must not be checkpointed."""
        return bool(self.stepped)


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
