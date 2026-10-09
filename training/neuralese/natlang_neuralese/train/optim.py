"""Optimiser policies for the port trainer.

`adamw` is the A–F pilot lineage: one torch AdamW, its state format unchanged. `muon` follows the project's
crisp-student optimiser (scripts/training_optimizers.py, MuonWithAdamW): Muon for hidden matrices, AdamW for
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
                 ns_steps: int = 5):
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
        if not muon:
            raise ValueError("the muon policy found no eligible hidden matrices")
        self.muon = torch.optim.Muon(muon, lr=lr, weight_decay=0.0, momentum=momentum, ns_steps=ns_steps,
                                     adjust_lr_fn="match_rms_adamw")
        self.auxiliary = torch.optim.AdamW(auxiliary, lr=lr, weight_decay=0.0) if auxiliary else None
        self._constructed = False
        super().__init__(self._groups(), {})
        self._constructed = True

    def _groups(self):
        return self.muon.param_groups + (self.auxiliary.param_groups if self.auxiliary else [])

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

    def state_dict(self):
        return {"format": FORMAT, "schema": self.schema, "muon": self.muon.state_dict(),
                "adamw": self.auxiliary.state_dict() if self.auxiliary else None}

    def load_state_dict(self, state_dict):
        if state_dict.get("format") != FORMAT or state_dict.get("schema") != self.schema:
            raise ValueError("port optimiser checkpoint parameter names/shapes/dtypes/partition differ")
        if (state_dict.get("adamw") is None) != (self.auxiliary is None):
            raise ValueError("port optimiser checkpoint AdamW partition differs")
        self.muon.load_state_dict(state_dict["muon"])
        if self.auxiliary is not None:
            self.auxiliary.load_state_dict(state_dict["adamw"])
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


class LionSR(torch.optim.Optimizer):
    """Lion (sign of interpolated momentum; one BF16 momentum buffer) for BF16 latents with stochastic-rounding
    writes: the memory-lean optimizer of Mellum's full-latent QAT (2 copies of the weights instead of 4-6). The update
    magnitude is ``lr`` per element, so for ternary latents ``lr`` is set in units of the codes' scale."""

    def __init__(self, params, lr: float, betas=(0.9, 0.99), weight_decay: float = 0.0, chunk: int = 1 << 26):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))
        self.chunk = chunk

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("LionSR does not support closures")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "momentum" not in state:
                    state["momentum"] = torch.zeros_like(p, dtype=torch.bfloat16)
                flat_p, flat_g, flat_m = p.view(-1), p.grad.view(-1), state["momentum"].view(-1)
                for start in range(0, flat_p.numel(), self.chunk):
                    end = start + self.chunk
                    g = flat_g[start:end].float()
                    m = flat_m[start:end].float()
                    update = (beta1 * m + (1 - beta1) * g).sign_()
                    value = flat_p[start:end].float()
                    if group["weight_decay"]:
                        value.mul_(1 - group["lr"] * group["weight_decay"])
                        value.add_(update, alpha=-group["lr"])
                    else:
                        value.add_(update, alpha=-group["lr"])
                    if flat_p.dtype == torch.bfloat16:
                        stochastic_round_(flat_p[start:end], value)
                    else:
                        flat_p[start:end].copy_(value)
                    flat_m[start:end].copy_(m.mul_(beta2).add_(g, alpha=1 - beta2))
