"""Checkpointable Muon for hidden matrices, with AdamW for other parameters."""
import torch


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
