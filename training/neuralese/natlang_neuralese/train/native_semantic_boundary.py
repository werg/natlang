"""Exact read-only boundary for a whole multi-model objective qualification."""
from __future__ import annotations
import copy
from ..eval.native_memory import _optimizer_versions, CapacityStateChangedError
from .loop import capture_training_rng_state, restore_training_rng_state


class ObjectiveBoundary:
    """Small state snapshots plus versions; no model/moment/gradient clones."""
    def __init__(self, participants, vae, optimizer, device):
        self.participants, self.vae, self.device = participants, vae, device
        self.rng = capture_training_rng_state(device)
        self.named = [(name+'.'+key,p) for name,entry in sorted(participants.items())
                      for key,p in entry.model.named_parameters()]
        self.named += [('vae.'+key,p) for key,p in vae.named_parameters()]
        if any(p.grad is not None for _,p in self.named):
            raise ValueError('semantic qualification requires an empty-gradient update boundary')
        self.versions = [(name,id(p),p._version,p.requires_grad) for name,p in self.named]
        opts = [entry.optimizer for entry in participants.values() if entry.optimizer is not None]
        if optimizer is not None: opts.append(optimizer)
        self.optimizers = [(opt,_optimizer_versions(opt)) for opt in opts]
        self.backends=[]
        for entry in participants.values():
            backend=entry.backend
            if backend is None:continue
            counters={key:(hasattr(backend,key),copy.deepcopy(getattr(backend,key,None)))
                      for key in ('row_counts','has_executed')}
            modules=[(module,hasattr(module,'_fp8_row_stats'),copy.deepcopy(getattr(module,'_fp8_row_stats',None)))
                     for module in getattr(backend,'linears',{}).values()]
            self.backends.append((backend,copy.deepcopy(backend.extra_state()),counters,modules))

    def restore(self):
        try:
            for _,p in self.named:p.grad=None
            for backend,state,counters,modules in self.backends:
                backend.restore_extra_state(copy.deepcopy(state))
                for key,(present,value) in counters.items():
                    if present:setattr(backend,key,copy.deepcopy(value))
                    elif hasattr(backend,key):delattr(backend,key)
                for module,present,value in modules:
                    if present:module._fp8_row_stats=copy.deepcopy(value)
                    elif hasattr(module,'_fp8_row_stats'):delattr(module,'_fp8_row_stats')
        finally:
            restore_training_rng_state(self.rng,self.device)
        current=[(name+'.'+key,id(p),p._version,p.requires_grad) for name,entry in sorted(self.participants.items())
                 for key,p in entry.model.named_parameters()]
        current += [('vae.'+key,id(p),p._version,p.requires_grad) for key,p in self.vae.named_parameters()]
        if current!=self.versions or any(_optimizer_versions(opt)!=saved for opt,saved in self.optimizers):
            raise CapacityStateChangedError('semantic qualification changed parameters or optimizer state')

