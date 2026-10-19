"""One shared cotangent lifetime for all models in a semantic objective."""
from __future__ import annotations

import contextlib
import torch

from ..model.native_weight_carrier import ParameterCotangentScope


class SemanticObjectiveScope:
    """Retain both models' complete adjoints until the whole backward finishes.

    A module container gives the existing carrier one inventory over the exact
    participant objects. It neither copies parameters nor changes their local
    names or optimizer ownership. Cotangent scopes cannot be nested.
    """
    def __init__(self, participants, vae, *, mode='update'):
        self.participants = participants
        self.mode = mode
        self.container = torch.nn.ModuleDict({
            'participants': torch.nn.ModuleList([participants[name].model for name in sorted(participants)]),
            'vae': vae,
        })
        self.stack = contextlib.ExitStack()
        self.cotangents = None

    def __enter__(self):
        try:
            self.cotangents = self.stack.enter_context(
                ParameterCotangentScope.for_model(self.container, mode=self.mode))
            for name in sorted(self.participants):
                backend = self.participants[name].backend
                if backend is not None:
                    self.stack.enter_context(backend.autocast())
            return self
        except BaseException:
            self.close()
            raise

    def finish(self):
        self.cotangents.finish()

    def close(self):
        self.stack.close()

    def __exit__(self, *error):
        return self.stack.__exit__(*error)


def qualify_joint_gradients(vae, participants, row, *, phase, options=None,
                            fetch_external=None, atol, rtol, device, optimizer=None):
    """Measure the real two-sided objective without consuming a training update.

    The existing complete comparator retains CPU reference derivatives and
    checks every trainable, including unused-derivative agreement. There is no
    synthetic input embedding argument in this objective.
    """
    from .native_semantic_boundary import ObjectiveBoundary
    from .native_semantic import joint_loss
    from ..eval.native_gradients import compare_complete_gradients
    from ..common.hashing import canonical_json_sha256_hex

    named = [(name+'.'+key, value) for name in sorted(participants)
             for key, value in participants[name].model.named_parameters() if value.requires_grad]
    named += [('vae.'+name, value) for name,value in vae.named_parameters() if value.requires_grad]
    boundary=ObjectiveBoundary(participants,vae,optimizer,device)
    versions = [(value, value._version) for _,value in named]

    flat_scope = None
    reference_scope = None
    try:
        flat_scope = SemanticObjectiveScope(participants,vae,mode='vjp').__enter__()
        flat_loss,_ = joint_loss(vae,participants,row,phase=phase,options=options,
                                fetch_external=fetch_external)
        actual_loss = float(flat_loss.detach())
        reference_losses = []
        def reference():
            nonlocal reference_scope
            boundary.restore()
            reference_scope=SemanticObjectiveScope(participants,vae,mode='vjp').__enter__()
            loss,_=joint_loss(vae,participants,row,phase=phase,options=options,
                              fetch_external=fetch_external,reference=True)
            reference_losses.append(float(loss.detach()))
            return loss,None,reference_scope
        proof=compare_complete_gradients(flat_loss,None,named,None,None,
            atol=atol,rtol=rtol,cached_factory=reference,flat_cotangent_scope=flat_scope)
        reference_loss=reference_losses[0]
        finite=bool(torch.isfinite(torch.tensor([actual_loss,reference_loss])).all())
        gap=abs(actual_loss-reference_loss)
        unchanged=all(value._version==version for value,version in versions)
        current=[(name+'.'+key,id(value)) for name in sorted(participants)
                 for key,value in participants[name].model.named_parameters() if value.requires_grad]
        current += [('vae.'+key,id(value)) for key,value in vae.named_parameters() if value.requires_grad]
        unchanged=unchanged and current==[(name,id(value)) for name,value in named]
        return {'schema':'natlang.native-semantic-complete-gradients/1',
            'passed':proof['passed'] and finite and gap<=atol+rtol*abs(reference_loss) and unchanged,
            'phase':phase,'row_sha256':canonical_json_sha256_hex(row),
            'participants':{name:participant.identity for name,participant in sorted(participants.items())},
            'options':dict(options or {}),'complete_gradients':proof,
            'actual_loss':actual_loss,'reference_loss':reference_loss,'loss_abs':gap,
            'parameters_unchanged':unchanged,'optimizer_updates':0,'exposure_updates':0,
            'numerical_measurement_performed':True,'reference':'explicit-cached-whole-semantic-objective'}
    finally:
        if flat_scope is not None:flat_scope.close()
        if reference_scope is not None:reference_scope.close()
        boundary.restore()
