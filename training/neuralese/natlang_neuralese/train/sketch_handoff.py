"""Explicit fresh shallow-latent channel from a qualified full-depth reference.

This is an architecture change, never an in-place checkpoint migration or an
inherited generated-channel qualification. Consumer training has fresh sketch,
content residual and shallow stop parameters.
"""
from ..model.heads import PortHeads


def install_latent_sketch(engine, *, cutoff, max_length=None):
    parent = engine.heads
    proof = dict(getattr(engine, 'foundation', None) or {})
    if parent.profile != 'raw-token-v1' or parent.cutoff != engine.backbone.num_layers:
        raise ValueError('sketch handoff requires a full-depth raw causal reference')
    if proof.get('qualified') is not True:
        raise ValueError('sketch handoff requires qualified foundation evidence')
    heads = PortHeads(engine.backbone, cutoff=cutoff, max_length=max_length or parent.max_length,
                      stop_source='shallow', stop_position=False, profile='latent-sketch-v1')
    heads.content.reference.load_state_dict(parent.feedback.state_dict())
    heads.to(device=engine.backbone.embedding_weight.device).eval()
    heads.configure_frozen_reference()
    engine.heads = heads
    engine.max_block = heads.max_length
    engine.dialect = heads.dialect
    engine.foundation = {**proof, 'runtime_qualified': False,
                        'autonomous_stopping_qualified': False,
                        'reference_foundation': proof,
                        'channel_profile': heads.profile,
                        'scope': 'Full-depth reference only; new shallow latent channel needs separate replay and consumer qualification.'}
    return heads
