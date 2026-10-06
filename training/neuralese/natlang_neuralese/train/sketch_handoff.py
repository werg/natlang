"""Explicit fresh shallow-latent channel from a qualified full-depth reference.

This is an architecture change, never an in-place checkpoint migration or an
inherited generated-channel qualification. Consumer training has fresh sketch,
content residual and stop parameters. The default layout is the autoregressive latent-sketch-v2 (owner 2026-10-06):
payload j from the top state at j - 1, the sketch from j - 1 predicting it, and the close token from the last top state.
latent-sketch-v1 (payload from each position's own top state, separate shallow stop head) stays available.
"""
from ..model.heads import PortHeads


def install_latent_sketch(engine, *, cutoff, max_length=None, profile='latent-sketch-v2'):
    parent = engine.heads
    proof = dict(getattr(engine, 'foundation', None) or {})
    if parent.profile != 'raw-token-v1' or parent.cutoff != engine.backbone.num_layers:
        raise ValueError('sketch handoff requires a full-depth raw causal reference')
    if proof.get('qualified') is not True:
        raise ValueError('sketch handoff requires qualified foundation evidence')
    heads = PortHeads(engine.backbone, cutoff=cutoff, max_length=max_length or parent.max_length,
                      stop_source='final' if profile == 'latent-sketch-v2' else 'shallow', stop_position=False,
                      profile=profile)
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
