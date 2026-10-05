"""Build a raw port from an exact, qualified foundation handoff.

The stop/content modules are fresh: this is a token-preserving initialization,
not a claim of autonomous stopping or compression quality.
"""
from pathlib import Path
import torch

from .recipe import require_foundation
from .output_embedding_projection import sha
from ..model.heads import PortHeads
from ..model.causal_feedback import load_projection_state
from ..serve import load_engine


def foundation_port(*, heads, checkpoint, certificate, device='cpu', max_length=128):
    proof = require_foundation(certificate, heads=heads, checkpoint=checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    engine = load_engine(heads_checkpoint=str(heads), device=device)
    port = PortHeads(engine.backbone, cutoff=saved['cutoff'], max_length=max_length,
                     stop_source='final', stop_position=False, profile='raw-token-v1').to(device).eval()
    load_projection_state(port.feedback, saved['projection'])
    for parameter in port.parameters():
        parameter.requires_grad_(False)
    engine.heads = port
    engine.max_block = max_length
    engine.dialect = 'nd:natlang-raw-token@1'
    engine.foundation = {'certificate_sha256': sha(certificate),
                         'source_heads_sha256': sha(heads),
                         'feedback_checkpoint_sha256': sha(checkpoint),
                         'qualified': proof['qualified'], 'runtime_qualified': False,
                         'autonomous_stopping_qualified': False}
    return engine


def save_foundation_port(engine, source_heads, output, *, runtime_report=None):
    output = Path(output)
    if output.exists():
        raise ValueError('fresh immutable port checkpoint required')
    old = torch.load(source_heads, map_location='cpu', weights_only=False, mmap=True)
    # Preserve only serving backbone lineage, never stale optimizer/phase claims.
    state = {key: old[key] for key in ('backbone', 'lora', 'lora_layers', 'lora_rank') if key in old}
    state.update({'schema': 'natlang.neuralese-raw-port-checkpoint/1',
                  'heads': {name: value.detach().cpu() for name, value in engine.heads.state_dict().items()},
                  'control_rows': engine.backbone.control_rows.detach().cpu(),
                  'port_config': engine.heads.port_config, 'foundation': dict(engine.foundation)})
    if runtime_report is not None:
        if runtime_report.get('runtime_transport_passed') is not True:
            raise ValueError('runtime transport gate failed')
        state['foundation']['runtime_qualified'] = True
        state['foundation']['runtime_report'] = runtime_report
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, output)
    return output
