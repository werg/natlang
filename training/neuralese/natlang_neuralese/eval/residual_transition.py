"""Qualify zero-residual activation at exact learned raw-parent weights."""
import argparse
import json
from pathlib import Path
import torch
from .raw_port_handoff import qualify_raw_transport
from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
from ..train.trajectory_state import initialize_content_residual
from ..train.output_embedding_projection import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('checkpoint', 'out'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        raise ValueError('fresh output required')
    torch.set_num_threads(2)
    engine, state = load_recurrence_checkpoint(a.checkpoint, device='cuda', dtype=torch.bfloat16)
    if engine.heads.content.transport != 'raw-identity':
        raise ValueError('requires previously bypassed raw residual')
    optimizer = torch.optim.AdamW(engine.heads.content.proj.parameters())
    # Exercise clearing populated optimizer slots, not just empty new state.
    for parameter in engine.heads.content.proj.parameters():
        optimizer.state[parameter] = {'sentinel': torch.ones_like(parameter)}
    reset = initialize_content_residual(engine.heads, optimizer)
    if optimizer.state:
        raise ValueError('residual optimizer slots were not cleared')
    engine.heads.set_content_transport('learned-residual')
    report = qualify_raw_transport(engine, ['The package was delivered yesterday.', 'A proposed fix is not a completed repair.'], limit=2)
    report.update(schema='natlang.zero-residual-transition-qualification/1',
                  parent_sha256=sha(a.checkpoint), parent_step=state['step'], reset_parameters=reset,
                  initialized_residual_optimizer_cleared=True, learned_weights_qualified=False,
                  scope='Exact channel at zero-residual transition only; subsequent learned weights require separate qualification.')
    a.out.mkdir(parents=True)
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['runtime_transport_passed']:
        raise SystemExit('zero-residual transition failed')


if __name__ == '__main__':
    main()
