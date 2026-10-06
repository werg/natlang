"""Requalify raw serving transport/replay against exact learned recurrent weights."""
import argparse
import json
from pathlib import Path

import torch

from .raw_port_handoff import qualify_raw_transport
from ..serve.recurrence_checkpoint import load_recurrence_checkpoint
from ..train.output_embedding_projection import sha
from ..train.trajectories import handover_notes, target_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'records', 'out'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--limit', type=int, default=16)
    args = parser.parse_args()
    if args.limit < 1 or args.out.exists():
        raise ValueError('positive limit and fresh output required')
    torch.set_num_threads(2)
    engine, state = load_recurrence_checkpoint(args.checkpoint, device=args.device, dtype=torch.bfloat16)
    if engine.heads.profile != 'raw-token-v1' or engine.heads.content.transport != 'raw-identity':
        raise ValueError('this gate only qualifies token-aligned raw-identity transport')
    texts = set()
    for row in map(json.loads, args.records.open()):
        name = target_write(row)
        if name and row.get('split') == 'test' and row.get('training_admission', {}).get('approved') is True:
            texts.add(handover_notes(row)[name])
    if not texts:
        raise ValueError('no admitted held source controls')
    report = qualify_raw_transport(engine, texts, limit=args.limit)
    report.update(schema='natlang.recurrence-runtime-requalification/1', checkpoint_step=state['step'],
                  pins={str(path): sha(path) for path in (args.checkpoint, args.records)},
                  weights_unmodified=True, ffn_chunk_tokens=engine.backbone.ffn_chunk_tokens,
                  qualification_inherited=False,
                  instruction_parameter_selection_qualified=False,
                  autonomous_stopping_qualified=False, semantic_compression_qualified=False)
    args.out.mkdir(parents=True)
    (args.out / 'runtime-report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)
    if not report['runtime_transport_passed']:
        raise SystemExit('learned runtime transport/replay gate failed; report preserved')


if __name__ == '__main__':
    main()
