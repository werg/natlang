"""Measure a trained stopping head on sealed prefixes without changing its policy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from ..serve import load_engine
from ..train.execution import prefill_batch, unroll_write


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


@torch.no_grad()
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--fixed-inputs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    inputs = torch.load(args.fixed_inputs, map_location='cpu', weights_only=False)
    engine = load_engine(heads_checkpoint=str(args.checkpoint), device=args.device)
    if args.device == 'cpu':
        engine.backbone.conv_kernel = None
    cases = []
    for prefix in inputs['prefixes']:
        producer = list(prefix) + [engine.backbone.controls.open_id]
        pre = prefill_batch(engine.backbone, engine.heads, [producer])
        written = unroll_write(engine.backbone, engine.heads, pre, sample=False, temperature=0.0)
        length = int(written.lengths[0])
        logits = written.stop_logits[0, :length].float().cpu()
        p = torch.sigmoid(logits)
        entropy = -(p * F.logsigmoid(logits) + (1 - p) * F.logsigmoid(-logits))
        cases.append({'prefix_tokens': len(prefix), 'length': length,
                      'truncated': bool(written.truncated[0]),
                      'stop_logits': logits.tolist(), 'stop_probabilities': p.tolist(),
                      'stop_entropy_nats': entropy.tolist(),
                      'log_probability_continue_before_boundary': float(F.logsigmoid(-logits[:-1]).sum()),
                      'probability_stop_before_boundary': float(-torch.expm1(F.logsigmoid(-logits[:-1]).sum()))})
    report = {'schema': 'natlang.stopping_diagnostics/1',
              'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': digest(args.checkpoint),
              'fixed_inputs_sha256': digest(args.fixed_inputs), 'device': args.device,
              'scope': 'probabilities on greedy paths for the fixed prefixes; not alternate-policy rollouts or a population estimate',
              'policy_changed': False, 'cases': cases}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'cases': len(cases), 'lengths': [c['length'] for c in cases],
                      'probability_stop_before_boundary': [c['probability_stop_before_boundary'] for c in cases]}), flush=True)


if __name__ == '__main__':
    main()
