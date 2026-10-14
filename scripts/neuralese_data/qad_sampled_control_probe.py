#!/usr/bin/env python3
"""Measure control-token choices on actual sampled QAD continuations.

This stops at the first ordinary EOS or Neuralese control. It is a diagnostic
of reachable serving logits, not a task evaluation or training-data source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import torch

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.eval.task_cadence import restore_engine

from qad_hf_port_parity import port_forward, selected_requests


@torch.inference_mode()
def sample(engine, item, *, seed: int, temperature: float, max_tokens: int) -> dict:
    device = engine.backbone.embedding_weight.device
    ids = torch.tensor([item['token_ids']], device=device)
    logits, cache = port_forward(engine, ids)
    generator = torch.Generator(device='cpu').manual_seed(seed)
    controls = (engine.backbone.controls.open_id, engine.backbone.controls.close_id)
    largest_control_mass = 0.0
    chosen = []
    end_reason = 'length'
    for _ in range(max_tokens):
        probabilities = torch.softmax(logits[0].float() / temperature, -1).cpu()
        largest_control_mass = max(largest_control_mass, float(probabilities[list(controls)].sum()))
        token = int(torch.multinomial(probabilities, 1, generator=generator))
        if token in engine.stop_ids:
            end_reason = 'ordinary_stop'
            break
        if token == controls[0]:
            end_reason = 'neuralese_open'
            break
        if token == controls[1]:
            end_reason = 'free_neuralese_close'
            break
        chosen.append(token)
        logits, cache = port_forward(engine, torch.tensor([[token]], device=device), cache)
    return {'seed': seed, 'generated_before_boundary': len(chosen),
            'end_reason': end_reason, 'max_control_mass_t': largest_control_mass,
            'generated_token_ids_sha256': hashlib.sha256(json.dumps(chosen).encode()).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--max-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=0.6)
    args = parser.parse_args()
    if args.seeds < 1 or args.max_tokens < 1 or args.temperature <= 0:
        parser.error('seeds, max-tokens and temperature must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory() as temporary:
        request = Path(temporary)
        (request / 'request.json').write_text(json.dumps({'weights': str(args.checkpoint.resolve())}))
        engine = restore_engine({'dir': str(request), 'heads': str(args.heads.resolve()),
                                 'device': 'cuda', 'prompt_rendering': 'full'})
        report = {'schema': 'natlang.qad-sampled-control-probe/1',
                  'checkpoint_sha256': sha256_file_hex(args.checkpoint),
                  'heads_sha256': sha256_file_hex(args.heads),
                  'requests_sha256': sha256_file_hex(args.requests),
                  'temperature': args.temperature, 'max_tokens': args.max_tokens,
                  'control_token_ids': [engine.backbone.controls.open_id,
                                        engine.backbone.controls.close_id],
                  'ordinary_stop_token_ids': sorted(engine.stop_ids), 'cases': []}
        for item in selected_requests(args.requests, engine):
            case = {key: value for key, value in item.items() if key != 'token_ids'}
            case['samples'] = []
            for index in range(args.seeds):
                case['samples'].append(sample(engine, item, seed=400000 + index,
                                              temperature=args.temperature, max_tokens=args.max_tokens))
            report['cases'].append(case)
            (args.out / 'report.partial.json').write_text(json.dumps(report, indent=2) + '\n')
            print(json.dumps({'case': case['label'], 'boundaries': [row['end_reason'] for row in case['samples']],
                              'max_control_mass': max(row['max_control_mass_t'] for row in case['samples'])}), flush=True)
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
