#!/usr/bin/env python3
"""Is a training lineage learning or diffusing? Coherence of its weight trajectory from saved checkpoints.

Given the base weights and checkpoints of one lineage in update order, for every hidden matrix:
  1. the window deltas between consecutive checkpoints (base -> c1 -> c2 ...), their norms and the cosine between
     consecutive windows (near 0: successive windows move in unrelated directions);
  2. the coherence ratio |base -> last| / sqrt(sum |window|^2): about 1 for a random walk, up to sqrt(windows) for
     a consistent direction;
  3. the BF16 stochastic-rounding share of the total change: sub-ulp updates rounded stochastically add energy
     proportional to each weight's BF16 spacing s, a learned direction does not. Fit E[d^2] = a + b*s over the
     base weights' exponent buckets; the share is b*mean(s)/mean(d^2).
The LFM3 step-8341 lineage measured window cosines 0.002-0.03, ratio ~1.0 and SR share ~0.35
(plans/neuralese/LFM3_REGRESSION_INVESTIGATION_20261029.md).

  weight_coherence.py --base <hf snapshot dir> CKPT1 CKPT2 [...] [--out report.json]
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file


def backbone(path):
    checkpoint = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    return checkpoint['backbone_state'], checkpoint.get('training', {}).get('global_step', checkpoint.get('step'))


def sr_share(w0, delta):
    """BF16 stochastic-rounding share of the change energy (exponent-bucket fit, buckets of >= 2000 elements)."""
    exponent = torch.floor(torch.log2(w0.abs().clamp_min(1e-30)))
    spacing = torch.pow(2.0, exponent - 7)
    squared = delta.square()
    rows = []
    for value in torch.unique(exponent).tolist():
        mask = exponent == value
        count = int(mask.sum())
        if count >= 2000:
            rows.append((count, 2.0 ** (value - 7), float(squared[mask].mean())))
    if len(rows) < 3:
        return None
    weights = torch.tensor([r[0] for r in rows], dtype=torch.float64).sqrt()
    design = torch.tensor([[1., r[1]] for r in rows], dtype=torch.float64) * weights[:, None]
    target = torch.tensor([r[2] for r in rows], dtype=torch.float64) * weights
    slope = float(torch.linalg.lstsq(design, target).solution[1])
    return max(0., slope * float(spacing.double().mean()) / max(float(squared.double().mean()), 1e-300))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, required=True, help='pinned base snapshot directory (model.safetensors)')
    parser.add_argument('checkpoints', type=Path, nargs='+', help='full-state checkpoints of one lineage, in order')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    base = load_file(str(args.base / 'model.safetensors'))
    states = [backbone(path) for path in args.checkpoints]
    names = [name for name, value in base.items() if value.ndim == 2 and min(value.shape) >= 256
             and 'embed' not in name and all(name in state for state, _ in states)]
    windows = len(states)
    cosines = [[] for _ in range(windows - 1)]
    norms = [[] for _ in range(windows)]
    ratios, shares = [], []
    for name in names:
        w0 = base[name].float()
        points = [w0] + [state[name].float() for state, _ in states]
        deltas = [(after - before).flatten() for before, after in zip(points, points[1:])]
        for index, delta in enumerate(deltas):
            norms[index].append(float(delta.norm()))
        for index in range(windows - 1):
            a, b = deltas[index], deltas[index + 1]
            cosines[index].append(float(a @ b / (a.norm() * b.norm()).clamp_min(1e-30)))
        total = (points[-1] - w0).flatten()
        ratios.append(float(total.norm()) / max(math.sqrt(sum(float(d.norm()) ** 2 for d in deltas)), 1e-30))
        share = sr_share(w0, points[-1] - w0)
        if share is not None:
            shares.append(share)
    mean = lambda values: sum(values) / len(values) if values else None
    report = {'schema': 'natlang.weight-coherence/1', 'base': str(args.base),
              'checkpoints': [{'path': str(path), 'global_step': step} for path, (_, step) in
                              zip(args.checkpoints, states)],
              'matrices': len(names),
              'window_norm_mean': [mean(values) for values in norms],
              'consecutive_window_cosine_mean': [mean(values) for values in cosines],
              'coherence_ratio_mean': mean(ratios), 'random_walk_ratio': 1.0,
              'coherent_ratio': math.sqrt(windows),
              'sr_noise_share_mean': mean(shares)}
    text = json.dumps(report, indent=1)
    if args.out is not None:
        args.out.write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
