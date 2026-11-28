#!/usr/bin/env python3
"""Distill ``NativeQueryProjector`` onto Mode-P ``.nzs`` header taps from 000000002's 45k-step ``query_Q.pt``.

Runs once (~15-30 min on 1 GPU) over prepared T2 site rows so online T3 Mode-P HKB retrieval runs 100% from
LFM3-1.7B's own header state taps with ZERO extra 0.6B PPLX query model in VRAM.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training' / 'neuralese'), str(REPO)]

from natlang_neuralese.compiler.compiled_site import default_taps, init_block_values, init_header_state, state_layout  # noqa: E402
from natlang_neuralese.compiler.live_kb_writer import PplxWarmstartWriter  # noqa: E402
from natlang_neuralese.compiler.native_query import NativeQueryProjector, distill_native_query_projector  # noqa: E402
from natlang_neuralese.compiler.run_t2 import _jsonl  # noqa: E402
from natlang_neuralese.serve import load_native_engine  # noqa: E402
from natlang_neuralese.serve.store import TensorStore  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--rows', type=Path, required=True, help='run_t2 prepare JSONL(.zst) rows')
    parser.add_argument('--checkpoint', type=Path, required=True, help='LFM3 native checkpoint')
    parser.add_argument('--base', type=Path, required=True, help='HF base directory')
    parser.add_argument('--warmstart-dir', type=Path, required=True, help='000000002 warmstart dir (writer_H.pt + query_Q.pt)')
    parser.add_argument('--pplx-revision', required=True, help='pinned HF revision for pplx-embed-v2-late-0.6b')
    parser.add_argument('--out', type=Path, required=True, help='output .pt path for distilled NativeQueryProjector')
    parser.add_argument('--store-dir', type=Path, default=None, help='optional TensorStore cache for header states')
    parser.add_argument('--max-sites', type=int, default=512)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=512)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--dtype', default='bfloat16')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args(argv)

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = [r for r in _jsonl(args.rows) if r.get('split') == 'train']
    by_site = {}
    for r in rows:
        by_site.setdefault(r['site'], r)
    unique_rows = list(by_site.values())[: int(args.max_sites)]
    if len(unique_rows) < 2:
        raise SystemExit(f'{args.rows}: need >= 2 unique train sites, got {len(unique_rows)}')

    engine = load_native_engine(
        str(args.checkpoint), base=str(args.base), device=args.device,
        dtype=getattr(torch, args.dtype), training_state=False,
    )
    store_dir = args.store_dir or (args.out.parent / 'nqp-store')
    engine.store = TensorStore(store_dir)

    writer = PplxWarmstartWriter.from_warmstart(
        revision=args.pplx_revision,
        warmstart_dir=str(args.warmstart_dir),
        device=args.device,
        dtype=getattr(torch, args.dtype),
    ).eval()

    # Precompute pooled .nzs state taps and 000000002 teacher query vectors once (zero online model during training).
    tap_names = None
    tap_dims = None
    cached_taps: list[list[torch.Tensor]] = []
    cached_q: list[torch.Tensor] = []
    with torch.no_grad():
        for row in unique_rows:
            sid = init_header_state(engine, row['messages'], row['tools'], row['header'])
            block = engine.store.get_native_state(sid).block
            if tap_names is None:
                layout = tuple(state_layout(block))
                shapes = dict(layout)
                tap_names = tuple(default_taps(layout))
                tap_dims = tuple(int(torch.tensor(shapes[n]).prod().item()) for n in tap_names)
            vals = init_block_values(block)
            pooled = [vals[n].float().flatten(1).mean(dim=0).cpu() for n in tap_names]
            q_text = row.get('experience_query') or row.get('header') or ''
            q_vec = writer.encode_query_texts(engine, [q_text])[0].detach().float().cpu()
            cached_taps.append(pooled)
            cached_q.append(q_vec)

    del writer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    projector = NativeQueryProjector(
        tap_dims=tap_dims,
        key_dim=int(cached_q[0].numel()),
        hidden_dim=int(args.hidden_dim),
        tap_source='state',
        tap_names=tap_names,
    ).to(args.device)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=float(args.lr), weight_decay=1e-4)

    first_metrics = last_metrics = None
    n = len(cached_taps)
    for step in range(1, int(args.steps) + 1):
        idxs = [rng.randrange(n) for _ in range(min(int(args.batch), n))]
        taps_batch = [
            torch.stack([cached_taps[i][t] for i in idxs], dim=0).to(args.device)
            for t in range(len(tap_dims))
        ]
        q_batch = torch.stack([cached_q[i] for i in idxs], dim=0).to(args.device)
        metrics = distill_native_query_projector(projector, optimizer, taps_batch, q_batch)
        if first_metrics is None:
            first_metrics = metrics
        last_metrics = metrics

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'schema': 'natlang.native-query-projector/1',
        'tap_dims': list(tap_dims),
        'tap_names': list(tap_names),
        'tap_source': 'state',
        'key_dim': int(cached_q[0].numel()),
        'hidden_dim': int(args.hidden_dim),
        'state_dict': {k: v.detach().cpu() for k, v in projector.state_dict().items()},
        'metrics': {'first': first_metrics, 'last': last_metrics, 'sites': n, 'steps': int(args.steps)},
    }, args.out)
    print(json.dumps({
        'out': str(args.out), 'sites': n, 'steps': int(args.steps),
        'first_cosine': round(first_metrics['mean_cosine'], 5),
        'last_cosine': round(last_metrics['mean_cosine'], 5),
    }))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
