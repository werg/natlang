"""Training-speed bench for the direct trainer's update on a Maple-family backbone (plans/neuralese/TRAINING_SPEED_20261012.md).

One packed update as ``train/direct_sft.py`` runs it (``--window-execution packed``): the backbone's ``forward_ids`` over
``rows`` right-padded windows, the chunked readout CE on the last ``--span`` positions, backward with the latents stepping
in backward (MuonSR or LionSR, ``step_in_backward(gated=True)``), then the routers/norms optimizer step. The model may be
truncated to ``--layers`` layers (``load_maple(layers=...)``) so the bench fits beside a running job; per-layer cost
extrapolates. Every update draws fresh random ids from a fixed seed, so arms see identical batches; ``--parity`` writes
the per-update losses so numerics-changing arms can be compared on the same batch (update 1 runs before any optimizer
step, so its loss is the exact fixed-batch comparison).

Arms: ``--checkpoint on|off``, ``--qat <point>:<mix>`` (a precision point of the recipe's quantization component),
``NATLANG_MAPLE_COMPILE`` in the environment, ``--profile`` (common/profiling.py breakdown of the measured updates).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    from ..common.paths import resolve_str
    p.add_argument('--model', default=resolve_str('models', 'mellum21-12b-a2.5b-thinking'))
    p.add_argument('--recipe', default='recipes/neuralese-direct-mellum-v3.json')
    p.add_argument('--layers', type=int, default=4)
    p.add_argument('--rows', type=int, default=6)
    p.add_argument('--row-tokens', type=int, default=4800)
    p.add_argument('--span', type=int, default=600, help='supervised positions per row (readout)')
    p.add_argument('--updates', type=int, default=6)
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--checkpoint', choices=('on', 'off'), default='on')
    p.add_argument('--micro-rows', type=int, default=0,
                   help='sequential accumulation (the trainer\'s sequential path): forward/backward this many rows at a '
                        'time, gradients accumulated, one optimizer step; 0: packed with the latents stepping in backward')
    p.add_argument('--qat', default='bf16:0', help='point:mix, e.g. q4:1.0 or ternary-experts:0.5')
    p.add_argument('--latent-optimizer', choices=('muonsr', 'lionsr'), default='muonsr')
    p.add_argument('--ns', default='cubic5')
    p.add_argument('--moe-kernel', default='grouped')
    p.add_argument('--profile', action='store_true')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    from ..common import profiling
    from ..maple.maple_port import MaplePortBackbone
    from ..maple.model import load_maple, set_dense_moe_kernel
    from ..model.hf_port import family_controls
    from .backbone_policy import configure_backbone_training, shared_parametrized_weights
    from .optim import PortMuonAdamW, latent_entries, latent_options, latent_split
    from .quantization import Quantization
    from .text_warmup import chunked_readout

    torch.manual_seed(a.seed)
    device = 'cuda'
    load_start = time.time()
    maple = load_maple(a.model, device=device, ternary_attention=False, layers=a.layers)
    maple.natlang_precision = 'bf16'
    tokenizer = AutoTokenizer.from_pretrained(a.model)
    backbone = MaplePortBackbone(maple, controls=family_controls(maple, tokenizer)).to(device)
    set_dense_moe_kernel(a.moe_kernel)
    recipe = json.loads(Path(a.recipe).read_text())
    component = dict(recipe['overrides']['quantization'],
                     stage={'id': 'direct_sft', 'progress': [0.0, 1.0], 'start': 0, 'end': 100})
    quant = Quantization(component)
    installed = quant.install(backbone.hf, layer_count=backbone.num_layers)
    named = [('backbone.' + n, q) for n, q in configure_backbone_training(backbone, 'latent')]
    latent_named, other = latent_split(named, a.latent_optimizer)
    latent = latent_entries(latent_named, a.latent_optimizer, 1e-5, maple.config.head_dim)
    from ..model.neuralese_projection import NeuraleseProjection
    projection = NeuraleseProjection(int(backbone.embedding_weight.shape[1])).to(device)  # the trainer's head (idle)
    head_named = [('neuralese_projection.' + n, q) for n, q in projection.named_parameters()]
    optimizer = PortMuonAdamW(head_named + other, lr=7.5e-6, vocab_size=int(backbone.embedding_weight.shape[0]), latent=latent,
                              latent_optimizer=a.latent_optimizer,
                              latent_options=latent_options(a.latent_optimizer, weight_decay=0.1, ns=a.ns))
    optimizer.latent.step_in_backward(gated=True)
    backbone.checkpoint_layers = a.checkpoint == 'on'
    point, _, mix = a.qat.partition(':')
    mix = float(mix or 0)
    norm_dtypes = sorted({str(m.weight.dtype) for n, m in maple.named_modules() if n.endswith('norm')})
    print(json.dumps({'event': 'bench_loaded', 'seconds': time.time() - load_start, 'installed': installed,
                      'latents': len(latent), 'norm_weight_dtypes': norm_dtypes,
                      'compile': os.environ.get('NATLANG_MAPLE_COMPILE', '1')}), flush=True)

    vocab = int(backbone.embedding_weight.shape[0])
    generator = torch.Generator().manual_seed(a.seed)
    profiler = profiling.UpdateProfiler(a.out, value=f'{a.warmup + 1}:{a.updates}' if a.profile else '')
    rows = []
    for update in range(1, a.updates + 1):
        ids = torch.randint(0, vocab - 512, (a.rows, a.row_tokens + 1), generator=generator).to(device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        profiler.begin(update)
        tick = time.perf_counter()
        def part(rows_ids, scale=1.0):
            with profiling.region('update/forward'):
                hidden = backbone.forward_ids(rows_ids[:, :-1], cutoff=backbone.num_layers, logits=False)['h_final']
                _, loss, _, _, _ = chunked_readout(backbone, hidden[:, -a.span:], rows_ids[:, -a.span:],
                                                   backbone.controls.close_id)
                loss = loss * scale
            with profiling.region('update/backward'), profiling.backward_phase():
                loss.backward()
            return float(loss)

        with quant.context(point, mix), shared_parametrized_weights(backbone.hf) as next_pass:
            if a.micro_rows:
                loss = sum(part(chunk, len(chunk) / a.rows) for chunk in ids.split(a.micro_rows))
            else:
                next_pass()
                with optimizer.latent.in_backward():
                    loss = part(ids)
            with profiling.region('update/optimizer'):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - tick
        profiler.update_done(update, seconds)
        profiler.maybe_finish(update)
        row = {'update': update, 'seconds': seconds, 'loss': float(loss),
               'peak_gb': torch.cuda.max_memory_allocated() / 2 ** 30}
        rows.append(row)
        print(json.dumps(row), flush=True)
    profiler.close(a.updates)
    measured = [r['seconds'] for r in rows[a.warmup:]]
    tokens = a.rows * a.row_tokens
    summary = {'arm': {k: str(v) for k, v in vars(a).items()}, 'compile': os.environ.get('NATLANG_MAPLE_COMPILE', '1'),
               'median_seconds': statistics.median(measured), 'mean_seconds': statistics.fmean(measured),
               'tokens_per_update': tokens, 'tokens_per_second': tokens / statistics.median(measured),
               'peak_gb': max(r['peak_gb'] for r in rows), 'losses': [r['loss'] for r in rows], 'updates': rows}
    (a.out / 'bench.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k != 'updates'}), flush=True)


if __name__ == '__main__':
    main()
