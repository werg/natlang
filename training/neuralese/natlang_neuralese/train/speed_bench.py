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
import contextlib
import json
import os
import statistics
import time
from pathlib import Path

import torch
from ..common.paths import resolve


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default=str(resolve('models', 'mellum21-12b-a2.5b-thinking')))
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
    p.add_argument('--readout', choices=('chunked', 'fused'), default='fused',
                   help="direct_sft's readout: chunked_readout + soft_token_mixture, or fused_readout (shared logits)")
    p.add_argument('--dump-grads', type=Path, help='save layer 0\'s expert and attention latent gradients of the '
                   'last update here (real gradients for optimizer numerics checks); the optimizer step is skipped')
    p.add_argument('--fp8-training', choices=('off', 'triton-experts'), default='off')
    p.add_argument('--latent-optimizer', choices=('muonsr', 'lionsr'), default='muonsr')
    p.add_argument('--ns', default='cubic5')
    p.add_argument('--moe-kernel', default='grouped')
    p.add_argument('--member', help='a family member update (maple/family.py), e.g. 2x32 or 1x32 on a 2-layer bench: '
                   'the BF16 teacher forward, then the member forward/backward with KL(full || member)')
    p.add_argument('--order', default=str(resolve('repo') / 'runs/mellum-nested-20261009/n0-v1/expert-order.pt'),
                   help='--member: the N0 expert order (its first --layers layers are used)')
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
    from .text_warmup import chunked_readout, fused_readout
    from ..model.neuralese_projection import soft_token_mixture

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
    fp8_backend = None
    if a.fp8_training != 'off':
        from .fp8_backend import make_backend
        fp8_backend = make_backend(a.fp8_training, backbone.hf, device=device)
    family = None
    if a.member:
        from ..maple.family import Family, member_kl
        from ..common.hashing import sha256_file_hex
        from .joint_kd import IGNORE
        order = a.out / 'expert-order.pt'
        torch.save({'orders': torch.load(a.order)['orders'][:a.layers]}, order)
        family = Family({'schema': 'natlang.neuralese-family/1', 'members': {a.member: 1.0}, 'kl_weight': 1.0,
                         'order': {'path': str(order), 'sha256': sha256_file_hex(order)}, 'stages': ['bench'],
                         'stage': 'bench'})
        family.install(backbone)
    point, _, mix = a.qat.partition(':')
    mix = float(mix or 0)
    norm_dtypes = sorted({str(m.weight.dtype) for n, m in maple.named_modules() if n.endswith('norm')})
    print(json.dumps({'event': 'bench_loaded', 'seconds': time.time() - load_start, 'installed': installed,
                      'latents': len(latent), 'norm_weight_dtypes': norm_dtypes,
                      'compile': os.environ.get('NATLANG_MAPLE_COMPILE', '1')}), flush=True)

    vocab = int(backbone.embedding_weight.shape[0])
    table = backbone.embedding_weight.detach().float().clone()  # direct_sft's mixture table (FP32)
    generator = torch.Generator().manual_seed(a.seed)
    profiler = profiling.UpdateProfiler(a.out, value=f'{a.warmup + 1}:{a.updates}' if a.profile else '')
    rows = []
    for update in range(1, a.updates + 1):
        ids = torch.randint(0, vocab - 512, (a.rows, a.row_tokens + 1), generator=generator).to(device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        profiler.begin(update)
        tick = time.perf_counter()
        teacher = None
        if family is not None:  # direct_sft's member update: the BF16 full model's span states, no graph
            with profiling.region('update/teacher'), torch.no_grad(), quant.context('bf16', 0.0), (
                    fp8_backend.native_modules() if fp8_backend is not None else contextlib.nullcontext()):
                teacher = backbone.forward_ids(ids[:, :-1], cutoff=backbone.num_layers,
                                               logits=False)['h_final'][:, -a.span:]

        def part(rows_ids, scale=1.0):
            with profiling.region('update/forward'):
                hidden = backbone.forward_ids(rows_ids[:, :-1], cutoff=backbone.num_layers, logits=False)['h_final']
                states, gold = hidden[:, -a.span:], rows_ids[:, -a.span:]
                weights = torch.ones_like(gold, dtype=torch.float32)
                if a.readout == 'fused':
                    loss, _, _, mixture = fused_readout(backbone, states, gold, backbone.controls.close_id,
                                                        position_weights=weights, mixture_table=table)
                else:
                    _, loss, _, _, _ = chunked_readout(backbone, states, gold, backbone.controls.close_id,
                                                       position_weights=weights)
                    mixture = soft_token_mixture(backbone.logits, table, states.detach())
                loss = loss + 0.0 * mixture.sum()
                if teacher is not None:
                    loss = loss + member_kl(backbone, states, teacher, gold, kl_weight=family.kl_weight)[0]
                loss = loss * scale
            with profiling.region('update/backward'), profiling.backward_phase():
                loss.backward()
            return float(loss)

        fp8 = fp8_backend.autocast() if fp8_backend is not None else contextlib.nullcontext()
        member = family.selected(backbone, a.member) if family is not None else contextlib.nullcontext()
        with member, fp8, quant.context(point, mix), shared_parametrized_weights(backbone.hf) as next_pass:
            if a.dump_grads is not None and update == a.updates:  # accumulate this update's real gradients
                loss = sum(part(chunk, len(chunk) / a.rows) for chunk in ids.split(a.micro_rows or a.rows))
                torch.save({name: q.grad.detach().cpu() for name, q in latent_named
                            if '.layers.0.' in name and q.grad is not None}, a.dump_grads)
                print(json.dumps({'event': 'gradients_dumped', 'path': str(a.dump_grads)}), flush=True)
                return
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
