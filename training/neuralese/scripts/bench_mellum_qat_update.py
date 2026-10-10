"""Update-time probe on real Mellum layers (plans/mellum-port.md, "Warm-up step profile"): the warm-up's update shape
(a main stream of batch 2 x ~500 tokens and a ~3k-token preserve stream, per-layer checkpointing, layer-lockstep
backward with LionSR stepping in backward) at bf16, q4 and ternary-experts precision, with the eager precision ramp
and loop MoE (before) and the fused ramp and grouped MoE (after).

    python bench_mellum_qat_update.py [--layers 3] [--reps 3]
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natlang_neuralese.common.paths import resolve  # noqa: E402
from natlang_neuralese.maple import ternary  # noqa: E402
from natlang_neuralese.maple.model import DENSE_MOE, DenseExperts, load_maple  # noqa: E402
from natlang_neuralese.maple.qat_convert import row_scale  # noqa: E402
from natlang_neuralese.model.layer_staging import LayerStaging  # noqa: E402
from natlang_neuralese.train.optim import LionSR  # noqa: E402

POINTS = {
    'bf16': None,
    'q4': {'experts': ('int4', 0.47, {'group': 32}), 'attention': ('int4', 0.47, {'group': 32})},
    'ternary-experts': {'experts': ('ternary', 0.5, {}), 'attention': ('int4', 1.0, {'group': 32})},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--main-tokens', type=int, default=500)
    p.add_argument('--preserve-tokens', type=int, default=3000)
    a = p.parse_args()
    torch.manual_seed(0)
    model = load_maple(resolve('models', 'mellum21-12b-a2.5b-thinking'), device='cuda', layers=a.layers,
                       ternary_attention=False, dense_experts=True)
    inner = model.model
    inner.checkpoint_layers = True
    latents = []
    for index, layer in enumerate(inner.layers):
        layer.mlp.experts.make_latent(quantize=False)
        layer.mlp.experts.precision_group = f'experts@{index}'
        latents += [layer.mlp.experts.gate_up, layer.mlp.experts.down]
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            module = getattr(layer.self_attn, name)
            module.weight.requires_grad_(True)
            torch.nn.utils.parametrize.register_parametrization(module, 'weight',
                                                                ternary.PrecisionSTE(f'attention@{index}'), unsafe=True)
            latents.append(module.parametrizations.weight.original)
    for p_ in model.parameters():
        if not any(p_ is q for q in latents):
            p_.requires_grad_(False)
    assert all(isinstance(layer.mlp.experts, DenseExperts) for layer in inner.layers)
    optimizer = LionSR([{'params': [q], 'lr': 3e-4, 'row_scale': row_scale(q)} for q in latents], lr=3e-4)
    optimizer.step_in_backward(gated=True)
    hidden = model.config.hidden_size
    main = (torch.randn(2, a.main_tokens, hidden, device='cuda') * 0.5).bfloat16()
    preserve = (torch.randn(1, a.preserve_tokens, hidden, device='cuda') * 0.5).bfloat16()

    def update():
        with LayerStaging() as tape, optimizer.in_backward() as armed:
            terms = [inner(inputs_embeds=x).last_hidden_state.float().square().mean() for x in (main, preserve)]
            (terms[0] + terms[1]).backward()
            tape.backward()
        armed.norm()

    results = {}
    for variant, fused, kernel in (('before', False, 'loop'), ('after', True, 'grouped')):
        ternary.PRECISION['fused'] = fused
        DENSE_MOE['kernel'] = kernel
        for point, groups in POINTS.items():
            with ternary.active_precision(groups, None if groups is None else point):
                update()  # warm-up (kernel compiles, allocator)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(a.reps):
                    update()
                torch.cuda.synchronize()
            seconds = (time.perf_counter() - start) / a.reps
            results[f'{variant}/{point}'] = {'seconds_per_update': round(seconds, 3),
                                             'per_layer': round(seconds / a.layers, 3),
                                             'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            print(variant, point, results[f'{variant}/{point}'], flush=True)
    for point in POINTS:
        results[f'speedup/{point}'] = round(results[f'before/{point}']['seconds_per_update'] /
                                            results[f'after/{point}']['seconds_per_update'], 2)
    print(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
