"""Gate-column probe on real Mellum layers (plans/mellum-port.md, "Stage-end eval cost"): the two parts of a
precision column's behaviour report, greedy decode through the port's KV cache (``generation_gate.greedy``) and a
held-row no-grad forward, at bf16, q4 and ternary-experts at deploy mix (lambda = 1), each with and without
``generation_gate.deployed_cache``.

    python bench_mellum_qat_column.py [--layers 3] [--decode 128] [--held-tokens 4096]
"""
import argparse
import contextlib
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natlang_neuralese.common.paths import resolve  # noqa: E402
from natlang_neuralese.maple import ternary  # noqa: E402
from natlang_neuralese.maple.generation_gate import deployed_cache, greedy  # noqa: E402
from natlang_neuralese.maple.model import DENSE_MOE, load_maple  # noqa: E402

POINTS = {
    'bf16': None,
    'q4': {'experts': ('int4', 1.0, {'group': 32}), 'attention': ('int4', 1.0, {'group': 32})},
    'ternary-experts': {'experts': ('ternary', 1.0, {}), 'attention': ('int4', 1.0, {'group': 32})},
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--decode', type=int, default=128)
    p.add_argument('--held-tokens', type=int, default=4096)
    a = p.parse_args()
    torch.manual_seed(0)
    DENSE_MOE['kernel'] = 'grouped'
    model = load_maple(resolve('models', 'mellum21-12b-a2.5b-thinking'), device='cuda', layers=a.layers,
                       ternary_attention=False, dense_experts=True)
    for index, layer in enumerate(model.model.layers):
        layer.mlp.experts.make_latent(quantize=False)
        layer.mlp.experts.precision_group = f'experts@{index}'
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj'):
            module = getattr(layer.self_attn, name)
            module.weight.requires_grad_(True)
            torch.nn.utils.parametrize.register_parametrization(module, 'weight',
                                                                ternary.PrecisionSTE(f'attention@{index}'), unsafe=True)
    prompt = list(range(1000, 1300))
    held = torch.arange(2000, 2000 + a.held_tokens, device='cuda')[None]

    def timed(fn):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        return time.perf_counter() - start

    results = {}
    for point, groups in POINTS.items():
        for cached in (False, True):
            scope = deployed_cache if cached else contextlib.nullcontext
            with ternary.active_precision(groups, None if groups is None else point), torch.no_grad():
                with scope():
                    greedy(model, prompt, 4, -1)  # warm-up
                    decode = timed(lambda: greedy(model, prompt, a.decode, -1))
                    model(input_ids=held[:, :256])
                    forward = timed(lambda: model(input_ids=held))
            row = {'decode_ms_per_token': round(1000 * decode / a.decode, 2), 'held_forward_s': round(forward, 3)}
            results[f'{point}{" cached" if cached else ""}'] = row
            print(json.dumps({'point': point, 'cached': cached, **row}), flush=True)
    print(json.dumps({'layers': a.layers, 'results': results}))


if __name__ == '__main__':
    main()
