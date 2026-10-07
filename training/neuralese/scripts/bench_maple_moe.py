"""Microbenchmark of Maple's grouped ternary expert GEMM on its training shapes (GB10): row block, tile and warp
configurations for the forward (A @ W_e^T) and input-gradient (A @ W_e) launches.

    python scripts/bench_maple_moe.py [--tokens 6270]
"""
import argparse
import itertools

import torch
import triton

from natlang_neuralese.maple.fused_moe import Plan, Weights, _grouped_ternary_mm


def launch(a, rows, weights, plan, transpose, BN, BK, warps, stages):
    codes, scale = weights.codes, weights.scale
    E, R, C = codes.shape
    N, K = (R, C) if not transpose else (C, R)
    out = torch.empty(plan.size, N, device=a.device, dtype=a.dtype)
    inner, outer = (codes.stride(2), codes.stride(1)) if not transpose else (codes.stride(1), codes.stride(2))
    grid = (plan.blocks, triton.cdiv(N, BN))
    _grouped_ternary_mm[grid](
        a, rows if rows is not None else a, codes, scale, out, plan.block_expert,
        plan.size, N, K, a.stride(0), a.stride(1), codes.stride(0), inner, outer,
        scale.stride(0), scale.stride(1), scale.stride(2), weights.block, out.stride(0), out.stride(1),
        HAS_ROWS=rows is not None, TRANS=transpose, SCALE_HALF=weights.half, BM=plan.block, BN=BN, BK=BK,
        num_warps=warps, num_stages=stages)
    return out


def timed(fn, reps=10):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / reps


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tokens', type=int, default=6270)
    p.add_argument('--experts', type=int, default=256)
    p.add_argument('--top-k', type=int, default=8)
    p.add_argument('--bm', type=int, nargs='*', default=[64, 128, 256])
    p.add_argument('--skew', type=float, default=0.0, help='std of a per-expert popularity bias on the router logits')
    p.add_argument('--dump', help='a call saved by fused_moe._dump_call: its real routing, inputs and weights')
    a = p.parse_args()
    torch.manual_seed(0)
    dev = 'cuda'
    hidden, ff, block = 2048, 512, 256
    shapes = {'gate_up': (2 * ff, hidden), 'down': (hidden, ff)}
    dump = torch.load(a.dump, weights_only=False) if a.dump else None
    if dump:
        index = dump['index'].to(dev)
        a.tokens, a.top_k = index.shape
        a.experts = dump['gate_up']['codes'].shape[0]
    else:
        bias = torch.randn(a.experts, device=dev) * a.skew
        index = (torch.randn(a.tokens, a.experts, device=dev) + bias).topk(a.top_k, -1).indices
    counts = torch.bincount(index.flatten(), minlength=a.experts).sort(descending=True).values
    print(f'skew {a.skew}: expert load max {int(counts[0])}, median {int(counts[len(counts)//2])}, '
          f'min {int(counts[-1])}, <64 rows: {int((counts < 64).sum())}/{a.experts}')
    dtype = dump['x'].dtype if dump else torch.bfloat16
    x = torch.randn(a.tokens + 1, hidden, device=dev, dtype=dtype)
    if dump:
        x[:-1] = dump['x'].to(dev)
        x[-1] = 0
    print('activations', dtype)
    for name, (rows_n, cols_k) in shapes.items():
        if dump:
            saved = dump[name]
            w = Weights(saved['codes'].to(dev), saved['scale'].to(dev), saved['block'] if saved['half'] else None)
            print(name, 'codes', tuple(w.codes.shape), w.codes.dtype, w.codes.stride(), 'scale', tuple(w.scale.shape),
                  w.scale.dtype, 'block', w.block, 'half', w.half)
        else:
            codes = torch.randint(-1, 2, (a.experts, rows_n, cols_k), device=dev, dtype=torch.int8)
            scale = torch.rand(a.experts, rows_n, cols_k // block, device=dev) * 0.02
            w = Weights(codes, scale, block)
        inp = x if name == 'gate_up' else torch.randn(a.tokens + 1, ff, device=dev, dtype=dtype)
        for transpose in (False, True):
            results = []
            for BM, BN, BK, warps, stages in itertools.product(a.bm, (64, 128, 256), (32, 64, 128),
                                                               (4, 8), (2, 3)):
                plan = Plan(index, a.experts, a.tokens, BM)
                if transpose:  # the input gradient: dense rows of the padded plan, inner dim = N of the forward
                    g = torch.randn(plan.size, rows_n, device=dev, dtype=dtype)
                    fn = lambda: launch(g, None, w, plan, True, BN, BK, warps, stages)
                else:
                    fn = lambda: launch(inp, plan.rows, w, plan, False, BN, BK, warps, stages)
                try:
                    ms = timed(fn)
                except Exception as error:  # out of shared memory / registers
                    continue
                results.append((ms, BM, BN, BK, warps, stages))
            results.sort()
            base = [r for r in results if r[1:] == ((64, 128, 64, 4, 2) if not transpose else (64, 128, 64, 4, 2))]
            print(f'{name} {"input-grad" if transpose else "forward"}: current {base[0][0]:.2f} ms' if base else f'{name} {"input-grad" if transpose else "forward"}')
            for r in results[:6]:
                print('   %.2f ms  BM=%d BN=%d BK=%d warps=%d stages=%d' % r)


if __name__ == '__main__':
    main()
