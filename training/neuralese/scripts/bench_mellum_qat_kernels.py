"""Kernel probe (plans/mellum-port.md, "warm-up step profile"): exactness of a compiled Q4_0 fake-quant, ternary
cost, grouped-GEMM MoE vs the per-expert loop, FP32 vs FP64 gradient norm. Synthetic one-layer Mellum expert
shapes; on the shared GPU the timings are contended."""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/home/werg/natlang/training/neuralese")
from natlang_neuralese.maple import ternary  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
E, D, F = 64, 2304, 896
gate_up = (torch.randn(E, 2 * F, D, device=dev) * 0.02).bfloat16()
down = (torch.randn(E, D, F, device=dev) * 0.02).bfloat16()
res = {}


def timeit(name, fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    res[name] = round((time.perf_counter() - t) / reps * 1000, 2)
    print(name, res[name], "ms", flush=True)


def q4_ramp(w, mix):
    q = ternary.q4_0(w, 32)
    x = w.float()
    return (x + mix * (q - x)).to(w.dtype)


def q4_only(w):
    return ternary.q4_0(w, 32)


with torch.no_grad():
    for name, fn in (("default", torch.compile(q4_only, dynamic=False)),):
        e, c = q4_only(gate_up), fn(gate_up)
        diff = (e != c)
        res[f"q4_compiled_{name}_mismatch_fraction"] = float(diff.float().mean())
        res[f"q4_compiled_{name}_max_abs_diff"] = float((e - c).abs().max())
    # Inductor without FMA contraction / with precise division
    import torch._inductor.config as ic
    torch._dynamo.reset()
    old = getattr(ic, "emulate_precision_casts", None)
    ic.emulate_precision_casts = True
    try:
        fn = torch.compile(q4_only, dynamic=False)
        c = fn(gate_up)
        res["q4_compiled_emulate_casts_mismatch_fraction"] = float((q4_only(gate_up) != c).float().mean())
    except Exception as exc:  # noqa: BLE001
        res["q4_compiled_emulate_casts_error"] = repr(exc)[:200]
    ic.emulate_precision_casts = old
    torch._dynamo.reset()

    # ternary point cost (experts' later precision)
    timeit("ternary_quantized_value_layer_eager_ms",
           lambda: (ternary.quantized_value(gate_up, "ternary"), ternary.quantized_value(down, "ternary")))
    tc = torch.compile(lambda w: ternary.quantized_value(w, "ternary"), dynamic=False)
    try:
        timeit("ternary_quantized_value_layer_compiled_ms", lambda: (tc(gate_up), tc(down)))
        res["ternary_compiled_mismatch_fraction"] = float(
            (ternary.quantized_value(gate_up, "ternary") != tc(gate_up)).float().mean())
    except Exception as exc:  # noqa: BLE001
        res["ternary_compiled_error"] = repr(exc)[:300]

# grouped GEMM MoE (fwd+bwd) vs per-expert loop, bf16, 3000 tokens, top-8
T, K = 3000, 8
x = torch.randn(T, D, device=dev, dtype=torch.bfloat16)
logits = torch.randn(T, E, device=dev)
top_w, top_i = logits.topk(K, -1)
top_w = top_w.softmax(-1)
gu = gate_up.clone().requires_grad_(True)
dn = down.clone().requires_grad_(True)


def loop_moe():
    flat = top_i.reshape(-1)
    order = flat.argsort()
    token = order // K
    counts = torch.bincount(flat, minlength=E).tolist()
    out = torch.zeros(flat.numel(), D, device=dev, dtype=x.dtype)
    gus, dns = gu.unbind(0), dn.unbind(0)
    start = 0
    for e, n in enumerate(counts):
        if n:
            rows = order[start:start + n]
            y = x[token[start:start + n]] @ gus[e].T
            out[rows] = (torch.nn.functional.silu(y[:, :F]) * y[:, F:]) @ dns[e].T
            start += n
    out = (out.view(-1, K, D).float() * top_w[..., None]).sum(1)
    out.square().mean().backward()
    gu.grad = dn.grad = None


timeit("moe_loop_fwd_bwd_3000tok_ms", loop_moe, reps=3)

if hasattr(torch, "_grouped_mm"):
    def grouped_moe():
        flat = top_i.reshape(-1)
        order = flat.argsort()
        token = order // K
        offs = torch.bincount(flat, minlength=E).cumsum(0).to(torch.int32)
        xs = x[token]
        y = torch._grouped_mm(xs, gu.transpose(1, 2), offs=offs)
        h = torch.nn.functional.silu(y[:, :F]) * y[:, F:]
        o = torch._grouped_mm(h, dn.transpose(1, 2), offs=offs)
        out = torch.empty_like(o).index_copy_(0, order, o)
        out = (out.view(-1, K, D).float() * top_w[..., None]).sum(1)
        out.square().mean().backward()
        gu.grad = dn.grad = None
    try:
        timeit("moe_grouped_mm_fwd_bwd_3000tok_ms", grouped_moe, reps=3)
    except Exception as exc:  # noqa: BLE001
        res["grouped_mm_error"] = repr(exc)[:300]

g = torch.randn(E, 2 * F, D, device=dev, dtype=torch.bfloat16)
timeit("grad_norm_fp64_gate_up_ms", lambda: torch.linalg.vector_norm(g, dtype=torch.float64))
timeit("grad_norm_fp32_gate_up_ms", lambda: torch.linalg.vector_norm(g, dtype=torch.float32).double())
print(json.dumps(res, indent=1))
