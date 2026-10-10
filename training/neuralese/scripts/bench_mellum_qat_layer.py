"""Timing probe of one Mellum QAT warm-up layer (plans/mellum-port.md, "warm-up step profile"): one synthetic
Mellum MoE layer with the real model classes.

Synthetic weights (timings do not depend on values). Runs on the shared GB10 next to the live job, so absolute
times are inflated by contention; ratios are the point.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from natlang_neuralese.maple.model import MapleConfig, SparseMoE  # noqa: E402
from natlang_neuralese.maple import ternary  # noqa: E402
from natlang_neuralese.train import optim  # noqa: E402

torch.manual_seed(0)
dev = "cuda"
cfg = MapleConfig(hidden_size=2304, num_experts=64, moe_intermediate_size=896, num_experts_per_tok=8,
                  dense_experts=True, swiglu_clamp=None)
moe = SparseMoE(cfg).to(dev)
with torch.no_grad():
    moe.experts.gate_up.normal_(0, 0.02)
    moe.experts.down.normal_(0, 0.02)
    moe.gate.weight.normal_(0, 0.02)
moe.gate.weight.requires_grad_(False)
latents = moe.experts.make_latent(quantize=False)
moe.experts.precision_group = "experts"
Q4 = {"experts": ("int4", 0.46, {"group": 32})}
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


# A. fake quant of one layer's experts (both tensors), eager, no grad
def fq_eager():
    with torch.no_grad(), ternary.active_precision(Q4, "q4"):
        a = moe.experts._value(moe.experts.gate_up, torch.bfloat16)
        b = moe.experts._value(moe.experts.down, torch.bfloat16)
    return a, b


timeit("fakequant_layer_eager_ms", fq_eager)
torch.cuda.reset_peak_memory_stats()
base = torch.cuda.memory_allocated()
fq_eager()
torch.cuda.synchronize()
res["fakequant_layer_eager_peak_extra_gb"] = round((torch.cuda.max_memory_allocated() - base) / 2**30, 2)


# B. the same math fused by torch.compile (bf16 in, bf16 out)
def q4_ramp(w, mix):
    q = ternary.q4_0(w, 32)
    x = w.float()
    return (x + mix * (q - x)).to(w.dtype)


q4c = torch.compile(q4_ramp, dynamic=False)


def fq_compiled():
    with torch.no_grad():
        return q4c(moe.experts.gate_up.detach(), 0.46), q4c(moe.experts.down.detach(), 0.46)


timeit("fakequant_layer_compiled_ms", fq_compiled)
with torch.no_grad():
    e = fq_eager()
    c = fq_compiled()
    res["compiled_matches_eager"] = bool(torch.equal(e[0], c[0]) and torch.equal(e[1], c[1]))


# C. bare weight-bandwidth floor: one bf16 read+write of both tensors
def copy_floor():
    with torch.no_grad():
        return moe.experts.gate_up.detach().clone(), moe.experts.down.detach().clone()


timeit("copy_layer_bf16_ms", copy_floor)


# D. MoE layer forward / forward+backward at the token counts of one update
def moe_run(tokens, precision, backward, checkpointed=False):
    x = torch.randn(1, tokens, 2304, device=dev, dtype=torch.bfloat16, requires_grad=backward)
    ctx = ternary.active_precision(Q4 if precision else None, "q4" if precision else None)

    def f():
        with ctx:
            if checkpointed:
                from torch.utils.checkpoint import checkpoint
                y = checkpoint(moe, x, use_reentrant=False)
            else:
                y = moe(x)
            if backward:
                y.float().square().mean().backward()
                for p in latents:
                    p.grad = None
            return y
    return f


for tokens in (1000, 3000):
    for precision in (False, True):
        tag = f"{tokens}tok_{'q4' if precision else 'bf16'}"
        with torch.no_grad():
            timeit(f"moe_fwd_nograd_{tag}_ms", moe_run(tokens, precision, False))
        timeit(f"moe_fwd_bwd_{tag}_ms", moe_run(tokens, precision, True), reps=3)
        timeit(f"moe_fwd_bwd_ckpt_{tag}_ms", moe_run(tokens, precision, True, True), reps=3)

# E. CPU-side launch cost of the per-expert loop: time with the GPU queue not draining
x = torch.randn(3000, 2304, device=dev, dtype=torch.bfloat16)
with torch.no_grad():
    moe(x[None])
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(3):
        moe(x[None])
    cpu = (time.perf_counter() - t) / 3
    torch.cuda.synchronize()
    res["moe_fwd_bf16_3000tok_host_ms_before_sync"] = round(cpu * 1000, 2)

# F. LionSR step on one layer's expert latents (compiled per-row kernel, as in the run)
from natlang_neuralese.maple.qat_convert import row_scale  # noqa: E402
opt = optim.LionSR([{"params": [p], "lr": 3e-4, "row_scale": row_scale(p)} for p in latents], lr=3e-4)
for p in latents:
    p.grad = torch.randn_like(p) * 1e-3


def lion():
    opt.step()


timeit("lionsr_step_layer_experts_ms", lion)


def norm64():
    return sum(torch.linalg.vector_norm(p.grad, dtype=torch.float64).square() for p in latents)


timeit("grad_norm_fp64_layer_experts_ms", norm64)
res["layer_expert_params"] = sum(p.numel() for p in latents)
res["device"] = torch.cuda.get_device_name()
print(json.dumps(res, indent=1))
