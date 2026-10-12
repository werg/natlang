# Training speed, 2026-10-12 (Mellum direct_sft on the DGX GB10)

Owner question (2026-10-12): what can we do to optimize training speed? Measure first, then fix what dominates.
Agent: dgx-perf2. Status: the first round has landed. v4 runs at ~30 s/update, down from 45.7 s.

## Tools

- **In-process profiler** (`common/profiling.py`, commit 9553e4f7). Set `NATLANG_PROFILE_UPDATES=a:b` (train.jsonl
  update numbers, absolute across resumes) on direct_sft. It writes `<stage out>/profile-a-b.json` and a gzipped
  Chrome trace.
  - GPU kernel time is broken down by model scope: attention, attention.sdpa, moe.route, moe.experts, norm, qat,
    readout, projection.
  - Each scope is split into forward, checkpoint recompute (`recompute/`) and backward (`bwd/`). Backward kernels are
    attributed to the forward scope through the autograd sequence number.
  - Optimizer scopes: `optim/muonsr.newton_schulz`, `optim/muonsr`, `optim/adamw`.
  - The report also gives GPU idle time (host gaps) and the top kernels.
  - Unset, every scope is a shared `nullcontext`.
  - Known gap: the `update/backward` phase region does not parent autograd-thread kernels, so read the categories, not
    the phases.
- **Bench** (`train/speed_bench.py`; runs via `runs/training-speed-20261012/bench.sh`). One packed direct_sft update
  on the real Mellum truncated to `--layers`, with the recipe's QAT component, the latent policy and MuonSR stepping in
  backward. Arms are set by env and flags (`--checkpoint`, `--micro-rows`, `--qat`, `--profile`). Fixed-seed batches
  give a fixed-batch loss for parity.
  - Caveat: arms g-/f-/h- ran beside the live Mellum run on the shared GPU. Their absolute times are inflated, so only
    back-to-back pairs compare.
- Microbenches in `runs/training-speed-20261012/`:
  - `muon_bench.py`: one layer's latents, times 28.
  - `ns_bench.py`, `ns_variant.py`: Newton-Schulz forms vs FP64.
  - `attn_bench.py`: SDPA backends and FlexAttention.
  - `gemm_bench.py`, `blas_probe.py`, `fp8_probe.py`, `nvfp4_probe.py`.

## Measured breakdown

### Before (v3 warm-up)

- 45.7 s/update median at ~29.7k forward tokens, layer checkpointing on.
- Bench at 4 layers × 32k tokens, old code: 7.24 s/update, 21 GB peak.
- Scaled to 28 layers per update:

| Component | Seconds per update |
|---|---|
| MuonSR | ~14.4 (Newton-Schulz 7.5, elementwise 6.2) |
| Sliding-window SDPA | ~10 (backward alone ~7.7: masked memory-efficient kernel, 6.6× its forward) |
| MoE experts | ~13 |
| RMSNorm | ~6.3 (eager, NATLANG_MAPLE_COMPILE=0) |
| Attention projections | ~4 |
| GPU idle | ≈0 |

### After (v4 profile, updates 4–6, snapshot b109e9d0)

- ~30 s/update at ~32k tokens. Single-window 32k-token updates take ~40 s: full-attention cost is quadratic.
- Per update (~33 s of GPU):

| Component | Seconds per update | Detail |
|---|---|---|
| MoE experts | 12.9 | fwd 3.7, recompute 2.7, bwd 6.6 |
| Newton-Schulz | 4.1 | |
| Attention projections and rope | 4.0 | |
| SDPA (flex + flash) | 2.7 | |
| Readout | 1.3 | |
| AccumulateGrad | 1.25 | |
| Norm | 1.1 | |
| NeuraleseProjection | 0.9 | |
| MoE route | 0.4 | |
| GPU idle | ~1 | |

The RMSNorm "Mismatch dtype" warning comes from NeuraleseProjection's `nn.RMSNorm` (span positions only), not from the
backbone, whose norm gains are BF16.

Fixed per-update cost: the old MuonSR step was 13.7 s of the ~16 s fixed cost. It is now 5.4 s on the microbench and
~5.6 s in v4's profile (Newton-Schulz plus elementwise).

## Changes (each with an env switch that restores the old path)

| Commit | Change | Measured | Numerics |
|---|---|---|---|
| 21ddf968 | Fused Triton RMSNorm fwd/bwd (`NATLANG_MAPLE_FUSED_NORM=0`) | norms 6.3 s → ~1.1 s/update (v4 profile) | outputs ≤1 BF16 step on ~5e-6 of elements; grad_x within 5% of row RMS on ~1e-5; checkpoint recompute bit-identical |
| 7ae8b5cb | MuonSR fused elementwise step (`NATLANG_MUON_FUSED=0`) | MuonSR 13.7 → 9.0 s/update | bit-identical (weights, momentum, expert grad norms) |
| 9c38e35e | Newton-Schulz cubic5 folded: M = aI + bXXᵀ by baddbmm, X ← MX (`NATLANG_MUON_NS_FOLDED=0`) | 9.0 → 5.4 s/update | closer to FP64 than before: 0.0067 vs 0.0095 (gaussian), 0.089 vs 0.119 (low-rank) |
| e7d1f7f0 | Predecessor state archived in a background thread; disk reserve deferred until it finishes | removes a 20+ min GPU-idle stall per phase hand-off | none |
| 01242289 | Sliding-window attention via FlexAttention, compiled with dynamic shapes (`NATLANG_FLEX_WINDOW=0`) | 395 → 52 ms fwd+bwd per sliding layer (8×4000); 2-layer bench 1.85 → 1.41 s | fixed-batch loss 15.26321 vs 15.26315; grads within 1% |
| a5fd7e21 | hdd_guard stages only data roles; load-once state read from the HDD (`read_from_hdd`); staged cache bounded LRU | avoids re-copying 44 GB checkpoints to the NVMe | none |
| 493c732d | Scatter-free MoE permutation glue (`NATLANG_MOE_GLUE=0`) | 2-layer bench 2.85 → 2.52 s; peak 9.98 → 8.15 GB | forward bit-identical; grads within 5e-3 |
| (this) | `maple/fp8_experts.GroupedLinear`: expert weight grads in parameter layout; optional FP8 (E4M3 row-scaled cuBLASLt per expert) forward and dgrad (`NATLANG_MOE_FP8=1`, default off) | FP8 is a wash at 32k tokens (4.016 vs 4.017 s, 2 layers) | FP8 fixed-batch loss 15.22067 vs 15.2217 |

Combined, v4 (which carries the commits through 01242289) went from 45.7 s to ~30 s per 32k-token update.

## Arms measured and not adopted

- **No checkpointing / micro-batched without checkpointing.** v3 data: non-checkpointed activations take ~9.5 MB/token
  (31 GB incremental at 3.3k tokens) vs ~0.55 MB/token checkpointed.
  - Accumulating gradients over micro-batches needs full BF16 latent gradient buffers (~22 GB), because in-backward
    stepping cannot accumulate.
  - Weights + momentum + grads + one 4k micro-batch ≈ 100 GB, which is over the claim.
  - A planner choice between the two is not worth it at this geometry. Even 1–2 non-checkpointed layers save ~1–2%.
- **Calibration inheritance across phases.** Already implemented (continue_from restores the estimator). It cannot
  change the decision, for the same memory reason.
- **QAT fake-quant once per update.** Attention weights are already built once per pass (`shared_parametrized_weights`).
  The fake-quant is one fused kernel. Packed mode runs one forward per update. Caching the expert values would hold
  ~22 GB through the backward to save one fused pass per update (<0.5 s).
- **torch.compile of the glue (`NATLANG_MAPLE_COMPILE=1`).** With the fused norm and Flex in place, the 2-layer bench
  shows no gain: 1.538 vs 1.407 s.
- **FP8 expert GEMMs** (owner 2026-10-12: try; numerics may change).
  - `torch._scaled_grouped_mm` is not built for sm_121.
  - cuBLASLt `_scaled_mm` reaches 64–79 TFLOP/s against BF16's 38–40, which is the GB10 BF16 GEMM ceiling for all
    layouts and for cuBLAS and cuBLASLt alike.
  - A per-expert FP8 loop is implemented. At 32k tokens its GEMM gain is eaten by the memory-bound quantization of
    activations (rowwise, ~10 ms per call) and of the weights (twice per call).
  - Next steps to make it pay:
    - quantize x per token before the gather (8× less work);
    - fuse the SwiGLU into the down projection's quantization;
    - cache the per-update weight quantization between forward and recompute;
    - FP8 wgrad.
  - Estimated ceiling: ~2 s/update.
- **NVFP4** (sm_121, container 26.09):

  | Path | TFLOP/s at 8192×2304×1792 |
  |---|---|
  | `torch._scaled_mm` with `float4_e2m1fn_x2` and E4M3 block scales | 94 |
  | MXFP8 block-scaled | 58 |
  | Triton `tl.dot_scaled` MXFP4 | 82 |

  Transformer Engine 2.19 reports NVFP4 available (`NVFP4BlockScaling`). NVFP4 is 2.4× BF16 on the GEMM. A grouped
  Triton `dot_scaled` kernel could fuse dequant and quant. The memory-bound quantization remains the limiter, as for
  FP8. Estimated gain over a tuned FP8 forward: ~1 s/update.

## Recommended configuration for the next resume

Code: origin/main at or after the commit that adds this file. No env needed: every adopted change is on by default.
Keep `NATLANG_MAPLE_COMPILE=0` (no measured gain) and leave `NATLANG_MOE_FP8` unset. Profile any resume with
`NATLANG_PROFILE_UPDATES=<next+3>:<next+5>`.

## Next (largest remaining phases, v4 profile)

1. MoE expert GEMMs (~8 s): the FP8 fusions above, then NVFP4 forward.
2. AccumulateGrad 1.25 s/update (still present with contiguous expert grads): find the parameter.
3. Attention projections and rope (4 s): fused QKV + rope.
4. Readout (1.3 s): fused linear + cross-entropy on supervised positions only.
5. NeuraleseProjection's soft-token mixture over the 98k vocabulary at every span position (0.9 s).

## Round 2 (2026-10-12 afternoon)

- **Profiler fix (48db6d59).** The v4 profile's "AccumulateGrad 1.25 s/update" was MuonSR. The latents step inside
  post-accumulate-grad hooks, and a record_function entered in a hook is not a cpu_parent of the ops under it. Real
  AccumulateGrad work is ~19 ms/update. The optimizer is therefore ~6.8 s/update, not 5.4. Scopes are now attributed
  by time nesting on the launching thread, and each report lists its top kernels per category.
- **GB10 GEMM rates.** The earlier "38–40 TFLOP/s BF16 ceiling" was contention with v4. Measured on an idle GPU at
  16k × 2304 × 1792:

  | Kernel | TFLOP/s |
  |---|---|
  | Triton FP8, pre-quantized, 128x128x64 | 173 |
  | cuBLASLt FP8 | 111 |
  | Triton BF16, 128x256x64, grouped-M swizzle | 97 |
  | cuBLAS BF16 | 59 |
  | Triton with in-kernel tile quantization | 57 (dropped) |

- **FP8 expert GEMMs.**
  - The cuBLASLt per-expert prototype was removed.
  - `maple/expert_gemm.py` has one Triton launch per projection over all experts.
  - Activations are quantized per token before the routing gather, and the kernel gathers FP8 rows itself.
  - Weights are quantized per row. The input gradient reuses the forward's FP8 weights, with their row scales folded
    into the gradient's quantization.
  - The weight gradient is a BF16 grouped GEMM in parameter layout. A Triton reduction kernel for it measured 2–5×
    slower.
  - Selection goes through the shared FP8 backend (`train/fp8_backend.py`, Pop's 10cc424d):
    `--fp8-training triton-experts` (`FP8ExpertAdapter`). It runs FP8 inside `autocast()` (training and recompute)
    and BF16 under `native_modules()` (evaluation).
  - It is stateless (current scaling), so the trainer and the recipe treat it as an execution choice like
    `moe_kernel`: a run switches it at a resume.
  - It is the default for Mellum (recipe neuralese-direct-mellum-v4; line-parity backbone-inherent).
  - Bench, 2 layers × 32k tokens, interleaved: 3.97 s vs 3.81 s per update (−4%). The MoE expert phase is a minority
    of a 2-layer update, so the full-model gain is larger in absolute terms.
  - Fixed-batch loss: 15.22067 vs 15.2217 (BF16). No NaN.
- **Resume fixes.**
  - 74a01986: input digests are cached by (realpath, size, mtime_ns), and an archived predecessor takes the digest
    from its receipt or the archive manifest. A resume never re-reads the 47 GB state.
  - 00b524d7: the resume identity is content-based (sorted input digests and options without input paths).
- **Shared tuned-BF16 GEMM dispatch: measured, not adopted.** Each shape on the hot path was measured beside the live
  v4 run (`runs/training-speed-20261012/dispatch_bench.py`, Triton swizzled kernel vs cuBLAS at 32k tokens):

  | GEMM | Triton vs cuBLAS |
  |---|---|
  | q forward | 0.80× |
  | q dgrad | 1.30× |
  | q wgrad | 0.88× |
  | k/v forward | 0.96× |
  | k/v dgrad | 1.17× |
  | k/v wgrad | 0.78× |
  | o forward | 1.01× |
  | o dgrad | 0.94× |
  | o wgrad | 1.51× |
  | Newton-Schulz gram | 0.83× |
  | Newton-Schulz M·X | 1.06× |
  | Newton-Schulz down gram | 0.89× |

  Relative error vs FP32 is 1.7e-3 for both. A per-shape dispatch would take only the wins: ≈8 ms per layer, ~0.2–0.3
  s/update (~1%). That is below the 5% bar, so it is not wired in. The idle-GPU 97 vs 59 TFLOP/s gap (one 16k ×
  2304 × 1792 shape) does not carry over to these shapes under load.
- **Remaining candidates (v4 profile).** Each is about 1–3% of the update:
  - Newton-Schulz 4.1 s. A Gram-iteration form would save ~20% of it, but its numerics need checking.
  - Attention projections, q/k norms and rope (~4 s): fusing QKV + rope.
  - Readout 1.3 s.
  - NeuraleseProjection 0.9 s. It recomputes the span logits that the readout already computed; sharing them would
    save about half.
