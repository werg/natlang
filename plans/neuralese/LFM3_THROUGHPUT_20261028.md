# LFM3-1.7B joint-SFT throughput, first time box (2026-10-28)

Goal: the fastest correct configuration of the shared native SFT trainer (`train/native_state_sft.py`) for the LFM3
joint SFT stage: BF16 stochastic-rounding profile, MuonSR on every layer matrix, AdamWSR on norms, conv kernels
and learned rows.

## Setup

- DGX Spark GB10 (sm_121, unified memory), torch 2.14 / CUDA 13. Ledger units `lfm3-tp-*`, `--priority high`.
- Data: the real LFM3 communication graphs, `/mnt/external/natlang-lfm3-data-20261028/communication/rest27-reviewed/
  graphs/train/native-graphs.jsonl.zst`. That is 135 two-node producer/consumer graphs of about 6.6k tokens each; the
  default `batch_tokens` packing gives about 13.2k tokens per update. Full backbone training (1.47B MuonSR latents),
  layer activation checkpointing on.
- Harness: a scratch copy of `eval/native_train_benchmark.py` that runs `native_training_update` and
  `common/profiling.UpdateProfiler` for the breakdown. Gradient checks run on the first batch without an update
  (`packed_graph_loss` + backward). They compare against an **FP32 reference gradient**: the same model loaded in
  FP32 on the same batch. Reported as global cosine and relative L2 over all trainable parameters.
- MFU basis: 6·N·T with N = 1.47B. Measured BF16 GEMM peak on GB10 is about 68 TFLOPS (4096^3).

## Baseline (main 06219535c)

| Metric | Value |
|---|---|
| Step (13.2k tokens) | 41-57 s |
| Throughput | ~300 tok/s (2.4 TFLOP/s model, 3.5% of the measured GEMM peak) |
| GPU busy / idle | 17.8 s busy / 39.2 s idle of a 57 s update |
| Biggest items | 5.5 s DtoH pageable copies; the fixed-tile Triton linear `_grouped_dense_mm` 2.6 s; NS 1.44 s |
| Peak CUDA | 10.3 GB |

The cause was the `cpu_file` parameter-cotangent carrier, the historical default of `native_training_update`. It
accumulates every BF16 parameter's cotangent in FP32 into file-backed CPU tensors. The GPU waited 39 s on host
writes per update. On top of that, the fixed-tile "invariant" linear replaced cuBLAS and split every dW per
invocation.

## Changes, in order (each measured on the same batches)

| # | Change | s/update | tok/s | vs FP32 gradient (cos / rel L2) | Peak GB | Status |
|---|---|---|---|---|---|---|
| 0 | baseline (`cpu_file` carrier, fixed-tile linear) | 41-57 | ~300 | 0.9757 / 0.219 | 10.3 | main |
| 1 | `--cotangent-accumulator-backend device_fp32` | 13.0 | ~1,000 | bit-identical losses to 0 | 10.3 | flag already on main |
| 2 | **`plain` backend (new default)**: ordinary autograd, cuBLAS forward/dX/dW, no carrier | 9.9 | 1,300 | **0.9840 / 0.178** | 10.7 | branch |
| 3 | **+ torch.compile of layer inputs/finish/finite conv (default in plain mode)** | 7.8 | 1,690 | 0.9850 / 0.173 | 10.0 | branch |
| 4 | **+ `--chunk-tokens` 1024 for new runs (was 128)** | 7.2 | 1,850 | exact (readout chunking) | 10.0 | branch |
| 5 | + 32k tokens per update (multi-graph packing) | — | 1,970 | — | 13.5 | config |
| 6 | + 64k tokens per update | 26-31 | 1,990 | — | 22.3 | config |
| — | no activation checkpointing (13.2k) | 6.6 | 1,985 | exact | 48.0 | not adopted (below) |

Numerics: in the BF16 profile, the plain path is closer to the FP32 gradient than the carried path it replaces.
Every trainable layer matrix is used once per layer-major graph pass, so one GEMM with FP32 accumulation and one
rounding is at least as accurate as per-invocation BF16 dW sections through the FP32 carrier. Losses over 4 real
updates track the baseline within 0.006 (5.091/4.479/3.828/3.631 baseline, 5.101/4.478/3.828/3.627 final config).

### Why each change is safe and how it defaults

- `plain` (model/native_weight_carrier.py `COTANGENT_BACKENDS`, `plain_training_arithmetic`): the default for new runs
  and for resumes. A resume records the transition in `cotangent_accumulator_history` with weights/optimizer/RNG/
  exposure preserved. Moving to plain needs no capacity re-qualification, because it holds no accumulators.
  `device_fp32`/`cpu_file` remain as an explicit opt-out. Adapter-fit scopes that require a carrier use device FP32
  when the model is plain (`carrier_backend`). The plain flag is process-wide, not a ContextVar: checkpoint
  recomputation runs on autograd device threads, and a ContextVar there broke recomputation with metadata mismatches.
- Compile (`model/shared_state._plain_compiled`): only while a plain update runs, and only on CUDA. Inference,
  reference arithmetic, exact carrier scopes and CPU execution keep the eager functions. Opt out with
  `NATLANG_NATIVE_COMPILE=0`.
- `--chunk-tokens`: omitted means 1024 for new runs. A resume keeps its saved value, or 128 for checkpoints from
  before it was recorded, because chunking is part of the resident geometry identity.

## Profile of the final DGX configuration (one 13.2k-token update, 7.13 s wall)

GPU busy 6.96 s, idle 0.18 s: the host is no longer a bottleneck.

| Item | GPU s | Note |
|---|---|---|
| Newton-Schulz (quintic, 5 steps, BF16) | 1.44 | about 70% of GEMM peak already; fixed per update, so it amortizes with batch tokens |
| GEMMs fwd + recompute + bwd | ~3.0 | ideal ~2.3 s at peak; the recompute share is ~0.5 s |
| Flex attention fwd+bwd | 0.41 | block-sparse graph masks, already sliding-window aware |
| Readout softmax/CE (128k vocab) | ~0.4 | |
| Width-7 finite convolution (gather + fused mul-sum + scatter backward) | ~0.3 | Inductor fuses it; a dedicated Triton kernel would save <0.2 s, not worth it now |
| Remaining elementwise/copies | ~1.4 | |

Data/collation and the graph-mask build are inside the 0.18 s of idle time.

## Recommendations for the DiLoCo launch

- Flags: default backend (`plain`), `--chunk-tokens 1024` (or omit it on a new run), `--checkpoint-layers` on.
  Use `--batch-tokens` of 32k-64k: NS is a fixed ~1.4 s per update on GB10, and the RTX PRO 6000 is proportionally
  faster. Memory is not the limit (13.5 GB at 32k).
- `local_steps`: sync volume for the full backbone is ~3.4 GB BF16 on the wire, plus the merge. Pick `local_steps`
  so that transfer + merge is at most 5% of the round. This needs the sandbox step time and link speed (below).

## Sandbox measurement (RTX PRO 6000 Blackwell, 96 GB, 2 CPUs Xeon 8562Y+)

Measured by the launch agent (`af805723812a2b210`), one sandbox shared rather than a second one, on the code before
24225f416 (`device_fp32` carrier, chunk 1024, 32k batch tokens, BF16-SR, MuonSR, EMA, matrix telemetry):
- 4.8 s/update, 6.6-6.8k tok/s per GPU (3.4-3.6x the GB10 at the same configuration class).
- GPU memory about 20.5 GB in nvidia-smi; torch peak allocated 15.1 GB.
- Host cgroup RAM: 8 GB steady, 16 GB after held eval, peak 26.8 GiB around the end-of-stage checkpoint save and
  reload (full-state checkpoint 12.26 GB).
- 2 CPUs at load ~1.0 with GPU utilization 98-100% at 1 s sampling: no loader starvation seen. Not measured: idle
  gaps within an update. The plain+compile seed run on 24225f416 replaces these numbers when it reports.
  Expectation from DGX: the carrier removal plus compile was 1.8x there. Part of that was host-bound file I/O
  that `device_fp32` had already removed, so expect roughly 1.3-1.5x on the sandbox.

Recommendations:
- Learner RAM minimum **48Gi**: the measured 26.8 GiB peak at checkpoint save/reload, plus room for a held eval
  overlapping a background snapshot. The provisional 64Gi template value is also fine.
- `local_steps`: a round is local_steps × step time. With about 4-4.8 s/update at 32k tokens, 128 local steps is a
  ~510-615 s round. The 5% rule then allows 26-31 s for pseudo-gradient transfer (3.4 GB BF16 each way) plus the
  merge. Keep **128** if the first round's measured transfer + merge is ≤ 25 s, which needs ≥ ~270 MB/s each way.
  Otherwise use local_steps = ceil(20 × (transfer + merge seconds) / step seconds), rounded up to a multiple of 32.
  Recompute after the seed run reports its plain-path step time.

## Plain + compile on the sandbox (seed run, 24225f416)

RTX PRO 6000, 2 CPUs, 48Gi, 32k tokens per update, own-output objective on joint-sft-v1: median **2.57 s/update,
12.3k tok/s** over 200 updates (6.7k before, with `device_fp32`). CUDA peak 20.9 GB, host cgroup peak 16.5 GiB,
GPU at 100% outside eval/checkpoint (2 CPUs do not starve it), first update after ~20 s. At 2.57 s/update the 5% rule
allows ~16 s of sync per 128-step round; size `local_steps` from the round-1 sync measurement.

## QAT in native SFT (FP8 + NVFP4, on by default for LFM3)

The fork's inline installation (`Quantization.install(model, inline=True)`: plain parameter names, `--quantization`
with `bind`, loss mode `sample`, calibration in the full state) plus:
- **Default for LFM3 SFT**: with `--quantization` omitted, an LFM3 run adopts the recipe component
  (`native_state_sft.default_lfm3_quantization`): FP8 0-0.15, NVFP4 0.1-0.5, bf16/fp8/nvfp4 weights 1/0.5/1. The
  bind covers the remaining planned updates of this process (DiLoCo: the learner's own shard), recorded as
  `quantization_history` `default-adopted` at that global update. A resume continues the recorded declaration;
  `--quantization off` opts out.
- **One weight fake quant per pass** (`maple/ternary.inline_precision`): the ramped weights `W + λ(Q(W) − W)` are
  computed once on entering the pass, by one compiled kernel set per weight shape, and reused by
  activation-checkpoint recomputation. λ is a device scalar, so the compiled layer pieces fuse the activation codecs
  without recompiling as the ramp moves. The recompile limit is raised to 256, because the default 8 sent later
  layers back to eager.
- **Export parity**: the compiled weight ramp uses eager division rounding, no flush-to-zero and emulated precision
  casts. With Inductor's default approximate division, 3,672 of 15.7M NVFP4 codes of one weight differed, and the
  FP8 BF16-division double rounding was skipped. At λ = 1 the trained weights equal the BF16 values of
  `export.lfm3_quantized` codes (`test_in_training_deploy_weights_equal_liquid_exports`, real LFM3 layers 0/13/23,
  w1/w2/w3, both formats), and those reproduce Liquid's checkpoints bit for bit. Activation codecs inside the
  compiled layers keep Inductor's division, like the deploy kernels' approximate reciprocals. Emulating eager there
  cost 1.2x per NVFP4 pass.
- **CPU/CUDA codec agreement** (`train/qformats._div`, `_true_div`): torch's CUDA tensor/scalar division is a
  reciprocal multiplication, and Liquid's weight scales match only that. Weight scale rules now use it on every
  device. Activation scales use correctly rounded division, as vLLM does.
  `test_liquid_codes_and_scales_reproduce_exactly` now also passes on CPU.

Cost per 13.2k-token update on GB10 (plain + compile, chunk 1024; 7.25 s at BF16):

| Pass | s/update | vs BF16 |
|---|---|---|
| FP8 λ = 1 | 7.3-7.4 | +2% |
| NVFP4 λ = 1 | 8.4 | +16% |
| Sampled mix (weights 1/0.5/1) | ~7.7 expected | ~+6% |

## Remaining ideas, ranked by expected gain

1. Larger `--batch-tokens`: NS falls from 20% of the update at 13k tokens to ~5% at 64k. Config only. GB10
   measured 1,970 tok/s at 32k and 1,990 at 64k, so the gain flattens past 32k on DGX.
2. Real FP8 GEMMs in the forward once the FP8 QAT point is at λ = 1 (rowwise FP8 is 1.9-2.1x BF16 on both GPUs).
   This is not an exact match to the FP8_BLOCK deploy scaling, which the hardware does not support, so it needs its
   own parity check. Could be ~25-35% of GEMM time.
3. NVFP4 activation codec (+16% per NVFP4 pass, ~6% averaged): a hand-written Triton group quantizer, or real
   NVFP4 GEMMs (`F.scaled_mm`, 4.5x on GB10, 2.8x on the sandbox) once λ = 1.
4. Batched Newton-Schulz over same-shape matrices after backward (~0.2-0.3 s of 1.44 s).
5. Selective activation checkpointing in the graph forward (keep the top K layers): at most ~7% (no checkpointing at
   all measured 6.6 vs 7.1 s), at ~3 GB per 1k tokens.
6. Fused linear + cross-entropy readout (vocab 128k): ~0.3 s.
