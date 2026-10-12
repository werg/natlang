# CoreWeave hardware optimization: RTX PRO 6000 Blackwell (sm_120)

Status 2026-10-12 (DGX, agent dgx-cw-hwopt). Owner goal: "diligently optimize everything we do on the sandboxes for
the hardware we have there to achieve fast training and inference."

The sandboxes, the controller, the bootstrap and the bundles are in [COREWEAVE_SANDBOXES.md](COREWEAVE_SANDBOXES.md)
(another agent owns them). This plan covers what runs inside a sandbox and how fast it runs. Everything that can be
built without the hardware is built (section 7). What only the first sandbox can decide is listed in section 8, and the
harness that decides it runs as a job as soon as the GPU smoke passes.

## 1. Hardware facts

One NVIDIA RTX PRO 6000 Blackwell Server Edition per sandbox (GB202, compute capability 12.0), about 20 sandboxes at
once, an x86_64 host, PCIe, no NVLink.

Sources:
- **WP**: NVIDIA, [RTX Blackwell PRO architecture whitepaper v1.1](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/quadro-product-literature/pdf/NVIDIA-RTX-Blackwell-PRO-GPU-Architecture-v1_1.pdf), Appendix A.
- **DS**: the [Server Edition datasheet](https://www.akamai.com/site/en/documents/corporate/nvidia-rtx-pro-6000-blackwell-server-edition-datasheet.pdf).
- **Lenovo**: the [Lenovo Press guide](https://lenovopress.lenovo.com/lp2263-thinksystem-nvidia-rtx-pro-6000-blackwell-server-edition-pcie-gen5-gpu).
- GB10 rows: our own measurements ([TRAINING_SPEED_20261012.md](neuralese/TRAINING_SPEED_20261012.md)).

| Item | RTX PRO 6000 Server | DGX GB10 (for comparison) | Source |
| - | - | - | - |
| SMs | 188 (24,064 CUDA cores, 752 5th-gen tensor cores) | 48 | WP, DS |
| BF16/FP16 dense, FP32 accumulate | ≈ 500 TFLOPS (Workstation 503.8; Server lists 1 PF "peak", i.e. sparse) | measured: 59 cuBLAS, 97 Triton | WP, Lenovo |
| FP8 dense | ≈ 1,000 TFLOPS (Workstation 1,007.6) | measured: 111 cuBLASLt, 173 Triton | WP, Lenovo |
| FP4 / NVFP4 dense | ≈ 2,000 TFLOPS (Workstation 2,015; Server "4 PF" sparse) | measured: 94 `_scaled_mm` NVFP4 | WP, DS |
| FP32-accumulate rate | full rate on RTX PRO (halved only on GeForce 5090) | n/a | WP, [arXiv 2608.27370](https://arxiv.org/pdf/2608.27370) |
| Memory | 96 GB GDDR7 ECC, 512-bit, **1,597 GB/s** (Workstation 1,792) | 121 GB unified LPDDR5X, ~273 GB/s | DS, WP |
| L2 | 128 MB | 24 MB (unverified) | WP |
| Shared memory | 128 KB L1 per SM; max **100 KB per SM / 99 KB per block** for CC 12.x (CC 10.0: 228/227 KB) | the same CC 12.x limits | [CUDA CC table](https://docs.nvidia.com/cuda/archive/13.2.1/cuda-programming-guide/05-appendices/compute-capabilities.html) |
| Power | up to 600 W (can be capped at 450 W) | ~140 W | DS, Lenovo |
| Host link | PCIe Gen5 x16, ~64 GB/s per direction | unified (no copy) | DS |

The Server boost clock is not published. FP32 is 120 against the Workstation's 126 TFLOPS, so plan with about 95% of
the Workstation tensor rates, and achieved rates of 70–80% of that on large GEMMs.

### Ratios that drive every decision

- **Compute: ~4–6× the GB10** on the GEMMs we run (BF16 ~350–400 against our measured 59–97 TFLOP/s; FP8 ~600–700
  against 111–173).
- **Memory bandwidth: 5.85× the GB10** (1,597 against 273 GB/s). Our memory-bound kernels speed up by about that.
  These are the optimizer elementwise passes, norms, quantization, routing glue and the readout softmax.
- **Memory capacity: 96 GB discrete against 121 GB unified.** The GPU has less memory than the GB10, but nothing else
  lives in it. Host RAM is separate and billed per GiB-hour.
- **Launch overhead does not scale.** A kernel that took 300 µs on the GB10 takes 60 µs here, so the CPU-side launch
  cost (5–10 µs) becomes a visible share. Steps that the GB10 hid behind slow kernels (GPU idle was 0.2 s of a 16 s
  update) can become 5–15% of a 3–4 s update. CUDA graphs and compile matter more here than on the GB10.

### sm_120 vs sm_100: what is missing, and what it means for kernels

- **No tcgen05, no TMEM, no 2-CTA MMA.** sm_120 runs warp-level `mma.sync` (like sm_80/sm_90 without wgmma), extended
  with block-scaled FP8/FP6/FP4 (`kind::mxf4nvf4.block_scale`, sm_120a). Kernels written for B200 (sm_100) do not run
  ([Colfax sm12x NVFP4 tutorial](https://research.colfax-intl.com/cutlass-tutorial-nvfp4-blockscaled-gemm-on-nvidia-rtx-pro-blackwell-gpus-sm12x/)).
  - FlashAttention 4's fast path is out.
  - So are DeepGEMM and CUTLASS sm_100 grouped GEMMs.
  - B200-tuned TMA warp-specialized tiles do not fit the 99 KB of shared memory.
- **TMA, clusters and Cluster Launch Control exist.** `compute_120f` covers both sm_120 and sm_121 (the GB10).
- **Consequence for us: sm_120 and the GB10's sm_121 are the same kernel target.** Our Triton kernels were tuned for
  CC 12.x shared memory (BM/BN/BK 128/128/64, 3 stages), so they run unchanged:
  - the FP8 experts (`maple/expert_gemm.py`);
  - fused RMSNorm, rope, SwiGLU and gradient quantization;
  - the Newton-Schulz kernels;
  - FlexAttention.

  Retuning block sizes for 188 SMs and a 128 MB L2 is an optimization; the bench's arm results show whether it is
  needed. Kernels that only exist for sm_90/sm_100 stay out of reach, and we do not depend on any of them.

### Library support on sm_120 (as of 2026-10)

| Library / kernel | sm_120 status | What we use | Source |
| - | - | - | - |
| PyTorch 2.14 (NGC 26.09), CUDA 13.4 | arch list includes `sm_120` (measured on our image) | yes | COREWEAVE_SANDBOXES.md §1 |
| Triton 3.8 | supported (sm_120 needs ≥ 3.3) | all our custom kernels, FlexAttention | [PyTorch forum](https://discuss.pytorch.org/t/pytorch-support-for-sm-120/222119) |
| SDPA flash / cuDNN / mem-efficient | flash (FA2) and cuDNN work on sm_120 | full-attention layers | [FA README](https://github.com/Dao-AILab/flash-attention) |
| FlashAttention 3 | Hopper only | no | FA README |
| FlashAttention 4 (CuTe DSL) | upstream now has `flash_fwd_sm120`/`flash_bwd_sm120`, about FA2 speed (mma.sync), head_dim ≤ 128 | no gain over SDPA flash, not adopted | [interface.py](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py) |
| `torch._grouped_mm` (BF16) | **runs, but through `_grouped_mm_fallback`**: a per-group mm loop. The CUTLASS fast path is sm90/sm100 only; the GB10 already used this fallback | BF16 expert path and the FP8 path's wgrad | [GroupedBlas.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/GroupedBlas.cpp) |
| `torch._scaled_grouped_mm` | **errors** ("compute capability = [9.0, 10.0]") | no; the reason our Triton FP8 grouped kernel exists | same |
| cuBLASLt FP8 `_scaled_mm` | supported on 12.x (TN only) | probe only | [cuBLAS docs](https://docs.nvidia.com/cuda/cublas/index.html) |
| cuBLASLt MXFP8 / NVFP4 block-scaled | listed for "10.0+"; works on sm_121 (we measured NVFP4 94 TFLOP/s on the GB10) | probe only | cuBLAS docs |
| cuBLASLt grouped matmul | 9.0, 10.x, 11.0 only, **not 12.x** | no | cuBLAS docs |
| Transformer Engine 2.19 | FP8 delayed/current scaling work. **MXFP8 disabled on ≥ 12.0** (`check_mxfp8_support`). NVFP4 recipe passes its check | `te-fp8` backend (LFM, informational) | [TE quantization.py](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/quantization.py) |
| torchao float8 | rowwise/tensorwise need SM ≥ 8.9, so fine; MXFP8 MoE training is SM100+ | no (our Triton FP8 is the one implementation) | [torchao #3163](https://github.com/pytorch/ao/pull/3163/files) |
| CUTLASS 4.6 | sm_120 grouped GEMMs: NVFP4 (example 79d), tensor/token-scaled FP8 ptr-array grouped GEMM (MoE), blockwise | candidate for the expert wgrad (section 2.5) | [CUTLASS CHANGELOG](https://github.com/NVIDIA/cutlass/blob/main/CHANGELOG.md) |
| DeepGEMM | sm_90/sm_100 only | no | [CloudRift](https://www.cloudrift.ai/blog/optimizing-qwen3-coder-rtx5090-pro6000) |

The hw_bench probe (section 6) re-measures each of these on the real part: rates, and which calls work.

## 2. Training plan: Mellum2.1-12B-A2.5B and LFM2.5-350M on one 96 GB GPU

### Rule: one implementation, hardware-selected operational knobs only

The recipes, objectives, data and trainers stay identical across DGX, Pop and CoreWeave (AGENTS.md, line parity).
What may differ by GPU is declared in **`training/neuralese/hardware-profiles.json`**, read by
`natlang_neuralese/train/hardware_profile.py`:

- **`stage_parameters`**: only `checkpoint_layers` and `window_execution`. Both are members of the recipe runner's
  `OPERATIONAL_STAGE_PARAMETERS`, so a lineage can hand off between machines in place without changing its identity.
- **`environment`**: only allocator, compile-cache and numerics-preserving compile switches (`PYTORCH_CUDA_ALLOC_CONF`,
  `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`, `NATLANG_MAPLE_COMPILE`, `CUDA_DEVICE_MAX_CONNECTIONS`).
- **`sandbox`**: recommended CPU, memory and scratch per job kind (documentation for `scripts/coreweave/jobs`).

How a profile is applied:
- A profile is selected by compute capability, minimum memory and integrated/discrete, or by
  `NATLANG_HARDWARE_PROFILE=<name>|none`.
- The validator refuses anything else, so `batch_tokens`, learning rates and FP8 numerics cannot become
  hardware-dependent.
- The recipe runner (`train/recipe.py`, both the lineage loop and the declared direct-stage path) applies the profile
  to the child command only.
- The plan keeps the recipe's own parameters, so the resume comparison is unchanged. It records `hardware_profile`
  (profile, device, each applied key with the recipe value it replaced) beside them.
- **The GB10 and Pop profiles are empty, so nothing changes for running lineages.**

### 2.1 Where the time goes today, and what this GPU changes

Mellum v4 on the GB10 (rounds 1–5): **15.96 s/update at 26.8k context tokens, 1,682 tokens/s.** The round-4 profile,
per update (19 s of GPU):

| Component | GB10 s | Bound by | Expected factor here | Expected s |
| - | - | - | - | - |
| Expert GEMMs (FP8 fwd/dgrad, BF16 wgrad) + glue | 6.9 | compute (GEMM) + memory (quant, gather, combine) | 4–5× | 1.4–1.7 |
| Attention projections, q/k norm, rope | 4.5 → ~2.5 after round 3 | compute | 4× | 0.6 |
| Newton-Schulz (MuonSR) | 3.95 → ~2.9 after Triton NS | compute (small GEMMs: launch- and occupancy-bound) | 3–4× | 0.8–1.0 |
| Other MuonSR (apply, momentum) | 1.4 → ~0.75 | memory | 5.8× | 0.13 |
| SDPA fwd/bwd (Flex window + flash) | 2.6 | compute | 4× | 0.65 |
| Norms, readout, projection, routing | ~2.1 | memory | 5–5.8× | 0.4 |
| Launch gaps (GPU idle) | 0.2 | host | 1× (does not shrink) | 0.2–0.6 |

**Expectation for Mellum: ~3.5–4.5 s per 26.8k-token update, i.e. 6,000–7,500 context tokens/s per sandbox (3.5–4.5×
the GB10).**
- With `train_target_tokens` 31,744 nearly every position of a whole-trajectory window is supervised. Supervised
  tokens/s is therefore about context tokens/s minus the per-window prefix.
- **Target: ≥ 6,000 supervised tokens/s; stretch 8,000.** The stretch needs the launch-overhead work in 2.4 and the
  wgrad work in 2.5.
- 20 sandboxes running independent arms give ~70× the GB10's aggregate Mellum throughput. One lineage is still one GPU,
  because there is no multi-GPU training without CKS (COREWEAVE_SANDBOXES.md §3.7).

**Expectation for LFM2.5-350M:**
- The work is ~6 × 0.35B × 16k tokens ≈ 3.4e13 FLOP, plus a 65k-vocabulary readout, per update.
- At 300 TFLOP/s that is ~0.15 s of GEMM. The real step is host- and launch-bound (16 layers, many small kernels,
  MuonSR over small matrices). Expect 0.3–0.5 s per 16k-token update: **35–55k supervised tokens/s** before
  compile/graph work.
- **Target ≥ 50k tokens/s.** On this GPU LFM is dominated by fixed per-update cost, so it benefits most from larger
  `batch_tokens` (2.2) and from CUDA graphs (2.4).

### 2.2 Batch and window token budgets

- **`batch_tokens` (32,768 Mellum, 16,384 LFM) is training math: it sets the update size.** It stays in the recipe
  and is never hardware-selected.
- The bench measures 16k, 32k and 64k per update (arms `m-16k`, `m-ref`, `m-64k`, `l-ref`, `l-32k-nockpt`). That
  shows how much of the update is fixed cost: the optimizer, at ~1 s for Mellum here, amortizes over twice the tokens
  at 64k. Expected: +10–20% tokens/s for Mellum at 64k, +50–100% for LFM at 32k.
- If the gain is large, raising `batch_tokens` is a **recipe decision for both lines together** (LR and β₂
  half-life are tied to tokens; `beta_for_half_life` already scales β₂ by tokens per update). It is a new recipe
  version briefed to Pop, never a profile key.
- **Window capacity (`tokens` 32,768) and `train_target_tokens` stay as they are.** They define what the model sees
  (whole-trajectory rule), and both fit: a single 32k window of Mellum is ~40 s on the GB10, ~10 s here.
- **Packing** (`window_execution`) is operational:
  - Mellum's recipe already declares `packed`. Its latents step inside backward, so there are no ~22 GB of
    accumulated gradients.
  - LFM's `auto` accumulates windows sequentially. On 96 GB, packed LFM fits easily and saves per-window launches.
    The bench's `l-ref` vs `l-nockpt` arms plus a planned trainer-level arm decide `window_execution: packed` for the
    sm_120 profile (pending).

### 2.3 Activation checkpointing

- **Mellum: keep it on, possibly selectively.**
  - Uncheckpointed activations take ~9.5 MB/token across 28 layers (~0.34 MB/token/layer), which is 255 GB at 26.8k
    tokens. Full-off is impossible on 96 GB.
  - The GB10 run held ~50 GB allocated and 68 GB reserved, with peaks +11–12.5 GB, so ~15–25 GB stay free on 96 GB.
    That is room for 2–3 uncheckpointed layers at 27k tokens (~9 GB each).
  - Recompute is ~1.4 s of expert and ~0.5 s of attention work per update on the GB10, ~0.5 s here in total. Keeping
    k of 28 layers saves k/28 of it, so 3 layers save ~0.05 s (~1–2%).
  - **Built:** `QwenPortBackbone.uncheckpointed_layers` (selective checkpointing: the top N layers keep their
    activations; numerically identical, tested). The bench measures 4, 8 and 14, expecting 8 and 14 to run out of
    memory at 32k; it records the boundary.
  - Wiring it into the direct_sft memory planner (`choose_layer_checkpointing` → a layer count instead of a bool) is
    worth it only if the bench shows > 3%. **Expected: not worth it.**
- **LFM: off.** The planner's `auto` mode already drops checkpointing when the estimate fits live CUDA memory. On
  96 GB it fits at 16–64k tokens (S3_PORT: ~40–50 GB of activations at 64k), so LFM gets the full ~25% recompute
  saving with no change. The bench's `l-ref`/`l-nockpt` pair measures that saving.
- **Built:** the page-cache release in `execution_plan` stays, because it is only reached on an integrated device when
  a fit fails. On a discrete GPU `device_bound` is the CUDA free memory, and dropping host page cache does not help.
  It runs at most once, and the ledger script is absent in a sandbox, so it costs one fadvise sweep. No change needed.

### 2.4 torch.compile, CUDA graphs and launch overhead

- On the GB10, `NATLANG_MAPLE_COMPILE=1` showed no gain (1.538 vs 1.407 s, 2 layers), because the kernels were slow
  enough to hide launches. Here they are not. Arm `m-compile` re-measures it, and `m-profile`/`l-profile` report GPU
  idle per update.
- **Rule:** if GPU idle exceeds 5% of the update, the next work is:
  1. **CUDA-graph the MuonSR step** (static shapes per latent; ~28 × (experts + attention) launches per update, all
     the same every update). It is numerics-identical. This is the largest launch-bound block, and the bench reports
     it as `optim/*`.
  2. **`torch.compile(mode="max-autotune-no-cudagraphs")` for the LFM layer stack**, whose shapes are bucketed by
     window length.
  3. **Persistent compile caches.** Sandboxes start cold: inductor and Triton autotune caches go to
     `TORCHINDUCTOR_CACHE_DIR`/`TRITON_CACHE_DIR` on scratch (set in the bench job). Publish them as a bundle keyed
     by (code sha, GPU) when the bench shows the cold start costs minutes (COREWEAVE_SANDBOXES.md §3.8).
- `NATLANG_MAPLE_COMPILE` is allowed in the profile's environment because it compiles numerics-preserving glue.

### 2.5 FP8 and the MoE dispatch kernel

- **Experts: keep `fp8_training: triton-experts` (Mellum).** It is our own Triton kernel, valid on CC 12.x, and the
  only grouped FP8 path that exists on sm_120 (`_scaled_grouped_mm` errors, cuBLASLt grouped excludes 12.x). The FP8
  forward and dgrad get the 2× FP8 rate. The `m-bf16` arm measures the gap here; numerics differ, so it is
  informational only.
- **The wgrad is the next target.** It runs as a BF16 `torch._grouped_mm` in parameter layout, which on sm_120 is the
  per-group fallback loop: 64 cuBLAS calls per projection per layer, ~5k launches per update. Candidates, in order:
  1. measure `grouped_mm_bf16` against `loop_mm_bf16` in the probe (if the fallback is close to the loop, it is fine);
  2. CUTLASS 4.6's sm_120 ptr-array grouped GEMM (FP8 tensor/token scaled) through a small extension, only if (1)
     shows the loop is launch-bound;
  3. a Triton grouped wgrad retuned for 188 SMs (it lost 2–5× on the GB10, which may not hold here).
- **Attention GEMMs: stay BF16.**
  - TE FP8 linears exclude QAT (direct_sft refuses `te-fp8` with a quantization component, and every recipe has QAT).
  - Attention projections are ~0.6 s of the expected update.
  - A per-shape FP8 projection path would save ≤ 0.3 s (~7%) at the risk of numerics. Revisit only after the bench,
    through the shared FP8 backend (`train/fp8_backend.py`): one implementation, no second FP8 path.
- **NVFP4/MXFP4 forward: not now.** On the GB10 the 4-bit forward saved ~2% (round 3) because the time is in the
  backward. The same arithmetic holds here.
- **MoE dispatch:**
  - The scatter-free permutation glue and fused SwiGLU/grad-quant (rounds 4–5) are bandwidth-bound and get the 5.8×.
  - The routing itself (topk, sort, offsets) is a handful of small kernels per layer: launch-bound, included in the
    graph work of 2.4.
  - Nothing to change before the profile.

### 2.6 Optimizer state placement and fused optimizers

- **Everything stays on the GPU.** Mellum's MuonSR momentum is BF16 (24 GB), the weights carry stochastic rounding
  in place, and there are no FP32 masters.
- CPU offload over PCIe Gen5 (~64 GB/s per direction) would move ≥ 48 GB per update, ~0.75 s, which is more than the
  whole expected optimizer cost. **Never offload.**
- The optimizer kernels are already fused:
  - Triton Newton-Schulz with a symmetric Gram;
  - fused elementwise apply with Philox stochastic-rounding noise;
  - stepping inside backward, so no gradient buffer.
- AdamW for routers and norms is tiny. Use `foreach=True` (the default), or the fused kernel if the profile shows it.
- **Expected optimizer time here: ~1 s per update** (~2.9 s Newton-Schulz → ~0.8; 0.75 elementwise → 0.13).

### 2.7 Host CPU, memory and the dataloader

- **Data path:** the trainer reads prepared text JSONL that already carries token ids (`--text-data`) and plans
  windows once at stage start. Per-update host work is negligible; there are no dataloader workers to size.
- **Startup:** parsing a 6.3 GB rendering takes minutes on one core, and its rows are released after mask
  preparation.
- **Host memory is dominated by state loads:**
  - `checkpoint.pt` (47 GB) is loaded with `map_location=cpu` and then moved;
  - the model safetensors (23 GB) are read through host memory;
  - a save writes host copies.
- **CPU:** compile workers (inductor uses all cores for parallel compile), the bundle publisher (sha256 of 47 GB per
  checkpoint) and pinned-memory copies want a handful of cores. GPU feeding is a single Python thread.

| Job | CPU | Memory | Scratch | Basis |
| - | - | - | - | - |
| Mellum training stage | 16 | 160 Gi | 400 Gi | 47 GB state load + 23 GB model + a save in flight; re-size from the bench's `host_peak_gb` and the first real stage (memory is billed on the request: 160 Gi costs $3.39/h, 128 Gi $2.71/h) |
| LFM training stage | 8 | 48 Gi | 100 Gi | small checkpoints, data read once |
| Task-success eval | 16 | 64 Gi | 120 Gi | node workers (as heldv2-base-mellum) |
| Teacher (vLLM) | 16 | 96 Gi | 150 Gi | weights staged through host memory; tokenizer and API server |
| hw_bench | 16 | 160 Gi | 200 Gi | Mellum loaded once per arm |

These are in the profile's `sandbox` section and in the two bench jobs.

### 2.8 Layer staging (end-to-end stages)

- `model/layer_staging.py` backpropagates layer by layer, one autograd engine call per layer per update, plus a
  pre-pass for cross-stream edges. Each call is host-side work of about a millisecond.
- On the GB10 that was noise. Here, against a 3–4 s update, 28 layers × (pre-pass + walk) is still < 2%. On LFM's
  0.3 s update it could reach 5–10%.
- **Measure it in the first E2E arm on a sandbox** (profiler `update/backward` with `LayerStaging` scopes). If it
  matters, overlap the per-layer optimizer step with the next layer's engine call (the step already runs in a hook).
  The hw_bench matrix does not include E2E stages yet; that is follow-up work for the E2E owner.

## 3. Teacher serving plan

Today: vLLM (DGX image `vllm-node`, arm64) on :8082 serves `nvidia/Qwen3.6-35B-A3B-NVFP4` with:
- Marlin MoE, FP8 KV, prefix caching, chunked prefill;
- 256 sequences, 8,192 batched tokens;
- **`--gpu-memory-utilization 0.45`**, because it shares the GB10 with training.

Measured production rate: 333–655 output tokens/s.

### Engine

- **vLLM v0.31.0** (pinned: `vllm/vllm-openai:v0.31.0@sha256:c1c9f6fd…`, multi-arch, amd64 resolved) is the first
  choice:
  - it is the engine our campaigns, tool parser and reasoning parser already target;
  - it is the most-used stack on sm_120 (FA2 attention, Marlin and FlashInfer FP4 MoE);
  - its prefix caching (hash-based, automatic) is what our shared program headers need.
- **SGLang v0.5.21** is a second candidate:
  - 0.5.8 produced NaNs on SM120;
  - NVIDIA's SGLang 26.03 container lists RTX PRO 6000 Server support;
  - its RadixAttention is the stronger prefix cache for tree-shaped prompt sharing (`lmsysorg/sglang:v0.5.21@sha256:b1259f3e…`).

  Benchmark it only if vLLM's prefix-cache hit rate on real campaign prompts is low.
- **llama.cpp:** best single-stream latency on sm_120 (native NVFP4 MMQ since b8967; gpt-oss-120b at ~196 tok/s
  decode), but its continuous batching is far behind vLLM at 64–256 concurrent requests. Use it only for latency-bound
  single sessions and for our own GGUF exports (Mellum fork parity), not for the collection teacher.

### Quantization (the 35B-class teacher fits at every precision)

| Format | Weights | KV room at 0.90 | Expected throughput | Quality | Plan |
| - | - | - | - | - | - |
| NVFP4 + Marlin W4A16 MoE | 22 GB | ~60 GB | published: 200 tok/s single stream, **1,581 tok/s at c=16** ([kyaky](https://huggingface.co/kyaky/Qwen3.6-35B-A3B-NVFP4)); expect 3–5k at c=128–256 | as DGX today | **baseline, DGX parity** (`vllm-nvfp4-marlin`) |
| NVFP4 native W4A4 (FlashInfer/CUTLASS sm_120) | 22 GB | ~60 GB | faster than Marlin at high concurrency if correct (FP4 tensor rate) | as above; check outputs (garbage was reported with older FlashInfer) | `vllm-nvfp4-native` |
| FP8 W8A8 | ~36 GB | ~45 GB | close to NVFP4 at high concurrency (Triton FP8 MoE on sm_120) | better than NVFP4 | `vllm-fp8` (needs the FP8 checkpoint downloaded) |
| AWQ int4 | ~20 GB | | similar to Marlin NVFP4 (also Marlin) | lower than NVFP4 | not benchmarked: no gain over NVFP4-Marlin |

Larger teachers also fit one GPU:
- Qwen3.5-122B-A10B or gpt-oss-120b in NVFP4/MXFP4 (63–70 GB weights, ~20 GB KV);
- Qwen3.5-397B does not fit.

A 120B-class teacher at ~1k tokens/s is an owner/quality decision (better teacher vs volume). The bench config list
takes it as one more entry.

### Speculative decoding

- Qwen3.6 has an MTP head. vLLM's `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'` helps latency
  at low concurrency and usually costs throughput at 128+ concurrent requests. Our collection runs at high
  concurrency, so the expected answer is "off for collection, on for interactive". The config `vllm-nvfp4-marlin-mtp2`
  measures both ends.
- One report saw MTP slow Marlin by 22% on SM120; the exact method name on v0.31 is unverified, and a failed start is
  recorded, not fatal.
- EAGLE-3 needs a trained draft head for Qwen3.6 and is not available.

### Prefix caching and concurrency

- `--enable-prefix-caching` with chunked prefill and `--max-num-batched-tokens 16384`. That is twice the DGX value:
  more prefill per step at high bandwidth.
- `--max-num-seqs 256`, `--gpu-memory-utilization 0.90` (the sandbox owns the GPU), FP8 KV cache, max model length
  65,536.
- The teacher bench sweeps concurrency 1, 8, 32, 64, 128 and 256 with 6k-token shared prefixes. It reads vLLM's
  `prefix_cache_hits/queries` counters, so the hit rate is measured, not assumed. Replay real campaign requests with
  `--prompts`.

**Expected: one sandbox teacher gives 5–8× the DGX :8082 rate.** Most of that comes from the GPU being whole (0.90
against 0.45) and from 5.8× the bandwidth on the decode-bound phase.

**Placement consequence.** A sandbox teacher is not reachable as `127.0.0.1:8082` from DGX. Either:
- run the collection worker inside the same sandbox (a campaign shard is a job: code bundle + vLLM image + ts-host),
  which is the natlang-native shape: the job is the program; or
- expose the sandbox through CoreWeave `services` (ingress; unverified for serverless).

Decide after the bench, with the placement program.

### Our Neuralese model serving (`natlang_neuralese.serve`)

- **Reference server (`serve.http`, `Engine`).** It is Python-orchestrated: per-sequence state machines and a batched
  `step_rows` decode for LFM; the Qwen/Maple port steps rows one by one. On this GPU it is host-bound long before the
  GPU is busy.
  - Run it with `--device cuda` (BF16), several replicas per GPU for LFM (1 GB of weights each: 8–16 processes share
    one GPU), and one for Mellum.
  - The fix that pays is CUDA graphs for the batched decode step (bucketed batch sizes), same math. Do it when serving
    becomes the bottleneck of a measured job.
- **Throughput path: `serve/vllm_rollout.py`.**
  - It decodes text on vLLM and runs writes on the reference port. It needs a vLLM image with this package on the path;
    the official amd64 v0.31.0 image plus the code bundle works.
  - For LFM2.5 (supported by vLLM) this is the fast path for RL and S5/S7 rollouts.
  - Mellum (model_type `mellum`) is not a vLLM architecture. Its throughput path is the llama.cpp fork (BF16/Q8_0
    parity passed) with sm_120 CUDA kernels.
- **Memory:** a sandbox hosting a training stage has no room for a second model. Rollout serving gets its own sandbox.

## 4. The image

- **Base:** NGC `pytorch:26.09-py3` (multi-arch; amd64 for CoreWeave). Measured on our arm64 copy: PyTorch 2.14.0a0,
  CUDA 13.4, Triton 3.8.0, TE 2.19, transformers 5.19.0, arch list `sm_80 sm_86 sm_90 sm_100 sm_110 sm_120
  compute_120`.
- **It covers everything the training stack needs on sm_120:**
  - Triton for every custom kernel and FlexAttention;
  - SDPA flash and cuDNN;
  - cuBLASLt FP8 and block-scaled formats;
  - TE FP8.
- **Nothing to add:** no FlashAttention wheel (FA3 is Hopper-only, FA4-sm120 is FA2-speed, SDPA covers it), no
  torchao, no DeepGEMM.
- **Built (`training/neuralese/docker/Dockerfile`, shared with DGX and Pop; one dependency list):**
  - `TORCH_CUDA_ARCH_LIST="8.9 12.0 12.1"` for any JIT extension build;
  - a build-time check that the base's PyTorch carries `sm_120` code (`torch._C._cuda_getArchFlags`, no GPU needed);
  - `/opt/natlang/pip-freeze.txt` records the exact resolved versions inside the image. Pinning is by image digest
    (which launches record) plus that freeze.
- **Build:** `scripts/coreweave/build_image.sh` (owned by the sandbox agent) builds `linux/amd64` with buildx. No change
  was needed there; the arch check runs inside the Dockerfile on any builder. On DGX it needs QEMU binfmt, or build
  on Pop (x86_64).
- **Teacher image:** the official `vllm/vllm-openai:v0.31.0` by digest. vLLM pins its own torch, so it is not mixed
  into the training image.

## 5. Expected speedups, summarized

| Workload | GB10 today | Expected on one RTX PRO 6000 | Main reasons |
| - | - | - | - |
| Mellum direct_sft (32k tokens/update) | 15.96 s, 1,682 tok/s | 3.5–4.5 s, **6–7.5k tok/s** (3.5–4.5×) | ~4–5× GEMM rate (FP8 2× BF16), 5.8× bandwidth for the optimizer and glue; launch gaps grow |
| LFM direct_sft (16k tokens/update) | ~1.3–1.9 s on Pop (Ada) | 0.3–0.5 s, **35–55k tok/s**, more with larger updates and graphs | no checkpointing; launch-bound |
| Teacher Qwen3.6-35B-A3B NVFP4 | 333–655 out tok/s (0.45 of the GB10) | **3–5k out tok/s** at c=128–256 | whole GPU, bandwidth, 16k batched tokens |
| 20 sandboxes in parallel | 1 GPU | ~70× the GB10's Mellum arm throughput | independent arms, evals and lines |

These figures are estimates from rates and the GB10 profile. The bench replaces them with measurements.

## 6. Benchmark harness (built, runs on the first sandbox)

- **`natlang_neuralese/train/hw_bench.py`** writes `hw-bench.json` (schema `natlang.neuralese-hw-bench/1`).
  - **probe** (its own process): device facts and versions; BF16 GEMM; FP8 rowwise; MXFP8 and NVFP4 `_scaled_mm`;
    `_grouped_mm` against a per-expert loop; `_scaled_grouped_mm` availability; FlexAttention window fwd+bwd; SDPA
    flash/cuDNN/mem-efficient fwd+bwd; device and pinned host bandwidth. Each test records its own failure.
  - **arms**: `train/speed_bench.py` (one real packed direct_sft update) per setting, each in its own process, so OOM
    is recorded.
  - **Mellum, 28 layers, v4 recipe (QAT q4), 4 × 8k tokens, whole-window supervision:** `m-ref` (FP8 experts,
    checkpointed: the GB10 configuration), `m-bf16`, `m-unckpt4/8/14`, `m-compile`, `m-16k`, `m-64k`, `m-profile`.
  - **LFM, LFM v3 recipe:** `l-ref` (16k, checkpointed), `l-nockpt`, `l-32k-nockpt`, `l-long-nockpt` (one 32k
    window), `l-te-fp8` (informational), `l-profile`.
  - **derived**: first-update loss agreement with each arm's reference (same math must give the same fixed-batch
    loss), the fastest numerically equal arm per backbone with its speedup, and the fastest arm overall.
- **`speed_bench.py` extended:** `--backbone lfm`, `--uncheckpointed-layers`, `--fp8-training te-fp8`, recipe
  resolution through `extends`, and device/host-peak/supervised tokens/s in `bench.json`.
- **`scripts/coreweave/teacher_bench.py`** (standard library only) writes `natlang.teacher-bench/1`.
  - It launches each configuration of `scripts/coreweave/teacher-configs.json` in turn: NVFP4-Marlin (DGX parity),
    NVFP4-native, FP8, NVFP4-Marlin + MTP2.
  - It sweeps concurrency over shared-prefix natlang-shaped prompts (or `--prompts` real requests) and reads prefix
    cache and spec-decode counters. It ranks configurations by output tokens/s and also reports single-stream decode
    and TTFT.
  - `--base-url` measures a running server, e.g. DGX :8082, for the like-for-like comparison.
- **Jobs (validated by `job.py`):**
  - `scripts/coreweave/jobs/hw-bench-train.json`: NGC image, 16 CPU / 160 Gi, bundle = code + Mellum + LFM snapshot,
    8 h lifetime, output `runs/coreweave-hw-bench/first`. Cost bound $63.52 at $2.50/GPU-h.
  - `scripts/coreweave/jobs/hw-bench-teacher.json`: vLLM v0.31.0 by digest, `NATLANG_CW_IMAGE_READY=1`, bundle = code +
    Qwen3.6 NVFP4, 6 h lifetime. Cost bound $39.50.
- **Run order after the GPU smoke passes:**
  `bundle.py plan … && bundle.py upload … && cwctl.py launch scripts/coreweave/jobs/hw-bench-train.json`. The bundle
  commands are in each job's `bundle_plan`. The teacher job can run at the same time in a second sandbox.
- **Smoke on DGX:** `python -m natlang_neuralese.train.hw_bench --matrix smoke --out /tmp/hwb` (2-layer Mellum and LFM
  at 2k tokens) checks every code path in a minute under the ledger.

## 7. What was built (this change)

- `training/neuralese/hardware-profiles.json` and `natlang_neuralese/train/hardware_profile.py`: declared hardware
  profiles (operational keys only). The recipe runner applies and records them.
- `model/hf_port.py`: `uncheckpointed_layers` (selective checkpointing, exact).
- `train/speed_bench.py`: LFM support, selective checkpointing, TE FP8 arm, richer output.
- `train/hw_bench.py`, `scripts/coreweave/teacher_bench.py`, `scripts/coreweave/teacher-configs.json`, and the two
  bench jobs.
- `training/neuralese/docker/Dockerfile`: arch list and sm_120 build check, pip freeze in the image.
- Tests: `tests/neuralese/test_hardware_optimization.py` covers profile validation and selection, refusal of
  training-math keys, application and recording, GB10 no-op, selective-checkpointing exactness, bench arms and
  derivation, the teacher bench against a fake streaming server, and job validation.

## 8. Decided only by the sandbox benchmark

| Question | Decided by | Lands in |
| - | - | - |
| Mellum: selective checkpointing worth wiring into the planner (> 3%)? Where does OOM start? | `m-unckpt*` vs `m-ref` | the planner (if yes), the profile's `checkpoint_layers` stays unset |
| LFM: packed vs sequential; does auto drop checkpointing as expected? | `l-ref`, `l-nockpt`, a trainer-level smoke | profile `window_execution` for `neuralese_direct` |
| `NATLANG_MAPLE_COMPILE=1` gain | `m-compile` | profile environment |
| GPU idle share → CUDA-graph the MuonSR step / compile LFM | `m-profile`, `l-profile` | next speed round (TRAINING_SPEED doc) |
| `_grouped_mm` fallback vs loop; is the expert wgrad launch-bound? | probe `grouped_mm_bf16`, `loop_mm_bf16` | CUTLASS sm_120 grouped wgrad or not |
| FP8 vs BF16 expert gain on this part | `m-bf16` vs `m-ref` | confirms `triton-experts` |
| Raise `batch_tokens`? | `m-16k/m-64k`, `l-32k` | a recipe version for both lines (owner-visible, briefed to Pop), never the profile |
| Real host-memory peak → sandbox memory request | `host_peak_gb` + first real stage | profile `sandbox`, job resources |
| Cold compile/autotune cost → publish caches as a bundle | arm wall vs median time | bootstrap/bundle (sandbox agent) |
| Teacher engine, quantization, MTP on/off, concurrency | `teacher-bench.json` ranking + output spot-check | teacher job config; placement of collection shards |
| Achieved BF16/FP8/NVFP4 rates on the Server Edition (clock unknown) | probe | this plan's §1 table |
