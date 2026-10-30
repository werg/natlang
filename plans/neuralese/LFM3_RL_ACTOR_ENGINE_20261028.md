# LFM3 RL actor serving engine (2026-10-28, revised 2026-10-29)

**Decision (owner, 2026-10-29).** The RL actors run on natlang's **SGLang fork**
([werg/sglang-natlang](https://github.com/werg/sglang-natlang), branch `natlang`, pin `training/neuralese/sglang-fork.json`).
LFM3 comes from Liquid's own upstream implementation, sgl-project/sglang PR #46858. The reason is dependability of
building on upstream-maintained code as much as speed: the model file, the width-7 CuTe DSL gated-conv kernel and the
hybrid conv/SWA/full-attention cache are Liquid's and SGLang's to maintain, and we follow their branch. The vLLM port of
2026-10-28 is dropped (its `lfm3.py` never loaded weights, so nothing about LFM3 numerics was established there).
The llama.cpp fork stays for local, browser and GGUF serving (committed decision 17).

Stages:

1. Ordinary-token episodes: SGLang serves the exact trained checkpoint (HF export with the native sparse rows merged)
   and returns the exact post-temperature sampled-token logprob and top-k compactly. Episodes that need Neuralese are
   refused explicitly by the front and stay on the Python native server.
2. Neuralese features in the order of the table below; each lands with parity against `serve/engine.py` and moves the
   remaining episode kinds over.

## Architecture

- **Fork** (`werg/sglang-natlang`): only mechanism. Hooks in the sampler/output path and, for Neuralese, in Liquid's
  `models/lfm3.py` and small scheduler/runner hooks. `NATLANG_FORK.md` there records the base, our commits and the
  rebase procedure.
- **Front** (`serve/sglang_actor.py`, `SglangEngine`): the native `Engine` with generation delegated. It renders and
  tokenizes requests, inserts privileged guidance, parses replies and builds the exact `x_natlang_rl_capture`, so the
  collector, receipts and learner replay are unchanged. Started by
  `scripts/neuralese_data/serve_native_checkpoint.py --engine sglang`, which exports the checkpoint
  (`export/native_hf.py`, cached by checkpoint SHA-256), launches the fork's server on `--port + 1000` and serves the
  native HTTP API on `--port`. `rl_policy_binding.serving_engine` names the fork commit, export weights SHA-256 and the
  banned role markers.
- **Export** (`export/native_hf.py`): the backbone state with the four role-marker rows and the reasoning-boundary
  override rows written into the (tied) embedding; exact for every ordinary token. The low-rank producer adapter is
  recorded as not represented; until feature 3 lands the front bans all four role markers (logit bias -1e30) and records
  `banned_token_ids` in every sample, so the learner replays the same law.

## SGLang fork status

Fork `werg/sglang-natlang` `natlang` = PR #46858 head `8f6196da0b` (upstream main `cb55d77d37` inside) + natlang
commit(s) (`training/neuralese/sglang-fork.json`). Current upstream main does not run the PR's model (its 2026-10-29
recurrent-state refactor removed the pool accessor `lfm3.py` uses: CUDA-graph capture fails), so the fork follows the
PR branch and rebases when Liquid updates it (`NATLANG_FORK.md`).

Measured on DGX (GB10, sm_121, aarch64, CUDA 13.0, torch 2.14.1+cu130, shared GPU with an RL helper, a llama-server
eval and data jobs running):

- **CuTe gated conv runs on GB10.** `_use_cute_conv` is true: the PR gates on sm90/sm100/sm120 and SGLang's sm120 test
  is `major == 12`, so sm_121 qualifies. The PR states numerical validation on H100 (sm_90) and RTX PRO 6000 (sm_120).
- **Logit parity, LFM3-1.7B BF16** (`check_sglang_lfm3_parity.py`, 6 real episode turns of 4.5k–7.2k tokens, chunked
  prefill 2048, 48 greedy decode steps; reference HF FP32; HF BF16 as the noise floor):

  | | SGLang vs FP32: mean / max \|Δ logp\| | top-1 | HF BF16 vs FP32: mean / max | top-1 |
  |---|---|---|---|---|
  | prefill (33,055 positions) | 0.078 / 5.77 | 0.962 | 0.091 / 4.42 | 0.957 |
  | cached decode (288) | 0.0061 / 0.064 | 0.993 | 0.0069 / 0.109 | 0.993 |

  SGLang is closer to FP32 than HF's own BF16 forward. Prompt tokens are low-probability text, where any BF16 forward
  is ~0.1 nats off; SGLang's decode and a re-prefill of its own generated tokens agree to 0.003–0.011 mean. Varying
  FlashInfer vs Triton attention and the fast input-logprob path changed nothing (ruled out).
- **Compact capture law** (`custom_params.natlang_top_k`, 576 sampled tokens at T=1.0 and 0.7 vs FP32
  `log_softmax(logits/T)`): sampled-token logp mean 0.013 / max 0.24, top-1 0.991, top-20 overlap 0.982, records
  sorted.
- **End to end actor** (`check_sglang_actor_capture.py`, RL checkpoint `jolt-r1/round-000000` sha `81b6d5af…`, HF
  export with merged rows, 16 real episode turns, T=1, k=128, 128 max tokens): 1,809 sampled tokens in 10.1 s at
  concurrency 16 (179 tok/s including the ~92k prompt tokens of prefill) while the GPU was shared; the native Python
  server serves ~12 tok/s. Replay against the native model (BF16 at the time): sampled logp mean 0.020, top-1 0.986,
  top-128 overlap 0.981. The capture is the native engine's schema (`natlang.native-rl-sampling/2`, behaviour top-k
  receipt, `banned_token_ids` per sample), so the learner replays it unchanged.

- **Memory (main 7e72b5b82).** SGLang pools are sized from need (`--sglang-max-running` 32, `--sglang-context` 32768,
  KV pool 4 x context, mamba cache 8 x running; `--sglang-mem-fraction` 0.3 is only a ceiling), and the native fallback
  model is released after the export and reloaded only for a turn SGLang cannot serve. Two actors running concurrently
  on the shared GB10 (32 recorded turns each, c=16, 256 tokens): SGLang 7.5–8.1 GiB CUDA each, front 0.19 GiB CUDA /
  4.4 GiB RSS each, 156 and 146 tok/s. Before: ~50 GB per actor (0.2 x 124 GB pool + resident native model), and a
  second actor failed at CUDA context creation. A lazily loaded native fallback adds ~5.4 GiB.
- **Recorded collector turns.** The collector sets `x_natlang_native_recording` on every RL turn; it is served on
  SGLang (before 7e72b5b82 it routed every turn to the native fallback). Replay of recorded turns vs FP32: mean 0.0133
  nats, top-1 0.986 (native BF16 floor 0.0138).
- **Streamed collector turns.** The collector streams every turn (`on_delta`); before b80a25954 this routed every RL
  turn to the native fallback. Streamed turns now use SGLang incremental streaming (`--incremental-streaming-output
  --stream-interval 8`), with compact top-k assembled per chunk. A real `rl.production.collect()` (jolt-r1 config, full
  prompt, top-k 128, 4 rows x group 2 + teacher, jolt-r1 round-1 checkpoint): 98 turns, 79,002 tokens, 0
  fallbacks, collection complete (8 task successes of 12 rollouts); every token has a sorted top-128 record, the
  sampled token is inside it for 99.84% of tokens, and its recorded logp matches the sample's within 0.004 (FP16).
- **Producer spans** (fork 1b910fb875, `check_sglang_producer_parity.py`). The checkpoint's adapter is the identity
  (`up` = 0 in every current LFM3 checkpoint), so the check replaces it by a seeded random adapter (update norm 0.5 x
  embedding norm), exports it and compares with `native_step` and its producing mask, FP32 reference, native BF16 as
  floor, over 5 real turns (2.0k-7.2k tokens). Forced spans (prefill, one crossing a 2048-token chunk boundary): mean
  |dlogp| 0.051 on producing positions vs floor 0.050 (no-adapter control 0.66). Decode at T=1 from prompts ending at
  the open marker, inside a span, after a closed span, and inside a span with a close-marker logit bias (decode crosses
  sampled close markers, back to ordinary inputs): producing 0.0164 vs floor 0.0166 (control 0.155), ordinary 0.0148
  vs floor 0.0138 (control 0.113).
- **Produce spans served on SGLang** (fork 15015589e4). No marker is banned (export schema 3, `banned_token_ids` []),
  so the SGLang law is the native law; the front follows the native decode-step protocol per sampled token and raises
  the native errors (state-open, -close, -nested, -consume-marker) and recoverable protocol failures. When a span
  closes, one capture request recomputes it (radix match capped by `logprob_start_len`) and the fork writes its
  native state; the front stores it as the native engine does. Span export vs `native_step` (5 spans, 49 tokens each,
  perturbed adapter): relative L2 raw K 0.0138 (native BF16 floor 0.0147), V 0.0157 (0.0177), conv inputs 0.0144
  (0.0155), raw embeddings exact; empty spans keep the layer layout. Unit tests: scripted token streams through the
  native engine and the front give identical samples, inputs, writes, reply content and stored blocks, also streamed,
  and identical failure descriptors for all five protocol violations.
- **Imported native state and `decide` on SGLang** (fork d7c7a01ce3). An exact block is fed as consume_open,
  content-hashed placeholder ids and consume_close; the fork replaces the placeholders' K/V (before RoPE) and conv
  inputs by the stored ones (`splice_block`), so the radix and conv-state caches stay on. Other-model blocks are
  re-encoded from text as natively. `decide`/`score` sum SGLang input logprobs of each continuation after the shared
  prompt. Parity (4 real prefixes of 3000 tokens, 49-token spans exported by SGLang, 16-token continuation): mean
  |dlogp| 0.058 vs native BF16 floor 0.056; another span under the same prefix 0.065; a radix-cached repeat 0.060;
  importing the other span instead (control) 0.54. Unit tests: prompt receipts, imports, decision capture equal the
  native engine's; imported positions occupy cache but not own positions. `decide` on GPU
  (`check_sglang_decide_parity.py`, 5 prompts x 6 options): per-token |dlogp| vs native FP32 0.021 (native BF16 floor
  0.022), max 0.076 (floor 0.125), same continuation tokens, same best option in 5/5, 0 fallbacks.
- **Native adapters on SGLang** (fork c98c324989 + dff145de20, actor flag `--sglang-adapter-slots N`, default 4; 0 opts out and runs
  adapted turns on the native engine). Batches without an adapted request replay the base-only (`nolora`) decode
  graph and skip the delta path eagerly, so an empty pool adds no per-step work; the overhead is measured on the live
  RL actor (decode tok/s and request rate, first round-6+ collection vs round 5). The front turns a request's adapter stack (or the actor's default stack) into one summed
  low-rank pair per fork module (`native_adapters.lowrank_factors`: xs `left@core`/`right^T`, free-lora
  `alpha/r*scale*B`/`A`; q|k|v and w1|w3 packed at their row offsets, export schema 5 `adapter_targets`), assigns
  one of N resident slots (never reassigned while a request uses it) and records the native effective identity and
  adapter binding. The fork adds `((x A^T) * slot mask) B^T` after every adaptable projection, CUDA-graph safe.
  Stacks whose summed rank per module exceeds `--sglang-adapter-rank` run natively ('adapter rank'). Parity: three
  random stacks (xs + free-lora, rank 4 each, attention/conv/MLP matrices of 4 layers) and unadapted requests
  concurrently through 2 slots, T=1 decode: sampled logp vs the native adapted forward 0.0135 (unadapted engine noise
  0.0159), no-adapter control 0.21, 0 fallbacks.
- **Determinism (for the decision gate): not achieved.** 8 cases, 128 greedy tokens, compared with each case served
  alone: plain mode 4/8 identical in a batch of 8, 3/8 mixed with 8 fillers; `--enable-deterministic-inference` 5/8
  and 5–6/8 with FlashInfer or Triton attention (so attention is not the only batch-dependent op; early divergence at
  token 3 points at prefill). Deterministic mode also needs `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0` on
  sm_120/121: DeepGEMM's sm120 BF16 GEMM fails to compile for LFM3's 16-wide attention output gate
  ("kNTilesPerWarp must be even"). The gate stays on llama.cpp `-np 1 --no-cache-prompt` until the LFM3 ops are
  batch-invariant (candidates: CuTe conv prefill, fused QK norm, the compiled head gate).

RL wiring: `rl.production` configs for LFM3 raw policies (no semantic adapter or coefficient policy) add
`--engine sglang` to `actor_command`. The actor exports the round's checkpoint (cached under
`<store-dir>/sglang-hf/<sha>`), starts the fork's server on `--port + 1000` and serves the native API on `--port`.
Turns SGLang cannot serve yet run on the native engine of the same checkpoint, loaded on first use (`/metrics`:
`natlang_native_fallback_requests_total`, `natlang_native_fallback_reason_total{reason=...}`,
`natlang_native_model_resident`). Still native: templated writes/decodes, guidance (rewind), forced teacher
replay, and adapter stacks beyond `--sglang-adapter-rank`. Declared `native_state_modes` (same-base adapter state)
import on SGLang through the ordinary import path with the native receipt (`adapter_state_input_receipt` with the
export's layout check).

## Neuralese feature plan on SGLang

Policy stays in the natlang front; the fork carries mechanism. Shared mechanism for the state features: a
content-addressed pseudo-modality "native_state". SGLang's multimodal path already replaces placeholder positions with
`pad_values` derived from the item's identity hash (`managers/schedule_batch.py` `_compute_pad_values`), so the radix
key of a prompt that imports a state is its content hash and the radix/mamba prefix cache stays on (this is how LFM3-VL
images are cached in the same PR). `input_embeds` (which forces `--disable-radix-cache`) is not used.

| # | Feature | Effort | Insertion point | Approach | Blocking unknowns |
|---|---|---|---|---|---|
| 1 | Compact exact top-k + sampled logp | done | `layers/sampler.py`, `sampling_batch_info.py` | see status | none |
| 2 | Privileged blocks | done (front) | `serve/sglang_actor.py` | `_insert_privileged` on the plan before token ids, as the native engine | none |
| 3 | Producer adapter + markers | done | `Lfm3Model.forward` embedding; extend mapping in `ScheduleBatch.prepare_for_extend`; per-slot span flag | Export precomputes the producing table `adapter(embed(t))` with the native module; producing token t is fed as id t+V. Extend chunks are mapped by scanning the request's tokens, decode tokens from a per-request-slot GPU flag (`inside' = (inside or tok==open) and tok!=close`), CUDA-graph safe. No marker is banned: the front follows the native marker protocol (`_sglang_token`) | none |
| 4a | Span export (KV + conv) | done | `NatlangCapture` in `models/lfm3.py` (attention after the K norm, conv `B_gate*x`), `custom_params.natlang_capture` | When a span closes the front sends one capture request: the sequence through the span with `logprob_start_len` just below it (the radix match is capped there), so the span is recomputed in an eager extend; the fork writes the span's pre-RoPE K, V, conv inputs and raw embeddings (FP32) to a safetensors file, which becomes the native engine's `NativeStateBlock` (`_native_finish_block`). A request evicted mid-span needs no special case: the capture recomputes from tokens, and re-prefill maps the span again | none |
| 4 | KV span import | done | `Lfm3Attention.forward` after the K norm | see status | none |
| 5 | Conv-state import | done | `Lfm3ShortConv.forward`: `B_gate = 1`, `x` = stored conv input | see status | none |
| 6 | `decide` with imported state | done | `SglangEngine.sglang_score` | input logprobs after the shared prompt | none |
| 7 | Native adapters (LoRA/xs, mixtures as stacks) | done | fork `NatlangAdapters` slots; front `_with_adapters` | summed low-rank delta per slot; see status | none |
| 8 | CUDA graphs | — | decode graphs | Imports and exports happen in extend (prefill) batches, which run eager; decode only needs the producing flag and the adapter as static buffers | piecewise prefill graphs if enabled |
| — | Guidance rewind | M–L | not in SGLang | Re-submit a truncated request (prefix cache makes it cheap), or keep guidance off for SGLang actors | conv-state checkpoint at the rewind point |

## 2026-10-28 measurement (sandbox, LFM2.5 proxy for vLLM/SGLang)

Sandbox `e74dc2d0-3715-48f1-922b-611a4c0c070f`:

- Hardware: RTX PRO 6000 Blackwell Server Edition, 97,887 MiB; 2 vCPUs assigned; 32 Gi cgroup.
- Request: one GPU and a 32Gi RAM minimum, CPU unspecified.
- Lifetime: 11:32:47 to 12:20:24 UTC (47.6 min), stopped after `cwctl collect`.

Four earlier launches ended within minutes and were stopped or completed:

- `7457fcb2`: assigned an H100, so the sm_120 binary could not run.
- `592af340`: the bundle lacked the wheelhouse.
- `b206116c`: input push interrupted.
- `3b706b03`: the code snapshot predated `prebuilt.py`.

All five show `completed`.

Inputs and outputs are in `runs/lfm3-engine-bench-20261028/` (gitignored):

- `inputs/run.sh`, `bench_client.py`, `record_proxy.py`, `make_requests.py`;
- `out/results.jsonl` and the per-engine logs;
- `out/capture.jsonl`.

**Workload: real natlang episode turns.**

- **Capture.** 32 dev-packet cases (`natlang-dev-task-packet-v1-20261028`, rows 0, 4, …, 124; two per shape) ran
  through the real collector (`scripts/held_task_budget_probe.py`). They ran against the fork's llama-server with LFM3-1.7B BF16, with a
  recording proxy in front, for 270 s. That captured 174 chat requests (messages + tools).
- **Replay.** Each captured request was tokenized with the model's own chat template (thinking on): 256 requests,
  prompts 4.5k–8.1k tokens (p50 5.7k).
- **Shared prefixes.** Every turn shares the same system preamble, and later turns of an episode extend earlier ones.
  The 256 requests cycle through 158–172 distinct turns.
- **Output length.** Each request's `max_tokens` is the captured completion length clipped to [256, 2048] (mean 490).
  The captured completions had p50 283 and max 2,958.
- **Sampling.** `ignore_eos`, so every engine generates the same token count. Temperature 1.0, top-p 1, no top-k or
  min-p.
- **Top-k capture.** Requested through the OpenAI `/v1/completions` `logprobs=k` (vLLM with `--max-logprobs 128
  --logprobs-mode processed_logprobs`; llama.cpp `n_probs`).

**Models.**

- No vLLM or SGLang release serves LFM3. vLLM 0.32.0 with `--trust-remote-code` fails: `Layer 0 (conv) does not dispatch
  through the Transformers attention interface and vLLM has no other way to handle it`.
- vLLM and SGLang were therefore measured on the closest supported sibling, **LFM2.5-1.2B-Thinking** (`Lfm2ForCausalLM`:
  gated short-conv + GQA hybrid, thinking, BF16, revision f313478934a7).
- llama.cpp was measured on LFM3-1.7B BF16 itself.
- LFM3-1.7B has about 1.5x the parameters (1.73B vs 1.17B) and twice the vocabulary (128k vs 65k). Expect LFM3 on vLLM
  at roughly 0.6–0.7x the proxy numbers. This is an estimate, not a measurement.

**Engines.**

- **llama.cpp.** The prebuilt fork artifact `llama-cpp-neuralese-fat` (85a2e7167, includes sm_120). Bench
  configuration: `-np 256 -kvu -c 2097152 -b 4096 -ub 1024 -fa on -ngl 999`.
- **vLLM 0.32.0** (torch 2.13.0+cu130). Flags: `--max-num-seqs 256 --gpu-memory-utilization 0.85 --enable-prefix-caching
  --max-model-len 16384`. Mamba cache mode was `align` (auto-set by vLLM), CUDA graphs on.
- **SGLang 0.5.22.** Flags: `--mem-fraction-static 0.85 --max-running-requests 256`. Unified radix cache (FULL+MAMBA).
- Both Python engines installed with `uv` into their own venvs: vLLM in 234 s, SGLang in 277 s.

### Results

Output tok/s is aggregate generated tokens divided by wall time; total time is for the whole batch.

| Engine (model) | Concurrency | top-k | Requests | Total time (s) | Output tok/s | p50 / p95 latency (s) | GPU mem (MiB) |
|---|---|---|---|---|---|---|---|
| vLLM (LFM2.5 proxy) | 1 | 128 | 4 | 3.6 | 287 | 0.86 / 1.10 | 83,077* |
| vLLM | 32 | 128 | 64 | 24.7 | 893 | 13.8 / 14.3 | 83,197* |
| vLLM | 128 | 128 | 256 | 201.2 | 615 | 80.3 / 130.8 | 83,201* |
| vLLM | 256 | 128 | 256 | 202.0 | 612 | 86.9 / 194.6 | 83,201* |
| vLLM | 256 | 20 | 256 | 33.7 | 3,669 | 32.2 / 32.8 | 83,203* |
| vLLM | 256 | 0 | 256 | **11.6** | **10,660** | 4.8 / 9.8 | 83,201* |
| SGLang (LFM2.5 proxy) | 1 | 128 | 4 | 3.9 | 264 | 0.98 / 1.07 | 85,139* |
| SGLang | 32 | 128 | 64 | 15.0 | 1,471 | 5.6 / 8.9 | 85,139* |
| SGLang | 256 | 0 | 256 | **28.0** | **4,422** | 19.0 / 25.6 | 85,139* |
| SGLang | 256 | 128 | — | not run (cut for the deadline; k=0 ran first) | | | |
| llama.cpp fork (LFM3 BF16) | 1 | 128 | 4 | 8.0 | ~180 per stream | 2.0 server-side (one request) | 33,793 |
| llama.cpp fork | 32 | 128 | 64 | 122.3 | ~155 | 46 / 80 server-side | 33,797 |
| llama.cpp fork | 128 | 128 | 95 of 256 done in 583 s | stopped | ~50 released (~116 decoding) | 340 / 506 server-side | 33,797 |
| llama.cpp fork | 256 | 0 | 1 of 256 done in 150 s | stopped | per-slot 0.94 tok/s | — | — |
| llama.cpp fork, capture run (`-np 32`, no `-kvu`, real episodes through chat, no logprobs) | ≤32 | 0 | 174 turns in 215 s | — | **397** (+1,353 prompt tok/s) | per-slot 17.6 tok/s | — |

\* vLLM and SGLang preallocate KV/state pools from `--gpu-memory-utilization` / `--mem-fraction-static` 0.85. These
figures are the pool, not need. vLLM reported 77.4 GiB available for KV, a capacity of 6.7M tokens (411x 16k-token
requests).

### How to read these numbers

- **top-k=128 over HTTP JSON is CPU-bound and collapses concurrency.**
  - With k=128, each response is about 1.76 MB (3.6 KB per generated token).
  - vLLM's engine log shows only 1–4 requests running while the 2-vCPU API server serializes. Generation stayed around
    600 tok/s, against 10,660 tok/s with k=0 and 3,669 with k=20.
  - The top-k itself is cheap on the GPU. The cost is Python object creation, detokenization and JSON on 2 CPUs.
  - The port must therefore return top-k as compact tensors (`common/sampling.encode_top_k` format: `<i4` ids, `<f2`
    logps) from the engine, not as OpenAI `Logprob` objects. The k=0 row is the throughput target.
  - SGLang's k=128 path is cheaper at concurrency 32 (1,471 vs 893 tok/s), but a compact path removes that difference.
- **Prefix caching.**
  - The vLLM k=0 run came after the k=128 runs on the same prompts, so its prefix cache was warm. The engine reported a
    97.9% prefix hit rate.
  - SGLang's k=0 ran first, with a cold cache that still shared the system preamble and repeated turns. **The vLLM vs
    SGLang k=0 numbers are therefore not like for like.**
  - Warm prefix reuse is realistic for RL, because a GRPO/JOLT+ group samples several rollouts from one prompt and every
    episode shares the natlang preamble.
  - llama.cpp has no cross-slot prefix sharing. Each slot only reuses its own previous prompt.
- **The llama.cpp high-concurrency rows are confounded by the bench configuration.**
  - I passed `-kvu` (one unified KV buffer). With hundreds of long sequences, each decode step then attends over the
    whole unified buffer.
  - The capture run without `-kvu` at 32 slots is the valid llama.cpp data point: 397 output tok/s on real episodes,
    per-slot decode 17.6 tok/s.
  - Even that is CPU-bound. The fork's backend (GPU) sampling (`-bs`) is disabled whenever pre-sampling `n_probs` or
    guidance is requested (`server-context.cpp`, `use_backend_sampling &= !need_pre_sample_logits; ... &= !guidance`),
    and those are exactly what RL actors use. At 32 slots the per-step sampler cost (128k-vocab softmax plus top-k per
    slot on one thread) matches the measured ~57 ms per decode step.
  - k=0 at 256 slots was no faster than k=128 (0.94 vs 0.91 tok/s per slot). The bottleneck there is the unified-KV
    attention and server loop, not `n_probs`.
  - A corrected fork configuration (no `-kvu`, backend sampling with post-sampling top-k) would improve the fork. It
    would not close a 10–25x gap to vLLM's batched GPU decode plus paged prefix cache. That is a judgment, not a
    measurement; a 15-minute llama-only sandbox would measure it if wanted.
- **2 vCPUs bind the Python engines too.** vLLM's API server and engine core each held about 50% CPU, and GPU
  utilization was 13–68%.
  - The actor sandboxes should get a CPU request once the compact top-k path exists. AGENTS.md keeps CPU unspecified,
    and committed decision 9 says to raise it with the owner only when starvation is shown. **This measurement shows it
    for actors.**
- **NVFP4 GGUF works.** `llama-quantize --tensor-type 'blk\..*=nvfp4' LFM3-1.7B-BF16.gguf out.gguf Q8_0` made a
  5.11-BPW GGUF in 33.6 s (1,054 MiB from 3,298 MiB). The server loaded it. Its bench points were cut for time.
  vLLM/SGLang NVFP4 and FP8 for LFM3 need our own `lfm3.py` first.
- **Single stream.** About 180 tok/s for the fork (on LFM3, k=128), 287 for vLLM and 264 for SGLang (on the proxy,
  k=128). This is not where the RL architecture is decided.

## Not established

- SGLang throughput on LFM3 against the Python native server and llama.cpp at RL shape on the sandbox GPU (H100 /
  RTX PRO 6000). DGX GB10 numbers are relative only.
- The CuTe gated-conv kernel on H100 (sm_90) and RTX PRO 6000 (sm_120) in our runs; the PR states it was validated
  numerically on both, and gates it on sm90/sm100/sm120 (sm_121 GB10 counts as sm120).
