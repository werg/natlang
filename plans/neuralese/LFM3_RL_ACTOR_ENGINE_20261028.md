# LFM3 RL actor serving engine: measurement and fork decision (2026-10-28)

**Decision.** Fork **vLLM 0.32.0** and port the Neuralese actor feature set to it. Move the RL actors over in stages:

1. While the port lands, actors stay on the llama.cpp fork.
2. Once `lfm3.py` and the compact top-k path pass parity, vLLM serves the episodes without Neuralese.
3. Once span/conv import and export, the producer adapter and the markers pass parity against `serve/engine.py` and the
   fork, vLLM serves all episodes.

The llama.cpp fork stays for local, browser and GGUF serving (committed decision 17). This supersedes the engine named in
committed decision 10.

The reasons, in order:

- **llama.cpp falls behind badly at RL concurrency.** It decodes at about 17.6 tok/s per slot with 32 slots. Its
  sampler and top-k run on the CPU in a single-threaded server loop. Its best valid aggregate on real dev-packet turns
  was about 400 output tok/s.
- **vLLM and SGLang are far faster on the same prompts.** vLLM produced 10,660 output tok/s at 256 concurrent requests
  without logprobs (prefix cache warm). SGLang produced 4,422 tok/s with a cold cache. Both measurements used the
  LFM2-family proxy.
- **vLLM is the more extensible base for our features.** Prompt embeddings compose with prefix caching in vLLM, while
  SGLang's `input_embeds` forces `--disable-radix-cache`. `processed_logprobs` is exactly our post-temperature law, and
  vLLM has plugin entry points for models, endpoints, logits processors and LoRA resolution.
- **Neither engine supports LFM3 today.** No vLLM release or vLLM main registers `Lfm3ForCausalLM`. SGLang support is
  only in the open PR #46858. Writing the vLLM `lfm3.py` is therefore the first port item.

## What was measured

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

## Engine choice for the fork

| | vLLM 0.32.0 | SGLang 0.5.22 (+ PR #46858) |
|---|---|---|
| LFM3 support | None (no model file, no open PR). We write `lfm3.py` from `lfm2.py` plus a width-7 conv kernel (Triton `causal_conv1d` branches only to width 5–6) | Unmerged PR #46858: 33 files, +4,234. The text-only part we would carry is about 1.6k lines (`models/lfm3.py` 965, `configs/lfm3.py` 147, `cutedsl_gated_conv.py` 390, overrides/hooks), on a moving target until it merges |
| Measured throughput (proxy) | 10,660 tok/s k=0 c=256 (warm), 3,669 k=20, 612 k=128 (API-bound) | 4,422 tok/s k=0 c=256 (cold), 1,471 k=128 c=32 |
| Prompt embeddings | Yes. Mixed token/embeds, prefix-cache hash includes an embeds digest (`kv_cache_utils._gen_prompt_embeds_extra_hash_keys`) | `input_embeds` requires `--disable-radix-cache`, which would lose the shared-prompt throughput RL depends on |
| Post-temperature top-k | `logprobs_mode=processed_logprobs`, the law after temperature/top-k/top-p | Post-temperature by default, `top_logprobs_num` arbitrary |
| Hybrid conv state | `ShortConv` as `MambaSpec`, align-mode prefix caching | `MambaPool` + unified radix cache (FULL+MAMBA) |
| LoRA on LFM | `Lfm2ForCausalLM` is `SupportsLoRA`; punica kernels; dynamic adapters | `lfm2.py` declares no LoRA modules |
| Plugin surface | `vllm.general_plugins` (model registry), endpoint and logits-processor plugins, `LoRAResolver` | `SGLANG_EXTERNAL_MODEL_PACKAGE`, custom logit processors |

**Fork vLLM.** It wins on throughput at k=0 (with a warm prefix cache, as in RL), and on the two features that are hard
in SGLang: prompt embeddings with prefix caching, and LoRA on the LFM family. Its plugin surface keeps most of our code
outside the fork. SGLang's measured advantage (cheaper k=128 JSON) disappears once top-k is returned compactly, which
the port does anyway. The cost of vLLM is writing `lfm3.py` ourselves rather than carrying the PR.

## Feature port: insertion points, approach, verification (vLLM 0.32.0)

Shape: a thin core patch set in the fork, about 500–800 lines estimated, plus a `natlang-vllm` plugin package holding
the model, endpoints, logits processors, LoRA resolver and the native state store.

**Shared mechanism.** A per-step native metadata object on the forward context, built in
`v1/worker/gpu_model_runner.py` (`GPUModelRunner._prepare_inputs`). It holds static-address GPU buffers so that CUDA
graphs remain valid:

- `producing[num_tokens]`;
- `import_mask[num_tokens]`;
- `import_rows[num_tokens]`, which indexes a GPU-resident `NativeStateStore` of span tensors;
- `capture_slot[num_tokens]`.

Policy stays in the plugin.

| # | Feature | Effort | Insertion point | Approach | Verify against | Blocking unknowns |
|---|---|---|---|---|---|---|
| 0 | LFM3 model | M | Plugin `Lfm3ForCausalLM`, registered with `ModelRegistry.register_model`; based on `model_executor/models/lfm2.py` and `layers/mamba/short_conv.py` | Per-layer type (conv / full attention with partial rotary 0.5 and θ 1e6 / sliding window 256 with θ 1e4); QK RMSNorm; per-head sigmoid output gate; sandwich norms; `embedding_norm`; tied head; width-7 (`conv_L_cache` 7) causal conv kernel, varlen and state-indexed (port HF `lfm3_conv.py` or the PR's `cutedsl_gated_conv.py`); hybrid KV groups full + SWA + Mamba via `HybridKVCacheCoordinator` | Logits vs HF `modeling_lfm3.py` and `serve/engine.py` on dev-packet prompts (max \|Δlogit\|, top-1 agreement); cached decode vs full forward; FP8 `LiquidAI/LFM3-1.7B-FP8` (compressed-tensors FP8_BLOCK) loads with unfused gate/conv names | 3-group hybrid cache plus align-mode prefix caching untested; conv kernel numerics |
| 1 | Positional KV span import/export | M | Plugin `Lfm3Attention.forward` (after qkv proj and q/k norm, **before** RoPE); scheduling through the multimodal placeholder path (`PlaceholderRange`, `mm_hash` = content-addressed state id) as a pseudo-modality "native_state" | `k = where(import_mask, store.k[layer][import_rows], k)` and the same for v, then RoPE at the actual consumer positions. The span's own later layers are also overridden, so this is exact (the same argument as the fork's virtual tokens). Export: `capture_slot` writes raw k, v (pre-RoPE) to a static staging buffer, copied asynchronously to host per request. Not the KV connector API: `KVConnectorBase_V1` loads post-RoPE prefix blocks only, with no mid-prompt splice | `model/shared_state.py::splice_block`, `serve/engine.py::_native_consume_impl`; fork `llama_state_span_register` (keys after the key norm, before RoPE; SWA keys rotated at their own position); `llama-state-span-dump` fixtures | Whether the encoder-cache budget tolerates a non-encoder modality (otherwise use an extra block-hash key); spans crossing align-mode Mamba block boundaries |
| 2 | LFM3 conv-state import/export | M | Plugin `Lfm3ShortConv` (from `short_conv.py`) | Import: `Bx = where(import_mask, store.bx[layer][import_rows], B*x)` before the conv kernel, so the state becomes `(prefix ‖ span)[-6:]`, exactly `splice_block`'s `ConvState`. Export: per-token `B*x` for the whole producer span into the capture staging buffer, not just the last-window state | `engine.py::_native_finish_block`, `_native_store_capture`; fork `llama_state_capture_get` kinds K, V, CONV, EMBD | Capture buffer sizing for long producer spans |
| 3 | Prompt embeddings + producer adapter | S (embeds) / M (adapter) | `EngineCoreRequest.prompt_is_token_ids` mixed prompts (existing); `Lfm3ForCausalLM.embed_input_ids`; GPU toggle in the runner's sampled-token path | Stored text Neuralese through mixed `prompt_embeds`. Adapter `h = e + producing·up(down(e))`, always computed (static graphs). `producing` is updated on the GPU from the sampled token: `inside' = (inside \| tok==open) & tok≠close` | `LowRankTokenEmbedding.forward`, `native_step(..., producing=)`, `_native_prefill_impl`; fork `llama_set_seq_producing` | Async scheduling: the toggle must stay on the GPU |
| 4 | Exact post-temperature top-k (k=128) | S, plus M for the compact output (required: the measured JSON path costs 17x) | `--logprobs-mode processed_logprobs --max-logprobs 128`; guidance bans as a v1 `LogitsProcessor` (`v1/sample/logits_processor/interface.py`); endpoint plugin encodes `LogprobsTensors` directly | Return `encode_top_k` arrays (`<i4` ids, `<f2` logps, base64), with no per-token `Logprob` objects or detokenization | `common/sampling.py::masked_log_distribution`, `behavior_top_k_declaration('post-processing-normalized')`, `engine.py::_native_token`; KL and top-k overlap on shared prompts. The RNG differs (only the law must match) | none |
| 5 | Privileged-block insertion | S | Plugin request renderer | Port `serve/engine.py::_insert_privileged` unchanged; it acts on the plan before token ids; segments are marked `privileged: True` in the capture | Same plans rendered by `serve/engine.py` | none |
| 6 | `decide` with imported state | M | Endpoint plugin `/v1/neuralese/decide[_many]`; small patch to the prompt-logprob path (`sampling_params.py` forces `skip_reading_prefix_cache=True`) | One request per option (plan + imports + option tokens, `max_tokens=1`, `prompt_logprobs`). Read the shared prefix from the cache and emit logprobs only past the uncached boundary. Score with **raw** log_softmax (no temperature) | `serve/native_learning.py::decide_native`, `_score_from_cache`; fork `server-decision.cpp` | Prompt logprobs with prefix-cache reads enabled |
| 7 | Adapters, native soft states, LoRA mixtures | S (stacks as LoRA) / M (per-request mixture coefficients) | Plugin `LoRAResolver` or `LoRAModel.from_lora_tensors`; `lora/punica_wrapper/punica_gpu.py` (`add_lora_linear`, between `add_shrink` and `add_expand`) | Each adapter stack becomes one LoRA (basis `left@core@rightᵀ·scale`, free LoRA `B·A·α/r`, stacks concatenated along rank). Bank mixtures: register the bank as one rank-ΣR LoRA and scale the shrink output by a per-request coefficient row. Never wrap the conv weight. Native soft states are stored blocks served by feature 1 | `model/native_adapters.py::_effective_replacements`, `InferenceReplacementCache`; fork `llama_set_adapter_lora_seq_scales` | LoRA with the width-7 conv and FP8 together; `max_lora_rank` |
| 8 | Producer/consumer markers | S | Tokenizer/checkpoint export; the toggle of feature 3 | Distinct begin/end markers as ordinary special tokens; the plugin emits consume-open / span / consume-close like `_native_consume_impl` | `provision_role_tokenizer`; GGUF `natlang.identity.*` ids | Vocabulary size if the markers extend past 128,000 |
| — | Guidance rewind (`native_guidance.check`, `seq.bans`) | M–L | Not in vLLM | Re-submit a truncated request, relying on the prefix cache; or keep guidance off for vLLM actors | `serve/guidance.py` behavior | Conv-state checkpointing for an in-place rewind |

Parity method for every row: the exact-reference approach used for the fork's native-state production (llama.cpp-neuralese
commit b153171a0): the same prompts, imports and adapters through `serve/engine.py`, the fork and the vLLM port; compare
logits, top-k laws and exported spans.

## Order of work for the port agent

1. `lfm3.py` plus the width-7 conv kernel, with logit parity against HF and `serve/engine.py`. Then run this benchmark
   again with LFM3 itself: `runs/lfm3-engine-bench-20261028/inputs`, with a vLLM-only run order.
2. Compact top-k output (feature 4). This alone makes vLLM usable for episodes without Neuralese.
3. Feature 5 (renderer), then features 3, 8, 1 and 2 (span/conv import and export, shared native metadata), then 6, then 7.
4. Actor job template: a CPU request for actor sandboxes. This bench showed 2 vCPUs binding both Python engines.

## Not established here

- LFM3 throughput on vLLM or SGLang. Those numbers are from the LFM2.5-1.2B sibling.
- A corrected-configuration llama.cpp high-concurrency number. The `-kvu` bench rows are confounded; the
  capture-run row is valid at ≤32 slots.
- SGLang k=128 at 256 concurrency, and the vLLM vs SGLang k=0 comparison at equal cache state.
- NVFP4/FP8 throughput on any engine.
