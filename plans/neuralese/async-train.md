# LFM3-1.7B Phase 2 Optimization Plan: G4 + DGX Spark Async RL, Quantized Rollouts & Score Centering

**Status:** Revised Phase 2 design brief and implementation plan (conditional on Phase 1 telemetry).  
**Date:** 22 November 2026  
**Prerequisite:** Completion of `LFM3_G4_LORA_LEARNER_DESIGN_AND_PLAN_V2.md` (Work Packages A–E: shared LoRA learner core, SFT/RL migration, atomic publication, and baseline G4/L4 profiling).  
**Primary topology:** **Colab G4** (NVIDIA RTX PRO 6000 Blackwell, $96\text{ GB}$ GDDR7) as the dedicated learner + **Local DGX Spark** (GB10, $128\text{ GB}$ unified LPDDR5X, 20-core Arm) as the dedicated rollout collector, harness runner, replay buffer, and held evaluator.  
**Target stack:** `werg/natlang` (`grpo.py`, `train/lfm3_fast.py`, `train/learner_state.py`, `serve/sglang_actor.py`, `rl/held_eval.py`).  
**Primary objective:** Eliminate metered Colab G4 idle time during rollouts and held evaluation, saturate DGX Spark's decode and harness capacity, and prevent off-policy staleness or quantization training–inference mismatch (TIM) from degrading held-out task performance.

---

## 1. Activation Triggers & System Synergy

Execute each pillar **only** when Phase 1 (`LFM3_G4_LORA_LEARNER_DESIGN_AND_PLAN_V2.md`) telemetry warrants it:

| Pillar | Quantitative Activation Trigger | Primary Payoff |
|---|---|---|
| **Pillar 1: Disaggregated G4 $\leftrightarrow$ DGX Spark Async Pipeline** | Rollout collection, TypeScript/NL harness wait, and `held_eval.py` account for $> 35\%$ of G4 wall time, burning metered CU/hr while the learner is idle. | Moves all $0\text{ CU/hr}$ CPU-bound harness work, rollouts, and held evals onto local DGX Spark; keeps G4 saturated on gradient steps. |
| **Pillar 2: High-Concurrency & Quantized Rollouts on Spark** | DGX Spark's $273\text{ GB/s}$ LPDDR5X bus cannot feed the G4 learner fast enough in BF16, leaving G4 starved for fresh rollout groups. | High concurrency ($32\text{--}64+$ episodes) + FP8 actor weights halve weight-read traffic ($\approx 3.22\text{ GiB} \to 1.61\text{ GiB}$), roughly doubling Spark decode throughput. |
| **Pillar 3: Score Centering (SC / RSC)** | Enabled **whenever** Pillar 1 runs with staleness $\Delta v \ge 2$ or Pillar 2 enables FP8 rollouts on Spark, **or** if actor-vs-learner KL drift causes reward collapse despite TIS. | Cancels off-policy and quantization TIM gradient drift additively without the variance spikes of uncapped importance sampling. |

### Why These Three Pillars Fit Together

Disaggregated G4 $\leftrightarrow$ Spark training and FP8 Spark rollouts are **cost and throughput multipliers** that deliberately introduce two sources of divergence between the sampler $q$ (SGLang on Spark) and the learner $\pi_\theta$ (`train/lfm3_fast.py` on G4):
1. **Temporal divergence (staleness):** Spark generates episodes under policy version $v_{t-k}$ while G4 updates $v_t$.
2. **Numerical divergence (quantization TIM):** Spark decodes with FP8 weights/KV while G4 computes gradients with a frozen BF16 base + FP32 LoRA factors.

Score Centering (composed with `grpo.py`'s existing Truncated Importance Sampling) is the **numerical stabilizer** that preserves held-out sample efficiency across both gaps.

---

## 2. Pillar 1: Disaggregated G4 Learner + DGX Spark Collector Architecture

### 2.1 Hardware Role Assignment

| Dimension | Colab G4 (RTX PRO 6000 Blackwell) | Local DGX Spark (GB10) | Assigned Role |
|---|---|---|---|
| **Memory & Bandwidth** | $96\text{ GB}$ GDDR7 @ $1{,}597\text{ GB/s}$ | $128\text{ GB}$ unified LPDDR5X @ $273\text{ GB/s}$ | **G4 = Learner:** $5.8\times$ memory bandwidth and dense BF16 Tensor Cores crush forward/backward/reference passes. |
| **CPU & Harness** | Ephemeral Colab vCPUs | 20-core Arm ($10\times$ X925 + $10\times$ A725) + $128\text{ GB}$ unified RAM | **Spark = Collector & Evaluator:** Runs high-concurrency NatLang TypeScript/NL sandboxes, SGLang decode, and `rl/held_eval.py`. |
| **Cost & Persistence** | Metered CU/hr ($\approx 5.1\times\text{--}5.9\times$ L4), preemptible | Sunk cost ($0\text{ CU/hr}$), persistent local NVMe | **Spark = Durable Anchor:** Holds the canonical base weights, local replay buffer, and continuous eval queue across Colab disconnects. |

*(Execution-profile isolation: Keep the repository's existing DGX combined host/device memory ledger strictly in the Spark collector/eval profile and out of the G4 learner profile. If DGX Spark is temporarily unavailable, fall back to a secondary single-G4 colocated profile with hard VRAM partitioning: e.g., $28\text{ GiB}$ SGLang pool, $52\text{ GiB}$ learner budget, $16\text{ GiB}$ reserve.)*

### 2.2 Weight Synchronization: Ship Deltas, Merge Locally on Spark

Pushing a full $3.22\text{ GiB}$ merged BF16 checkpoint over WAN on every RL step is slow and unnecessary—and you do **not** need SGLang multi-LoRA slots to avoid it:

1. **Pinned Immutable Base on Both NVMe Drives:** Both G4 and DGX Spark hold the exact same $3.22\text{ GiB}$ immutable BF16 base (`base_weights_sha256` verified at startup).
2. **Delta-Only WAN Upload from G4:** On every committed optimizer step $v$, G4 uploads only the LoRA factor delta:
   - At $r = 16$: $\approx 61.4\text{ MiB}$ (FP32) or $\approx 30.7\text{ MiB}$ (BF16).
   - At $r = 32$: $\approx 122.8\text{ MiB}$ (FP32) or $\approx 61.4\text{ MiB}$ (BF16).
   *(Always upload FP32 factors for resumable learner state; optionally upload BF16-cast delta or let Spark merge in FP32 from the FP32 delta—sub-second to a few seconds on standard broadband.)*
3. **Local Merge & Refit on Spark:** DGX Spark downloads the delta, computes the merged effective weights locally in unified memory using FP32 accumulation:
   $$W_{\text{eff}} = \operatorname{cast}_{\text{bf16}}\!\left(\operatorname{fp32}(W_0) + \frac{\alpha}{r} B_{\text{fp32}} A_{\text{fp32}}\right)$$
   and executes the Phase 1 atomic publication transaction against its local SGLang instance (pause/drain, refit, invalidate KV + `conv1d` caches, **rebind all four routes including `readout: decision` and native fallback**, attest `effective_policy_sha256`, resume).

**Why this wins initially:** You get $25\times\text{--}100\times$ smaller WAN transfers while completely avoiding SGLang LoRA packed-rank constraints ($3r=96$ for fused `qkv`, $2r=64$ for fused `w1/w3`) and hybrid `conv1d` LoRA-kernel risks. Upgrade to SGLang refcounted LoRA slots on Spark **only** if local merge + refit exceeds $5\%$ of Spark's rollout cycle time.

### 2.3 Preemption-Proof Object-Storage Bridge (GCS / S3 / R2)

Because Colab VMs are ephemeral and DGX Spark sits behind local NAT, use a cloud object-storage bucket as the asynchronous transport queue rather than brittle direct TCP/RPC (optional Tailscale direct HTTP can be added later only if bucket polling latency becomes measurable):

```text
gs://<natlang-rl-bucket>/<run_id>/
  base/
    base_manifest.json                  # SHA-256 digests of immutable base + interface
  adapters/
    v000042/
      adapter_factors_fp32.safetensors  # 61–123 MiB delta (uploaded first)
      policy_manifest.json              # effective_policy_sha256, learner_step, git_sha
    latest_policy.json                  # Atomic pointer updated ONLY after blobs are durable
  rollouts/
    v000042/
      group_<uuid>.zst.parquet          # Completed, scored group + optional top-K binary blob
      group_<uuid>.done                 # Atomic completion marker
  checkpoints/
    step_000042/                        # Full resumable learner_state.py snapshot
  evals/
    v000042/held_eval_result.json       # Written asynchronously by Spark
```

**Operational Properties:**
- **Colab preemption immunity:** If Colab disconnects mid-step, DGX Spark does not crash. It continues filling `rollouts/` up to the backpressure cap. When a new Colab G4 session starts, it pulls `checkpoints/latest` and immediately resumes draining fresh groups from `rollouts/`.
- **Atomic visibility:** Both sides write data blobs first, verify checksums, and commit small JSON pointer/marker files last.

### 2.4 Bounded Staleness, Backpressure & Mid-Episode Attribution

1. **Strict staleness window ($\Delta v_{\max}$):** Every rollout group carries `policy_version` ($v_{\text{beh}}$). G4 only trains on groups satisfying:
   $$\Delta v = v_{\text{cur}} - v_{\text{beh}} \le \Delta v_{\max}$$
   Start with $\Delta v_{\max} = 1$ (1-step pipelining). Relax to $\Delta v_{\max} \in \{2, 4, 8\}$ only after enabling Pillar 3 (TIS + SC) and confirming held-eval stability. Archive/skip expired groups and log `dropped_stale_groups`.
2. **Queue backpressure:** Spark pauses new rollout launches whenever the number of unconsumed non-stale groups in `rollouts/` exceeds $2\times$ the learner's optimizer batch size (preventing wasted compute on groups that will expire before G4 reads them).
3. **Mid-episode version attribution:** Under local-merge-on-Spark, pause/drain in-flight episodes at a turn boundary before refitting SGLang, **or** explicitly record per-token `(version_id, behavior_logp)` spans if retract/resume is used. Never attribute a mixed-version episode wholly to $v_{k+1}$.
4. **Zero double-training on resume:** Persist `committed_group_ids` and `consumed_rollout_uris` inside G4's `learner_manifest.json` on every optimizer step. On restart, G4 ignores any bucket object whose `group_id` is already in `committed_group_ids`.

---

## 3. Pillar 2: High-Concurrency & Quantized Rollouts on DGX Spark

Quantization applies **strictly to the DGX Spark rollout actor**. The G4 learner remains frozen BF16 base + FP32 LoRA factors + FP32 AdamW state so training numerics never degrade.

### 3.1 Overcoming Spark's $273\text{ GB/s}$ Decode Bandwidth

On LFM3-1.7B, single-stream BF16 decode on Spark's $273\text{ GB/s}$ LPDDR5X bus is memory-bandwidth bound ($\approx 50\text{--}65\text{ tok/s}$). Two levers let Spark keep up with G4:

1. **High Rollout Concurrency (First Lever — Zero Numerical Risk):** With $128\text{ GB}$ of unified memory and a $3.22\text{ GiB}$ model, run $32\text{--}64+$ concurrent NatLang episodes in SGLang (subject to the DGX combined memory ledger so TypeScript/NL sandboxes and OS buffers remain safe). Batched decode amortizes weight reads across requests, shifting LFM3-1.7B from bandwidth-bound toward GB10's compute ceiling.
2. **Staged Quantization on Spark (Second Lever — When Concurrency Alone Isn't Enough):**

| Stage | Spark Actor Configuration | Bottleneck Solved | Expected Gain on Spark |
|---|---|---|---|
| **Stage 2a** | **FP8 weights** (W8A8 / W8A16) + BF16 KV | **$273\text{ GB/s}$ weight-read bandwidth** (halves model reads from $3.22\text{ GiB} \to 1.61\text{ GiB}$). | $\approx 1.6\times\text{--}1.9\times$ decode throughput on bandwidth-limited batches. |
| **Stage 2b** | FP8 weights + **FP8 KV cache** (`e4m3fn`) | Long-context KV footprint when running $64+$ concurrent multi-turn episodes. | Doubles KV capacity and cuts attention KV read bandwidth in half. |

*(Note: Unlike on a $24\text{ GB}$ L4 where KV VRAM is the first wall, on a $128\text{ GB}$ DGX Spark **weight-read bandwidth** is the primary bottleneck, so FP8 weights are prioritized as Stage 2a.)*

### 3.2 Hybrid LFM3 Architectural Safeguards

1. **Keep causal `conv1d` recurrent states in BF16/FP32:** Convolution states are tiny relative to attention KV caches and sensitive to multi-turn quantization drift. Exempt `conv1d` states from FP8 quantization.
2. **Preserve `readout: decision` & logprob precision:** Keep final LM-head logits and softmax probabilities in FP32/BF16 inside SGLang so captured behavior logprobs and top-$K$ distributions remain accurate.
3. **Local merge $\to$ online FP8 quantization:** When Spark merges $W_0 + \frac{\alpha}{r}BA$ locally into BF16, apply fast per-channel/block FP8 quantization during the SGLang refit step (measure and require quantization + refit time $< 3\%$ of rollout cycle time). Alternatively, mount a frozen FP8 base with BF16 SGLang LoRA slots once Milestone 2 slot qualification is complete.

### 3.3 Mandatory Pre-RL Quantization Gate

Before enabling quantized Spark rollouts in live RL, run a fixed $200$-prompt NatLang diagnostic suite comparing the quantized Spark actor against the BF16 G4 learner:
- Measure token-level forward KL $D_{\text{KL}}(q_{\text{spark,fp8}} \parallel \pi_{\text{g4,bf16}})$, max $|\log q - \log \pi|$, and `readout: decision` option agreement rate.
- Compare `rl/held_eval.py` pass rates between BF16 and FP8 Spark actors on the same checkpoint.
- **Gate:** If held-eval pass rate drops outside noise or KL drift is elevated, require Pillar 3 (Score Centering) to be enabled alongside Stage 2a/2b.

---

## 4. Pillar 3: Score Centering (SC) & Regularized Score Centering (RSC)

### 4.1 Mathematical Formulation (`natlang_grpo_v2_tis_sc`)

When Spark samples $a_t \sim q(\cdot \mid s_t)$ (under FP8 and/or staleness $\Delta v > 0$) while G4 updates $\pi_\theta(\cdot \mid s_t)$ (BF16), the policy gradient decomposes into a true reward-covariance signal plus an unwanted off-policy drift term:

$$\mathbb{E}_{a_t \sim q}\big[A_t \nabla_\theta \log \pi_\theta(a_t \mid s_t)\big] = \operatorname{Cov}_q\!\big(A_t, \nabla_\theta \log \pi_\theta(a_t \mid s_t)\big) + \mathbb{E}_q[A_t]\,\mathbb{E}_{v \sim q}\big[\nabla_\theta \log \pi_\theta(v \mid s_t)\big]$$

Even with group-centered GRPO advantages, $\mathbb{E}_q[A_t \mid s_t] \neq 0$ at intermediate prefixes $s_t$. **Regularized Score Centering (RSC)** with strength $\lambda_{\text{sc}} \in [0, 1]$ ($\lambda_{\text{sc}} = 1$ gives full SC; $\lambda_{\text{sc}} = 0$ recovers `natlang_grpo_v1`) subtracts the expected trainer score under the sampler's distribution, cancelling drift additively at every prefix:

$$\tilde{s}_t^{(\lambda)}(\theta) = \nabla_\theta \log \pi_\theta(a_t \mid s_t) - \lambda_{\text{sc}} \, \mathbb{E}_{v \sim \tilde{q}(\cdot \mid s_t)}\big[\nabla_\theta \log \pi_\theta(v \mid s_t)\big]$$

**Composition with `grpo.py`'s Existing TIS & k2 Reference Penalty:**

$$\mathcal{L}_{\text{policy}, t}^{\text{TIS+SC}} = -w_{\text{tis}, t} \cdot \hat{A}_t \cdot \left[\log \pi_\theta^{\text{pol}}(a_t \mid s_t) - \lambda_{\text{sc}} \sum_{v \in \mathcal{V}} \operatorname{sg}\!\big(\tilde{q}(v \mid s_t)\big) \log \pi_\theta^{\text{pol}}(v \mid s_t)\right]$$

$$\mathcal{L}_{\text{ref}, t} = \frac{\beta}{2}\left(\log \pi_{\text{ref}}^{\text{full}}(a_t \mid s_t) - \log \pi_\theta^{\text{full}}(a_t \mid s_t)\right)^2$$

where $w_{\text{tis}, t} = \operatorname{sg}(\operatorname{trunc\_weight}(\rho_t, m_{\text{tail}, t}))$. Keep $\mathcal{L}_{\text{ref}, t}$ uncentered and unmasked by $m_{\text{tail}, t}$, and keep ratio-free teacher/replay rows uncentered.

### 4.2 Top-$K$ Tail-Renormalized Approximation & WAN Payload Budget

Spark captures the top-$K$ sampled probabilities $\{(v_j, q_j)\}_{j=1}^{K}$ per generated token (default $K = 128$; test $K = 64$ if WAN upload is tight), with residual tail mass $q_{\text{tail}} = \max\!\left(0,\; 1 - \sum_{j \in \mathcal{T}_K} q_j\right)$, and approximates the tail using G4's stop-gradient renormalized tail:

$$\tilde{q}(v \mid s_t) = \begin{cases} q_v & \text{if } v \in \mathcal{T}_K \\[6pt] q_{\text{tail}} \cdot \dfrac{\pi_\theta(v \mid s_t)}{1 - \sum_{u \in \mathcal{T}_K} \pi_\theta(u \mid s_t) + \epsilon} & \text{if } v \notin \mathcal{T}_K \end{cases}$$

**WAN Bandwidth Check (Spark $\to$ Bucket $\to$ G4):**
- Uncompressed flat binary arrays (`topk_ids: int32[T_gen, K]`, `topk_logps: float16[T_gen, K]`): $6K\text{ bytes/token}$.
- At $K = 128$: $768\text{ B/tok}$ raw ($\approx 350\text{--}450\text{ B/tok}$ zstd-compressed). A $4{,}096$-token episode adds $\approx 1.5\text{--}1.8\text{ MiB}$ compressed.
- At $K = 64$: $\approx 0.75\text{--}0.9\text{ MiB}$ compressed per $4{,}096$-token episode.
- Store top-$K$ tensors strictly in compressed binary columns inside `group_<uuid>.zst.parquet` (never JSON), and prune them from the bucket after commit.

### 4.3 $O(1)$-Overhead Chunked Readout Backward in `train/lfm3_fast.py`

Because $\nabla_z \log \pi_\theta(v) = \mathbf{e}_v - \pi_\theta$ and $\sum_v \tilde{q}(v) = 1$, the gradient with respect to raw vocabulary logits $z \in \mathbb{R}^{V}$ simplifies analytically:

$$\frac{\partial \mathcal{L}_{\text{policy}, t}^{\text{TIS+SC}}}{\partial z_v} = -w_{\text{tis}, t} \hat{A}_t \left[\big(\mathbf{1}[v = a_t] - \pi_\theta(v)\big) - \lambda_{\text{sc}}\big(\tilde{q}(v) - \pi_\theta(v)\big)\right]$$

For full SC ($\lambda_{\text{sc}} = 1$):

$$\frac{\partial \mathcal{L}_{\text{policy}, t}^{\text{TIS+SC}}}{\partial z_v} = -w_{\text{tis}, t} \hat{A}_t \cdot \begin{cases} \mathbf{1}[v = a_t] - q_v & \text{if } v \in \mathcal{T}_K \\[6pt] \mathbf{1}[v = a_t] - \gamma_t \, \pi_\theta(v) & \text{if } v \notin \mathcal{T}_K \end{cases}, \qquad \gamma_t = \frac{q_{\text{tail}}}{1 - \pi_\theta(\mathcal{T}_K) + \epsilon}$$

**Implementation advantage:** `train/lfm3_fast.py`'s selected-position chunked readout already computes $\pi_\theta(v)$ chunk-by-chunk without materializing $[T, 128\text{K}]$ logits. Supporting SC/RSC requires only:
1. Gathering $\pi_\theta(\mathcal{T}_K)$ over the $K$ top indices per position to compute $\gamma_t$.
2. Scaling each chunk's $\pi_\theta$ gradient contribution by $(1 - \lambda_{\text{sc}}) + \lambda_{\text{sc}}\gamma_t$ and scatter-subtracting $\lambda_{\text{sc}} q_v$ at the $K$ indices in that chunk.
3. Retaining the frozen tied head with zero head weight-gradient GEMM and $< 5\%$ readout time overhead.

*(Keep decision-option softmax losses centered directly over all $M$ discrete options—exact centering with zero top-$K$ approximation. For student tokens under `natlang_grpo_v2_tis_sc`, ensure $\tilde{q}$ and $\pi_\theta^{\text{pol}}$ share identical support—either full-softmax or matched nucleus truncation.)*

---

## 5. Phased Implementation Plan (Work Packages G–J)

Work packages are ordered so you get the **0-CU/hr DGX Spark offload first** (immediate economic win at $\Delta v_{\max} = 1$ in BF16), and layer on FP8 Spark rollouts and Score Centering only as needed.

### Work Package G: G4 $\leftrightarrow$ Bucket $\leftrightarrow$ DGX Spark Async Bridge & Local Merge
- **Touch:** `scripts/neuralese_data/grpo.py`, `train/learner_state.py`, `serve/sglang_actor.py`, new `train/async_bridge.py`.
- **Deliverables:**
  - Implement the object-storage bridge (`adapters/`, `rollouts/`, `checkpoints/`, `evals/`) with atomic JSON pointer/marker commits and checksum verification.
  - Implement **delta-only upload from G4** ($31\text{--}123\text{ MiB}$) and **local FP32-accumulated merge + atomic SGLang refit on DGX Spark** (preserving multi-route attestation including `readout: decision` and native fallback).
  - Enforce $\Delta v_{\max} = 1$ staleness filtering, $2\times$-batch backpressure on Spark, and `committed_group_ids` deduplication in G4 checkpoints.
  - Offload `rl/held_eval.py` asynchronously to DGX Spark so G4 never pauses for evaluation.
- **Exit Gate:** End-to-end G4 $\leftrightarrow$ Spark RL loop runs continuously at $\Delta v_{\max} = 1$ in BF16; survives a forced Colab G4 disconnect/reconnect with zero duplicate group training and zero Spark crashes; reduces G4 idle wall time by $> 50\%$ vs. synchronous baseline.

### Work Package H: Top-$K$ Capture & Score Centering (SC / RSC)
- **Touch:** `serve/sglang_actor.py`, `train/lfm3_fast.py`, `scripts/neuralese_data/grpo.py`, unit tests.
- **Deliverables:**
  - Add toggleable compressed binary top-$K$ ($K \in \{64, 128\}$) capture on Spark's SGLang generation route; benchmark Spark decode overhead and WAN upload latency.
  - Implement the $O(1)$-overhead chunked readout backward for SC / RSC ($\lambda_{\text{sc}} \in [0, 1]$) in `train/lfm3_fast.py` and register `rl_objective_version = "natlang_grpo_v2_tis_sc"`.
  - Add exact option-level centering for `readout: decision` losses.
  - Log `mean_q_tail`, `p99_q_tail`, and $D_{\text{KL}}(q \parallel \pi)$ per step.
- **Exit Gate:**
  1. Chunked SC/RSC backward matches eager FP32 autograd on $\nabla_z$ and FP32 LoRA factor gradients.
  2. **Zero-drift invariant test:** Under constant advantages ($A_t = +1.0$) and an artificially biased sampler $q \neq \pi_\theta$, full SC ($\lambda_{\text{sc}} = 1$, exact $q$) drives expected policy gradient norm to $\approx 0$.
  3. Readout overhead on G4 is $< 5\%$; top-$K$ decode + WAN overhead on Spark is $< 10\%$.

### Work Package I: High-Concurrency & FP8 Quantized Rollouts on DGX Spark
- **Touch:** `serve/sglang_actor.py` (Spark profile), `ExecutionProfile`, pre-RL diagnostic harness.
- **Deliverables:**
  - Tune Spark rollout concurrency ($32\text{--}64+$ concurrent episodes) within the DGX combined memory ledger; measure G4 starvation rate.
  - Qualify **Stage 2a (FP8 weights + BF16 KV, keeping `conv1d` in BF16/FP32)** through the $200$-prompt pre-RL quantization gate; measure local-merge + FP8-refit latency and decode speedup.
  - Conditionally qualify **Stage 2b (FP8 weights + FP8 KV)** if high concurrency becomes KV-capacity bound.
  - Run a bounded RL comparison on G4 $\leftrightarrow$ Spark: `(BF16 Spark + TIS)` vs. `(FP8 Spark + TIS)` vs. `(FP8 Spark + TIS + SC)`.
- **Exit Gate:** Spark feeds G4 at target utilization without starving the learner, passes multi-route attestation, and matches or beats BF16 held-eval improvement per G4 CU.

### Work Package J: Relaxed Staleness Tuning & Optional SGLang LoRA Slots
- **Touch:** `train/async_bridge.py`, `serve/sglang_actor.py`.
- **Deliverables:**
  - Benchmark $\Delta v_{\max} \in \{1, 2, 4\}$ with `natlang_grpo_v2_tis_sc` to find the highest-throughput staleness cap that preserves held-eval sample efficiency.
  - Only if local merge + refit on Spark exceeds $5\%$ of rollout cycle time: enable refcounted immutable SGLang LoRA slots on Spark (with derived packed ranks $3r = 96$ / $2r = 64$) so in-flight episodes pin cleanly to their starting slot without pausing decode.
- **Exit Gate:** Documented optimal $(\Delta v_{\max}, \lambda_{\text{sc}}, \text{quant\_stage})$ operating point showing clear improvement in **held-eval gain per G4 CU and wall hour** over Phase 1.

---

## 6. Risk Register & Promotion Controls

| Risk | Likelihood / Impact | Concrete Mitigation |
|---|---|---|
| **1. Spark-vs-G4 base weight or tokenizer mismatch** | Low / Critical | Verify identical `base_weights_sha256` and `native_interface_sha256` from `base_manifest.json` on both machines at startup; refuse to launch on mismatch. |
| **2. DGX Spark unified memory OOM (SGLang + 64 TypeScript sandboxes)** | Medium / High | Retain the DGX combined-memory-ledger exclusively on Spark; cap sandbox concurrency and SGLang KV pool jointly with explicit OS headroom. |
| **3. Quantizing LFM3 `conv1d` state corrupts multi-turn trajectories** | High (if unchecked) / Critical | Explicitly exempt `conv1d` recurrent buffers from FP8 quantization; enforce the $200$-prompt multi-turn parity gate in WP-I. |
| **4. Top-$K$ WAN upload or SGLang capture slows Spark more than FP8 speeds it up** | Medium / High | Measure $K \in \{64, 128\}$ decode + zstd upload cost in WP-H before enabling in live RL; keep top-$K$ off when BF16 at $\Delta v_{\max} = 1$ is already stable. |
| **5. High tail mass $q_{\text{tail}}$ weakens Score Centering on high-entropy tokens** | Medium / Medium | Track `p99_q_tail` per step; tune RSC $\lambda_{\text{sc}} < 1$ or cap centering when $q_{\text{tail}} > 0.15$. |
| **6. Async staleness inflates throughput while eroding sample efficiency** | Medium / High | Judge every configuration strictly by **held-eval improvement per G4 CU and wall hour**, never by raw tokens/sec. |

### Final Promotion Rule

Promote the G4 $\leftrightarrow$ DGX Spark Phase 2 configuration to the default RL pipeline **only** if it:
1. Passes all automated correctness, multi-route attestation, and Colab-preemption recovery gates,
2. Meets the existing `rl/held_eval.py` policy-selection rule, and
3. Delivers a clear net reduction in **G4 Compute Units and wall time to a qualifying policy** compared to the Phase 1 synchronous G4 LoRA baseline.