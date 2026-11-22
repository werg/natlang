# LFM3-1.7B Learner Optimization: Comprehensive Design Brief & Implementation Plan

**Status:** Revised implementation specification (v2), grounded in the pinned read-only code audit.  
**Date:** 22 November 2026  
**Audited repository snapshot:** `werg/natlang` at `bfc5a7a7d2020823c6df4ad7d78b9e85093d6cfe` (`[R0]`).  
**Primary execution target:** Single Colab G4 (NVIDIA RTX PRO 6000 Blackwell); retain an L4 execution profile.  
**Fixed constraint:** Train `LFM3-1.7B`. LoRA parameterization, recipe adjustments, and internal refactoring are in scope; held-evaluation comparability is non-negotiable.  
**Primary objective:** Maximize held-out task improvement per unit of wall time and Compute Unit (CU) budget, with deterministic replay and reliable interruption recovery.

---

## Executive Summary & Key Upgrades in v2

Build **one small, adapter-aware LFM3 learner core** shared by the existing SFT (`exp0_sft.py`) and RL (`grpo.py`) drivers. Adopt **BF16 frozen-base LoRA with FP32 trainable factors and FP32 AdamW state** as the initial G4 recipe, while preserving the repository's fast compiled kernels, document packing, prefix-tree sharing, selected-position chunked vocabulary losses, and Neuralese producer/consumer pass-through.

Compared to the v1 draft, this specification adds:
1. **Explicit interface and data-structure contracts** (`AdapterSpec`, `LayerAdapterBundle`, `EffectivePolicy`, `ExecutionProfile`) designed to pass cleanly through `torch.compile` and `torch.utils.checkpoint(..., use_reentrant=False)` without graph breaks or freed-graph errors.
2. **Exact mathematical formulations** for factored projections, global microbatch normalization, Ambient SFT, custom GRPO/decision losses, and G4-vs-L4 CU break-even analysis.
3. **Parameterized memory budgets and a deterministic OOM retry state machine** replacing the legacy full-SFT `88e3` bytes/token/layer heuristic.
4. **An atomic actor-publication state machine and route-attestation matrix** that treats the `sglang_actor.py` native-fallback staleness bug (`[R10]`) as a hard release blocker.
5. **Concrete checkpoint layouts, resume compatibility rules, and milestone exit gates** with explicit risk ownership.

---

## 1. Strategic Decision & Architectural Rationale

### 1.1 Core Decision

- **Unify the mechanism, keep two drivers:** SFT and RL keep their respective objective orchestration, data mixtures, and rollout collectors, while sharing one adapter-aware LFM3 computation, trainable-state, reference-view, checkpoint, and export core.
- **Initial G4 recipe:** Frozen BF16 base weights (the current selected native `LFM3-1.7B` checkpoint), FP32 LoRA factors ($r=32$, $\alpha=64$) across all nine hybrid linear projection families, and resident FP32 AdamW moments.
- **Hardware positioning:** G4 is the primary initial deployment due to team familiarity and high VRAM capacity (NVIDIA specifies $96\text{ GB}$ for the RTX PRO 6000 Blackwell Server Edition `[E1]`; confirm allocated SKU and usable GiB at runtime). Whether G4 beats L4 on CU-efficiency is treated as an empirical measurement, not an assumption.

### 1.2 Structural Gains vs. Non-Goals

| Structural Gain | Mechanism |
|---|---|
| Eliminate full-backbone optimizer state | Remove FP32 master weights, dense gradients, and Adam moments for $\approx 1.7\times 10^9$ frozen parameters. |
| Eliminate frozen weight-gradient GEMMs | Backward computes $\partial \mathcal{L} / \partial x$ through frozen $W_0$ to reach earlier adapters, skipping $\partial \mathcal{L} / \partial W_0$. |
| Trade freed VRAM for compute | Retain more layer activations and prefix states on G4, cutting activation recomputation. |
| Lightweight reference policy | Replace full-weight reference swapping with small frozen FP32 reference-factor snapshots when base weights match. |
| Frequent, cheap recovery | Resumable learner state shrinks to adapter factors, optimizer moments, and metadata, enabling per-step saves. |

**Explicit Non-Goals (Do Not Build):**
- Do not replace the custom SFT/RL loops with a generic HF/PEFT/TRL trainer (would break weighted targets, prefix trees, Neuralese state pass-through, and custom decision/nucleus laws).
- Do not pull the hierarchical program-tuning controller (`tune/parameterization.py`) into global SFT/RL; reuse only its small arithmetic/init primitives.
- Do not introduce FP8/FP4 training, QLoRA on G4, custom Triton kernels, or a TPU accelerator abstraction in the initial delivery.

---

## 2. Codebase Audit & Refactoring Scope

Findings are pinned to `bfc5a7a7d2020823c6df4ad7d78b9e85093d6cfe`. **Pre-flight requirement:** Reconcile `HEAD` against this SHA before editing.

| Component | Audited Behavior | Refactoring Consequence |
|---|---|---|
| `scripts/neuralese_data/exp0_sft.py` `[R1]` | Loads native checkpoints into vendored HF LFM3; builds weighted per-call chains; allocates FP32 masters/moments for *all* named params; runs Ambient SFT; checkpoint writer records source step rather than learner step. | Keep data/objective orchestration; extract model, parameterization, reference, and state lifecycle; fix learner-step provenance. |
| `scripts/neuralese_data/grpo.py` `[R3]` | Active RL driver importing chain/loss/packing from SFT; owns rollouts, rewards, decisions, `ReferenceWeights`, `WeightSync`, and partial state saves. | Share core mechanisms with SFT; keep custom RL objective, collector, and staleness tracking; harden recovery and publication. |
| `scripts/neuralese_data/lfm3_fast_train.py` `[R2]` | Compiled attention, causal conv, MLPs, `levels` prefix trees, imported state, selected-row chunked readout, and `88e3` B/tok/layer memory planner. `_layer_call` reads `.weight` directly into `F.linear`. | Move to `train/lfm3_fast.py`; pass explicit adapter factor bundles into all four layer paths (ordinary/past $\times$ conv/attn) and MLPs; recalibrate memory planner. |
| `model/native_adapters.py` & `tune/parameterization.py` `[R4][R5]` | FP32 free LoRA factors, scaling, composition, and factored projection math for native program tuning. | Extract one pure factored-projection helper; keep native wrappers calling it; ensure scale $\alpha/r$ is applied exactly once. |
| `tune/targets.py`, `tune/export.py`, `export/native_hf.py` `[R6][R7][R8]` | LFM3 target inventory, PEFT/GGUF conversion, native interface merge, packed SGLang weight mappings. | Single canonical target inventory and merge/scaling contract across training, export, and serving. |
| `scripts/neuralese_data/expert_iteration.py` `[R9]` | Launches SFT with DGX-specific paths and combined host/device memory budgets. | Introduce an `ExecutionProfile` boundary (`dgx`, `colab_g4`, `colab_l4`) without changing learner objectives. |
| `serve/sglang_actor.py` & `grpo.py::WeightSync` `[R3][R10]` | Dense weight rebind ignores LoRA deltas; native fallback stays pinned to start checkpoint after dense rebind. | Publish immutable merged effective checkpoints first; update **all** serving routes (including native fallback) atomically. |
| Probes (`behaviour_probe.py`, `sanity_logprob_paths.py`, `operator_ablation.py`, `passthrough_check.py`) `[R11]` | Inspect behavior, logprobs, and native-state use via shared fast-forward path. | Must load the effective adapter policy or its merged export; probing bare base weights produces invalid diagnostics. |
| `rl/held_eval.py` & engine consumers `[R0][R12]` | Judges served checkpoints under pinned engine, sampling, and scoring identities. | Preserve full export parity across native, SGLang, managed `llama-server`, and in-page engines. |

### Six Blocking Structural Findings

1. **Module wrappers are bypassed:** `lfm3_fast_train._layer_call` invokes `F.linear(x, mod.weight, ...)` directly `[R2]`. Neither PEFT nor native wrapper hooks fire on this path.
2. **Full-tuning optimizer lifecycle:** `exp0_sft.py` allocates FP32 masters and Adam states for every parameter regardless of `requires_grad` `[R1]`.
3. **Obsolete memory constant:** The `88e3` bytes/token/layer planner assumes full-tuning activation retention and wastes G4 memory on unnecessary recomputation `[R2]`.
4. **No valid stock oracle:** SFT's `--stock-model` swaps nucleus likelihood for full softmax and lacks imported-state parity `[R1]`. A dedicated eager reference test is required.
5. **Incomplete recovery & provenance bug:** SFT saves inference-only weights and writes the source checkpoint's step instead of the learner step; RL lacks a complete RNG/collector recovery contract `[R1][R3]`.
6. **Serving route split-brain:** `sglang_actor.py`'s dense rebind leaves its native fallback on the start checkpoint `[R10]`. Front-end SHA verification alone cannot catch this.

---

## 3. Target Architecture & Interface Contracts

### 3.1 Module Layout

| Location | Responsibility |
|---|---|
| `model/native_adapters.py` (existing) | Pure `factored_linear(x, base_w, lora_a, lora_b, scale, bias=None)` primitive and scaling/merge utilities; existing wrappers delegate to it. |
| `tune/targets.py` (or moved to `model/targets.py` if needed to avoid circular imports) | Single canonical LFM3 projection inventory, target resolver, and packed-rank calculator. |
| `train/lfm3_learner.py` (new) | Effective-policy loader, trainable-set builder, precision policy, AdamW factory, and immutable reference-view manager. |
| `train/lfm3_fast.py` (moved from `scripts/neuralese_data/lfm3_fast_train.py`) | Shared adapter-aware fast kernels (ordinary/past conv & attn, MLP), `levels` prefix tree, chunked readout, and parameterized memory planner. Update all imports in the same commit. |
| `train/learner_state.py` (new) | Versioned resumable state serialization, atomic staging, immutable merged BF16 export, and provenance manifest writer. |
| `scripts/neuralese_data/exp0_sft.py` & `grpo.py` (existing) | Objective-specific batch construction, loss math, rollout collection, and CLI entry points. |
| `scripts/neuralese_data/expert_iteration.py` (existing) | Selects `ExecutionProfile` (`dgx`, `colab_g4`, `colab_l4`) and dispatches resolved recipes. |

### 3.2 Core Data Contracts

Keep contracts as lightweight frozen dataclasses and flat tuples so they are transparent to `torch.compile` and `torch.utils.checkpoint`:

```python
@dataclass(frozen=True)
class AdapterSpec:
    target_families: tuple[str, ...]  # 9 canonical LFM3 linear families
    rank: int                         # Initial: 32
    alpha: float                      # Initial: 64.0
    dropout: float = 0.0              # Fixed 0.0 for deterministic prefix & RL parity
    gate_rank: int | None = None      # None = frozen initially (max <= 16 if enabled)
    dtype: str = "float32"            # Canonical trainable storage dtype

@dataclass(frozen=True)
class LinearFactorLeaf:
    """Canonical FP32 leaf tensors. Never cache derived scaled/cast tensors here."""
    a: torch.Tensor                   # [rank, in_features], FP32 leaf
    b: torch.Tensor                   # [out_features, rank], FP32 leaf
    scale: float                      # alpha / rank (Python float, applied once)

@dataclass(frozen=True)
class LayerAdapterBundle:
    """Flat, fixed-topology per-layer factor container passed into compiled kernels."""
    layer_idx: int
    layer_kind: Literal["attn", "conv"]
    # Attention projections (None on conv layers)
    q_proj: LinearFactorLeaf | None = None
    k_proj: LinearFactorLeaf | None = None
    v_proj: LinearFactorLeaf | None = None
    o_proj: LinearFactorLeaf | None = None
    # Convolution projections (None on attn layers)
    in_proj: LinearFactorLeaf | None = None
    out_proj: LinearFactorLeaf | None = None
    # Shared MLP projections
    w1: LinearFactorLeaf | None = None
    w2: LinearFactorLeaf | None = None
    w3: LinearFactorLeaf | None = None

@dataclass(frozen=True)
class EffectivePolicySpec:
    base_checkpoint_id: str
    base_weights_sha256: str
    tokenizer_sha256: str
    native_interface_sha256: str
    parameterization: Literal["lora", "full"]
    adapter_spec: AdapterSpec | None
    policy_version: int
    effective_policy_sha256: str      # Deterministic digest of base + sorted factor tensors
```

### 3.3 First-Class Factored Projection & Autograd Contract

For any targeted linear matrix $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ with canonical FP32 leaf factors $A \in \mathbb{R}^{r \times d_{\text{in}}}$, $B \in \mathbb{R}^{d_{\text{out}} \times r}$, and scale $s = \alpha / r$:

$$y = \operatorname{cast}_{\text{out}}\!\left(\operatorname{fp32}\!\left(W_0 x\right) + s \cdot B_{\text{fp32}}\!\left(A_{\text{fp32}} \, x_{\text{fp32}}\right)\right)$$

**Mandatory Execution Rules:**
1. **Explicit FP32 low-rank reference branch:** Wrap the low-rank matmuls and residual addition inside `torch.amp.autocast("cuda", enabled=False)` so outer BF16 autocast cannot silently downcast FP32 factor GEMMs `[E3]`. Record `torch.backends.cuda.matmul.allow_tf32` / `float32_matmul_precision` in the run manifest; the small mathematical oracle uses strict FP32 (`allow_tf32=False`).
2. **Apply scaling exactly once:** Because some existing helpers in `tune/export.py` fold $s$ into $B$ `[R7]`, canonical training leaves store *unscaled* $A$ and $B$ alongside scalar $s = \alpha / r$. Folding occurs only inside explicit export/merge functions.
3. **No dense $W_0 + s B A$ materialization in training:** Keep $W_0$ frozen in BF16 without `torch.no_grad()` around the backbone so $\partial \mathcal{L} / \partial x$ flows unobstructed to earlier layers' adapters.
4. **Leaf-only bundles for multi-backward prefix trees:** `levels` prefix execution runs separate `.backward()` calls per child leaf before backpropagating into shared prefixes `[R2]`. Bundles must pass only canonical leaf tensors `(a, b)` and Python float `scale`. Never precompute and cache a non-leaf `b_scaled = b * scale` or cast tensor across backward calls—doing so triggers freed-graph errors on the second child backward.
5. **Direct FP32 gradient accumulation:** Accumulate directly into leaf `.grad` in FP32, zero once per optimizer step, and clip/step once. Remove the legacy BF16-to-master post-accumulate gradient hook from FP32 LoRA leaves `[R1]`.
6. **Non-reentrant checkpointing consistency:** Keep `torch.utils.checkpoint(..., use_reentrant=False)` `[R2][E4]`. Pass leaf factor tensors as explicit positional arguments (flattened tuple) into the checkpointed function rather than mutating module attributes, ensuring recomputation always observes identical factors, reference view, and numerical flags.

---

## 4. Initial Training Recipe & Controlled Tradeoffs

All values below are **starting proposals** to validate correctness and establish a G4 baseline, not claimed optima.

| Parameter | Initial Setting | Rationale & Follow-up Rule |
|---|---|---|
| Base model | Selected native `LFM3-1.7B` checkpoint, frozen BF16 | Preserves the trained starting policy; HF checkpoint provides config only. |
| Target families (9) | Attn: `q_proj`, `k_proj`, `v_proj`, `o_proj`; Conv: `in_proj`, `out_proj`; MLP: `w1`, `w2`, `w3` | Covers all major hybrid linear maps; aligns with Liquid's LFM3 recipe `[E2]` and broad-coverage LoRA guidance `[E8]`. Fail fast at startup if any target fails to resolve. |
| Rank & Alpha | $r = 32$, $\alpha = 64$ ($s = 2.0$) | Moderate global adaptation capacity. Log exact trainable parameter count $P$ at startup. Do not auto-sweep ranks; test $r=16$ only for cost reduction or $r=64$ only if held quality is capacity-limited after LR/data checks. |
| Attention gate | Frozen | Output width is $16$; native constructor rejects $r > \min(d_{\text{in}}, d_{\text{out}})$ `[R5][R6]`. If enabled later, require explicit `gate_rank` $\le 16$ and per-target scaling. |
| Norms, conv kernels, tied embed/head | Frozen | Removes massive head weight-gradient GEMM and optimizer state. Deliberate change from full SFT; revisit trainable rows only if held eval shows capacity limits. |
| Marker/override rows & native interface | Preserve existing frozen/override semantics `[R1][R8]` | Prevents silent drift in Neuralese special tokens and interface tensors. |
| Initialization | $A \sim \text{Kaiming/Gaussian}$, $B = 0$ `[E11]` | Exact identity with starting base at step $0$. Expect $\|\nabla_A \mathcal{L}\| = 0$ on step $0$ while $B$ moves first. |
| LoRA dropout | $0.0$ | Preserves deterministic prefix reuse and train/serve logprob parity. |
| Optimizer | FP32 AdamW, resident on GPU; weight decay $0.0$; $\epsilon = 10^{-8}$; grad clip norm $1.0$ | Retain SFT $\beta = (0.9, 0.95)$ and RL $\beta = (0.9, 0.99)$. Record zero weight decay as an explicit recipe choice. |
| SFT Learning Rate | $1 \times 10^{-4}$ | LoRA requires higher LR than full SFT ($5 \times 10^{-6}$). Note that changing $r$ while holding $\alpha = 2r$ changes effective scaling and requires LR re-verification `[E8]`. |
| RL Learning Rate | $1 \times 10^{-5}$ | Conservative migration start preserving spread-based LR scaling; inspect advantage-directed KL/update norms before increasing. |
| Effective Batch Size | SFT: $32{,}000$ weighted target units `[R9]`; RL: preserved groups & spread scaling | Increase physical microbatch packing on G4 to reduce accumulation steps without altering effective optimizer batch or source weights. |
| Context & Auxiliary Losses | Retain admitted contexts and enabled Ambient/RL auxiliary terms | Use length/token buckets; no blanket truncation. Objective changes are separate experiments after parity is proven. |

### Deferred Alternatives & Revisit Triggers

- **QLoRA:** Deferred. A $1.7\text{B}$ BF16 base occupies $\approx 3.2\text{ GiB}$; 4-bit quantization saves $< 2.4\text{ GiB}$ before metadata—negligible on a $96\text{ GB}$ G4. Reconsider only if an L4 workload genuinely fails to fit after activation/packing tuning.
- **BF16 Adapter GEMMs:** Deferred until profiling shows FP32 low-rank matmuls are a material fraction of step time; requires passing the full probability/gradient parity suite.
- **Full Tuning:** Retained as a first-class mode (`parameterization="full"`) in `train/lfm3_learner.py` for regression comparison and fallback if held metrics show LoRA capacity limits.

---

## 5. Preserving Learning Semantics During Migration

### 5.1 SFT Contract (`exp0_sft.py`)

1. **Exact sequence & mask preservation:** Retain per-call token sequences, prefix-extension chain merging, empty-think / repeated-close / post-end masks, source weights, and zero-weight causal context (prompt, tool, harness tokens).
2. **Global weighted denominator across microbatches:** For an optimizer batch partitioned into $M$ physical microbatches $\mathcal{B}_1, \dots, \mathcal{B}_M$, accumulate *weighted loss numerators* and divide by the global batch weight sum $W_{\text{total}}$:
   $$\mathcal{L}_{\text{batch}} = \frac{\sum_{m=1}^{M} \sum_{i \in \mathcal{B}_m} w_i \, \ell_i}{W_{\text{total}}}, \quad W_{\text{total}} = \sum_{m=1}^{M} \sum_{i \in \mathcal{B}_m} w_i$$
   Never average unequal microbatch means.
3. **Probability law invariance:** Preserve student-sampled nucleus likelihood vs. full-softmax teacher/action likelihood regardless of execution backend `[R1]`.
4. **Ambient SFT semantics:** Preserve $p_{\text{clean}} = 0.75$ and $T = 2.5$ (without a $T^2$ multiplier), making one deterministic per-optimizer-step choice between clean weighted NLL and full-vocabulary temperature-matched teacher-to-student distillation on the same weighted positions `[R1]`. Persist Ambient RNG seed, step, and recipe in the learner checkpoint.
5. **Reference view correctness:** Disabling adapters (`bundle = None`) is valid for the teacher/reference pass **only** when the intended reference equals the bare frozen base. Continuing an existing adapter run requires a frozen FP32 snapshot of its run-start factors (`ref_bundle`).

### 5.2 RL Contract (`grpo.py`)

Preserve the explicit custom objective version (`rl_objective_version = "natlang_grpo_v1"`) rather than substituting a generic PPO/TRL loss `[R3][E7]`:

$$\rho_t = \exp\!\left(\operatorname{sg}\!\left(\log \pi_\theta^{\text{full}}(a_t \mid s_t)\right) - \log \pi_{\text{beh}}^{\text{full}}(a_t \mid s_t)\right)$$

$$\mathcal{L}_{\text{policy}, t} = -\operatorname{sg}\!\left(\operatorname{trunc\_weight}(\rho_t, m_{\text{tail}, t})\right) \cdot \hat{A}_t \cdot \log \pi_\theta^{\text{pol}}(a_t \mid s_t)$$

$$\mathcal{L}_{\text{ref}, t} = \frac{\beta}{2}\left(\log \pi_{\text{ref}}^{\text{full}}(a_t \mid s_t) - \log \pi_\theta^{\text{full}}(a_t \mid s_t)\right)^2$$

where $\operatorname{sg}(\cdot)$ denotes stop-gradient (`.detach()`), $\pi_\theta^{\text{pol}}$ uses the learner's nucleus law for student samples, and $\pi^{\text{full}}$ uses untruncated probabilities.

**Mandatory RL Invariants:**
- Never apply the policy tail mask $m_{\text{tail}, t}$ to $\mathcal{L}_{\text{ref}, t}$.
- Preserve group-relative advantages, zero-spread group handling, spread-based LR scaling, decision-option softmax losses (with captured behavior option probabilities and chosen-option penalties), and ratio-free teacher/replay demonstration rows.
- Keep three policy identities strictly distinct:
  1. **Current policy ($\pi_\theta$):** Frozen base + live trainable factors.
  2. **Behavior policy ($\pi_{\text{beh}}$):** Effective policy that sampled each token/decision; captured logprobs remain authoritative across staleness/version spans.
  3. **Reference policy ($\pi_{\text{ref}}$):** Fixed run-start effective policy; a checkpoint resume must never reset $\pi_{\text{ref}}$ to the latest policy or bare base.
- **Reference numerical parity:** Execute the reference pass through the same compiled fast path with detached outputs and immutable `ref_bundle` (completing and discarding the reference forward before running any policy forward that retains a backward graph).

### 5.3 Packing, Prefix Trees & Neuralese Dependencies

- **Dual isolation:** Packed documents must enforce strict boundaries in **both** attention masks/positions (including sliding windows) and causal convolution state resets `[R2]`.
- **Prefix reuse validity:** Reuse `levels` prefix states strictly within a single optimizer step under identical `(effective_policy_sha256, numerical_flags, inputs, rng_state)`. Aggregate all child adjoints into a prefix node before triggering its backward pass. Invalidate all prefix/KV caches immediately on optimizer step or policy/reference switch.
- **Differentiable Neuralese pass-through:** Preserve differentiable recomputation of recorded producer documents into consumer adapters `[R1][R2][R3]`. Explicitly tag archived/detached states so they are never confused with differentiable pass-through; verify frozen embeddings do not sever producer-to-consumer adapter gradients.

---

## 6. G4 Memory Engineering, Compilation & Hardware Economics

### 6.1 Parameterization-Aware Memory Model

Let $P$ be the count of trainable FP32 adapter parameters (for `LFM3-1.7B` at $r=32$ across 9 projection families, estimate $P$ dynamically at startup; typically $P \approx 30\text{M}\text{--}60\text{M}$ depending on exact hidden/MLP widths). Static resident storage is:

| Component | Full SFT (BF16 + FP32 Masters) | LoRA ($r=32$, FP32 Trainables) | Notes |
|---|---|---|---|
| Base weights (BF16) | $\approx 3.2\text{ GiB}$ | $\approx 3.2\text{ GiB}$ | Frozen in LoRA |
| Trainable FP32 params + grads + 2 Adam moments | $\approx 27.2\text{ GiB}$ ($16 \times 1.7\text{B}$) | $16P\text{ bytes}$ ($\approx 0.5\text{--}1.0\text{ GiB}$) | Resident on GPU; no CPU offload needed |
| Reference policy state | $\approx 3.2\text{ GiB}$ (full copy/swap) | $4P\text{ bytes}$ ($\approx 0.12\text{--}0.25\text{ GiB}$) | $0\text{ B}$ if reference equals bare base |
| **Static learner subtotal** | **$\approx 33.6\text{ GiB}$** | **$\approx 3.8\text{--}4.5\text{ GiB}$** | Frees $\approx 29\text{ GiB}$ before activations |

All new memory telemetry must be reported in **GiB** ($1024^3$ bytes), not ambiguously labeled `*_gb` `[R1][R2][R3]`. Host RAM on Colab discrete G4 is a separate budget from VRAM; remove DGX unified-memory assumptions from the Colab execution profile.

**Replacing the `88e3` Constant:**
- Replace the single full-SFT constant with a small calibration table keyed by `(parameterization, adapter_arithmetic, layer_kind)` measuring bytes/token/layer for: (a) retained activations on uncheckpointed layers, (b) checkpointed boundary activations, (c) prefix/history state, and (d) chunked $128\text{K}$-vocabulary readout workspace.
- Reserve $10\%\text{--}15\%$ of usable VRAM plus known resident actor/export staging needs as an initial launch safety margin, then tune from observed `max_memory_allocated` / `max_memory_reserved` peaks.
- Keep selected-position chunked readout (both NLL and Ambient distillation) so the frozen tied head never materializes full-sequence $128\text{K}$ logits or computes head weight gradients `[R2]`.

### 6.2 Deterministic OOM Retry State Machine

Never silently drop or truncate difficult long-context examples on `torch.OutOfMemoryError`. Execute the following bounded recovery loop per optimizer step:

1. **Snapshot step-entry state:** Record CPU/CUDA/Python/NumPy RNG states, Ambient step choice, and current physical plan `(max_tokens_per_microbatch, checkpoint_ratio)`.
2. **On `torch.OutOfMemoryError`:**
   - Immediately clear `optimizer.zero_grad(set_to_none=True)` and discard all retained prefix trees, autograd graphs, and partial step accumulators.
   - Call `gc.collect()` and `torch.cuda.empty_cache()`.
   - Restore step-entry RNG states and Ambient choice.
   - Reduce physical plan deterministically: first increase activation checkpointing ratio toward $1.0$, then halve `max_tokens_per_microbatch` down to a single indivisible chain/prefix group.
   - Retry the **exact same optimizer batch**. Count wall time toward cost telemetry, but commit supervision exposure **only once** upon successful `optimizer.step()`.
3. **Hard abort:** If a single indivisible chain OOMs at full activation checkpointing, fail loudly with a structured diagnostic dump rather than silently dropping data.

### 6.3 Compilation & Input Pipeline Discipline

- **Fixed graph topology:** Install the adapter layout once before `torch.compile`; never mutate which modules carry adapters inside a run `[E5]`.
- **Bucketed shapes over blanket `dynamic=True`:** Use a small set of meaningful token/length buckets and selective dynamic dimensions where shape churn is measured `[E6]`. Log compile time, graph breaks, and recompile counts separately from steady-state step time.
- **Local SSD staging:** Stage immutable tokenized/rendered datasets on local SSD outside the optimizer loop. Measure data-wait vs. H2D transfer vs. GPU kernel time before tuning CPU worker counts or host-pinned memory.

### 6.4 G4 vs. L4 Economic Comparison Formula

Keep the exact same objective, target layout, effective batch size, and canonical unit definition (e.g., original chain-token exposure with fixed targets and masks—*not* prefix-deduplicated computed tokens) across both GPUs. Profiles vary only physical packing, checkpointing ratio, readout chunk size, and actor residency.

$$\text{CU per } 10^6 \text{ units} = \text{CU\_rate}_{\text{obs}}\;[\text{CU/h}] \times \frac{t_{\text{elapsed}}\;[\text{s}]}{3600} \times \frac{10^6}{N_{\text{canonical}}}$$

G4 is more CU-efficient than L4 on equal canonical work if and only if:

$$\frac{\text{Throughput}_{\text{G4}}}{\text{Throughput}_{\text{L4}}} > \frac{\text{CU\_rate}_{\text{G4}}}{\text{CU\_rate}_{\text{L4}}}$$

Final allocation decisions must amortize **full-cycle** time and CU (cold start, compilation, training, checkpointing, export/refit, rollout collection, and held evaluation).

---

## 7. Actor Publication, Serving Parity & Scheduling

### 7.1 Milestone 1 Publication: Immutable Merged BF16 Export

Do not merge/unmerge weights in place on the live training model. Construct an immutable effective checkpoint from `(frozen base, frozen factor snapshot)` using FP32 accumulation:

$$W_{\text{eff}} = \operatorname{cast}_{\text{bf16}}\!\left(\operatorname{fp32}(W_0) + \frac{\alpha}{r} B_{\text{fp32}} A_{\text{fp32}}\right)$$

Preserve tied embeddings/head and native interface conventions `[R4][R7][R8]`. Refactor `WeightSync` to consume this immutable effective export rather than walking the live training module `[R3]`.

**Atomic Publication State Machine:**
1. **`SNAPSHOT`:** Freeze committed optimizer step $k$ factors and compute `effective_policy_sha256`.
2. **`EXPORT`:** Materialize complete merged BF16 artifact in a staging directory; verify tensor checksums and target count; atomically rename to immutable version path.
3. **`PAUSE`:** Pause rollout generation at a safe engine boundary. If the pinned actor cannot accurately record mixed-version probability spans for in-flight requests, drain/finish affected requests before refit—never attribute a mixed request solely to version $k$ `[R3][R10][E9]`.
4. **`REFIT & INVALIDATE`:** Push merged weights to the actor; flush KV caches, convolution/prefix states, and any adapter caches required by the pinned SGLang fork.
5. **`BIND ALL ROUTES` (Release Blocker):** Update **all four** serving routes to version $k$:
   - Generation route
   - Decision / option-scoring route
   - Logprob capture route
   - **Native fallback route** (explicitly fixing the audited `sglang_actor.py` split-brain defect `[R10]`)
6. **`ATTEST & RESUME`:** Run a lightweight multi-route attestation check (verifying `effective_policy_sha256` and a sentinel logprob hash across active routes). Resume collection only if all routes attest version $k$.
7. **`ROLLBACK` (On any failure in steps 2–6):** Restore the last verified complete version $k-1$ across all routes and halt or retry before admitting new rollouts.

### 7.2 Milestone 2 Optimization (Conditional): SGLang Adapter Slots

Enable only if profiling shows merged export + refit is a material fraction of RL cycle time. Reuse existing `_sglang_adapter_file`, acquire/release lifecycle, and per-request adapter bindings `[R10]`, with three mandatory safeguards:
1. **Packed-rank capacity derivation:** Because `export/native_hf.py` packs fused projections, compute required serving rank from the packed layout `[R8][R10]`:
   - Fused `q_proj` + `k_proj` + `v_proj`: $r_{\text{packed}} = 3r$ (nominal $r=32 \implies r_{\text{packed}} = 96$).
   - Fused `w1` + `w3`: $r_{\text{packed}} = 2r$ (nominal $r=32 \implies r_{\text{packed}} = 64$).
   The existing default actor rank capacity of $32$ is insufficient and must be derived automatically from `AdapterSpec`.
2. **Immutable slot versioning:** Never overwrite an adapter slot held by an in-flight request; load into a new immutable slot, atomically flip the default pointer for new requests, and release the old slot once its refcount reaches zero.
3. **Exact base binding:** Reject loading any global or per-program adapter whose `base_weights_sha256` or `native_interface_sha256` does not match the mounted base.

### 7.3 Learner/Actor Scheduling

- Start with **exclusive GPU-heavy learner phases** (actor paused or resident-idle if VRAM permits and residency saves reload time).
- Evaluate concurrent collection/training only after profiling demonstrates real GPU/CPU slack, while preserving staleness caps and exact provenance. Do not blindly carry over GB10 concurrency flags to G4 `[R3][R14]`.

---

## 8. Recovery, Provenance & Lineage Contract

### 8.1 Three Distinct Artifacts

| Artifact | Storage Layout & Lifecycle |
|---|---|
| **Immutable Base** | Stored once per lineage outside ephemeral VM: `weights.safetensors`, `config.json`, tokenizer, native interface, and `base_manifest.json` with SHA-256 digests. |
| **Resumable Learner State** | Saved atomically per committed optimizer step (keep latest + previous verified version): `adapter_factors_fp32.safetensors`, `optimizer_state.pt`, `rng_and_cursors.pt`, `ref_factors_fp32.safetensors` (if non-base), and `learner_manifest.json`. |
| **Inference Export** | Immutable merged BF16 checkpoint (or qualified base+adapter bundle) with `effective_policy_sha256`, actual `learner_step`, and `ops/current-policy.json` updated **only** via the held-evaluation promotion gate. |

**Required `learner_manifest.json` Fields:**
`schema_version`, `git_sha`, `learner_step` (actual optimizer steps, fixing `[R1]`), `source_checkpoint_id`, `base_weights_sha256`, `effective_policy_sha256`, `reference_policy_sha256`, `adapter_spec`, `optimizer_config`, `objective_version`, `ambient_state` (seed, step, $p_{\text{clean}}$, $T$), `data_manifest_sha256`, `batch_cursor`, `rl_rollout_ledger` (consumed IDs, pending/buffered group IDs, auxiliary buffer offsets), and `hardware_profile` (GPU SKU, usable GiB, PyTorch/CUDA/SGLang versions, `allow_tf32`).

### 8.2 Recovery Guarantees & Lineage Transitions

- **Deterministic Learner Replay:** Restoring a committed checkpoint on the same software/hardware stack and replaying the same captured batch reproduces the exact schedule, Ambient choice, state transitions, and numerically equivalent parameter update.
- **Valid Live Resume:** Restores exact policy/reference/optimizer lineage and rollout-consumption ledger (preventing duplicate training on replayed RL groups), while explicitly logging any in-flight external episodes abandoned across a VM restart.

| Transition Type | Policy Weights | Optimizer & Schedule | Reference Policy ($\pi_{\text{ref}}$) | Lineage Status |
|---|---|---|---|---|
| **Same-run resume** | Restored factors | Restored exactly | Restored unchanged | Exact continuation |
| **SFT $\to$ RL handoff** | Promoted SFT effective policy | Fresh optimizer & schedule | Frozen at RL run start | New stage |
| **Merge-and-rebase** | Merged BF16 becomes new base; fresh $B=0$ factors | Fresh optimizer & schedule | Explicitly declared | New lineage |
| **Rank / target change** | Re-initialized or projected explicitly | Fresh optimizer & schedule | Explicitly declared | New lineage (never exact resume) |

On object storage (e.g., GCS/Drive), write immutable data blobs first, verify durability, and commit `learner_manifest.json` last via atomic pointer update.

---

## 9. Phased Implementation Plan

Work packages are structured as small, independently mergeable PRs/commits. Ready improvements enter the next pinned run without halting useful training.

### Pre-Flight: Repository Reconciliation
- Diff `HEAD` against audited SHA `bfc5a7a7d2020823c6df4ad7d78b9e85093d6cfe` across `exp0_sft.py`, `grpo.py`, `lfm3_fast_train.py`, `native_adapters.py`, `sglang_actor.py`, and `expert_iteration.py`. Update target line/symbol mappings before starting Work Package A.

### Work Package A: Contracts & Shared Core Extraction
- **Touch:** `exp0_sft.py`, `grpo.py`, `lfm3_fast_train.py` $\to$ `train/lfm3_fast.py`, `train/lfm3_learner.py`, probes, tests.
- **Deliverables:**
  - Capture one representative SFT batch fixture and one RL replay batch fixture (including masks, weights, decisions, native imports, and resolved loss config).
  - Move `lfm3_fast_train.py` to `train/lfm3_fast.py` and update all production and diagnostic callers in one atomic commit.
  - Extract model loading and parameterization-independent trainable enumeration into `train/lfm3_learner.py`, keeping `parameterization="full"` fully functional.
- **Exit Gate:** Both captured full-tuning replay fixtures run through the extracted core with identical target counts, loss values, and optimizer behavior within the existing numerical envelope.

### Work Package B: Adapter-Aware Computation & Trainable State
- **Touch:** `model/native_adapters.py`, `tune/targets.py`, `train/lfm3_fast.py`, `train/lfm3_learner.py`.
- **Deliverables:**
  - Implement `AdapterSpec`, `LinearFactorLeaf`, `LayerAdapterBundle`, and the pure FP32-reference `factored_linear` primitive (with `autocast(enabled=False)`).
  - Wire explicit `LayerAdapterBundle` tuples into all four fast layer paths (ordinary/past $\times$ conv/attn) and MLPs; pass leaf tensors flat through `checkpoint(use_reentrant=False)`.
  - Enforce 9-family target resolution with startup parameter-count logging and fail-fast validation.
  - Allocate AdamW state strictly for FP32 trainable leaves; remove legacy BF16-master gradient hooks on FP32 leaves.
  - Build a small independent eager FP32 mathematical oracle (matching exact nucleus/full-vocab laws, masks, and native imports—*not* `--stock-model`).
- **Exit Gate:** All 9 target families update; frozen base/head/interface tensors have zero drift and zero optimizer entries; packed, prefix-tree (multi-child separate backwards), and Neuralese pass-through paths match the eager FP32 oracle; reference-view switching preserves exact equal-policy logprobs.

### Work Package C: SFT LoRA Delivery, Recovery & Merged Export
- **Touch:** `exp0_sft.py`, `train/learner_state.py`, `export/native_hf.py`, `expert_iteration.py`, probes.
- **Deliverables:**
  - Route SFT through LoRA trainables ($r=32$, $\alpha=64$, LR $10^{-4}$) preserving weighted microbatch numerators, nucleus/full-vocab laws, and Ambient SFT ($p_{\text{clean}}=0.75$, $T=2.5$).
  - Implement `train/learner_state.py` atomic saves, fix the `learner_step` provenance bug `[R1]`, and implement immutable merged BF16 export.
  - Add `ExecutionProfile` (`dgx`, `colab_g4`, `colab_l4`) to `expert_iteration.py` removing DGX-only assumptions from Colab runs.
  - Update probes to load effective adapter policies / merged exports.
- **Exit Gate:** End-to-end G4 SFT cycle trains, survives a forced interrupt/resume with deterministic replay of the next batch, exports a merged BF16 checkpoint verified against the factored learner, and completes held evaluation under the existing promotion rule.

### Work Package D: RL Learner Integration & Atomic Publication
- **Touch:** `grpo.py`, `WeightSync`, `serve/sglang_actor.py`, `train/learner_state.py`, probes.
- **Deliverables:**
  - Switch RL learner mechanics to shared LoRA core while keeping `natlang_grpo_v1` objective, decision losses, and enabled auxiliary terms intact.
  - Replace full-weight `ReferenceWeights` I/O with immutable `ref_bundle` snapshots when base weights match; remove full-model moment offload on G4.
  - Implement the 7-step atomic publication state machine using merged BF16 exports; **fix `sglang_actor.py` native-fallback route binding** and add multi-route attestation.
  - Persist RL rollout ledger and RNG state for valid live resume and zero double-training.
- **Exit Gate:** Captured RL replay matches factor gradients and decision losses; a bounded live `collect -> update -> publish -> attest -> collect` cycle passes route attestation and behavior probes.

### Work Package E: G4 Performance Tuning & Economic Comparison
- **Touch:** `train/lfm3_fast.py` (memory planner), compile config, `ExecutionProfile`, optional SGLang adapter slots.
- **Deliverables:**
  - Calibrate LoRA activation/prefix memory profiles from measured G4 peaks; implement the deterministic OOM retry state machine.
  - Increase physical packing and reduce checkpointed layers while holding effective batch constant; audit `torch.compile` graph breaks and bucket shapes.
  - Conditionally evaluate (only if profiling justifies): BF16 adapter GEMMs or SGLang adapter slots (with derived packed ranks $3r=96$ / $2r=64$).
  - Run a matched canonical-work L4 segment when deciding default hardware allocation.
- **Exit Gate:** Chosen profile improves canonical useful work per wall hour or CU without objective drift or held-quality regression; record exact GPU SKU, usable GiB, CU rates, and software versions.

### Work Package F: Consolidation & Cleanup
- Promote the qualifying recipe/profile via the held-selection rule; update operational docs and `DECISIONS.md` with measured results; remove temporary migration shims.

### Dependency Graph & Planning Allowance

- **Critical path:** `Pre-Flight -> A -> B -> C -> D (publication) -> E -> F`.
- **Parallelizable:** `train/learner_state.py` export/manifest work and `ExecutionProfile` wiring can proceed alongside Work Package B against the frozen contracts in Section 3.2. Work Package E can begin profiling immediately after the first working SFT cycle in C.
- **Estimated effort:** $1\text{--}2$ engineer-weeks for A–D (shared core, SFT/recovery, RL integration & publication), plus a focused tuning interval for E–F. Highest schedule uncertainties are hybrid conv/attn kernel parity under `torch.compile` and `sglang_actor.py` native-fallback binding.

---

## 10. Verification Suite, Telemetry & Risk Register

### 10.1 Automated Correctness Gates

| Failure Mode | Decisive Automated Check |
|---|---|
| Adapter bypass in fast kernels | Nonzero $(A, B)$ changes output for each of the 9 target families; $B=0$ reproduces base exactly. Verify $\nabla_B \neq 0$ on step $0$ and $\nabla_A \neq 0$ on step $1$. |
| Frozen-base / head regression | Assert $0$ frozen tensors in optimizer param groups, $0$ parameter drift after step, and nonzero $\nabla$ on layer-$0$ adapters. |
| Prefix-tree / packing corruption | Whole vs. packed vs. multi-child shared-prefix loss and FP32 factor gradients match at equal global denominator, with $s \neq 1$, separate child backwards, and conv/attn doc boundaries. |
| Severed Neuralese dependency | Consumer loss produces nonzero gradients on producer-document adapters in pass-through mode; detached mode produces exact zero. |
| Reference view / backward aliasing | Interleaved live $\to$ ref $\to$ live calls restore cleanly; pending backward uses its original factor tuple; equal-policy logprobs agree within dtype-calibrated tolerance. |
| Objective drift | Mixed student/teacher/replay and decision fixtures match nucleus/full-vocab logprobs, tail masks (excluded from ref penalty), ratio caps, and global normalization. |
| Merge / scaling mismatch | Round-trip $\alpha/r$ once; verify tied rows and native interface. Compare (1) factored learner vs. merged actor and (2) merged reference vs. merged actor on full/nucleus logprobs and decision options. |
| Split-brain actor identity | Generation, scoring, capture, and **native fallback** all attest identical `effective_policy_sha256` and sentinel logprob hash after refit. |
| Broken interruption recovery | Interrupt at optimizer boundary; restored run reproduces exact next batch, LR, Ambient choice, factor/optimizer state, and RL consumed-rollout ledger. |

Set numerical tolerances from dtype and repeated-reference variance before judging candidates. Never apply loose BF16 forward tolerances to RL importance-ratio checks.

### 10.2 Required Performance & Quality Telemetry

Log every run's manifest alongside:
- **Phase wall times:** data prep, reference forward, policy forward/backward, optimizer step, checkpoint save, merge/publish/attest, rollout collection, held eval.
- **Work counters (named separately):** canonical chain tokens, prefix-deduplicated computed tokens, supervised tokens, weighted supervision mass.
- **Memory & compiler health:** peak CUDA allocated/reserved (GiB), host RAM (GiB), actor KV (GiB), retained vs. recomputed layer-tokens, compile time, recompile count, OOM retry count, input stall time.
- **Cost & learning health:** observed CU/h, cold-start-to-complete CU, held task scores, reasoning/termination rates, per-family grad/update norms, policy-vs-ref KL, train-vs-actor logprob mismatch, and rollout staleness.

### 10.3 Top-5 High-Blast-Radius Risk Register

| Risk | Likelihood / Impact | Concrete Mitigation |
|---|---|---|
| 1. Prefix-tree multi-backward freed-graph error | High / High | Enforce leaf-only `LinearFactorLeaf` bundles; forbid caching scaled/cast non-leaf tensors; test multi-child backward in Work Package B. |
| 2. `sglang_actor.py` native fallback serves stale start weights | High / Critical | Treat multi-route attestation (including native fallback) as a hard release blocker in Work Package D before any live RL run. |
| 3. Silent objective drift from microbatch averaging or nucleus/full-vocab swap | Medium / High | Enforce global denominator accumulation; pin `rl_objective_version`; test against independent eager FP32 oracle. |
| 4. SGLang LoRA packed-rank overflow ($3r=96$, $2r=64$) | High (if M2 enabled) / High | Derive actor rank capacity automatically from `tune/targets.py` packed mapping; block actor launch if capacity $< r_{\text{packed}}$. |
| 5. `torch.compile` recompile storms on variable-length packed batches | Medium / Medium | Fixed adapter topology, bucketed token lengths, selective dynamic dims, and mandatory recompile-count telemetry in Work Package E. |

---

## 11. Decision Triggers (Revisit Only on Evidence)

- **Held quality lags full tuning:** Audit supervision exposure, masks, update norms, and LR first; only then test $r=64$, selective trainable rows, or fallback to `parameterization="full"`.
- **Reference forward dominates wall time:** Propose a separately versioned Ambient/KL objective experiment acknowledging the learning tradeoff.
- **Merge/refit dominates RL cycle:** Qualify SGLang adapter slots (Milestone 2) with packed-rank derivation and full route attestation.
- **Rollouts dominate RL cycle:** Optimize actor scheduling, task throughput, or remote collection placement before micro-optimizing learner GEMMs.
- **G4 loses to L4 on CU efficiency:** Retain G4 for turnaround-sensitive runs and run long budget-sensitive jobs on L4 using the same learner core and checkpoint format.

---

## Sources & Audit References

All repository citations refer to `werg/natlang` at `bfc5a7a7d2020823c6df4ad7d78b9e85093d6cfe`. External references were consulted on 22 November 2026 as design guidance, not proof of pinned-fork compatibility.

- **`[R0]`** `AGENTS.md` — project conventions, precision, provenance, evaluation, and engineering rules.
- **`[R1]`** `scripts/neuralese_data/exp0_sft.py` — `load_model`, `step_backward`, `ambient_teacher_rows`, `write_checkpoint`, `train`.
- **`[R2]`** `scripts/neuralese_data/lfm3_fast_train.py` — `_layer_call`, normal/past kernels, `run_prefix_tree`, activation planner, chunked readout.
- **`[R3]`** `scripts/neuralese_data/grpo.py` — RL objective, `ReferenceWeights`, `WeightSync`, state save/restore, policy/reference passes.
- **`[R4]`–`[R8]`** `model/native_adapters.py`, `tune/parameterization.py`, `tune/targets.py`, `tune/export.py`, `export/native_hf.py`.
- **`[R9]`–`[R12]`** `scripts/neuralese_data/expert_iteration.py`, `serve/sglang_actor.py`, `scripts/neuralese_data/behaviour_probe.py`, `docs/ENGINE_PARITY.md`.
- **`[R13]`–`[R14]`** `plans/neuralese/LFM3_THROUGHPUT_20261028.md`, `plans/neuralese/GRPO_STEP_THROUGHPUT_20261117.md` (historical full-SFT context only).
- **`[E1]`** NVIDIA RTX PRO 6000 Blackwell Server Edition specification ($96\text{ GB}$).
- **`[E2]`** LiquidAI LFM3 SFT cookbook notebook (`b9abf690d969d9ec5a436c69dcd6e047b203780d`).
- **`[E3]` / `[E11]`** Hugging Face PEFT precision troubleshooting & LoRA reference.
- **`[E4]` / `[E5]` / `[E6]`** PyTorch activation checkpointing (`use_reentrant=False`), PEFT `torch.compile`, and PyTorch dynamic shapes guides.
- **`[E7]`** TRL GRPO documentation.
- **`[E8]`** Thinking Machines Lab: *LoRA Without Regret*.
- **`[E9]` / `[E10]`** SGLang for RL systems & SGLang LoRA serving documentation.