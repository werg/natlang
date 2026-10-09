# Mellum 2.1 as a Maple replacement — evaluation and port plan (2026-10-08)

Owner request: evaluate JetBrains Mellum2.1-12B-A2.5B-Thinking (Apache-2.0). Switch from Maple if it trains faster
and does acceptably at raw agentic coding in our harness out of the box. If adopted: relatively aggressive QAT.

## The model

- 28 layers, width 2304, GQA 32/4, head_dim 128, q/k RMSNorm.
- 64 experts, top-8, renormalized; expert FFN 896; no shared expert; untied head; vocab 98,304.
- 3 sliding (window 1024) : 1 global, the same pattern as Maple.
  - RoPE θ=500k on all dimensions; YaRN (factor 16, original 8192) on global layers only.
- Released in BF16. 12B total, 2.5B active per token. Maple: 20B total, about 1B active, published ternary.
- Self-reported scores: SWE-bench Verified 47.0, Terminal-Bench 2.1 17.4, BFCL v4 62.3, LiveCodeBench v6 82.0.
- Mellum 2 (June 2026) also ships Base/Instruct/Thinking, intermediate pre-training checkpoints and a technical report
  (arXiv 2605.31268). No training code.

## Existing support (reusable as is)

| Stack | Status |
|---|---|
| vLLM (`vllm-node` image) | `MellumForCausalLM`, per-layer sliding window and per-layer RoPE; `--reasoning-parser qwen3 --tool-call-parser hermes` |
| transformers 5.19 (training container) | native `models/mellum`; exact reference; expert loop is Python unless `experts_implementation=grouped_mm` |
| Our llama.cpp fork | `LLM_ARCH_MELLUM` (MoE, SWA pattern 4, plain RoPE on sliding / YaRN on global), `mellum2` pre-tokenizer, HF→GGUF converter `conversion/mellum.py` |
| JetBrains GGUFs | BF16, Q8_0, Q6_K, Q4_K_M, MXFP4_MOE |

## Done

- 9d55780b — the Maple family implementation loads Mellum exactly:
  - `MapleConfig` gains `model_type`, `swiglu_clamp`, `rope_parameters` and `dense_experts`.
  - Per-layer-type RoPE uses transformers' own `MellumRotaryEmbedding`.
  - `DenseExperts` keeps exact BF16 experts; the SwiGLU clamp is optional.
  - Exact logit parity with transformers on a tiny random Mellum (`tests/neuralese/test_mellum_port.py`).
  - So the Maple port runner, trainer, fused-kernel path and QAT code apply to Mellum.
- `training/neuralese/scripts/bench_hf_training_step.py`: native-transformers forward / forward+backward timing on a
  real token window.
- Evaluation harness prepared: `runs/mellum-evaluation-20261008/` (vLLM serve script, runner copy).
  - Same 23-task protected packet, limits and greedy decoding as the Maple/Ling comparison
    (`plans/student-candidate-comparison-20261005.md`).

- Real-checkpoint parity (`scripts/mellum_parity.py`; weights verified against the HF LFS hashes,
  `runs/mellum-evaluation-20261008/weights-verified.json`).
  - In FP32, four layers: the port matches transformers to 1e-5 relative residual error.
  - In BF16, the full model differs by ~0.4% per layer. Our router runs in FP32; the reference's runs in BF16, so
    near-tie expert choices differ.
  - On 2048 held tokens: CE 4.49 (transformers) vs 4.54 (ours), argmax agreement 0.88.
  - Truncated loads (`layers=`) were fixed for the per-type rotary.
- Neuralese markers: `family_controls` picks `<|extra_token_7|>`/`<|extra_token_8|>` (IDs 33/34) for Mellum and
  keeps 151669/151670 for Maple. Used in `serve.load_engine` and `foundation_heads`.
- Harness evaluation set up: `runs/mellum-evaluation-20261008/evaluation-plan-{thinking,no-thinking}-v1.json`.
  - Same packet, runtime and limits as Maple/Ling.
  - The pinned source review was recovered from git (02ff4cc2) because the live file has moved on.

- Harness result, thinking on (`runs/mellum-evaluation-20261008/execution-thinking-v1/report.json`): **13/23**
  complete_success (Maple 12/23, Ling 11/23 on the same packet and limits).
  - Other outcomes: 2 semantic failures, 3 contract failures, 5 incomplete, 1 policy-held.
  - Semantic accuracy 0.87 on completed cases.
  - Median 84 s per case on GB10 vLLM BF16, eager.

## Next, in order

1. (done) Real-checkpoint parity.
2. Harness evaluation: vLLM BF16, thinking on (as Ling), then thinking off.
   - Baselines: Maple 12/23, Ling 11/23 / 10/23.
3. Training speed:
   - Our trainer's warm-up step on the same windows as Maple v10. Maple v10: ~11 s per two-pass update over ~5.5k
     tokens with QAT.
   - Native-transformers bench as a cross-check.
4. Neuralese markers: Mellum has no unused vocabulary rows; use the spare added tokens `<|extra_token_7|>` and
   `<|extra_token_8|>` (IDs from the tokenizer). `serve.load_engine` and `foundation_heads` currently hard-code
   Qwen/Maple marker IDs (151669/151670) and need a per-tokenizer choice.
5. GGUF: convert with the fork's converter, then check perplexity/logit parity vs our port (Maple's M0.3 equivalent).

## Nested family (owner 2026-10-08: same goal as Maple)

Mellum would carry the same nested family as Maple (`plans/neuralese/MAPLE_NESTED.md`): one model holding the full
model and members `LxE` (first L layers, first E experts per layer in an N0 usage order; depth members exit early
through the final norm and shared head), trained jointly with every member as a real objective (§4a). The code is
`maple/nested_train.py` plus the member machinery in `maple/model.py` (`set_member`, private router rows and norm
gains, expert ordering), which now loads Mellum.

Mellum-specific:
- Only 64 experts instead of 256, so prefixes are coarser. Candidates are 28x16, 28x24 and 28x32, plus a depth member
  such as 14x16. N0 decides from routing coverage on our corpus.
- Bootstrap and joint phases run inside QAT. Maple's shared QAT was scale-only; Mellum's is full latent ternary (below),
  so every member's gradient flows into the latent expert weights.
- `TernaryExperts.learn_scales` and expert permutation exist; `DenseExperts` already permutes. The QAT expert module
  must keep the member hooks.

Gap shared with Maple (MAPLE_NESTED §9 item 5, still open): the Neuralese trainers (`train/text_warmup.py`,
`train/trajectories.py`) run on the frozen nested state but train only the full model, and do not evaluate members.
Their QAT updates move shared weights the members depend on. Member objectives and per-member evaluation have to
enter the Neuralese stages before either backbone's long runs.

## QAT design (if adopted)

- Maple's experts were published ternary, so its expert QAT is scale-only (fixed codes, learned 256-column block
  scales).
- Mellum is BF16, so aggressive QAT needs latent weights whose ternary codes are re-derived every step (STE).
  - Experts: ternary codes from latent BF16 weights with absmean scales per 128-column block.
    - The fused kernel's 256 blocks do not divide 896; 128 does (896 = 7×128, 2304 = 18×128).
    - The kernel takes the block size as a parameter.
  - Attention: the Maple attention QAT path (ternarize plus LoRA) applies unchanged.
  - Memory: about 24 GB BF16 latents + 24 GB gradients + Muon momentum 24 GB, plus activations. This fits the GB10
    with checkpointing.
    - Alternatives if tight: QAT layer groups in turn, or train LoRA on frozen ternary codes after a short full-latent
      phase.
- Distill from the BF16 original (KL on logits) during QAT. Same recipe, data and Neuralese stages as Maple; only
  backbone-inherent differences are declared.

## 2026-10-09 — CE smell test and first QAT conversion trials

- Teacher dump (v7 text, 512+32 windows of 2048, top-64): CE 3.93 train / 4.19 held. Not a port bug:
  transformers BF16 4.49, FP32 4.58 and our port 4.54 on the same corpus window, while the model README scores 1.00
  and a Python file 1.89 in all three. As with base Maple (CE 5.07, DECISIONS.md 2026-10-08), ~86% of corpus tokens
  are the shared system prompt, which chat models do not predict; 15.6% of reference tokens fall outside the
  teacher's top-64 (20-30 nat outliers), median top-1 probability 0.64.
- Naive full ternarization (λ=1): held CE 11.35, KL 9.9. λ=0 reproduces the teacher exactly.
- Trials at Lion lr 3e-3·α and 3e-4·α (α = per-latent ternary scale) both lost the teacher within a few updates at
  λ≈0 (KL 0.02 → ~1): one lr 3e-4·α stochastic-rounding step alone raises held KL to 0.97. Under diagnosis.
- Speed: fused compiled ramp (24x eager) brings the forward to 1.3 s per 2048 tokens; backward is 63 s (next target).
- Implication: conversion recovery data should be text the teacher models well (assistant turns, code, prose),
  not system-prompt-dominated windows.
- Update-isolation probes (one gradient on one train window, held KL at λ=0, base 0.021): writing identical values
  0.021; round-to-nearest lr 3e-4·α 0.48; stochastic rounding at lr 3e-5·α 0.43. Damage does not scale with step
  size and is diffuse: per group, all attention 0.20, all experts 0.30, gate_up 0.21, o_proj 0.11, every layer
  quarter 0.04–0.11. Some gate_up rows (layers 0, 2, 3, 27) have near-zero RMS, so a per-tensor α step changes
  them by up to 3.6× their RMS. Next: per-row step scale, recovery windows from assistant/code/prose text instead
  of the system prompt, and a check of KL sensitivity on README/Python text.
- Resolution (2026-10-09 evening): the "damage" was CE learning, not broken weights — one update on
  prompt-dominated windows memorises the shared system prompt (train CE 4.5 → 1.96), which the teacher does not
  predict, so held KL rises. Writing identical values leaves KL at 0.021. Fixes (fabc8eda): recovery windows drop
  each chat's system message and pack repo TS/Python files (teacher CE 1.63 train / 1.76 held, was 3.93 / 4.19);
  Lion steps in units of each row's ternary scale. Backward fix (592c1819): dense experts ramp each tensor once
  per layer and unbind, instead of per-expert indexing that zero-filled a full-size gradient 64x per tensor.
- v2 trial (runs/mellum-qat-convert-20261009-v2.sh): teacher-v7-nosys-code, lr 3e-4·α_row, ramp 1000, constant KL,
  CE 0.25, 2000 steps; queued for admission (queue script retries every 5 min).

## 2026-10-09 — Pipeline handoff, recipes and N0 (Mellum-pipeline sub-agent)

- Handoff (97667c90, ordered export 2026-10-09): `python -m natlang_neuralese.maple.qat_export --model BF16_DIR
  --checkpoint CONVERT/checkpoint.pt --out EXPORT [--order runs/mellum-nested-20261009/n0-v1/expert-order.pt]` writes
  the conversion's λ=1 ternary values in the source HF layout (config flag `natlang_deployed_ternary`); `load_maple`
  loads it as ternary codes, so Maple's QAT policy, foundation heads, N0 and the fused ternary MoE (trainable block
  scales, unclamped SwiGLU) apply unchanged. Tested: identical λ=1 logits on the tiny Mellum; ordered export equals
  ordering after load.
- Fix: `load_student`, `routing.py` and `nested_train.py` defaulted to Maple's converted-weights cache for any model
  path, so Mellum heads/engines/N0 would have silently loaded Maple. The cache now applies to published Maple only.
- Recipes (5dfff36b): `raw-recurrence-v2` (v1 + read adapter + projection-anchor decay 256 for every line; the
  recurrence handler whitelist now accepts both), `raw-recurrence-mellum-v1` (backbone-inherent differences only:
  cutoff 27 / 512 contexts, ternary QAT with qat_latent_lr 0.003, member terms with system masking and full-model
  anchor), `foundation-mellum-v1` (cutoff 27). Rationale in recipes/HISTORY.md.
- N0 on BF16 Mellum (runs/mellum-nested-20261009/n0-v1/expert-order.pt; 327 rendered rows of
  maple-joint-20261005/qwen3-render-v1, 4096 tokens): routing mass covered by the first k ordered experts per layer,
  mean (worst layer): 8: .50 (.35), 16: .71 (.56), 24: .83 (.70), 32: .91 (.81), 48: .98 (.94); no unused experts.
  Mellum routes broadly. Proposed members: 28x32 (half width, 91% of mass), 28x24, 28x16 (aggressive), plus a depth
  member (14x16 or 21x32) pending N1. N1 (held NLL/KL/agreement per prefix and per LxE member, `routing.py
  --members`) was stopped to keep the GPU for the conversion; rerun after it with
  `--members 28x16,28x24,28x32,21x32,14x16,14x32`.
- Launch sequence once the conversion qualifies: qat_export --order → maple/foundation_heads --model EXPORT
  --cutoff 27 → recipe raw-recurrence-mellum-v1 (or foundation-mellum-v1 first) via natlang_neuralese.train.recipe.
  Not yet done: a GPU smoke of the warm-up path on an export, and the step profile (#46).
- v2 trial results (held: 32 windows of system-free chat turns + repo code; BF16 teacher CE 1.756). Deployed = all
  attention and experts ternary (λ=1); current = the ramp's λ:

  | step | λ | deployed CE | current-λ CE | deployed KL to BF16 |
  |---|---|---|---|---|
  | 0 | 0.0 | 10.74 | 1.756 | 9.86 |
  | 300 | 0.3 | 3.45 | 1.426 | — |
  | 500 | 0.5 | 2.70 | 1.487 | — |
  | 700 | 0.7 | 1.89 | 1.468 | 1.07 |
  | 800 | 0.8 | 1.74 | 1.497 | 0.95 |
  | 900 | 0.9 | 1.64 | 1.546 | 0.88 |

  The fully ternary model beats the BF16 original on this held set from step 800. The KL stays high: it is learning
  our text, not copying the original. Not yet evidence of general/agentic quality — the harness eval (24-case packet)
  on the exported λ=1 model decides qualification. Steps 1000–2000 continue at λ=1.

## 2026-10-10 — Qualification of the ternary conversion (criterion fixed before the run)

Pipeline: `runs/mellum-qualify-20261010/pipeline.sh` (armed while conversion v2 runs). Stages: merge the qat_convert
checkpoint-policy branch → `qat_export --order` (N0) to `/home/werg/data/models/mellum21-ternary-convert-v2`, held-CE
check against the trainer's final deployed CE (≤ 0.01), packed copy → the protected 24-case harness packet on vLLM,
served exactly as the BF16 eval (same packet, runtime, runner, template, 1024-token budget, greedy, thinking on/off) →
artifact registration with the verdict → N1 with members 28x16,28x24,28x32,21x32,14x16,14x32 → foundation heads
(cutoff 27), step profile, recipe smoke through core_text_warmup.

Qualification criterion (paired against BF16 Mellum, runs/mellum-evaluation-20261008), all must hold:
- thinking on: complete successes ≥ BF16 − 2 (BF16 13/23), semantic accuracy ≥ BF16 − 0.10 (BF16 0.867), paired
  regressions (BF16 success, ternary not) minus paired gains ≤ 2;
- thinking off: complete successes ≥ BF16 − 2 (BF16 9/23);
- either mode: contract failures ≤ BF16 + 2 (protocol regression guard).
The 23-case packet resolves only large differences (±2 cases is within run-to-run noise at temperature 0 across
stacks); passing it means "not detectably worse", not "equal".
