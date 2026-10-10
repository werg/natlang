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
- Early stop (coordinator, 2026-10-10 ~02:20): conversion v2 was stopped at step ~1248 of 2000 on a λ=1 plateau —
  deployed held CE 1.555 @1000, 1.551 @1100, 1.551 @1200 — while train CE fell to ~0.3 (memorising the 512 train
  windows, ~2.4 epochs). Final state = `convert-v2/checkpoint.pt` at step 1200 (λ=1, deployed held CE 1.551, KL 0.80
  to the BF16 original; teacher CE 1.756). The qualification pipeline exports step 1200 and checks the export against
  1.551. A longer conversion needs more (or fresh) recovery windows, not more steps on these 512.

## 2026-10-10 — Conversion v2 FAILED qualification (degenerate generation)

- Pipeline (runs/mellum-qualify-20261010): export of step 1200 in N0 order (`/home/werg/data/models/mellum21-ternary-
  convert-v2`, 5488 matrices); export held CE matched the trainer's 1.551 within 0.01; packed copy 3.6 GB.
- Harness (protected 24-case packet, vLLM, served exactly as BF16): **0/23 complete in both modes** (BF16 13/23
  thinking, 9/23 no thinking); all 23 cases `incomplete_task` (median 40 s thinking, 54 s no thinking). Criterion
  fails on every check.
- Coordinator diagnosis: the export is consistent (plain transformers held CE 1.512 vs BF16 1.902 on 8 windows; no
  router/expert mismatch), but greedy generation is degenerate. "What is 17+25" loops "Preserve the original question
  and answers." with no `<think>` and no `<|im_end|>`; a tool prompt loops nested JSON. BF16 answers `<think>…</think>
  42<|im_end|>` and emits a correct `<tool_call>`.
- Causes: recovery windows were Maple-rendered corpus text with system messages stripped, so chat-structure positions
  (assistant start/`<think>`, tool-call format, end of turn) were barely or wrongly trained; CE weight 0.25 pulled
  toward that text; 512 windows memorised; the gate was teacher-forced CE, not generation. Held KL 0.80 to the BF16
  original was the warning sign: a low CE with a high KL means the model moved, not that it recovered.
- Artifact `mellum21-ternary-convert-v2-20261010` registered with qualification `failed` (files kept).
- N1 on the v2 export (no longer the target weights, but the structure carries over; held NLL / KL from full / top-1):
  28x32 1.26 / 0.18 / 0.87, 28x24 1.48 / 0.42 / 0.80, 28x16 2.38 / 1.36 / 0.60, 21x32 7.00 / 5.82 / 0.23,
  14x16 13.7 / 12.7 / 0.00, 14x32 13.0 / 12.0 / 0.00 (full model 1.05). Untrained depth members are unusable (early
  exit without training); width members 28x32/28x24 are viable starting points. Rerun on the v3 export.
- Step profile on the v2 export, 4 layers × 8192 tokens: 2.22 s/step, 15.6 GB peak (TernaryExperts fused path).
- Next: conversion v3 — recovery data = BF16 Mellum's own generations in Mellum's chat template (system prompts
  kept), KL-only distillation with prompt positions down-weighted, and a generation gate during training.

## 2026-10-10 — Conversion v3: data, teacher, split check

- Recovery data `mellum21-self-distill-v3-20261010` (registered, manifest e17b56d6): 6742 prompts → 6536 BF16
  responses via vLLM (T 0.6, top-p 0.95; max 1024 tokens, 2048 thinking), 15.0M tokens, 3213 with thinking on. By
  source: natlang native records 2395 (Luna v17 v9 assembled-v3, train split, tools and system prompts kept), tool
  calls 900, code 841, math 700, general 700, user turns 600, SQL 400. 206 prompts over 7168 tokens omitted (205
  natlang, 1 code; receipt). 681/841 code responses hit the length cap; kept as BF16 behaviour.
- Teacher `/home/werg/data/mellum-qat/teacher-v3`: top-64 over whole sequences, 6472 train / 64 held records.
- Split check before training (runs/mellum-convert-v3-20261010/split-check.json, after harness-bench's S1 re-split
  749cf77c): no distill-v3 prompt comes from harness-bench. All 3200 natlang/user-turn prompts are train in the
  source snapshot and in the newer Luna v9 view (2026-10-10). cross_corpus.place over their source groups and ids
  against the S1 index and the harness-bench v4 index places none; the namespaces are disjoint (S1: task,
  agenttrove-task, hotpotqa-q, …; Luna: inline-curriculum, authored-bounded-decisions-v1, luna-v2x…), so that
  check is vacuous rather than proof. The 24-case packet (execution-eval-v3) shares no source group with the
  used records, none of its 112 group/source ids occurs in them or in any prompt, and none of its 214 long text
  leaves occurs in any prompt. Gate probes' natlang cases come from the test split and are never trained.

## 2026-10-10 — QAT moves into the stages (owner); the conversion no longer gates the foundation

Owner decision (plans/neuralese/DECISIONS.md, 2026-10-10 "QAT inside every training stage"): the Mellum line starts
the Neuralese foundation from **BF16 Mellum**, not from a qualified ternary conversion. Conversion v3 and its
qualification pipeline (runs/mellum-qualify-v3-20261010) run out unchanged; their results are a data point for how
far a short conversion gets, not a precondition.

New lineage:
1. Heads: `python -m natlang_neuralese.maple.foundation_heads --model <bf16 Mellum> --precision bf16 --cutoff 27
   --stop-source final --out heads.pt` (the backbone identity records `precision: bf16`; `serve.load_engine` loads
   dense experts and unternarized attention).
2. Recipe `raw-recurrence-mellum-v5` (foundation stages = `token-preserving-foundation-mellum-v2`): the `latent`
   policy and the shared quantization component of `raw-recurrence-v6`: `q4` (Q4_0 everywhere, required, ramp
   0.15–0.3, i.e. the late warm-up), `ternary-experts` (experts ternary by Maple's rule, attention int4; required,
   ramp 0.3–0.7 through the AR fixup into the recurrence), `ternary` (everything ternary; reported, ramp 0.55–0.95).
   Sampled multi-precision objective, BF16 the teacher, backbone rate 7.5e-6.
3. Conversion v3 reused: init from its best latents when they pass the init gate at λ = 0 (else BF16), its teacher
   shards as the behaviour-preservation KL stream (weight 0.5, prompt 0.25, 4,096 tokens), its generation gate and
   held KL as every precision's gate column (pass: ≥ 0.65 generation pass rate against BF16's 0.75, held KL ≤ 0.25;
   init gate: ≤ 0.05 drop, held KL ≤ 0.05). These thresholds are proposals, not yet measured on a QAT stage.
4. Launch script: runs/mellum-foundation-qat-20261010/launch.sh (prepared, not launched): registers the v3 latents,
   runs the init gate, builds heads with the decision, then the recipe through `adapted_runtime`. Waits for memory: v3
   holds 85 GB until ~15:05 and its qualification pipeline runs after it.

Open: nested-family members have no BF16 state yet (member terms are off in v5); per-latent learning rates in units
of the ternary scale for the `latent` policy (measure on the gate columns first); deploy exports at a chosen
precision (Q4_0 GGUF; TQ2_0 via qat_export) from latents.

## 2026-10-10 — Conversion v3 stopped by decision (QAT moved into the training stages)

- Run `/home/werg/data/mellum-qat/convert-v3` (runs/mellum-convert-v3-20261010/train.sh): KL only to BF16 on
  mellum21-self-distill-v3-20261010 (prompt positions 0.25), ramp 0→1 over 1200 updates, Lion 3e-4 per-row scale,
  generation gate (24 probes) every 100 updates at λ=1. Step 0: BF16 (mix 0) gate 0.75 (natlang probes partly run
  past the 256/512-token limit), held response KL 0.0002; untrained ternary (λ=1) gate 0/24, held KL 10.0.
- λ=1 trend: held KL 10.0 → 6.2 → 4.5 → 5.3 → 4.5 → 4.0 → **2.5** (steps 0–600), greedy-token agreement 0.03 → 0.53,
  gate 0/24 throughout; at mix 0.5 the thinking check (`<think>` opens and closes) reached 0.79, end/tool/exact 0.
  KL at the current mix held at ~0.95–1.0 from step 300 to 600 while the mix doubled (latents kept pace with the ramp).
  Peak 69 GB with the gate's deployed-weights cache; ~21 min per 100 updates including evaluations.
- 11:28: the memory guard stopped the run (free memory below the floor, caused by another probe). The SIGTERM
  checkpoint (47 GB) cannot be written before docker's kill (exit 137, partial `checkpoint.pending` removed);
  the rolling slot (step 500) was intact. Resumed from step 500 with a 20-minute slot cadence.
- **Stopped at step ~520 (last logged 516) by coordinator decision**: under the owner's new plan QAT runs inside the
  training stages (1324fd2b, recipe raw-recurrence-mellum-v5). The best latents (`best-weights.pt`, step 600,
  metric = (1 − gate) + held KL = 3.52) are handed to the foundation init gate
  (runs/mellum-foundation-qat-20261010/launch.sh), which registers them, gates them at λ=0 against BF16 and continues
  the ramp in the recipe. The v3 export/harness pipeline (runs/mellum-qualify-v3-20261010) was never run.
- Lesson for long-running QAT jobs: a signal checkpoint of the full latent+optimizer state does not fit docker's stop
  window; rely on a short rolling cadence (or a weights-only signal snapshot).
