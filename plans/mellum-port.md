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

## 2026-10-10 — Foundation embedding_distillation gate failure (cutoff 27): a recipe error, not a Mellum limit

`runs/mellum-foundation-qat-20261010` (raw-recurrence-mellum-v5, BF16 start) passed token_identity, trained
embedding_distillation for 8,192 steps, and failed the gate (agreement ≥ 0.9 and KL ≤ 0.25 per stratum). The
failed attempt is preserved in `run/embedding_distillation*`.

| stratum (held) | positions | KL | agreement | gate |
|---|---|---|---|---|
| source | 4,833 | 0.105 | 0.857 (plateau 0.852–0.857 for the last ~1,000 steps) | fail |
| context | 524,800 | 0.072 | 0.915 | pass |

**Cause.** The metric (causal_bootstrap.metrics) compares the argmax of a projection of the layer-`cutoff` state
with the argmax of the full 28-layer model at the same position. At cutoff 27 the projection has to emulate the last
layer, including its attention over the prefix, from one position's state, so its ceiling is below 1. Owner
decision 2026-10-06 (DECISIONS.md "out port at the top") makes the foundation reference `cutoff: "full"` on every
backbone; the shared base `foundation-v1` declares `cutoff: "full"` with `stop_on_gate`. The Mellum recipes
(raw-recurrence-mellum-v1…v5, HISTORY.md "cutoff 27 = layers - 1") re-introduced the Maple c12–c23 shallow cutoff as a
"backbone-inherent" override. That lesson came from lineages the 2026-10-06 decision had already declared to be
"measuring the wrong thing", so the override is not backbone-inherent and breaks the unify rule.

**Evidence** (ledgered, 2026-10-10, same heads/records/pieces/512 contexts; `runs/mellum-foundation-qat-20261010/diagnostics/`):
- Cutoff `full` with `--stop-on-gate`: `qualified_at_initialization`, KL 0, agreement 1.0 on all 529,633 held positions
  (source 4,833, context 524,800), bit-exact, 0 optimizer updates (`full-depth-qualification-eval.jsonl`). Pop's LFM
  qualified the same way (132,883 positions).
- Gate diagnostic of the cutoff-27 checkpoint (`gate-diagnostic-c27.json`, maple/gate_diagnostic.py): the projection's
  token is in the teacher's top 5 at 99.3% (source) / 99.7% (context). Disagreement sits where the teacher is
  uncertain: source has 30.6% of positions with top-1 probability < 0.3 (agreement 0.66) and 19.3% at 0.3–0.5 (0.84);
  context has 13% / 14%. Above 0.7 agreement is ≥ 0.99. Source (program text) is higher-entropy, so it sits lower
  under any shallow cutoff; 3.7% of source positions are exact bf16 ties.
- Shallow cutoffs failed this gate on every backbone: Maple c12–c23 best 0.856 overall / source 0.79; LFM cutoff 14
  62% (82% with contexts), LFM cutoff 15 >92% aggregate but ~88% source (HANDOVER 2026-10-05). Mellum c27's 0.914 /
  0.857 is the best shallow result so far. The 0.9 bound is not LFM-calibrated: no backbone passes it at a shallow
  cutoff, and every backbone passes it exactly at full depth.

**Fix (next launch).** Drop the cutoff override; inherit the shared foundation unchanged. Recipes are immutable, so add
`raw-recurrence-mellum-v6` = `raw-recurrence-mellum-v5` minus `overrides.stages.embedding_distillation.parameters.cutoff`
(27) and minus `...contexts` (512). Full depth qualifies at initialization, so the windows only change the held sample
size; keep 512 only if the extra held positions are wanted, declared as such. Heads: `foundation_heads --cutoff` must
match (full depth). Relaunch the lineage from token_identity with v6 (the heads/init-gate receipts are reusable if the
heads' cutoff is not baked in; otherwise rebuild them). No threshold change: a shallow-cutoff projection is a separate,
optional efficiency variant (sketch/MTP initializer), gated on its own and not part of the foundation.

## 2026-10-10 — Warm-up step profile (core_text_warmup backbone phase, run-v9)

Question: where does a whole_transformer_adaptation update of `runs/mellum-foundation-qat-20261010/run-v9`
(raw-recurrence-mellum-v10, text_warmup.py, LionSR in backward) spend its time, and why does the ledger see ~40 GB
more than CUDA-allocated memory. Sources: `core_text_warmup/train.jsonl` (steps 2561–2606), py-spy (150 s,
3,660 samples) of the live trainer, `/proc` and cgroup memory of the container, and two ledgered one-layer probes
(`training/neuralese/scripts/bench_mellum_qat_layer.py`, `bench_mellum_qat_kernels.py`; synthetic Mellum expert
shapes, ≤ 12 GB). The probes shared the GPU with the live job (~2.3× slower than alone); ratios are reliable.

### Step time

The 42.9 s of step 2561 is not typical. It was the first backbone update (one-time `torch.compile` of
`_fused_lion_rows` per latent shape). Over the next 45 updates, time depends on the sampled precision point:

| point | updates | mean | median | range |
|---|---|---|---|---|
| q4 (mix ≈ 0.47) | 14 | 34.1 s | 33.4 s | 28–52 s |
| bf16 | 31 | 9.7 s | 8.8 s | 5–15 s |
| all | 45 | 17.3 s | | |

At the current mix, the ~1,535 remaining warm-up updates take about 7.4 h, not 18 h. The step gets slower later in
the stage: ternary-experts (weight 1.0) starts at curriculum progress 0.3 (step ~3,510), and sequence passes ramp
from 1 to 3.

**Cause: eager fake-quant of the experts, recomputed at every forward.** `DenseExperts._value` →
`precision_value` → `q4_0` + `_PrecisionRamp` runs eagerly in FP32 over all 64 experts of a layer
(396 M elements). It makes about 15 full-size FP32 passes over memory, uses 3.9 GB of temporaries, and takes
452 ms per layer in the probe. A bf16 copy of the same weights takes 21 ms. `shared_parametrized_weights` caches
only the parametrized attention weights. The experts (`precision_group`) are re-quantized at each of the 4
forwards of a q4 update: main stream, its checkpoint recompute, preserve stream, and its recompute. The q4−bf16
gap is 24.4 s = 4 passes × 28 layers × ~0.22 s. That is ~72% of a q4 update and ~45% of the mean update. With
3 sequence passes it becomes 8 passes, ~48 s per quantized update. py-spy agrees: 58% of samples wait at
`maple/model.py:441` (`bincount(...).tolist()`, the first host sync after the queued quantization kernels). GPU
utilisation is 96%: the update is GPU-bound and memory-bandwidth-bound, not host-bound.

**bf16 update (~9.7 s)**, from probe ratios scaled to the live run:
- per-expert MoE loop forward + recompute + backward (main ~1k tokens, preserve ~3k tokens): ~5 s;
- FP64 gradient norm in the LionSR hook (`vector_norm(dtype=float64)`; FP64 is slow on GB10): ~1.2 s;
- LionSR step: ~1 s;
- attention, readout/KL chunks, Python and the rest: ~2.5 s.

Not significant now:
- eval: ~52 s per 256 updates, < 1% in this phase (the logged 14% share comes from the fast projection phase);
- host data/tokenisation: < 1% of samples;
- unified-memory paging: no swapping of trainer pages, offload 0 bytes.

### Memory

| | GB |
|---|---|
| CUDA allocated at update start | 48.9 (BF16 weights ~24.3; LionSR BF16 momentum ~23.4; heads + Muon/AdamW ~1.2) |
| CUDA peak allocated / reserved | 55.5 / 57.2 (+6.6 GB per update; ~3.9 GB of it is the eager fake-quant temporaries of one layer) |
| nvidia-smi (what the ledger counts for CUDA) | 55.8 ≈ reserved + context; reserved − allocated ≈ 7.8 GB at update start (cached blocks, expandable segments) |
| container host anon (cgroup) | 28.7; trainer RSS 27.8 |
| of which `LazyFree` | **22.6**: host memory freed by torch's CPU allocator (mimalloc in `libc10.so`, `[anon:mimalloc]` arenas) with `MADV_FREE`; still counted as RSS until the kernel reclaims it |
| live host anon | ~4.6 (Private_Dirty): records, Python, CUDA/driver, inductor |
| teacher top-64 shards | file-backed mmap (not anon) |

The ~40 GB gap is: ~8 GB of caching-allocator reserve, ~22.6 GB of lazily freed host pages, ~4.6 GB of live host
memory, and transient host spikes. The likely sources of the freed pages: host staging of tensors at load
(`safe_open(device="cpu")` / `torch.load` before `.to(cuda)`) and the checkpoint writers' host copies. VmHWM is
50.8 GB. The 20 forked inductor compile workers at the first backbone update (Lion kernels) add a short spike.
These pages are reclaimable, but the ledger counts them. On GB10 a CUDA allocation may also fail before the kernel
reclaims them, as with page cache.

### Ranked speedups (estimates for the live mix: mean 17.3 s/update)

1. **Fused fake-quant** (restart). Compute `w + mix·(Q(w) − w)` in one kernel, bf16 in and bf16 out: `torch.compile`
   of `q4_0` + ramp, or a hand-written Triton kernel. The probe shows 452 → 21 ms per layer, at the bandwidth floor.
   The compiled kernel matched eager bit-exactly on a 0.4 G-element layer only with
   `torch._inductor.config.emulate_precision_casts = True`; the default differs on 1.4e-4 of elements, by one
   int4 step (FMA contraction). Keep the eager rule as the exact verifier and add an equivalence test over every
   expert/attention tensor of the real checkpoint. Ternary is the same story: 419 → 23 ms, but compiled differs
   on 58% of elements (row-mean/alpha reduction order flips the BF16 row scale). Compute the row reductions with
   the eager kernels and fuse only the elementwise part, or make one shared rule for training and export.
   Gain: about −24 s per q4 update; mean 17.3 → ~9.6 s (−45%); remaining stage 7.4 h → ~4.1 h. The gain is larger
   later (ternary-experts, 3 passes). It also removes ~3.9 GB of peak temporaries.
2. **Grouped-GEMM MoE for trainable dense experts** (restart). Replace the 64-way Python loop + `unbind` with
   `torch._grouped_mm` over expert-sorted tokens. Extend `maple/fused_moe.py`, which today serves only frozen
   ternary experts, so the MoE keeps one implementation. The probe shows 222 → 81 ms per layer forward+backward at
   3k tokens (2.75×). It also removes the per-layer `bincount().tolist()` host sync. Gain: ~−3 s per update (all
   points). Needs a BF16 equivalence test against the loop (rounding-level differences).
3. **FP32 gradient norm in the LionSR hook** (`train/optim.py:211`): `vector_norm(dtype=float32)` per tensor,
   accumulated in FP64. The probe shows 47 → 5 ms on one expert tensor (9×). Gain: ~−1 s per update. A trivial
   change.
4. **Main stream without per-layer checkpointing** (restart). Keep checkpointing for the 4k-token preserve stream.
   After (1), the main stream's recompute is compute only. Gain ~−0.5 s; cost ~+4.5 GB (28 × ~0.16 GB per 1k
   tokens).
5. **LionSR kernel**: 38 ms per layer against a ~16 ms bandwidth floor. Gain ≤ 0.5 s. Low priority.
6. **Not recommended**: caching all 28 layers' quantized experts for the whole update. It would save the 3 repeat
   quantizations but costs ~22 GB, and after (1) the gain is < 1 s.

Combined (1)–(3): ~17.3 → ~5.5–6.5 s per mean update (~3×). The remaining stage then takes ~2.5 h at the current
mix, against ~7.4 h now and more than 12 h once ternary and 3 passes arrive.

**Memory actions** (no speed effect):
- (a) The ledger can subtract `LazyFree` (from the container processes' `smaps_rollup`) as it subtracts page
  cache. This needs no restart and frees ~22 GB of admission headroom.
- (b) At the next restart, test `MIMALLOC_PURGE_DECOMMITS=1` (purge with `MADV_DONTNEED` instead of `MADV_FREE`).
  Alternatively, load latents straight to the device (`safe_open(..., device="cuda")`) so the host copies are
  never made.
- (c) The 100 GB claim can come down to ~70 GB once (a) or (b) holds.

Every code change, (3) included, touches the code the trainer imported at start (the `run-v9/runtime` snapshot), so
it takes effect only at the next resume from checkpoint. The ledger change (a) is the only one that needs no restart. Per "newest code always",
restart at the next checkpoint after (1) lands with its exactness test.

### Implemented (2026-10-10, same day)

- **Ledger:** `LazyFree` is left out of a job's use (36255319); the guard was restarted. The live job read 59.8 GB
  instead of ~81 GB. Its claim's learned peak (95.1 GB, measured before the change) still counts until it is
  re-learned or reset.
- **(3) FP32 gradient norm** in the LionSR hook, with the total accumulated in FP64 (175c313c).
- **(1) Fused precision ramp.** The rule's reductions stay eager and are shared with export (`q4_scales` behind
  `q4_0`, `ternary_row_stats` behind `ternary_codes`); the elementwise rest is one Triton kernel
  (`maple/precision_kernels.py`; IEEE division, no FMA contraction). It differs from the eager rule on zero
  elements: tested on 2 real Mellum layers (attention and experts), q4 group 32/64 and ternary, mixes 1.0/0.47/0.03.
- **(2) Grouped-GEMM dense MoE** (`moe_kernel: grouped`, `fused_moe.dense_grouped_experts`). Against the loop at
  Mellum shape (bf16/q4/ternary), the relative error is < 1e-2 in the output and in all gradients. torch's grouped
  GEMM still syncs with the host on the GB10.
- **Recipe `raw-recurrence-mellum-v11`** = v10 + `moe_kernel: grouped` for the text warm-up and recurrence stages.

Update time on 3 real Mellum layers (`scripts/bench_mellum_qat_update.py`). Each update is the warm-up's shape: main
stream 2×500 tokens, preserve stream 3k tokens, per-layer checkpointing, layer-lockstep backward, LionSR in backward.
The GPU was shared with the live job.

| point | before | after | speedup |
|---|---|---|---|
| bf16 | 1.92 s | 1.21 s | 1.59× |
| q4 (mix 0.47) | 7.27 s | 2.23 s | 3.26× |
| ternary-experts (mix 0.5, attention int4) | 9.39 s | 3.86 s | 2.43× |

Ternary stays the slowest point. The eager row reductions of Maple's rule materialise FP32 copies of the weight. A
dedicated row-statistics kernel would have to become the one rule for export as well, which changes the published
Maple codes at reduction-order ties. That needs an owner decision.

Projected onto the live step times (q4 34.2 s, bf16 9.4 s, ternary ≈ 44 s):
- after the change: q4 ≈ 10.5 s, bf16 ≈ 5.9 s, ternary ≈ 18 s;
- now until ternary-experts starts (~step 3,510): mean 17.3 → ~7.5 s per update (2.3×);
- after that: ~28 → ~12 s per update.

### Restart onto v11 without losing progress (2026-10-10/11)

**Stop path (checked before stopping).** SIGTERM reaches the trainer: systemd stops the unit, `docker start -a`
forwards the signal, `ExecStopPost` runs `docker stop -t 600`, the recipe runner forwards it to its child, and
`TimeoutStopSec` is 10.5 min. The trainer finishes its update and writes the full state:
- latents, heads, Muon/AdamW state and the LionSR momenta;
- Python/torch/CUDA RNG, schedule, step, best, memory estimator.

The sampler is stateless (python RNG), and the preserve record and precision draw are keyed by step. The step-2842
state was 47.0 GB, written in 307 s, with 196 LionSR momenta (23.4 GB) and 316 latents (23.5 GB).

**Handoff.** `train/recipe.py --handoff` (171003f6) resumes the lineage on the newest code. The code is a clean
origin/main snapshot under `/home/werg/data/code-snapshots/`. The recipe is raw-recurrence-mellum-v11, which differs
from v10 only in an operational parameter. The lineage's `recipe_sha256` is kept, and the change goes into
`recipe-plan.json` "handoffs". The trainer logged `code_handoff` with `options_added: {moe_kernel: grouped}`.
`launch-v10.sh` drives it; it now also waits through "deactivating", which launch-v9 treated as finished.

**Problems hit and fixed on the way:**
1. **Registry not found (4a48df8b).** The snapshot resolved `training/neuralese_artifacts.json` relative to itself,
   giving "unknown artifact" for the preserve teacher. `artifacts.REPO` is now `common.paths.root('repo')`.
   - Finding: recipe children had always run the live checkout's code, not `<out>/runtime`. `python -m` puts the cwd
     (`training/neuralese`) first on `sys.path`.
2. **Resume leaked 38–48 GB of host anon memory** and stalled the preflight at step 2871.
   - Cause (085713c7): main()'s loop variable `v`, a view of the `mmap=True` checkpoint, stayed bound after
     `student_parameters` were restored. It kept the 47 GB mapping and every page the device copies dirtied. Found
     with `py-spy dump --locals`.
   - The copy now runs in `copy_restored_values`.
   - Defensive fixes on the same path: LionSR owns its row scales (6fe98c8d); optimizer restore clones host tensors
     (53fa56d6).
   - The preflight now frees page cache through the ledger's `release-cache` before waiting (6fe98c8d).
   - No progress was lost: an emergency state at 2870 and signal states at 3068 and 3141 carried the run over.
   - After the fix, the trainer's RSS is 2.8 GB and the ledger reads 54.8 GB, against ~100 GB with the leak.

**Measured on the live run** (seconds per update, mean):

| range | code | bf16 | q4 | all |
|---|---|---|---|---|
| 2562–2842 | v10 | 9.4 (n=189) | 33.9 (n=92) | 17.4 |
| 2843–3141 | v11, 1 sequence pass | 5.9 (n=201) | 9.3 (n=98) | 7.05 |
| 3142–3162 | v11, 2 sequence passes | 7.0 (n=11) | 13.9 (n=10) | 10.3 |

- **Speedup at 1 pass:** 1.6× bf16, 3.6× q4 and 2.5× overall, in line with the 3-layer bench (1.59×, 3.26×).
- **Peak GPU memory:** 54.0 GB allocated / 55.1 GB reserved, against 56.4 / 58.2 GB on v10.
- **Loss continuity across the resume:** no jump beyond per-batch noise.
  - Last 60 v10 updates: CE 0.64, bf16 loss 1.34, q4 loss 0.82 (preserve KL 0.55).
  - First 60 v11 updates: 0.87 / 1.52 / 1.83 (0.70).
  - From 3142: 0.86 / 1.31 / 1.17 (0.60).
  - Two things moved during the window. The q4 mix ramped from 0.47 to 0.80, and passes went from 1 to 2.
- **Disk:** NVMe has 12 GB free beside the 49.6 GB checkpoint reserve. The step-2842 state was replaced in place by
  the later writes.
