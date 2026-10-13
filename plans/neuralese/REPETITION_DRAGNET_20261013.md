# Repetition dragnet, 2026-10-13 (DGX, dgx-repeat)

Symptom: both students loop. Pop QAD v7 step 1000: 11 of 12 budget-exhausted held tasks end in >=8 identical
(action, observation) turns (102-127 repeats). Harness bench: the Muon LFM student repeats an earlier identical pi
call in 93-95% of calls. Mellum v4 step 2000: 4 of 5 budget exhaustions are exact loops (identical suffixes of 46/48,
14/18, 45/48, 23/25 turns); the fifth (greenhouse) polls `greenhouse.read()` without a control action. Step 1600 and
its repeat show the same loops.

## Hypotheses

| # | Hypothesis | Verdict | Evidence |
|---|---|---|---|
| 1 | Training targets contain loops | **ruled out** | Scan of all four r3 renders (train split, `/tmp/repeat-dragnet/h1_scan.py`): program-header 1,127/357,632 supervised turns (0.32%) repeat an earlier identical call, 291 (0.08%) on an identical observation; whole-trajectory 2,570/577,117 (0.45%), 603 (0.10%). Documents ending in >=8 identical turns: 4 (PH) / 0 (WT). Failed/blocked final returns: 227 PH, 118 WT (honest stops). Twins equal within 1%. |
| 2a | Observations missing before supervised actions | **ruled out** | Every r3 PH row has a tool/user observation before its supervised turn (45 assistant-preceded). feedback_weight 0.25 only scales loss on observation tokens; they stay in context. merge_prefix_rows is causal-exact; no cross-document attention (B x L padded windows). |
| 2b | Truncated context in long documents | **ruled in (bug, fixed opt-in)** | `document_windows` with tokens 32768 and train_target_tokens 31744 gives every later window only 1,024 tokens of context: 20.1M Mellum / 7.9M LFM PH tokens (2.5%), 8.2M / 3.6M WT, all pi harness documents (1,120 / 695 PH docs). Matches the pi-harness symptom (copying openings, repeated calls). Held windows keep full context, so held CE cannot see it. Fix babd28a2: `train_min_context_tokens` windowing parameter. |
| 3a | Eval prompt differs from training (Mellum) | **ruled in (Mellum v4 only)** | Mellum v4 runs a frozen snapshot (`natlang-8e431f14`) without 4ffd9a4f, so its task-eval queue has no `prompt_rendering`; the server kept the collector's 5 tool schemas: ~1,070-token `<tools>` block in the system turn that PH training never has (case-0004 rendered 1,927 tokens = logged prompt_tokens; 857 without). Pop's eval has `prompt_rendering: program-header` (662-token first prompt) and loops too, so this is not the loop cause, but it invalidates Mellum's eval comparability. Fix: Mellum v4 to current code at its next handoff. |
| 3b | Position-0 token | open (measurement pending, CPU A/B) | Training windows are `[open_id] + text + [close_id]` (Neuralese open marker at position 0, close after the final `<|im_end|>`); serving prepends nothing. Every training document differs from serving at the attention-sink position. |
| 3c | Other rendering | ruled out | Role layout, `<tool_response>` wrapping, call format, history reasoning policy, generation prompt, observation text match (Mellum case-0004 token comparison). Bare-call (no think) replies are in distribution (48% of Mellum PH suffixes; 69% of LFM). |
| 4 | Decoding | **ruled in (amplifier)** | task_cadence sends no temperature; the engine default is argmax. Mellum step-1600 and 1600-repeat1 are identical action for action, so the noise repeat measured nothing. Greedy + byte-identical observation = deterministic fixed point. Harness at T=0.6 still repeats 93-95%, so greedy is not the whole cause. Fix ecd1815a: `task_eval.temperature` (default 0 unchanged), sampled repeats draw new seeds. |
| 5 | Runtime feedback on repeats | partly ruled in, low leverage | A successful identical rerun returns a byte-identical observation (redeclared const allowed; the `Stored local` line drops after the first run) with no note. Failed reruns already get "This is the code of your previous eval, and it failed the same way: change it before running it again." That note is present in 5 of Pop's 11 loops and Mellum's frontier loop and is ignored 23-127 times. `return_result` with a ProofResult status in its own status field gets "rejected status: bad-action..." which does not teach putting it in `value`. |

## What the loops are (Mellum step 2000, Pop step 1000)

- No-progress successful reads: multihop `graph.outgoing(parent, "owns", 1)` (empty: the relation is `owned_by`,
  incoming), route_replan `world.look()`, late_argmax `page_count()`, entailment `facts.page(1)`, frontier `graph.links`.
- Errors with the existing repeat note: undeclared fields, `const x = x.open()`, syntax errors, `forbidden-loop`,
  array grown while iterating.
- Rejected results: wrong `return_result` status (45x), type-mismatched record (126x), `compact_history` same note (102x).

The common pattern is a policy that does not condition its next action on the observation.

## Fixes landed

- babd28a2 `train_min_context_tokens` (recipe/CLI windowing parameter, default 0 = old geometry so running lineages
  resume unchanged; handoff moves a running stage by a windowing continuation). Recommended 16384 for both lines at
  their next handoff and for r4-based lineages.
- ecd1815a `task_eval.temperature`. Recommended: declare 0.6 (harness value) at the next eval handoff, keeping a greedy
  run for continuity if wanted; until then the "noise" of greedy repeats is zero by construction.

## Not landed (needs live measurement, MODEL_FACING_CHANGES rule)

- Runtime note for an identical successful rerun with identical output (`ts-host/src/native/runtime.ts` `evaluate()`,
  beside `failureDebug`; same for identical rejected tool calls in `scopeTool`/`record()`), e.g. "This eval repeats your
  previous one and returned the same output; use that output to decide a different next step." And teach the
  `return_result` status fix in its error. Expected effect small (the failure repeat note is already ignored). It would
  also change in-run eval comparability mid-lineage, so measure it in a teacher window (~48 samples per variant) first.

## Recommendations

1. Mellum v4: move to current code at its next handoff (eval rendering 4ffd9a4f); its step 1200-2000 task scores were
   measured with a tool-schema prompt.
2. Both lines: `train_min_context_tokens` 16384 at the next handoff; eval temperature 0.6.
3. r4 admission: optional hygiene only (drop the 4+12 documents with >=8 identical streaks and 135 PH documents ending
   in >=2-streaks, 92 of them from the s72 curriculum); the data is not the cause.
4. Behaviour data: recovery supervision (an observation that shows no progress followed by a different action) is the
   remaining lever; the targets almost never contain a repeated action, so the model never sees "same call twice, then
   change" either.

## Pop QAD step-8000 follow-up — 2026-10-13

Owner: Pop. This is an active investigation, not a verdict that the model cannot learn recovery. Keep the checkpoint lineage and investigate unexplained gaps against exact references.

## Observed state

- Direct SFT v12 step 8000: held CE 0.675470 (base 2.679), gold accuracy 0.86277, response KL 1.12542. Full-system-prompt task eval: 0/16 success, ten exact repeated-action stops, two invalid actions, three semantic mismatches, one budget exhaustion, zero infrastructure errors. The task gate has not shown success at earlier steps either.
- The ten repeated cases each have many identical adjacent `eval` actions (10–41), with fresh, unique request hashes on every turn. The unchanged observation appears in the saved partial. Example: workbench tickets repeated the same TypeScript syntax error despite explicit “change it before running it again” feedback; its prompt grew from 5,413 to 11,185 tokens. Across repeated cases the largest saved prompt was 18,801 tokens, within the eval's declared 32,768 context.
- The supervisor's `identical_turn_limit=8` matches the same invocation, exact tool-call JSON and exact observation hash over eight saved turns. It does not classify a reused action on another item as repetition. The stop labels are grounded in actual no-progress loops; their underlying cause remains open.
- Of 418 saved assistant turns in the step-8000 task packet, zero contain `reasoning_content` or `<think>` in the server's returned message. The DGX whole-corpus scan above found 69% of LFM program-header suffixes are bare calls, so this is **not by itself a template bug**. Check whether the other 31% and the live prompt would predict any thinking before assigning a cause.

## Training and rendering checks so far

- The declared v12 direct SFT objective uses the r3 `program-header` input, then r3 `whole-trajectory`; the stage sampler declares recovery cohort weight 0.5 with two repeats. The handoff entered at step 7014. The `windows` record says 11,580 recovery candidate documents among 357,632, with 50/50 cohort sampling policy. Actual recovery draws since step 7014 still need measurement.
- In the first 150,000 r3 `program-header` rows, 8,775 contain the exact runtime feedback “Nothing else from this eval was kept”; the corresponding whole-trajectory slice has 774. Representative rows put the next assistant action after the observation. Program-header rows declare `supervised_suffix_start` at that assistant action, and `direct_sft.py` gives positive suffix weight while masking the failed preceding assistant action. The actual sampled token mass on these rows still needs counting.
- In the first 40,000 r3 rows, 32,707 program-header rows contain nonempty `<think>` (46,060 nonempty blocks) and 12,024 whole-trajectory rows contain nonempty `<think>`. A first-50,000-row r2 slice contains 40,579 such rows. **These prefix slices are corpus-order-skewed and not estimates of the full mix**; use the DGX full-corpus 69% bare-call suffix count above. A sampled r3 program-header row's `token_ids` decoded byte-for-byte to `text`; its supervised suffix starts at `<think>`. This rules out a blanket absence of reasoning targets or a simple token/text mismatch in that sample.
- The QAD LFM chat template's generation prefix is `<|im_start|>assistant\n`; it permits `<think>` as the first generated content. The template renders a supplied `reasoning_content` as `<think>…</think>`. LFM's declared `last_turn_only` policy drops older turns' thoughts in training and serving, while retaining the latest turn. The server's `build_message` extracts a generated `<think>` block to `reasoning_content`; current eval responses never expose one. More direct inspection of raw generated tokens and first-token logits is needed before ruling out a parser/generation bug.
- The task evaluator uses the same training engine through a local HTTP server, full prompt rendering, temperature 0.6, 32,768 context, and no explicit reasoning disable. It only requires a tool call near its last/compaction turns. Need compare the *exact* prompt prefix and request settings for a repeated state with a known-good train example.
- Pop confirmed a substantial declared prompt-distribution gap: the active r3 LFM program-header text-data rows have `system_prompt_mode: none` and begin with a user turn; the whole-trajectory twin likewise begins with a user turn. The step-8000 task evaluator uses `prompt_rendering: full`; case-0001's first model request contains a 16,126-character system turn and five tool schemas. This does not alone prove the cause of zero reasoning or repeated calls. Run the same protected task packet with pinned program-header rendering for a direct A/B while retaining the full-prompt result as a deployment measure.
- In one inspected step-8000 case, the second action received the exact `nl-not-tag` runtime error, after which the model repeated essentially the same invalid `eval` code. The returned `model_response` has no reasoning field; the assistant sometimes writes planning prose in ordinary content. The error was visible to the model, so this is a concrete failure to condition on feedback.

## Next checks

1. Reconstruct two exact repeated requests from durable evidence and compare model-visible messages/tool schemas with training rows; check tool/error observation placement, full versus program-header prompt, tokenization, and cache positions. Inspect raw generated token IDs before `build_message`, especially the probability of `<think>` versus direct tool-call opening.
2. Count actual recovery source windows and weighted supervised tokens drawn between 7014 and 8551. In selected post-error windows, verify nonzero positive weights on the correction and zero weight on the failed action.
3. At a checkpoint, run a bounded three-state decode sweep (greedy and a few temperatures/top-p settings) with the identical full prompt, then remove only the repeated observation from a diagnostic copy. Preserve the protected eval packet and do not train on those cases.
4. Compare the same states with a full recomputation against the normal request cache path to rule out stale KV. Inspect stop counter and context compaction evidence per case.
5. If a mechanism is confirmed, fix the shared trainer/serving path, remeasure on the live executor, and coordinate DGX Mellum checks. Do not patch an isolated eval path as a substitute for the shared fix.

## Training interruption

At step 8552 the adaptive memory estimator predicted 5,516,523,000 bytes incremental against a 5,463,919,820-byte reusable ledger claim and raised before forward execution. This was a preflight capacity stop, not a numerical/model regression. Pop restarted the same v12 checkpoint lineage with discrete CUDA envelope 7.3 GiB (previously 7.2); preserve exact optimizer/RNG resume and check that training advances again. A first relaunch refused the recipe code pin because the shared `common/hdd_guard.py` changed after v12 was frozen. The pinned frozen runtime matches every recipe package hash, so the next launch uses that active-lineage runtime for this resume; this is not an adoption of changed shared code.

That relaunch restored step 8360 and ran through 8362, but step 8363 OOMed in the chunked CE/KL gradient (`grad_logits.T @ h`, 256 MiB allocation): 7.35 GiB used, only 259 MiB free and 479 MiB PyTorch reserved but unallocated. Its plan estimated 5.591 GiB incremental against 5.640 GiB reusable, too close for this allocation shape. The systemd ledger supplied `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to its wrapper, but Docker did not inherit that variable. A second relaunch passed it into Docker and raised the discrete claim to 7.35 GiB, but refused before loading the checkpoint: the shared parent recipe had since deleted `neuralese_weight` and `value_margin`, while the frozen lineage plan still declares them, so fresh resolution of its inherited v12 recipe is no longer handoff-identical. Pop wrote the exact already-recorded resolved v12 recipe to the run directory (`resume-v12-frozen.json`) and relaunched with its pinned frozen runtime, preserving the existing weights, optimizer and data cursor. That unit was admitted at 18:18 CEST. Verify allocator config in the next `train.jsonl` step and that it passes steps 8363 and 8552 before considering this fixed. The inheritance drift is a recipe reproducibility bug to eliminate at the next shared-code handoff.
