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
