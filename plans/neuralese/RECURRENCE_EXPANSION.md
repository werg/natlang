# Rich recurrence corpus — 2026-10-05

## Findings

The v13-r2 native teacher file contains 111,301 decisions, 45,940 eval targets,
6,905 child decisions, 2,122 successful child `return_result` targets, and 847
text/structured return targets at least 16 characters long. Only 1,388 eval
targets contain inline-lambda syntax (~3%); 560 match named-function call syntax.
These are lexical candidates, not successful invocation counts. Count targets,
not repeated history. There are also 121 plain child replies (69 long): the
converter and trajectory writer currently support tool-shaped final writes only.

The restrictive semantic prompt originated in `df1f679a` (Oct 1). It survived
subsequent additions of lambda examples. The user requested its removal and an
increase from three to **five** ad hoc layers. Current source promotes inline
judgments, extraction, transformations and itemwise calls. At layer five, prompts
and help remove creation guidance; layer six is refused. Every file-backed `.nl`
function starts a new root budget. Frozen campaigns retain their pinned prompt;
new campaigns need a rebuilt, verified runtime before these changes take effect.

Conversion v4 recognizes named functions from pinned program files and scopes
result names by trajectory identity plus value, preventing cross-run collisions.
A no-output audit on the same teacher file raises counted writes 153→197 and
reads 342→582. These are occurrences including history, not unique producers.
Remaining counts: producer-missing 2,020; crisp-value 248; value-not-printed 1,336.
Newly recognized calls expose additional omissions. Do not invent missing returns
or treat formatting differences as evidence of execution. Equal values within a
run can still have ambiguous producers. Native export now preserves invocation
IDs from the raw trajectory for subsequent precise recovery.

## Implementation and live work

- `inline-curriculum/recurrence.mjs`: exact-oracle triage, delivery evidence and
  cancellation policy; private leaf services, nested named functions, paired
  hidden worlds with differing answers and identical parent openings. Both named
  graphs and graphs with anonymous semantic judgment lambdas are registered.
- Depth 1–4, width 1–3, varying source padding. Padding size is **not** verified
  model context length: runtime pagination can shorten what the model sees.
- Source siblings share split groups; held-out seeds differ from training seeds.
- `verify-recurrence-pool.mjs` checks holds, runtime replay, native linking against
  a specified collector runtime. Proof is not model-output admission.
- `audit_natlang_delegation.py`: streaming actual-target frequency audit.
- `audit_neuralese_recurrence.py`: producer closure, ambiguity, cycles, graph/source
  split checks. Missing historic compaction producers can legitimately use crisp
  source fallback; distinguish those from missing child writers before admission.
- 300 initial named cases and 48 held-out cases verified. Both 32-case named and
  inline pilots verified. 600 expanded training cases across both families verified with clearer
  once-per-helper wording and a separate seed. Published source-only snapshot:
  `recurrence-task-pool-20261005-v1` (plus48 held-out cases), not model trajectories.
- Space Bunny 32-case named pilot is running in
  `runs/recurrence-expansion-20261005/space-bunny/`, one request, free Stealth only,
  no distillation flag, normal retry delay/backoff. Frozen v38 runtime. At review,
  20 completed / 3 accepted: root duplicated a child call in an inspected failure.
  Keep raw failures; do not loosen exact array/type/label grading or scale blindly.
- Local writer-control GPU run reached step 200/512 with zero terminal errors;
  no inference job competes with it. DGX job ownership remains with the DGX agent.

## Next required work

1. Finish pilot rejection clustering (wrong labels, duplicated children, incorrect
   result shape, scope mistakes). Verify native admission and observed call graph,
   not collector completion alone. Register sealed accepted outputs separately.
2. Rebuild/freeze a current runtime with the new prompt and five-layer budget;
   replay proof, then a fresh inline pilot. Coordinate external/DGX scheduling.
3. Recover complete valid historical child finals from raw traces using preserved
   invocation identity; exclude ambiguity and protected split crossings. Support
   plain final writes in converter **and trainer/template rendering together**.
   Match structured displayed values using the runtime's exact printer, not fuzzy
   whitespace/text matching. Escaped JSON strings also need exact treatment.
4. Add shared-child reuse, argument/instruction handoffs and deeper inline chains;
   extend to grounded summaries, translation/editing and skill improvement.
   Current three small constructed domains are a starting calibration suite.
5. Implement correct/shuffled/zero/removed-return and counterfactual evaluation.
   Controls currently declared in case metadata are **not an implemented evaluator**.
   Measure content dependence and task pass rates, not merely lower target CE.
6. Build a closed, graph-group-split recurrence training snapshot after admission,
   publish SHA manifests, synchronize, and train/evaluate on that snapshot. Do not
   count case pools, reference replays, replicas and generated trajectories as
   separate independent data.

Evidence: `training/audits/delegation-20261005.json` and
`training/audits/recurrence-conversion-v4-20261005.json`.
