# Pop QAD behavior dragnet, 2026-10-14

The step-15000 LFM QAD direct checkpoint is diagnostic. It was trained before
the fused BF16 gradient-normalization correction (`b6d262af`). Do not reject
the direct approach or call the old checkpoint a training gain on this evidence.

## Matched task evidence

Both arms used the same 20 train-split tasks, two root-generated samples per
task, seeds, temperature 0.6, full prompt, collector and serving image. The
base arm restored the clean QAD base; the trained arm restored the exact
step-15000 checkpoint and clean base heads. The base arm's two MuSiQue samples
were quarantined by the new family-level source hold before generation; they
are unscored. Two trained samples have no comparable score (one unresolved
recurrence case and one unresolved workbench case). On the 36 pairs scored in
both arms, base passed two and trained passed one. The wins were both samples
of one `policy_after_measure` task: both passed one seed, and base alone passed
the other. There were no student-only wins.

Across all 40 pair slots, base outcomes were 2 success, 29 semantic mismatch,
4 invalid action, 2 request-budget exhaustion, 1 model incomplete and 2
source-held unresolved. Trained outcomes were 1 success, 14 semantic mismatch,
17 invalid action, 6 request-budget exhaustion and 2 unresolved. The broader
60-task trained packet, two samples each, was operator-closed at 115/120 exact
terminal journals to release the GPU from its serial long tail: 114 scored,
5 successes; five missing queue entries and one wall-limit terminal are
unscored. The closeout receipts and exact merged-journal hashes live beside
the run, under `natlang-artifacts/qad-student-train-rollouts-20261014-v2/`.

`invalid_action` includes wrong typed final values and malformed or misplaced
tool use, not only syntax errors. A trained greenhouse sample repeatedly tried
an invalid `eval` expression with a nonexistent Python helper and finally
returned the string `greenhouse.certificate()`; the matched base sample
returned a narrative claim without invoking the certificate. Both failed, by
different mechanisms. Inspect action traces before interpreting outcome counts.

## Ruled out and still open

- Prior exact serving/training prefix checks found no prompt-prefix drift at
  the sampled boundary. FP32 versus BF16 serving drift was similar. Those
  observations do not validate the training gradient.
- The fused BF16 normalization bug is concrete: the corrected kernel now
  matches the ordinary backward bitwise on the actual 63-label comparison.
  The step-15000 checkpoint predates that fix and cannot certify the corrected
  implementation.
- The old training run showed lower held CE but higher response CE/KL and
  repetition. The matched task result is consistent with degraded action
  behavior, but does not localize which training stage caused it.
- Source quality is independently at risk: MuSiQue answer occurrence was used
  as a proxy for relation entailment. Both hop families are held and DGX is
  rebuilding the target union without their rendered member.
- **Measured:** the first PH rollout finished 2/38 scored but used a
  1,000-per-row seed stride while the full/base arms used 3,000; it is only
  exploratory. The corrected PH run copied seeds by task ID and sample from
  the clean-base queue, verified all 40 mappings plus identical packet SHA,
  and finished 2/38 scored. Across the 40 exact task/sample/seed slots, base
  had 2 successes, trained full prompt 1, and trained program header 2.
  Program header recovered one `policy_after_measure` seed and added one
  `inline_avoid_crisp` success, but lost the other base-successful
  `policy_after_measure` seed. It had 16 invalid actions and 9 budget
  exhaustions versus full prompt's 17 and 6, so rendering alone did not
  restore broad action behavior. Both PH runs had two source-held MuSiQue
  slots with no model result, and no DPO or training admission. The corrected
  run's journal SHA and source-hold receipt are under
  `round2-trained-program-header20-matched-seeds/diagnostic-closeout.json`.
- **Done:** the image-backed full gate passed end to end at `be4d58c9`:
  Python general 962 pass, Neuralese 1290 pass, GPU 14 pass, TS and
  conformance green. Host/image isolation fixes are on main.
- **Held:** DGX's post-build audit found that excluding MuSiQue from R7 also
  released held-group protection on 409 header, 104 whole-trajectory and four
  full examples. Pop had started fresh clean-QAD-base R7 training before
  receiving the hold, but stopped exact unit `natlang-pop-qad-r7-clean-main-v2`
  after 38 warm-up updates, deleted its optimizer checkpoint and other run
  artifacts, and withdrew the R7 recipe. Do not use its weights. DGX is
  rebuilding a new immutable union with excluded-source held references.
  The container also required host-mounted `zstd` for compressed data; the
  corrected launch verified it, and future launches need the same facility.
  Once the replacement union passes source and hash review, compare held
  response CE/KL, repetition, exact
  task behavior, and the mask/gradient receipts stage by stage. Do not restore
  optimizer or weights from the old step-15000 checkpoint.

Future divergence checks should compare exact known-good references for data
and masks, prompt rendering, dtype/upcasts, state/cache between steps, kernels,
optimizer/resume, gradient flow, loss weighting, and scoring. A behavioral
regression is a bug to localize until those checks rule it out.

## Protected R8 warm-up, matched train-only student roots (2026-10-14)

The independently qualified clean QAD R8 warm-up is a starting point for PH
SFT, not a claim of task competence. Pop sampled two rollouts for each of 11
held-disjoint train families in `qad-r8-warmup-student-root-20261014-v4`, then
ran the pinned LFM base on the exact same packet and seeds in
`qad-r8-base-matched-student-root-20261014-v4`. Both use the full prompt and
the same serving image. The packet SHA-256 is
`21652ee30befca810efb1283436cd4a869faead6f13874397432c8da28c8d280`.

Both arms passed **0/22**. The warm-up arm had 18 `invalid_action`, three
`semantic_mismatch`, one `budget_exhaustion` (score SHA-256
`0b11b288e1ac450ac0e482bfc8527340e9d06f95652287b0cadb08f7b29820b2`).
The base had one `invalid_action`, 19 `semantic_mismatch`, one
`budget_exhaustion`, one `model_incomplete` (score SHA-256
`0bf1366fe534ae8f8b48cc937e1a5026878c29c4ccf72ae2bccf624ad0d03f55`).
This batch is too hard to use as a pass-rate gate at warm-up. The different
failure mix still warrants an action-validity dragnet after PH SFT. The
`invalid_action` scorer reports any rejected action in the trajectory before
the final answer; for example, one route-planning run repeatedly tried to
return the `world` service object as a string. It does not mean all 18 runs
ended with a syntax error.

The v4 exact replay produced 43 handoff sites (25 failed actions, 18 wrong
results), skipped four whose replay diverged, and selected 11 distinct
train-only source groups for teacher continuation. The selected source SHA is
`08ee65459f6a59f91413fc99607ea63c91682260af0684d73c62ea0c37c70aac`.
The earlier v3 produced 27 exact sites and selected 11. The PH CoreWeave
continuation should be judged stage by stage against this exact behavior
baseline, including action diagnostics and post-observation recovery.

## Recovery review and continuation feed, 2026-10-14

The v3 Luna correction campaign yielded 11/11 exact-accepted parent outcomes.
Native decision review admitted eight parents and 41 successful teacher turns
as `qad-r8-student-root-recovery-v3-20261014-v2` (approved-turns SHA-256
`a9f8d662a25cfafd948fbe79f8c5f88630506ac3d35c52859e5ab776feaa2c94`).
The protected held-48 program, source ID and source-group intersections are
empty. Three exact-answer parents are held: composed process used an arbitrary
10,000-step loop instead of required `iterateOn`; edit stream bypassed the
required one-instruction-at-a-time `iterateOn` and NL edit; counterexample
revision hardcoded a record ID instead of expressing a reusable identity rule.
One further target repeated `world.look()` after an identical observation;
it was excluded while later route recovery actions remain. Unapproved
student-prefix and failed decisions are not training targets. The initially
published v1 source and LFM render were held in the registry once this repeat
was found. The corrected v2 source and LFM render manifests were verified
after sync to DGX. The corrected render contains 41 program-header, 23
whole-trajectory and three sampled full train documents, with zero local
decontamination omissions and 41 equal text/serving token boundaries. The
Mellum tokenizer twin and global held/source-quality proof remain before a
later protected data continuation; the active PH snapshot is unchanged.
The shared curator's exact-source-bound parent and individual-decision hold
path independently selected the same 41 decision IDs from all 57 materialized
turns; its same-invocation duplicate-action check now requires an explicit
review for later campaigns (`dc120214`).

The v4 campaign has 11 distinct replay-checked train-only handoffs. Two Luna
correction workers started on Pop; the replenisher now grants them first claim
on openings within the same six-worker cap (`466eebf4`). Their outputs have no
training admission yet.
