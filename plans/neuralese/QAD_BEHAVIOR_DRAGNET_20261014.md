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
- **Open:** quantify the program-header versus full-prompt serving effect on
  the exact checkpoint with the same 20 tasks and seeds. The first PH rollout
  (`natlang-pop-qad-trained-program-header20-v1`) finished 2/38 scored, but
  used the preparer's 1,000-per-row seed stride while the full/base arms used
  3,000 per row. Only the first task's two samples had the same seed. This is
  exploratory, not a matched ablation; its two source-held MuSiQue samples
  remain unscored. A new preparation copies seeds by task ID and sample from
  the clean-base reference and verifies all 40 mappings plus identical packet
  SHA. Rerun the PH arm on those exact seeds before interpreting rendering.
- **Open:** run the current-main full gate with the pinned training-image
  interpreter; the first Pop attempt lacked a host Python 3.11+ torch stack.
  The image-backed run is queued under the ledger.
- **Open:** qualify a fresh clean-QAD-base lineage on the corrected shared
  code and filtered R7 union. Compare held response CE/KL, repetition, exact
  task behavior, and the mask/gradient receipts stage by stage. Do not restore
  optimizer or weights from the old step-15000 checkpoint.

Future divergence checks should compare exact known-good references for data
and masks, prompt rendering, dtype/upcasts, state/cache between steps, kernels,
optimizer/resume, gradient flow, loss weighting, and scoring. A behavioral
regression is a bug to localize until those checks rule it out.
