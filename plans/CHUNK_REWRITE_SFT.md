# Threshold-triggered chunk rewriting

Status: implemented; development pilots, not automatically published or promoted.
Entry point: `ts-host/scripts/rewrite-student-chunks.mjs PLAN --execute PLAN_SHA`.
Build a normal render/token-audit graph with `scripts/create_student_rewrite_pipeline.py`.
Method is `verified-threshold-chunk-rewrite/1`, a heuristic rather than MH.

## Contract

- Freeze student weights and exact train-only source closure/runtime/operator pins.
- Replay admitted teacher references fresh. Score only approved assistant targets;
  failed actions and detours remain context, not perplexity objectives or SFT labels.
- Ordinary student context has no teacher hints. EOS is scored at complete action
  boundaries. Parsed spans use exact tokenizer offsets, failing closed on mismatch.
- Gate on trajectory mean NLL and chunk mean/worst-token NLL. Choose first difficult
  parsed tool name, argument, code line or prose sentence. If whole difficulty is
  high but no fine span qualifies, coarsen to a complete assistant action.
- Generate bounded proposals with teacher (or student when teacher is absent).
  Teacher candidates are ranked by temperature1 prompt-logprob NLL of the complete
  JSON replacement response in the rewrite context. This is NOT teacher likelihood
  under the student's task prompt. Wrapper/type/length effects require evaluation.
- Pre-score candidates with student; reject too-difficult chunks before fresh replay.
  Then require exact task/native/curriculum admission and no trajectory NLL regression.
- Changed non-assistant messages/tool surface trigger fresh teacher continuation,
  conservatively including changes that might only be presentation. Re-execute every
  action and recompute actual observations. Never retain an invented old tool result.
- Continue one complete teacher action at a time, score with student before execution,
  stop after configured consecutive difficult actions. Stop is persistent across runtime
  repair. Without teacher, changed observations force prefix fallback.
- Bound edits, retries, calls and output lengths. Retain best valid full trajectory
  despite exploratory teacher paths. Record accepted chunks separately from continuations.
- Fallback SFT includes approved decisions through one complete corrective action in
  the last validated trajectory. New unvalidated partial continuations are review-only.
  No artificial EOS halfway through code or a tool argument. The durable native
  `provenance.student_chunk_rewrite.supervision_cutoff_decision` prevents re-export from
  accidentally restoring suffix targets. Invalid cutoffs fail closed.

## Pilot findings and limits

Initial six-case pilot exercised vanilla reuse, teacher-ranked local rewrites, teacher
continuation and prefix fallback. One code workflow fell from0.6642 to0.6532mean NLL;
API-discovery workflow0.8121 to0.5872. These are preliminary data-shaping results,
NOT measured post-training accuracy gains. Earlier pilots used all trajectory decisions
in the objective and lacked durable cutoff markers; they remain held for review.
The corrected version masks context decisions and records prefix boundaries durably.
128token teacher continuation was too small and produced empty parsed tool arguments;
512token allowance used in the next pilot. No task oracle or exact numerical condition
was weakened. Thresholds remain exploratory and need train-only per-modality calibration.

Batch preprocessing first; integration does not reuse a training autograd graph.
Within-action streaming early scoring, richer JS AST statement chunks, dependency-aware
presentation equivalence, empirical threshold calibration, resumable mid-case search
and demonstrated downstream accuracy/cost benefit are remaining work. The operator
uses fresh attempt directories; completed/interrupted previous pilot artifacts stay intact.
