# Foundation-boundary task execution receipt

The communication-foundation policy already requires a current-weight runtime
certificate and a held-task execution receipt before publishing readiness. The
natural-language recipe remains the authority for which held tasks run. The
missing receipt is a recipe omission, not a reason to add a task-success bar or
to rewrite trainer policy.

For a foundation SFT continuation, retain the registered held `task_eval`
specification and set `task_eval_every` to `0`. The existing trainer interprets
zero as no periodic task evaluations; it still runs the task evaluator at the
foundation refresh boundary using the current checkpoint. This keeps task
execution off routine optimizer updates while making readiness executable.

The exact preparation mechanism takes a byte-bound concrete parent recipe and
a byte-bound registered recipe containing the held task-evaluation spec. It
checks that the target stage is a communication-foundation native SFT stage,
that the parent currently has no conflicting task-evaluation spec, that the
registered packet is valid, and that the target cadence is zero. It copies the
parent recipe and changes only `task_eval` and `task_eval_every`, then writes a
durable proof binding both inputs and the output. It never edits a live recipe,
checkpoint, exposure record, or optimizer state.

Task execution eligibility is unchanged: all selected task outcomes must be
accounted, and infrastructure or unresolved outcomes remain ineligible. A
completed unsuccessful task outcome is still an honest execution receipt;
task success remains telemetry and is not a foundation qualification barrier.

## Natlang decomposition

1. The recipe's natural-language policy names a registered held packet and the
   stage's foundation-boundary evaluation cadence.
2. The existing evaluator runs authored tasks against the current served model
   and records exact outcome accounting. No new model-facing prompt is added.
3. A small crisp preparer verifies input recipe/packet identities and performs
   the exact two-field recipe copy with durable output bindings. These checks
   touch external recipe files and exact hashes, so they remain crisp plumbing.
4. The existing receipt validator and foundation readiness check decide whether
   the observed task execution is eligible. They do not impose a success-count
   threshold.
