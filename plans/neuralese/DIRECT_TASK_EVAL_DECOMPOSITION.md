# Direct checkpoint task evaluation decomposition

## Goal

Evaluate a direct-SFT checkpoint on the fixed, protected 48-case authored task
packet. Execute the authored root program and sample its child decisions through
the shared teacher collector and native runtime. Keep the task's expected value
and effects in evaluator metadata. Compare only runs that use the same packet,
collector runtime, base snapshot, serving precision, and execution settings.

## Stages

1. **Bind the checkpoint** — crisp exact verification. Copy the rolling checkpoint
   to a fresh immutable candidate path, hash it, and export it through
   `train.direct_export.export_direct_checkpoint` with the launch receipt and
   exact local base snapshot. The existing strict exporter owns checkpoint and
   serving-artifact identity.
2. **Run the tasks** — the authored task roots are already natlang programs.
   The existing collector verifies each root's declared source hash, executes
   that root, asks the model for the child actions, and lets the host interpreter
   execute/check task values and effects. A fixed queue and frozen collector
   runtime supply durability and repeatable execution; no sampled planner or
   separate model-serving path is introduced.
3. **Classify outcomes** — crisp exact verification of collector terminal rows.
   Accepted full task results count as success. Explicit collector evidence
   distinguishes invalid actions, repetition, resource exhaustion, and
   infrastructure failures; other completed mismatches remain their own class.
4. **Choose the retained checkpoint** — the policy is: among fully accounted
   evaluations of the same packet/configuration, keep the checkpoint with the
   greatest count of accepted tasks; on a tie, keep the earlier checkpoint.
   The score is computed from exact collector `outcome.accepted` values. Crisp
   code verifies the complete 48-row set and atomically publishes a hard link
   from an immutable candidate checkpoint, preserving the full optimizer state.

## Exclusions

The task scorer never returns expected answers to the model. The protected
packet remains test-only and is not a training source. Incomplete or invalid
evaluation runs cannot replace the current best. The task score does not claim
native-latent writer qualification; it measures only the ordinary text/tool
serving path of a `direct-writer-pending` artifact.
