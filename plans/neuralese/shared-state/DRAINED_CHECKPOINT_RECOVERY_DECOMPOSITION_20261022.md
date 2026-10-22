# Drained full-checkpoint recovery decomposition

## Purpose and boundary

Recover the newest independently verified full RAW optimizer checkpoint after
its real owner has drained, then continue the existing recipe without granting
stage completion, exposure credit, or readiness that the checkpoint did not
earn. This is a supervisor recovery mechanism. It does not claim production
completion; the actual remote recovery invocation and resumed cloud learner
remain to be performed and verified by the root supervisor.

## Policy and mechanism

The supervisor's natural-language policy is: accept only the exact checkpoint
whose full handoff and schema-4 recipe transition are verified; require observed
owner drain and no live descendants; reject an active RAW/RL/semantic operation;
preserve the existing recipe lineage and all saved optimizer, RNG, exposure,
and training state; restart mandatory supervised work at its first stage with
zero completion credit; reserve a fresh operation namespace; and continue only
through the configured runtime authority that consumes this receipt once.
These are recovery decisions owned by the supervisor, not learned policy.

The crisp helpers implement only exact external-world checks and durable state
changes:

1. `import_drained_checkpoint` verifies the captured drain, model/base,
   historical recipe reader and source bindings, then prepares exact state and
   cursor before-images and candidates.
2. The existing controller lease, durable intent, compare-and-swap checks and
   atomic per-file writes publish the checkpoint into the current RAW lineage.
   Retries validate the same request, snapshots and candidates before resuming.
3. `verify_drained_checkpoint_import` authenticates the no-credit receipt and
   candidate transition against the exact current configuration and full
   checkpoint handoff.
4. `prepare_runtime_handoff` carries that receipt through the normal runtime
   proof. `_owned_cycle_boundary` admits the new boundary only with mandatory
   supervised stages present and no pending operation.
5. `student_cycle` consumes the receipt once at the fresh SFT boundary and
   rejects stale work in the reserved namespace before the learner can update.

The selected policy stays in the existing coordinator/runtime path; the new
code is limited to verification, durability, custody plumbing, and receipt
consumption. Historical recipe interpretation reuses the captured historical
package reader, while actual executed-runtime proof remains separately bound.

## Evidence and status

Local focused checks cover transaction rejection/retry, schema-4 runtime
handoff, receipt consumption, and a real next-optimizer update for both LFM
and Mellum. Those checks establish mechanism behavior on the verified fixtures;
they do not establish that the remote checkpoint has been imported or that
production training has resumed. Root supervision owns the actual cloud call,
deployment, and confirmation of subsequent optimizer progress.
