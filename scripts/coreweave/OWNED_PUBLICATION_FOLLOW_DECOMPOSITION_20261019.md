# Follow the currently supervised checkpoint publisher

The supervisor's policy selects one active native learner and one exact parent
for each quantum. Publication backup follows that selection; it does not choose
a model, infer a parent, or grant training/readiness credit.

1. At the existing supervised-command boundary, expand the command with the
   same shared argv substitution used for launch. Bind the exact selected
   checkpoint, output root, runtime inventory, publication producers and
   operation identity. Keep the command and manifest immutable under that
   operation's output directory.
2. Atomically replace one role-specific current-owner pointer with the binding
   of the new manifest. The pointer is only a selector. The owner intent is not
   proof of liveness, optimizer progress or publication success.
3. The external exporter reads the pointer and immutable manifest, verifies all
   bindings and invokes the existing `observe_publication` verifier. That
   verifier still requires the checkpoint's successful status and exact
   checkpoint control against the selected parent and runtime producers. Read
   the pointer again before returning; retry if it changed during observation.
4. Give every distinct manifest generation its own exporter journal. Preserve
   previous generations and their pending full-object uploads. This supports a
   new supervised output without resetting an old watermark or mistaking a
   stale learner's output for the current one.
5. Derive read-only roots only from the declared current recipe's authenticated
   external source roots. The current LF recipe explicitly needs the two D24
   trees under `capacity-envelope-candidate-v2` and `d24-whole-pending-20261017`;
   never authorize their `/work/data-preparation` ancestor. The existing
   production namespace remains the authority for `/work/native-production`.
   Start the follower with `--root /work/native-production` and the same
   role-specific `/work/native-production/control/...` pointer configured by
   the supervised controller; the older shared-deployment root is not the
   current owner namespace. Keep the follower's per-role journal in its own
   durable state directory.

Exact durable file binding, the atomic selector, and remote publication checks
are crisp mechanisms. The controller's choice of learner, parent, next stage
and backup cadence remains in the existing supervisor policy. No model-facing
behavior changes.

The existing asynchronous task evaluator consumes the same owner follower
without introducing a second checkpoint watermark. Its export selector reads
only the index's current generation, verifies that generation's immutable
selection record and independently read-back private receipt, and then passes
that exact receipt to the existing transfer/evaluation path. If a transfer is
already pending, the evaluator continues using its sealed transfer receipt;
the owner may advance independently, and the next selection happens only after
that evaluation boundary drains.
