# Export the current owned learner's published checkpoints

`exportOwnedPublication(owner, publication, durability)` follows the supervisor's
declared current learner, including recovery to an earlier step. A step from a
discarded trajectory must never hide useful checkpoints from that learner.

1. Bind the current learner's effective command, parent, runtime and publication
   producers before starting its exporter. Use a separate durability journal for
   this owned continuation; do not reset or relabel an older journal.
2. Read only this learner's successful checkpoint-publication status. Verify the
   existing source-bound control, saved parent and producer selection. Never
   decode a second copy of the optimizer to discover its step.
3. Preserve the operator's declared storage reserve before allocating the full
   copy. Copy the actual full source to an independent immutable SHA-named pin. Verify
   that pin through the same publication verifier before offering it to the
   existing private uploader and independent full readback.
4. Retain a pending pin across transport retries. Mark durability only after the
   existing uploader's SHA and size checks succeed. Do not grant training, task
   quality, exposure or readiness credit.
5. Actor evaluations remain separately bound to their actual full parent. The
   publication-only control does not assert that a compact actor exists.

This policy is the operator's natural-language continuation decision. The
Python module implements exact file identity, producer verification and atomic
pinning; it does not select a model, alter training or introduce model prompts.
It reuses `native_checkpoint_control.verify_checkpoint_publication`,
`cursor_checkpoint_export.bound_output_checkpoint` and the existing private
export journal/uploader rather than inventing alternate checkpoint proofs.
