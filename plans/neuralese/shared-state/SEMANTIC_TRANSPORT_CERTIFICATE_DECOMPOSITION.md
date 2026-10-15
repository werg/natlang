# Learned semantic transport runtime certificate

The shared adaptation stage must exercise its learned message channel before
publishing a usable adapter. A task score alone is insufficient: a task can
finish without producing any Neuralese. This certificate is a runtime parity
check, not evidence that the learned message improves task success.

1. Select one nonempty admitted payload for each participant in the declared
   semantic curriculum. Keep the complete payload and its source provenance.
2. Bind the existing serving engine to the live, exact exported semantic
   adapter and participant. Reuse the normal serving binder; load no second
   backbone or VAE. Verify actual embedding/model identity before use.
3. Produce each payload with the real native writer, then serialize and decode
   its compact message through the existing artifact format. Require a learned
   code and no raw KV/convolution layer tensors on the wire.
4. For every directed pair of distinct participants, consume that message with
   the real serving consumer. Independently apply the declared semantic decoder
   and positional state splice, then compare logits, next-token logits and all
   attention/convolution cache states. An explicit tolerance belongs to the
   caller; do not widen it after a failure.
5. Record exact identities, adapter digest, artifact IDs, directions and errors.
   Preserve all model modes and RNG state. A failure prevents a passing runtime
   certificate; it never becomes an unmeasured task-quality assertion.

Selection and publication policy stay in the semantic curriculum/stage.
`eval/native_semantic_runtime.py` is crisp because it verifies exact transport,
serialization and numerical agreement against the shared decoder/splice
reference. Existing engines, stores and semantic decoding remain the only
implementations of those mechanisms.
