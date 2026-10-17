# Restore the declared foundation of the requested supervised stage

The recipe policy is unchanged: initial foundation activation must cover the
entire current admitted communication cohort. Later admitted communication stays
mandatory without changing the original finite foundation. A policy operation
and each supervised stage retain separate schedule metadata under one current
optimizer/RNG history.

1. The existing collect_sft_progress call authenticates/selects the checkpoint's
   effective stage frames. Reuse that already materialized result; do not decode
   or deep-copy the checkpoint again.
2. At the exact initial-cohort verifier, look up only the requested stage key
   (including its existing native-graph-ce default). Read its saved foundation
   declaration. A different stage's flag or the current policy's top-level flag
   cannot activate this exemption.
3. Keep the existing exact cohort verifier and subsequent source, optimizer,
   RNG, exposure and readiness checks unchanged. Crisp code performs the exact
   stage-key lookup and existing boolean handoff; it chooses no new curriculum.

Focused real tiny-model tests cover policy-top false with carried declaration,
ordinary resume, the default stage, missing/wrong-stage declarations and a fresh
activation with an incomplete cohort. Positive resumes retain the saved finite
cohort, consume only actual remaining rows and never grant task readiness.
