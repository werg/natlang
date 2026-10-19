# Reuse an unchanged memory-check numerical reference

The existing supervisor policy selects complete source-authenticated capacity cases. This change does not select cases or change that policy. It removes repeated exact verification work within one owned, sequential restore boundary.

1. Authenticate each requested case and its numerical reference using the existing source, row, index, parent, execution and runtime checks.
2. Before measuring the reference, capture the exact declared invocation and the current versioned model, optimizer, backend and RNG identities.
3. If this same invocation already passed during this one call and every captured identity still matches, carry that original numerical-reference result and identify its original request explicitly.
4. Otherwise run the existing two-backward numerical-reference comparison. Retain only a successful result whose boundary identities are unchanged afterward. Do not retain model, optimizer, checkpoint or gradient tensors in the reuse record.
5. Always execute every requested complete capacity case. Keep the existing per-case certificates and state-preservation checks. Discard the reference reuse table when this call ends.

The identity verifier, call-local table, exact evidence links and adversarial regression tests are crisp mechanisms: they perform equality and durability checks, not model-facing or discretionary policy. There is no persistent cross-owner cache, new readiness claim, changed tolerance, or inherited capacity-case execution.
