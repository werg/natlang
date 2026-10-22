# Completed scheduling repair through cached configuration

The production recipe chooses scheduling policy. Cached bootstrap evidence is
an optimization of verification; it cannot establish that a historical scheduling
control points to the current final plan.

1. For a completed continuation, compare all present scheduling-control plans
   against the actual current plan before accepting cached metadata.
2. If independent RL is declared, require its control to exist and to carry the
   declared policy. Reject cached reuse on any mismatch.
3. Let the existing completed configurator authenticate and apply its exact
   zero-update repair under the continuously held controller lease. Unknown
   mutations continue to fail the full verifier.

This is a crisp cache-validity check, not new recovery or scheduling policy.
It changes no numerical code, source policy, model, exposure or earned state.
