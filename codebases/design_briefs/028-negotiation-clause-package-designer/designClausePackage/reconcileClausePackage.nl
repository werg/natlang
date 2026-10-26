---
args: { prior: ClausePackageCheckpoint|null, request: NegotiationPackageRequest, vault: ClauseTermVault, positions: NegotiationPositionMap, graph: ClauseEffectGraph, packages: ClausePackageSet|null, redlines: RedlineReviewSet|null, approval: NegotiationGuardrail, offer: ClauseOfferPlan|null }
returns: ClausePackageCheckpoint
---
Preserve each party mandate and clause graph edition. A global defined-term change reopens only dependent clauses. Keep every conditional acceptance attached to its prerequisite and exact clause hash. Bind approval only to the exact rendered and redlined package bytes; a changed hash makes the former approval stale. Record offer circulation only from a dispatch receipt matching the approved hash and intended recipients. Duplicate evidence replay is a no-op.
