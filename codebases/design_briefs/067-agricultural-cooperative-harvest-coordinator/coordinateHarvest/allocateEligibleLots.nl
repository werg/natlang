---
description: Construct a provenance-bound lot-to-order eligibility graph and conserve accepted supply.
args: { desk: HarvestDeskState; readiness: HarvestReadiness[]; commitments: HarvestCommitment[] }
returns: LotAllocation
---
Call `buildLotEligibilityEdges(desk, readiness, commitments)` and bind every candidate edge with its exact lot, commitment, source evidence, and exclusions. Call `conserveLotSupply(edges, readiness, commitments)` to enforce quantity limits across the graph. Never allocate a lot twice or exceed usable quantity. Return rejected edges with the precise variety, grade, timing, or evidence mismatch; preserve each commitment's unmatched balance as unmet demand.
