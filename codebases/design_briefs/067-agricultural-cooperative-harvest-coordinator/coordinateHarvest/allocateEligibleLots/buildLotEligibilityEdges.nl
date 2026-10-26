---
description: Check each lot against customer quality, time window, substitutions and current field evidence.
args: { desk: HarvestDeskState; readiness: HarvestReadiness[]; commitments: HarvestCommitment[] }
returns: LotSuitability[]
---
For each changed lot and affected commitment, compare recorded variety identity and grade evidence with customer required quality. Compare readiness and delivery windows only when both have current source support. A substitution may be considered only if the commitment explicitly lists it and the customer has not rejected it. Mark eligibility false whenever one of those dimensions is unknown or contradictory; state which source evidence is missing. Preserve exact lots even when crop names match. Bind a `proposedQuantity` only when current source data states the exact amount this lot can serve for this commitment in the same unit; otherwise leave it absent for the allocation proposal to resolve from explicit records.
