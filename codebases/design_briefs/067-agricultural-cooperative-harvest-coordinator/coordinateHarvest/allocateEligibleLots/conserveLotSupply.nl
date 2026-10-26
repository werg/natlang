---
description: Enforce exact supply and demand conservation across the accepted eligibility graph.
args: { edges: LotSuitability[]; readiness: HarvestReadiness[]; commitments: HarvestCommitment[] }
returns: LotAllocation
---
Use exact decimal arithmetic for quantities in matching units. For each lot, total assigned to accepted commitments must not exceed usable quantity; for each commitment, total assigned must not exceed open quantity. Preserve unmatched demand rather than rounding it away. Return a blocked edge when unit conversion is not evidenced or arithmetic cannot be exact.
