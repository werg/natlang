---
description: Enforce exact supply and demand conservation across the accepted eligibility graph.
args: { edges: LotSuitability[]; readiness: HarvestReadiness[]; commitments: HarvestCommitment[] }
returns: LotAllocation
uses: [coordinateHarvest/exactHarvestArithmetic/verifyQuantityLimits]
---
Bind candidate quantities from eligible edges only when a source-backed exact amount is stated; leave `proposedQuantity` absent when the amount would require an unsupported conversion or speculative split. Build one `lot-supply` check per lot and one `commitment-demand` check per commitment, with exact available quantities, every proposed edge amount, and the evidenceRefs that bind each limit. Call `verifyQuantityLimits(checks)` for exact summation and unit/limit validation. Accept only edges whose two quantity checks pass; mark every flow on an over-allocated lot or commitment blocked, preserve all unmet demand, and attach the exact checks to the returned allocation.
