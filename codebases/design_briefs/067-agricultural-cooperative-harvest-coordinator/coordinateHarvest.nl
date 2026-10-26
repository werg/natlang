---
description: Rebalance one cooperative harvest round across perishable lots, customer commitments and exact resource capacities.
args: { desk: HarvestDeskState; change: HarvestChange; coordinator: HarvestCoordinator }
returns: HarvestControlProposal
uses: [coordinateHarvest/exactHarvestArithmetic/sumMovementQuantities]
---
Treat `desk.round` as the current round identity. If desk.checkpoint is supplied, use it as the authoritative carried harvest ledger. On first use only, seed the checkpoint from the source-bound ledger fields already present in desk: set priorRevision and revision to desk.revision, and carry desk.priorLots, readiness, commitments, matches, plans and movementReceipts unchanged with unresolved as an empty list. Do not infer any additional receipt or resolution. Bind currentDesk = { round: desk.round, checkpoint, priorLots: checkpoint.priorLots, readiness: checkpoint.readiness, commitments: checkpoint.commitments, matches: checkpoint.matches, plans: checkpoint.plans, movementReceipts: checkpoint.movementReceipts, changedSourceIds: desk.changedSourceIds, revision: checkpoint.revision }. Use currentDesk for every child call, so a stale duplicated projection cannot override saved state. Preserve previous lot estimates and each harvest, storage and delivery receipt. Interpret only the supplied change; a grower forecast is not committed supply.

First Bind delta = `readFieldChange(currentDesk, change)`  as the lot-scoped delta. If no current source identifies the lot and estimate version, return a blocked proposal with the prior assignments untouched.

Bind readiness = `refreshReadiness(currentDesk, delta)`. Keep exact variety identity, buyer grade, excluded quantity and readiness window separate. If quality or maturity evidence conflicts, do not make that lot allocatable; retain the old estimate and ask for the specific discriminating assessment.

Bind commitments = `reconcileCommitments(currentDesk, delta, readiness, change)` to bind each demand to its customer, quality, quantity, window and explicit substitution permissions. A customer rejection closes that substitution path and leaves the original unmet demand visible. Never replace it with a convenient alternate order.

Bind allocation = `allocateEligibleLots(currentDesk, readiness, commitments)` only for edges that satisfy the complete lot-to-commitment suitability relation. Bind every rejected or ambiguous edge with its missing quality or identity fact. Bind resourcePlan = `fitHarvestResources(currentDesk, delta, allocation)` and feed its accepted flow into resource planning; require both equipment and a qualified operator, plus non-overlapping harvest, cold-storage and transport windows. Do not schedule beyond exact capacity.

If the change contains actual yield or movement receipts, bind movementTotals = `sumMovementQuantities(change.physicalReceipts)` with the exact arithmetic mechanism, then bind movement = `reconcilePhysicalMovement(currentDesk, change.physicalReceipts, allocation, movementTotals)`. Keep harvested, stored and delivered amounts in separate totals. Compare them through the exact conservation checker; unexplained loss or over-delivery blocks fulfillment status and leaves the last accepted commitment intact.

Return `HarvestControlProposal` for `coordinator.id`. Include the changed lot's prior and revised estimate, affected commitment IDs, new feasible allocation edges, resource holds, unmet demand, and an append-only `resumeState` carrying lot versions and physical receipts. Mark it ready only if every allocated edge is supported and every capacity and quantity check passes. A ready result remains a review proposal, not a committed delivery.
