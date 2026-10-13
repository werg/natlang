---
description: Compute cancellation entitlements across handover boundaries and shared stock ownership.
args:
  request: InvestigationRequest
  index: EvidenceIndex
  correlation: Correlation
  ledger: EffectLedger
returns: ObligationGraph
---
1. Read the pool opening balance, complete-snapshot scope and signed custody assertions. Sum distinct committed
   reservations once. Compute inUse from signed handovers, currentReserved from unconsumed reserved units,
   and currentFree = initialStock - inUse - currentReserved. Keep maintenance and other nontarget reservations.
2. For each target order, join its cancellation, committed reservation, rental payment, deposit and refunds using
   the correlation identities and ledger references. A shared key alone is not a join condition.
3. Compare the normalized cancellation and signed handover times under the cancellation contract. A printed
   manifest does not transfer custody. Unknown order remains unresolved if it changes the entitlement.
4. Interpret the rollback boundary: determine which rental shares are void and which remain payable, then use
   exact arithmetic to compute totalRefundEntitlement and releaseQuantity for unconsumed reserved units.
5. Keep deposits separate from rental shares. Protect a deposit whenever the source says custody still exists;
   a cancellation is not evidence of a kit return. Preserve nontarget reservation effects independently.
6. Sum confirmed refunds with the same referenced payment and cancellation cause. Record uncertain refunds
   separately by ledgerId; subtract neither zero nor their full amount before an observation branch resolves them.
7. Create one Obligation per supported target cancellation, using the source cancellation ID as obligationId.
   Include payment/reservation IDs, amounts, protected effects and witnesses establishing boundary and price.
8. Compute confirmedCash as committed rental plus deposit charges minus confirmed refunds. Collect all bounded
   uncertain refund ledger IDs for later world accounting. Do not count rejected or replayed attempts again.
9. Check conservation, entitlement bounds and completeness. Negative quantities, refund exceeding payment,
   missing prices or absent required scope become unresolved with their source references.
10. Return the obligation graph, pool state, confirmed cash and witnesses. This stage decides what must be
    repaired; action-key selection, observations and plan ordering belong to the downstream policy branch.
