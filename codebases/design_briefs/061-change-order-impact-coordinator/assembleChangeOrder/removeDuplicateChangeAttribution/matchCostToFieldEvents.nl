---
args: { item: ChangeCostItem, events: WorkEvent[] }
returns: CostEventMatch
---
1. Match only events with the exact item.workPackageId and a contemporaneous evidenceRef that supports the described work; a shared overlapGroup alone does not prove the same work occurred.
2. Bind quantityRateBasis to the supplied quantity and rate without converting incomparable units or rates. Keep ambiguous units unresolved.
3. If no event supports the cost item, return disposition "unresolved" with no accepted event IDs. If several items claim the same evidenced work, return "duplicate-candidate" for the overlapping candidates so the parent can select at most one supported attribution.
4. Return cost ID, package, matching event IDs, overlap group, quantity-rate basis, evidenceMatch, disposition, and reasons.
