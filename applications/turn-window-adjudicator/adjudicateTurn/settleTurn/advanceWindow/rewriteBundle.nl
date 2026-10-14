---
description: Set a finite source-governed replacement frontier and consume its rewritten parts.
args:
  bundle: EventBundle
  beforeBoard: BoardSnapshot
  source: TurnSource
  protocol: WindowProtocol
returns: RewrittenBundle
---
1. Call resolveOrdering(bundle, beforeBoard, source, protocol). Consume the actual finite OrderPlan,
   source constraints, legal candidates and original choice clauses. Keep a pristine copy of the
   supplied bundle/beforeBoard for all previews; ordering does not change the live turn.
2. If legality is unresolved, return the original bundle with no decisions or consumed tokens and an
   OrderResolution containing the plan, no fabricated previews and a null unresolved choice. Preserve
   the specific missing source premise before any replacement or batch is committed.
3. For each legal candidate create a separate ReplacementState with that candidate's application list,
   the original bundle/beforeBoard, empty decisions/reservations and the same source/protocol. No state
   object or mutable reservation collection is shared between candidates.
4. Invoke the existing applyReplacement.iterateOn on this candidate state, measured by
   remainingApplications.length. Consume each actual result until its list is empty or unresolved
   stops that branch. This is the same semantic replacement child used by ordinary resolution.
5. Store OrderPreview with candidateId parent-bound from this candidate, original beforeRevision,
   actual returned bundle, decisions, reservedTokenIds and unresolved. Exact checks require unchanged
   beforeBoard and original source token statuses. Preview reservations do not spend live tokens.
6. After all finite candidates have been processed, call compareOrderPreviews(plan, previews,
   beforeBoard, source). Consume the actual OrderResolution and its selected candidate identity.
   Do not replace a failed preview with a predicted final part or favor the first completed branch.
7. If comparison or source choice is unresolved, return the original bundle, empty committed decisions
   and consumed tokens, plus the actual OrderResolution. The parent records this uncommitted dispute
   and finishes without resolving a batch or activating a consequence from any preview.
8. Otherwise locate the actual selected preview. Check lineage preservation, nonnegative integral
   quantities and capacity IDs. Return its exact bundle, decisions and reserved tokens as the selected
   RewrittenBundle, with OrderResolution retained. No second semantic replay can change the choice.
9. The caller applies these chosen deltas only after simultaneous outcomes succeed. Hypothetical
   histories remain distinguishable from the selected actual history in the certificate.
