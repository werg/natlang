---
description: Derive simultaneous effect outcomes from one shared before-state and reconcile their actual proposals.
args:
  rewritten: RewrittenBundle
  beforeBoard: BoardSnapshot
  source: TurnSource
returns: BatchDelta
---
1. Require the actual rewritten bundle's beforeRevision to identify beforeBoard. Preserve its current
   parts and replacement decisions; original announced parts are provenance, not the resolving effects.
2. For each final part call judgeOutcome(part, beforeBoard, source). Give every sibling the same board.
   Collect the actual returned damage/growth quantities, object/resource proposals and unresolved fields.
3. Call reconcileDeltas with this whole outcome collection, beforeBoard, source and bundle identity.
   Its semantic join determines accumulation, simultaneous departures and any conflicting proposals;
   a sequential application of sibling proposed snapshots is not this join's meaning.
4. Consume its merged objects, resources and departures. Add the actual rewritten consumedTokenIds
   once; a shared token reservation is not independently spent again for each part outcome.
5. Return BatchDelta with the original beforeRevision and every actual PartOutcome, merged changes,
   departures, token consumption and exact witnesses. Any decisive unresolved outcome prevents a
   complete batch commit; the owning window decides no partial state before source repair.
