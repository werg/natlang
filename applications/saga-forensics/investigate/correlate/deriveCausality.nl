---
description: Create witnessed directed links from explicit execution, retry and compensation relations.
args:
  index: EvidenceIndex
  identity: IdentityPartition
  timeline: Timeline
returns: CausalEdge[]
---
1. Initialize an empty edge set keyed by beforeId, afterId and kind. Read explicit references in the atoms.
2. For each committed receipt naming an attempt, add an execution edge from that attempt to the named effect.
   For a response reporting an earlier effect, preserve its retry/report relation without a new execution effect.
3. For each source-described retry, connect the original attempt to its retry with kind retry. A same-key pair
   classified independent supplies no retry edge even if its timestamps are close.
4. Connect a refund to its referenced payment and cause through compensation edges in the causal direction:
   payment or cancellation precedes refund. Keep an unresolved attempt ID when no effect is established.
5. Connect a signed handover to its consumed reservation through a custody edge from reservation to handover.
   Printed dispatch manifests do not establish this edge without the source's custody evidence.
6. Add policy-boundary edges only for explicit contractual ordering dependencies; simple temporal order is not
   itself a cause. Retain witnesses for both the rule and its application when the edge depends on a rule.
7. Deduplicate identical edges by combining witnesses. Do not sort edge endpoints: their direction is semantic.
8. Return the edge set. The caller checks known-time contradictions using timeline; this stage does not resolve
   conflicting source claims by deleting whichever edge is inconvenient for the proposed repair.
