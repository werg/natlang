---
description: Compare actual isolated replacement previews and consume a source-governed selection.
args:
  plan: OrderPlan
  previews: OrderPreview[]
  beforeBoard: BoardSnapshot
  source: TurnSource
returns: OrderResolution
---
1. Join actual previews to declared candidates by candidateId. Require beforeRevision to equal
   beforeBoard.revision. Missing or unresolved previews are missing evidence, not losing alternatives.
2. For every completed pair call compareConsequenceProjection(first, second, beforeBoard, source).
   Append its actual OrderDifference with false and null distinct. Equal eventual counters cannot
   erase different source capacities or quantities inspected by a trigger.
3. For one complete legal candidate bind sole-legal-order with the plan's source witnesses. Otherwise
   call bindSubmittedChoice(plan, previews, differences, source). Consume its selected identity/basis.
4. A stable presentation choice is allowed only after actual differences establish equivalence.
   Any consequential difference requires a source-backed submission. Unknown comparison stays unresolved.
5. Check that selection names an actual complete candidate and cited choice IDs belong to the plan's
   choice clauses. These exact domain checks validate the child's source interpretation, not replace it.
6. Return plan, actual previews, differences, choice and unresolved findings. This stage changes no
   board or capacity. Its parent consumes the selected preview or returns an uncommitted rewrite.
