---
description: Bind an actually submitted legal relative-order choice to a completed preview.
args:
  plan: OrderPlan
  previews: OrderPreview[]
  differences: OrderDifference[]
  source: TurnSource
returns: OrderChoiceBinding
---
1. Read source-defined chooser, choice window and scope from actual choice clauses and governing
   rules. Separate submissions from examples, permission to choose and choices about another event.
2. Preserve unresolved previews or differences. A submission cannot certify an unexecuted candidate
   or settle missing knowledge about what an order did.
3. If actual differences are all equivalent, choose a stable presentation identity and return
   equivalent-orders with comparison witnesses. Do not claim that a player submitted this choice.
4. Otherwise interpret each actual relative-order submission against candidate application identities.
   Require chooser and event/lineage scope to match source authority. Resolve physical target references
   from source identity history, without choosing the interpretation with a desirable outcome.
5. Retain candidates satisfying every applicable submitted constraint. Select the actual identity when
   one semantic alternative remains; equivalent residual candidates may use a stable display identity.
   Return submitted-choice, original choice clause IDs and exact supporting witnesses.
6. If a consequential difference remains without an applicable submission, return null selection and
   unresolved with the precise outstanding choice. Contradictory or illegal submissions also remain
   unresolved; numeric advantage or source-file order cannot supply the missing player action.
7. Return the actual binding. The parent selects an already produced preview and never rewrites one
   to fit this result. Source choice identity and generated candidate identity remain separate fields.
