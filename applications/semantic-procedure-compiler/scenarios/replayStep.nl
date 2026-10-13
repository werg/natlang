---
description: Consume one scenario event and apply only returned generated-rule decisions to typed state.
args:
  state: ScenarioFrontier
returns: ScenarioFrontier
---
1. Select the next original event by explicit time, preserving simultaneous alternatives if the
   source history does not order them. Remove this event ID from a copied finite frontier exactly once.
2. Find generated modules/branches responsible for that event's actor/action. Read the actual emitted
   body together with its ModulePlan; an emitted unconditional effect cannot be hidden by a correct model.
3. For each relevant rule call judgeGuard(event, atoms, branch, model, generatedModule). Consume the
   actual returned value, satisfied/missing/waived premises, effects and localDecision.
4. Apply only source-licensed returned effects. Copy original StateAtom ownership and witnesses;
   update the specified keys for their specified actor at this event time. Preserve unrelated fields
   and unknown facts, and do not create another actor's readiness from an object custody transfer.
5. A false guard records the denied action and leaves permission-dependent effects unapplied. A null
   guard preserves unknown permission and names the missing source fact; future events remain processable.
6. Record ScenarioTrace with original event identity, before atoms, actual decisions, after atoms,
   observable outcome and exact witnesses. No trace receipt is created before the actual guard returns.
7. Return copied remainingEventIds, atoms, traces and unresolved with scenario/program/model retained.
   The next step consumes these actual state updates rather than reinterpreting the entire history anew.
