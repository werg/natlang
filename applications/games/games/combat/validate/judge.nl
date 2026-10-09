---
description: Judge whether one fighter's tactic is legal in the arena.
args:
  state: CombatState
  submission: CombatSubmission
returns: Verdict
---
Judge submission.plan, the tactic of the fighter submission.actor in state.

1. The actor is a fighter of state.fighters with hp above 0; otherwise not ok, reason "the actor is a living fighter
   of this arena".
2. plan.move is "left", "stay" or "right"; otherwise not ok, reason "the move is left, stay or right".
3. plan.action is "attack", "guard" or "rest"; otherwise not ok, reason "the action is attack, guard or rest".
4. For an attack, plan.target is the id of a fighter of state.fighters that differs from the actor and has hp above
   0; otherwise not ok, reason "an attack names another living fighter as target".
5. Otherwise ok, reason "a legal tactic".

Return actor as submission.actor, ok and reason.
