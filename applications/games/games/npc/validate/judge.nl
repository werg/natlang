---
description: Judge whether an NPC's plan is legal for what it holds.
args:
  observation: NpcObservation
  plan: NpcPlan
returns: Verdict
---
Judge plan, the plan of the NPC observation.actor.

1. plan.say is a string (it may be empty); otherwise not ok, reason "the plan has a reply".
2. plan.action is "none", "give" or "promise"; otherwise not ok, reason "the action is none, give or promise".
3. For "give": item and target are strings that start with a letter and continue with letters, digits, "_" or "-"
   (reason "a give names an item and a recipient"), and observation.inventory[item] is at least 1 (reason "a give
   names an item the NPC holds").
4. For "promise": target is such a name and detail is a string with a visible character in it; otherwise reason "a
   promise names a recipient and a concrete detail".
5. Otherwise ok, reason "a legal plan".

When a step fails the verdict is not ok with that step's reason. Return actor as observation.actor, ok and reason.
