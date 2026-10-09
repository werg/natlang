---
description: Phase 2 of a combat round. List the attacks that land, judged on the cells after movement and on the health and cooldown from before the round.
args:
  moved: Fighter[]
  submissions: CombatSubmission[]
returns: Hit[]
---
Find the hits. moved holds each fighter at its cell after movement, with the hp and cooldown it had at the start of
the round. Every attack is judged on that same state, so the order of attackers does not matter.

For each submission whose plan.action is "attack", with attacker the fighter of moved whose id is submission.actor and
target the fighter whose id is plan.target, the attack lands when all of these hold:
1. attacker.hp is above 0 and attacker.cooldown is 0.
2. target exists and target.hp is above 0.
3. The distance between attacker.x and target.x is at most 1 (Math.abs in eval).

A landed attack is the hit { attacker: attacker.id, target: target.id, amount }. amount is 1 when the submission of
target has action "guard", and 2 otherwise (also when target has no submission). One attacker lands at most one hit.

Return the hits in the order of submissions.
