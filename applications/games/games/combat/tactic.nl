---
description: A fighter's tactic for the round. Choose a step and an action from what the fighter sees.
args:
  observation: CombatObservation
returns: CombatPlan
---
Choose the tactic of the fighter observation.self for round observation.round. Everything it sees is in observation:
its own cell x (cells run 0 to observation.width - 1), health hp and cooldown, and the cell and health of each
opponent in observation.others.

1. move is "left" (x decreases by 1), "right" (x increases by 1) or "stay". A fighter attacks an opponent whose cell
   is at most 1 away from its own cell after every fighter has moved, so compare the cells you expect after moving.
2. action is "attack", "guard" or "rest". Attack only while observation.self.cooldown is 0, naming as target the id of
   an opponent in observation.others whose hp is above 0. Guard halves the damage the fighter takes this round. Rest
   leaves the fighter as it is.
3. Weigh health, distance and cooldown, and return { move, action } with target set for an attack.
