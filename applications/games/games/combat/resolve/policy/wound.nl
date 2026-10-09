---
description: Phase 3 of a combat round. Subtract the damage of all hits from the health of their targets, to a floor of 0.
args:
  fighters: Fighter[]
  hits: Hit[]
returns: Fighter[]
---
Wound the fighters. All hits apply together to the health the fighters hold now.

For each fighter in fighters, in order, produce a copy with the same id, x and cooldown and with hp set by:
1. taken = the sum of hit.amount over the hits whose target is the fighter's id (0 when there is none), added up in eval.
2. hp becomes the fighter's hp minus taken, raised to 0 if it is below 0.

Return the copies in the order of fighters.
