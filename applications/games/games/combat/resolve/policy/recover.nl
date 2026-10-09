---
description: Phase 4 of a combat round. Advance the cooldown timers; a fighter that struck starts its cooldown.
args:
  fighters: Fighter[]
  hits: Hit[]
returns: Fighter[]
---
Advance the cooldown timers of the fighters by one round.

For each fighter in fighters, in order, produce a copy with the same id, x and hp and with cooldown set by:
1. The fighter's id is the attacker of some hit in hits: cooldown becomes 1.
2. Otherwise cooldown becomes the fighter's cooldown minus 1, raised to 0 if it is below 0.

Return the copies in the order of fighters.
