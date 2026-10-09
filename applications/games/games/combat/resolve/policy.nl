---
description: Resolve one combat round in four phases (movement, strikes, wounds, cooldown timers), all fighters at once. Returns the fighters afterwards, the hits and the damage per target.
args:
  fighters: Fighter[]
  width: number
  submissions: CombatSubmission[]
  problem: string
returns: CombatResolution
---
Resolve the round with the stages in your folder. fighters are the fighters at the start of the round, width is the
arena's width, and submissions are the tactics (a fighter without one acts as nothing). problem is empty, or the check
that rejected an earlier resolution of this round; resolve so that this check holds.

1. moved = move(fighters, submissions, width): every fighter's cell after all movement.
2. hits = strike(moved, submissions): the attacks that land, judged on the cells after movement.
3. damage: for each target that appears in hits, the sum of the amounts of the hits on it; an object from target id to
   that sum, with no entry for a fighter that was not hit.
4. wounded = wound(moved, hits).
5. final = recover(wounded, hits).

Return { fighters: final, hits, damage }. final lists the same fighters in the order of fighters.
