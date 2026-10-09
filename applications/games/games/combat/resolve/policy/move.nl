---
description: Phase 1 of a combat round. Every living fighter with a tactic steps one cell or stays, inside the arena; all steps count together.
args:
  fighters: Fighter[]
  submissions: CombatSubmission[]
  width: number
returns: Fighter[]
---
Move the fighters. Every step is decided from the cells at the start of the round, so no fighter's step depends on
another's.

For each fighter in fighters, in order, produce a copy with the same id, hp and cooldown and with x set by:
1. The fighter has no submission in submissions, or hp is 0 or less: x stays.
2. Otherwise step is -1 for move "left", +1 for "right" and 0 for "stay". x becomes x + step, raised to 0 if it is
   below 0 and lowered to width - 1 if it is above that.

Return the copies in the order of fighters.
