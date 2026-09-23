---
description: Translate a player's spell into an ordered, typed effect plan or ask for clarification.
args:
  utterance: string
  observation: World
  rules: Rules
returns: Interpretation
---
Read the player's utterance as a request within `rules` and the visible `observation`.
For an unambiguous lawful request, write status "plan" and an ordered effect list with exact actor IDs.
Use "clarify" when the target or requested action has more than one plausible reading.
Use "impossible" when the utterance asks for an action outside damage, shielding and movement.
For either non-plan status, use an empty effect list. Set plan.revision to observation.revision.
An effect's unused amount or coordinates are zero. Do not invent actors, rules or hidden state.
Explain the interpretation briefly. The exact resolver will still check costs and world validity.
