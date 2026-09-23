---
description: Interpret a spoken spell, then commit only an exactly valid plan
  against the observed arena revision.
args:
  utterance: string
  observation: World
  rules: Rules
returns: CastResult
---
function cast(utterance, observation, rules) -> CastResult
  meaning = interpret(utterance, observation, rules)
  return resolve(observation, rules, meaning)
