import { interpret } from "./cast/interpret";
import { resolve } from "./cast/resolve";
---
description: Interpret a spoken spell, then commit only an exactly valid plan against the observed arena revision.
args:
  utterance: Text
  observation: World
  rules: Rules
returns: CastResult
types:
  Actor: '{ id: Text, hp: Num, shield: Num, x: Num, y: Num }'
  World: '{ revision: Num, energy: Num, actors: Actor[] }'
  Rules: '{ max_damage: Num, max_shield: Num, max_move: Num, damage_cost: Num, shield_cost: Num, move_cost: Num }'
  Effect: '{ kind: "damage" | "shield" | "move", target: Text, amount: Num, x: Num, y: Num }'
  Plan: '{ revision: Num, effects: Effect[] }'
  Interpretation: '{ status: "plan" | "clarify" | "impossible", plan: Plan, explanation: Text }'
  CastResult: '{ status: "applied" | "clarify" | "impossible" | "invalid" | "stale" | "insufficient", world: World, cost: Num, explanation: Text }'
---
function cast(utterance, observation, rules) -> CastResult
  meaning = interpret(utterance, observation, rules)
  return resolve(observation, rules, meaning)
