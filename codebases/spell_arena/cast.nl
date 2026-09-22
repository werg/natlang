import { interpret } from "./cast/interpret";
import { resolve } from "./cast/resolve";
---
description: Interpret a spoken spell, then commit only an exactly valid plan against the observed arena revision.
args:
  utterance: string
  observation: World
  rules: Rules
returns: CastResult
types:
  Actor: '{ id: string, hp: number, shield: number, x: number, y: number }'
  World: '{ revision: number, energy: number, actors: Actor[] }'
  Rules: '{ max_damage: number, max_shield: number, max_move: number, damage_cost: number, shield_cost: number, move_cost: number }'
  Effect: '{ kind: "damage" | "shield" | "move", target: string, amount: number, x: number, y: number }'
  Plan: '{ revision: number, effects: Effect[] }'
  Interpretation: '{ status: "plan" | "clarify" | "impossible", plan: Plan, explanation: string }'
  CastResult: '{ status: "applied" | "clarify" | "impossible" | "invalid" | "stale" | "insufficient", world: World, cost: number, explanation: string }'
---
function cast(utterance, observation, rules) -> CastResult
  meaning = interpret(utterance, observation, rules)
  return resolve(observation, rules, meaning)
