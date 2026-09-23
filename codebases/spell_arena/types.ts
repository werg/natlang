export type Actor = { id: string, hp: number, shield: number, x: number, y: number };
export type World = { revision: number, energy: number, actors: Actor[] };
export type Rules = { max_damage: number, max_shield: number, max_move: number, damage_cost: number, shield_cost: number, move_cost: number };
export type Effect = { kind: "damage" | "shield" | "move", target: string, amount: number, x: number, y: number };
export type Plan = { revision: number, effects: Effect[] };
export type Interpretation = { status: "plan" | "clarify" | "impossible", plan: Plan, explanation: string };
export type CastResult = { status: "applied" | "clarify" | "impossible" | "invalid" | "stale" | "insufficient", world: World, cost: number, explanation: string };
