export type Fighter = { id: string, x: number, hp: number, cooldown: number };
export type Opponent = { id: string, x: number, hp: number };
export type CombatObservation = { actor: string, round: number, width: number,
  self: Fighter, others: Opponent[] };
export type CombatPlan = { move: string, action: string, target?: string };
export type CombatReceipt = { status: string, actor: string, round: number };
