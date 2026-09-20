export type Fighter = { id: Text, x: Num, hp: Num, cooldown: Num };
export type Opponent = { id: Text, x: Num, hp: Num };
export type CombatObservation = { actor: Text, round: Num, width: Num,
  self: Fighter, others: Opponent[] };
export type CombatPlan = { move: Text, action: Text, target?: Text };
export type CombatReceipt = { status: Text, actor: Text, round: Num };
