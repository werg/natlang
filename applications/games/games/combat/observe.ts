import type { CombatObservation, CombatState } from "../../types.js";
/** What one fighter sees: itself and the cell and health of every other fighter. The information boundary. */
export default function observe(state: CombatState, actor: string): CombatObservation {
  const self = state.fighters.find(row => row.id === actor);
  if (!self) throw new Error('unknown fighter');
  return { actor, round: state.round, width: state.width, self: structuredClone(self),
    others: state.fighters.filter(row => row.id !== actor).map(row => ({ id: row.id, x: row.x, hp: row.hp })) };
}
