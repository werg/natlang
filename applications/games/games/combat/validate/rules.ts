import type { CombatState, CombatSubmission, Verdict } from '../../../types.js';

/** The crisp rules of a tactic: known step and action; an attack names another fighter who is alive. */
export default function rules(state: CombatState, submission: CombatSubmission): Verdict {
  const { actor, plan } = submission;
  const verdict = (ok: boolean, reason: string): Verdict => ({ actor, ok, reason });
  const self = state.fighters.find(row => row.id === actor);
  if (!self || self.hp <= 0) return verdict(false, 'the actor is a living fighter of this arena');
  if (!['left', 'stay', 'right'].includes(plan?.move)) return verdict(false, 'the move is left, stay or right');
  if (!['attack', 'guard', 'rest'].includes(plan.action)) return verdict(false, 'the action is attack, guard or rest');
  if (plan.action === 'attack') {
    const target = state.fighters.find(row => row.id === plan.target);
    if (!target || target.id === actor || target.hp <= 0) return verdict(false, 'an attack names another living fighter as target');
  }
  return verdict(true, 'a legal tactic');
}
