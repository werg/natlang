import type { NpcObservation, NpcPlan, Verdict } from '../../../types.js';

const isName = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(value);

/** The crisp rules of a plan: a reply and one action the inventory and the names allow. */
export default function rules(observation: NpcObservation, plan: NpcPlan): Verdict {
  const verdict = (ok: boolean, reason: string): Verdict => ({ actor: observation.actor, ok, reason });
  if (typeof plan?.say !== 'string') return verdict(false, 'the plan has a reply');
  if (!['none', 'give', 'promise'].includes(plan.action)) return verdict(false, 'the action is none, give or promise');
  if (plan.action === 'give') {
    if (!isName(plan.item) || !isName(plan.target)) return verdict(false, 'a give names an item and a recipient');
    if (!((observation.inventory[plan.item] ?? 0) >= 1)) return verdict(false, 'a give names an item the NPC holds');
  }
  if (plan.action === 'promise' && (!isName(plan.target) || typeof plan.detail !== 'string' || !plan.detail.trim()))
    return verdict(false, 'a promise names a recipient and a concrete detail');
  return verdict(true, 'a legal plan');
}
