/**
 * Resolution of a round: simultaneous movement, strikes, wounds and cooldown timers. Pluggable hot path ("resolve"):
 * `resolve/policy.nl` by default, with settings 'crisp' `resolve/reference.ts`, and 'shadow' runs both and records whether they agree. problem is the check that rejected an
 * earlier resolution of this round ('' for none).
 */
import { pluggable } from '@natlang/node';
import policy from './resolve/policy.nl';
import reference from './resolve/reference.js';
import type { CombatResolution, CombatState, CombatSubmission, Settings } from '../../types.js';

export default async function resolve(state: CombatState, submissions: CombatSubmission[], settings: Settings, problem: string): Promise<CombatResolution> {
  return pluggable({ crisp: () => reference(state.fighters, state.width, submissions), nl: () => policy(state.fighters, state.width, submissions, problem) },
    settings.resolve, { name: 'combat.resolve' })();
}
