/**
 * Resolution of a round: simultaneous movement, strikes, wounds and cooldown timers. Pluggable hot path ("resolve"):
 * `resolve/policy.nl` by default, with settings 'crisp' `resolve/reference.ts`. problem is the check that rejected an
 * earlier resolution of this round ('' for none).
 */
import policy from './resolve/policy.nl';
import reference from './resolve/reference.js';
import type { CombatResolution, CombatState, CombatSubmission, Settings } from '../../types.js';

export default async function resolve(state: CombatState, submissions: CombatSubmission[], settings: Settings, problem: string): Promise<CombatResolution> {
  if (settings.resolve === 'crisp') return reference(state.fighters, state.width, submissions);
  return policy(state.fighters, state.width, submissions, problem);
}
