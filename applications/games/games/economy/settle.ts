import type { EconomyState, Settings, Settlement, Submission } from "../../types.js";
/**
 * Settlement of a tick's intents in order. Pluggable hot path ("settle"): `settle/policy.nl` by default, with
 * settings 'crisp' `settle/reference.ts`. problem is the check that rejected an earlier settlement ('' for none).
 */
import policy from './settle/policy.nl';
import reference from './settle/reference.js';

export default async function settle(state: EconomyState, ordered: Submission[], settings: Settings, problem: string): Promise<Settlement> {
  if (settings.settle === 'crisp') return reference(state.merchants, ordered);
  return policy(state.merchants, ordered, problem);
}
