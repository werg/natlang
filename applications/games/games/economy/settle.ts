import type { EconomyState, Settings, Settlement, Submission } from "../../types.js";
/**
 * Settlement of a tick's intents in order. Pluggable hot path ("settle"): `settle/policy.nl` by default, with
 * settings 'crisp' `settle/reference.ts`, and 'shadow' runs both and records whether they agree (see `pluggable`). problem is the check that rejected an earlier settlement ('' for none).
 */
import { pluggable } from '@natlang/node';
import policy from './settle/policy.nl';
import reference from './settle/reference.js';

export default async function settle(state: EconomyState, ordered: Submission[], settings: Settings, problem: string): Promise<Settlement> {
  return pluggable({ crisp: () => reference(state.merchants, ordered), nl: () => policy(state.merchants, ordered, problem) },
    settings.settle, { name: 'economy.settle' })();
}
