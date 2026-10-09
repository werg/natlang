/**
 * Plan validation. Pluggable hot path ("validate"): the crisp rules with settings 'crisp', otherwise the
 * natural-language judge, which states the same rules.
 */
import judge from './validate/judge.nl';
import rules from './validate/rules.js';
import type { NpcObservation, NpcPlan, Settings, Verdict } from '../../types.js';

export default async function validate(observation: NpcObservation, plan: NpcPlan, settings: Settings): Promise<Verdict> {
  if (settings.validate === 'crisp') return rules(observation, plan);
  return judge(observation, plan);
}
