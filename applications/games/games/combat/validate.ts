/**
 * Tactic validation. Pluggable hot path ("validate"): the crisp rules with settings 'crisp', otherwise the
 * natural-language judge, which states the same rules.
 */
import judge from './validate/judge.nl';
import rules from './validate/rules.js';
import type { CombatState, CombatSubmission, Settings, Verdict } from '../../types.js';

export default async function validate(state: CombatState, submission: CombatSubmission, settings: Settings): Promise<Verdict> {
  if (settings.validate === 'crisp') return rules(state, submission);
  return judge(state, submission);
}
