/**
 * Tactic validation. Pluggable hot path ("validate"): the crisp rules with settings 'crisp', otherwise the
 * natural-language judge, which states the same rules; 'shadow' runs both and records whether they agree on ok.
 */
import { pluggable } from '@natlang/node';
import judge from './validate/judge.nl';
import rules from './validate/rules.js';
import type { CombatState, CombatSubmission, Settings, Verdict } from '../../types.js';

export default async function validate(state: CombatState, submission: CombatSubmission, settings: Settings): Promise<Verdict> {
  return pluggable({ crisp: () => rules(state, submission), nl: () => judge(state, submission) },
    settings.validate, { name: 'combat.validate', same: (crisp, nl) => crisp.ok === nl.ok })();
}
