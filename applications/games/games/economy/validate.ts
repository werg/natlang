import type { Settings, Submission, Verdict } from "../../types.js";
/**
 * Intent validation. Pluggable hot path ("validate"): the crisp rules by default settings 'crisp', otherwise the
 * natural-language judge, which states the same rules; 'shadow' runs both and records whether they agree on ok.
 */
import { pluggable } from '@natlang/node';
import judge from './validate/judge.nl';
import rules from './validate/rules.js';

export default async function validate(known: string[], submission: Submission, settings: Settings): Promise<Verdict> {
  return pluggable({ crisp: () => rules(known, submission), nl: () => judge(known, submission) },
    settings.validate, { name: 'economy.validate', same: (crisp, nl) => crisp.ok === nl.ok })();
}
