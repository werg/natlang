import type { Settings, Submission, Verdict } from "../../types.js";
/**
 * Intent validation. Pluggable hot path ("validate"): the crisp rules by default settings 'crisp', otherwise the
 * natural-language judge, which states the same rules.
 */
import judge from './validate/judge.nl';
import rules from './validate/rules.js';

export default async function validate(known: string[], submission: Submission, settings: Settings): Promise<Verdict> {
  if (settings.validate === 'crisp') return rules(known, submission);
  return judge(known, submission);
}
