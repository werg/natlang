/**
 * Which facts a consumer may see at all: an exact data rule, not a preference, so it is crisp. The runtime filters by it
 * before `selectGuidanceFacts` runs. `rl/privileged_context.applicable` (training/neuralese/natlang_neuralese) is the
 * verifier of the same rule: the engine re-checks every guidance request with it and the learner re-checks every
 * captured block, so a change here is a change there.
 *
 * - `child_value`: only a child call, and only when every `match` string occurs in its prompt text, or, when the
 *   reference names the inputs it is about (identifier match strings: a quoted value such as `"M1"` or a token with a
 *   digit such as `d81028_evidence_94_0`), when every such identifier occurs: a policy's own child call is matched by
 *   what it is called on, not by the reference's wording. Several children that match with different answers give
 *   none (crispSelection).
 * - `attempt_feedback`, `fit_hindsight`: a call-specific fact (`invocation` set) only for that call; otherwise for any.
 * - every other kind: the root call and learned-optimizer bindings.
 */
import type { GuidanceConsumer, GuidanceFact } from './types.js';

/** A match string naming an input value rather than wording: a quoted value or a token containing a digit. */
export const identifierMatch = (item: string): boolean => /^"[^"\s]+"$/.test(item) || (/^[A-Za-z0-9][\w.:-]*$/.test(item) && /\d/.test(item));

/** Whether a child reference's match strings select this prompt: all of them, or all of its identifier strings. */
export function childMatches(match: readonly string[], text: string): boolean {
  if (!match.length) return false;
  if (match.every(item => text.includes(item))) return true;
  const ids = match.filter(identifierMatch);
  return ids.length > 0 && ids.every(item => text.includes(item));
}

export function visible(fact: GuidanceFact, consumer: GuidanceConsumer): boolean {
  if (fact.kind === 'child_value')
    return consumer.role === 'child' && childMatches(fact.match ?? [], consumer.text);
  if (fact.kind === 'attempt_feedback' || fact.kind === 'fit_hindsight')
    return fact.invocation === undefined || fact.invocation === null || fact.invocation === consumer.name;
  return consumer.role === 'root' || consumer.role === 'binding';
}

/** The facts `consumer` may see, in their given order. */
export function visibleFacts(consumer: GuidanceConsumer, facts: readonly GuidanceFact[]): GuidanceFact[] {
  return facts.filter(fact => visible(fact, consumer));
}
