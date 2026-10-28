/**
 * Which facts a consumer may see at all: an exact data rule, not a preference, so it is crisp. The runtime filters by it
 * before `selectGuidanceFacts` runs. `rl/privileged_context.applicable` (training/neuralese/natlang_neuralese) is the
 * verifier of the same rule: the engine re-checks every guidance request with it and the learner re-checks every
 * captured block, so a change here is a change there.
 *
 * - `child_value`: only a child call, and only when every `match` string occurs in its prompt text.
 * - `attempt_feedback`, `fit_hindsight`: a call-specific fact (`invocation` set) only for that call; otherwise for any.
 * - every other kind: the root call and learned-optimizer bindings.
 */
import type { GuidanceConsumer, GuidanceFact } from './types.js';

export function visible(fact: GuidanceFact, consumer: GuidanceConsumer): boolean {
  if (fact.kind === 'child_value')
    return consumer.role === 'child' && (fact.match ?? []).every(item => consumer.text.includes(item));
  if (fact.kind === 'attempt_feedback' || fact.kind === 'fit_hindsight')
    return fact.invocation === undefined || fact.invocation === null || fact.invocation === consumer.name;
  return consumer.role === 'root' || consumer.role === 'binding';
}

/** The facts `consumer` may see, in their given order. */
export function visibleFacts(consumer: GuidanceConsumer, facts: readonly GuidanceFact[]): GuidanceFact[] {
  return facts.filter(fact => visible(fact, consumer));
}
