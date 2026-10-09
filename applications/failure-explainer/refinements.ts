/**
 * Crisp checkers for the refinement predicates of types.ts. Each key is a predicate exactly as `Is<T, "...">` states it.
 * Checks that need the call's arguments (the tag is one of the given tags, a quote occurs in the card) are in index.ts.
 */
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const sentenceCount = (text: string): number => text.split(/(?<=[.!?])\s+/).filter(part => part.trim()).length;
const SNAKE = /^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$/;

export const refinements: Record<string, (value: any) => boolean | undefined> = {
  'an explanation whose proposed_tag is a lowercase snake_case name exactly when its tag is new, whose why is one to three sentences, and whose evidence holds at least one non-empty quote': value => {
    if (!isRecord(value) || typeof value.tag !== 'string' || typeof value.why !== 'string' || !Array.isArray(value.evidence)) return false;
    const named = typeof value.proposed_tag === 'string' && SNAKE.test(value.proposed_tag);
    if (value.tag === 'new' ? !named : value.proposed_tag !== null) return false;
    const count = sentenceCount(value.why);
    return count >= 1 && count <= 3 && value.evidence.length > 0 &&
      value.evidence.every(item => isRecord(item) && typeof item.quote === 'string' && item.quote.trim().length > 0);
  },
  'proposals each of whose match_prefix is a non-empty prefix of its reason': value =>
    isRecord(value) && Array.isArray(value.proposals) && value.proposals.every(item => isRecord(item) &&
      typeof item.reason === 'string' && typeof item.match_prefix === 'string' && item.match_prefix.length > 0 && item.reason.startsWith(item.match_prefix)),
  'an explanation with gate_unchanged true, a pattern of one to three sentences, at least one candidate cause and at least one next check': value => {
    if (!isRecord(value) || value.gate_unchanged !== true || typeof value.pattern !== 'string') return false;
    const count = sentenceCount(value.pattern);
    return count >= 1 && count <= 3 && Array.isArray(value.candidate_causes) && value.candidate_causes.length > 0 &&
      Array.isArray(value.next_checks) && value.next_checks.length > 0;
  },
};
