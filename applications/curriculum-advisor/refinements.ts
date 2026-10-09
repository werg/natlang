/** Crisp checkers for the refinement predicates of types.ts; checks that need the call's arguments are in index.ts. */
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const sentenceCount = (text: string): number => text.split(/(?<=[.!?])\s+/).filter(part => part.trim()).length;

export const refinements: Record<string, (value: any) => boolean | undefined> = {
  'a judgment whose verdict is equivalent, different or unsure, whose reason is one sentence, and whose evidence holds at least one non-empty quote unless the verdict is unsure': value => {
    if (!isRecord(value) || !['equivalent', 'different', 'unsure'].includes(value.verdict as string) || typeof value.reason !== 'string' || !Array.isArray(value.evidence)) return false;
    if (sentenceCount(value.reason) !== 1) return false;
    const quotes = value.evidence.every(item => isRecord(item) && typeof item.quote === 'string' && item.quote.trim().length > 0);
    return quotes && (value.verdict === 'unsure' || value.evidence.length > 0);
  },
  'a proposal whose batch lines each have a positive whole count and a rationale, and whose note is a string': value =>
    isRecord(value) && typeof value.note === 'string' && Array.isArray(value.batch) && value.batch.every(line => isRecord(line) &&
      Number.isSafeInteger(line.count) && (line.count as number) > 0 && typeof line.rationale === 'string' && line.rationale.trim().length > 0),
};
