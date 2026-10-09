/**
 * Crisp checkers for the refinement predicates of types.ts (plans/REFINEMENT_TYPES.md). Each key is a predicate exactly
 * as `Is<T, "...">` states it; a checker decides from the value alone, without a model call. The part of a predicate
 * that needs a judge ("a product area", "addressed to the customer") is not checked here; the instructions state it.
 */
const words = (text: string) => text.trim().split(/\s+/).filter(Boolean);

/** Sentences of a plain text: stretches that end in . ! or ? followed by a space or the end (3.5 is one number, not two sentences). */
export function sentenceCount(text: string): number {
  return text.trim().split(/(?<=[.!?])\s+/).filter(part => /[\p{L}\p{N}]/u.test(part)).length;
}

/** Plain text carries no markdown: no code marks, bold, headings, bullets or links. */
const plain = (text: string) => !/`|\*\*|^\s{0,3}(?:#{1,6}|[-*+]|\d+[.)])\s/m.test(text) && !/\[[^\]]*\]\([^)]*\)/.test(text);

export const refinements: Record<string, (value: any) => boolean | undefined> = {
  'two to four words naming a product area': value =>
    typeof value === 'string' && words(value).length >= 2 && words(value).length <= 4,
  'one sentence that states what is wrong or wanted and what the customer already tried': value =>
    typeof value === 'string' && sentenceCount(value) === 1,
  'two to five sentences of plain text addressed to the customer': value =>
    typeof value === 'string' && sentenceCount(value) >= 2 && sentenceCount(value) <= 5 && plain(value),
  'a positive whole number of minutes': value => typeof value === 'number' && Number.isSafeInteger(value) && value >= 1,
};
