/**
 * Crisp checkers for the refinement predicates of types.ts (plans/REFINEMENT_TYPES.md). Each key is a predicate exactly
 * as `Is<T, "...">` states it; a checker decides it from the value alone, without a model call.
 */
export const refinements: Record<string, (value: any) => boolean | undefined> = {
  'two or three search phrases': value => Array.isArray(value) && value.length >= 2 && value.length <= 3 &&
    value.every(item => typeof item === 'string' && item.trim() !== ''),
  'each id listed once': value => Array.isArray(value) && value.every(item => typeof item === 'string') &&
    new Set(value).size === value.length,
};
