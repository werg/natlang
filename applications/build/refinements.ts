/**
 * Crisp checkers for the refinement predicates of types.ts (plans/REFINEMENT_TYPES.md). The launcher loads this
 * module's `refinements` table; each key is a predicate exactly as `Is<T, "...">` states it, and a checker decides it
 * from the value alone, without a model call. A checker that returns `undefined` leaves the value to the judge.
 */
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const strings = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === 'string');
const sortedUnique = (list: string[]) => list.every((item, i) => i === 0 || list[i - 1]! < item);

const CAUSES = ['command-failed', 'missing-input', 'missing-tool', 'output-conflict', 'input-mutated', 'path-escape', 'declaration-error', 'interrupted', 'other'];
const REASONS = ['no record of an earlier run', 'declaration changed'];
const PATHS = /^(input changed|output missing|output modified): \S/;

export const refinements: Record<string, (value: any) => boolean | undefined> = {
  'a plan whose needed list is sorted without duplicates and contains the goal, and whose order and cycle together hold each needed task exactly once': value => {
    if (!isRecord(value) || typeof value.goal !== 'string' || !strings(value.needed) || !strings(value.order) || !strings(value.cycle)) return false;
    const needed = new Set(value.needed), placed = [...value.order, ...value.cycle];
    return sortedUnique(value.needed) && needed.has(value.goal) && placed.length === needed.size && placed.every(id => needed.has(id)) && new Set(placed).size === placed.length;
  },
  "valid exactly when the reason is up to date; an invalid verdict's reason is no record of an earlier run, declaration changed, or input changed, output missing or output modified followed by a colon, a space and a path": value => {
    if (!isRecord(value) || typeof value.valid !== 'boolean' || typeof value.reason !== 'string') return false;
    return value.valid ? value.reason === 'up to date' : REASONS.includes(value.reason) || PATHS.test(value.reason);
  },
  'a diagnosis whose cause is command-failed, missing-input, missing-tool, output-conflict, input-mutated, path-escape, declaration-error, interrupted or other, and whose retry is inspect-first for interrupted, no for other and after-fix for every other cause': value => {
    if (!isRecord(value) || typeof value.cause !== 'string' || !CAUSES.includes(value.cause)) return false;
    return value.retry === (value.cause === 'interrupted' ? 'inspect-first' : value.cause === 'other' ? 'no' : 'after-fix');
  },
  'a report whose status is done only when order contains the goal and every result has status ok': value => {
    if (!isRecord(value)) return false;
    if (value.status !== 'done') return true;
    return Array.isArray(value.order) && value.order.includes(value.goal) && Array.isArray(value.results) &&
      value.results.every(result => isRecord(result) && result.status === 'ok');
  },
};
