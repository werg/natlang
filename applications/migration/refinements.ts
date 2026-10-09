/**
 * Crisp checkers for the refinement predicates of types.ts (plans/REFINEMENT_TYPES.md). The launcher loads this
 * module's `refinements` table; each key is a predicate exactly as `Is<T, "...">` states it, and a checker decides it
 * from the value alone, without a model call. A checker that returns `undefined` leaves the value to the judge.
 */
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const text = (value: unknown): value is string => typeof value === 'string';
const filled = (value: unknown): value is string => typeof value === 'string' && value.trim() !== '';
const whole = (value: unknown, least: number) => typeof value === 'number' && Number.isSafeInteger(value) && value >= least;
const unique = (list: unknown[]) => new Set(list).size === list.length;

const PATTERNS = ['declaration', 'call', 'import', 'export', 'type-position', 'property-access', 'comment-or-doc', 'string-literal', 'test', 'unrelated'];
const REPAIRABLE = ['missed-site', 'wrong-edit', 'test-expectation'];
const KINDS = [...REPAIRABLE, 'environment', 'unrelated'];

export const refinements: Record<string, (value: any) => boolean | undefined> = {
  'patches that each name a path, replace non-empty old text, and give new text that differs from the old': value =>
    Array.isArray(value) && value.every(patch => isRecord(patch) && filled(patch.path) && text(patch.old) && patch.old !== '' && text(patch.new) && patch.new !== patch.old),
  'an intent whose queries are non-empty strings, each listed once': value =>
    isRecord(value) && Array.isArray(value.queries) && value.queries.every(filled) && unique(value.queries),
  'a site whose from and to are line numbers from 1 with from at most to, and whose hits is a whole number': value =>
    isRecord(value) && whole(value.from, 1) && whole(value.to, 1) && (value.from as number) <= (value.to as number) && whole(value.hits, 0),
  'a usage whose pattern is declaration, call, import, export, type-position, property-access, comment-or-doc, string-literal, test or unrelated, and whose reason is not empty': value =>
    isRecord(value) && PATTERNS.includes(value.pattern as string) && filled(value.reason),
  'a plan whose edits list each site id once, whose leave entries each give a site id and a reason, and in which no site id is both edited and left': value => {
    if (!isRecord(value) || !Array.isArray(value.edits) || !Array.isArray(value.leave)) return false;
    if (!value.edits.every(filled) || !unique(value.edits)) return false;
    if (!value.leave.every(entry => isRecord(entry) && filled(entry.site) && filled(entry.reason))) return false;
    const left = (value.leave as { site: string }[]).map(entry => entry.site);
    return unique(left) && left.every(site => !(value.edits as string[]).includes(site));
  },
  'a finding whose kind is missed-site, wrong-edit, test-expectation, environment or unrelated, and which is repairable exactly when the kind is missed-site, wrong-edit or test-expectation': value =>
    isRecord(value) && KINDS.includes(value.kind as string) && value.repairable === REPAIRABLE.includes(value.kind as string),
  'a verdict whose problem is empty when exact is true and is a sentence when exact is false': value =>
    isRecord(value) && typeof value.exact === 'boolean' && text(value.problem) && (value.exact ? value.problem === '' : filled(value.problem)),
  'a state whose remaining budget is a non-negative whole number': value => isRecord(value) && whole(value.remaining, 0),
};
