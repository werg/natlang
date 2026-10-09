/**
 * Crisp checkers for the refinement predicates of types.ts (plans/REFINEMENT_TYPES.md). A host that runs the planner in
 * its own runtime passes this table as `refinements: { crisp }`; each key is a predicate exactly as `Is<T, "...">` states
 * it, and a checker decides it from the value alone, without a model call.
 */
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);
const whole = (value: unknown, least = -Infinity) => typeof value === 'number' && Number.isSafeInteger(value) && value >= least;
const unique = (list: unknown[]) => new Set(list).size === list.length;
const span = (item: unknown) => isRecord(item) && whole(item.start) && whole(item.end) && (item.start as number) < (item.end as number);

export const refinements: Record<string, (value: any) => boolean | undefined> = {
  'a reading whose tasks each have an id that starts with a letter and has only letters, digits, underscores and hyphens, a duration that is a whole number of minutes of at least 1, and an id no other task has': value => {
    if (!isRecord(value) || !Array.isArray(value.tasks)) return false;
    const tasks = value.tasks as unknown[];
    return tasks.every(task => isRecord(task) && typeof task.id === 'string' && /^[A-Za-z][A-Za-z0-9_-]*$/.test(task.id) && whole(task.minutes, 1)) &&
      unique(tasks.map(task => (task as { id: string }).id));
  },
  'a reading whose limits each name a task, give a reason and set at least one of notBefore, endBy or after, and whose blocks each start a whole number of minutes before they end': value =>
    isRecord(value) && Array.isArray(value.limits) && Array.isArray(value.blocks) &&
    value.limits.every(limit => isRecord(limit) && typeof limit.task === 'string' && limit.task !== '' && typeof limit.reason === 'string' && limit.reason !== '' &&
      (whole(limit.notBefore) || whole(limit.endBy) || (Array.isArray(limit.after) && limit.after.length > 0))) &&
    value.blocks.every(span),
  'a reading whose preferences are numbered p1, p2 and so on in order, and each have a weight of 1, 2 or 3 and a list of task ids': value =>
    isRecord(value) && Array.isArray(value.preferences) &&
    value.preferences.every((wish, i) => isRecord(wish) && wish.id === `p${i + 1}` && [1, 2, 3].includes(wish.weight as number) &&
      Array.isArray(wish.tasks) && wish.tasks.every(id => typeof id === 'string')),
  "domains whose spans are ascending and disjoint, each start a whole number of minutes before its end and each at least as long as the task's minutes": value =>
    Array.isArray(value) && value.every(domain => {
      if (!isRecord(domain) || !Array.isArray(domain.spans) || !whole(domain.minutes, 1)) return false;
      const spans = domain.spans as { start: number, end: number }[];
      return spans.every(item => span(item) && item.end - item.start >= (domain.minutes as number)) &&
        spans.every((item, i) => i === 0 || spans[i - 1]!.end <= item.start);
    }),
  'placements that each start a whole number of minutes before they end, with no task placed twice': value =>
    Array.isArray(value) && value.every(item => span(item) && typeof (item as { id?: unknown }).id === 'string') &&
    unique(value.map(item => item.id)),
  'a proposal that carries placements and no questions when it is chosen, questions and no placements when it is unclear, and neither when it is infeasible': value => {
    if (!isRecord(value) || !Array.isArray(value.placements) || !Array.isArray(value.questions)) return false;
    const placed = value.placements.length > 0, asked = value.questions.length > 0;
    if (value.status === 'chosen') return placed && !asked;
    if (value.status === 'unclear') return !placed && asked;
    return value.status === 'infeasible' && !placed && !asked;
  },
};
