/**
 * Crisp checkers for the refinement predicates of types.ts (plans/REFINEMENT_TYPES.md). A host that runs the policy in
 * its own runtime passes this table as `refinements: { crisp }`; each key is a predicate exactly as `Is<T, "...">`
 * states it, and a checker decides it from the value alone, without a model call.
 */
const isRecord = (value: unknown): value is Record<string, unknown> => typeof value === 'object' && value !== null && !Array.isArray(value);

export const refinements: Record<string, (value: any) => boolean | undefined> = {
  'a decision whose waitMs, when given, is a positive whole number of milliseconds': value =>
    isRecord(value) && (value.waitMs === undefined || (typeof value.waitMs === 'number' && Number.isSafeInteger(value.waitMs) && value.waitMs >= 1)),
  'steps that list refund before release, each at most once, with a reason for each': value => {
    if (!isRecord(value) || !Array.isArray(value.steps)) return false;
    const steps = value.steps as unknown[];
    const actions = steps.map(step => isRecord(step) ? step.action : undefined);
    return steps.every(step => isRecord(step) && (step.action === 'refund' || step.action === 'release') && typeof step.reason === 'string' && step.reason !== '') &&
      new Set(actions).size === actions.length && !(actions.includes('refund') && actions.includes('release') && actions.indexOf('release') < actions.indexOf('refund'));
  },
  'a message whose subject is one line of at most 60 characters without a trailing period, and whose body is not empty': value =>
    isRecord(value) && typeof value.subject === 'string' && value.subject.length >= 1 && value.subject.length <= 60 && !/[\r\n]/.test(value.subject) &&
    !value.subject.trimEnd().endsWith('.') && typeof value.body === 'string' && value.body.trim() !== '',
};
