/**
 * Convergence combinators (plans/semantic-fp/03_SEMANTIC_STDLIB_AND_PATTERNS.md S2; ts-host/std/DECOMPOSITION.md):
 * `semanticRepair` (an exact check feeds a fix until it holds), `fixedPoint` (revise until the meaning settles) and
 * `semanticMerge` (a three-way merge whose conflicts are judged by intent). The loops run through `iterateOn`; the
 * judgments are the caller's functions or the shipped ones in judgments.ts. Values are passed by reference, never
 * serialised, so a `Neuralese<T>` state flows through unchanged.
 */
import { IterationLimitError, iterateOn, stableHash } from '../runtime/iterate.js';
import { decideInFrame } from '../runtime/runtime.js';
import type { NatlangFunction } from '../runtime/surface.js';
import { combine as shippedCombine, compatible as shippedCompatible, intentOf as shippedIntentOf, settled as shippedSettled,
  type Edit, type Intent } from './judgments.js';
import { StdSpan, stdError } from './trace.js';
import { isPlainObject } from '../runtime/plain-object.js';

/** What an exact check found wrong with a value: one sentence per problem, none when the value passes. */
export type Problems = readonly string[];
type Maybe<T> = T | Promise<T>;

function problemsOf(found: unknown, who: string): string[] {
  if (!Array.isArray(found) || found.some(item => typeof item !== 'string'))
    throw stdError('std-check-result', `${who} returns the list of problems it found, as strings, and an empty list when the value passes.`);
  return [...found];
}
function boundOf(value: unknown, name: string, combinator: string): number {
  if (!Number.isSafeInteger(value) || (value as number) < 0)
    throw stdError('iteration-unbounded', `${combinator} needs ${name}, a whole number of at least 0: the most steps it may take before it returns its closest value.`);
  return value as number;
}

// --- semanticRepair -------------------------------------------------------------------------------------------------

export type RepairOptions<T> = {
  /** The exact check (compiler, schema, tests): the problems it finds in a value, none when it passes. */
  check: (value: T) => Maybe<Problems>;
  /** The fix: a value without those problems. Usually an inline `nl` such as `nl<(draft: T, problems: string[]) => T>`. */
  fix: (value: T, problems: string[]) => Maybe<T>;
  /** The most fixes to try. When they run out, the result has `ok: false` and the value with the fewest problems. */
  attempts: number;
  /** How two values are compared to see that a fix repeated an earlier value (which ends the repair). Default: by content. */
  identity?: (value: T) => string;
};
export type RepairAttempt<T> = { value: T; problems: string[] };
export type Repair<T> = {
  /** Whether the returned value passes the check. */
  ok: boolean;
  /** The passing value, or else the value with the fewest problems (the latest among equals). */
  value: T;
  /** The check's problems with `value`. */
  problems: string[];
  /** How many fixes ran. */
  attempts: number;
  /** The checked values in order: the initial one, then one per fix. */
  history: RepairAttempt<T>[];
};

/**
 * Check `value` exactly; while the check finds problems, hand the value and its problems to `fix` and check its answer.
 * Stops when a value passes, when a fix repeats an earlier value, or after `attempts` fixes.
 */
export async function semanticRepair<T>(value: T, options: RepairOptions<T>): Promise<Repair<T>> {
  if (typeof options?.check !== 'function' || typeof options.fix !== 'function')
    throw stdError('std-arguments', 'semanticRepair(value, { check, fix, attempts }) needs an exact check and a fix function.');
  const attempts = boundOf(options.attempts, 'attempts', 'semanticRepair');
  const identity = options.identity ?? stableHash;
  const span = new StdSpan('semanticRepair');
  return span.around(async () => {
    const checked = async (candidate: T) => problemsOf(await span.run(() => options.check(candidate)), 'check');
    const history: RepairAttempt<T>[] = [{ value, problems: await checked(value) }];
    span.stage('check', { attempt: 0, problems: history[0]!.problems.length });
    const seen = new Set([identity(value)]);
    type State = RepairAttempt<T> & { repeated: boolean };
    const step = async (state: State): Promise<State> => {
      const next = await span.run(() => options.fix(state.value, state.problems));
      const problems = await checked(next);
      const key = identity(next), repeated = seen.has(key);
      seen.add(key);
      history.push({ value: next, problems });
      span.stage('fix', { attempt: history.length - 1, problems: problems.length, repeated });
      return { value: next, problems, repeated };
    };
    try {
      await span.run(() => iterateOn(step, { ...history[0]!, repeated: false }).withLimit({ maxSteps: attempts })
        .until(state => state.problems.length === 0 || state.repeated));
    } catch (error) { if (!(error instanceof IterationLimitError)) throw error; }
    const passing = history.find(attempt => attempt.problems.length === 0);
    const best = passing ?? history.reduce((closest, attempt) => attempt.problems.length <= closest.problems.length ? attempt : closest);
    return { ok: !!passing, value: best.value, problems: best.problems, attempts: history.length - 1, history };
  }, result => `${result.ok ? 'passed' : 'failed'} after ${result.attempts} fixes`);
}

// --- fixedPoint -----------------------------------------------------------------------------------------------------

/** One revision: the value before and after a step. */
export type Revision<T> = { previous: T; current: T };
export type FixedPointOptions<T> = {
  /**
   * The judgment that the revision only reworded its value. Default: the shipped `settled` (`@natlang/std/settled.nl`).
   * A natural-language judgment needs no step bound: the progress judge watches long runs.
   */
  judge?: NatlangFunction<[revision: Revision<T>], boolean>;
  /** An exact settledness test instead of the judgment, such as equality of the values. It needs `maxSteps`. */
  settled?: (previous: T, current: T) => Maybe<boolean>;
  /** The most steps. Reaching it returns the latest value with `settled: false`. */
  maxSteps?: number;
};
export type FixedPoint<T> = { value: T; steps: number; settled: boolean };

/** Apply `step` to its own output until a revision no longer changes the value's meaning (or `settled` says so). */
export async function fixedPoint<T>(step: (value: T) => Maybe<T>, initial: T, options: FixedPointOptions<T> = {}): Promise<FixedPoint<T>> {
  if (typeof step !== 'function') throw stdError('std-arguments', 'fixedPoint(step, initial, options) needs a step function.');
  if (options.settled && options.maxSteps === undefined)
    throw stdError('iteration-unbounded', 'fixedPoint with an exact settled test needs maxSteps: the most steps before it returns the latest value; leave settled out to judge settledness in natural language.');
  const maxSteps = options.maxSteps === undefined ? undefined : boundOf(options.maxSteps, 'maxSteps', 'fixedPoint');
  const span = new StdSpan('fixedPoint');
  return span.around(async () => {
    if (maxSteps === 0) return { value: initial, steps: 0, settled: false };
    let steps = 0;
    const revise = async (revision: Revision<T>): Promise<Revision<T>> => {
      const current = await span.run(() => step(revision.current));
      steps++;
      span.stage('step', { step: steps });
      return { previous: revision.current, current };
    };
    const first = await revise({ previous: initial, current: initial });
    if (stableHash(first.current) === stableHash(initial)) return { value: first.current, steps, settled: true };
    let iteration = iterateOn(revise, first);
    if (maxSteps !== undefined) iteration = iteration.withLimit({ maxSteps: maxSteps - 1 });
    const settled = options.settled;
    try {
      const last = await span.run(() => iteration.until(settled ? revision => settled(revision.previous, revision.current) :
        (options.judge ?? shippedSettled) as unknown as (revision: Revision<T>) => Promise<boolean>));
      return { value: last.current, steps, settled: true };
    } catch (error) {
      if (error instanceof IterationLimitError) return { value: (error.lastState as Revision<T>).current, steps, settled: false };
      throw error;
    }
  }, result => `${result.settled ? 'settled' : 'unsettled'} after ${result.steps} steps`);
}

// --- semanticMerge --------------------------------------------------------------------------------------------------

export type MergeOptions = {
  /** The least probability that the two edits are compatible at which they are combined. Default 0.5. */
  floor?: number;
  /** An exact check of a combined value at its path: its problems, none when it holds. */
  check?: (value: unknown, path: string[]) => Maybe<Problems>;
  /** What the merged value holds where a conflict stays unresolved. Default `base`: only vetted edits enter. */
  keep?: 'base' | 'ours' | 'theirs';
  /** Replacements for the shipped judgments, such as `intentOf.in(context)` or a function on another model. */
  intentOf?: (before: unknown, after: unknown) => Promise<Intent>;
  compatible?: (base: unknown, ours: Edit, theirs: Edit) => Promise<boolean>;
  combine?: (base: unknown, ours: Edit, theirs: Edit) => Promise<unknown>;
};
export type MergeConflict = { path: string[]; base: unknown; ours: unknown; theirs: unknown };
export type Merged = MergeConflict & { value: unknown; intents: { ours: Intent; theirs: Intent }; compatibility: number };
export type Unresolved = MergeConflict & {
  /** `incompatible`: the purposes exclude each other; `unsure`: compatibility below the floor; `check`: the combined value failed. */
  reason: 'incompatible' | 'unsure' | 'check';
  intents: { ours: Intent; theirs: Intent };
  compatibility: number;
  problems?: string[];
};
export type Merge<T> = { value: T; merged: Merged[]; unresolved: Unresolved[] };

const isRecord = isPlainObject;
const kindOf = (value: unknown) => value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;

/**
 * Merge two edits of `base`. Records merge key by key; a side that left a part unchanged takes the other side's edit,
 * equal edits are one edit. Every other part is a conflict: each edit's intent is stated, the two are judged
 * compatible or not, and compatible ones are combined and checked. What is not merged is listed in `unresolved`.
 */
export async function semanticMerge<T>(base: T, ours: T, theirs: T, options: MergeOptions = {}): Promise<Merge<T>> {
  const floor = options.floor ?? 0.5;
  if (!(floor >= 0 && floor <= 1)) throw stdError('std-arguments', 'semanticMerge floor is a probability between 0 and 1.');
  const span = new StdSpan('semanticMerge');
  return span.around(async () => {
    const same = (a: unknown, b: unknown) => a === b || stableHash(a) === stableHash(b);
    const conflicts: MergeConflict[] = [];
    const slots = new Map<object, MergeConflict>(), built = new WeakSet<object>();
    const walk = (b: unknown, o: unknown, t: unknown, path: string[]): unknown => {
      if (same(o, t) || same(t, b)) return o;
      if (same(o, b)) return t;
      if (isRecord(b) && isRecord(o) && isRecord(t)) {
        const result: Record<string, unknown> = {};
        built.add(result);
        for (const key of new Set([...Object.keys(b), ...Object.keys(o), ...Object.keys(t)])) {
          const value = walk(b[key], o[key], t[key], [...path, key]);
          if (value !== undefined) result[key] = value;
        }
        return result;
      }
      const conflict = { path, base: b, ours: o, theirs: t }, slot = {};
      conflicts.push(conflict); slots.set(slot, conflict);
      return slot;
    };
    const draft = walk(base, ours, theirs, []);
    span.stage('walk', { conflicts: conflicts.map(conflict => conflict.path) });
    const intentOf = options.intentOf ?? shippedIntentOf, compatible = options.compatible ?? shippedCompatible;
    const combine = options.combine ?? shippedCombine;
    const merged: Merged[] = [], unresolved: Unresolved[] = [];
    const resolution = new Map<MergeConflict, unknown>();
    await Promise.all(conflicts.map(async conflict => {
      const [oursIntent, theirsIntent] = await Promise.all([span.run(() => intentOf(conflict.base, conflict.ours)),
        span.run(() => intentOf(conflict.base, conflict.theirs))]);
      const intents = { ours: oursIntent, theirs: theirsIntent };
      const edits: [Edit, Edit] = [{ after: conflict.ours, intent: oursIntent }, { after: conflict.theirs, intent: theirsIntent }];
      const decision = await decideInFrame(span.frame, compatible, [conflict.base, ...edits]);
      const compatibility = decision.probabilities.find(item => item.value === true)?.probability ?? 0;
      const keep = () => resolution.set(conflict, conflict[options.keep ?? 'base']);
      if (compatibility < floor) {
        unresolved.push({ ...conflict, reason: decision.value === true ? 'unsure' : 'incompatible', intents, compatibility });
        return keep();
      }
      const value = await span.run(() => combine(conflict.base, ...edits));
      const kinds = [conflict.ours, conflict.theirs].map(kindOf);
      const problems = kinds.includes(kindOf(value)) ? options.check ? problemsOf(await span.run(() => options.check!(value, conflict.path)), 'check') : [] :
        [`The combined value is ${kindOf(value)}; the edits are ${[...new Set(kinds)].join(' and ')}.`];
      if (problems.length) {
        unresolved.push({ ...conflict, reason: 'check', intents, compatibility, problems });
        return keep();
      }
      merged.push({ ...conflict, value, intents, compatibility });
      resolution.set(conflict, value);
    }));
    span.stage('resolve', { merged: merged.map(item => item.path), unresolved: unresolved.map(item => ({ path: item.path, reason: item.reason })) });
    const fill = (value: unknown): unknown => {
      if (value !== null && typeof value === 'object' && slots.has(value)) return resolution.get(slots.get(value)!);
      if (value !== null && typeof value === 'object' && built.has(value))
        return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, fill(item)]).filter(([, item]) => item !== undefined));
      return value;
    };
    const byPath = (a: { path: string[] }, b: { path: string[] }) => a.path.join('\0') < b.path.join('\0') ? -1 : 1;
    return { value: fill(draft) as T, merged: merged.sort(byPath), unresolved: unresolved.sort(byPath) };
  }, result => `${result.merged.length} merged, ${result.unresolved.length} unresolved`);
}
