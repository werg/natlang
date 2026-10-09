/**
 * The TypeScript surface of natlang. Keep these declarations in step with
 * `compiler/intrinsics.ts`, which supplies the same types to eval programs.
 */
import { Iteration, iterateOn as iterate, type IterationEvent, type IterationTrajectory, type ProgressVerdict } from './iterate.js';
import { inline, uncompiled } from './lowered.js';
import { resolveFrame } from './runtime.js';
import { traceFor } from '../native/graph.js';
import { RefinementError, canonicalValue, failureError } from '../native/refinement.js';
import { hexDigest } from '../native/hash.js';
import { normalizePredicate } from '../native/types.js';
export { Deopt } from '../calls/dispatch.js';

export type { IterationEvent, IterationTrajectory, ProgressVerdict };
export type { Iteration };

/** Marker for an unspecified `nl` type argument. */
export interface NlUnspecified { readonly __natlangUnspecified: true }
export interface NatlangStepError<M extends string> { readonly __natlangStepError: M }
type NatlangIsAny<T> = 0 extends (1 & T) ? true : false;
export type NatlangStepState<A extends unknown[], R> =
  number extends A['length'] ? (NatlangIsAny<R> extends true ? any : A[0]) :
  A extends [infer S, ...unknown[]] ? ([R] extends [S] ? S :
    NatlangStepError<'iterateOn requires the step return type to be assignable to its first parameter'>) :
  NatlangStepError<'iterateOn requires a step whose first parameter is the state'>;
export type NatlangStepArgs<A extends unknown[]> = number extends A['length'] ? any[] :
  A extends [unknown, ...infer Rest] ? Rest : never[];

/** Methods every compiled natlang callable provides. */
export interface NatlangCallableMethods<A extends unknown[], R> {
  iterateOn(initial: NatlangStepState<A, R>, ...args: NatlangStepArgs<A>): Iteration<NatlangStepState<A, R>>;
}
/** A compiled natural-language function: asynchronous, typed, and monitored. */
export type NatlangFunction<A extends unknown[] = any[], R = any> = ((...args: A) => Promise<R>) & NatlangCallableMethods<A, R>;
/** An \`nl\` function without a type argument; the compiler infers its signature from its uses. */
export type NatlangUntypedFunction = ((...args: any[]) => Promise<any>) & {
  iterateOn<S>(initial: S, ...args: any[]): Iteration<S>;
};
export type NlResult<F> = [F] extends [NlUnspecified] ? NatlangUntypedFunction :
  [F] extends [(...args: infer A) => infer R] ? NatlangFunction<A, Awaited<R>> : NatlangFunction<any[], F>;

/** The body of a soft function literal: a model-written Neuralese block in an `nl` template. */
export interface NeuraleseBody { readonly __natlangNeuraleseBody: true }
/** The template tag `nl.with({ … })` returns: a function whose captures are exactly the listed ones. */
export interface NlWithTag<F> {
  (strings: TemplateStringsArray, body: NeuraleseBody): any;
  <G = F>(strings: TemplateStringsArray, ...values: unknown[]): NlResult<G>;
}
/** The `nl` template tag. */
export interface NlTag {
  <F = NlUnspecified>(strings: TemplateStringsArray, ...values: unknown[]): NlResult<F>;
  /**
   * An `nl` function whose captures are exactly these: plain entries are snapshots taken now; `live(x)` entries
   * are read at each call and written back. No other names are captured.
   * `nl.with<CaptureRecord, Result>(captures)` can check the finite capture schema separately from the child result;
   * child arguments remain the values passed when the returned callable is invoked.
   * @natlangIntrinsic nl.with
   */
  with<CaptureRecord extends object, Result>(captures: CaptureRecord): NlWithTag<Result>;
  with<F = NlUnspecified>(captures: { readonly [name: string]: unknown }): NlWithTag<F>;
}

/**
 * Create an anonymous natural-language function. Exact mentions of visible names capture live bindings;
 * `nl.with({ … })` lists the captures explicitly instead.
 * @natlangIntrinsic nl
 */
export const nl: NlTag = Object.assign(function nl(strings: TemplateStringsArray, ...values: unknown[]) {
  void strings; void values;
  return uncompiled();
}, { with: (captures: { readonly [name: string]: unknown }) => { void captures; return uncompiled(); } }) as NlTag;
// Compiled `nl` expressions call `nl.__inline(plan, values, accessors, context)`.
Object.defineProperty(nl, '__inline', { value: inline });

/**
 * Run a step repeatedly from an initial state until a predicate holds, with progress review.
 * @natlangIntrinsic iterateOn
 */
export function iterateOn<T, A extends unknown[]>(step: (state: T, ...args: A) => T | Promise<T>, initial: T, ...args: A): Iteration<T> {
  return iterate(step as never, initial, ...args);
}

declare const natlangRefinement: unique symbol;
/**
 * A `T` whose value satisfies the natural-language predicate `P` (plans/REFINEMENT_TYPES.md). It is a `T` everywhere;
 * crisp code obtains one from a refined natlang result, from `refine(value, predicate)` or from `assume(value, predicate)`.
 */
export type Is<T, P extends string> = T & { readonly [natlangRefinement]: { [K in P]: true } };

const checkedPredicate = (predicate: unknown): string => {
  const text = typeof predicate === 'string' ? normalizePredicate(predicate) : '';
  if (!text) throw new RefinementError('refinement-predicate-invalid',
    'Is<T, P> takes a nonempty string literal as P, for example refine(subject, "one line of at most 60 characters").');
  return text;
};
const emitRefinement = (frame: ReturnType<typeof resolveFrame>, kind: string, data: Record<string, unknown>) => {
  const trace = traceFor(frame.parentCallId);
  if (trace) trace.emit(kind, data); else frame.task.refinementEvents.push({ kind, data });
};

/**
 * Check `value` against `predicate` with the judge and return it as an `Is<T, P>`; throws `refinement-unsatisfied` (or
 * `refinement-undecided` inside the uncertainty band). Must run inside a natlang task. `refine<Is<T, "p">>(value)` is
 * lowered by the compiler to `refine(value, "p")`.
 * @natlangIntrinsic refine
 */
export async function refine<T, P extends string>(value: T, predicate: P): Promise<Is<T, P>>;
export async function refine<R extends Is<unknown, string>>(value: unknown): Promise<R>;
export async function refine(value: unknown, predicate?: string): Promise<unknown> {
  const text = checkedPredicate(predicate);
  const frame = resolveFrame();
  const task = frame.task;
  const failures = await task.refinementChecker().check([{ path: 'value', predicate: text, value }],
    { phase: 'refine', ...task.refinementJudges(task.model(), frame), callId: frame.parentCallId ?? null, signal: frame.signal ?? task.signal,
      emit: (kind, data) => emitRefinement(frame, kind, data) });
  if (failures.length) throw failureError(failures[0]!);
  return value;
}

/**
 * Declare that `value` satisfies `predicate` without checking it. The assumption is recorded in the trace.
 * @natlangIntrinsic assume
 */
export function assume<T, P extends string>(value: T, predicate: P): Is<T, P>;
export function assume<R extends Is<unknown, string>>(value: unknown): R;
export function assume(value: unknown, predicate?: string): unknown {
  const text = checkedPredicate(predicate);
  const frame = resolveFrame();
  const shown = canonicalValue(value);
  emitRefinement(frame, 'refinement_assumed', { call_id: frame.parentCallId ?? null, predicate: text,
    value: shown.length > 400 ? `${shown.slice(0, 400)} … (${shown.length} chars)` : shown, value_sha256: hexDigest(shown) });
  return value;
}

/** Host and application helper: one entry point over a crisp and a natural-language implementation (see ./pluggable.ts). */
export { pluggable } from "./pluggable.js";
export type { PluggableMode, PluggableImplementations, PluggableOptions } from "./pluggable.js";
