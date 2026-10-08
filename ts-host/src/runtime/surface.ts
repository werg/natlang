/**
 * The TypeScript surface of natlang. Keep these declarations in step with
 * `compiler/intrinsics.ts`, which supplies the same types to eval programs.
 */
import { Iteration, iterateOn as iterate, type IterationEvent, type IterationTrajectory, type ProgressVerdict } from './iterate.js';
import { inline, uncompiled } from './lowered.js';

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
