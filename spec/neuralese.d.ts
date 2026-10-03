/**
 * Surface declarations for Neuralese, contexts and learning (spec/SPEC.md, "Neuralese",
 * "Contexts"). These declarations give the TypeScript checker the type rules it can enforce
 * itself: opacity of soft values, applicability of soft functions, nested-type rejection and
 * dialect mismatch. The natlang compiler adds the rules TypeScript cannot express
 * (`neuralese-condition`, `neuralese-interpolation`, comparisons, `neuralese-untyped-literal`,
 * `type-recursive-function`, `context-new-executable`).
 *
 * Model-emitted literals `<|neuralese|>⟦…⟧<|/neuralese|>` never reach the checker: the runtime
 * replaces each one with a reference expression (`__neuralese.value(id)` or
 * `__neuralese.body(id)`) before type checking.
 */

declare module 'natlang:neuralese' {
  /** A dialect version tag, e.g. `'nd:natlang@1'` (spec/NEURALESE_DIALECTS.md). */
  export type Dialect = `nd:${string}@${number}`;

  /**
   * Program configuration binds the default dialect by augmenting this interface:
   * `declare module 'natlang:neuralese' { interface NeuraleseConfig { dialect: 'nd:natlang@2' } }`.
   */
  export interface NeuraleseConfig {}
  export type DefaultDialect = NeuraleseConfig extends { dialect: infer D extends Dialect } ? D : 'nd:natlang@1';

  const softBrand: unique symbol;
  /** The opaque part of every soft value. It has no readable members. */
  export interface SoftValue<T, D extends Dialect> {
    readonly [softBrand]: { readonly type: T; readonly dialect: D };
  }


  /**
   * A soft value of type `T` in dialect `D`. Not a `T`; a `T` is not one either.
   * Soft functions are callable with their declared parameters and result.
   * `Neuralese<Neuralese<T>>` is `never` (`neuralese-nested`).
   */
  export type Neuralese<T, D extends Dialect = DefaultDialect> =
    [T] extends [SoftValue<any, any>] ? never :
    [T] extends [(...args: infer P) => infer R]
      ? SoftValue<T, D> & ((...args: P) => Promise<Awaited<R>>)
      : SoftValue<T, D>;

  /** Opaque soft instructions: the body of a Neuralese function literal. */
  export interface SoftBody { readonly [softBrand]: { readonly body: true } }

  // --- Combinators (system natural-language functions with trainable soft bodies) ---

  export function map<A, B, D extends Dialect = DefaultDialect>(
    v: Neuralese<A, D>, f: (a: A) => Promise<B>): Promise<Neuralese<B, D>>;
  export function zip<A, B, D extends Dialect = DefaultDialect>(
    a: Neuralese<A, D>, b: Neuralese<B, D>): Promise<Neuralese<[A, B], D>>;
  export function ap<A, B, D extends Dialect = DefaultDialect>(
    f: Neuralese<(a: A) => Promise<B>, D>, a: Neuralese<A, D> | A): Promise<Neuralese<B, D>>;
  export function combine<T, D extends Dialect = DefaultDialect>(
    ...vs: Neuralese<T, D>[]): Promise<Neuralese<T, D>>;
  export function empty<T, D extends Dialect = DefaultDialect>(): Neuralese<T, D>;
  export function split<T extends object, D extends Dialect = DefaultDialect>(
    v: Neuralese<T, D>): Promise<{ [K in keyof T]: Neuralese<T[K], D> }>;
  export function splitList<E, D extends Dialect = DefaultDialect>(
    v: Neuralese<E[], D>): Promise<Neuralese<E, D>[]>;
  /** The only way out: a checked `T`, or a `NatlangCallError`. */
  export function read<T, D extends Dialect = DefaultDialect>(v: Neuralese<T, D>): Promise<T>;
  export function convert<T, D1 extends Dialect, D2 extends Dialect>(
    v: Neuralese<T, D1>, to: D2): Promise<Neuralese<T, D2>>;
  /** Diagnostic text for people. Not a value form; never shown to a model as the value. */
  export function gloss(v: SoftValue<unknown, Dialect>): Promise<string>;

  /** Marks a `let` capture of a Neuralese function literal as live (read per call, written back). */
  export function live<T>(binding: T): T;

  /** The reference form of a soft value outside the model. */
  export interface NeuraleseRef { readonly $neuralese: { readonly type: string; readonly id: string } }
}

/** Compiler-emitted replacements for model-written literals. Not for authors. */
declare const __neuralese: {
  value<T, D extends import('natlang:neuralese').Dialect = import('natlang:neuralese').DefaultDialect>(
    id: string): import('natlang:neuralese').Neuralese<T, D>;
  body(id: string): import('natlang:neuralese').SoftBody;
};

declare module 'natlang:context' {
  /** An immutable, content-addressed folder value of definitions and data. */
  export interface Context {
    readonly id: string;
    /** A context with data entries replaced or added, and executable nodes edited in place. */
    with(entries: Record<string, unknown>): Context;
    /** A union of contexts loaded from files (and of their subsets). */
    union(...others: Context[]): Context;
    /** A subset by entry name. */
    pick(...names: string[]): Context;
  }

  /** A natural-language function bound to its context. */
  export type Bound<F extends (...args: any[]) => Promise<any>> = F & {
    /** The same function bound to another context, checked against its context interface. */
    in(context: Context): Bound<F>;
    readonly context: Context;
  };
}

declare module 'natlang:learning' {
  import type { Neuralese, SoftValue, Dialect } from 'natlang:neuralese';

  const lossBrand: unique symbol;
  const gradBrand: unique symbol;
  /** An opaque scalar loss. */
  export interface Loss { readonly [lossBrand]: true }
  /** An opaque gradient with the shape of `A`'s soft parts. */
  export interface Gradient<A> { readonly [gradBrand]: A }
  /** A recorded trajectory of one execution (spec/NEURALESE_GRAPH.md). */
  export interface Trajectory { readonly id: string }

  export type LawName =
    | 'map-identity' | 'map-fusion' | 'read-map' | 'combine-associativity' | 'combine-identity' | 'split-zip';

  export function grad<A>(f: (a: A) => Promise<Loss>, a: A, options?: { order?: 1 | 2 }): Promise<Gradient<A>>;
  export function valueAndGrad<A>(
    f: (a: A) => Promise<Loss>, a: A, options?: { order?: 1 | 2 }): Promise<{ loss: Loss; grad: Gradient<A> }>;
  export function stopGradient<T>(v: T): T;

  export const objectives: {
    crossEntropy(output: Promise<unknown>, expected: unknown): Promise<Loss>;
    selfDistill(output: Promise<unknown>, withFullSource: () => Promise<unknown>): Promise<Loss>;
    /** Log-probability of a recorded trajectory: discrete choices and sampled Neuralese payloads (log N(z; μ, τ²σ²)). */
    logLikelihood(trajectory: Trajectory, weight?: number): Promise<Loss>;
    /** KL divergence of written or stored Gaussian blocks to the standard normal prior (VAE-style regulariser). */
    klPrior(blocks: Neuralese<unknown, any> | Neuralese<unknown, any>[]): Promise<Loss>;
    law(name: LawName, ...operands: SoftValue<unknown, Dialect>[]): Promise<Loss>;
    sum(...losses: Loss[]): Promise<Loss>;
    scale(loss: Loss, weight: number): Promise<Loss>;
  };

  const optBrand: unique symbol;
  export interface OptimizerState<A> { readonly [optBrand]: A }
  export interface Optimizer {
    init<A>(a: A): OptimizerState<A>;
    step<A>(state: { value: A; opt: OptimizerState<A> }, grad: Gradient<A>): { value: A; opt: OptimizerState<A> };
  }
  export const optimizers: {
    sgd(options: { lr: number; momentum?: number }): Optimizer;
    adam(options: { lr: number; betas?: [number, number]; weightDecay?: number }): Optimizer;
  };

  /** A written `.nz` file (spec/NEURALESE_FILES.md). */
  export interface NzFile { readonly path: string; readonly exports: readonly string[] }
  export function save(path: string, exports: Record<string, unknown>): Promise<NzFile>;

  export type { Neuralese };
}
