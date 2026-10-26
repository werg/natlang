/**
 * Surface declarations for Neuralese, contexts and learning (spec/ext/neuralese.md, spec/ext/learning.md, spec/SPEC.md
 * "Contexts"). These declarations give the TypeScript checker the type rules it can enforce itself: opacity of soft
 * values, applicability of soft functions, nested-type rejection and dialect mismatch. The natlang compiler adds the
 * rules TypeScript cannot express (`neuralese-opaque-access`, `neuralese-condition`, `neuralese-readout-sync`,
 * `neuralese-untyped-literal`, `type-recursive-function`, `context-new-executable`) and lowers the implicit readouts
 * the Neuralese extension lists.
 *
 * A soft value is an opaque reference to an archived native state (`nzs1_…`): the text, token IDs, token embeddings and
 * positional KV and hybrid state of one producer span. Model-written states never reach the checker as tokens: the
 * runtime replaces each one with a reference expression (`__neuralese("nzs1_…")`, or `__neuralese.body("nzs1_…")` for a
 * soft function body) before type checking.
 */

declare module 'natlang:neuralese' {
  /**
   * A dialect tag: `native-state/<architecture>` for a native model's state, or `semantic-state/<adapter SHA-256>` for a
   * state written through a configured learned semantic adapter (spec/NEURALESE_DIALECTS.md).
   */
  export type Dialect = string;

  /** The dialect a `Neuralese<T>` without a second argument names; the runtime binds it to its reader dialect. */
  export type DefaultDialect = 'DefaultDialect';

  const softBrand: unique symbol;
  /** The opaque part of every soft value. It has no readable members. */
  export interface SoftValue<T, D extends Dialect> {
    readonly [softBrand]: { readonly type: T; readonly dialect: D };
  }

  /** Receiver-only text methods on a soft string; the compiler reads the value first. */
  export type SoftStringMethods<T> = [T] extends [string] ? Pick<String,
    'trim' | 'trimStart' | 'trimEnd' | 'toLowerCase' | 'toUpperCase' |
    'includes' | 'startsWith' | 'endsWith' | 'indexOf' | 'lastIndexOf' | 'slice' | 'substring'> : {};

  /**
   * A soft value of type `T` in dialect `D`. Not a `T`; a `T` is not one either.
   * Soft functions are callable with their declared parameters and result.
   * `Neuralese<Neuralese<T>>` is `never` (`neuralese-nested`).
   */
  export type Neuralese<T, D extends Dialect = DefaultDialect> =
    [T] extends [SoftValue<any, any>] ? never :
    [T] extends [(...args: infer P) => infer R]
      ? SoftValue<T, D> & ((...args: P) => Promise<Awaited<R>>)
      : SoftValue<T, D> & SoftStringMethods<T>;

  /** Opaque soft instructions: the body of a Neuralese function literal. */
  export interface SoftBody { readonly [softBrand]: { readonly body: true } }

  // --- Combinators (system natural-language functions whose bodies are native states) ---

  export function map<A, B, D extends Dialect = DefaultDialect>(
    v: Neuralese<A, D>, f: (a: A) => Promise<B>): Promise<Neuralese<B, D>>;
  export function zip<A, B, D extends Dialect = DefaultDialect>(
    a: Neuralese<A, D>, b: Neuralese<B, D>): Promise<Neuralese<[A, B], D>>;
  export function ap<A, B, D extends Dialect = DefaultDialect>(
    f: Neuralese<(a: A) => Promise<B>, D>, a: Neuralese<A, D> | A): Promise<Neuralese<B, D>>;
  export function combine<T, D extends Dialect = DefaultDialect>(
    ...vs: Neuralese<T, D>[]): Promise<Neuralese<T, D>>;
  /** The library's archived zero-token native state, the identity of `combine`. */
  export function empty<T, D extends Dialect = DefaultDialect>(): Neuralese<T, D>;
  export function split<T extends object, D extends Dialect = DefaultDialect>(
    v: Neuralese<T, D>): Promise<{ [K in keyof T]: Neuralese<T[K], D> }>;
  export function splitList<E, D extends Dialect = DefaultDialect>(
    v: Neuralese<E[], D>): Promise<Neuralese<E, D>[]>;
  /** The typed way out: a checked `T`, or a `NatlangCallError`. */
  export function read<T, D extends Dialect = DefaultDialect>(v: Neuralese<T, D>): Promise<T>;
  export function convert<T, D1 extends Dialect, D2 extends Dialect>(
    v: Neuralese<T, D1>, to: D2): Promise<Neuralese<T, D2>>;
  /** `f` after `g`, exactly: the right side of the map-fusion rewrite (NEURALESE_REWRITES.md). */
  export function compose<A, B, C>(f: (b: B) => C | Promise<C>, g: (a: A) => B | Promise<B>): (a: A) => Promise<C>;
  /** Diagnostic text for people. Not a value form; never shown to a model as the value. */
  export function gloss(v: SoftValue<unknown, Dialect>): Promise<string>;
  /** The query operator, defined as `read(map(v, question))`: the answer to `question` from what `v` holds. */
  export function ask<T, D extends Dialect = DefaultDialect>(v: Neuralese<T, D>, question: string): Promise<string>;

  /** The reference form of a soft value outside the model. */
  export interface NeuraleseRef {
    readonly $neuralese: {
      readonly type: string;
      /** `nzs1_` and 52 base32 characters: the content address of the native state. */
      readonly id: string;
      /** A constant use: this occurrence receives no gradient. */
      readonly stop_gradient?: true;
      /** The producing graph occurrence of this use. */
      readonly source_occurrence?: string;
    };
  }
}

/** Eval globals provided by the compiler (intrinsics). */
/** Marks a `let` capture of a Neuralese function literal as live (read per call, written back). */
declare function live<T>(binding: T): T;
/** Read the ordinary typed value of a soft value; an ordinary string is returned unchanged. */
declare function read<T>(value: import('natlang:neuralese').Neuralese<T>): Promise<T>;
declare function read(value: string): Promise<string>;

/** Compiler-emitted replacements for model-written states. Not for authors. */
declare const __neuralese: {
  <T, D extends import('natlang:neuralese').Dialect = import('natlang:neuralese').DefaultDialect>(
    id: string, constant?: true, sourceOccurrence?: string): import('natlang:neuralese').Neuralese<T, D>;
  body(id: string, constant?: true, sourceOccurrence?: string): import('natlang:neuralese').SoftBody;
};

// Contexts are exported by the runtime package (`@natlang/node`, `@natlang/browser`; runtime/contexts.ts), not by a
// `natlang:` module.
declare module '@natlang/node' {
  /** An immutable, content-addressed folder value of definitions and data. */
  export interface Context {
    readonly id: string;
    /** Entry names: executable items and data, sorted. */
    readonly names: string[];
    /** A context with data entries replaced or added, and executable nodes edited in place. */
    with(entries: Record<string, unknown>): Context;
    /** A context without the named entries. */
    without(...names: string[]): Context;
    /** A context whose executable node `name` is replaced by an edited definition that compiles against it. */
    edit(name: string, text: string): Context;
    /** A subset by entry name. */
    pick(...names: string[]): Context;
    /** The union of two contexts; an entry present in both must be the same entry (`context-conflict`). */
    union(other: Context): Context;
    /** Whether every entry of `other` is in this context. */
    contains(other: Context): boolean;
  }

  /** A natural-language function bound to its context. */
  export type Bound<F extends (...args: any[]) => Promise<any>> = F & {
    /** The same function bound to another context, checked against its context interface. */
    in(context: Context): Bound<F>;
    readonly context: Context;
  };
}

declare module 'natlang:learning' {
  import type { Neuralese } from 'natlang:neuralese';

  /** A named native adapter coefficient artifact (`nnt1_`). Not a soft value. */
  export interface NativeAdapter { readonly $nativeAdapter: { readonly id: string } }
  export type NativeTensorKind = 'parameter-set' | 'parameter-gradient' | 'parameter-update' | 'state-gradient' |
    'optimizer-state' | 'adapter-basis' | 'adapter-gradient' | 'derivative-program';
  /** A named native tensor artifact (`nnt1_`) other than adapter coefficients. Not a soft value. */
  export interface NativeTensor { readonly $nativeTensor: { readonly id: string; readonly kind: NativeTensorKind } }
  /** The exact model identity a native state or artifact is bound to. */
  export type NativeIdentity = { architecture: string; checkpoint_sha256: string; tokenizer_fingerprint: string;
    embedding_weight_sha256: string };

  const lossBrand: unique symbol;
  const gradBrand: unique symbol;
  /** An opaque scalar loss; its value is known after `grad` or `valueAndGrad` evaluates it. */
  export interface Loss { readonly [lossBrand]: true; valueOf(): number }
  /** An opaque gradient with the shape of `A`'s native roots (archived native states and adapter coefficients). */
  export interface Gradient<A> { readonly [gradBrand]: A }
  /** The recorded native model turns of one execution (spec/NEURALESE_GRAPH.md). */
  export interface Trajectory { readonly id: string }

  /** Law names as `objectives.law` takes them; rewrite rules use the kebab-case names of NEURALESE_REWRITES.md. */
  export type LawName =
    | 'mapIdentity' | 'mapFusion' | 'readMapCommutation' | 'combineAssociativity' | 'combineIdentity' | 'splitZip';

  export function grad<A>(f: (a: A) => Promise<Loss>, a: A, options?: { order?: 1 | 2 }): Promise<Gradient<A>>;
  export function valueAndGrad<A>(
    f: (a: A) => Promise<Loss>, a: A, options?: { order?: 1 | 2 }): Promise<{ loss: Loss; grad: Gradient<A> }>;
  export function stopGradient<T>(v: T): T;
  /** The model turns one execution makes, recorded as a trajectory for `logLikelihood` and `policyGradient`. */
  export function trajectory(run: Promise<unknown> | (() => Promise<unknown>)): Promise<Trajectory>;

  /** Run `fn` with archived native states imported into every model call. Changes call context, not weights. */
  export function withNativeStates<T>(states: string | readonly string[], fn: () => Promise<T> | T): Promise<T>;
  /** Run `fn` with native adapter coefficients bound to every native model call. */
  export function withNativeAdapters<T>(adapters: NativeAdapter | readonly (NativeAdapter | { adapter: NativeAdapter; scale?: number })[],
    fn: () => Promise<T> | T): Promise<T>;
  /** Run `fn` with soft forms of the runtime prompt pieces (piece ID → Neuralese<string>). */
  export function withSystemPrompts<T>(prompts: Record<string, Neuralese<string>>, fn: () => Promise<T> | T): Promise<T>;

  export const nativeStates: {
    gradient(graph: { schema: 'natlang.neuralese-native-graph/1'; split: string }, id: string, nativeIdentity: NativeIdentity,
      options?: { nativeStateModes?: Record<string, 'same-base-adapter-state'>; objective?: 'graph-ce' }): Promise<unknown>;
    optimize(gradient: unknown, nativeIdentity: NativeIdentity,
      optimizer: { name: 'sgd' | 'adam'; learning_rate?: number; weight_decay?: number; beta1?: number; beta2?: number; eps?: number; momentum?: number },
      optimizerStateId?: string): Promise<unknown>;
  };
  export const nativeGraph: {
    gradient(f: () => Promise<Loss>,
      target: { kind: 'model-parameters'; names: string[] } | { kind: 'native-state' | 'adapter'; id: string; names?: string[] },
      nativeIdentity: NativeIdentity, options?: { nativeStateModes?: Record<string, 'same-base-adapter-state'> }): Promise<unknown>;
  };
  export const nativeAdapters: {
    create(nativeIdentity: NativeIdentity,
      options: { parameterNames: string[]; kind?: 'xs'; rank?: number; seed?: number }): Promise<NativeAdapter>;
    ref(id: string): NativeAdapter;
  };

  export const objectives: {
    crossEntropy(output: Promise<unknown>, expected: unknown): Promise<Loss>;
    selfDistill(output: Promise<unknown>, withFullSource: () => Promise<unknown>): Promise<Loss>;
    conditionedDistill(student: Promise<unknown> | (() => Promise<unknown>), teacher: Promise<unknown> | (() => Promise<unknown>),
      options?: { privileged?: unknown; rule?: 'logLoss' | 'brier' | 'rps' }): Promise<Loss>;
    /** A proper scoring rule on a decision readout. */
    decision(output: Promise<unknown> | (() => Promise<unknown>), expected: unknown, rule?: 'logLoss' | 'brier' | 'rps'): Promise<Loss>;
    /** Expected reward under a decision readout's option distribution. */
    expectedReward(output: Promise<unknown> | (() => Promise<unknown>), reward: (value: unknown) => number | Promise<number>): Promise<Loss>;
    /** Reward-weighted log-likelihood of sampled trajectories, at their recorded sampling temperature. */
    policyGradient(samples: { trajectory: Trajectory; reward: number }[], options?: { baseline?: 'mean' | number }): Promise<Loss>;
    /** Log-probability of a recorded trajectory's sampled tokens under the model (temperature 1). */
    logLikelihood(trajectory: Trajectory, weight?: number): Promise<Loss>;
    /** A combinator law as a cross-entropy consistency loss measured through `read`. */
    law(name: LawName, ...operands: unknown[]): Promise<Loss>;
    sum(...losses: Loss[]): Promise<Loss>;
    scale(loss: Loss, weight: number): Promise<Loss>;
  };

  const optBrand: unique symbol;
  export interface OptimizerState<A> { readonly [optBrand]: A }
  export interface Optimizer {
    init<A>(a: A): OptimizerState<A>;
    /** One update: new artifacts and new state; nothing changes in place. */
    step<A>(state: { value: A; opt: OptimizerState<A> }, grad: Gradient<A>): Promise<{ value: A; opt: OptimizerState<A> }>;
  }
  export const optimizers: {
    sgd(options: { lr: number; momentum?: number; weightDecay?: number }): Optimizer;
    adam(options: { lr: number; betas?: [number, number]; weightDecay?: number }): Optimizer;
  };

  /** Write native states and named coefficient artifacts, with their dependencies, to a `.nz` file (spec/NEURALESE_FILES.md). */
  export function save(path: string, exports: Record<string, unknown>): Promise<{ path: string; exports: string[] }>;

  export type { Neuralese };
}
