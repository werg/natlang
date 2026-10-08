/**
 * Declarations for the natlang TypeScript surface.
 *
 * `nl` and `iterateOn` are ordinary TypeScript declarations whose extra meaning is
 * supplied by the natlang compiler. The compiler recognizes them by the
 * `@natlangIntrinsic` JSDoc tag on the resolved declaration, never by spelling alone.
 */

/** Version of generated lowering. A runtime refuses output from another major version. */
export const NATLANG_COMPILE_VERSION = 7 as const;

/** Shared type contract for free `iterateOn` and eval's normalized `fn.iterateOn` analysis form. */
export const ITERATE_ON_SIGNATURE = '<T, A extends unknown[]>(step: (state: T, ...args: A) => T | Promise<T>, initial: T, ...args: A): Iteration<T>';

/** Names that cannot be used as child attributes of a natlang callable object. */
export const RESERVED_CALLABLE_PROPERTIES: ReadonlySet<string> = new Set([
  'call', 'apply', 'bind', 'name', 'length', 'prototype', 'constructor', '__proto__', 'caller', 'arguments',
  'toString', 'toLocaleString', 'valueOf', 'hasOwnProperty', 'isPrototypeOf', 'propertyIsEnumerable',
  '__defineGetter__', '__defineSetter__', '__lookupGetter__', '__lookupSetter__', 'then', 'iterateOn', 'in', 'with',
]);

/** Exact ambient declaration for the compiler-provided Neuralese type, also shown by read_code("Neuralese"). */
export const NEURALESE_TYPE_DECLARATION = String.raw`
/** The dialect a Neuralese<T> without a second argument names; the program's configuration binds it. */
type DefaultDialect = "DefaultDialect";
/** Brand of a soft value. It has no semantic fields to inspect; typed string conversions use readout. */
interface NeuraleseValue<T, D extends string> { readonly __natlangNeuralese: { readonly type: T; readonly dialect: D } }
/** Native, receiver-only text operations available on typed soft strings. The compiler reads the payload first. */
type NeuraleseStringMethods<T> = [T] extends [string] ? Pick<String,
  'trim' | 'trimStart' | 'trimEnd' | 'toLowerCase' | 'toUpperCase' |
  'includes' | 'startsWith' | 'endsWith' | 'indexOf' | 'lastIndexOf' | 'slice' | 'substring'> : {};
/**
 * A soft value of type T in dialect D: an opaque reference to a stored block of vectors that a model reads.
 * A Neuralese of a function type is callable with the function's parameters.
 */
type Neuralese<T, D extends string = DefaultDialect> = [T] extends [(...args: infer A) => infer R] ?
  NeuraleseValue<T, D> & ((...args: A) => Promise<Awaited<R>>) : NeuraleseValue<T, D> & NeuraleseStringMethods<T>;
`;

export const NEURALESE_TYPE_DOCUMENTATION =
  `Built-in compile-time type declaration (not a runtime value):\n${NEURALESE_TYPE_DECLARATION}\n` +
  'Neuralese<T> is an opaque typed reference. The brand does not expose payload fields. When the configured Neuralese standard library is available, ' +
  'neuralese.read(value) in eval (neuralese.read<T>(value: Neuralese<T>): Promise<T>) returns the ordinary typed value; use String(value) or another supported text-conversion context for Neuralese<string> ' +
  'when ordinary text is needed. These operations run the configured typed readout; neuralese.textReadSource is read-only provenance metadata, not a method, ' +
  'and neuralese.bodies.read is only the configured reader body ID. ' +
  'A direct string argument to a declared Neuralese<string> parameter is ' +
  'materialized through the configured writer; this does not convert nested fields or other payload types. This declaration grants no call or service capability.';

const DECLARATIONS = String.raw`
/** Marker for an unspecified \`nl\` type argument. */
interface NlUnspecified { readonly __natlangUnspecified: true }
interface NatlangStepError<M extends string> { readonly __natlangStepError: M }
type NatlangIsAny<T> = 0 extends (1 & T) ? true : false;
type NatlangStepState<A extends unknown[], R> =
  number extends A['length'] ? (NatlangIsAny<R> extends true ? any : A[0]) :
  A extends [infer S, ...unknown[]] ? ([R] extends [S] ? S :
    NatlangStepError<'iterateOn requires the step return type to be assignable to its first parameter'>) :
  NatlangStepError<'iterateOn requires a step whose first parameter is the state'>;
type NatlangStepArgs<A extends unknown[]> = number extends A['length'] ? any[] :
  A extends [unknown, ...infer Rest] ? Rest : never[];

/** Read-only observation of one iteration boundary. */
type IterationEvent<T> = Readonly<{
  kind: 'initial' | 'step' | 'review' | 'done' | 'error';
  sequence: number;
  iteration: number;
  iterationId: string;
  siteId: string;
  state?: T;
  review?: { verdict: 'continue' | 'divergent'; reason: string };
  error?: string;
}>;

/** Read-only trajectory supplied to a progress judge. */
interface IterationTrajectory<T> {
  readonly iterationId: string;
  readonly siteId: string;
  readonly length: number;
  state(index: number): T;
  states(start?: number, end?: number): T[];
  steps(start?: number, end?: number): { iteration: number; callId: string; elapsedMs: number; stateHash: string }[];
  repeats(): { iteration: number; firstSeen: number }[];
  summary(): { iterations: number; activeMs: number; repeatedStates: number; reviews: number; statistics: unknown };
}

type ProgressVerdict = { verdict: 'continue' | 'divergent'; reason: string };
type ProgressJudge<T> = (trajectory: IterationTrajectory<T>) => Promise<ProgressVerdict>;

/** A single-use monitored iteration plan. */
interface Iteration<T> {
  until(done: (state: T) => boolean | Promise<boolean>): Promise<T>;
  streamUntil(done: (state: T) => boolean | Promise<boolean>): AsyncIterable<IterationEvent<T>> & { readonly __natlangIterationStream: true };
  onStep(observer: (event: IterationEvent<T>) => void | Promise<void>): Iteration<T>;
  checkProgress(judge: ProgressJudge<T> | "off"): Iteration<T>;
  withMeasure(remaining: (state: T) => number): Iteration<T>;
  withSiteId(id: string): Iteration<T>;
  withLimit(limit: { maxSteps?: number; deadlineMs?: number }): Iteration<T>;
}

/** Methods every compiled natlang callable provides. */
interface NatlangCallableMethods<A extends unknown[], R> {
  iterateOn(initial: NatlangStepState<A, R>, ...args: NatlangStepArgs<A>): Iteration<NatlangStepState<A, R>>;
}

/** A compiled natural-language function: asynchronous, typed, and monitored. */
type NatlangFunction<A extends unknown[] = any[], R = any> = ((...args: A) => Promise<R>) & NatlangCallableMethods<A, R>;
/** An inline \`nl\` value can be instantiated again with a fresh snapshot of the same named captures. */
type InlineNatlangFunction<A extends unknown[] = any[], R = any> = NatlangFunction<A, R> & {
  with<C extends Record<string, unknown>>(captures: C): InlineNatlangFunction<A, R>;
};

/** An nl function without a type argument; the compiler infers its signature from its uses. */
type NatlangUntypedFunction = ((...args: any[]) => Promise<any>) & {
  iterateOn<S>(initial: S, ...args: any[]): Iteration<S>;
};
type NlResult<F> = [F] extends [NlUnspecified] ? NatlangUntypedFunction :
  [F] extends [(...args: infer A) => infer R] ? InlineNatlangFunction<A, Awaited<R>> : InlineNatlangFunction<any[], F>;

/** A folder handle with directory-reducer authority. */
interface FolderSnapshot extends Folder { readonly digest: string; branch(): Folder; }
interface FolderProposal<R> { readonly folder: FolderSnapshot; readonly value: R; readonly baseRevision: number; readonly diff: unknown; }
interface FolderIterationResult<S> { readonly folder: FolderSnapshot; readonly state: S; }
interface FolderIteration<S> {
  until(done: (state: S, folder: FolderSnapshot) => boolean | Promise<boolean>): Promise<FolderIterationResult<S>>;
  streamUntil(done: (state: S, folder: FolderSnapshot) => boolean | Promise<boolean>): AsyncIterable<IterationEvent<FolderIterationResult<S>>> & { readonly __natlangIterationStream: true };
  checkProgress(judge: ProgressJudge<FolderIterationResult<S>> | "off"): FolderIteration<S>;
  withMeasure(remaining: (state: S, folder: FolderSnapshot) => number): FolderIteration<S>;
  withLimit(limit: { maxSteps?: number; deadlineMs?: number }): FolderIteration<S>;
  withSiteId(id: string): FolderIteration<S>;
  onStep(observer: (event: IterationEvent<FolderIterationResult<S>>) => void | Promise<void>): FolderIteration<S>;
}
interface Folder {
  readonly path: string;
  readonly name: string;
  file(path: string): FileHandle;
  dir(path: string): Folder;
  exists(): Promise<boolean>;
  entries(pattern?: string): Promise<(Folder | FileHandle)[]>;
  files(pattern?: string): Promise<FileHandle[]>;
  folders(pattern?: string): Promise<Folder[]>;
  diff(): Promise<unknown>;
  remove(): Promise<void>;
  moveTo(destination: string): Promise<void>;
  snapshot(): FolderSnapshot; at(sourceId:string):FolderSnapshot;
  propose<A extends unknown[], R>(reducer: NatlangFunction<[Folder, ...A], R>, ...inputs: A): Promise<FolderProposal<R>>;
  accept<R>(proposal: FolderProposal<R>): Promise<FolderSnapshot>;
  select(snapshot: FolderSnapshot): Promise<void>;
  iterateOn<S, A extends unknown[]>(reducer: NatlangFunction<[Folder, S, ...A], S>, initial: S, ...inputs: A): FolderIteration<S>;
  apply<A extends unknown[], R>(reducer: NatlangFunction<[Folder, ...A], R>, ...inputs: A): Promise<R>;
}
interface FileHandle {
  readonly path: string;
  readonly name: string;
  exists(): Promise<boolean>;
  readText(startLine?: number, endLine?: number): Promise<string>;
  readJson(): Promise<unknown>;
  writeText(content: string): Promise<void>;
  writeJson(value: unknown): Promise<void>;
  editText(find: string, replaceWith: string, fuzzy?: boolean): Promise<unknown>;
  remove(): Promise<void>;
  moveTo(destination: string): Promise<void>;
}
type Blob = string;

/** The body of a soft function literal: a model-written Neuralese block in an \`nl\` template. */
interface NeuraleseBody { readonly __natlangNeuraleseBody: true }
/** The template tag \`nl.with({ … })\` returns: a function whose captures are exactly the listed ones. */
interface NlWithTag<F> {
  (strings: TemplateStringsArray, body: NeuraleseBody): any;
  <G = F>(strings: TemplateStringsArray, ...values: unknown[]): NlResult<G>;
}
/** The \`nl\` template tag. */
interface NlTag {
  <F = NlUnspecified>(strings: TemplateStringsArray, ...values: unknown[]): NlResult<F>;
  /**
   * An \`nl\` function whose captures are exactly these: plain entries are snapshots taken now; \`live(x)\` entries
   * are read at each call and written back. No other names are captured. Use \`<CaptureRecord, Result>\` to check the
   * finite capture record separately from the result type; child arguments are passed to the returned function and
   * inferred from that call. A full callable signature can still be supplied as the one type argument.
   * @natlangIntrinsic nl.with
   */
  with<CaptureRecord extends object, Result>(captures: CaptureRecord): NlWithTag<Result>;
  with<F = NlUnspecified>(captures: { readonly [name: string]: unknown }): NlWithTag<F>;
}

${NEURALESE_TYPE_DECLARATION}
`;

const FUNCTIONS = String.raw`
/**
 * Create an anonymous natural-language function. Exact mentions of visible names capture live bindings;
 * \`nl.with({ … })\` lists the captures explicitly instead.
 * @natlangIntrinsic nl
 */
const nl: NlTag;
/**
 * Mark an explicit capture of a let binding as live: read at each call and written back after a successful eval.
 * Only meaningful inside \`nl.with({ … })\`.
 * @natlangIntrinsic live
 */
function live<T>(binding: T): T;
/**
 * Run a step repeatedly from an initial state until a predicate holds, with progress review.
 * @natlangIntrinsic iterateOn
 */
function iterateOn${ITERATE_ON_SIGNATURE};
`;

/** Ambient global declarations used by eval programs and virtual projects. */
const STRING_NEURALESE_CONCAT = `interface String {
  concat<T, D extends string>(...strings: (string | Neuralese<T, D>)[]): string;
}\n`;
export const INTRINSICS_GLOBAL_DTS = `${DECLARATIONS}\n${FUNCTIONS.replace(/\n(function|const) /g, '\ndeclare $1 ')}\n${STRING_NEURALESE_CONCAT}`;

/** Module declaration text for packages that export the natlang surface. */
export const INTRINSICS_MODULE_DTS = `${DECLARATIONS.replace(/\n(interface|type) /g, '\nexport $1 ')}\n` +
  FUNCTIONS.replace(/\n(function|const) /g, '\nexport declare $1 ');

export const INTRINSICS_FILE = '/__natlang__/intrinsics.d.ts';
/** Module form, resolved for `@natlang/*` imports in virtual programs. */
export const SURFACE_MODULE_FILE = '/__natlang__/surface.d.ts';
