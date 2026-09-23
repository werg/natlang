/**
 * Declarations for the natlang TypeScript surface.
 *
 * `nl` and `iterateOn` are ordinary TypeScript declarations whose extra meaning is
 * supplied by the natlang compiler. The compiler recognizes them by the
 * `@natlangIntrinsic` JSDoc tag on the resolved declaration, never by spelling alone.
 */

/** Version of generated lowering. A runtime refuses output from another major version. */
export const NATLANG_COMPILE_VERSION = 2 as const;

/** Names that cannot be used as child attributes of a natlang callable object. */
export const RESERVED_CALLABLE_PROPERTIES: ReadonlySet<string> = new Set([
  'call', 'apply', 'bind', 'name', 'length', 'prototype', 'constructor', '__proto__', 'caller', 'arguments',
  'toString', 'toLocaleString', 'valueOf', 'hasOwnProperty', 'isPrototypeOf', 'propertyIsEnumerable',
  '__defineGetter__', '__defineSetter__', '__lookupGetter__', '__lookupSetter__', 'then', 'iterateOn',
]);

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
  checkProgress(judge: ProgressJudge<T>): Iteration<T>;
  withSiteId(id: string): Iteration<T>;
  withLimit(limit: { maxSteps?: number; deadlineMs?: number }): Iteration<T>;
}

/** Methods every compiled natlang callable provides. */
interface NatlangCallableMethods<A extends unknown[], R> {
  iterateOn(initial: NatlangStepState<A, R>, ...args: NatlangStepArgs<A>): Iteration<NatlangStepState<A, R>>;
}

/** A compiled natural-language function: asynchronous, typed, and monitored. */
type NatlangFunction<A extends unknown[] = any[], R = any> = ((...args: A) => Promise<R>) & NatlangCallableMethods<A, R>;

/** An nl function without a type argument; the compiler infers its signature from its uses. */
type NatlangUntypedFunction = ((...args: any[]) => Promise<any>) & {
  iterateOn<S>(initial: S, ...args: any[]): Iteration<S>;
};
type NlResult<F> = [F] extends [NlUnspecified] ? NatlangUntypedFunction :
  [F] extends [(...args: infer A) => infer R] ? NatlangFunction<A, Awaited<R>> : NatlangFunction<any[], F>;

/** A folder handle with directory-reducer authority. */
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
`;

const FUNCTIONS = String.raw`
/**
 * Create an anonymous natural-language function. Exact mentions of visible names capture live bindings.
 * @natlangIntrinsic nl
 */
function nl<F = NlUnspecified>(strings: TemplateStringsArray, ...values: unknown[]): NlResult<F>;
/**
 * Run a step repeatedly from an initial state until a predicate holds, with progress review.
 * @natlangIntrinsic iterateOn
 */
function iterateOn<T, A extends unknown[]>(step: (state: T, ...args: A) => T | Promise<T>, initial: T, ...args: A): Iteration<T>;
`;

/** Ambient global declarations used by eval programs and virtual projects. */
export const INTRINSICS_GLOBAL_DTS = `${DECLARATIONS}\n${FUNCTIONS.replace(/\nfunction /g, '\ndeclare function ')}`;

/** Module declaration text for packages that export the natlang surface. */
export const INTRINSICS_MODULE_DTS = `${DECLARATIONS.replace(/\n(interface|type) /g, '\nexport $1 ')}\n` +
  FUNCTIONS.replace(/\nfunction /g, '\nexport declare function ');

export const INTRINSICS_FILE = '/__natlang__/intrinsics.d.ts';
/** Module form, resolved for `@natlang/*` imports in virtual programs. */
export const SURFACE_MODULE_FILE = '/__natlang__/surface.d.ts';
