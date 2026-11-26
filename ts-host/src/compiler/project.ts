/**
 * `natlang check` and `natlang build`: whole-project analysis and lowering.
 *
 * - Every `.nl` function gets a generated `foo.d.nl.ts` so TypeScript (and plain editors) see typed imports.
 * - Callable folders (`foo/` beside `foo.nl`, and `natlang.d/`) are checked with the constrained-source
 *   policy and embedded as records; they execute at runtime as module instances.
 * - Ordinary project files are type-checked, `nl` expressions are planned against the checker, and the
 *   emitted JavaScript calls the runtime through `__natlang`.
 */
import { checkSequentialNlLoops } from './sequential-loops.js';
import { checkFixedCursorIterations, checkNlIdiomsIn } from './idiom.js';
import { refinedSlotsOf, type RefinedSlot } from './refined-slots.js';
import ts from 'typescript';
import {withSourceLoadSession,sourceTypeExports} from '../runtime/source-session.js';
import {sourceTypeReuseEnabled,type SourceTypeExports} from '../runtime/source-type-context.js';
import { REWRITE_RULES, RewriteGate, rewriteCombinators, type RewriteRecord, type RewriteRule, type RuleMeasurement } from './rewrites.js';
import { describeProgram, namedDescriptor, inlineDescriptor } from '../adaptation/inventory.js';
import type { ComponentDescriptor, ProgramDescriptor } from '../adaptation/types.js';
import { declarationNamespace } from '../native/external.js';
import { analyzeInlineLambdas, fromTypeScript, spanOf, type InlineLambdaPlan, type NatlangDiagnostic } from './inline.js';
import { checkConstrainedSource } from './policy.js';
import { fusionCallSites } from './call-flow.js';
import { analysisLowering, natlangTransformer } from './lower.js';
import { typeScriptText } from './eval-check.js';
import { INTRINSICS_FILE, NATLANG_COMPILE_VERSION, STD_MODULE_FILE, SURFACE_MODULE_FILE } from './intrinsics.js';
import { createNatlangCompilerHost } from './host.js';
import { loadNzSync, nzDeclaration, toBase64 } from '../native/nz-file.js';
import { compileModule } from '../runtime/modules.js';
import { findApplicationContext, loadCallableFolder, loadNamedFunction, NatlangSourceError, usesTargetsOf, findPackageRoot, resolveImportedSource,
  type ItemRecord, type NatlangRecord } from '../runtime/loader.js';
import type { SourceFiles } from '../runtime/loader.js';
import { joinSourcePath as join, normalizeSourcePath as normalize } from '../runtime/source-path.js';
import { NEUTRAL_RUNTIME_SPECIFIER } from './runtime-specifier.js';

/** File access for the project compiler: the real filesystem on Node, virtual files in browsers. */
export interface ProjectFiles {
  isFile(path: string): boolean;
  isDirectory(path: string): boolean;
  list(dir: string): string[];
  read(path: string): string;
  /** Binary reads, for `.nz` files. */
  readBytes?(path: string): Uint8Array;
  write?(path: string, text: string): void;
  /** Underlying compiler access for libraries and dependencies (Node: ts.sys). */
  compiler?: import('./host.js').CompilerFiles;
}

// POSIX path helpers (absolute paths); the Node adapter normalizes platform paths before calling in.
const dirname = (path: string) => { const at = path.lastIndexOf('/'); return at <= 0 ? '/' : path.slice(0, at); };
const resolve = (...parts: string[]) => { let path = ''; for (const part of parts) path = part.startsWith('/') ? part : `${path}/${part}`; return normalize(path); };
const relative = (from: string, to: string) => {
  const a = normalize(from).split('/').filter(Boolean), b = normalize(to).split('/').filter(Boolean);
  let index = 0; while (index < a.length && a[index] === b[index]) index++;
  return [...a.slice(index).map(() => '..'), ...b.slice(index)].join('/');
};
const sep = '/';
const basename = (path: string, extension?: string) => { const base = path.slice(path.lastIndexOf('/') + 1);
  return extension && base.endsWith(extension) ? base.slice(0, -extension.length) : base; };
const extname = (path: string) => { const base = basename(path); const at = base.lastIndexOf('.'); return at > 0 ? base.slice(at) : ''; };

function sourceFiles(files: ProjectFiles, root: string): SourceFiles {
  return { join, dirname, basename, extname, isFile: path => files.isFile(path), isDirectory: path => files.isDirectory(path),
    read: path => files.read(path), list: path => files.list(path), relative: path => relative(root, path), root };
}

export { NEUTRAL_RUNTIME_SPECIFIER };

export type BuildOptions = {
  /** Diagnostic opt-out of reusable pure linked-type metadata. */
  reuseSourceTypes?: boolean;
  /** Opt out of shared generated record modules (diagnostic/reference mode). */
  inlineRecords?: boolean;
  /** Enforce the finite generated-program profile on every editable application module. */
  constrained?: boolean;
  services?: ProgramDescriptor['services'];
  programId?: string;
  guidance?: string;
  importedGuidancePrograms?: readonly string[];
  /** Collect the program descriptor for adaptation. Defaults to true. */
  adaptation?: boolean;
  /** Project directory or tsconfig path (absolute POSIX path when `files` is supplied). */
  project: string;
  /** File access; defaults to the Node filesystem (see `buildProject` in compiler/node-project.ts). */
  files: ProjectFiles;
  /** Emitted module format; browsers executing a virtual project use `commonjs`. */
  module?: 'esm' | 'commonjs';
  outDir?: string;
  target?: 'node' | 'browser';
  /** Specifier of the runtime package the emitted code imports (default: the one files import `nl` from). */
  runtimeSpecifier?: string;
  /** Write `.d.nl.ts` declarations and emitted JavaScript. */
  write?: boolean;
  /** Emit JavaScript (build) or only check. */
  emit?: boolean;
  /** Return/write generated declaration documents (default true). Internal
   * signatures are still generated and checked when false. */
  declarations?: boolean;
  /** Write generated `.d.nl.ts` files beside their sources (default true; read-only installed packages use false). */
  writeDeclarations?: boolean;
  /**
   * Inline each `uses:` function into the declarations that use it instead of an `import type` of its own declaration
   * (the form before WS1 L6). Default false; `NATLANG_INLINE_USES_DECLARATIONS=1` also sets it.
   */
  inlineUses?: boolean;
  /**
   * Bind the project to a specific runtime module: these specifiers type-check against `types` and are
   * emitted as `url`, so the compiled app shares one runtime instance with its launcher. The neutral specifier
   * (`NEUTRAL_RUNTIME_SPECIFIER`) is always bound with them.
   */
  runtimeModule?: { specifiers: string[]; url: string; types: string; path?: string;
    /** The standard library bound with it (`<specifier>/std`); by default `std/index.js` beside an `index.js` runtime. */
    std?: { url: string; types: string; path?: string } };
  /** Type these runtime specifiers, and the neutral one, against the built-in surface declarations (virtual projects). */
  surfaceSpecifiers?: string[];
  /** Type-check these specifiers against a runtime's declaration file when the project cannot resolve them. */
  runtimeTypes?: { specifiers: string[]; types: string };
  /** Type roots shipped with the runtime (Node's `@types`), used for `types` entries the project cannot resolve. */
  runtimeTypeRoots?: string[];
  /**
   * Law-based combinator rewrites (spec/NEURALESE_REWRITES.md) applied to emitted code that imports `natlang:neuralese`.
   * Without a gate, or with no recorded measurement for the current model and dialect, no rule fires.
   */
  rewriteGate?: RewriteGate;
};

export type DefinitionManifest = { version: typeof NATLANG_COMPILE_VERSION;
  inline: { id: string; source: string; line: number; signature: string }[];
  named: { id: string; source: string; signature: string }[];
  sites: string[];
  adaptation?: ProgramDescriptor };

export type BuildResult = { ok: boolean; diagnostics: NatlangDiagnostic[]; outputs: Record<string, string>;
  linkedTypes?: SourceTypeExports;
  declarations: Record<string, string>; manifest: DefinitionManifest; outDir: string;
  /** Rewrites applied per emitted file (empty until a rule is enabled). */
  rewrites?: Record<string, RewriteRecord[]>;
  /** Every `Is<T, "predicate">` slot of the named and inline functions and the predicate the runtime will check there. */
  refinedSlots?: RefinedSlot[] };

const SKIPPED_DIRS = new Set(['node_modules', 'dist', '.git', '.natlang']);

/** Directories that are callable folders: `natlang.d/`, and `foo/` beside `foo.nl`. */
function callableFolders(files: ProjectFiles, root: string): { named: string[]; contexts: string[]; folders: Set<string> } {
  const named: string[] = [], contexts: string[] = [], folders = new Set<string>();
  const mark = (dir: string) => {
    folders.add(dir);
    for (const entry of files.list(dir)) {
      const path = join(dir, entry);
      if (files.isDirectory(path)) mark(path);
    }
  };
  // A package nested inside another package (its own natlang.json below one) is its own project, e.g. a task
  // repository shipped as data; the member packages of a workspace root without natlang.json (applications/) are built.
  const walk = (dir: string, inPackage: boolean) => {
    for (const entry of files.list(dir).sort()) {
      if (SKIPPED_DIRS.has(entry) || entry.startsWith('.')) continue;
      const path = join(dir, entry);
      if (files.isDirectory(path)) {
        if (folders.has(path)) continue;
        const isPackage = files.isFile(join(path, 'natlang.json'));
        if (isPackage && inPackage) continue;
        if (entry === 'natlang.d') { contexts.push(path); mark(path); continue; }
        if (files.isFile(`${path}.nl`)) { mark(path); continue; }
        walk(path, inPackage || isPackage);
      } else if (entry.endsWith('.nl')) named.push(path);
    }
  };
  walk(root, files.isFile(join(root, 'natlang.json')));
  return { named, contexts, folders };
}

const aliasBlock = (types: Record<string, string>) => {
  const known = new Set(Object.keys(types));
  return Object.entries(types).map(([name, text]) => `type ${name} = ${typeScriptText(text, known)};`).join('\n');
};

/** A used function that has its own generated declaration: the declaration refers to it instead of inlining it. */
export type DeclarationReference = { name: string; specifier: string };

/** TypeScript declaration text for an item and its callable-folder children. */
function itemType(record: ItemRecord, known: ReadonlySet<string>, references?: ReadonlyMap<ItemRecord, DeclarationReference>): string {
  const children = Object.entries(record.codebase).map(([name, child]) => {
    const reference = references?.get(child);
    return `readonly ${name}: ${reference ? `typeof ${reference.name}` : itemType(child, known, references)}`;
  });
  const members = children.length ? ` & { ${children.join('; ')} }` : '';
  if (record.kind === 'namespace') return `{ ${children.join('; ')} }`;
  if (record.kind === 'module') {
    const exports = Object.entries(record.exports).filter(([name]) => name !== 'default').map(([name, spec]) =>
      spec.kind === 'function' ? `readonly ${name}: (${Object.entries(spec.args).map(([raw, type]) =>
        `${spec.rest===raw?'...':''}${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`).join(', ')}) => ` +
        `${spec.async ? `Promise<${typeScriptText(spec.returns, known)}>` : typeScriptText(spec.returns, known)}` :
        `readonly ${name}: ${spec.type ? typeScriptText(spec.type, known) : 'unknown'}`);
    const main = record.exports.default;
    const body = `{ ${[...exports, ...children].join('; ')} }`;
    if (main?.kind !== 'function') return body;
    const params = Object.entries(main.args).map(([raw, type]) => `${main.rest===raw?'...':''}${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`);
    return `((${params.join(', ')}) => ${main.async ? `Promise<${typeScriptText(main.returns, known)}>` : typeScriptText(main.returns, known)}) & ${body}`;
  }
  const params = Object.entries(record.args).map(([raw, type]) => `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`);
  if (record.subtype === 'directory-reducer') params.unshift('folder: Folder');
  // A representation-generic result: each call's expected type instantiates it (DECISIONS.md 2026-10-09).
  if (record.generic) return `NatlangGenericFunction<[${params.join(', ')}], ${typeScriptText(record.generic.constraint, known)}, ` +
    `${typeScriptText(record.returns, known)}>${members}`;
  return `NatlangFunction<[${params.join(', ')}], ${typeScriptText(record.returns, known)}>${members}`;
}

/**
 * Generated declaration for `foo.nl`, read by TypeScript as `foo.d.nl.ts`. A `uses:` item in `references` (a function
 * of the same build with its own declaration) is an `import type` of that declaration rather than an inlined copy, so
 * its aliases stay its own and each function is declared once (WS1 L6).
 */
export function natlangDeclaration(record: NatlangRecord, runtimeSpecifier: string,
    references?: ReadonlyMap<ItemRecord, DeclarationReference>): string {
  const collect: Record<string, string> = {};
  const gathered=new Set<ItemRecord>();
  const gather = (item: ItemRecord) => {
    if(gathered.has(item))return;gathered.add(item);
    if (item.kind !== 'namespace') Object.assign(collect, item.types);
    for (const child of Object.values(item.codebase)) if (!references?.has(child)) gather(child);
  };
  gather(record);
  const known = new Set(Object.keys(collect));
  const used = references ? [...new Map([...references.values()].map(reference => [reference.name, reference])).values()] : [];
  const imports = used.map(reference => `import type ${reference.name} from ${JSON.stringify(reference.specifier)};\n`).join('');
  const body = `${aliasBlock(collect)}\ndeclare const fn: ${itemType(record, known, references)};\nexport default fn;\n`;
  // `Is<T, "predicate">` is the refinement brand exported by the runtime (plans/REFINEMENT_TYPES.md).
  const refined = (/\bIs\s*</.test(body) ? ', Is' : '') + (/\bUntrusted\s*</.test(body) ? ', Untrusted' : '') +
    (/\bNatlangGenericFunction</.test(body) ? ', NatlangGenericFunction' : '');
  return `// Generated by natlang from ${record.name}.nl; do not edit.\n` +
    `import type { NatlangFunction, Folder as FolderStore, FolderHandle${refined} } from ${JSON.stringify(runtimeSpecifier)};\n` + imports +
    '/** A natlang `Folder` accepts a whole folder or a directory handle within one. */\ntype Folder = FolderStore | FolderHandle;\n' + body;
}

/** Declaration of `natlang:services` falls to the application; this supplies an empty default. */
const SERVICES_FALLBACK = '/__natlang__/services.d.ts';
/** No measurements: every rewrite rule is off. */
const DEFAULT_REWRITE_GATE = new RewriteGate({ model: 'unmeasured', dialect: 'unmeasured' });

const isRewriteRule = (value: unknown): value is RewriteRule =>
  typeof value === 'string' && (REWRITE_RULES as readonly string[]).includes(value);

const isRuleMeasurement = (value: unknown): value is RuleMeasurement => {
  if (!value || typeof value !== 'object') return false;
  const item = value as Record<string, unknown>;
  return isRewriteRule(item.rule) && typeof item.model === 'string' && typeof item.dialect === 'string' &&
    typeof item.comparison === 'string' && (item.verdict === 'no-harm' || item.verdict === 'harm' || item.verdict === 'inconclusive');
};

/**
 * Build a `RewriteGate` from `natlang.json`'s optional `"rewrites": { "measurements": PATH }` entry (WS2 R5).
 * Rules remain off by default whenever the key or measurements file is absent, malformed, or lacks a certified
 * `no-harm` measurement for the active model and dialect.
 */
export function loadRewriteGate(files: ProjectFiles, root: string): RewriteGate {
  const manifestPath = join(root, 'natlang.json');
  if (!files.isFile(manifestPath)) return DEFAULT_REWRITE_GATE;
  let manifest: Record<string, unknown>;
  try { manifest = JSON.parse(files.read(manifestPath)) as Record<string, unknown>; } catch { return DEFAULT_REWRITE_GATE; }
  const rewritesConfig = manifest?.rewrites;
  if (!rewritesConfig || typeof rewritesConfig !== 'object' || Array.isArray(rewritesConfig)) return DEFAULT_REWRITE_GATE;
  const config = rewritesConfig as Record<string, unknown>;
  if (typeof config.measurements !== 'string' || !config.measurements.trim()) return DEFAULT_REWRITE_GATE;
  const measurementsPath = resolve(root, config.measurements);
  if (!files.isFile(measurementsPath)) return DEFAULT_REWRITE_GATE;
  let parsed: unknown;
  try { parsed = JSON.parse(files.read(measurementsPath)); } catch { return DEFAULT_REWRITE_GATE; }
  const record = parsed && typeof parsed === 'object' && !Array.isArray(parsed) ? parsed as Record<string, unknown> : undefined;
  const rawList = Array.isArray(parsed) ? parsed : Array.isArray(record?.measurements) ? record.measurements : [];
  const measurements = rawList.filter(isRuleMeasurement);
  if (!measurements.length) return DEFAULT_REWRITE_GATE;
  const manifestModel = typeof manifest.model === 'string' ? manifest.model :
    manifest.model && typeof manifest.model === 'object' && !Array.isArray(manifest.model) ?
      ((manifest.model as Record<string, unknown>).model ?? (manifest.model as Record<string, unknown>).stock ?? (manifest.model as Record<string, unknown>).profile) :
      undefined;
  const parsedVersion = record?.version && typeof record.version === 'object' && !Array.isArray(record.version) ?
    record.version as Record<string, unknown> : undefined;
  const model = typeof config.model === 'string' ? config.model :
    typeof parsedVersion?.model === 'string' ? parsedVersion.model :
    typeof record?.model === 'string' ? record.model :
    typeof manifestModel === 'string' ? manifestModel : 'unmeasured';
  const dialect = typeof config.dialect === 'string' ? config.dialect :
    typeof parsedVersion?.dialect === 'string' ? parsedVersion.dialect :
    typeof record?.dialect === 'string' ? record.dialect : 'unmeasured';
  const disabledRaw = Array.isArray(config.disabled) ? config.disabled : Array.isArray(record?.disabled) ? record.disabled : [];
  const disabled = disabledRaw.filter(isRewriteRule);
  return new RewriteGate({ model, dialect }, measurements, disabled.length ? { disabled } : {});
}
const NEURALESE_MODULES = '/__natlang__/neuralese-modules.d.ts';
/** `natlang:neuralese` and `natlang:learning` in terms of the intrinsic `Neuralese` type (spec/neuralese.d.ts). */
const NEURALESE_MODULES_SOURCE = `
declare module 'natlang:neuralese' {
  type N<T> = Neuralese<T>;
  export function map<A, B>(v: N<A>, f: (a: A) => Promise<B>): Promise<N<B>>;
  export function zip<A, B>(a: N<A>, b: N<B>): Promise<N<[A, B]>>;
  export function ap<A, B>(f: N<(a: A) => Promise<B>>, a: N<A> | A): Promise<N<B>>;
  export function combine<T>(...vs: N<T>[]): Promise<N<T>>;
  export function empty<T>(): N<T>;
  export function split<T extends object>(v: N<T>): Promise<{ [K in keyof T]: N<T[K]> }>;
  export function splitList<E>(v: N<E[]>): Promise<N<E>[]>;
  export function read<T>(v: N<T>): Promise<T>;
  export function convert<T>(v: N<T>, to: string): Promise<N<T>>;
  /** f after g, exactly; the map-fusion rewrite writes map(v, compose(f, g)) for map(map(v, g), f). */
  export function compose<A, B, C>(f: (b: B) => C | Promise<C>, g: (a: A) => B | Promise<B>): (a: A) => Promise<C>;
  export function gloss(v: N<unknown>): Promise<string>;
  /** The query operator, read(map(v, question)): the answer to question from what v holds. */
  export function ask(v: N<unknown>, question: string): Promise<string>;
}
declare module 'natlang:learning' {
  export interface NativeAdapter { readonly $nativeAdapter: { readonly id: string } }
  export type NativeTensorKind = 'parameter-set' | 'parameter-gradient' | 'parameter-update' | 'state-gradient' |
    'optimizer-state' | 'adapter-basis' | 'adapter-gradient' | 'derivative-program';
  export interface NativeTensor { readonly $nativeTensor: { readonly id: string; readonly kind: NativeTensorKind } }
  export interface Loss { readonly __natlangLoss: true; valueOf(): number }
  export interface Gradient<A> { readonly __natlangGradient: A }
  export interface Trajectory { readonly id: string }
  export interface OptimizerState<A> { readonly __natlangOptimizer: A }
  export interface Optimizer {
    init<A>(a: A): OptimizerState<A>;
    step<A>(state: { value: A; opt: OptimizerState<A> }, grad: Gradient<A>): Promise<{ value: A; opt: OptimizerState<A> }>;
  }
  export function grad<A>(f: (a: A) => Promise<Loss>, a: A, options?: { order?: 1 | 2 }): Promise<Gradient<A>>;
  export function valueAndGrad<A>(f: (a: A) => Promise<Loss>, a: A, options?: { order?: 1 | 2 }): Promise<{ loss: Loss; grad: Gradient<A> }>;
  export function stopGradient<T>(v: T): T;
  export function withNativeStates<T>(states: string | readonly string[], fn: () => Promise<T> | T): Promise<T>;
  export function withNativeAdapters<T>(adapters: NativeAdapter | readonly (NativeAdapter | { adapter: NativeAdapter; scale?: number })[], fn: () => Promise<T> | T): Promise<T>;
  export const nativeStates: {
    gradient(graph: { schema: 'natlang.neuralese-native-graph/1'; split: string }, id: string,
      nativeIdentity: { architecture: string; checkpoint_sha256: string; tokenizer_fingerprint: string; embedding_weight_sha256: string },
      options?: { nativeStateModes?: Record<string, 'same-base-adapter-state'>; objective?: 'graph-ce' }): Promise<unknown>;
    optimize(gradient: unknown,
      nativeIdentity: { architecture: string; checkpoint_sha256: string; tokenizer_fingerprint: string; embedding_weight_sha256: string },
      optimizer: { name: 'sgd' | 'adam'; learning_rate?: number; weight_decay?: number; beta1?: number; beta2?: number; eps?: number; momentum?: number },
      optimizerStateId?: string): Promise<unknown>;
  };
  export const nativeGraph: {
    gradient(f: () => Promise<Loss>, target: { kind: 'model-parameters'; names: string[] } | { kind: 'native-state' | 'adapter'; id: string; names?: string[] },
      nativeIdentity: { architecture: string; checkpoint_sha256: string; tokenizer_fingerprint: string; embedding_weight_sha256: string },
      options?: { nativeStateModes?: Record<string, 'same-base-adapter-state'> }): Promise<unknown>;
  };
  /** Run fn with soft forms of the runtime prompt pieces (piece ID → Neuralese<SystemPrompt>); differentiable like any soft value. */
  export function withSystemPrompts<T>(prompts: Record<string, Neuralese<string>>, fn: () => Promise<T> | T): Promise<T>;
  export const nativeAdapters: {
    create(nativeIdentity: { architecture: string; checkpoint_sha256: string; tokenizer_fingerprint: string; embedding_weight_sha256: string },
      options: { parameterNames: string[]; kind?: 'xs'; rank?: number; seed?: number }): Promise<NativeAdapter>;
    ref(id: string): NativeAdapter;
    optimize(gradient: unknown,
      nativeIdentity: { architecture: string; checkpoint_sha256: string; tokenizer_fingerprint: string; embedding_weight_sha256: string },
      optimizer: { name: 'sgd' | 'adam'; learning_rate?: number; weight_decay?: number; beta1?: number; beta2?: number; eps?: number; momentum?: number },
      optimizerStateId?: string): Promise<unknown>;
  };
  export type FilePatch = { readonly schema: 'natlang.file-patch/1'; readonly base: string; readonly changes: Readonly<Record<string, string | null>> };
  export const deltas: {
    diff(after: NativeAdapter, base: NativeAdapter): Promise<NativeAdapter>;
    apply(base: NativeAdapter, delta: NativeAdapter, scale?: number): Promise<NativeAdapter>;
    compose(deltas: readonly (NativeAdapter | { delta: NativeAdapter; scale?: number })[]): Promise<NativeAdapter>;
    interference(base: NativeAdapter, deltas: readonly NativeAdapter[], measure: (value: NativeAdapter) => Promise<number>):
      Promise<{ base: number; alone: number[]; joint: number; interference: number }>;
    learnMerge(base: NativeAdapter, deltas: readonly NativeAdapter[], loss: (value: NativeAdapter) => Promise<Loss>,
      options?: { steps?: number; lr?: number; init?: number | number[] }):
      Promise<{ coefficients: number[]; value: NativeAdapter; trace: { loss: number; coefficients: number[] }[] }>;
    diffFiles(after: Readonly<Record<string, string>>, base: Readonly<Record<string, string>>): FilePatch;
    applyPatch(files: Readonly<Record<string, string>>, patch: FilePatch, options?: { strict?: boolean }): Record<string, string>;
    composePatches(patches: readonly FilePatch[]): FilePatch;
  };
  export function trajectory(run: Promise<unknown> | (() => Promise<unknown>)): Promise<Trajectory>;
  export const objectives: {
    crossEntropy(output: Promise<unknown>, expected: unknown): Promise<Loss>;
    selfDistill(output: Promise<unknown>, withFullSource: () => Promise<unknown>): Promise<Loss>;
    decision(output: Promise<unknown> | (() => Promise<unknown>), expected: unknown, rule?: 'logLoss' | 'brier' | 'rps'): Promise<Loss>;
    expectedReward(output: Promise<unknown> | (() => Promise<unknown>), reward: (value: unknown) => number | Promise<number>): Promise<Loss>;
    policyGradient(samples: { trajectory: Trajectory; reward: number }[], options?: { baseline?: 'mean' | number }): Promise<Loss>;
    conditionedDistill(student: Promise<unknown> | (() => Promise<unknown>), teacher: Promise<unknown> | (() => Promise<unknown>), options?: { privileged?: unknown; rule?: 'logLoss' | 'brier' | 'rps' }): Promise<Loss>;
    logLikelihood(trajectory: Trajectory, weight?: number): Promise<Loss>;
    law(name: 'mapIdentity' | 'mapFusion' | 'readMapCommutation' | 'combineAssociativity' | 'combineIdentity' | 'splitZip',
      ...args: unknown[]): Promise<Loss>;
    sum(...losses: Loss[]): Promise<Loss>;
    scale(loss: Loss, weight: number): Promise<Loss>;
  };
  export const optimizers: {
    sgd(options: { lr: number; momentum?: number; weightDecay?: number }): Optimizer;
    adam(options: { lr: number; betas?: [number, number]; weightDecay?: number }): Optimizer;
  };
  export function save(path: string, exports: Record<string, unknown>): Promise<{ path: string; exports: string[] }>;
  export function load(path: string): Promise<{ path: string; header: unknown; exports: Record<string, unknown>; types: Record<string, string> }>;
}
`;

const GEPA_MODULE = '/__natlang__/gepa-module.d.ts';
/** `natlang:gepa`: the GEPA selection math shared by the component-search engine and the program improver (src/gepa). */
const GEPA_MODULE_SOURCE = `
declare module 'natlang:gepa' {
  export type Score = { caseId: string; quality: number };
  export type Member = { id: string; quality: number; scores: readonly Score[]; cost?: number; modelCalls?: number };
  export type Objective = 'quality' | 'source-size' | 'model-calls';
  export type BetterRule<T> = { eligible?: (member: T) => boolean; measure?: (member: T) => number | null };
  export function caseIds(members: readonly Member[]): string[];
  export function frontier(members: readonly Member[]): string[];
  export function frontierSet(members: readonly Member[]): Set<string>;
  export function wonCases(members: readonly Member[]): Record<string, string[]>;
  export function draw(seed: number): { value: number; next: number };
  export function pick<T>(choices: readonly T[], value: number): T;
  export function parentChoices(members: readonly Member[]): string[];
  export function drawParent(members: readonly Member[], seed: number): { id: string; next: number };
  export function better<T extends { quality: number | null }>(left: T, right: T, rule?: BetterRule<T>): boolean;
  export function objectiveMeasure(objective?: Objective): ((member: { cost?: number; modelCalls?: number }) => number | null) | undefined;
  export function leaders(members: readonly Member[], objective?: Objective): string[];
  export function chooseIncumbent(members: readonly Member[], incumbent: string, objective?: Objective): string;
  export function prune<T extends Member>(members: readonly T[], protectedIds: Iterable<string>, limit: number): T[];
  export function componentProblem(parent: Record<string, unknown>, candidate: unknown, keys: readonly string[]): string;
}
`;

function findTsconfig(files: ProjectFiles, project: string): { configPath?: string; root: string } {
  const absolute = normalize(project);
  if (files.isFile(absolute)) return { configPath: absolute, root: dirname(absolute) };
  const candidate = join(absolute, 'tsconfig.json');
  return files.isFile(candidate) ? { configPath: candidate, root: absolute } : { root: absolute };
}

/** Check and optionally compile a project. File access comes from `options.files`. */
export function compileProject(options: BuildOptions): BuildResult {
  const files=sourceFiles(options.files,findTsconfig(options.files,options.project).root);
  return withSourceLoadSession(files,()=>{
    const result=compileProjectInSession(options,files);
    return options.reuseSourceTypes!==false&&sourceTypeReuseEnabled()?{...result,linkedTypes:sourceTypeExports(files)}:result;
  });
}
function compileProjectInSession(options: BuildOptions, files: SourceFiles): BuildResult {
  const fs = options.files;
  const { configPath, root } = findTsconfig(fs, options.project);
  const diagnostics: NatlangDiagnostic[] = [];
  const rel = (path: string) => relative(root, path);
  const problem = (file: string, message: string, code: NatlangDiagnostic['code'] = 'callable-scope', line = 1, hint?: string) =>
    diagnostics.push({ file, start: 0, end: 0, line, column: 1, code, message, severity: 'error', ...(hint ? { hint } : {}) });
  // A source error names its own file (a folder's item, not the folder) and, when it knows it, the line.
  // Positioned diagnostics (a refused loop, a TypeScript syntax error) keep their own line, column and code.
  const sourceProblem = (fallback: string, error: unknown) => {
    if (!(error instanceof NatlangSourceError)) return problem(rel(fallback), String(error));
    if (!error.diagnostics?.length)
      return problem(rel(error.path), error.reason, (error.code ?? 'callable-scope') as NatlangDiagnostic['code'], error.line, error.hint);
    for (const item of error.diagnostics)
      diagnostics.push({ file: rel(error.path), start: 0, end: 0, line: item.line, column: item.column,
        code: item.code as NatlangDiagnostic['code'], message: item.message, severity: 'error' });
  };

  // TypeScript program over ordinary project files, with generated `.nl` declarations.
  const configHost: ts.ParseConfigFileHost = { useCaseSensitiveFileNames: true, getCurrentDirectory: () => root,
    fileExists: path => fs.isFile(path), readFile: path => fs.isFile(path) ? fs.read(path) : undefined,
    readDirectory: (dir, extensions, excludes, includes, depth) => fs.compiler?.readDirectory ?
      fs.compiler.readDirectory(dir, extensions, excludes, includes, depth) : walkTs(fs, dir),
    onUnRecoverableConfigFileDiagnostic: diagnostic => diagnostics.push(fromTypeScript(diagnostic, rel(configPath!))) };
  const parsed = configPath ? ts.getParsedCommandLineOfConfigFile(configPath, {}, configHost) : undefined;
  const compilerOptions: ts.CompilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext, strict: true, skipLibCheck: true,
    ...(parsed?.options ?? {}), allowArbitraryExtensions: true, noEmit: false, declaration: false,
    ...(options.module === 'commonjs' ? { module: ts.ModuleKind.CommonJS, moduleResolution: ts.ModuleResolutionKind.Node10 } : {}) };
  const outDir = resolve(options.outDir ?? compilerOptions.outDir ?? join(root, 'dist'));
  compilerOptions.outDir = outDir;
  compilerOptions.rootDir ??= root;

  // Callable folders and named functions.
  const layout = callableFolders(fs, root);
  const namedRecords = new Map<string, NatlangRecord>();
  for (const path of layout.named) {
    if ([...layout.folders].some(folder => path.startsWith(folder + sep))) continue;
    try { namedRecords.set(path, loadNamedFunction(path, files)); }
    catch (error) { sourceProblem(path, error); }
  }
  const contextRecords = new Map<string, Record<string, ItemRecord>>();
  for (const dir of layout.contexts) {
    try { contextRecords.set(dir, loadCallableFolder(dir, files)); }
    catch (error) { sourceProblem(dir, error); }
  }
  // Nested packages remain independent until an actual TS import (or a TS
  // helper in an available NL codebase) reaches them. Keep their manifests:
  // package-local types/exports still own name resolution.
  const inCallableFolder = (path: string) => [...layout.folders].some(folder => path.startsWith(folder + sep) || path === folder);
  const ordinaryRoots = (parsed?.fileNames?.length ? parsed.fileNames : walkTs(fs, root)).filter(path =>
    (!inCallableFolder(path) || path.endsWith('.d.ts')) && !path.endsWith('.d.nl.ts') && !path.endsWith('.d.nz.ts'));
  const importedSources = new Set(ordinaryRoots), compilerSources = new Set(ordinaryRoots), pendingSources = [...ordinaryRoots], visitedSources = new Set<string>();
  const packageLayouts = new Map<string, ReturnType<typeof callableFolders>>([[root, layout]]), declarationPackages = new Set<string>();
  const enqueue = (path: string, standalone = false): void => {
    const promoted = standalone && !compilerSources.has(path);
    if (standalone) compilerSources.add(path);
    if (promoted && visitedSources.delete(path)) pendingSources.push(path);
    if (!importedSources.has(path)) { importedSources.add(path); pendingSources.push(path); }
  };
  const scannedRecords=new Set<ItemRecord>();
  const scanCodebase = (level: Record<string, ItemRecord>): void => {
    for (const record of Object.values(level)) {
      if(scannedRecords.has(record))continue;scannedRecords.add(record);
      if (record.kind === 'module') enqueue(resolve(root, record.source));
      if(record.kind==='natlang'){
        const path=resolve(root,record.source),owner=findPackageRoot(dirname(path),files)??root;
        let ownedLayout=packageLayouts.get(owner);
        if(!ownedLayout){ownedLayout=callableFolders(fs,owner);packageLayouts.set(owner,ownedLayout);}
        // Public shared dependencies receive their own declaration and lexical
        // alias environment. Private children remain members of their owner.
        if(ownedLayout.named.includes(path)&&!namedRecords.has(path)){namedRecords.set(path,record);enqueue(path);}
      }
      scanCodebase(record.codebase);
    }
  };
  for (const record of namedRecords.values()) scanCodebase(record.codebase);
  for (const level of contextRecords.values()) scanCodebase(level);
  for (let cursor = 0; cursor < pendingSources.length; cursor++) {
    const source = pendingSources[cursor]!;
    if (visitedSources.has(source) || !fs.isFile(source)) continue;
    visitedSources.add(source);
    const owner = findPackageRoot(dirname(source), files) ?? root;
    if (owner && !declarationPackages.has(owner)) {
      declarationPackages.add(owner);
      for (const declaration of walkTs(fs, owner)) if (declaration.endsWith('.d.ts')) enqueue(declaration, true);
    }
    if (source.endsWith('.nl')) continue;
    for (const imported of ts.preProcessFile(fs.read(source), true, true).importedFiles) {
      const target = resolveImportedSource(imported.fileName, source, files);
      if (!target || !/\.(?:ts|nl)$/.test(target) || relative(root, target).startsWith('..')) continue;
      if (target.endsWith('.nl')) {
        if (!compilerSources.has(source) || namedRecords.has(target)) continue;
        const packageDirectory = findPackageRoot(dirname(target), files) ?? root;
        let targetLayout = packageLayouts.get(packageDirectory);
        if (!targetLayout) { targetLayout = callableFolders(fs, packageDirectory); packageLayouts.set(packageDirectory, targetLayout); }
        // Private NL children remain properties of their actual public parent.
        if (!targetLayout.named.includes(target)) continue;
        try { const record = loadNamedFunction(target, files); namedRecords.set(target, record); scanCodebase(record.codebase); enqueue(target); }
        catch (error) { sourceProblem(target, error); }
      } else enqueue(target, compilerSources.has(source));
    }
  }
  const collectAdaptation = options.adaptation !== false;
  const optimizationComponents: ComponentDescriptor[] = [];
  const optimizationSources: Record<string, string> = {};
  const programId = options.programId ?? basename(root);
  // Compile every callable-folder module now so errors surface at build time rather than first call.
  // A function listed in several functions' `uses:` (or also loaded as a named function) is described once.
  const described = new Set<string>();
  const describeNamed = (record: NatlangRecord) => {
    if (!collectAdaptation || described.has(record.source)) return;
    described.add(record.source);
    optimizationComponents.push(namedDescriptor(programId, record));
  };
  const checkedItems=new Set<ItemRecord>();
  const checkModules = (level: Record<string, ItemRecord>) => {
    for (const item of Object.values(level)) {
      if(checkedItems.has(item))continue;checkedItems.add(item);
      if (item.kind !== 'namespace') item.programId = programId;
      if (collectAdaptation && item.kind !== 'namespace') optimizationSources[item.source] = item.text;
      if (item.kind === 'natlang') describeNamed(item);
      if (item.kind === 'module' && !described.has(item.source)) {
        described.add(item.source);
        try { compileModule(item, level, collectAdaptation ? plans => optimizationComponents.push(...plans.map(plan => inlineDescriptor(programId, plan))) : undefined); }
        catch (error) { problem(item.source, error instanceof Error ? error.message : String(error)); }
      }
      checkModules(item.codebase);
    }
  };
  for (const record of namedRecords.values()) {
    record.programId = programId;
    if (collectAdaptation) optimizationSources[record.source] = record.text;
    describeNamed(record); checkModules(record.codebase);
  }
  for (const records of contextRecords.values()) checkModules(records);

  const inFolder = (path: string) => [...layout.folders].some(folder => path.startsWith(folder + sep) || path === folder);
  const rootNames = [...compilerSources].filter(path => !path.endsWith('.nl') && !path.endsWith('.d.nl.ts') && !path.endsWith('.d.nz.ts'));
  const withNeutral = (specifiers: readonly string[]) => [...new Set([...specifiers, NEUTRAL_RUNTIME_SPECIFIER])];
  const bound = options.runtimeModule && { ...options.runtimeModule, specifiers: withNeutral(options.runtimeModule.specifiers) };
  // `@natlang/node/std` is emitted as the library module of the same build as the bound runtime, so both share one instance.
  const sibling = (path: string | undefined, from: RegExp, to: string) => path && from.test(path) ? path.replace(from, to) : undefined;
  const stdModule = bound && (bound.std ?? (() => {
    const url = sibling(bound.url, /\/index\.js$/, '/std/index.js'), types = sibling(bound.types, /([\\/])index\.d\.ts$/, '$1std$1index.d.ts');
    return url && types && fs.isFile(types) ? { url, types, path: sibling(bound.path, /([\\/])index\.js$/, '$1std$1index.js') } : undefined;
  })());
  const stdSpecifiers = stdModule ? options.runtimeModule!.specifiers.map(specifier => `${specifier}/std`) : [];
  if (options.runtimeTypeRoots?.length) {
    const resolutionHost = ts.sys ?? { fileExists: (path: string) => fs.isFile(path), readFile: (path: string) => fs.isFile(path) ? fs.read(path) : undefined };
    const missing = (compilerOptions.types ?? ['node']).some(name =>
      !ts.resolveTypeReferenceDirective(name, join(root, 'index.ts'), compilerOptions, resolutionHost).resolvedTypeReferenceDirective);
    if (missing) compilerOptions.typeRoots = [...(compilerOptions.typeRoots ?? []), ...options.runtimeTypeRoots];
  }
  // A program outside a checkout or install (no node_modules above it) resolves the runtime and its standard library
  // (`<specifier>/std`, the sibling std/index.d.ts) to the CLI's own declarations.
  if (options.runtimeTypes) for (const specifier of options.runtimeTypes.specifiers) {
    const stdTypes = sibling(options.runtimeTypes.types, /([\\/])index\.d\.ts$/, '$1std$1index.d.ts');
    for (const [name, types] of [[specifier, options.runtimeTypes.types], [`${specifier}/std`, stdTypes && fs.isFile(stdTypes) ? stdTypes : undefined]] as const) {
      if (!types) continue;
      const resolved = ts.resolveModuleName(name, join(root, 'index.ts'), compilerOptions, ts.sys ?? {
        fileExists: path => fs.isFile(path), readFile: path => fs.isFile(path) ? fs.read(path) : undefined });
      if (!resolved.resolvedModule) compilerOptions.paths = { ...(compilerOptions.paths ?? {}), [name]: [types] };
    }
  }
  if (options.surfaceSpecifiers?.length) compilerOptions.paths = { ...(compilerOptions.paths ?? {}),
    ...Object.fromEntries(withNeutral(options.surfaceSpecifiers).map(specifier => [specifier, [SURFACE_MODULE_FILE]])),
    ...Object.fromEntries(options.surfaceSpecifiers.map(specifier => [`${specifier}/std`, [STD_MODULE_FILE]])) };
  if (bound) compilerOptions.paths = { ...(compilerOptions.paths ?? {}),
    ...Object.fromEntries(bound.specifiers.map(specifier => [specifier, [bound.types]])),
    ...Object.fromEntries(stdSpecifiers.map(specifier => [specifier, [stdModule!.types]])) };
  const runtimeSpecifier = bound?.url ?? options.runtimeSpecifier ?? detectRuntimeSpecifier(fs, rootNames) ?? '@natlang/node';
  const declarationSpecifier = bound?.specifiers[0] ?? runtimeSpecifier;
  const rewrite = new Map((bound?.specifiers ?? []).map(specifier => [specifier, bound!.url]));
  const declarations: Record<string, string> = {};
  const virtual = new Map<string, string>([[SERVICES_FALLBACK, "declare module 'natlang:services' { const services: Record<string, any>; export = services; }\n"]]);
  // A `uses:` item that is itself a named function of this build is referenced by its own declaration (`import type`).
  const declared = new Map([...namedRecords].map(([path, record]) => [resolve(path), record]));
  const referencesOf = (path: string, record: NatlangRecord): Map<ItemRecord, DeclarationReference> | undefined => {
    if (options.inlineUses ?? (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NATLANG_INLINE_USES_DECLARATIONS === '1') return undefined;
    const references = new Map<ItemRecord, DeclarationReference>();
    const visit = (item: ItemRecord) => {
      if (item.kind === 'natlang') for (const [alias, target] of Object.entries(usesTargetsOf(item) ?? {})) {
        const child = item.codebase[alias];
        if (!child || child.kind !== 'natlang' || !target.endsWith('.nl') || !declared.has(resolve(target))) continue;
        let specifier = relative(dirname(path), resolve(target)).split(sep).join('/');
        if (!specifier.startsWith('.')) specifier = `./${specifier}`;
        references.set(child, { name: `__uses_${alias.replace(/[^A-Za-z0-9_$]/g, '_')}_${references.size}`, specifier });
      }
      for (const child of Object.values(item.codebase)) if (!references.has(child)) visit(child);
    };
    visit(record);
    return references.size ? references : undefined;
  };
  for (const [path, record] of namedRecords) {
    const declaration = natlangDeclaration(record, declarationSpecifier, referencesOf(path, record));
    const target = path.replace(/\.nl$/, '.d.nl.ts');
    if (options.declarations !== false) declarations[target] = declaration;
    virtual.set(target, declaration);
  }
  // `.nz` files: typed declarations from their headers, read by TypeScript as `name.d.nz.ts`.
  const nzFiles = new Map<string, Uint8Array>();
  for (const path of walkNz(fs, root)) {
    if (!fs.readBytes) { problem(rel(path), 'reading .nz files needs binary file access'); continue; }
    try {
      const bytes = fs.readBytes(path);
      const loaded = loadNzSync(bytes);
      nzFiles.set(path, bytes);
      const declaration = nzDeclaration(loaded.header, basename(path), typeScriptText);
      const target = path.replace(/\.nz$/, '.d.nz.ts');
      if (options.declarations !== false) declarations[target] = declaration;
      virtual.set(target, declaration);
    } catch (error) { problem(rel(path), error instanceof Error ? error.message : String(error), 'neuralese-file'); }
  }
  const usesNeuraleseModules = rootNames.some(path => /['"]natlang:(neuralese|learning)['"]/.test(fs.read(path)));
  if (usesNeuraleseModules) virtual.set(NEURALESE_MODULES, NEURALESE_MODULES_SOURCE);
  const usesGepa = rootNames.some(path => /['"]natlang:gepa['"]/.test(fs.read(path)));
  if (usesGepa) virtual.set(GEPA_MODULE, GEPA_MODULE_SOURCE);
  const host = createNatlangCompilerHost({ options: compilerOptions, virtual, currentDirectory: root,
    files: fs.compiler ?? { readFile: path => fs.isFile(path) ? fs.read(path) : undefined, fileExists: path => fs.isFile(path),
      directoryExists: path => fs.isDirectory(path), getDirectories: path => fs.isDirectory(path) ?
        fs.list(path).filter(entry => fs.isDirectory(join(path, entry))) : [] } });
  const hasServicesDeclaration = rootNames.some(path => /declare\s+module\s+['"]natlang:services['"]/.test(fs.read(path)));
  const program = ts.createProgram({ rootNames: [INTRINSICS_FILE, ...(hasServicesDeclaration ? [] : [SERVICES_FALLBACK]),
    ...(usesNeuraleseModules ? [NEURALESE_MODULES] : []), ...(usesGepa ? [GEPA_MODULE] : []), ...rootNames],
    options: compilerOptions, host });
  for (const diagnostic of ts.getPreEmitDiagnostics(program))
    if (diagnostic.category === ts.DiagnosticCategory.Error) diagnostics.push(fromTypeScript(diagnostic, diagnostic.file ? rel(diagnostic.file.fileName) : ''));
  const sources = rootNames.map(path => program.getSourceFile(path)).filter((file): file is ts.SourceFile => !!file);
  // Discovery determines entry points, not the extent of the application's real
  // imports. A local source explicitly imported through a tsconfig exclusion or
  // an evaluation test-file filter still needs lowering and provenance hashing.
  const discovered = new Set(sources.map(file => file.fileName));
  for (const file of program.getSourceFiles().filter(file => !discovered.has(file.fileName) && !file.isDeclarationFile &&
      file.fileName.startsWith(root + sep) && !inFolder(file.fileName) && !rel(file.fileName).split(sep).includes('node_modules') &&
      /\.(ts|tsx|mts)$/.test(file.fileName)).sort((a, b) => a.fileName.localeCompare(b.fileName))) sources.push(file);
  if (options.constrained) for (const file of sources) diagnostics.push(...checkConstrainedSource(file, { checker: program.getTypeChecker(), displayPath: source => rel(source.fileName) }));
  for (const file of sources) diagnostics.push(...checkSequentialNlLoops(file, source => rel(source.fileName)));
  // The idiom floor (compiler/idiom.ts): exact, model-free warnings.
  for (const file of sources) diagnostics.push(...checkFixedCursorIterations(file, source => rel(source.fileName)));
  diagnostics.push(...checkNlIdiomsIn([...[...namedRecords.values()].map(record => ({ [record.name]: record })), ...contextRecords.values()],
    source => source.startsWith('/') ? rel(source) : source));
  const revision = (file: ts.SourceFile) => `${rel(file.fileName)}@${file.text.length}`;
  const plans = new Map<ts.SourceFile, InlineLambdaPlan[]>();
  const readouts = new Map<ts.SourceFile, import('./neuralese.js').NeuraleseReadout[]>();
  const instantiations = new Map<ts.SourceFile, import('./neuralese.js').GenericInstantiation[]>();
  for (const file of sources) {
    const analysis = analyzeInlineLambdas(program, [file], { displayPath: source => rel(source.fileName), sourceRevision: revision(file), authored: true });
    diagnostics.push(...analysis.diagnostics);
    analysis.plans.forEach(plan => { plan.programId = programId; });
    plans.set(file, analysis.plans);
    readouts.set(file, analysis.readouts);
    instantiations.set(file, analysis.instantiations);
    if (collectAdaptation) {
      optimizationSources[rel(file.fileName)] = file.text;
      optimizationComponents.push(...analysis.plans.map(plan => inlineDescriptor(programId, plan)));
    }
  }
  // Include dependency resolutions/configuration and conservatively all authored project files.
  const rootIsPackage = fs.isFile(join(root, 'natlang.json'));
  const collectSources = (dir: string): void => {
    for (const name of fs.list(dir).sort()) {
      if (SKIPPED_DIRS.has(name) || name.startsWith('.') || name === 'adaptations' || name.endsWith('.d.nl.ts')) continue;
      const path = join(dir, name);
      if (fs.isDirectory(path)) {
        // A package nested inside this one is its own program (walkSources).
        if (rootIsPackage && fs.isFile(join(path, 'natlang.json'))) continue;
        const runManifest = join(path, 'manifest.json');
        if (fs.isFile(runManifest)) {
          try { if (JSON.parse(fs.read(runManifest))?.schema === 'natlang.adaptation-run/v1') continue; }
          catch { /* An ordinary manifest still contributes to the source fingerprint. */ }
        }
        collectSources(path);
      }
      // The program's examples.json is its test and tuning data, not a source: saving an example changes no build.
      else if (dir === root && name === 'examples.json') continue;
      else if (fs.isFile(path) && /\.(?:[cm]?ts|[cm]?js|nl|json|yaml|yml)$/.test(name)) {
        let text = fs.read(path);
        // Portable artifacts may live at any declared relative path. Their
        // contents cannot participate in the build hash they themselves bind.
        if (name.endsWith('.json')) {
          try { if (/^natlang\.adaptation(?:\/|-)/.test(JSON.parse(text)?.schema ?? '')) continue; } catch { /* Ordinary malformed JSON remains part of the conservative fingerprint. */ }
        }
        if (name === 'natlang.json') {
          try { const manifest = JSON.parse(text); if (manifest.targets) for (const target of Object.values(manifest.targets) as Record<string, unknown>[]) delete target.adaptation;
            text = JSON.stringify(manifest); } catch { /* Parsing diagnostics belong to the manifest loader. */ }
        }
        optimizationSources[rel(path)] = text;
      }
    }
  };
  if (collectAdaptation) {
    collectSources(root);
    // Imported packages retain their actual manifests and source identity too.
    for (const owner of declarationPackages) if (owner !== root && !relative(root, owner).startsWith('..')) collectSources(owner);
    let dependencyRoot = root;
    while (true) {
      for (const lock of ['package-lock.json', 'pnpm-lock.yaml', 'yarn.lock']) {
        const path = join(dependencyRoot, lock);
        if (fs.isFile(path)) optimizationSources['@dependency-lock/' + lock] = fs.read(path);
      }
      const parent = dirname(dependencyRoot); if (parent === dependencyRoot) break; dependencyRoot = parent;
    }
  }
  const manifest: DefinitionManifest = { version: NATLANG_COMPILE_VERSION,
    inline: [...plans.values()].flat().map(plan => ({ id: plan.definitionId, source: plan.sourceSpan.file, line: plan.sourceSpan.line,
      signature: `(${plan.parameters.map(parameter => `${parameter.name}: ${parameter.type.text}`).join(', ')}) => ${plan.returns.text}` })),
    named: [...namedRecords.values()].map(record => ({ id: record.id, source: record.source,
      signature: `(${Object.entries(record.args).map(([name, type]) => `${name}: ${type}`).join(', ')}) => ${record.returns}` })),
    sites: [], ...(collectAdaptation ? { adaptation: describeProgram(programId, optimizationSources, optimizationComponents, options.guidance, options.importedGuidancePrograms,
      options.services ? { declarations: Object.fromEntries(Object.entries(options.services.declarations).map(([name, text]) =>
        [name, /^\s*declare (?:namespace|const) /.test(text) ? text.trim() : declarationNamespace(name, text)])), scopes: options.services.scopes } : undefined) } : {}) };
  const refinedSlots: RefinedSlot[] = [];
  const collectSlots = (level: Record<string, ItemRecord>) => {
    for (const item of Object.values(level)) {
      if (item.kind === 'natlang') refinedSlots.push(...refinedSlotsOf(item.source, item.name, item.args, item.returns, item.types),
        // Postconditions (`ensures:`) are checked at run time too: crisp checks first, then judged predicates.
        ...(item.ensures ?? []).map(clause => ({ source: item.source, function: item.name, slot: 'ensures',
          predicate: 'check' in clause ? clause.check : clause.predicate + (clause.against?.length ? ` (given ${clause.against.join(', ')})` : '') })));
      collectSlots(item.codebase);
    }
  };
  for (const record of namedRecords.values()) { collectSlots({ [record.name]: record }); }
  for (const records of contextRecords.values()) collectSlots(records);
  for (const plan of [...plans.values()].flat()) {
    const planAliases: Record<string, string> = { ...plan.returns.aliases };
    for (const parameter of plan.parameters) Object.assign(planAliases, parameter.type.aliases);
    refinedSlots.push(...refinedSlotsOf(plan.sourceSpan.file, plan.definitionId,
      Object.fromEntries(plan.parameters.map(parameter => [parameter.name, parameter.type.natlang ?? parameter.type.text])),
      plan.returns.natlang ?? plan.returns.text, planAliases));
  }
  // Validate authored static imports in both check and emit modes. A stale declaration can make
  // TypeScript accept an import without its source. Keep rewrite targets only for actual JS emit.
  const emitting = options.emit !== false;
  const generated = new Map<string, { record: NatlangRecord; commonjs: boolean; target: string }>();
  const generatedNz = new Map<string, { bytes: Uint8Array; commonjs: boolean }>();
  const importsBySource = new Map<ts.SourceFile, ReadonlySet<string>>();
  const noNatlangImports = new Set<string>();
  for (const file of program.getSourceFiles()) {
    if (file.isDeclarationFile || program.isSourceFileFromExternalLibrary(file)) continue;
    const found = new Set<string>();
    const commonjs = file.impliedNodeFormat === ts.ModuleKind.CommonJS || compilerOptions.module === ts.ModuleKind.CommonJS;
    for (const statement of file.statements) {
      if ((!ts.isImportDeclaration(statement) && !ts.isExportDeclaration(statement)) ||
        !statement.moduleSpecifier || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
      const specifier = statement.moduleSpecifier.text;
      if (specifier.endsWith('.nz')) {
        const target = resolve(dirname(file.fileName), specifier);
        const bytes = nzFiles.get(target);
        if (!bytes) { problem(rel(file.fileName), `cannot import ${specifier}: no such .nz file`); continue; }
        if (emitting) {
          found.add(specifier);
          generatedNz.set(join(outDir, relative(compilerOptions.rootDir!, target)) + '.js', { bytes, commonjs });
        }
      } else if (specifier.endsWith('.nl')) {
        const target = resolve(dirname(file.fileName), specifier);
        const record = namedRecords.get(target);
        if (!record) { problem(rel(file.fileName), `cannot import ${specifier}: no such natlang function`); continue; }
        if (emitting) {
          found.add(specifier);
          generated.set(join(outDir, relative(compilerOptions.rootDir!, target)) + '.js', { record, target, commonjs });
        }
      }
    }
    if (found.size) importsBySource.set(file, found);
  }
  const ok = !diagnostics.some(item => item.severity === 'error');
  const outputs: Record<string, string> = {};
  const rewrites: Record<string, RewriteRecord[]> = {};
  if (options.write !== false && options.writeDeclarations !== false && fs.write) for (const [path, text] of Object.entries(declarations))
    if (!fs.isFile(path) || fs.read(path) !== text) fs.write(path, text);
  if (ok && emitting) {
    // Calls of named functions that make up a possible hand-off (compiler/call-flow.ts): the runtime matches them to planned edges.
    const fusionSitesOf = (file: ts.SourceFile): Map<string, string> => {
      const names = new Map<string, ItemRecord>();
      for (const statement of file.statements) {
        if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier) || !statement.moduleSpecifier.text.endsWith('.nl')) continue;
        const record = namedRecords.get(resolve(dirname(file.fileName), statement.moduleSpecifier.text));
        if (record && statement.importClause?.name) names.set(statement.importClause.name.text, record);
      }
      return fusionCallSites(file, rel(file.fileName), names);
    };
    const specifierFor = (commonjs: boolean) => bound ? (commonjs && bound.path ? bound.path : bound.url) : runtimeSpecifier;
    const byFile = new Map(sources.map(file => [file.fileName, file]));
    const sharedRecords = !options.inlineRecords && (globalThis as {process?:{env?:Record<string,string|undefined>}}).process?.env?.NATLANG_INLINE_SOURCE_RECORDS !== '1';
    const graphNodes:ItemRecord[]=[], graphIds=new Map<ItemRecord,number>(),graphModes=new Set<boolean>();
    const graphId=(record:ItemRecord):number=>{const prior=graphIds.get(record);if(prior!==undefined)return prior;const id=graphNodes.length;graphIds.set(record,id);graphNodes.push(record);for(const child of Object.values(record.codebase))graphId(child);return id;};
    for(const record of namedRecords.values())graphId(record);
    const contextIds=new Map([...contextRecords].map(([path,codebase])=>[path,graphId({kind:'namespace',name:basename(path),source:rel(path),codebase})]));
    const graphPath=(commonjs:boolean)=>join(outDir,`.natlang-records.${commonjs?'cjs':'mjs'}`);
    const graphSpecifier=(from:string,commonjs:boolean)=>{graphModes.add(commonjs);const path=relative(dirname(from),graphPath(commonjs));return path.startsWith('./')||path.startsWith('../')?path:`./${path}`;};

    const transformer: ts.TransformerFactory<ts.SourceFile> = context => file => {
      const source = byFile.get(file.fileName);
      if (!source) return file;
      const contextDir = findApplicationContext(dirname(file.fileName), files);
      const filePlans = plans.get(source) ?? [];
      const fileReadouts = readouts.get(source) ?? [];
      const commonjs = source.impliedNodeFormat === ts.ModuleKind.CommonJS || compilerOptions.module === ts.ModuleKind.CommonJS;
      const rewriteFor = new Map([...[...rewrite.keys()].map(specifier => [specifier, specifierFor(commonjs)] as const),
        ...stdSpecifiers.map(specifier => [specifier, commonjs && stdModule!.path ? stdModule!.path : stdModule!.url] as const)]);
      const contextIndex=contextDir?contextIds.get(contextDir):undefined;
      let contextName='__natlang_context_records';while(source.text.includes(contextName))contextName+='_';
      const emittedPath=join(outDir,relative(compilerOptions.rootDir!,source.fileName)).replace(/\.[cm]?tsx?$/,'.js');
      const graph=contextIndex!==undefined&&sharedRecords?graphSpecifier(emittedPath,commonjs):undefined;
      const transformed=natlangTransformer({
        ...analysisLowering({ plans: filePlans, readouts: fileReadouts, instantiations: instantiations.get(source) ?? [] },
          item => `${rel(file.fileName)}:${item.line}`),
        fusionSites: fusionSitesOf(source),
        checker: program.getTypeChecker(), runtime: '__natlang',
        context: graph ? (commonjs?`require(${JSON.stringify(graph)}).records[${contextIndex}].codebase`:`${contextName}[${contextIndex}].codebase`) : contextDir && contextRecords.has(contextDir) ? JSON.stringify(contextRecords.get(contextDir)) : undefined,
        constrained: options.constrained ?? false, guardPrefix: JSON.stringify([programId, rel(file.fileName)]), modulePath: rel(file.fileName), browser: options.target === 'browser',
        module: { natlangImports: importsBySource.get(source) ?? noNatlangImports, rewrite: rewriteFor } })(context)(file);
      if(graph&&!commonjs){
        const imported=ts.factory.createImportDeclaration(undefined,ts.factory.createImportClause(false,undefined,ts.factory.createNamedImports([
          ts.factory.createImportSpecifier(false,ts.factory.createIdentifier('records'),ts.factory.createIdentifier(contextName))
        ])),ts.factory.createStringLiteral(graph),undefined);
        return ts.factory.updateSourceFile(transformed,[imported,...transformed.statements]);
      }
      return transformed;
    };
    const rewriteGate = options.rewriteGate ?? loadRewriteGate(fs, root);
    const emit = (fileName: string, text: string) => {
      if ((options.constrained || /\b__natlang\.(?:fuseSite|instantiate|own)\(/.test(text)) && /\.js$/.test(fileName) && !/(?:\b(?:const|let|var)\s+__natlang\b|\bimport\s*\{[^}]*\b__natlang\b)/.test(text)) {
        const commonjs = compilerOptions.module === ts.ModuleKind.CommonJS || /Object\.defineProperty\(exports/.test(text);
        text = (commonjs ? `const __natlang = require(${JSON.stringify(specifierFor(true))}).__natlang;\n` :
          `import { __natlang } from ${JSON.stringify(specifierFor(false))};\n`) + text;
      }
      if (/\.[cm]?js$/.test(fileName) && /['"]natlang:neuralese['"]/.test(text)) {
        const rewritten = rewriteCombinators(text, { gate: rewriteGate, fileName });
        text = rewritten.code;
        if (rewritten.applied.length) rewrites[fileName] = rewritten.applied;
      }
      outputs[fileName] = text;
      if (options.write !== false) fs.write?.(fileName, text);
    };
    const emitted = program.emit(undefined, emit, undefined, false, { before: [transformer] });
    // The crisp checkers of a program's refined predicates (plans/REFINEMENT_TYPES.md section 3): the nearest
    // `refinements.ts` above a `.nl` module is loaded with it, so every host that imports the function finds them.
    const sourceNames = new Set(program.getSourceFiles().map(file => resolve(file.fileName)));
    const refinementsFor = (target: string): string | undefined => {
      const top = resolve(compilerOptions.rootDir!);
      for (let dir = dirname(resolve(target)); dir === top || dir.startsWith(top + sep); dir = dirname(dir)) {
        const candidate = join(dir, 'refinements.ts');
        if (sourceNames.has(candidate)) return candidate;
        if (dir === top) break;
      }
      return undefined;
    };
    for (const [path, { record, commonjs, target }] of generated) {
      const specifier = JSON.stringify(specifierFor(commonjs));
      const graph=sharedRecords?JSON.stringify(graphSpecifier(path,commonjs)):undefined;
      const json=sharedRecords?`__records[${graphId(record)}]`:JSON.stringify(record);
      const crisp = refinementsFor(target);
      let local = '';
      if (crisp) {
        const emitted = join(outDir, relative(compilerOptions.rootDir!, crisp)).replace(/\.ts$/, '.js');
        const from = relative(dirname(path), emitted).split(sep).join('/');
        local = JSON.stringify(from.startsWith('.') ? from : `./${from}`);
      }
      emit(path, commonjs ?
        `"use strict";\nObject.defineProperty(exports, "__esModule", { value: true });\n` +
          (graph?`const __records=require(${graph}).records;\n`:'')+
          (crisp ? `require(${specifier}).__natlang.registerCrisp(require(${local}).refinements, ${local});\n` : '') +
          `exports.default = require(${specifier}).__natlang.named(${JSON.stringify(record.name)}, ${json});\n` :
        `import { __natlang } from ${specifier};\n` + (graph?`import {records as __records} from ${graph};\n`:'') + (crisp ? `import { refinements as __crisp } from ${local};\n__natlang.registerCrisp(__crisp, ${local});\n` : '') +
          `export default __natlang.named(${JSON.stringify(record.name)}, ${json});\n`);
    }
    if(sharedRecords)for(const commonjs of graphModes){
      const aliases:string[]=[],aliasIds=new Map<string,number>();
      const rows=graphNodes.map(record=>{
        const {codebase,...body}=record;
        if('types' in body){const text=JSON.stringify(body.types);if(!aliasIds.has(text)){aliasIds.set(text,aliases.length);aliases.push(text);}delete (body as Partial<typeof body>).types;}
        return body;
      });
      let text=`const __types=[${aliases.join(',')}];\nconst records=${JSON.stringify(rows)};\n`;
      graphNodes.forEach((record,index)=>{
        text+=`records[${index}].codebase={${Object.entries(record.codebase).map(([key,child])=>`${JSON.stringify(key)}:records[${graphId(child)}]`).join(',')}};\n`;
        if('types' in record)text+=`records[${index}].types={...__types[${aliasIds.get(JSON.stringify(record.types))}]};\n`;
      });
      emit(graphPath(commonjs),text+(commonjs?'exports.records=records;\n':'export {records};\n'));
    }
    for (const [path, { bytes, commonjs }] of generatedNz) {
      const specifier = JSON.stringify(specifierFor(commonjs));
      const names = Object.keys(loadNzSync(bytes).header.exports);
      const load = `__natlang.nzModule(${JSON.stringify(toBase64(bytes))})`;
      emit(path, commonjs ?
        `"use strict";\nObject.defineProperty(exports, "__esModule", { value: true });\nconst __nz = require(${specifier}).${load};\n` +
          names.map(name => `exports.${name} = __nz.${name};\n`).join('') :
        `import { __natlang } from ${specifier};\nconst __nz = ${load};\n` + names.map(name => `export const ${name} = __nz.${name};\n`).join(''));
    }
    for (const diagnostic of emitted.diagnostics) if (diagnostic.category === ts.DiagnosticCategory.Error)
      diagnostics.push(fromTypeScript(diagnostic, diagnostic.file ? rel(diagnostic.file.fileName) : ''));
    if (options.write !== false) fs.write?.(join(outDir, 'natlang-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  }
  diagnostics.sort((a, b) => a.file.localeCompare(b.file) || a.start - b.start);
  return { ok: !diagnostics.some(item => item.severity === 'error'), diagnostics, outputs, declarations, manifest, outDir, rewrites,
    ...(refinedSlots.length ? { refinedSlots } : {}) };
}

function walkNz(fs: ProjectFiles, root: string): string[] {
  return walkSources(fs, root, entry => entry.endsWith('.nz'));
}

function walkTs(fs: ProjectFiles, root: string): string[] {
  return walkSources(fs, root, entry => /\.(ts|tsx|mts)$/.test(entry));
}

/**
 * The files below `root` that `wanted` selects. A package nested inside another package (its own natlang.json below
 * one, such as a program `natlang init` created inside this one) is its own project and is left out, as in
 * `callableFolders`.
 */
function walkSources(fs: ProjectFiles, root: string, wanted: (entry: string) => boolean): string[] {
  const out: string[] = [];
  const walk = (dir: string, inPackage: boolean) => {
    for (const entry of fs.list(dir)) {
      if (SKIPPED_DIRS.has(entry) || entry.startsWith('.')) continue;
      const path = join(dir, entry);
      if (fs.isDirectory(path)) {
        const isPackage = fs.isFile(join(path, 'natlang.json'));
        if (!(isPackage && inPackage)) walk(path, inPackage || isPackage);
      } else if (wanted(entry)) out.push(path);
    }
  };
  walk(root, fs.isFile(join(root, 'natlang.json')));
  return out;
}

function detectRuntimeSpecifier(fs: ProjectFiles, files: string[]): string | undefined {
  for (const path of files) {
    const match = /import\s*\{[^}]*\b(?:nl|iterateOn|createNatlangRuntime)\b[^}]*\}\s*from\s*['"]([^'"]+)['"]/.exec(fs.read(path));
    if (match) return match[1];
  }
  return;
}

/** Format diagnostics for a terminal. */
export function formatDiagnostics(diagnostics: readonly NatlangDiagnostic[]): string {
  return diagnostics.map(item => `${item.file}:${item.line}:${item.column} ${item.severity} ${item.code}: ${item.message}` +
    (item.detail ? `\n${item.detail.split('\n').map(line => `  ${line}`).join('\n')}` : '') + (item.hint ? `\n  hint: ${item.hint}` : '')).join('\n');
}
export { spanOf };
