/**
 * `natlang check` and `natlang build`: whole-project analysis and lowering.
 *
 * - Every `.nl` function gets a generated `foo.d.nl.ts` so TypeScript (and plain editors) see typed imports.
 * - Callable folders (`foo/` beside `foo.nl`, and `natlang.d/`) are checked with the constrained-source
 *   policy and embedded as records; they execute at runtime as module instances.
 * - Ordinary project files are type-checked, `nl` expressions are planned against the checker, and the
 *   emitted JavaScript calls the runtime through `__natlang`.
 */
import ts from 'typescript';
import { RewriteGate, rewriteCombinators, type RewriteRecord } from './rewrites.js';
import { describeProgram, namedDescriptor, inlineDescriptor } from '../adaptation/inventory.js';
import type { ComponentDescriptor, ProgramDescriptor } from '../adaptation/types.js';
import { declarationNamespace } from '../native/external.js';
import { analyzeInlineLambdas, spanOf, type InlineLambdaPlan, type NatlangDiagnostic } from './inline.js';
import { checkConstrainedSource } from './policy.js';
import { natlangTransformer } from './lower.js';
import { typeScriptText } from './eval-check.js';
import { INTRINSICS_FILE, NATLANG_COMPILE_VERSION, SURFACE_MODULE_FILE } from './intrinsics.js';
import { createNatlangCompilerHost } from './host.js';
import { loadNzSync, nzDeclaration, toBase64 } from '../native/nz-file.js';
import { compileModule } from '../runtime/modules.js';
import { findApplicationContext, loadCallableFolder, loadNamedFunction, NatlangSourceError,
  type ItemRecord, type NatlangRecord } from '../runtime/loader.js';
import type { SourceFiles } from '../runtime/loader.js';

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
const join = (...parts: string[]) => normalize(parts.filter(Boolean).join('/'));
const normalize = (path: string) => {
  const absolute = path.startsWith('/'), out: string[] = [];
  for (const part of path.split('/')) { if (!part || part === '.') continue; if (part === '..') out.pop(); else out.push(part); }
  return (absolute ? '/' : '') + out.join('/');
};
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
    read: path => files.read(path), list: path => files.list(path), relative: path => relative(root, path) };
}

export type BuildOptions = {
  /** Enforce the finite generated-program profile on every editable application module. */
  constrained?: boolean;
  services?: ProgramDescriptor['services'];
  programId?: string;
  guidance?: string;
  importedGuidancePrograms?: readonly string[];
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
  /** Write generated `.d.nl.ts` files beside their sources (default true; read-only installed packages use false). */
  writeDeclarations?: boolean;
  /**
   * Bind the project to a specific runtime module: these specifiers type-check against `types` and are
   * emitted as `url`, so the compiled app shares one runtime instance with its launcher.
   */
  runtimeModule?: { specifiers: string[]; url: string; types: string; path?: string };
  /** Type these runtime specifiers against the built-in surface declarations (virtual projects). */
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
  declarations: Record<string, string>; manifest: DefinitionManifest; outDir: string;
  /** Rewrites applied per emitted file (empty until a rule is enabled). */
  rewrites?: Record<string, RewriteRecord[]> };

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
  const walk = (dir: string) => {
    for (const entry of files.list(dir).sort()) {
      if (SKIPPED_DIRS.has(entry) || entry.startsWith('.')) continue;
      const path = join(dir, entry);
      if (files.isDirectory(path)) {
        if (folders.has(path)) continue;
        if (entry === 'natlang.d') { contexts.push(path); mark(path); continue; }
        if (files.isFile(`${path}.nl`)) { mark(path); continue; }
        walk(path);
      } else if (entry.endsWith('.nl')) named.push(path);
    }
  };
  walk(root);
  return { named, contexts, folders };
}

const aliasBlock = (types: Record<string, string>) => {
  const known = new Set(Object.keys(types));
  return Object.entries(types).map(([name, text]) => `type ${name} = ${typeScriptText(text, known)};`).join('\n');
};

/** TypeScript declaration text for an item and its callable-folder children. */
function itemType(record: ItemRecord, known: ReadonlySet<string>): string {
  const children = Object.entries(record.codebase).map(([name, child]) => `readonly ${name}: ${itemType(child, known)}`);
  const members = children.length ? ` & { ${children.join('; ')} }` : '';
  if (record.kind === 'namespace') return `{ ${children.join('; ')} }`;
  if (record.kind === 'module') {
    const exports = Object.entries(record.exports).filter(([name]) => name !== 'default').map(([name, spec]) =>
      spec.kind === 'function' ? `readonly ${name}: (${Object.entries(spec.args).map(([raw, type]) =>
        `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`).join(', ')}) => ` +
        `${spec.async ? `Promise<${typeScriptText(spec.returns, known)}>` : typeScriptText(spec.returns, known)}` :
        `readonly ${name}: ${spec.type ? typeScriptText(spec.type, known) : 'unknown'}`);
    const main = record.exports.default;
    const body = `{ ${[...exports, ...children].join('; ')} }`;
    if (main?.kind !== 'function') return body;
    const params = Object.entries(main.args).map(([raw, type]) => `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`);
    return `((${params.join(', ')}) => ${main.async ? `Promise<${typeScriptText(main.returns, known)}>` : typeScriptText(main.returns, known)}) & ${body}`;
  }
  const params = Object.entries(record.args).map(([raw, type]) => `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`);
  if (record.subtype === 'directory-reducer') params.unshift('folder: Folder');
  return `NatlangFunction<[${params.join(', ')}], ${typeScriptText(record.returns, known)}>${members}`;
}

/** Generated declaration for `foo.nl`, read by TypeScript as `foo.d.nl.ts`. */
export function natlangDeclaration(record: NatlangRecord, runtimeSpecifier: string): string {
  const collect: Record<string, string> = {};
  const gather = (item: ItemRecord) => {
    if (item.kind !== 'namespace') Object.assign(collect, item.types);
    for (const child of Object.values(item.codebase)) gather(child);
  };
  gather(record);
  const known = new Set(Object.keys(collect));
  return `// Generated by natlang from ${record.name}.nl; do not edit.\n` +
    `import type { NatlangFunction, Folder as FolderStore, FolderHandle } from ${JSON.stringify(runtimeSpecifier)};\n` +
    '/** A natlang `Folder` accepts a whole folder or a directory handle within one. */\ntype Folder = FolderStore | FolderHandle;\n' +
    `${aliasBlock(collect)}\ndeclare const fn: ${itemType(record, known)};\nexport default fn;\n`;
}

/** Declaration of `natlang:services` falls to the application; this supplies an empty default. */
const SERVICES_FALLBACK = '/__natlang__/services.d.ts';
/** No measurements: every rewrite rule is off. */
const DEFAULT_REWRITE_GATE = new RewriteGate({ model: 'unmeasured', dialect: 'unmeasured' });
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
  export function gloss(v: N<unknown>): Promise<string>;
}
declare module 'natlang:learning' {
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
  /** A weight adapter value: coefficients of a tiny adapter of the serving model (\`Adapter\` in natlang types). */
  export type Adapter = Neuralese<{ $adapter: 'adapter' }>;
  export function withAdapters<T>(adapters: Adapter | { adapter: Adapter; scale?: number } | Array<Adapter | { adapter: Adapter; scale?: number }>, fn: () => Promise<T> | T): Promise<T>;
  /** Run fn with soft forms of the runtime prompt pieces (piece ID → Neuralese<SystemPrompt>); differentiable like any soft value. */
  export function withSystemPrompts<T>(prompts: Record<string, Neuralese<string>>, fn: () => Promise<T> | T): Promise<T>;
  export const adapters: {
    create(options?: { kind?: 'xs' | 'tiny'; rank?: number; u?: number; layers?: number[]; targets?: Array<'out' | 'ffn_down' | 'ffn_up'>; seed?: number }): Promise<Adapter>;
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
    klPrior(blocks: Neuralese<unknown> | Neuralese<unknown>[]): Promise<Loss>;
    sum(...losses: Loss[]): Promise<Loss>;
    scale(loss: Loss, weight: number): Promise<Loss>;
  };
  export const optimizers: {
    sgd(options: { lr: number; momentum?: number; weightDecay?: number }): Optimizer;
    adam(options: { lr: number; betas?: [number, number]; weightDecay?: number }): Optimizer;
  };
  export function save(path: string, exports: Record<string, unknown>): Promise<{ path: string; exports: string[] }>;
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
  const fs = options.files;
  const { configPath, root } = findTsconfig(fs, options.project);
  const diagnostics: NatlangDiagnostic[] = [];
  const files = sourceFiles(fs, root);
  const rel = (path: string) => relative(root, path);
  const problem = (file: string, message: string, code: NatlangDiagnostic['code'] = 'callable-scope') =>
    diagnostics.push({ file, start: 0, end: 0, line: 1, column: 1, code, message, severity: 'error' });

  // Callable folders and named functions.
  const layout = callableFolders(fs, root);
  const namedRecords = new Map<string, NatlangRecord>();
  for (const path of layout.named) {
    if ([...layout.folders].some(folder => path.startsWith(folder + sep))) continue;
    try { namedRecords.set(path, loadNamedFunction(path, files)); }
    catch (error) { problem(rel(path), error instanceof NatlangSourceError ? error.message : String(error)); }
  }
  const contextRecords = new Map<string, Record<string, ItemRecord>>();
  for (const dir of layout.contexts) {
    try { contextRecords.set(dir, loadCallableFolder(dir, files)); }
    catch (error) { problem(rel(dir), error instanceof NatlangSourceError ? error.message : String(error)); }
  }
  const optimizationComponents: ComponentDescriptor[] = [];
  const optimizationSources: Record<string, string> = {};
  const programId = options.programId ?? basename(root);
  // Compile every callable-folder module now so errors surface at build time rather than first call.
  const checkModules = (level: Record<string, ItemRecord>) => {
    for (const item of Object.values(level)) {
      if (item.kind !== 'namespace') item.programId = programId;
      if (item.kind !== 'namespace') optimizationSources[item.source] = item.text;
      if (item.kind === 'natlang') optimizationComponents.push(namedDescriptor(programId, item));
      if (item.kind === 'module') {
        try { compileModule(item, level, plans => optimizationComponents.push(...plans.map(plan => inlineDescriptor(programId, plan)))); }
        catch (error) { problem(item.source, error instanceof Error ? error.message : String(error)); }
      }
      checkModules(item.codebase);
    }
  };
  for (const record of namedRecords.values()) {
    record.programId = programId;
    optimizationSources[record.source] = record.text; optimizationComponents.push(namedDescriptor(programId, record)); checkModules(record.codebase);
  }
  for (const records of contextRecords.values()) checkModules(records);

  // TypeScript program over ordinary project files, with generated `.nl` declarations.
  const configHost: ts.ParseConfigFileHost = { useCaseSensitiveFileNames: true, getCurrentDirectory: () => root,
    fileExists: path => fs.isFile(path), readFile: path => fs.isFile(path) ? fs.read(path) : undefined,
    readDirectory: (dir, extensions, excludes, includes, depth) => fs.compiler ?
      ts.sys.readDirectory(dir, extensions, excludes, includes, depth) : walkTs(fs, dir),
    onUnRecoverableConfigFileDiagnostic: diagnostic => problem(rel(configPath!), ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'), 'typescript') };
  const parsed = configPath ? ts.getParsedCommandLineOfConfigFile(configPath, {}, configHost) : undefined;
  const compilerOptions: ts.CompilerOptions = { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.NodeNext,
    moduleResolution: ts.ModuleResolutionKind.NodeNext, strict: true, skipLibCheck: true,
    ...(parsed?.options ?? {}), allowArbitraryExtensions: true, noEmit: false, declaration: false,
    ...(options.module === 'commonjs' ? { module: ts.ModuleKind.CommonJS, moduleResolution: ts.ModuleResolutionKind.Node10 } : {}) };
  const outDir = resolve(options.outDir ?? compilerOptions.outDir ?? join(root, 'dist'));
  compilerOptions.outDir = outDir;
  compilerOptions.rootDir ??= root;
  const inFolder = (path: string) => [...layout.folders].some(folder => path.startsWith(folder + sep) || path === folder);
  const rootNames = (parsed?.fileNames?.length ? parsed.fileNames : walkTs(fs, root)).filter(path => !inFolder(path) &&
    !path.endsWith('.d.nl.ts') && !path.endsWith('.d.nz.ts'));
  const bound = options.runtimeModule;
  if (options.runtimeTypeRoots?.length) {
    const resolutionHost = ts.sys ?? { fileExists: (path: string) => fs.isFile(path), readFile: (path: string) => fs.isFile(path) ? fs.read(path) : undefined };
    const missing = (compilerOptions.types ?? ['node']).some(name =>
      !ts.resolveTypeReferenceDirective(name, join(root, 'index.ts'), compilerOptions, resolutionHost).resolvedTypeReferenceDirective);
    if (missing) compilerOptions.typeRoots = [...(compilerOptions.typeRoots ?? []), ...options.runtimeTypeRoots];
  }
  if (options.runtimeTypes) for (const specifier of options.runtimeTypes.specifiers) {
    const resolved = ts.resolveModuleName(specifier, join(root, 'index.ts'), compilerOptions, ts.sys ?? {
      fileExists: path => fs.isFile(path), readFile: path => fs.isFile(path) ? fs.read(path) : undefined });
    if (!resolved.resolvedModule) compilerOptions.paths = { ...(compilerOptions.paths ?? {}), [specifier]: [options.runtimeTypes.types] };
  }
  if (options.surfaceSpecifiers?.length) compilerOptions.paths = { ...(compilerOptions.paths ?? {}),
    ...Object.fromEntries(options.surfaceSpecifiers.map(specifier => [specifier, [SURFACE_MODULE_FILE]])) };
  if (bound) compilerOptions.paths = { ...(compilerOptions.paths ?? {}),
    ...Object.fromEntries(bound.specifiers.map(specifier => [specifier, [bound.types]])) };
  const runtimeSpecifier = bound?.url ?? options.runtimeSpecifier ?? detectRuntimeSpecifier(fs, rootNames) ?? '@natlang/node';
  const declarationSpecifier = bound?.specifiers[0] ?? runtimeSpecifier;
  const rewrite = new Map((bound?.specifiers ?? []).map(specifier => [specifier, bound!.url]));
  const declarations: Record<string, string> = {};
  const virtual = new Map<string, string>([[SERVICES_FALLBACK, "declare module 'natlang:services' { const services: Record<string, any>; export = services; }\n"]]);
  for (const [path, record] of namedRecords) {
    const declaration = natlangDeclaration(record, declarationSpecifier);
    const target = path.replace(/\.nl$/, '.d.nl.ts');
    declarations[target] = declaration;
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
      declarations[target] = declaration;
      virtual.set(target, declaration);
    } catch (error) { problem(rel(path), error instanceof Error ? error.message : String(error), 'neuralese-file'); }
  }
  const usesNeuraleseModules = rootNames.some(path => /['"]natlang:(neuralese|learning)['"]/.test(fs.read(path)));
  if (usesNeuraleseModules) virtual.set(NEURALESE_MODULES, NEURALESE_MODULES_SOURCE);
  const host = createNatlangCompilerHost({ options: compilerOptions, virtual, currentDirectory: root,
    files: fs.compiler ?? { readFile: path => fs.isFile(path) ? fs.read(path) : undefined, fileExists: path => fs.isFile(path),
      directoryExists: path => fs.isDirectory(path), getDirectories: path => fs.isDirectory(path) ?
        fs.list(path).filter(entry => fs.isDirectory(join(path, entry))) : [] } });
  const hasServicesDeclaration = rootNames.some(path => /declare\s+module\s+['"]natlang:services['"]/.test(fs.read(path)));
  const program = ts.createProgram({ rootNames: [INTRINSICS_FILE, ...(hasServicesDeclaration ? [] : [SERVICES_FALLBACK]),
    ...(usesNeuraleseModules ? [NEURALESE_MODULES] : []), ...rootNames],
    options: compilerOptions, host });
  for (const diagnostic of ts.getPreEmitDiagnostics(program)) {
    if (diagnostic.category !== ts.DiagnosticCategory.Error) continue;
    const file = diagnostic.file;
    const position = file && diagnostic.start !== undefined ? file.getLineAndCharacterOfPosition(diagnostic.start) : { line: 0, character: 0 };
    diagnostics.push({ file: file ? rel(file.fileName) : '', start: diagnostic.start ?? 0, end: (diagnostic.start ?? 0) + (diagnostic.length ?? 0),
      line: position.line + 1, column: position.character + 1, code: 'typescript', severity: 'error',
      message: ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n') });
  }
  const sources = rootNames.map(path => program.getSourceFile(path)).filter((file): file is ts.SourceFile => !!file);
  if (options.constrained) for (const file of sources) diagnostics.push(...checkConstrainedSource(file, { checker: program.getTypeChecker(), displayPath: source => rel(source.fileName) }));
  const revision = (file: ts.SourceFile) => `${rel(file.fileName)}@${file.text.length}`;
  const plans = new Map<ts.SourceFile, InlineLambdaPlan[]>();
  const readouts = new Map<ts.SourceFile, import('./neuralese.js').NeuraleseReadout[]>();
  for (const file of sources) {
    const analysis = analyzeInlineLambdas(program, [file], { displayPath: source => rel(source.fileName), sourceRevision: revision(file), authored: true });
    diagnostics.push(...analysis.diagnostics);
    analysis.plans.forEach(plan => { plan.programId = programId; });
    plans.set(file, analysis.plans);
    readouts.set(file, analysis.readouts);
    optimizationSources[rel(file.fileName)] = file.text;
    optimizationComponents.push(...analysis.plans.map(plan => inlineDescriptor(programId, plan)));
  }
  // Include dependency resolutions/configuration and conservatively all authored project files.
  const collectSources = (dir: string): void => {
    for (const name of fs.list(dir).sort()) {
      if (SKIPPED_DIRS.has(name) || name.startsWith('.') || name === 'adaptations' || name.endsWith('.d.nl.ts')) continue;
      const path = join(dir, name);
      if (fs.isDirectory(path)) {
        const runManifest = join(path, 'manifest.json');
        if (fs.isFile(runManifest)) {
          try { if (JSON.parse(fs.read(runManifest))?.schema === 'natlang.adaptation-run/v1') continue; }
          catch { /* An ordinary manifest still contributes to the source fingerprint. */ }
        }
        collectSources(path);
      }
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
  collectSources(root);
  let dependencyRoot = root;
  while (true) {
    for (const lock of ['package-lock.json', 'pnpm-lock.yaml', 'yarn.lock']) {
      const path = join(dependencyRoot, lock);
      if (fs.isFile(path)) optimizationSources['@dependency-lock/' + lock] = fs.read(path);
    }
    const parent = dirname(dependencyRoot); if (parent === dependencyRoot) break; dependencyRoot = parent;
  }
  const manifest: DefinitionManifest = { version: NATLANG_COMPILE_VERSION,
    inline: [...plans.values()].flat().map(plan => ({ id: plan.definitionId, source: plan.sourceSpan.file, line: plan.sourceSpan.line,
      signature: `(${plan.parameters.map(parameter => `${parameter.name}: ${parameter.type.text}`).join(', ')}) => ${plan.returns.text}` })),
    named: [...namedRecords.values()].map(record => ({ id: record.id, source: record.source,
      signature: `(${Object.entries(record.args).map(([name, type]) => `${name}: ${type}`).join(', ')}) => ${record.returns}` })),
    sites: [], adaptation: describeProgram(programId, optimizationSources, optimizationComponents, options.guidance, options.importedGuidancePrograms,
      options.services ? { declarations: Object.fromEntries(Object.entries(options.services.declarations).map(([name, text]) =>
        [name, /^\s*declare (?:namespace|const) /.test(text) ? text.trim() : declarationNamespace(name, text)])), scopes: options.services.scopes } : undefined) };
  const ok = !diagnostics.some(item => item.severity === 'error');
  const outputs: Record<string, string> = {};
  const rewrites: Record<string, RewriteRecord[]> = {};
  if (options.write !== false && options.writeDeclarations !== false && fs.write) for (const [path, text] of Object.entries(declarations))
    if (!fs.isFile(path) || fs.read(path) !== text) fs.write(path, text);
  if (ok && options.emit !== false) {
    const generated = new Map<string, { record: NatlangRecord; commonjs: boolean }>();
    const generatedNz = new Map<string, { bytes: Uint8Array; commonjs: boolean }>();
    const natlangImports = (file: ts.SourceFile) => {
      const found = new Set<string>();
      for (const statement of file.statements) {
        if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier)) continue;
        const specifier = statement.moduleSpecifier.text;
        if (specifier.endsWith('.nz')) {
          const target = resolve(dirname(file.fileName), specifier);
          const bytes = nzFiles.get(target);
          if (!bytes) { problem(rel(file.fileName), `cannot import ${specifier}: no such .nz file`); continue; }
          found.add(specifier);
          generatedNz.set(join(outDir, relative(compilerOptions.rootDir!, target)) + '.js',
            { bytes, commonjs: file.impliedNodeFormat === ts.ModuleKind.CommonJS || compilerOptions.module === ts.ModuleKind.CommonJS });
          continue;
        }
        if (!specifier.endsWith('.nl')) continue;
        const target = resolve(dirname(file.fileName), specifier);
        const record = namedRecords.get(target);
        if (!record) { problem(rel(file.fileName), `cannot import ${specifier}: no such natlang function`); continue; }
        found.add(specifier);
        generated.set(join(outDir, relative(compilerOptions.rootDir!, target)) + '.js',
          { record, commonjs: file.impliedNodeFormat === ts.ModuleKind.CommonJS || compilerOptions.module === ts.ModuleKind.CommonJS });
      }
      return found;
    };
    const specifierFor = (commonjs: boolean) => bound ? (commonjs && bound.path ? bound.path : bound.url) : runtimeSpecifier;
    const byFile = new Map(sources.map(file => [file.fileName, file]));
    const transformer: ts.TransformerFactory<ts.SourceFile> = context => file => {
      const source = byFile.get(file.fileName);
      if (!source) return file;
      const contextDir = findApplicationContext(dirname(file.fileName), files);
      const filePlans = plans.get(source) ?? [];
      const fileReadouts = readouts.get(source) ?? [];
      const commonjs = source.impliedNodeFormat === ts.ModuleKind.CommonJS || compilerOptions.module === ts.ModuleKind.CommonJS;
      const rewriteFor = new Map([...rewrite.keys()].map(specifier => [specifier, specifierFor(commonjs)]));
      return natlangTransformer({ plans: new Map(filePlans.map(plan => [`${plan.sourceSpan.start}:${plan.sourceSpan.end}`, plan])),
        readouts: new Set(fileReadouts.filter(item => !item.kind).map(item => `${item.start}:${item.end}`)),
        joins: new Set(fileReadouts.filter(item => item.kind === 'join').map(item => `${item.start}:${item.end}`)),
        checker: program.getTypeChecker(), runtime: '__natlang',
        context: contextDir && contextRecords.has(contextDir) ? JSON.stringify(contextRecords.get(contextDir)) : undefined,
        constrained: options.constrained ?? false, guardPrefix: JSON.stringify([programId, rel(file.fileName)]), modulePath: rel(file.fileName), browser: options.target === 'browser',
        module: { natlangImports: natlangImports(source), rewrite: rewriteFor } })(context)(file);
    };
    const emit = (fileName: string, text: string) => {
      if (options.constrained && /\.js$/.test(fileName) && !/(?:\b(?:const|let|var)\s+__natlang\b|\bimport\s*\{[^}]*\b__natlang\b)/.test(text)) {
        const commonjs = compilerOptions.module === ts.ModuleKind.CommonJS || /Object\.defineProperty\(exports/.test(text);
        text = (commonjs ? `const __natlang = require(${JSON.stringify(specifierFor(true))}).__natlang;\n` :
          `import { __natlang } from ${JSON.stringify(specifierFor(false))};\n`) + text;
      }
      if (/\.[cm]?js$/.test(fileName) && /['"]natlang:neuralese['"]/.test(text)) {
        const rewritten = rewriteCombinators(text, { gate: options.rewriteGate ?? DEFAULT_REWRITE_GATE, fileName });
        text = rewritten.code;
        if (rewritten.applied.length) rewrites[fileName] = rewritten.applied;
      }
      outputs[fileName] = text;
      if (options.write !== false) fs.write?.(fileName, text);
    };
    const emitted = program.emit(undefined, emit, undefined, false, { before: [transformer] });
    for (const [path, { record, commonjs }] of generated) {
      const specifier = JSON.stringify(specifierFor(commonjs)), json = JSON.stringify(record);
      emit(path, commonjs ?
        `"use strict";\nObject.defineProperty(exports, "__esModule", { value: true });\nexports.default = require(${specifier}).__natlang.named(${JSON.stringify(record.name)}, ${json});\n` :
        `import { __natlang } from ${specifier};\nexport default __natlang.named(${JSON.stringify(record.name)}, ${json});\n`);
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
      problem(diagnostic.file ? rel(diagnostic.file.fileName) : '', ts.flattenDiagnosticMessageText(diagnostic.messageText, '\n'), 'typescript');
    if (options.write !== false) fs.write?.(join(outDir, 'natlang-manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  }
  diagnostics.sort((a, b) => a.file.localeCompare(b.file) || a.start - b.start);
  return { ok: !diagnostics.some(item => item.severity === 'error'), diagnostics, outputs, declarations, manifest, outDir, rewrites };
}

function walkNz(fs: ProjectFiles, root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const entry of fs.list(dir)) {
      if (SKIPPED_DIRS.has(entry) || entry.startsWith('.')) continue;
      const path = join(dir, entry);
      if (fs.isDirectory(path)) walk(path);
      else if (entry.endsWith('.nz')) out.push(path);
    }
  };
  walk(root);
  return out;
}

function walkTs(fs: ProjectFiles, root: string): string[] {
  const out: string[] = [];
  const walk = (dir: string) => {
    for (const entry of fs.list(dir)) {
      if (SKIPPED_DIRS.has(entry) || entry.startsWith('.')) continue;
      const path = join(dir, entry);
      if (fs.isDirectory(path)) walk(path);
      else if (/\.(ts|tsx|mts)$/.test(entry) && !entry.endsWith('.d.ts')) out.push(path);
    }
  };
  walk(root);
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
  return diagnostics.map(item => `${item.file}:${item.line}:${item.column} ${item.severity} ${item.code}: ${item.message}`).join('\n');
}
export { spanOf };
