/**
 * Module instances for callable-folder TypeScript. A module is compiled with the shared lowering
 * (entry guards, finite iteration, `nl`, `iterateOn` sites) and evaluated once per source revision.
 */
import ts from 'typescript';
import { currentFrame } from './context.js';
import type { EvalEnvironment } from '../native/evaluator.js';
import { createVirtualProgram, EVAL_COMPILER_OPTIONS } from '../compiler/host.js';
import { analyzeInlineLambdas, type InlineLambdaPlan } from '../compiler/inline.js';
import { natlangTransformer } from '../compiler/lower.js';
import { NEUTRAL_RUNTIME_SPECIFIER } from '../compiler/runtime-specifier.js';
import { checkConstrainedSource } from '../compiler/policy.js';
import { typeScriptText } from '../compiler/eval-check.js';
import { NatlangSourceError, type ItemRecord, type ModuleRecord } from './loader.js';
import * as lowered from './lowered.js';
import * as surface from './surface.js';
import { inlineDescriptor } from '../adaptation/inventory.js';
import { fingerprint } from '../adaptation/identity.js';
import { namedCallable, callableTree } from './callable.js';

let defaultRealm: (() => EvalEnvironment) | undefined;
let moduleTarget: 'node' | 'browser' = 'node';
let packageLoader: ((specifier: string) => unknown) | undefined;
/** Installed by the platform: how callable-folder modules load packages. */
const builtinModules = new Map<string, () => unknown>();
/**
 * Platform modules such as `natlang:neuralese` and `natlang:learning`, registered by the platform wiring
 * (runtime/node.ts, browser/index.ts); a platform that cannot provide one registers a factory that throws why.
 */
export function registerBuiltinModule(specifier: string, factory: () => unknown): void { builtinModules.set(specifier, factory); }
/** Importing a platform module this platform cannot provide (code `module-unavailable`); `reason` teaches what to do. */
export class ModuleUnavailableError extends Error {
  readonly code = 'module-unavailable';
  constructor(readonly specifier: string, reason: string) {
    super(`${specifier} is not available on this platform: ${reason}`);
    this.name = 'ModuleUnavailableError';
  }
}
export function hasBuiltinModule(specifier: string): boolean { return builtinModules.has(specifier); }
export function builtinModule(specifier: string): unknown {
  const factory = builtinModules.get(specifier);
  if (!factory) throw new ModuleUnavailableError(specifier, 'this platform registers no such module');
  return factory();
}

export function setPackageLoader(loader: (specifier: string) => unknown): void { packageLoader = loader; }
/** Browsers restore the task context after each await in callable-folder code. */
export function setModuleTarget(target: 'node' | 'browser'): void { moduleTarget = target; }
let realm: EvalEnvironment | undefined;
let realmGeneration = 0;
export function moduleRealmGeneration(): number { return realmGeneration; }
export class ModuleRealmChangedError extends Error {
  readonly code = 'module-realm-changed';
  constructor() {
    super('The module execution authority changed or closed; restart the task and reload its callable modules.');
    this.name = 'ModuleRealmChangedError';
  }
}
/** Installed by the platform entry point: the evaluator in which module instances live. */
export function setModuleRealm(factory: () => EvalEnvironment): void {
  if(typeof factory!=='function')throw new TypeError('Module evaluator factory required');
  // A cutover must never hand back CommonJS exports created by the previous
  // realm (which may have been native and untraced). Invalidate the shared
  // module-instance cache and close the old realm before installing the new
  // authority. Active task-local caches cannot be migrated in place.
  const previous=realm;
  realmGeneration++;
  defaultRealm=factory;
  realm=undefined;
  instances=new WeakMap();
  const failures: unknown[] = [];
  for (const scope of [...asyncScopes]) {
    try { scope.close(); } catch (error) { failures.push(error); }
  }
  asyncScopes.clear();
  sharedAsyncScope=undefined;
  asyncTaskScopes=new WeakMap();
  try { previous?.close(); } catch (error) { failures.push(error); }
  if (failures.length) throw new AggregateError(failures, 'Module authority changed but resource cleanup failed');
}
function moduleRealm(): EvalEnvironment {
  if (!defaultRealm) throw new Error('no module realm is installed for this platform');
  return realm ??= defaultRealm();
}

// A loop over a value whose type decides whether it is a bounded collection needs the checker.
const FOR_OF = /\bfor\s*(?:await\s*)?\([^;)]*\bof\b/;
// The exact intrinsic resolver decides what a tag means. A text prefilter missed nl.with tags and aliases.
function containsTaggedTemplate(node: ts.Node): boolean {
  return ts.isTaggedTemplateExpression(node) || !!ts.forEachChild(node, containsTaggedTemplate);
}
const FOLDER = '/__natlang__/folder';

function natlangDeclaration(record: ItemRecord): string {
  if (record.kind !== 'natlang') return '';
  const known = new Set(Object.keys(record.types));
  const params = Object.entries(record.args).map(([raw, type]) =>
    `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`);
  if (record.subtype === 'directory-reducer') params.unshift('folder: Folder');
  const type = record.generic ? `NatlangGenericFunction<[${params.join(', ')}], ${typeScriptText(record.generic.constraint, known)}, ` +
    `${typeScriptText(record.returns, known)}>` : `NatlangFunction<[${params.join(', ')}], ${typeScriptText(record.returns, known)}>`;
  return `declare const fn: ${type};\nexport default fn;\n`;
}

/** Compile one module to CommonJS-style JavaScript for evaluation. */
export function compileModule(record: ModuleRecord, level: Record<string, ItemRecord>, inventory?: (plans: InlineLambdaPlan[]) => void): string {
  let plans = new Map<string, InlineLambdaPlan>();
  let readouts = new Set<string>();
  let conditionalReadouts = new Set<string>();
  let joins = new Set<string>();
  let arrayStrings = new Set<string>();
  let arrayMaps = new Set<string>();
  let arrayMapCallbacks = new Set<string>();
  let concats = new Set<string>();
  let jsons = new Set<string>();
  let errorReadouts = new Set<string>();
  let stringArguments = new Map<string, number>();
  let scalarConversions = new Map<string, { argument: number; conversion: 'Number' | 'Boolean'; conditional?: true }>();
  let stringReplaces = new Map<string, { conditional?: true }>();
  let instantiations = new Map<string, { representation: import('../native/representation.js').Representation; site: string }>();
  let checker: ts.TypeChecker | undefined;
  const path = `${FOLDER}/${record.name}.ts`;
  const softTypes = JSON.stringify(record.types);
  // A sibling with a representation-generic result: its call sites need the checker to pick their instance.
  const generic = Object.values(level).some(item => item.kind === 'natlang' && !!item.generic);
  const parsed = ts.createSourceFile(path, record.text, ts.ScriptTarget.ES2022, true);
  if (containsTaggedTemplate(parsed) || FOR_OF.test(record.text) || /\bNeuralese\s*</.test(`${record.text}\n${softTypes}`) || generic) {
    const files: Record<string, string> = { [path]: record.text };
    for (const [name, item] of Object.entries(level)) {
      if (name === record.name) continue;
      if (item.kind === 'module') files[`${FOLDER}/${name}.ts`] = item.text;
      if (item.kind === 'natlang') files[`${FOLDER}/${name}.d.nl.ts`] = natlangDeclaration(item);
    }
    const known = new Set(Object.keys(record.types));
    files[`${FOLDER}/__types.d.ts`] = Object.entries(record.types)
      .map(([name, text]) => `type ${name} = ${typeScriptText(text, known)};`).join('\n') + '\n';
    // The same aliases are importable as the folder's types module (`import type { Ticket } from "./types"`),
    // which is what TypeScript authors write; without it the import resolves to nothing and types become any.
    files[`${FOLDER}/types.ts`] ??= Object.entries(record.types)
      .map(([name, text]) => `export type ${name} = ${typeScriptText(text, known)};`).join('\n') + '\nexport {};\n';
    const external = new Map<string, Set<string>>();
    for (const statement of parsed.statements) if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier) &&
        !statement.moduleSpecifier.text.startsWith('.')) {
      const names = external.get(statement.moduleSpecifier.text) ?? new Set<string>();
      const bindings = statement.importClause?.namedBindings;
      if (bindings && ts.isNamedImports(bindings)) for (const element of bindings.elements) names.add((element.propertyName ?? element.name).text);
      external.set(statement.moduleSpecifier.text, names);
    }
    files[`${FOLDER}/__external.d.ts`] = [...external].filter(([specifier]) => !/^@natlang\//.test(specifier) && specifier !== NEUTRAL_RUNTIME_SPECIFIER)
      .map(([specifier, names]) => `declare module ${JSON.stringify(specifier)} {\n${[...names].map(name =>
        `  export const ${name}: any;`).join('\n')}\n  const value: any;\n  export default value;\n}`).join('\n');
    const program = createVirtualProgram(files, { ...EVAL_COMPILER_OPTIONS, paths: { '@natlang/*': ['/__natlang__/surface.d.ts'],
      [NEUTRAL_RUNTIME_SPECIFIER]: ['/__natlang__/surface.d.ts'] } });
    const file = program.getSourceFile(path)!;
    const analysis = analyzeInlineLambdas(program, [file], { displayPath: () => record.source,
      sourceRevision: record.revision, authored: true });
    // Iteration sources are classified by type, so a loop the run would refuse is refused when the project is checked.
    const iteration = checkConstrainedSource(file, { checker: program.getTypeChecker(), displayPath: () => record.source });
    if (iteration.length) throw new NatlangSourceError(record.source, iteration.map(item => `${item.line}:${item.column} ${item.message}`).join('\n'));
    const errors = analysis.diagnostics.filter(item => item.severity === 'error');
    if (errors.length) throw new NatlangSourceError(record.source, errors.map(item => `${item.line}:${item.column} ${item.message}`).join('\n'));
    readouts = new Set(analysis.readouts.filter(item => !item.kind).map(item => `${item.start}:${item.end}`));
    conditionalReadouts = new Set(analysis.readouts.filter(item => item.conditional).map(item => `${item.start}:${item.end}`));
    joins = new Set(analysis.readouts.filter(item => item.kind === 'join').map(item => `${item.start}:${item.end}`));
    arrayStrings = new Set(analysis.readouts.filter(item => item.kind === 'array-string').map(item => `${item.start}:${item.end}`));
    arrayMaps = new Set(analysis.readouts.filter(item => item.kind === 'array-map').map(item => `${item.start}:${item.end}`));
    arrayMapCallbacks = new Set(analysis.readouts.filter(item => item.kind === 'array-map' && item.callback)
      .map(item => `${item.callback!.start}:${item.callback!.end}`));
    concats = new Set(analysis.readouts.filter(item => item.kind === 'concat').map(item => `${item.start}:${item.end}`));
    jsons = new Set(analysis.readouts.filter(item => item.kind === 'json').map(item => `${item.start}:${item.end}`));
    errorReadouts = new Set(analysis.readouts.filter(item => item.kind === 'error').map(item => `${item.start}:${item.end}`));
    stringArguments = new Map(analysis.readouts.filter(item => item.kind === 'string-argument')
      .map(item => [`${item.start}:${item.end}`, item.argument ?? 0]));
    scalarConversions = new Map(analysis.readouts.filter(item => item.kind === 'scalar-conversion' && item.conversion)
      .map(item => [`${item.start}:${item.end}`, { argument: item.argument ?? 0, conversion: item.conversion!,
        ...(item.conditional ? { conditional: true as const } : {}) }]));
    stringReplaces = new Map(analysis.readouts.filter(item => item.kind === 'string-replace')
      .map(item => [`${item.start}:${item.end}`, { ...(item.conditional ? { conditional: true as const } : {}) }]));
    instantiations = new Map(analysis.instantiations.map(item => [`${item.start}:${item.end}`,
      { representation: item.representation, site: `${record.source}:${item.line}` }]));
    analysis.plans.forEach(plan => { if (record.programId) plan.programId = record.programId; });
    inventory?.(analysis.plans);
    plans = new Map(analysis.plans.map(plan => [`${plan.sourceSpan.start}:${plan.sourceSpan.end}`, plan]));
    checker = program.getTypeChecker();
  }
  const output = ts.transpileModule(record.text, { fileName: `${record.name}.ts`, reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true, isolatedModules: true },
    transformers: { before: [natlangTransformer({ plans, checker, readouts, conditionalReadouts, joins, arrayStrings, arrayMaps, concats, jsons, instantiations,
      errors: errorReadouts, stringArguments, scalarConversions, stringReplaces, runtime: '__natlang', context: '__natlang_context',
      arrayMapCallbacks, constrained: true, guardPrefix: record.programId ? JSON.stringify([record.programId, record.id]) : record.id,
      modulePath: record.source, browser: moduleTarget === 'browser' })] } });
  const errors = (output.diagnostics ?? []).filter(item => item.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new NatlangSourceError(record.source, errors.map(item => ts.flattenDiagnosticMessageText(item.messageText, '\n')).join('; '));
  return output.outputText;
}

let instances = new WeakMap<ModuleRecord, { exports: Record<string, unknown>; ready: boolean }>();
// Compilation is immutable and shareable; mutable exports remain in the task's instance cache.
const compiled = new Map<string, string>();
function compiledCode(record: ModuleRecord, level: Record<string, ItemRecord>): string {
  const key = fingerprint({ target: moduleTarget, source: record.source, id: record.id, revision: record.revision,
    programId: record.programId ?? null, text: record.text, types: record.types,
    siblings: Object.entries(level).map(([name, item]) => ({ name, kind: item.kind,
      text: item.kind === 'namespace' ? null : item.text,
      ...(item.kind === 'natlang' ? { args: item.args, returns: item.returns, types: item.types, subtype: item.subtype,
        ...(item.generic ? { generic: item.generic.constraint } : {}) } : {}) })) });
  const cached = compiled.get(key);
  if (cached !== undefined) return cached;
  const code = compileModule(record, level);
  if (compiled.size >= 256) compiled.delete(compiled.keys().next().value!);
  compiled.set(key, code); return code;
}

const servicesModule = new Proxy(Object.create(null), {
  get: (_, name) => typeof name === 'string' && name !== '__esModule' && name !== 'then' ? lowered.service(name) : undefined,
});
function selectedServicesModule(record: ModuleRecord): object {
  const file = ts.createSourceFile(record.source, record.text, ts.ScriptTarget.ES2022, true);
  const names = new Set<string>();
  for (const statement of file.statements) {
    if (!ts.isImportDeclaration(statement) || !ts.isStringLiteral(statement.moduleSpecifier) ||
        statement.moduleSpecifier.text !== 'natlang:services') continue;
    const bindings = statement.importClause?.namedBindings;
    if (bindings && ts.isNamedImports(bindings)) for (const binding of bindings.elements)
      names.add((binding.propertyName ?? binding.name).text);
    else for (const name of Object.keys(currentFrame()?.task.services ?? {})) names.add(name);
  }
  return Object.fromEntries([...names].map(name => [name, lowered.selectedService(name)]));
}

/** The live exports of a module in its folder `level` (its sibling items). */
export function moduleInstance(record: ModuleRecord, level: Record<string, ItemRecord>): Record<string, unknown> {
  const task = currentFrame()?.task;
  const cache = task && (task.programView.binding || task.runtime.options.isolateModules || task.programView.revisionId) ?
    task.moduleInstances : instances;
  record = task?.programView.original(record) ?? record;
  const patched = task?.programView.patched(record.source);
  if (patched) record = task!.programView.record(record);
  const existing = cache.get(record);
  if (existing) return existing.exports;
  if (task?.programView.binding && !patched) {
    const projected = task.programView.record(record);
    if (projected.text !== record.text) {
      let originalPlans: InlineLambdaPlan[] = [], projectedPlans: InlineLambdaPlan[] = [];
      compileModule(record, level, plans => { originalPlans = plans; });
      compileModule(projected, task.programView.tree(level), plans => { projectedPlans = plans; });
      if (originalPlans.length !== projectedPlans.length || originalPlans.some((plan, index) =>
        inlineDescriptor(task.programView.program!.id, plan).contractHash !==
        inlineDescriptor(task.programView.program!.id, projectedPlans[index]!).contractHash))
        throw new NatlangSourceError(record.source, 'adapted module projection changed frozen inline contracts');
    }
  }
  const code = compiledCode(record, level);
  const exports: Record<string, unknown> = {};
  const state = { exports, ready: false };
  cache.set(record, state);
  const environment = moduleRealm();
  const require = (specifier: string): unknown => {
    if (specifier === 'natlang:services') return servicesModule;
    if (builtinModules.has(specifier)) return builtinModule(specifier);
    // Every runtime specifier, the neutral one included, is the natlang surface in callable-folder code.
    if (specifier === NEUTRAL_RUNTIME_SPECIFIER || /^@natlang\/(node|browser|core)$/.test(specifier)) return { __esModule: true, ...surface };
    if (specifier.startsWith('./')) {
      const parts = specifier.slice(2).replace(/\.(ts|js|nl)$/, '').split('/');
      let scope: Record<string, ItemRecord> = level, parent = level, item: ItemRecord | undefined;
      for (const part of parts) {
        parent = scope;
        item = scope[part];
        if (!item) break;
        scope = item.codebase;
      }
      if (!item) throw new NatlangSourceError(record.source, `cannot import ${JSON.stringify(specifier)}: only items in this callable folder can be imported`);
      if (item.kind === 'natlang') return { __esModule: true, default: namedCallable(item.name, item as never) };
      if (item.kind === 'module') return moduleInstance(item, parent);
      return { __esModule: true, ...callableTree(item.codebase as never) };
    }
    if (specifier.startsWith('.') || specifier.startsWith('/'))
      throw new NatlangSourceError(record.source, `cannot import ${JSON.stringify(specifier)}: callable-folder code may import its sibling items and packages only`);
    if (!packageLoader) throw new NatlangSourceError(record.source, `package imports are unavailable here: ${specifier}`);
    return packageLoader(specifier);
  };
  const module = { exports };
  try {
    environment.evaluateModule(code, { exports, module, require, __natlang: lowered, __natlang_context: level,
      nl: surface.nl, iterateOn: surface.iterateOn });
  } catch (error) {
    cache.delete(record);
    throw error;
  }
  if (module.exports !== exports) Object.assign(exports, module.exports);
  Object.defineProperty(exports, '__esModule', { value: true });
  state.ready = true;
  return exports;
}


/** Async scopes own both their evaluator and cache. Task isolation never reuses
 * another task's live module state. A graph is initialized as one serial turn;
 * recursive imports share partial exports, while unrelated roots cannot deadlock
 * by waiting on each other's pending initialization. */
type AsyncModuleState = {
  raw: Record<string, unknown>;
  exports: Record<string, unknown>;
  ready: boolean;
  failure?: unknown;
};
type AsyncModuleScope = {
  environment: EvalEnvironment;
  generation: number;
  closed: boolean;
  instances: WeakMap<ModuleRecord, AsyncModuleState>;
  pending: WeakMap<ModuleRecord, Promise<Record<string, unknown>>>;
  tail: Promise<unknown>;
  check(): void;
  close(): void;
};
let sharedAsyncScope: AsyncModuleScope | undefined;
let asyncTaskScopes = new WeakMap<object, AsyncModuleScope>();
const asyncScopes = new Set<AsyncModuleScope>();

export function moduleRealmUsesAsync(): boolean {
  return typeof moduleRealm().evaluateModuleAsync === 'function';
}

function relativeModuleItem(specifier: string, level: Record<string, ItemRecord>,
    source: string): {item: ItemRecord; parent: Record<string, ItemRecord>} {
  if (!specifier.startsWith('./')) throw new NatlangSourceError(source, 'module import must be relative');
  const parts = specifier.slice(2).replace(/\.(ts|js|nl)$/, '').split('/');
  let scope: Record<string, ItemRecord> = level;
  let parent = level;
  let item: ItemRecord | undefined;
  for (const part of parts) {
    parent = scope;
    item = scope[part];
    if (!item) throw new NatlangSourceError(source,
      `cannot import ${JSON.stringify(specifier)}: no matching callable-folder item`);
    scope = item.codebase;
  }
  return {item: item!, parent};
}

function asyncScope(): AsyncModuleScope {
  const task = currentFrame()?.task;
  const isolated = !!task && (task.programView.binding || task.runtime.options.isolateModules || task.programView.revisionId);
  const existing = isolated ? asyncTaskScopes.get(task!) : sharedAsyncScope;
  if (existing) { existing.check(); return existing; }
  const environment = isolated ? defaultRealm?.() : moduleRealm();
  if (!environment || typeof environment.evaluateModuleAsync !== 'function')
    throw new Error('Selected asynchronous module evaluation is not installed');
  const scope: AsyncModuleScope = {
    environment, generation: realmGeneration, closed: false,
    instances: new WeakMap(), pending: new WeakMap(), tail: Promise.resolve(),
    check() {
      if (scope.closed || scope.generation !== realmGeneration || (isolated && task?.isClosed))
        throw new ModuleRealmChangedError();
    },
    close() {
      if (scope.closed) return;
      scope.closed = true;
      asyncScopes.delete(scope);
      if (isolated) environment.close();
    },
  };
  asyncScopes.add(scope);
  if (isolated) {
    asyncTaskScopes.set(task!, scope);
    task!.onClose(() => scope.close());
  } else sharedAsyncScope = scope;
  return scope;
}

/** Preserve descriptor getters and callable identity rather than snapshotting
 * exports. The checks also cover references handed out before authority cutover. */
function liveModuleExports(raw: Record<string, unknown>, scope: AsyncModuleScope): Record<string, unknown> {
  const functions = new WeakMap<Function, Function>();
  const expose = (value: unknown): unknown => {
    if (typeof value !== 'function') return value;
    let wrapped = functions.get(value);
    if (!wrapped) {
      wrapped = function (this: unknown, ...args: unknown[]) {
        scope.check();
        return Reflect.apply(value, this === facade ? raw : this, args);
      };
      functions.set(value, wrapped);
    }
    return wrapped;
  };
  const facade = new Proxy(Object.create(null) as Record<string, unknown>, {
    get(_target, key) { scope.check(); return expose(Reflect.get(raw, key, raw)); },
    has(_target, key) { scope.check(); return Reflect.has(raw, key); },
    ownKeys() { scope.check(); return Reflect.ownKeys(raw); },
    getOwnPropertyDescriptor(_target, key) {
      scope.check();
      const descriptor = Object.getOwnPropertyDescriptor(raw, key);
      if (!descriptor) return undefined;
      return { enumerable: descriptor.enumerable, configurable: true,
        get: () => { scope.check(); return expose(Reflect.get(raw, key, raw)); } };
    },
    set() { throw new TypeError('Module namespace bindings are read-only'); },
    defineProperty() { throw new TypeError('Module namespace bindings are read-only'); },
    deleteProperty() { throw new TypeError('Module namespace bindings are read-only'); },
  });
  return facade;
}

/** Initialize a callable module and all static sibling dependencies through the
 * asynchronous evaluator. No guest source executes in the native loader. */
export async function moduleInstanceAsync(record: ModuleRecord,
    level: Record<string, ItemRecord>): Promise<Record<string, unknown>> {
  const task = currentFrame()?.task;
  const scope = asyncScope();
  const project = (input: ModuleRecord): ModuleRecord => {
    const original = task?.programView.original(input) ?? input;
    const projected = task?.programView.record(original) ?? original;
    if (task?.programView.binding && !task.programView.patched(original.source) && projected.text !== original.text) {
      let originalPlans: InlineLambdaPlan[] = [], projectedPlans: InlineLambdaPlan[] = [];
      compileModule(original, level, plans => { originalPlans = plans; });
      compileModule(projected, task.programView.tree(level), plans => { projectedPlans = plans; });
      if (originalPlans.length !== projectedPlans.length || originalPlans.some((plan, index) =>
        inlineDescriptor(task.programView.program!.id, plan).contractHash !==
        inlineDescriptor(task.programView.program!.id, projectedPlans[index]!).contractHash))
        throw new NatlangSourceError(original.source, 'adapted module projection changed frozen inline contracts');
    }
    return projected;
  };
  record = project(record);
  const completed = scope.instances.get(record);
  if (completed?.ready) return completed.exports;
  const pending = scope.pending.get(record);
  if (pending) return pending;
  const initialize = async (current: ModuleRecord, siblings: Record<string, ItemRecord>,
      visiting: ReadonlySet<ModuleRecord>): Promise<Record<string, unknown>> => {
    scope.check();
    current = project(current);
    const found = scope.instances.get(current);
    if (found) {
      if (Object.hasOwn(found, 'failure')) throw found.failure;
      if (found.ready || visiting.has(current)) return found.exports;
      throw new Error('Module initialization escaped its graph transaction');
    }
    const code = compiledCode(current, siblings);
    const raw: Record<string, unknown> = {};
    // A dependency in the same import cycle can publish its selected binding
    // surface before this initializer runs. Declare the statically known names
    // as lazy cells now so its imported bindings continue to read this namespace
    // after initialization replaces the cell descriptors with live exports.
    for (const name of Object.keys(current.exports)) Object.defineProperty(raw, name, {
      get: () => undefined,
      set: value => { Object.defineProperty(raw, name, { value, writable: true, enumerable: true, configurable: true }); },
      enumerable: true, configurable: true,
    });
    const state: AsyncModuleState = {raw, exports: liveModuleExports(raw, scope), ready:false};
    scope.instances.set(current, state);
    try {
      const nested = new Set(visiting); nested.add(current);
      const dependencies = new Map<string, unknown>();
      for (const specifier of current.imports) {
        if (!specifier.startsWith('./')) continue;
        const {item, parent} = relativeModuleItem(specifier, siblings, current.source);
        if (item.kind === 'module') dependencies.set(specifier, await initialize(item, parent, nested));
        else if (item.kind === 'natlang') dependencies.set(specifier,
          {__esModule:true, default:namedCallable(item.name, item as never)});
        else dependencies.set(specifier, {__esModule:true, ...callableTree(item.codebase as never)});
      }
      const require = (specifier: string): unknown => {
        scope.check();
        if (dependencies.has(specifier)) return dependencies.get(specifier);
        if (specifier === 'natlang:services') return selectedServicesModule(current);
        if (builtinModules.has(specifier)) return builtinModule(specifier);
        if (specifier === NEUTRAL_RUNTIME_SPECIFIER || /^@natlang\/(node|browser|core)$/.test(specifier))
          return {__esModule:true, ...surface};
        if (specifier.startsWith('.') || specifier.startsWith('/'))
          throw new NatlangSourceError(current.source, `cannot import ${JSON.stringify(specifier)}: initialize a declared sibling import first`);
        if (!packageLoader) throw new NatlangSourceError(current.source, `package imports are unavailable here: ${specifier}`);
        return packageLoader(specifier);
      };
      const module = {exports:raw};
      const runtimeBindings = scope.environment.authority.startsWith('selected-semantic/') ? {
        ...lowered,
        // Context is loader authority, not a guest-synthesized module tree.
        inline: (plan: InlineLambdaPlan, values: unknown[], accessors: Record<string, unknown>, _context: unknown, version?: number) =>
          lowered.inlineSelected(plan, values, accessors, siblings, version),
      } : lowered;
      await scope.environment.evaluateModuleAsync!(code, {exports:raw, module, require,
        __natlang:runtimeBindings, __natlang_context:siblings, nl:surface.nl, iterateOn:surface.iterateOn,
        __natlang_module:{id:current.id,source:current.source,revision:current.revision,
          originalSource:current.text,imports:[...current.imports]}});
      scope.check();
      if (module.exports !== raw) Object.defineProperties(raw, Object.getOwnPropertyDescriptors(module.exports));
      if (!Object.hasOwn(raw, '__esModule')) Object.defineProperty(raw, '__esModule', {value:true});
      state.ready = true;
      return state.exports;
    } catch (error) {
      // Retrying initialization could repeat a live write. A failed revision
      // stays failed until explicit realm cutover or a new source revision.
      state.failure = error;
      throw error;
    }
  };
  const work = scope.tail.then(() => initialize(record, level, new Set()));
  scope.pending.set(record, work);
  scope.tail = work.catch(() => {});
  try { return await work; }
  finally { scope.pending.delete(record); }
}
