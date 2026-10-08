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
/** Platform modules such as `natlang:learning` (registered by the Node wiring) and `natlang:neuralese`. */
export function registerBuiltinModule(specifier: string, factory: () => unknown): void { builtinModules.set(specifier, factory); }
export function builtinModule(specifier: string): unknown {
  const factory = builtinModules.get(specifier);
  if (!factory) throw new Error(`${specifier} is not available on this platform`);
  return factory();
}

export function setPackageLoader(loader: (specifier: string) => unknown): void { packageLoader = loader; }
/** Browsers restore the task context after each await in callable-folder code. */
export function setModuleTarget(target: 'node' | 'browser'): void { moduleTarget = target; }
let realm: EvalEnvironment | undefined;
/** Installed by the platform entry point: the evaluator in which module instances live. */
export function setModuleRealm(factory: () => EvalEnvironment): void { defaultRealm = factory; realm = undefined; }
function moduleRealm(): EvalEnvironment {
  if (!defaultRealm) throw new Error('no module realm is installed for this platform');
  return realm ??= defaultRealm();
}

const NL_TAG = /\bnl\s*(?:<[^`]*>)?\s*`/;
const FOLDER = '/__natlang__/folder';

function natlangDeclaration(record: ItemRecord): string {
  if (record.kind !== 'natlang') return '';
  const known = new Set(Object.keys(record.types));
  const params = Object.entries(record.args).map(([raw, type]) =>
    `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${typeScriptText(type, known)}`);
  if (record.subtype === 'directory-reducer') params.unshift('folder: Folder');
  return `declare const fn: NatlangFunction<[${params.join(', ')}], ${typeScriptText(record.returns, known)}>;\nexport default fn;\n`;
}

/** Compile one module to CommonJS-style JavaScript for evaluation. */
export function compileModule(record: ModuleRecord, level: Record<string, ItemRecord>, inventory?: (plans: InlineLambdaPlan[]) => void): string {
  let plans = new Map<string, InlineLambdaPlan>();
  let readouts = new Set<string>();
  let conditionalReadouts = new Set<string>();
  let joins = new Set<string>();
  let concats = new Set<string>();
  let checker: ts.TypeChecker | undefined;
  const path = `${FOLDER}/${record.name}.ts`;
  const softTypes = JSON.stringify(record.types);
  if (NL_TAG.test(record.text) || /\bNeuralese\s*</.test(`${record.text}\n${softTypes}`)) {
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
    const parsed = ts.createSourceFile(path, record.text, ts.ScriptTarget.ES2022, true);
    for (const statement of parsed.statements) if (ts.isImportDeclaration(statement) && ts.isStringLiteral(statement.moduleSpecifier) &&
        !statement.moduleSpecifier.text.startsWith('.')) {
      const names = external.get(statement.moduleSpecifier.text) ?? new Set<string>();
      const bindings = statement.importClause?.namedBindings;
      if (bindings && ts.isNamedImports(bindings)) for (const element of bindings.elements) names.add((element.propertyName ?? element.name).text);
      external.set(statement.moduleSpecifier.text, names);
    }
    files[`${FOLDER}/__external.d.ts`] = [...external].filter(([specifier]) => !/^@natlang\//.test(specifier))
      .map(([specifier, names]) => `declare module ${JSON.stringify(specifier)} {\n${[...names].map(name =>
        `  export const ${name}: any;`).join('\n')}\n  const value: any;\n  export default value;\n}`).join('\n');
    const program = createVirtualProgram(files, { ...EVAL_COMPILER_OPTIONS, paths: { '@natlang/*': ['/__natlang__/surface.d.ts'] } });
    const file = program.getSourceFile(path)!;
    const analysis = analyzeInlineLambdas(program, [file], { displayPath: () => record.source,
      sourceRevision: record.revision, authored: true });
    const errors = analysis.diagnostics.filter(item => item.severity === 'error');
    if (errors.length) throw new NatlangSourceError(record.source, errors.map(item => `${item.line}:${item.column} ${item.message}`).join('\n'));
    readouts = new Set(analysis.readouts.filter(item => !item.kind).map(item => `${item.start}:${item.end}`));
    conditionalReadouts = new Set(analysis.readouts.filter(item => item.conditional).map(item => `${item.start}:${item.end}`));
    joins = new Set(analysis.readouts.filter(item => item.kind === 'join').map(item => `${item.start}:${item.end}`));
    concats = new Set(analysis.readouts.filter(item => item.kind === 'concat').map(item => `${item.start}:${item.end}`));
    analysis.plans.forEach(plan => { if (record.programId) plan.programId = record.programId; });
    inventory?.(analysis.plans);
    plans = new Map(analysis.plans.map(plan => [`${plan.sourceSpan.start}:${plan.sourceSpan.end}`, plan]));
    checker = program.getTypeChecker();
  }
  const output = ts.transpileModule(record.text, { fileName: `${record.name}.ts`, reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true, isolatedModules: true },
    transformers: { before: [natlangTransformer({ plans, checker, readouts, conditionalReadouts, joins, concats, runtime: '__natlang', context: '__natlang_context',
      constrained: true, guardPrefix: record.programId ? JSON.stringify([record.programId, record.id]) : record.id,
      modulePath: record.source, browser: moduleTarget === 'browser' })] } });
  const errors = (output.diagnostics ?? []).filter(item => item.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new NatlangSourceError(record.source, errors.map(item => ts.flattenDiagnosticMessageText(item.messageText, '\n')).join('; '));
  return output.outputText;
}

const instances = new WeakMap<ModuleRecord, { exports: Record<string, unknown>; ready: boolean }>();
// Compilation is immutable and shareable; mutable exports remain in the task's instance cache.
const compiled = new Map<string, string>();
function compiledCode(record: ModuleRecord, level: Record<string, ItemRecord>): string {
  const key = fingerprint({ target: moduleTarget, source: record.source, id: record.id, revision: record.revision,
    programId: record.programId ?? null, text: record.text, types: record.types,
    siblings: Object.entries(level).map(([name, item]) => ({ name, kind: item.kind,
      text: item.kind === 'namespace' ? null : item.text,
      ...(item.kind === 'natlang' ? { args: item.args, returns: item.returns, types: item.types, subtype: item.subtype } : {}) })) });
  const cached = compiled.get(key);
  if (cached !== undefined) return cached;
  const code = compileModule(record, level);
  if (compiled.size >= 256) compiled.delete(compiled.keys().next().value!);
  compiled.set(key, code); return code;
}

const servicesModule = new Proxy(Object.create(null), {
  get: (_, name) => typeof name === 'string' && name !== '__esModule' && name !== 'then' ? lowered.service(name) : undefined,
});

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
    if (/^@natlang\/(node|browser|core)$/.test(specifier)) return { __esModule: true, ...surface };
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
