/**
 * Module instances for callable-folder TypeScript. A module is compiled with the shared lowering
 * (entry guards, finite iteration, `nl`, `iterateOn` sites) and evaluated once per source revision.
 */
import ts from 'typescript';
import type { EvalEnvironment } from '../native/evaluator.js';
import { createVirtualProgram, EVAL_COMPILER_OPTIONS } from '../compiler/host.js';
import { analyzeInlineLambdas, type InlineLambdaPlan } from '../compiler/inline.js';
import { natlangTransformer } from '../compiler/lower.js';
import { typeScriptText } from '../compiler/eval-check.js';
import { NatlangSourceError, type ItemRecord, type ModuleRecord } from './loader.js';
import * as lowered from './lowered.js';
import * as surface from './surface.js';
import { namedCallable, callableTree } from './callable.js';

let defaultRealm: (() => EvalEnvironment) | undefined;
let moduleTarget: 'node' | 'browser' = 'node';
let packageLoader: ((specifier: string) => unknown) | undefined;
/** Installed by the platform: how callable-folder modules load packages. */
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
export function compileModule(record: ModuleRecord, level: Record<string, ItemRecord>): string {
  let plans = new Map<string, InlineLambdaPlan>();
  let checker: ts.TypeChecker | undefined;
  const path = `${FOLDER}/${record.name}.ts`;
  if (NL_TAG.test(record.text)) {
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
      sourceRevision: record.revision });
    const errors = analysis.diagnostics.filter(item => item.severity === 'error');
    if (errors.length) throw new NatlangSourceError(record.source, errors.map(item => `${item.line}:${item.column} ${item.message}`).join('\n'));
    plans = new Map(analysis.plans.map(plan => [`${plan.sourceSpan.start}:${plan.sourceSpan.end}`, plan]));
    checker = program.getTypeChecker();
  }
  const output = ts.transpileModule(record.text, { fileName: `${record.name}.ts`, reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS, esModuleInterop: true, isolatedModules: true },
    transformers: { before: [natlangTransformer({ plans, checker, runtime: '__natlang', context: '__natlang_context',
      constrained: true, guardPrefix: record.id, modulePath: record.source, browser: moduleTarget === 'browser' })] } });
  const errors = (output.diagnostics ?? []).filter(item => item.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new NatlangSourceError(record.source, errors.map(item => ts.flattenDiagnosticMessageText(item.messageText, '\n')).join('; '));
  return output.outputText;
}

const instances = new WeakMap<ModuleRecord, { exports: Record<string, unknown>; ready: boolean }>();

const servicesModule = new Proxy(Object.create(null), {
  get: (_, name) => typeof name === 'string' && name !== '__esModule' && name !== 'then' ? lowered.service(name) : undefined,
});

/** The live exports of a module in its folder `level` (its sibling items). */
export function moduleInstance(record: ModuleRecord, level: Record<string, ItemRecord>): Record<string, unknown> {
  const existing = instances.get(record);
  if (existing) return existing.exports;
  const code = compileModule(record, level);
  const exports: Record<string, unknown> = {};
  const state = { exports, ready: false };
  instances.set(record, state);
  const environment = moduleRealm();
  const require = (specifier: string): unknown => {
    if (specifier === 'natlang:services') return servicesModule;
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
    instances.delete(record);
    throw error;
  }
  if (module.exports !== exports) Object.assign(exports, module.exports);
  Object.defineProperty(exports, '__esModule', { value: true });
  state.ready = true;
  return exports;
}
