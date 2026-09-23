import { readFileSync } from 'node:fs';
import { createContext, runInContext, runInThisContext, type Context } from 'node:vm';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { ApplicationPackages, findPackageWorkspace } from './application-packages.js';
import { EvalFailure, type EnvironmentMode, type EvalEnvironment, type EvalRequest,
  type EvalResult, type HostEvent } from './native/evaluator.js';
export { EvalFailure } from './native/evaluator.js';
export type { EnvironmentMode, EvalEnvironment, EvalRequest, EvalResult, HostEvent } from './native/evaluator.js';

const prelude = readFileSync(fileURLToPath(new URL('../prelude.js', import.meta.url)), 'utf8');

function snapshot(value: unknown): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value))) return value;
  if (Array.isArray(value)) return Object.freeze(Array.from(value, snapshot));
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
    const out: Record<string, unknown> = Object.create(null);
    for (const [key, item] of Object.entries(value)) out[key] = snapshot(item);
    return Object.freeze(out);
  }
  throw new TypeError('eval scope contains a nonportable value');
}

export function portable(value: unknown, seen = new Set<object>(), path = '$'): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value))) return value;
  if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) || value instanceof ArrayBuffer)
    throw new TypeError(`eval result ${path} contains an unsupported or inexact ${typeof value} value`);
  if (typeof (value as Promise<unknown>).then === 'function')
    throw new TypeError('await the asynchronous imported function call');
  if (seen.has(value)) throw new TypeError('eval result contains a cycle');
  seen.add(value);
  if (Array.isArray(value)) {
    const out = Array.from(value, (item, index) => portable(item, seen, `${path}[${index}]`));
    seen.delete(value); return out;
  }
  if (Object.prototype.toString.call(value) !== '[object Object]')
    throw new TypeError('eval result contains a native object');
  const proto = Object.getPrototypeOf(value);
  if (proto !== null && (Object.getPrototypeOf(proto) !== null || proto.constructor?.name !== 'Object'))
    throw new TypeError('eval result contains a native class instance');
  const out: Record<string, unknown> = Object.create(null);
  for (const [key, item] of Object.entries(value)) out[key] = portable(item, seen, `${path}.${key}`);
  seen.delete(value);
  return out;
}

function lowerStaticImports(source: string, validateImport: (specifier: string) => void): string {
  const file = ts.createSourceFile('imports.ts', source, ts.ScriptTarget.Latest, true);
  const edits: { start: number; end: number; text: string }[] = [];
  const visit = (node: ts.Node) => {
    if (ts.isImportDeclaration(node)) {
      if (!ts.isStringLiteral(node.moduleSpecifier)) throw new Error('Package import must use a string literal');
      validateImport(node.moduleSpecifier.text);
      const clause = node.importClause;
      const call = `await __natlang_import(${node.moduleSpecifier.getText(file)})`;
      const fields: string[] = [];
      let text = '';
      const typeOnlyBindings = clause?.namedBindings && ts.isNamedImports(clause.namedBindings)
        && clause.namedBindings.elements.length > 0 && clause.namedBindings.elements.every(binding => binding.isTypeOnly);
      if (!clause?.isTypeOnly && !(typeOnlyBindings && !clause?.name)) {
        if (clause?.name) fields.push(`default: ${clause.name.text}`);
        const bindings = clause?.namedBindings;
        if (bindings && ts.isNamedImports(bindings)) for (const binding of bindings.elements)
          if (!binding.isTypeOnly) fields.push(`${binding.propertyName?.text ?? binding.name.text}: ${binding.name.text}`);
        if (fields.length) text += `const { ${fields.join(', ')} } = ${call};`;
        if (bindings && ts.isNamespaceImport(bindings)) text += `const ${bindings.name.text} = ${call};`;
        if (!text) text = `${call};`;
      }
      edits.push({start:node.getStart(file),end:node.end,text});
    } else ts.forEachChild(node,visit);
  };
  visit(file);
  for (const edit of edits.sort((a,b)=>b.start-a.start)) source = source.slice(0,edit.start)+edit.text+source.slice(edit.end);
  return source;
}

function compile(code: string, body: boolean, asyncBody = false, modules = false,
  validateImport: (specifier: string) => void = () => {}): string {
  let source = body ? `${asyncBody ? 'async ' : ''}function __natlang_body(self: unknown) {\n${code}\n}\n__natlang_body(self)` : code;
  // Reparse lowered imports before TS binding; otherwise TS rewrites uses to removed import aliases.
  if (modules) source = lowerStaticImports(source, validateImport);
  const result = ts.transpileModule(source, {
    fileName: 'natlang-eval.ts', reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
      isolatedModules: true, removeComments: false },
    transformers: modules ? { before: [context => root => {
      const visit: ts.Visitor = node => {
        if (ts.isCallExpression(node) && node.expression.kind === ts.SyntaxKind.ImportKeyword)
          return ts.factory.updateCallExpression(node, ts.factory.createIdentifier('__natlang_import'), undefined,
            node.arguments.map(arg => ts.visitNode(arg, visit) as ts.Expression));
        return ts.visitEachChild(node, visit, context);
      };
      return ts.visitNode(root, visit) as ts.SourceFile;
    }] } : undefined,
  });
  const errors = (result.diagnostics ?? []).filter(d => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new SyntaxError('TypeScript: ' + errors.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')).join('; '));
  return result.outputText;
}

/**
 * Node evaluator: a `vm` context in this process. Portable scope data arrives as a frozen snapshot
 * (`self`); live objects, callables, captures and the output sink arrive by reference (`__live`).
 */
export class TypeScriptEnvironment implements EvalEnvironment {
  readonly authority = 'shared-node-host';
  readonly mode: EnvironmentMode;
  readonly timeoutMs: number;
  readonly scopeCapabilities: { allowModules: boolean; allowNetwork: boolean };
  readonly packages?: ApplicationPackages;
  private readonly workspace?: string;
  private readonly packageEvents: HostEvent[] = [];
  private context?: Context;
  private disposed = false;
  private readonly observe?: (event: HostEvent) => void;

  constructor(options: { mode?: EnvironmentMode; timeoutMs?: number; observe?: (event: HostEvent) => void;
    workspace?: string; network?: boolean } = {}) {
    this.mode = options.mode ?? 'fresh';
    this.timeoutMs = options.timeoutMs ?? 2000;
    this.observe = options.observe;
    this.workspace = options.workspace ?? findPackageWorkspace(process.cwd());
    this.scopeCapabilities = { allowModules: true, allowNetwork: options.network ?? !!this.workspace };
    if (this.workspace) this.packages = new ApplicationPackages(this.workspace, event => this.packageEvents.push(event));
    if (!Number.isInteger(this.timeoutMs) || this.timeoutMs < 1) throw new RangeError('timeoutMs must be positive');
  }

  fork(): TypeScriptEnvironment { return new TypeScriptEnvironment({ mode: 'fresh',
    timeoutMs: this.timeoutMs, observe: this.observe, workspace: this.workspace, network: this.scopeCapabilities.allowNetwork }); }

  installPackages(specifiers: string[]): ReturnType<ApplicationPackages['installPackages']> {
    if (!this.packages) throw new Error('Package installation requires an application workspace');
    return this.packages.installPackages(specifiers);
  }

  private makeContext(): Context {
    const context = createContext({ console: undefined });
    runInContext(prelude, context, { timeout: this.timeoutMs });
    context.__natlang_import = (specifier: string) => {
      if (!this.packages) throw new Error('Package imports require a project package.json');
      return this.packages.importModule(specifier);
    };
    if (this.packages) context.installPackages = (specifiers: string[]) => this.installPackages(specifiers);
    if (this.scopeCapabilities.allowNetwork) Object.assign(context, {
      fetch: async (input: string | URL | Request, init?: RequestInit) => {
        const url = new URL(input instanceof Request ? input.url : String(input));
        const event = { operation: 'network.fetch', url: url.origin + url.pathname,
          method: init?.method ?? (input instanceof Request ? input.method : 'GET'), replayable: false };
        try {
          const response = await globalThis.fetch(input, init);
          this.packageEvents.push({ ...event, status: response.status });
          return response;
        } catch (error) { this.packageEvents.push({ ...event, status: 'failed' }); throw error; }
      }, URL, URLSearchParams, Headers, Request, Response, AbortController, AbortSignal,
    });
    return context;
  }

  private captureConsole(context: Context): string[] {
    const logs: string[] = [];
    const write = (...values: unknown[]) => {
      if (logs.length >= 32) return;
      const line = values.map(value => {
        if (typeof value === 'string') return value;
        if (value === undefined) return 'undefined';
        try { return JSON.stringify(portable(value)); }
        catch { return String(value); }
      }).join(' ');
      logs.push(line.length > 2000 ? `${line.slice(0, 2000)} …` : line);
    };
    context.console = Object.freeze({ log: write, info: write, warn: write, error: write });
    return logs;
  }

  private prepare(request: EvalRequest): { context: Context; logs: string[] } {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    if (typeof request.code !== 'string' || typeof request.scope !== 'object' || request.scope === null)
      throw new TypeError('invalid eval request');
    const context = this.mode === 'retained' ? (this.context ??= this.makeContext()) : this.makeContext();
    const logs = this.captureConsole(context);
    const scope = snapshot(request.scope) as Record<string, unknown>;
    context.self = scope;
    context.__live = request.live ?? {};
    return { context, logs };
  }

  private failure(request: EvalRequest, error: unknown, logs: string[]): EvalFailure {
    return new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed'),
      { sourceStack: error && typeof error === 'object' && 'stack' in error && typeof error.stack === 'string' ?
        error.stack : undefined, logs });
  }

  execute(request: EvalRequest): EvalResult {
    let logs: string[] = [];
    try {
      const prepared = this.prepare(request); logs = prepared.logs;
      const code = '"use strict";\n' + compile(request.code, request.body);
      const value = runInContext(code, prepared.context, { timeout: this.timeoutMs, displayErrors: true });
      if (value && typeof value === 'object' && typeof (value as Promise<unknown>).then === 'function')
        throw new TypeError('an asynchronous result needs executeAsync');
      return { result: value === undefined ? null : value, events: this.capture(request, 'completed'), logs };
    } catch (error) { throw this.failure(request, error, logs); }
  }

  async executeAsync(request: EvalRequest): Promise<EvalResult> {
    let logs: string[] = [];
    try {
      const prepared = this.prepare(request); logs = prepared.logs;
      const code = '"use strict";\n' + compile(request.code, request.body, request.body, true, specifier => {
        if (!this.packages) throw new Error('Package imports require a project package.json');
        this.packages.validateImportSpecifier(specifier);
      });
      const value = await runInContext(code, prepared.context, { timeout: this.timeoutMs, displayErrors: true });
      return { result: value === undefined ? null : value, events: this.capture(request, 'completed'), logs };
    } catch (error) { throw this.failure(request, error, logs); }
  }

  /**
   * Evaluate a callable-folder module body in the host realm, so its values interoperate with
   * application code (arrays, classes, Promises). Bindings are passed by reference.
   */
  evaluateModule(code: string, bindings: Record<string, unknown>): unknown {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    const names = Object.keys(bindings);
    const factory = runInThisContext(`(function (${names.join(', ')}) {\n"use strict";\n${code}\n})`,
      { displayErrors: true }) as (...values: unknown[]) => unknown;
    return factory(...names.map(name => bindings[name]));
  }



  private capture(request: EvalRequest, status: string): HostEvent[] {
    const events = this.packageEvents.splice(0);
    for (const event of events) this.observe?.(event);
    const evalEvent = { operation: 'typescript.eval', mode: this.mode, body: request.body, status };
    this.observe?.(evalEvent);
    return [...events, evalEvent];
  }

  close(): void { this.disposed = true; this.context = undefined; }
}
