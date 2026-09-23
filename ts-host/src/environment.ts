import { readFileSync } from 'node:fs';
import { createContext, runInContext, runInThisContext, type Context } from 'node:vm';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { WorkspaceModules, findPackageWorkspace } from './workspace-modules.js';
import { packageNameFromSpecifier } from './package-specifier.js';
import { EvalFailure, consoleWriter, withinTimeout, type EnvironmentMode, type EvalEnvironment, type EvalRequest,
  type EvalResult, type HostEvent } from './native/evaluator.js';
export { EvalFailure } from './native/evaluator.js';
export type { EnvironmentMode, EvalEnvironment, EvalRequest, EvalResult, HostEvent } from './native/evaluator.js';

const ARGUMENTS_KEY = '__natlang_arguments';
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
  // The body is a function expression; the scope, live values, and import hook arrive as its parameters,
  // so nothing of the runtime's plumbing is a global that eval code could find on globalThis.
  let source = body ? `(${asyncBody ? 'async ' : ''}function (self: any, __live: any, __natlang_import: any) {\n${code}\n})` +
    `(...(globalThis as any)[${JSON.stringify(ARGUMENTS_KEY)}])` : code;
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
  /** Synchronous run time allowed per eval; unlimited unless set. */
  readonly timeoutMs?: number;
  readonly scopeCapabilities: { allowModules: boolean; allowNetwork: boolean };
  readonly modules?: WorkspaceModules;
  private readonly workspace?: string;
  private readonly packageEvents: HostEvent[] = [];
  private context?: Context;
  private disposed = false;
  private readonly observe?: (event: HostEvent) => void;

  constructor(options: { mode?: EnvironmentMode; timeoutMs?: number; observe?: (event: HostEvent) => void;
    workspace?: string; network?: boolean } = {}) {
    this.mode = options.mode ?? 'fresh';
    this.timeoutMs = options.timeoutMs;
    this.observe = options.observe;
    this.workspace = options.workspace ?? findPackageWorkspace(process.cwd());
    this.scopeCapabilities = { allowModules: true, allowNetwork: options.network ?? true };
    if (this.workspace) this.modules = new WorkspaceModules(this.workspace, event => this.packageEvents.push(event));
    if (this.timeoutMs !== undefined && (!Number.isInteger(this.timeoutMs) || this.timeoutMs < 1)) throw new RangeError('timeoutMs must be positive');
  }

  fork(): TypeScriptEnvironment { return new TypeScriptEnvironment({ mode: 'fresh',
    timeoutMs: this.timeoutMs, observe: this.observe, workspace: this.workspace, network: this.scopeCapabilities.allowNetwork }); }

  private makeContext(): Context {
    // Node's host globals are available to eval code as they are to any module in the application.
    // A callback that eval code schedules may throw after its eval has finished; that is recorded as a host
    // event instead of becoming an uncaught exception that ends the application.
    const guarded = (callback: unknown) => typeof callback !== 'function' ? callback : (...args: unknown[]) => {
      const report = (error: unknown) => this.packageEvents.push({ operation: 'eval.callback-error',
        message: error instanceof Error ? error.message : String(error) });
      try { const value = (callback as (...items: unknown[]) => unknown)(...args);
        if (value && typeof (value as Promise<unknown>).catch === 'function') (value as Promise<unknown>).catch(report); }
      catch (error) { report(error); }
    };
    const context = createContext({ console: undefined, process, Buffer, clearTimeout, clearInterval, clearImmediate,
      setTimeout: (callback: unknown, ...rest: unknown[]) => setTimeout(guarded(callback) as () => void, ...rest as [number]),
      setInterval: (callback: unknown, ...rest: unknown[]) => setInterval(guarded(callback) as () => void, ...rest as [number]),
      setImmediate: (callback: unknown, ...rest: unknown[]) => setImmediate(guarded(callback) as (...items: unknown[]) => void, ...rest as []),
      queueMicrotask: (callback: unknown) => queueMicrotask(guarded(callback) as () => void),
      structuredClone, performance, crypto: globalThis.crypto,
      TextEncoder, TextDecoder, URL, URLSearchParams, AbortController, AbortSignal, Blob });
    runInContext(prelude, context);
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
    const write = consoleWriter(logs, value => { try { return JSON.stringify(portable(value)); } catch { return String(value); } });
    context.console = Object.freeze({ log: write, info: write, warn: write, error: write });
    return logs;
  }

  private readonly importModule = async (specifier: string) => {
    if (!this.modules) throw new Error(`cannot import ${specifier}: no package.json above ${process.cwd()}`);
    return this.modules.load(specifier);
  };

  private prepare(request: EvalRequest): { context: Context; logs: string[]; start: (code: string, timeout?: number) => unknown } {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    if (typeof request.code !== 'string' || typeof request.scope !== 'object' || request.scope === null)
      throw new TypeError('invalid eval request');
    const context = this.mode === 'retained' ? (this.context ??= this.makeContext()) : this.makeContext();
    const logs = this.captureConsole(context);
    const scope = snapshot(request.scope) as Record<string, unknown>;
    // The body picks its arguments up as it starts; the property is not enumerable and is removed right after.
    Object.defineProperty(context, ARGUMENTS_KEY, { value: [scope, request.live ?? {}, this.importModule],
      enumerable: false, configurable: true });
    return { context, logs, start: (code: string, timeout?: number) => {
      try { return runInContext(code, context, { ...(timeout === undefined ? {} : { timeout }), displayErrors: true }); }
      finally { delete (context as Record<string, unknown>)[ARGUMENTS_KEY]; }
    } };
  }

  private failure(request: EvalRequest, error: unknown, logs: string[]): EvalFailure {
    return new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed'),
      { sourceStack: error && typeof error === 'object' && 'stack' in error && typeof error.stack === 'string' ?
        error.stack : undefined, logs });
  }

  execute(request: EvalRequest): EvalResult {
    let logs: string[] = [];
    const timeout = request.timeoutMs ?? this.timeoutMs;
    try {
      const prepared = this.prepare(request); logs = prepared.logs;
      const code = '"use strict";\n' + compile(request.code, request.body);
      const value = prepared.start(code, timeout);
      if (value && typeof value === 'object' && typeof (value as Promise<unknown>).then === 'function')
        throw new TypeError('an asynchronous result needs executeAsync');
      return { result: value === undefined ? null : value, events: this.capture(request, 'completed'), logs };
    } catch (error) { throw this.failure(request, error, logs); }
  }

  async executeAsync(request: EvalRequest): Promise<EvalResult> {
    let logs: string[] = [];
    const timeout = request.timeoutMs ?? this.timeoutMs;
    try {
      const prepared = this.prepare(request); logs = prepared.logs;
      const code = '"use strict";\n' + compile(request.code, request.body, request.body, true, specifier => {
        packageNameFromSpecifier(specifier);
        if (!this.modules) throw new Error(`cannot import ${specifier}: no package.json above ${process.cwd()}`);
      });
      const value = await withinTimeout(Promise.resolve(prepared.start(code, timeout)), timeout);
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
