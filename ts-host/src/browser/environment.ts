import ts from 'typescript';
import { bindAwait } from '../runtime/context.js';
import { EvalFailure, consoleWriter, withinTimeout, type EnvironmentMode, type EvalEnvironment, type EvalRequest,
  type EvalResult, type HostEvent } from '../native/evaluator.js';

declare const __NATLANG_PRELUDE__: string;

function portable(value: unknown, seen = new Set<object>(), path = '$', scope = false): unknown {
  if (scope && value === undefined) return value;
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value))) return value;
  if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) || value instanceof ArrayBuffer)
    throw new TypeError(`eval result ${path} contains an unsupported or inexact ${typeof value} value`);
  if (typeof (value as Promise<unknown>).then === 'function')
    throw new TypeError('await the asynchronous imported function call');
  if (seen.has(value)) throw new TypeError('eval result contains a cycle');
  seen.add(value);
  if (Array.isArray(value)) {
    const out = value.map((item, index) => portable(item, seen, `${path}[${index}]`, scope));
    seen.delete(value); return out;
  }
  if (Object.prototype.toString.call(value) !== '[object Object]')
    throw new TypeError('eval result contains a native object');
  const proto = Object.getPrototypeOf(value);
  if (proto !== null && (Object.getPrototypeOf(proto) !== null || proto.constructor?.name !== 'Object'))
    throw new TypeError('eval result contains a native class instance');
  const out: Record<string, unknown> = Object.create(null);
  for (const [key, child] of Object.entries(value)) out[key] = portable(child, seen, `${path}.${key}`, scope);
  seen.delete(value); return out;
}

function snapshot(value: unknown): unknown {
  const copy = portable(value, new Set(), '$', true);
  function freeze(item: unknown): void {
    if (item && typeof item === 'object') { for (const child of Object.values(item)) freeze(child); Object.freeze(item); }
  }
  freeze(copy); return copy;
}

/** Browsers have no async context: restore the natlang task frame after every await in eval code. */
const restoreAfterAwait: ts.TransformerFactory<ts.SourceFile> = context => file => {
  const f = context.factory;
  const visit: ts.Visitor = node => ts.isAwaitExpression(node) ?
    f.createCallExpression(f.createParenthesizedExpression(f.createAwaitExpression(f.createCallExpression(
      f.createIdentifier('__natlang_bindAwait'), undefined, [ts.visitNode(node.expression, visit) as ts.Expression]))), undefined, []) :
    ts.visitEachChild(node, visit, context);
  return ts.visitNode(file, visit) as ts.SourceFile;
};

function compile(code: string, body: boolean, asyncBody: boolean): string {
  const source = body ? `${asyncBody ? 'async ' : ''}function __natlang_body(self: unknown) {\n${code}\n}\n__natlang_body(self)` : code;
  const result = ts.transpileModule(source, { fileName: 'natlang-eval.ts', reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
      isolatedModules: true, removeComments: false }, transformers: { before: [restoreAfterAwait] } });
  const errors = (result.diagnostics ?? []).filter(d => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new SyntaxError('TypeScript: ' + errors.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')).join('; '));
  return result.outputText;
}

type Evaluator = (scope: Record<string, unknown>, code: string, log: (...values: unknown[]) => void,
  live: Record<string, unknown>) => unknown;

/**
 * Browser evaluator. Portable scope data arrives as a frozen snapshot (`self`); live objects,
 * callables, captures and the output sink arrive by reference (`__live`).
 */
export class TypeScriptEnvironment implements EvalEnvironment {
  readonly authority = 'shared-browser-host';
  readonly mode: EnvironmentMode;
  private evaluator?: Evaluator;
  private disposed = false;
  private readonly observe?: (event: HostEvent) => void;

  constructor(options: { mode?: EnvironmentMode; observe?: (event: HostEvent) => void } = {}) {
    this.mode = options.mode ?? 'fresh';
    this.observe = options.observe;
  }

  fork(): TypeScriptEnvironment { return new TypeScriptEnvironment({ mode: 'fresh', observe: this.observe }); }

  /** Cancellers of timers eval code scheduled that have not run: cleared when the call ends or an eval fails. */
  private readonly timers = new Set<() => void>();

  /** setTimeout and clearTimeout for eval code: the timers belong to this environment's call. */
  private timerBindings(): { setTimeout: unknown; clearTimeout: unknown } {
    const timers = this.timers, cancellers = new Map<unknown, () => void>();
    return {
      setTimeout: (callback: unknown, ms?: number, ...args: unknown[]) => {
        // A callback that fails after its eval finished is recorded, not thrown into the page.
        const report = (error: unknown) => this.observe?.({ operation: 'eval.callback-error',
          message: error instanceof Error ? error.message : String(error) });
        const handle = setTimeout(() => {
          timers.delete(cancel); cancellers.delete(handle);
          if (typeof callback !== 'function') return;
          try { const value = (callback as (...items: unknown[]) => unknown)(...args);
            if (value && typeof (value as Promise<unknown>).catch === 'function') (value as Promise<unknown>).catch(report); }
          catch (error) { report(error); }
        }, ms);
        const cancel = () => clearTimeout(handle);
        timers.add(cancel); cancellers.set(handle, cancel);
        return handle;
      },
      clearTimeout: (handle: unknown) => {
        const cancel = cancellers.get(handle);
        if (cancel) { timers.delete(cancel); cancellers.delete(handle); }
        clearTimeout(handle as ReturnType<typeof setTimeout>);
      },
    };
  }

  private clearTimers(): void { for (const cancel of this.timers) cancel(); this.timers.clear(); }

  private makeEvaluator(): Evaluator {
    // Eval code sees these timer bindings instead of the page's; setInterval is not offered (repeat with iterateOn).
    const factory = new Function('__natlang_bindAwait', '__natlang_timers', `${__NATLANG_PRELUDE__}\nlet self, __live;\n` +
      `const { setTimeout, clearTimeout } = __natlang_timers; const setInterval = undefined, clearInterval = undefined;\n` +
      `function* evaluate() { let job=yield; while(true) {\n` +
      `const {scope,code,log,live}=job; self=scope; __live=live;\n` +
      `const console=Object.freeze({log,info:log,warn:log,error:log});\n` +
      `job=yield eval(code);\n} }\n` +
      `const runner=evaluate(); runner.next();\n` +
      `return function(scope,code,log,live) { return runner.next({scope,code,log,live}).value; }`) as (bind: typeof bindAwait, timers: unknown) => Evaluator;
    return factory(bindAwait, this.timerBindings());
  }

  private capture(request: EvalRequest, status: string): HostEvent[] {
    const evalEvent = { operation: 'typescript.eval', mode: this.mode, body: request.body, status };
    this.observe?.(evalEvent);
    return [evalEvent];
  }

  private run(request: EvalRequest, asyncBody: boolean, logs: string[]): unknown {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    if (typeof request.code !== 'string' || typeof request.scope !== 'object' || request.scope === null)
      throw new TypeError('invalid eval request');
    const evaluator = this.mode === 'retained' ? (this.evaluator ??= this.makeEvaluator()) : this.makeEvaluator();
    const log = consoleWriter(logs, value => { try { return JSON.stringify(portable(value)); } catch { return String(value); } });
    return evaluator(snapshot(request.scope) as Record<string, unknown>, compile(request.code, request.body, asyncBody),
      log, request.live ?? {});
  }

  private failure(request: EvalRequest, error: unknown, logs: string[]): EvalFailure {
    return new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed'),
      { sourceStack: error && typeof error === 'object' && 'stack' in error && typeof error.stack === 'string' ?
        error.stack : undefined, logs });
  }

  execute(request: EvalRequest): EvalResult {
    const logs: string[] = [];
    try {
      const value = this.run(request, false, logs);
      if (value && typeof value === 'object' && typeof (value as Promise<unknown>).then === 'function')
        throw new TypeError('an asynchronous result needs executeAsync');
      return { result: value === undefined ? null : value, events: this.capture(request, 'completed'), logs };
    } catch (error) { throw this.failure(request, error, logs); }
  }

  async executeAsync(request: EvalRequest): Promise<EvalResult> {
    const logs: string[] = [];
    try {
      const value = await withinTimeout(Promise.resolve(this.run(request, request.body, logs)), request.timeoutMs);
      return { result: value === undefined ? null : value, events: this.capture(request, 'completed'), logs };
    } catch (error) { this.clearTimers(); throw this.failure(request, error, logs); }
  }

  /** Evaluate a callable-folder module body in the page realm; bindings are passed by reference. */
  evaluateModule(code: string, bindings: Record<string, unknown>): unknown {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    const names = Object.keys(bindings);
    return new Function(...names, `"use strict";\n${code}`)(...names.map(name => bindings[name]));
  }

  close(): void { this.disposed = true; this.evaluator = undefined; this.clearTimers(); }
}
