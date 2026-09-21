import { readFileSync } from 'node:fs';
import { createContext, runInContext, type Context } from 'node:vm';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { EvalFailure, type EnvironmentMode, type EvalEnvironment, type EvalRequest,
  type EvalResult, type HostEvent } from './native/evaluator.js';
export { EvalFailure } from './native/evaluator.js';
export type { EnvironmentMode, EvalEnvironment, EvalRequest, EvalResult, HostEvent } from './native/evaluator.js';

const prelude = readFileSync(fileURLToPath(new URL('../prelude.js', import.meta.url)), 'utf8');

function snapshot(value: unknown): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value))) return value;
  if (Array.isArray(value)) return Object.freeze(value.map(snapshot));
  if (value && typeof value === 'object' && Object.getPrototypeOf(value) === Object.prototype) {
    const out: Record<string, unknown> = Object.create(null);
    for (const [key, item] of Object.entries(value)) out[key] = snapshot(item);
    return Object.freeze(out);
  }
  throw new TypeError('eval scope contains a nonportable value');
}

export function portable(value: unknown, seen = new Set<object>()): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value))) return value;
  if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) || value instanceof ArrayBuffer)
    throw new TypeError('eval result contains an unsupported or inexact value');
  if (seen.has(value)) throw new TypeError('eval result contains a cycle');
  seen.add(value);
  if (Array.isArray(value)) {
    const out = value.map(item => portable(item, seen));
    seen.delete(value); return out;
  }
  if (Object.prototype.toString.call(value) !== '[object Object]')
    throw new TypeError('eval result contains a native object');
  const proto = Object.getPrototypeOf(value);
  if (proto !== null && (Object.getPrototypeOf(proto) !== null || proto.constructor?.name !== 'Object'))
    throw new TypeError('eval result contains a native class instance');
  const out: Record<string, unknown> = Object.create(null);
  for (const [key, item] of Object.entries(value)) out[key] = portable(item, seen);
  seen.delete(value);
  return out;
}

function compile(code: string, body: boolean, asyncBody = false): string {
  const source = body ? `${asyncBody ? 'async ' : ''}function __natlang_body(self: unknown, args: unknown, fx: unknown, host: unknown) {\n${code}\n}\n__natlang_body(self,args,fx,host)` : code;
  const result = ts.transpileModule(source, {
    fileName: 'natlang-eval.ts', reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
      isolatedModules: true, removeComments: false },
  });
  const errors = (result.diagnostics ?? []).filter(d => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new SyntaxError('TypeScript: ' + errors.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')).join('; '));
  return result.outputText;
}

/** Trusted VM context. `host` is a direct reference to caller-owned native objects. */
export class TypeScriptEnvironment implements EvalEnvironment {
  readonly authority = 'shared-node-host';
  readonly mode: EnvironmentMode;
  readonly host: object;
  readonly timeoutMs: number;
  private context?: Context;
  private disposed = false;
  private readonly observe?: (event: HostEvent) => void;
  private effect?: (capability: string, operation: string, args: unknown[]) => unknown;

  constructor(options: { mode?: EnvironmentMode; host?: object; timeoutMs?: number;
    observe?: (event: HostEvent) => void;
    effect?: (capability: string, operation: string, args: unknown[]) => unknown } = {}) {
    this.mode = options.mode ?? 'fresh';
    this.host = options.host ?? Object.freeze({});
    this.timeoutMs = options.timeoutMs ?? 2000;
    this.observe = options.observe;
    this.effect = options.effect;
    if (!Number.isInteger(this.timeoutMs) || this.timeoutMs < 1) throw new RangeError('timeoutMs must be positive');
  }

  bindEffect(handler: (capability: string, operation: string, args: unknown[]) => unknown): () => void {
    const prior = this.effect;
    this.effect = handler;
    return () => { if (this.effect === handler) this.effect = prior; };
  }

  fork(): TypeScriptEnvironment { return new TypeScriptEnvironment({ mode: 'fresh', host: this.host,
    timeoutMs: this.timeoutMs, observe: this.observe }); }

  private makeContext(): Context {
    const context = createContext({ host: this.host, __fx: (cap: string, fn: string, raw: string) => {
      if (!this.effect) return JSON.stringify({ __error: 'effect-undeclared' });
      try { return JSON.stringify({ value: portable(this.effect(cap, fn, JSON.parse(raw))) }); }
      catch (error) { return JSON.stringify({ __error: error instanceof Error ? error.message : String(error) }); }
    },
      console: undefined });
    runInContext(prelude, context, { timeout: this.timeoutMs });
    return context;
  }

  execute(request: EvalRequest): EvalResult {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    if (typeof request.code !== 'string' || typeof request.scope !== 'object' || request.scope === null)
      throw new TypeError('invalid eval request');
    try {
      const context = this.mode === 'retained' ? (this.context ??= this.makeContext()) : this.makeContext();
      const scope = snapshot(request.scope) as Record<string, unknown>;
      context.self = scope;
      context.args = scope.args;
      context.locals = scope.let ?? {};
      const code = '"use strict";\n' + compile(request.code, request.body);
      const value = runInContext(code, context, { timeout: this.timeoutMs, displayErrors: true });
      if (value && typeof value === 'object' && typeof (value as Promise<unknown>).then === 'function')
        throw new TypeError('async eval results require a host job and later poll');
      const result = portable(value === undefined ? null : value);
      return { result, events: this.capture(request, 'completed') };
    } catch (error) {
      throw new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed'));
    }
  }

  async executeAsync(request: EvalRequest): Promise<EvalResult> {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    if (typeof request.code !== 'string' || typeof request.scope !== 'object' || request.scope === null)
      throw new TypeError('invalid eval request');
    try {
      const context = this.mode === 'retained' ? (this.context ??= this.makeContext()) : this.makeContext();
      const scope = snapshot(request.scope) as Record<string, unknown>;
      context.self = scope; context.args = scope.args; context.locals = scope.let ?? {};
      const previousFx = context.fx;
      context.fx = new Proxy({}, { get: (_, capability) => new Proxy({}, {
        get: (_, operation) => (...rawArgs: unknown[]) => {
          if (!this.effect) throw new Error('NATLANG:effect-undeclared');
          const args = JSON.parse(JSON.stringify(portable(rawArgs))) as unknown[];
          const result = this.effect(String(capability), String(operation), args);
          if (result && typeof result === 'object' && typeof (result as Promise<unknown>).then === 'function')
            return Promise.resolve(result).then(portable);
          return portable(result);
        },
      }) });
      const code = '"use strict";\n' + compile(request.code, request.body, request.body);
      try {
        const pending = runInContext(code, context, { timeout: this.timeoutMs, displayErrors: true });
        const value = await pending;
        const result = portable(value === undefined ? null : value);
        return { result, events: this.capture(request, 'completed') };
      } finally { context.fx = previousFx; }
    } catch (error) {
      throw new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed'));
    }
  }

  private capture(request: EvalRequest, status: string): HostEvent[] {
    const host = this.host as { drainEvents?: () => HostEvent[] };
    const events = typeof host.drainEvents === 'function' ? host.drainEvents() : [];
    for (const event of events) this.observe?.(event);
    const evalEvent = { operation: 'typescript.eval', mode: this.mode, body: request.body,
      effectful: request.effectful, sharedHost: true, status, nativeEffectsReplayable: false };
    this.observe?.(evalEvent);
    return [...events, evalEvent];
  }

  close(): void { this.disposed = true; this.context = undefined; }
}
