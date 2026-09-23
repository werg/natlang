import ts from 'typescript';
import { EvalFailure, type EnvironmentMode, type EvalEnvironment, type EvalRequest,
  type EvalResult, type HostEvent } from '../native/evaluator.js';

declare const __NATLANG_PRELUDE__: string;

function portable(value: unknown, seen = new Set<object>(), path = '$'): unknown {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number' && Number.isFinite(value) && (!Number.isInteger(value) || Number.isSafeInteger(value))) return value;
  if (!value || typeof value !== 'object' || ArrayBuffer.isView(value) || value instanceof ArrayBuffer)
    throw new TypeError(`eval result ${path} contains an unsupported or inexact ${typeof value} value`);
  if (typeof (value as Promise<unknown>).then === 'function')
    throw new TypeError('await the asynchronous imported function call');
  if (seen.has(value)) throw new TypeError('eval result contains a cycle');
  seen.add(value);
  if (Array.isArray(value)) {
    const out = value.map((item, index) => portable(item, seen, `${path}[${index}]`));
    seen.delete(value); return out;
  }
  if (Object.prototype.toString.call(value) !== '[object Object]')
    throw new TypeError('eval result contains a native object');
  const proto = Object.getPrototypeOf(value);
  if (proto !== null && (Object.getPrototypeOf(proto) !== null || proto.constructor?.name !== 'Object'))
    throw new TypeError('eval result contains a native class instance');
  const out: Record<string, unknown> = Object.create(null);
  for (const [key, child] of Object.entries(value)) out[key] = portable(child, seen, `${path}.${key}`);
  seen.delete(value); return out;
}

function snapshot(value: unknown): unknown {
  const copy = portable(value);
  function freeze(item: unknown): void {
    if (item && typeof item === 'object') { for (const child of Object.values(item)) freeze(child); Object.freeze(item); }
  }
  freeze(copy); return copy;
}

function scopeBridgeValue(value: unknown): unknown {
  if (value === undefined || value === null || typeof value === 'string' ||
      typeof value === 'boolean' || typeof value === 'number') return value;
  if (Array.isArray(value)) return Array.from(value, scopeBridgeValue);
  if (typeof value === 'object' || typeof value === 'function')
    return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, scopeBridgeValue(item)]));
  throw new TypeError(`scope bridge contains unsupported ${typeof value} value`);
}

function compile(code: string, body: boolean, asyncBody: boolean): string {
  const source = body ? `${asyncBody ? 'async ' : ''}function __natlang_body(self: unknown, fx: unknown, host: unknown) {\n${code}\n}\n__natlang_body(self,fx,host)` : code;
  const result = ts.transpileModule(source, { fileName: 'natlang-eval.ts', reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
      isolatedModules: true, removeComments: false } });
  const errors = (result.diagnostics ?? []).filter(d => d.category === ts.DiagnosticCategory.Error);
  if (errors.length) throw new SyntaxError('TypeScript: ' + errors.map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n')).join('; '));
  return result.outputText;
}

/** Parse a browser-host TypeScript function body without evaluating it. */
export function checkTypeScriptBody(code: string): string[] {
  const source = `async function __natlang_body(self: unknown, fx: unknown, host: unknown) {\n${code}\n}`;
  const result = ts.transpileModule(source, { fileName: 'natlang-source.ts', reportDiagnostics: true,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None,
      isolatedModules: true } });
  return (result.diagnostics ?? []).filter(d => d.category === ts.DiagnosticCategory.Error)
    .map(d => ts.flattenDiagnosticMessageText(d.messageText, '\n'));
}

type Evaluator = (scope: Record<string, unknown>, code: string, effect: (cap: string, fn: string, args: unknown[]) => unknown) => unknown;
/** Trusted browser evaluator. The host object is shared by identity and may be mutated. */
export class TypeScriptEnvironment implements EvalEnvironment {
  readonly authority = 'shared-browser-host';
  readonly mode: EnvironmentMode;
  readonly host: object;
  private evaluator?: Evaluator;
  private effect?: (capability: string, operation: string, args: unknown[]) => unknown;
  private disposed = false;
  private readonly observe?: (event: HostEvent) => void;

  constructor(options: { mode?: EnvironmentMode; host?: object; observe?: (event: HostEvent) => void;
    effect?: (capability: string, operation: string, args: unknown[]) => unknown } = {}) {
    this.mode = options.mode ?? 'fresh'; this.host = options.host ?? Object.freeze({});
    this.observe = options.observe; this.effect = options.effect;
  }

  bindEffect(handler: (capability: string, operation: string, args: unknown[]) => unknown): () => void {
    const previous = this.effect; this.effect = handler;
    return () => { if (this.effect === handler) this.effect = previous; };
  }

  fork(): TypeScriptEnvironment { return new TypeScriptEnvironment({ mode: 'fresh', host: this.host,
    observe: this.observe }); }

  private makeEvaluator(): Evaluator {
    const prelude = __NATLANG_PRELUDE__.split('const __deepFreeze')[0];
    const factory = new Function('host', `${prelude}\nlet self, locals;\n` +
      `function* evaluate() { let job=yield; while(true) {\n` +
      `const {scope,code,effect}=job; self=scope; locals=scope.let || {};\n` +
      `const fx=new Proxy({}, {get:(_,cap)=>new Proxy({}, {get:(_,fn)=>(...raw)=>effect(String(cap),String(fn),raw)})});\n` +
      `job=yield eval(code);\n} }\n` +
      `const runner=evaluate(); runner.next();\n` +
      `return function(scope,code,effect) { return runner.next({scope,code,effect}).value; }`) as (host: object) => Evaluator;
    return factory(this.host);
  }

  private capture(request: EvalRequest, status: string): HostEvent[] {
    const host = this.host as { drainEvents?: () => HostEvent[] };
    const events = typeof host.drainEvents === 'function' ? host.drainEvents() : [];
    for (const event of events) this.observe?.(event);
    const evalEvent = { operation: 'typescript.eval', mode: this.mode, body: request.body,
      effectful: request.effectful, sharedHost: true, status, nativeEffectsReplayable: false };
    this.observe?.(evalEvent); return [...events, evalEvent];
  }

  private run(request: EvalRequest, asyncBody: boolean): unknown {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    if (typeof request.code !== 'string' || typeof request.scope !== 'object' || request.scope === null)
      throw new TypeError('invalid eval request');
    const evaluator = this.mode === 'retained' ? (this.evaluator ??= this.makeEvaluator()) : this.makeEvaluator();
    const scope = snapshot(request.scope) as Record<string, unknown>;
    return evaluator(scope, compile(request.code, request.body, asyncBody), (cap, fn, raw) => {
      if (!this.effect) throw new Error('NATLANG:effect-undeclared');
      const args = cap === 'natlang' && fn === 'scope' ? scopeBridgeValue(raw) as unknown[] : portable(raw) as unknown[];
      const result = this.effect(cap, fn, args);
      if (result && typeof result === 'object' && typeof (result as Promise<unknown>).then === 'function')
        return Promise.resolve(result).then(portable);
      return portable(result);
    });
  }

  execute(request: EvalRequest): EvalResult {
    try {
      const value = this.run(request, false);
      if (value && typeof value === 'object' && typeof (value as Promise<unknown>).then === 'function')
        throw new TypeError('async eval results require a host job and later poll');
      return { result: portable(value === undefined ? null : value), events: this.capture(request, 'completed') };
    } catch (error) { throw new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed')); }
  }

  async executeAsync(request: EvalRequest): Promise<EvalResult> {
    try {
      const value = await this.run(request, request.body);
      return { result: portable(value === undefined ? null : value), events: this.capture(request, 'completed') };
    } catch (error) { throw new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed')); }
  }

  close(): void { this.disposed = true; this.evaluator = undefined; }
}
