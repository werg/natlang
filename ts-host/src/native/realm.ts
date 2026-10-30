/**
 * The evaluator over a realm. A platform supplies only a `Realm`, a way to run JavaScript with named bindings in its
 * global environment (a `node:vm` context, the Function constructor, a host-provided function constructor); everything
 * else an evaluator does is shared: TypeScript compilation (eval-compile.ts), the portable scope snapshot, the console,
 * the result checks and the failure reports.
 */
import { EvalFailure, consoleWriter, withinTimeout, type EnvironmentMode, type EvalEnvironment, type EvalRequest,
  type EvalResult, type HostEvent } from './evaluator.js';
import { compileEval, evalRealmBody } from './eval-compile.js';

/** Where compiled eval code runs: a JavaScript realm's global environment. */
export interface Realm {
  /**
   * Run `body` as the body of a strict-mode function whose parameters are `parameters`, called with `args`, with this
   * realm's global object as its global scope (the Function constructor's contract, strict), and return what it returns.
   * A promise it returns is the caller's to await.
   *
   * `timeoutMs` bounds the synchronous part of the call where the realm can interrupt running code (a `node:vm`
   * context can); a realm that cannot ignores it, and the caller bounds the asynchronous result itself.
   */
  call(body: string, parameters: readonly string[], args: readonly unknown[], options?: { timeoutMs?: number }): unknown;
}

/** A realm over a host-provided function constructor: `newFunction(body, name, ...parameterNames)`. */
export function constructorRealm(newFunction: (body: string, name: string, ...parameters: string[]) => (...args: unknown[]) => unknown,
    name = 'natlang'): Realm {
  return { call: (body, parameters, args) => newFunction(`"use strict";\n${body}`, name, ...parameters)(...args) };
}

export function portable(value: unknown, seen = new Set<object>(), path = '$', scope = false): unknown {
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

/** Eval scope follows TypeScript: an absent optional property may be undefined. Published results use `portable`. */
export function snapshot(value: unknown): unknown {
  const copy = portable(value, new Set(), '$', true);
  function freeze(item: unknown): void {
    if (item && typeof item === 'object') { for (const child of Object.values(item)) freeze(child); Object.freeze(item); }
  }
  freeze(copy); return copy;
}

/** Static configuration of a `RealmEnvironment`; what depends on the instance is a protected method a subclass overrides. */
export type RealmEnvironmentOptions = {
  authority: string;
  mode?: EnvironmentMode;
  /** The realm eval code runs in: asked for each evaluation in `fresh` mode, once in `retained` mode. */
  realm?: () => Realm;
  /** Synchronous run time allowed per eval where the realm can interrupt; unlimited unless set. */
  timeoutMs?: number;
  scopeCapabilities?: { allowModules: boolean; allowNetwork: boolean };
  observe?: (event: HostEvent) => void;
  /** Restores the natlang task after each `await` in eval code (platforms whose context store cannot follow promises). */
  bindAwait?: <T>(value: T | PromiseLike<T>) => Promise<() => T>;
  /** A new environment of the same kind, for `fork()`. */
  fork?: () => EvalEnvironment;
};

/** Package imports in eval: validates a specifier (throws for one that may not be imported) and loads it. */
export type EvalImports = { validate(specifier: string): void; load(specifier: string): Promise<unknown> };

/**
 * The evaluator over a realm. Portable scope data arrives as a frozen snapshot (`self`); live objects, callables,
 * captures and the output sink arrive by reference (`__live`). A subclass adds what its realm needs around that: its
 * own realm, package imports, extra bindings, host events and cleanup.
 */
export class RealmEnvironment implements EvalEnvironment {
  readonly authority: string;
  readonly mode: EnvironmentMode;
  readonly timeoutMs?: number;
  readonly scopeCapabilities: { allowModules: boolean; allowNetwork: boolean };
  protected disposed = false;
  private retained?: Realm;

  constructor(protected readonly options: RealmEnvironmentOptions) {
    this.authority = options.authority;
    this.mode = options.mode ?? 'fresh';
    this.timeoutMs = options.timeoutMs;
    this.scopeCapabilities = options.scopeCapabilities ?? { allowModules: false, allowNetwork: false };
    if (this.timeoutMs !== undefined && (!Number.isInteger(this.timeoutMs) || this.timeoutMs < 1)) throw new RangeError('timeoutMs must be positive');
  }

  fork(): EvalEnvironment {
    if (!this.options.fork) throw new Error(`${this.authority} cannot fork`);
    return this.options.fork();
  }

  /** A realm for one evaluation (`fresh`) or for the environment's life (`retained`). */
  protected makeRealm(): Realm {
    if (!this.options.realm) throw new Error(`${this.authority} has no realm`);
    return this.options.realm();
  }
  /** The realm callable-folder modules run in; by default the eval realm. */
  protected moduleRealm(): Realm { return this.realm(); }
  /** Package imports for async evals, if the platform has packages. */
  protected importer(): EvalImports | undefined { return undefined; }
  /** Names the realm binds for eval code besides the standard ones (timers, for example). */
  protected extraBindings(): Record<string, unknown> { return {}; }
  /** Events the realm's host side recorded since the last eval (callback errors, unhandled rejections, fetches). */
  protected hostEvents(): HostEvent[] { return []; }
  /** Called when an asynchronous eval fails: cancel what the eval scheduled. */
  protected abandon(): void {}
  /** Called by `close()`: release what the environment owns. */
  protected release(): void {}

  private realm(): Realm { return this.mode === 'retained' ? (this.retained ??= this.makeRealm()) : this.makeRealm(); }

  private capture(request: EvalRequest, status: string): HostEvent[] {
    const events = this.hostEvents();
    for (const event of events) this.options.observe?.(event);
    const evalEvent = { operation: 'typescript.eval', mode: this.mode, body: request.body, status };
    this.options.observe?.(evalEvent);
    return [...events, evalEvent];
  }

  private failure(request: EvalRequest, error: unknown, logs: string[]): EvalFailure {
    return new EvalFailure(error instanceof Error ? error.message : String(error), this.capture(request, 'failed'),
      { sourceStack: error && typeof error === 'object' && 'stack' in error && typeof error.stack === 'string' ?
        error.stack : undefined, logs });
  }

  private run(request: EvalRequest, asyncBody: boolean, logs: string[]): unknown {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    if (typeof request.code !== 'string' || typeof request.scope !== 'object' || request.scope === null)
      throw new TypeError('invalid eval request');
    const imports = asyncBody ? this.importer() : undefined;
    const compiled = compileEval(request.code, { body: request.body, asyncBody, imports: imports?.validate.bind(imports),
      restoreAfterAwait: !!this.options.bindAwait });
    const realm = this.realm();
    const write = consoleWriter(logs, value => { try { return JSON.stringify(portable(value)); } catch { return String(value); } });
    const logger = Object.freeze({ log: write, info: write, warn: write, error: write });
    const bound = { self: snapshot(request.scope), __live: request.live ?? {}, __natlang_import: (specifier: string) => {
      if (!imports) throw new Error(`cannot import ${specifier}: eval has no package imports here`);
      return imports.load(specifier);
    }, console: logger, ...(this.options.bindAwait ? { __natlang_bindAwait: this.options.bindAwait } : {}), ...this.extraBindings() };
    return realm.call(evalRealmBody(compiled, request.body), Object.keys(bound), Object.values(bound),
      { timeoutMs: request.timeoutMs ?? this.timeoutMs });
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
      const timeout = request.timeoutMs ?? this.timeoutMs;
      const value = await withinTimeout(Promise.resolve(this.run(request, request.body, logs)), timeout);
      return { result: value === undefined ? null : value, events: this.capture(request, 'completed'), logs };
    } catch (error) { this.abandon(); throw this.failure(request, error, logs); }
  }

  /** Evaluate a callable-folder module body; bindings are passed by reference. */
  evaluateModule(code: string, bindings: Record<string, unknown>): unknown {
    if (this.disposed) throw new Error('TypeScript environment is disposed');
    return this.moduleRealm().call(code, Object.keys(bindings), Object.values(bindings));
  }

  close(): void { this.disposed = true; this.retained = undefined; this.release(); }
}
