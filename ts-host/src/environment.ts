import { createContext, runInContext, runInThisContext, type Context } from 'node:vm';
import { WorkspaceModules, findPackageWorkspace, packageDeclaration } from './workspace-modules.js';
import { packageNameFromSpecifier } from './package-specifier.js';
import { markRaces } from './runtime/context.js';
import type { EnvironmentMode, HostEvent } from './native/evaluator.js';
import { RealmEnvironment, type EvalImports, type Realm } from './native/realm.js';
export { EvalFailure } from './native/evaluator.js';
export type { EnvironmentMode, EvalEnvironment, EvalRequest, EvalResult, HostEvent } from './native/evaluator.js';

const ARGUMENTS_KEY = '__natlang_arguments';

/** Raw native bindings and process-ending calls that model-written eval code must not reach: a malformed call into
 * process.binding('fs') aborted the whole collector natively (2026-10-07, 96 workers lost). Everything else on
 * process (env, cwd, argv, hrtime, ...) stays as it was. */
const HIDDEN_PROCESS_MEMBERS = new Set(['binding', '_linkedBinding', 'dlopen', 'abort', 'reallyExit', '_kill',
  'kill', 'exit', 'chdir', 'setuid', 'setgid', 'seteuid', 'setegid', 'setgroups', 'initgroups']);
const evalProcess = new Proxy(process, {
  get(target, key, receiver) {
    if (typeof key === 'string' && HIDDEN_PROCESS_MEMBERS.has(key))
      return () => { throw new Error(`process.${key} is not available to evaluated code`); };
    return Reflect.get(target, key, receiver);
  },
});

/**
 * Rejections nothing handles of promises eval code made: its realm's Promise is theirs. Model code can drop a promise
 * (an async callback given to forEach) whose rejection would end the host as an unhandled rejection; it is recorded
 * as a host event of the eval's environment instead. Any other unhandled rejection is the host's own and, with no
 * other listener, still ends the process as it would without this one.
 *
 * `process` has one `unhandledRejection` event, so the routing is shared, but each open environment owns its share
 * of it: the listener is installed when the first environment that ran eval code is open and removed when the last
 * one closes.
 */
class RejectionRouter {
  private readonly owners = new WeakMap<object, (reason: unknown) => void>();
  private open = 0;
  private readonly listener = (reason: unknown, promise: Promise<unknown>) => {
    const owner = this.owners.get((promise as { constructor?: object }).constructor ?? {});
    if (owner) return owner(reason);
    if (process.listenerCount('unhandledRejection') === 1) throw reason;
  };
  /** Take a share of the routing until the returned release is called. */
  acquire(): () => void {
    if (this.open++ === 0) process.on('unhandledRejection', this.listener);
    let released = false;
    return () => {
      if (released) return;
      released = true;
      if (--this.open === 0) process.off('unhandledRejection', this.listener);
    };
  }
  route(realmPromise: PromiseConstructor, report: (reason: unknown) => void): void { this.owners.set(realmPromise, report); }
}
const rejections = new RejectionRouter();

/** A `node:vm` context as a realm. The call's arguments reach the function through a property that is removed at once. */
class VmRealm implements Realm {
  constructor(private readonly context: Context) {}
  call(body: string, parameters: readonly string[], args: readonly unknown[], options: { timeoutMs?: number } = {}): unknown {
    Object.defineProperty(this.context, ARGUMENTS_KEY, { value: args, enumerable: false, configurable: true });
    try {
      return runInContext(`(function (${parameters.join(', ')}) {\n"use strict";\n${body}\n})(...globalThis[${JSON.stringify(ARGUMENTS_KEY)}])`,
        this.context, { ...(options.timeoutMs === undefined ? {} : { timeout: options.timeoutMs }), displayErrors: true });
    } finally { delete (this.context as Record<string, unknown>)[ARGUMENTS_KEY]; }
  }
}

/** The host realm: callable-folder module bodies run in the application's own realm, so their values interoperate with its. */
const hostRealm: Realm = {
  call: (body, parameters, args) => (runInThisContext(`(function (${parameters.join(', ')}) {\n"use strict";\n${body}\n})`,
    { displayErrors: true }) as (...values: unknown[]) => unknown)(...args),
};

export type TypeScriptEnvironmentOptions = { mode?: EnvironmentMode; timeoutMs?: number; observe?: (event: HostEvent) => void;
  workspace?: string; network?: boolean };

/**
 * Node evaluator: a `vm` context in this process, under the shared realm evaluator. Eval sees Node's host globals as
 * any module in the application does; its timers belong to the call and are cleared when it ends.
 */
export class TypeScriptEnvironment extends RealmEnvironment {
  readonly modules?: WorkspaceModules;
  private readonly workspace?: string;
  private readonly packageEvents: HostEvent[] = [];
  /** Cancellers of timers that eval code scheduled and that have not run yet. */
  private readonly timers = new Set<() => void>();
  private releaseRejections?: () => void;

  constructor(options: TypeScriptEnvironmentOptions = {}) {
    const workspace = options.workspace ?? findPackageWorkspace(process.cwd());
    super({ authority: 'shared-node-host', mode: options.mode, timeoutMs: options.timeoutMs, observe: options.observe,
      scopeCapabilities: { allowModules: true, allowNetwork: options.network ?? true },
      fork: () => new TypeScriptEnvironment({ mode: 'fresh', timeoutMs: options.timeoutMs, observe: options.observe,
        workspace, network: options.network }) });
    this.workspace = workspace;
    if (workspace) this.modules = new WorkspaceModules(workspace, event => this.packageEvents.push(event));
  }

  protected override makeRealm(): Realm { return new VmRealm(this.makeContext()); }
  protected override moduleRealm(): Realm { return hostRealm; }
  protected override hostEvents(): HostEvent[] { return this.packageEvents.splice(0); }
  protected override abandon(): void { this.clearTimers(); }
  protected override release(): void { this.clearTimers(); this.releaseRejections?.(); this.releaseRejections = undefined; }

  protected override importer(): EvalImports {
    return {
      validate: specifier => {
        packageNameFromSpecifier(specifier);
        if (!this.modules) throw new Error(`cannot import ${specifier}: no package.json above ${process.cwd()}`);
      },
      load: async specifier => {
        if (!this.modules) throw new Error(`cannot import ${specifier}: no package.json above ${process.cwd()}`);
        return this.modules.load(specifier);
      },
    };
  }

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
    // Timers belong to the call: close() clears those still pending, so nothing eval code scheduled outlives it.
    // setInterval is not offered; repeated work is an iterateOn loop.
    const timers = this.timers, cancellers = new WeakMap<object, () => void>();
    const scheduled = <H extends object>(start: (run: () => void) => H, stop: (handle: never) => void, callback: unknown): H => {
      const run = guarded(callback) as () => void;
      const handle = start(() => { timers.delete(cancel); run(); });
      const cancel = () => stop(handle as never);
      timers.add(cancel);
      cancellers.set(handle, cancel);
      return handle;
    };
    const cancelling = (stop: (handle: never) => void) => (handle: unknown) => {
      const cancel = handle && typeof handle === 'object' ? cancellers.get(handle) : undefined;
      if (cancel) { timers.delete(cancel); cancel(); } else stop(handle as never);
    };
    const context = createContext({ console: undefined, process: evalProcess, Buffer,
      clearTimeout: cancelling(clearTimeout), clearImmediate: cancelling(clearImmediate),
      setTimeout: (callback: unknown, ...rest: unknown[]) => scheduled(run => setTimeout(run, ...rest as [number]), clearTimeout, callback),
      setImmediate: (callback: unknown) => scheduled(run => setImmediate(run), clearImmediate, callback),
      queueMicrotask: (callback: unknown) => queueMicrotask(guarded(callback) as () => void),
      structuredClone, performance, crypto: globalThis.crypto,
      TextEncoder, TextDecoder, URL, URLSearchParams, AbortController, AbortSignal, Blob });
    const realmPromise = runInContext('Promise', context) as PromiseConstructor;
    markRaces(realmPromise);
    this.releaseRejections ??= rejections.acquire();
    rejections.route(realmPromise, reason => this.packageEvents.push({
      operation: 'eval.unhandled-rejection', message: reason instanceof Error ? reason.message : String(reason) }));
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

  declarationOf(name: string): string | undefined {
    return this.modules && this.scopeCapabilities.allowModules ? packageDeclaration(this.modules.workspace, name) : undefined;
  }

  /** Cancel the timers eval code scheduled that have not run: when the call ends, or when an eval fails. */
  private clearTimers(): void {
    if (!this.timers.size) return;
    this.options.observe?.({ operation: 'eval.timers-cleared', count: this.timers.size });
    for (const cancel of this.timers) cancel();
    this.timers.clear();
  }
}
