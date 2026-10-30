import type { EnvironmentMode, HostEvent } from '../native/evaluator.js';
import { RealmEnvironment, type Realm, type RealmEnvironmentOptions } from '../native/realm.js';

/** The page's Function constructor as a realm: eval code runs in the page's global scope. */
export const functionRealm: Realm = {
  call: (body, parameters, args) => new Function(...parameters, `"use strict";\n${body}`)(...args),
};

/**
 * Browser evaluator, under the shared realm evaluator. Compiled code restores the natlang task after each await. Eval
 * sees the page's globals except for timers, which belong to the call and are cleared when it ends.
 */
export class TypeScriptEnvironment extends RealmEnvironment {
  /** Cancellers of timers eval code scheduled that have not run: cleared when the call ends or an eval fails. */
  private readonly timers = new Set<() => void>();
  private readonly timerBindings: Record<string, unknown>;

  /** `bindAwait` is the platform's context slot's: eval awaits restore the natlang task through it. */
  constructor(options: { bindAwait: NonNullable<RealmEnvironmentOptions['bindAwait']>; mode?: EnvironmentMode; observe?: (event: HostEvent) => void }) {
    super({ authority: 'shared-browser-host', mode: options.mode, observe: options.observe, realm: () => functionRealm, bindAwait: options.bindAwait,
      fork: () => new TypeScriptEnvironment({ bindAwait: options.bindAwait, mode: 'fresh', observe: options.observe }) });
    this.timerBindings = this.makeTimerBindings();
  }

  /** setTimeout and clearTimeout for eval code: the timers belong to this environment's call. setInterval is not offered (repeat with iterateOn). */
  private makeTimerBindings(): Record<string, unknown> {
    const timers = this.timers, cancellers = new Map<unknown, () => void>();
    return {
      setTimeout: (callback: unknown, ms?: number, ...args: unknown[]) => {
        // A callback that fails after its eval finished is recorded, not thrown into the page.
        const report = (error: unknown) => this.options.observe?.({ operation: 'eval.callback-error',
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
      setInterval: undefined, clearInterval: undefined,
    };
  }

  protected override extraBindings(): Record<string, unknown> { return this.timerBindings; }
  protected override abandon(): void { this.clearTimers(); }
  protected override release(): void { this.clearTimers(); }
  private clearTimers(): void { for (const cancel of this.timers) cancel(); this.timers.clear(); }
}
