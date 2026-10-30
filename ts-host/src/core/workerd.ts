/**
 * The workerd platform: natlang in a Cloudflare-style isolate with the `nodejs_compat` flag (AsyncLocalStorage). The
 * isolate forbids code generation from strings, so the host supplies the `Realm` its eval code and compiled modules run
 * in: typically workerd's `UNSAFE_EVAL` binding (`newFunction(body, name, ...parameters)`, see `constructorRealm`) with
 * the host's own confinement of globals. Everything else is the shared evaluator.
 */
import { AsyncLocalStorage } from 'node:async_hooks';
import type { CallStoreLike } from '../calls/recorder.js';
import type { ContextStore, Frame } from '../runtime/context.js';
import { realmPlatform, type Platform } from '../runtime/platform.js';
import type { Realm } from '../native/realm.js';

// One store for the instance: every workerd platform in it shares it, so a call finds its task whichever platform
// value started it.
const storage = new AsyncLocalStorage<Frame | undefined>();
const workerdContextStore: ContextStore = { current: () => storage.getStore(), run: (frame, fn) => storage.run(frame, fn) };

export type WorkerdPlatformOptions = {
  /** Where eval code and compiled modules run. */
  realm: Realm;
  /** The call record store of runtimes that name none; absent: calls are not recorded. */
  callStore?: () => CallStoreLike | undefined;
  /** Platform modules callable-folder code can import. */
  builtinModules?: Platform['builtinModules'];
};

/** A platform for runtimes in a workerd isolate; module state belongs to the value returned, so keep one per realm. */
export function workerdPlatform(options: WorkerdPlatformOptions): Platform {
  return realmPlatform({ name: 'workerd', contextStore: workerdContextStore, realm: options.realm,
    callStore: options.callStore, builtinModules: options.builtinModules });
}
