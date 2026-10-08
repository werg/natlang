/**
 * Recording host service calls as effects, so a failed eval can report what already happened. Arguments reach the
 * service as plain data of the host realm (eval runs in its own realm), and methods a service lists under
 * ONCE_EFFECTS run once per distinct arguments.
 */
import { hostCopy } from './values.js';

/** One service call, as recorded: requested, then completed or failed. */
export type EffectEvent = { phase: 'requested' | 'completed' | 'failed'; service: string; method: string;
  args?: unknown; result?: unknown; error?: string };

/** Marks services that are already wrapped, so they are never recorded twice. */
const RECORDED = Symbol('natlang.recorded-services');
export const isRecording = (services: object) => Object.hasOwn(services, RECORDED);

/**
 * A service's external effects that must happen once: `{ [ONCE_EFFECTS]: ['send'], send(...) {...} }`. A call with the
 * same arguments as an earlier one on the same service object (an executor re-running eval code, a parent calling a
 * function again) returns the earlier result instead of acting again; a call that failed may run again. The service
 * object's lifetime is the scope: hand a fresh one to each unit of work that may repeat the effect deliberately.
 */
export const ONCE_EFFECTS = Symbol.for('natlang.once-effects');
const onceResults = new WeakMap<object, Map<string, unknown>>();

/** The cache key of a call's arguments, or undefined when they hold anything but plain data. */
function argumentsKey(method: string, args: unknown[]): string | undefined {
  let plain = true;
  const text = JSON.stringify([method, args], (_key, value) => {
    if (typeof value === 'function' || typeof value === 'symbol' || typeof value === 'bigint') plain = false;
    else if (value && typeof value === 'object' && !Array.isArray(value)) {
      const prototype = Object.getPrototypeOf(value);
      if (prototype !== null && Object.getPrototypeOf(prototype) !== null) plain = false;
    }
    return value;
  });
  return plain ? text : undefined;
}

/**
 * Wrap services so each method call is recorded as an effect. Objects are wrapped one level deep
 * per property access; the original receiver is preserved.
 */
export function recordingServices<T extends Record<string, unknown>>(services: T,
  emit: (event: EffectEvent) => void): T {
  const preview = (value: unknown): unknown => {
    try {
      const text = JSON.stringify(value);
      return text === undefined ? String(value) : text.length > 400 ? `${text.slice(0, 400)} … (${text.length} chars)` : JSON.parse(text);
    } catch { return String(value); }
  };
  const wrap = (target: object, label: string): object => new Proxy(target, {
    get(object, property, receiver) {
      const value = Reflect.get(object, property, receiver);
      if (typeof property === 'symbol') return value;
      if (typeof value === 'function') return (...given: unknown[]) => {
        const args = given.map(item => hostCopy(item));
        emit({ phase: 'requested', service: label, method: property, args: preview(args) });
        const once = (object as Record<symbol, unknown>)[ONCE_EFFECTS];
        const key = Array.isArray(once) && once.includes(property) ? argumentsKey(property, args) : undefined;
        let results = key === undefined ? undefined : onceResults.get(object);
        if (key !== undefined && !results) onceResults.set(object, results = new Map());
        if (key !== undefined && results!.has(key)) {
          const earlier = results!.get(key);
          emit({ phase: 'completed', service: label, method: property, result: 'the result of the earlier identical call (this effect happens once)' });
          return earlier;
        }
        try {
          let result = (value as Function).apply(object, args);
          if (key !== undefined) {
            if (result && typeof (result as PromiseLike<unknown>).then === 'function')
              result = Promise.resolve(result).catch(error => { results!.delete(key); throw error; });
            results!.set(key, result);
          }
          if (result && typeof (result as PromiseLike<unknown>).then === 'function')
            return Promise.resolve(result).then(resolved => { emit({ phase: 'completed', service: label, method: property, result: preview(resolved) }); return resolved; },
              error => { emit({ phase: 'failed', service: label, method: property, error: String(error?.message ?? error) }); throw error; });
          emit({ phase: 'completed', service: label, method: property, result: preview(result) });
          return result;
        } catch (error) {
          emit({ phase: 'failed', service: label, method: property, error: String((error as Error)?.message ?? error) });
          throw error;
        }
      };
      return value;
    },
  });
  const wrapped = Object.fromEntries(Object.entries(services).map(([name, service]) =>
    [name, service && typeof service === 'object' ? wrap(service, name) : service]));
  Object.defineProperty(wrapped, RECORDED, { value: true });
  return Object.freeze(wrapped) as T;
}
