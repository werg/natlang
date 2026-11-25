/**
 * Recording host service calls as effects, so a failed eval can report what already happened. Arguments reach the
 * service as plain data of the host realm (eval runs in its own realm), and methods a service lists under
 * ONCE_EFFECTS run once per distinct arguments.
 */
import { hostCopy } from './values.js';

/** One service call, as recorded: requested, then completed or failed. */
export type EffectEvent = { phase: 'requested' | 'completed' | 'failed'; service: string; method: string;
  args?: unknown; result?: unknown; error?: string;
  /** Pairs a request with its completion. */
  seq?: number;
  /** The exact arguments and result, for call records (§3.1). Traces keep only the previews above. */
  exact?: { args?: unknown[]; result?: unknown; async?: boolean } };

/** Marks services that are already wrapped, so they are never recorded twice. */
const RECORDED = Symbol('natlang.recorded-services');
export const isRecording = (services: object) => Object.hasOwn(services, RECORDED);
// Private provenance removes only our observer when a descendant becomes the
// executing call. The retained map is a snapshot of exactly the granted names;
// it never substitutes the task's wider authority or discovers later additions.
const recordingSources = new WeakMap<object, Readonly<Record<string, unknown>>>();
const recordingPorts = new WeakMap<object, object>();
export function unrecordedServices<T extends Record<string, unknown>>(services: T): T {
  const source = recordingSources.get(services);
  if (source) return source as T;
  return Object.freeze(Object.fromEntries(Object.entries(services).map(([name, port]) =>
    [name, port && typeof port === 'object' ? recordingPorts.get(port) ?? port : port]))) as T;
}
const selectedSurfaces = new WeakMap<object, object>();
/** Only recorder-owned proxies have this declared selected-execution surface. */
export function selectedServiceSurface(value: unknown): unknown {
  return value && (typeof value === 'object' || typeof value === 'function') ? selectedSurfaces.get(value) ?? value : value;
}

/**
 * A service's external effects that must happen once: `{ [ONCE_EFFECTS]: ['send'], send(...) {...} }`. A call with the
 * same arguments as an earlier one on the same service object (an executor re-running eval code, a parent calling a
 * function again) returns the earlier result instead of acting again; a call that failed may run again. The service
 * object's lifetime is the scope: hand a fresh one to each unit of work that may repeat the effect deliberately.
 */
export const ONCE_EFFECTS = Symbol.for('natlang.once-effects');
const onceResults = new WeakMap<object, Map<string, unknown>>();

// Result-checking wrappers must run for every invocation, including a cached
// native effect. Only the underlying service object owns the once cache.
const resultCheckedPorts = new WeakSet<object>();
export function resultCheckingService<T extends object>(port: T): T {
  resultCheckedPorts.add(port);
  return port;
}

/** Invoke the supplied native method with its original receiver and once scope. */
export function invokeServiceMethod(object: object, method: string, value: Function, args: unknown[]): unknown {
  const once = resultCheckedPorts.has(object) ? undefined : (object as Record<symbol, unknown>)[ONCE_EFFECTS];
  const key = Array.isArray(once) && once.includes(method) ? argumentsKey(method, args) : undefined;
  let results = key === undefined ? undefined : onceResults.get(object);
  if (key !== undefined && !results) onceResults.set(object, results = new Map());
  if (key !== undefined && results!.has(key)) return results!.get(key);
  let result = value.apply(object, args);
  if (key !== undefined) {
    if (result && typeof (result as PromiseLike<unknown>).then === 'function')
      result = Promise.resolve(result).catch(error => { results!.delete(key); throw error; });
    results!.set(key, result);
  }
  return result;
}

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
  emit: (event: EffectEvent) => unknown,
  authority?: T): T {
  // Checked service-result wrappers belong to this invocation. Its descendants
  // reapply their own checks to the same lexical ports, rather than executing
  // through an ancestor's checker and observer.
  const source = unrecordedServices(authority ?? services);
  if (!authority) services = source;
  const preview = (value: unknown): unknown => {
    try {
      const text = JSON.stringify(value);
      return text === undefined ? String(value) : text.length > 400 ? `${text.slice(0, 400)} … (${text.length} chars)` : JSON.parse(text);
    } catch { return String(value); }
  };
  let sequence = 0;
  const wrap = (target: object, label: string): object => {
    // The proxy stands over an empty shell, not over the service: a host may hand over frozen or hardened objects, and a
    // proxy over a non-configurable, non-writable property must return that property's own value.
    const object = target;
    const proxy = new Proxy(Object.create(Object.getPrototypeOf(target)), {
    has: (_shell, property) => Reflect.has(object, property),
    ownKeys: () => Reflect.ownKeys(object),
    getOwnPropertyDescriptor: (_shell, property) => {
      const descriptor = Reflect.getOwnPropertyDescriptor(object, property);
      return descriptor && { ...descriptor, configurable: true };
    },
    set: (_shell, property, value) => Reflect.set(object, property, value),
    get(_shell, property, receiver) {
      const value = Reflect.get(object, property, object);
      if (typeof property === 'symbol') return value;
      if (typeof value === 'function') {
        const recorded = (...given: unknown[]) => {
        const args = given.map(item => hostCopy(item));
        const seq = sequence++;
        emit({ phase: 'requested', service: label, method: property, args: preview(args), seq, exact: { args } });
        try {
          const result = invokeServiceMethod(object, property, value, args);
          if (result && typeof (result as PromiseLike<unknown>).then === 'function')
            return Promise.resolve(result).then(async resolved => { await emit({ phase: 'completed', service: label, method: property, result: preview(resolved), seq,
              exact: { result: resolved, async: true } }); return resolved; },
              async error => { await emit({ phase: 'failed', service: label, method: property, error: String(error?.message ?? error), seq, exact:{async:true} }); throw error; });
          emit({ phase: 'completed', service: label, method: property, result: preview(result), seq, exact: { result, async: false } });
          return result;
        } catch (error) {
          emit({ phase: 'failed', service: label, method: property, error: String((error as Error)?.message ?? error), seq });
          throw error;
        }
        };
        Object.defineProperty(recorded, Symbol.for('natlang.native-call-kind'), { value:
          Object.getOwnPropertyDescriptor(value, Symbol.for('natlang.native-call-kind'))?.value ??
          (Object.getPrototypeOf(value)?.constructor?.name === 'AsyncFunction' ? 'promise' : 'synchronous') });
        return recorded;
      }
      return value;
    },
    });
    const surface = Object.create(null);
    // The surface names come from the supplied service, not arbitrary property
    // lookup on a guest proxy. Accessors remain lazy broker cell reads.
    for (const key of Object.getOwnPropertyNames(target)) {
      const descriptor = Object.getOwnPropertyDescriptor(target, key)!;
      if ('value' in descriptor && typeof descriptor.value === 'function')
        Object.defineProperty(surface, key, { value: Reflect.get(proxy, key), enumerable: true });
      else Object.defineProperty(surface, key, { get: () => Reflect.get(proxy, key),
        ...(descriptor.set || ('value' in descriptor && descriptor.writable) ? { set: (value: unknown) => Reflect.set(proxy, key, value) } : {}),
        enumerable: true });
    }
    selectedSurfaces.set(proxy, surface);
    const origin = source[label];
    if (origin && typeof origin === 'object') recordingPorts.set(proxy, origin);
    return proxy;
  };
  const wrapped = Object.fromEntries(Object.entries(services).map(([name, service]) =>
    [name, service && typeof service === 'object' ? wrap(service, name) : service]));
  recordingSources.set(wrapped, source);
  Object.defineProperty(wrapped, RECORDED, { value: true });
  return Object.freeze(wrapped) as T;
}
