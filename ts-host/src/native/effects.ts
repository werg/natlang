/** Recording host service calls as effects, so a failed eval can report what already happened. */

/** One service call, as recorded: requested, then completed or failed. */
export type EffectEvent = { phase: 'requested' | 'completed' | 'failed'; service: string; method: string;
  args?: unknown; result?: unknown; error?: string };

/** Marks services that are already wrapped, so they are never recorded twice. */
const RECORDED = Symbol('natlang.recorded-services');
export const isRecording = (services: object) => Object.hasOwn(services, RECORDED);

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
      if (typeof value === 'function') return (...args: unknown[]) => {
        emit({ phase: 'requested', service: label, method: property, args: preview(args) });
        try {
          const result = (value as Function).apply(object, args);
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
