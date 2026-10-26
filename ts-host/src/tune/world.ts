/**
 * Captured worlds for tuning (plans/HIERARCHICAL_LORA_TUNING.md §4.2, §4.4): a root's host service calls are journaled
 * exactly (service, method, arguments, result or error, in completion order), so a later run of the same root, such as
 * an on-policy repair, re-executes in the same world without acting on it again. A replay answers each call from the
 * journal by service, method and arguments (repeated identical calls in their recorded order); a call the journal
 * does not hold throws and marks the replay divergent, so a repair that would need a new external effect is rejected
 * instead of acting on the live world.
 *
 * The journaling proxy is deliberately not a `recordingServices` wrapper: the runtime still wraps the services it is
 * given and records its own effects.
 */
import type { Services } from '../runtime/runtime.js';

export type WorldCall = { readonly service: string; readonly method: string; readonly args: unknown;
  readonly result?: unknown; readonly error?: string; readonly async: boolean };
export type World = { readonly schema: 'natlang.tune-world/v1'; readonly calls: WorldCall[] };

const copy = (value: unknown): unknown => value === undefined ? undefined : JSON.parse(JSON.stringify(value));
const key = (service: string, method: string, args: unknown) => JSON.stringify([service, method, args]);

/** Methods of a service object (own and inherited, excluding Object's), by name. */
function methods(service: object): string[] {
  const names = new Set<string>();
  for (let proto: object | null = service; proto && proto !== Object.prototype; proto = Object.getPrototypeOf(proto))
    for (const name of Object.getOwnPropertyNames(proto))
      if (name !== 'constructor' && typeof (service as Record<string, unknown>)[name] === 'function') names.add(name);
  return [...names];
}

/** Services whose every method call is appended to `world` (arguments and results as JSON data). */
export function journaledServices(services: Services, world: World): Services {
  return Object.fromEntries(Object.entries(services).map(([name, service]) => [name, Object.fromEntries(methods(service).map(method =>
    [method, (...args: unknown[]) => {
      const record = (outcome: { result?: unknown; error?: string }, async: boolean) => {
        world.calls.push({ service: name, method, args: copy(args), ...(outcome.error !== undefined ? { error: outcome.error } : { result: copy(outcome.result) }), async });
      };
      let value: unknown;
      try { value = (service as Record<string, (...a: unknown[]) => unknown>)[method]!(...args); }
      catch (error) { record({ error: (error as Error)?.message ?? String(error) }, false); throw error; }
      if (value && typeof (value as PromiseLike<unknown>).then === 'function')
        return Promise.resolve(value).then(result => { record({ result }, true); return result; },
          error => { record({ error: (error as Error)?.message ?? String(error) }, true); throw error; });
      record({ result: value }, false);
      return value;
    }]))]));
}

export class WorldDivergence extends Error {
  constructor(message: string) { super(`world-divergence: ${message}`); this.name = 'WorldDivergence'; }
}

/**
 * Services with the shape of `shape` (its service and method names; nothing on it is called) that answer from `world`.
 * `divergence()` reports the first call the journal could not answer.
 */
export function replayServices(shape: Services, world: World): { services: Services; divergence(): string | null } {
  const queues = new Map<string, WorldCall[]>();
  for (const call of world.calls) { const k = key(call.service, call.method, call.args); queues.set(k, [...queues.get(k) ?? [], call]); }
  let divergent: string | null = null;
  const services = Object.fromEntries(Object.entries(shape).map(([name, service]) => [name, Object.fromEntries(methods(service).map(method =>
    [method, (...args: unknown[]) => {
      const recorded = queues.get(key(name, method, copy(args)))?.shift();
      if (!recorded) {
        divergent ??= `${name}.${method} was called with arguments the captured world never saw`;
        throw new WorldDivergence(divergent);
      }
      const answer = () => { if (recorded.error !== undefined) throw new Error(recorded.error); return copy(recorded.result); };
      return recorded.async ? Promise.resolve().then(answer) : answer();
    }]))]));
  return { services, divergence: () => divergent };
}
