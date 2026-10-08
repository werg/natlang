/**
 * The replay harness (§3.7): run a case (or the agent) on a recorded call's inputs, with services that answer from the
 * recorded effects instead of acting. A service call the record cannot answer is a divergence: it is reported and the
 * call fails, never silently allowed. Folder inputs are rebuilt in memory from the record.
 */
import { Folder } from '../native/scoped-fs.js';
import { currentFrame, runInFrame } from '../runtime/context.js';
import type { NatlangRuntime } from '../runtime/runtime.js';
import type { CallRecord, EffectRecord, ValueRef } from './types.js';

/** What a replay needs of the store. */
export interface ValueSource { value(ref: ValueRef | null | undefined): unknown; call(callId: string): CallRecord | undefined;
  children(callId: string): { call_id: string }[] }

export class ReplayDivergence extends Error {
  constructor(message: string) { super(message); this.name = 'ReplayDivergence'; }
}

/** One recorded service call, with its values loaded. */
export type RecordedEffect = { service: string; method: string; args: unknown; result?: unknown; async: boolean; error?: string;
  complete: boolean };
/** A service call as made during a replay. */
export type ObservedEffect = { service: string; method: string; args: unknown };

const canonical = (value: unknown): string => JSON.stringify(value, (_key, item) =>
  item && typeof item === 'object' && !Array.isArray(item) ? Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);
/** Equality up to how scalars are written: `999` and `"999"` are the same argument (§3.7, representation only). */
const loose = (value: unknown): string => JSON.stringify(value, (_key, item) =>
  typeof item === 'number' ? String(item) : item && typeof item === 'object' && !Array.isArray(item) ?
    Object.fromEntries(Object.entries(item).sort(([a], [b]) => a.localeCompare(b))) : item);
/** Deep equality of JSON data, ignoring key order. */
export const sameData = (left: unknown, right: unknown): boolean => {
  try { return canonical(left ?? null) === canonical(right ?? null); } catch { return false; }
};
const plain = (value: unknown): unknown => { try { return value === undefined ? undefined : JSON.parse(JSON.stringify(value)); } catch { return value; } };

/** The recorded effects of a call and of every call below it, in the order they started. */
export function subtreeEffects(store: ValueSource, record: CallRecord, depth = 0): RecordedEffect[] {
  const own = record.effects.map(effect => loadEffect(store, effect));
  if (depth > 16) return own;
  const below = store.children(record.call_id).flatMap(child => {
    const childRecord = store.call(child.call_id);
    return childRecord ? subtreeEffects(store, childRecord, depth + 1) : [];
  });
  return [...own, ...below];
}
function loadEffect(store: ValueSource, effect: EffectRecord): RecordedEffect {
  return { service: effect.service, method: effect.method, args: store.value(effect.args), result: store.value(effect.result),
    async: effect.async ?? false, ...(effect.error ? { error: effect.error } : {}),
    complete: effect.args.complete && (effect.error !== undefined || !!effect.result?.complete) };
}

/** Services that answer from recorded effects. Each recorded effect answers at most one call. */
export class ReplayServices {
  readonly services: Record<string, object>;
  readonly observed: ObservedEffect[] = [];
  readonly divergences: string[] = [];
  private readonly used = new Set<number>();
  constructor(readonly effects: readonly RecordedEffect[], names: readonly string[] = []) {
    const all = [...new Set([...names, ...effects.map(effect => effect.service)])];
    this.services = Object.fromEntries(all.map(name => [name, new Proxy(Object.create(null), {
      get: (_target, property) => typeof property !== 'string' || property === 'then' ? undefined :
        (...args: unknown[]) => this.answer(name, property, args),
      has: () => true,
    })]));
  }
  private answer(service: string, method: string, given: unknown[]): unknown {
    const args = plain(given);
    this.observed.push({ service, method, args });
    const open = (effect: RecordedEffect, at: number) => !this.used.has(at) && effect.service === service && effect.method === method && effect.complete;
    let index = this.effects.findIndex((effect, at) => open(effect, at) && sameData(effect.args, args));
    // The same call with its scalars written differently (an id as a number or a string) is answered by the record. It
    // is observed with its own arguments, so the comparison still shows the difference and the judge decides it.
    if (index < 0) index = this.effects.findIndex((effect, at) => {
      try { return open(effect, at) && loose(effect.args) === loose(args); } catch { return false; }
    });
    if (index < 0) {
      const message = `no recorded result for ${service}.${method}(${JSON.stringify(args).slice(1, -1).slice(0, 300)})`;
      this.divergences.push(message);
      throw new ReplayDivergence(`${message}: a replay cannot perform new effects`);
    }
    this.used.add(index);
    const effect = this.effects[index]!;
    if (effect.error !== undefined) {
      const error = new Error(effect.error);
      if (effect.async) return Promise.reject(error);
      throw error;
    }
    const result = plain(effect.result);
    return effect.async ? Promise.resolve(result) : result;
  }
  /** Recorded effects no replayed call used. */
  unused(): RecordedEffect[] { return this.effects.filter((_effect, index) => !this.used.has(index)); }
}

/** The inputs of a recorded call as host values, with a folder rebuilt in memory. `undefined` when one is incomplete. */
export function recordedArguments(store: ValueSource, record: CallRecord): Record<string, unknown> | undefined {
  const args: Record<string, unknown> = {};
  for (const [name, ref] of [...Object.entries(record.captures), ...Object.entries(record.inputs)]) {
    if (!ref.complete) return undefined;
    const value = store.value(ref);
    if (name === 'folder' && value && typeof value === 'object' && '$folder' in (value as object)) {
      const files: Record<string, string> = {};
      for (const [path, file] of Object.entries((value as { $folder: Record<string, ValueRef> }).$folder)) {
        if (!file.complete) return undefined;
        files[path] = store.value(file) as string;
      }
      args.folder = Folder.fromFiles(files).root();
      continue;
    }
    if (value === undefined) return undefined;
    args[name] = value;
  }
  return args;
}

/** What a replayed run did. */
export type ReplayOutcome = { value?: unknown; error?: string; observed: ObservedEffect[]; divergences: string[];
  captureWrites: Record<string, unknown>; files?: { path: string; kind: string; text?: string }[] };

/**
 * Run `run(args)` against recorded effects in a task of `runtime` whose services are the replay services; natural-language
 * calls it makes run live (on the runtime's model) under the same services, with compilations off.
 */
export async function replay(runtime: NatlangRuntime, run: (args: Record<string, unknown>) => unknown, args: Record<string, unknown>,
  effects: readonly RecordedEffect[], options: { serviceNames?: string[]; captureNames?: string[];
    taskOptions?: { auditOf?: string; name?: string } } = {}): Promise<ReplayOutcome> {
  const services = new ReplayServices(effects, options.serviceNames);
  const before = Object.fromEntries((options.captureNames ?? []).map(name => [name, args[name]]));
  const folder = args.folder as { folder?: Folder } | undefined;
  let value: unknown, error: string | undefined;
  try {
    value = await runtime.run(async () => {
      const frame = currentFrame()!;
      return runInFrame({ ...frame, services: services.services }, () => run(args));
    }, { name: 'replay', ...options.taskOptions, services: services.services, specialization: 'off' });
    value = plain(value);
  } catch (caught) { error = caught instanceof Error ? `${caught.name === 'Error' ? '' : `${caught.name}: `}${caught.message}` : String(caught); }
  const captureWrites = Object.fromEntries(Object.entries(before).filter(([name, old]) => !Object.is(args[name], old))
    .map(([name]) => [name, plain(args[name])]));
  const files = folder?.folder ? folder.folder.diffSync().changes.map(change => ({ path: change.path, kind: change.kind,
    ...(change.after ? { text: new TextDecoder().decode(change.after) } : {}) })) : undefined;
  return { ...(error === undefined ? { value } : { error }), observed: services.observed, divergences: services.divergences, captureWrites,
    ...(files ? { files } : {}) };
}
