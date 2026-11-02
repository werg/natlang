/**
 * The replay harness (§3.7): run a case (or the agent) on a recorded call's inputs, with services that answer from the
 * recorded effects instead of acting. A service call the record cannot answer is a divergence: it is reported and the
 * call fails, never silently allowed. Folder inputs are rebuilt in memory from the record.
 */
import { Folder } from '../native/scoped-fs.js';
import { hexDigest } from '../native/hash.js';
import { graphCustodyDigest } from './graph-custody.js';
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
/** Deep equality of JSON data, ignoring key order. */
export const sameData = (left: unknown, right: unknown): boolean => {
  try { return canonical(left ?? null) === canonical(right ?? null); } catch { return false; }
};
const plain = (value: unknown): unknown => { try { return value === undefined ? undefined : JSON.parse(JSON.stringify(value)); } catch { return value; } };

/** Gather the complete recorded subtree and reconstruct only authenticated request chronology. */
export function subtreeEffects(store: ValueSource, record: CallRecord): RecordedEffect[] {
  const pending = [record], records: CallRecord[] = [], seen = new Set<string>();
  while (pending.length) {
    const current = pending.pop()!;
    if (seen.has(current.call_id)) throw new ReplayDivergence('call graph repeats an invocation or contains a cycle');
    seen.add(current.call_id);
    if (current.task_id !== record.task_id) throw new ReplayDivergence('call graph crosses task identities');
    records.push(current);
    const children = store.children(current.call_id);
    const ids = children.map(child => child.call_id);
    if (new Set(ids).size !== ids.length) throw new ReplayDivergence('call store repeats a child identity');
    const custody = current.graph_custody;
    if (custody) {
      if (custody.schema !== 'natlang.call-graph-chronology/1' || custody.task_id !== current.task_id ||
          !Array.isArray(custody.child_call_ids) || new Set(custody.child_call_ids).size !== custody.child_call_ids.length ||
          custody.child_call_ids.length !== ids.length || custody.child_call_ids.some(id => !ids.includes(id)) ||
          custody.sha256 !== graphCustodyDigest(current.task_id, current.call_id, current.parent_call_id, custody.child_call_ids, current.effects))
        throw new ReplayDivergence('captured child inventory differs from stored call graph');
    }
    for (const id of ids) {
      const child = store.call(id);
      if (!child || child.parent_call_id !== current.call_id) throw new ReplayDivergence('recorded child call is missing or has another parent');
      pending.push(child);
    }
  }
  const current = records.some(item => !!item.graph_custody);
  if (current && records.some(item => !item.graph_custody))
    throw new ReplayDivergence('call graph has incomplete chronology custody');
  const effects = records.flatMap(item => item.effects.map(effect => ({ effect, call: item.call_id })));
  if (!current) {
    if (records.length > 1 && effects.length > 0)
      throw new ReplayDivergence('historical cross-call effect chronology is unknown');
    if (effects.length > 0)
      throw new ReplayDivergence('historical cached-effect value custody is unknown');
    return [];
  }
  const orders = new Set<number>();
  for (const {effect} of effects) {
    const order = effect.graph_order;
    if (!Number.isSafeInteger(order) || order! < 0 || orders.has(order!))
      throw new ReplayDivergence('task-wide effect chronology is missing or repeated');
    orders.add(order!);
  }
  return effects.sort((a,b) => a.effect.graph_order! - b.effect.graph_order!).map(item => loadEffect(store,item.effect));
}
/** Read and independently verify the exact content-addressed call-store value. */
function checkedValue(store: ValueSource, ref: ValueRef | null | undefined): unknown {
  if (!ref?.complete) return undefined;
  const value = store.value(ref);
  if (value === undefined) throw new ReplayDivergence('recorded value is missing');
  const text = JSON.stringify(value);
  if (typeof text !== 'string' || hexDigest(text) !== ref.hash || new TextEncoder().encode(text).byteLength !== ref.bytes)
    throw new ReplayDivergence('recorded value differs from its content binding');
  return value;
}
function loadEffect(store: ValueSource, effect: EffectRecord): RecordedEffect {
  return { service: effect.service, method: effect.method, args: checkedValue(store, effect.args), result: checkedValue(store, effect.result),
    async: effect.async ?? false, ...(effect.error ? { error: effect.error } : {}),
    complete: effect.args.complete && (effect.error !== undefined || !!effect.result?.complete) };
}

/** Explicit, source-bound claim that independent exact lookup is valid for these methods. */
export type IndependentReplayContract = { schema: 'natlang.independent-service-replay/1';
  semantics: 'stateless-independent'; methods: string[]; effects_sha256: string; sha256: string };

/** Services that answer from recorded effects. Each recorded effect answers at most one call. */
export class ReplayServices {
  readonly services: Record<string, object>;
  readonly observed: ObservedEffect[] = [];
  readonly divergences: string[] = [];
  private readonly used = new Set<number>();
  constructor(readonly effects: readonly RecordedEffect[], names: readonly string[] = [],
    private readonly options: { mode?: 'ordered-exact' | 'independent-exact'; contract?: IndependentReplayContract } = {}) {
    if (options.mode === 'independent-exact') {
      const contract = options.contract;
      if (!contract || contract.schema !== 'natlang.independent-service-replay/1' ||
          contract.semantics !== 'stateless-independent' || !Array.isArray(contract.methods) ||
          contract.methods.some(method => typeof method !== 'string') ||
          new Set(contract.methods).size !== contract.methods.length ||
          effects.some(effect => !contract.methods.includes(`${effect.service}.${effect.method}`)) ||
          contract.effects_sha256 !== hexDigest(canonical(effects)) ||
          contract.sha256 !== hexDigest(canonical({ schema: contract.schema, semantics: contract.semantics,
            methods: contract.methods, effects_sha256: contract.effects_sha256 })))
        throw new Error('independent replay requires an exact source-bound stateless service contract');
    }
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
    // A source-derived observation sequence may describe stateful services: the
    // same exact request can return different results after prior actions. In
    // this explicit mode the next complete observation is the only legal one.
    const ordered = this.options.mode !== 'independent-exact';
    let index = ordered ? this.used.size :
      this.effects.findIndex((effect, at) => open(effect, at) && sameData(effect.args, args));
    if (ordered) {
      const expected = this.effects[index];
      if (!expected || !open(expected, index) || !sameData(expected.args, args)) index = -1;
    }
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
    const value = checkedValue(store, ref);
    if (name === 'folder' && value && typeof value === 'object' && '$folder' in (value as object)) {
      const files: Record<string, string> = {};
      for (const [path, file] of Object.entries((value as { $folder: Record<string, ValueRef> }).$folder)) {
        if (!file.complete) return undefined;
        files[path] = checkedValue(store, file) as string;
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
    replayMode?: 'ordered-exact' | 'independent-exact'; contract?: IndependentReplayContract;
    taskOptions?: { auditOf?: string; name?: string } } = {}): Promise<ReplayOutcome> {
  const services = new ReplayServices(effects, options.serviceNames, { mode: options.replayMode, contract: options.contract });
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
