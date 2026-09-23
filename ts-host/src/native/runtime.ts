import YAML from 'yaml';
import { EvalFailure, type EvalEnvironment, type HostEvent } from './evaluator.js';
import { hexDigest } from './hash.js';
import { TypeEnv, fitsType, formatType, parseType, resultType, type Type } from './types.js';
import { MISSING, Reject, buildPending, cloneValue, coerce, dump, dumpState, isPending, loadProgram,
  partType, problems, unboundParts, type LambdaNode, type Pending, type Value } from './values.js';
import { changes, NativeTraceRecorder } from './trace.js';
import { isLazyDict } from './host-tree.js';
import { FileHandle, Folder, FolderHandle, FolderBusyError, type FolderTransaction } from './scoped-fs.js';
import { compileScopeSnippet } from '../scope-compiler.js';
import { parseCrispModule } from './source-core.js';

export type NativeOutcome = { path: string; kind: 'done' | 'quiesced' | 'waiting' | 'replaced'; detail: string; value?: Value };
export type NativeResult = { kind: string; text: string; value?: Value; codes?: string[] };
export type NativeAgent = (session: NativeSession) => Promise<string | void> | string | void;
export type NativePoll = { kind: 'item'; value: unknown } | { kind: 'empty' } | { kind: 'closed' } | { kind: 'failed'; detail: string };
export interface NativeStream { poll(): Promise<NativePoll> | NativePoll }
export type NativeRuntimeOptions = { environment: EvalEnvironment; agent?: NativeAgent;
  capabilities?: Record<string, (args: unknown[]) => unknown>; maxEpisodes?: number;
  maxDepth?: number; maxActions?: number; maxToolCalls?: number;
  runId?: string; stream?: NativeStream; signal?: AbortSignal; timeoutMs?: number;
  sourceRevision?: string; parentCallId?: string;
  sharedEpisodeBudget?: { limit?: number; used: number };
  mapWorkers?: number; parallelModelSafe?: boolean;
  seedPolicy?: { mode: 'compatibility' | 'derived' | 'backend'; root?: number } };

type Ref = { path: string; type?: Type; env: TypeEnv; deny?: string;
  get(): Value; set(value: Value): void; del(): void };
const pending = (value: Value): value is Pending => isPending(value);
const hash = (value: unknown) => hexDigest(JSON.stringify(value));
function fuzzyEditSpan(text: string, remembered: string): string | undefined {
  if (remembered.trim().length < 4) return;
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  const score = (left: string, right: string): number => {
    const a = left.trim(), b = right.trim();
    if (!a.length || !b.length) return 0;
    const width = Math.max(a.length, b.length), row = Array.from({ length: b.length + 1 }, (_, i) => i);
    for (let i = 1; i <= a.length; i++) {
      let previous = row[0]!; row[0] = i;
      for (let j = 1; j <= b.length; j++) {
        const saved = row[j]!;
        row[j] = Math.min(row[j]! + 1, row[j - 1]! + 1, previous + (a[i - 1] === b[j - 1] ? 0 : 1));
        previous = saved;
      }
    }
    return 1 - row[b.length]! / width;
  };
  const candidates: [number, string][] = [];
  for (let width = 1; width <= 3; width++) for (let start = 0; start + width <= lines.length; start++) {
    const span = lines.slice(start, start + width).join('');
    if (span.trim() && text.split(span).length - 1 === 1) candidates.push([score(remembered, span), span]);
  }
  candidates.sort((a, b) => b[0] - a[0]);
  if (!candidates.length || candidates[0]![0] < 0.72 ||
      (candidates.length > 1 && candidates[0]![0] - candidates[1]![0] < 0.08)) return;
  return candidates[0]![1];
}
function oneLine(value: unknown): string {
  if (isLazyDict(value)) return `[${value.label}; lazy read-only Dict]`;
  if (Array.isArray(value)) return `${value.length} items`;
  if (value && typeof value === 'object') return `{ ${Object.entries(value).slice(0, 4)
    .map(([key, item]) => `${key}: ${item && typeof item === 'object' ? '…' : oneLine(item)}`).join(', ')} }`;
  if (typeof value === 'string') {
    const clean = value.replace(/\n+$/, '');
    if (clean.includes('\n')) return `${JSON.stringify(clean.split('\n')[0]!.slice(0, 80))} (${value.split(/\r?\n/).length} lines)`;
    return clean.length <= 80 ? JSON.stringify(clean) : `${JSON.stringify(clean.slice(0, 80))} … (${clean.length} chars)`;
  }
  return JSON.stringify(value);
}
function importInvokePrelude(codebase: Record<string, unknown>): string {
  const synchronous = Object.entries(codebase).filter(([, raw]) =>
    !!raw && typeof raw === 'object' && (raw as Record<string, unknown>).async === false &&
    Object.hasOwn(raw, 'code')).map(([name]) => name);
  return `const __syncImports = new Set(${JSON.stringify(synchronous)});\n` +
    `const __invoke = (name: string, args: unknown[]) => __syncImports.has(name) ? ` +
    `fx.natlang.scope(self.__natlangScopeToken, "call", [name, args]) : ` +
    `({ then: (resolve: (value: unknown) => void, reject: (reason: unknown) => void) => { ` +
    `try { Promise.resolve(fx.natlang.scope(self.__natlangScopeToken, "call", [name, args]))` +
    `.then(resolve, reject); } catch (error) { reject(error); } } });\n`;
}
function pythonJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(', ')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value)
    .map(([key, item]) => `${JSON.stringify(key)}: ${pythonJson(item)}`).join(', ')}}`;
  return JSON.stringify(value);
}
const nodeType = (node: Pending) => ({ lambda: 'Lambda', map: 'MapNode', fold: 'FoldNode',
  iterate: 'IterateNode' })[node.nodeKind];
const DIAGNOSTIC_HINTS: Record<string, string> = {
  'commit-holes': 'Fill what is still missing with write, then finish your turn.',
  'commit-pending': 'A sub-task has not produced its result yet: call run on it, then finish your turn.',
  'not-writable': 'That value cannot be changed at this location.',
  frozen: 'That sub-task is running; its args cannot change now.',
  'type-mismatch': 'Pass the value itself with the type shown as expected, not wrapped in another object: for boolean use `true`, for number use `42.5`, for string use text, and for a record use an object with exactly its fields.',
  'type-does-not-fit-slot': 'That slot needs the type shown as expected.',
  'unknown-field': 'Use one of the fields listed as expected.',
  'unbound-param': 'Give the sub-task its inputs first: copy a value into the path shown, or define it with args_from.',
  'unbound-part': 'The sub-task is missing the part shown: for a Map or Fold, pass the list with over_from or copy it to .../over.',
  'no-such-path': 'Use a path that appears in the state.',
  'old-not-found': 'Copy `old` exactly from the text, including punctuation.',
  'old-not-unique': 'Make `old` longer so that it occurs only once.',
  'no-origin': 'Only a result that a sub-task produced can be retried. Write the value again instead.',
  'too-large': 'Read a part of it with `from` and `to`, or define a Map over it so that each sub-task sees one item.',
  'stuck-dependency': 'An input of this sub-task could not be produced: read its note, fix it, run it again.',
  'unchanged-retry': 'Edit the function or change its inputs first. To deliberately sample another reproducible attempt, evaluate `await retry(local); local`.',
};
function rejected(error: Reject): NativeResult {
  const hint = error.diagnostics.map(diagnostic => DIAGNOSTIC_HINTS[diagnostic.code]).find(Boolean);
  return { kind: 'rejected', text: `rejected\n${error.message}${hint ? `\nhint: ${hint}` : ''}`,
    codes: error.diagnostics.map(diagnostic => diagnostic.code) };
}
function programListing(body: string, marks: Record<number, string>, window = 3): string {
  const lines = body.replace(/^\n+|\n+$/g, '').split('\n');
  const width = String(lines.length).length, output: string[] = [], closed: number[] = [];
  const flush = () => {
    if (!closed.length) return;
    const kinds = new Set(closed.map(number => marks[number]).filter(Boolean));
    const box = kinds.size === 1 && kinds.has('done') ? '[x]' :
      kinds.size === 1 && kinds.has('skipped') ? '[-]' : '[x/-]';
    const first = closed[0]!, last = closed.at(-1)!;
    output.push(`${String(first === last ? first : `${first}-${last}`).padStart(width)} ${box}`);
    closed.length = 0;
  };
  for (const [index, raw] of lines.entries()) {
    const number = index + 1, text = raw.trimEnd(), trimmed = text.trim();
    const markable = !!trimmed && !trimmed.startsWith('#') && !trimmed.startsWith('function ');
    if (Object.hasOwn(marks, number) || (!markable && closed.length)) { closed.push(number); continue; }
    flush();
    const box = markable ? marks[number] === 'done' ? '[x]' : marks[number] === 'skipped' ? '[-]' : '[ ]' : '   ';
    output.push(`${String(number).padStart(width)} ${box} ${text}`.trimEnd());
  }
  flush();
  if (window) {
    const open = output.flatMap((line, index) => line.slice(0, width + 5).includes('[ ]') ? [index] : []);
    if (open.length > window) return [...output.slice(0, open[window]),
      `${' '.repeat(width)} … ${open.length - window} more lines to do`].join('\n');
  }
  return output.join('\n');
}
function pendingProgramLines(body: string, marks: Record<number, string>): number[] {
  return body.replace(/^\n+|\n+$/g, '').split('\n').flatMap((raw, index) => {
    const text = raw.trim();
    const markable = !!text && !text.startsWith('#') && !text.startsWith('function ');
    return markable && !Object.hasOwn(marks, index + 1) ? [index + 1] : [];
  });
}
function jsView(value: Value): unknown {
  if (value === MISSING) return null;
  if (isLazyDict(value) || value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle)
    throw new TypeError('an opaque host handle cannot enter ordinary crisp data; use its injected host API');
  if (pending(value)) return { $pending: formatType(value.type), status: value.status };
  if (Array.isArray(value)) return value.map(jsView);
  if (value !== null && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, jsView(v)]));
  return value;
}
function materializeLazyDictFlat(value: import('./host-tree.js').LazyDict,
  prefix = ''): Record<string, unknown> {
  return Object.fromEntries(value.entries().flatMap(entry => {
    const path = prefix ? `${prefix}/${entry.name}` : entry.name;
    const child = value.child(entry.name);
    return isLazyDict(child) ? Object.entries(materializeLazyDictFlat(child, path)) : [[path, dump(child as Value)]];
  }));
}
function materializeHostDicts(value: Value): Value {
  if (isLazyDict(value)) return materializeLazyDictFlat(value) as Value;
  if (Array.isArray(value)) return value.map(materializeHostDicts);
  if (value && typeof value === 'object' && !(value instanceof Folder) &&
      !(value instanceof FolderHandle) && !(value instanceof FileHandle))
    return Object.fromEntries(Object.entries(value).map(([key, item]) =>
      [key, materializeHostDicts(item as Value)])) as Value;
  return value;
}
const OMIT_HOST_VALUE = Symbol('omit-host-value');
function inlineEvalView(value: Value): unknown {
  if (value === MISSING) return null;
  if (isLazyDict(value) || value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle)
    return OMIT_HOST_VALUE;
  if (pending(value)) return { $pending: formatType(value.type), status: value.status };
  if (Array.isArray(value)) {
    const items = value.map(inlineEvalView);
    if (items.includes(OMIT_HOST_VALUE))
      throw new TypeError('a host-backed Dict inside a list cannot enter crisp eval; use the crisp host API');
    return items;
  }
  if (value !== null && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .flatMap(([key, item]) => { const projected = inlineEvalView(item); return projected === OMIT_HOST_VALUE ? [] : [[key, projected]]; }));
  return value;
}
const itemRef = (container: Record<string, Value> | Value[], key: string | number, type: Type, env: TypeEnv,
  path: string, deny = ''): Ref => ({ path, type, env, deny,
    get: () => Object.hasOwn(container, String(key)) ? (container as Record<string, Value>)[String(key)]! : MISSING,
    set: value => { (container as Record<string, Value>)[String(key)] = value; },
    del: () => { if (Array.isArray(container)) container.splice(Number(key), 1); else delete container[String(key)]; } });

export class NativeRuntime {
  readonly events: HostEvent[] = [];
  readonly trace: NativeTraceRecorder;
  readonly emitted: unknown[] = [];
  readonly origins = new Map<string, LambdaNode>();
  readonly options: { maxEpisodes?: number; maxDepth?: number; maxActions?: number;
    maxToolCalls?: number; runId: string; mapWorkers: number; parallelModelSafe: boolean };
  readonly seedPolicy: { mode: 'compatibility' | 'derived' | 'backend'; root?: number };
  readonly environment: EvalEnvironment;
  readonly agent?: NativeAgent;
  readonly capabilities: Record<string, (args: unknown[]) => unknown>;
  readonly episodeBudget: { limit?: number; used: number };
  private releaseEffect: () => void;
  private stack: string[] = [];
  private invocationPaths: string[] = [];
  private depth = 0;
  private localEpisodesStarted = 0;
  get episodesStarted(): number { return this.episodeBudget.used; }
  currentCallId?: string;
  private root?: { value: Value };
  private lastObserved?: unknown;
  private stream?: NativeStream;
  private streamCurrent: Value | undefined;
  private streamPosition = 0;
  private readonly signal?: AbortSignal;
  private readonly deadline?: number;
  private readonly scopeBridges = new Map<string, (operation: string, args: unknown[]) => unknown>();
  private scopeBridgeSequence = 0;

  constructor(options: NativeRuntimeOptions) {
    this.options = { maxEpisodes: options.maxEpisodes, maxDepth: options.maxDepth,
      maxActions: options.maxActions, maxToolCalls: options.maxToolCalls,
      runId: options.runId ?? 'native-run', mapWorkers: options.mapWorkers ?? 1,
      parallelModelSafe: options.parallelModelSafe ?? false };
    for (const [name, value] of Object.entries({ maxEpisodes: this.options.maxEpisodes,
      maxDepth: this.options.maxDepth, maxActions: this.options.maxActions,
      maxToolCalls: this.options.maxToolCalls }))
      if (value !== undefined && (!Number.isInteger(value) || value < 1))
        throw new RangeError(`${name} must be a positive integer`);
    if (!Number.isInteger(this.options.mapWorkers) || this.options.mapWorkers < 1)
      throw new RangeError('mapWorkers must be positive');
    this.episodeBudget = options.sharedEpisodeBudget ?? { limit: this.options.maxEpisodes, used: 0 };
    this.seedPolicy = options.seedPolicy ?? { mode: 'compatibility' };
    if (this.seedPolicy.mode === 'derived' && !Number.isInteger(this.seedPolicy.root))
      throw new TypeError('derived seed policy requires an integer root');
    this.trace = new NativeTraceRecorder({ run_id: this.options.runId, tool_schema: 'scope-eval-v1',
      ...(options.sourceRevision ? { source_revision: options.sourceRevision } : {}),
      ...(options.parentCallId ? { parent_call_id: options.parentCallId } : {}),
      engines: ['typescript-host'], engine_contracts: { 'typescript-host': {
        environment_mode: options.environment?.mode ?? 'fresh',
        authority: options.environment?.authority ?? 'shared-node-host', native_state_replayable: false } },
      seed_policy: this.seedPolicy, coverage: 'natlang-state-and-observed-host-effects' });
    this.agent = options.agent;
    this.capabilities = options.capabilities ?? {};
    this.environment = options.environment;
    this.releaseEffect = this.environment.bindEffect((cap, fn, args) => this.effect(cap, fn, args));
    this.stream = options.stream;
    this.signal = options.signal;
    if (options.timeoutMs !== undefined) {
      if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 1) throw new RangeError('timeoutMs must be positive');
      this.deadline = Date.now() + options.timeoutMs;
    }
  }

  close(): void { this.releaseEffect(); }

  checkInterruption(): void {
    if (this.signal?.aborted) throw new Error('natlang run aborted; external effects may have occurred');
    if (this.deadline !== undefined && Date.now() >= this.deadline)
      throw new Error('natlang run timed out; external effects may have occurred');
  }

  evalFor(node: LambdaNode, code: string, body: boolean, path: string, scope: Record<string, unknown>) {
    this.checkInterruption();
    const previous = this.acting;
    this.acting = node;
    try {
      const result = this.environment.execute({ code, body, path, effectful: node.effects.length > 0, scope });
      this.recordHostEvents(path, result.events);
      return result;
    } catch (error) {
      if (error instanceof EvalFailure) this.recordHostEvents(path, error.events);
      throw error;
    } finally { this.acting = previous; }
  }

  async evalForAsync(node: LambdaNode, code: string, path: string, scope: Record<string, unknown>, body = true) {
    this.checkInterruption();
    const previous = this.acting;
    const previousCallId = this.currentCallId;
    this.acting = node;
    this.currentCallId = `${path || '$root'}@${node.attempts || 1}`;
    try {
      const result = await this.environment.executeAsync({ code, body, path,
        effectful: node.effects.length > 0, scope });
      this.recordHostEvents(path, result.events);
      this.checkInterruption();
      return result;
    } catch (error) {
      if (error instanceof EvalFailure) this.recordHostEvents(path, error.events);
      throw error;
    } finally { this.acting = previous; this.currentCallId = previousCallId; }
  }

  async evalScopeFor(node: LambdaNode, code: string, path: string, scope: Record<string, unknown>,
    bridge: (operation: string, args: unknown[]) => unknown) {
    const token = `${this.options.runId}:scope:${++this.scopeBridgeSequence}`;
    this.scopeBridges.set(token, bridge);
    try {
      return await this.evalForAsync(node, code, path, { ...scope, __natlangScopeToken: token }, true);
    } finally { this.scopeBridges.delete(token); }
  }

  evalScopeForSync(node: LambdaNode, code: string, path: string, scope: Record<string, unknown>,
    bridge: (operation: string, args: unknown[]) => unknown) {
    const token = `${this.options.runId}:scope:${++this.scopeBridgeSequence}`;
    this.scopeBridges.set(token, bridge);
    const isolated = this.environment.fork();
    const release = isolated.bindEffect((cap, fn, args) => this.effect(cap, fn, args));
    const previous = this.acting;
    this.acting = node;
    try {
      const result = isolated.execute({ code, body: true, path,
        effectful: node.effects.length > 0, scope: { ...scope, __natlangScopeToken: token } });
      this.recordHostEvents(path, result.events);
      return result;
    } catch (error) {
      if (error instanceof EvalFailure) this.recordHostEvents(path, error.events);
      throw error;
    } finally {
      this.acting = previous;
      release();
      isolated.close();
      this.scopeBridges.delete(token);
    }
  }

  private recordHostEvents(path: string, events: HostEvent[]): void {
    this.events.push(...events);
    for (const event of events) if (event.operation !== 'typescript.eval')
      this.trace.emit('host', { path, event });
  }

  private acting?: LambdaNode;
  private effect(cap: string, fn: string, args: unknown[]): unknown {
    this.checkInterruption();
    if (cap === 'natlang' && fn === 'scope') {
      const [token, operation, raw] = args;
      const bridge = this.scopeBridges.get(String(token ?? ''));
      if (!bridge) throw new Error('scope bridge is unavailable');
      if (!Array.isArray(raw)) throw new Error('scope bridge arguments must be positional');
      return bridge(String(operation ?? ''), raw);
    }
    const name = `${cap}.${fn}`, node = this.acting;
    if (!node || !node.effects.includes(name)) throw new Error('effect-undeclared');
    const entry = { seq: node.journal.length + 1, capability: name, function: fn,
      args_preview: pythonJson(args).slice(0, 80), status: 'pending' };
    node.journal.push(entry);
    this.events.push({ operation: 'effect.requested', capability: name, args });
    const callId = this.currentCallId?.startsWith('$root@') ? null : this.currentCallId ?? null;
    this.trace.emit('effect', { phase: 'requested', call_id: callId, capability: name, sequence: entry.seq, args });
    try {
      const value = name === 'out.emit' ? (this.emitted.push(args[0]), null) : this.capabilities[name]?.(args);
      if (value === undefined && name !== 'out.emit') throw new Error('effect-unavailable');
      const complete = (resolved: unknown) => {
        entry.status = 'ok'; this.events.push({ operation: 'effect.completed', capability: name, result: resolved });
        this.trace.emit('effect', { phase: 'completed', call_id: callId, capability: name, sequence: entry.seq, result: resolved });
        return resolved;
      };
      const fail = (error: unknown): never => {
        entry.status = 'error'; this.events.push({ operation: 'effect.failed', capability: name });
        this.trace.emit('effect', { phase: 'failed', call_id: callId, capability: name, sequence: entry.seq });
        throw error;
      };
      if (value && typeof value === 'object' && typeof (value as Promise<unknown>).then === 'function')
        return Promise.resolve(value).then(complete, fail);
      return complete(value);
    } catch (error) {
      entry.status = 'error'; this.events.push({ operation: 'effect.failed', capability: name });
      this.trace.emit('effect', { phase: 'failed', capability: name, sequence: entry.seq });
      throw error;
    }
  }

  async runRoot(root: Pending | Record<string, unknown>): Promise<{ outcome: NativeOutcome; value: Value; events: HostEvent[]; emitted: unknown[] }> {
    this.checkInterruption();
    const source = isPending(root) ? root : loadProgram(root);
    if (!this.root) this.root = { value: source };
    else this.root.value = source;
    const box = this.root;
    const before = this.traceView(box.value);
    this.trace.emit('state', { phase: 'initial', value: before });
    this.lastObserved = before;
    const ref: Ref = { path: '', env: new TypeEnv(), get: () => box.value,
      set: value => { box.value = value; }, del: () => { box.value = MISSING; } };
    const outcome = await this.trigger(ref);
    this.observeState('final', outcome.kind);
    return { outcome, value: box.value, events: this.events, emitted: this.emitted };
  }

  private traceView(value: Value): unknown {
    const state = dumpState(value);
    if (this.stream && pending(value) && value.nodeKind === 'fold' && state && typeof state === 'object') {
      const body = (state as Record<string, Record<string, unknown>>).$fold;
      if (body) body.over = { $stream: { position: this.streamPosition,
        admitted: this.streamCurrent === undefined ? null : 'item', history: 'not-captured' } };
    }
    return state;
  }

  observeState(phase: string, outcome?: string): void {
    if (!this.root) return;
    const value = this.traceView(this.root.value);
    const delta = changes(this.lastObserved, value);
    if (delta.length) this.trace.emit('reduction', { phase, changes: delta });
    this.trace.emit('state', { phase, value, ...(outcome ? { outcome } : {}) });
    this.lastObserved = value;
  }

  private done(ref: Ref, node: Pending, value: Value): NativeOutcome {
    node.status = 'done'; ref.set(value);
    this.trace.emit('node', { path: ref.path, transition: 'done', node_type: nodeType(node) });
    if (node.nodeKind === 'lambda') this.origins.set(ref.path, node);
    return { path: ref.path, kind: 'done', detail: oneLine(dump(value)), value };
  }
  private quiesce(ref: Ref, node: Pending, detail: string): NativeOutcome {
    node.status = 'quiesced'; node.note = detail;
    this.trace.emit('node', { path: ref.path, transition: 'quiesced', node_type: nodeType(node), detail });
    return { path: ref.path, kind: 'quiesced', detail };
  }

  async trigger(ref: Ref): Promise<NativeOutcome> {
    this.checkInterruption();
    const node = ref.get();
    if (!pending(node)) throw new Reject([{ path: ref.path, code: 'no-such-path', expected: 'a pending node' }]);
    if (node.status === 'running') throw new Reject([{ path: ref.path, code: 'frozen' }]);
    const env = ref.env.child(node.types);
    if (node.nodeKind === 'lambda' && node.type.kind === 'lambda') {
      for (const field of node.type.params.fields) {
        const input = node.args[field.name];
        if (input !== undefined && pending(input)) {
          const child = itemRef(node.args, field.name, field.type, env, `${ref.path}/args/${field.name}`);
          const out = await this.trigger(child);
          if (out.kind !== 'done') return this.quiesce(ref, node, `stuck dependency: ${child.path}: ${out.detail}`);
        }
      }
    } else if (node.nodeKind !== 'lambda') {
      const parts = node.nodeKind === 'map' ? ['over'] : node.nodeKind === 'fold' ? ['init'] : ['init'];
      for (const part of parts) {
        const childValue = (node as unknown as Record<string, Value>)[part];
        if (childValue !== undefined && pending(childValue)) {
          const child = itemRef(node as unknown as Record<string, Value>, part, partType(node, part), env, `${ref.path}/${part}`);
          const out = await this.trigger(child);
          if (out.kind !== 'done') return this.quiesce(ref, node, `stuck dependency: ${child.path}: ${out.detail}`);
        }
      }
    }
    const unbound = unboundParts(node, ref.env, ref.path).filter(d =>
      !(this.stream && !ref.path && node.nodeKind === 'fold' && d.path === '/over'));
    if (unbound.length) return this.quiesce(ref, node, `unbound: ${unbound.map(d => d.path).join(', ')}`);
    switch (node.nodeKind) {
      case 'lambda': return node.kind === 'code' ? this.crisp(ref, node, env) : this.episode(ref, node);
      case 'map': return this.map(ref, node, env);
      case 'fold': return this.fold(ref, node, env);
      case 'iterate': return this.iterate(ref, node, env);
    }
  }

  private async crisp(ref: Ref, node: LambdaNode, env: TypeEnv): Promise<NativeOutcome> {
    node.status = 'running';
    node.originalBody ??= node.body;
    this.trace.emit('eval', { phase: 'start', path: ref.path, mode: 'body', engine: 'typescript-host',
      ...(node.engine !== 'typescript-host' ? { declared_engine: node.engine } : {}),
      code: node.body, effectful: node.effects.length > 0 });
    try {
      if (node.type.kind !== 'lambda') throw new Error('invalid lambda type');
      const names = node.type.params.fields.map(field => field.name);
      const helpers = importInvokePrelude(node.codebase) + Object.keys(node.codebase).map(name =>
        `const ${name} = Object.assign((...values: unknown[]) => ` +
        `__invoke(${JSON.stringify(name)}, values), ` +
        `{ __natlangFunction: ${JSON.stringify(name)} });`).join('\n');
      const source = `${names.length ? `let { ${names.join(', ')} } = JSON.parse(JSON.stringify(self.inputs));` : ''}\n` +
        `${helpers}\nreturn await (async () => {\n${node.body}\n})();`;
      const session = new NativeSession(this, node, env, ref.path);
      const result = await this.evalScopeFor(node, source, ref.path,
        { inputs: jsView(materializeHostDicts(node.args as Value)) },
        (operation, values) => session.scopeBridge(operation, values));
      let value: Value;
      try { value = coerce(result.result, node.type.returns, env, ref.path); }
      catch (error) {
        if (error instanceof Reject) {
          this.trace.emit('eval', { phase: 'rejected', path: ref.path, error: error.message });
          return this.quiesce(ref, node, `rejected: ${error.message}`);
        }
        throw error;
      }
      if (pending(value)) {
        node.status = 'done'; ref.set(value);
        this.trace.emit('eval', { phase: 'completed', path: ref.path, value: dump(value) });
        return { path: ref.path, kind: 'replaced', detail: `${formatType(value.type)} unreduced` };
      }
      const missing = problems(value, node.type.returns, env, ref.path);
      if (missing.holes.length) return this.quiesce(ref, node, 'returned value is incomplete');
      this.trace.emit('eval', { phase: 'completed', path: ref.path, value: dump(value) });
      return this.done(ref, node, value);
    } catch (error) {
      this.trace.emit('eval', { phase: 'failed', path: ref.path, error: error instanceof Error ? error.message : String(error) });
      return this.quiesce(ref, node, `code error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  private async episode(ref: Ref, node: LambdaNode): Promise<NativeOutcome> {
    if (this.options.maxDepth !== undefined && this.depth >= this.options.maxDepth)
      return this.quiesce(ref, node, `run budget: episodes nested deeper than ${this.options.maxDepth}`);
    if ((this.options.maxEpisodes !== undefined && this.localEpisodesStarted >= this.options.maxEpisodes) ||
        (this.episodeBudget.limit !== undefined && this.episodeBudget.used >= this.episodeBudget.limit))
      return this.quiesce(ref, node, `run budget: more than ${this.options.maxEpisodes} episodes`);
    const key = hash({ body: node.body, args: dump(node.args as Value), type: formatType(node.type) });
    if (this.stack.includes(key)) return this.quiesce(ref, node, 'identical to a lambda already being reduced above it');
    if (!this.agent) return this.quiesce(ref, node, 'no native model or agent driver supplied');
    node.status = 'running'; node.note = ''; node.attempts++;
    node.originalBody ??= node.body;
    this.localEpisodesStarted++; this.episodeBudget.used++; this.depth++; this.stack.push(key);
    const callId = `${ref.path || '$root'}@${node.attempts}`;
    const previousCallId = this.currentCallId;
    this.currentCallId = callId;
    const parentPath = this.invocationPaths.at(-1) ?? null;
    this.invocationPaths.push(ref.path);
    this.trace.emit('invocation', { phase: 'start', call_id: callId, path: ref.path,
      attempt: node.attempts, parent_path: parentPath });
    const session = new NativeSession(this, node, ref.env, ref.path);
    let note: string | void;
    try {
      note = await this.agent(session);
      this.checkInterruption();
    } catch (error) {
      if (node.projectTransaction?.open) {
        node.projectTransaction.abort();
        this.trace.emit('folder', { call_id: callId, phase: 'discarded', mode: node.reducerMode,
          reason: 'interpreter exception' });
      }
      throw error;
    } finally { this.trace.emit('invocation', { phase: 'end', call_id: callId });
      this.stack.pop(); this.invocationPaths.pop(); this.depth--; this.currentCallId = previousCallId; }
    // Ending the interpreter turn is the completion signal.  A custom agent and
    // the model-backed agent follow the same lifecycle: the runtime validates
    // the typed result and line marks, then commits any reducer transaction.
    if (session.completed || session.finish()) return this.done(ref, node, node.return);
    if (node.projectTransaction?.open) {
      node.projectTransaction.abort();
      this.trace.emit('folder', { call_id: callId, phase: 'discarded', mode: node.reducerMode,
        reason: String(note || 'incomplete invocation') });
    }
    return this.quiesce(ref, node, String(note || 'budget exhausted'));
  }

  private async map(ref: Ref, node: Extract<Pending, { nodeKind: 'map' }>, env: TypeEnv): Promise<NativeOutcome> {
    if (!Array.isArray(node.over) || !pending(node.fn) || node.fn.nodeKind !== 'lambda' || node.type.kind !== 'map')
      return this.quiesce(ref, node, 'invalid Map bindings');
    node.slots ??= node.over.map((item, i) => {
      const step = cloneValue(node.fn as LambdaNode);
      step.status = 'unreduced'; step.attempts = 0;
      step.args[node.itemName] = cloneValue(item);
      if (step.type.kind === 'lambda' && step.type.params.fields.some(f => f.name === 'index')) step.args.index = i;
      return step;
    });
    node.status = 'running';
    if (this.canParallelMap(node)) return this.parallelMap(ref, node, env);
    const stuck: NativeOutcome[] = [];
    for (let i = 0; i < node.slots.length; i++) {
      if (!pending(node.slots[i]!)) continue;
      const child = itemRef(node.slots, i, node.type.b, env, `${ref.path}/${i}`);
      const out = await this.trigger(child);
      if (out.kind !== 'done') stuck.push(out);
    }
    if (!stuck.length) return this.done(ref, node, node.slots);
    return this.quiesce(ref, node, `${node.slots.length - stuck.length} of ${node.slots.length} reduced\n${stuck.map(o => `${o.path}: ${o.kind} "${o.detail}"`).join('\n')}`);
  }

  private canParallelMap(node: Extract<Pending, { nodeKind: 'map' }>): boolean {
    if (this.options.mapWorkers <= 1 || !this.options.parallelModelSafe || this.environment.mode !== 'fresh' ||
        Object.keys(this.environment.host).length || !pending(node.fn) || node.fn.nodeKind !== 'lambda') return false;
    const pure = (definition: unknown): boolean => {
      if (!definition || typeof definition !== 'object') return true;
      const fn = definition as Record<string, unknown>;
      if (Array.isArray(fn.effects) && fn.effects.length) return false;
      return Object.values(fn.codebase as Record<string, unknown> ?? {}).every(pure);
    };
    return !node.fn.effects.length && Object.values(node.fn.codebase).every(pure);
  }

  private async parallelMap(ref: Ref, node: Extract<Pending, { nodeKind: 'map' }>, env: TypeEnv): Promise<NativeOutcome> {
    if (!node.slots || node.type.kind !== 'map') throw new Error('Map slots unavailable');
    const mapType = node.type;
    const indices = node.slots.flatMap((value, index) => pending(value) ? [index] : []);
    const results = new Map<number, { outcome: NativeOutcome; events: { kind: string; [key: string]: unknown }[];
      origins: Map<string, LambdaNode> }>();
    const abort = new AbortController();
    const signal = this.signal ? AbortSignal.any([this.signal, abort.signal]) : abort.signal;
    let cursor = 0;
    const worker = async () => {
      while (cursor < indices.length) {
        this.checkInterruption();
        const index = indices[cursor++]!;
        const child = new NativeRuntime({ environment: this.environment.fork(),
          agent: this.agent, maxEpisodes: this.options.maxEpisodes, maxDepth: this.options.maxDepth,
          maxActions: this.options.maxActions, maxToolCalls: this.options.maxToolCalls,
          seedPolicy: this.seedPolicy, sharedEpisodeBudget: this.episodeBudget,
          runId: this.options.runId, capabilities: {}, signal,
          timeoutMs: this.deadline === undefined ? undefined : Math.max(1, this.deadline - Date.now()) });
        child.depth = this.depth; child.stack = [...this.stack];
        try {
          const slot = itemRef(node.slots!, index, mapType.b, env, `${ref.path}/${index}`);
          const outcome = await child.trigger(slot);
          results.set(index, { outcome, events: child.trace.events.filter(event => event.kind !== 'manifest'),
            origins: child.origins });
        } catch (error) { abort.abort(); throw error; }
        finally { child.close(); child.environment.close(); }
      }
    };
    await Promise.all(Array.from({ length: Math.min(indices.length, this.options.mapWorkers) }, () => worker()));
    for (const index of indices) {
      const result = results.get(index)!;
      for (const event of result.events) {
        const { kind, version: _version, seq: _seq, ...data } = event;
        this.trace.emit(kind, data);
      }
      for (const [path, origin] of result.origins) this.origins.set(path, origin);
      this.trace.emit('map_slot', { path: `${ref.path}/${index}`, slot: index,
        outcome: result.outcome.kind, value: dump(node.slots[index]!) });
    }
    const stuck = indices.map(index => results.get(index)!.outcome).filter(outcome => outcome.kind !== 'done');
    if (!stuck.length) return this.done(ref, node, node.slots);
    return this.quiesce(ref, node, `${node.slots.length - stuck.length} of ${node.slots.length} reduced\n${stuck.map(o => `${o.path}: ${o.kind} "${o.detail}"`).join('\n')}`);
  }

  private async fold(ref: Ref, node: Extract<Pending, { nodeKind: 'fold' }>, env: TypeEnv): Promise<NativeOutcome> {
    if (!pending(node.step) || node.step.nodeKind !== 'lambda' || node.type.kind !== 'fold')
      return this.quiesce(ref, node, 'invalid Fold bindings');
    node.status = 'running';
    if (node.acc === MISSING) { node.acc = cloneValue(node.init); node.at = 0; }
    while (true) {
      let item: Value;
      if (this.stream && !ref.path) {
        if (this.streamCurrent === undefined) {
          const polled = await this.stream.poll();
          if (polled.kind === 'empty') {
            node.status = 'waiting'; node.note = 'waiting for stream input';
            this.trace.emit('stream', { path: ref.path, phase: 'waiting', position: this.streamPosition });
            return { path: ref.path, kind: 'waiting', detail: node.note };
          }
          if (polled.kind === 'closed') {
            this.trace.emit('stream', { path: ref.path, phase: 'closed', position: this.streamPosition });
            return this.done(ref, node, node.acc);
          }
          if (polled.kind === 'failed') {
            this.trace.emit('stream', { path: ref.path, phase: 'failed', position: this.streamPosition,
              detail: polled.detail });
            return this.quiesce(ref, node, `stream failed: ${polled.detail}`);
          }
          this.streamCurrent = coerce(polled.value, node.type.a, env, `${ref.path}/over/${node.at}`);
          this.trace.emit('stream', { path: ref.path, phase: 'admitted', position: this.streamPosition,
            value: dump(this.streamCurrent) });
        }
        item = this.streamCurrent;
      } else {
        if (!Array.isArray(node.over)) return this.quiesce(ref, node, 'invalid Fold input');
        if (node.at >= node.over.length) return this.done(ref, node, node.acc);
        item = node.over[node.at]!;
      }
      if (node.current === null) {
        const step = cloneValue(node.step as LambdaNode);
        step.args[node.accName] = cloneValue(node.acc); step.args[node.itemName] = cloneValue(item);
        node.current = step;
      }
      const child = itemRef(node as unknown as Record<string, Value>, 'current', node.type.s, env, `${ref.path}/step/${node.at}`);
      const out = await this.trigger(child);
      if (out.kind !== 'done') return this.quiesce(ref, node, `step ${node.at} ${out.kind}: ${out.detail}`);
      node.acc = node.current; node.current = null; node.at++;
      if (this.stream && !ref.path) {
        this.streamCurrent = undefined; this.streamPosition++;
        this.trace.emit('stream', { path: ref.path, phase: 'consumed', position: this.streamPosition });
      }
    }
  }

  private async iterate(ref: Ref, node: Extract<Pending, { nodeKind: 'iterate' }>, env: TypeEnv): Promise<NativeOutcome> {
    if (!pending(node.step) || node.step.nodeKind !== 'lambda' || !pending(node.check) || node.check.nodeKind !== 'lambda' ||
        node.type.kind !== 'iterate' || typeof node.max !== 'number') return this.quiesce(ref, node, 'invalid Iterate bindings');
    node.status = 'running';
    if (node.state === MISSING) { node.state = cloneValue(node.init); node.iteration = 0;
      node.recent = [cloneValue(node.init)]; node.seenHashes = [hash(dump(node.init))]; }
    while (node.iteration < node.max) {
      if (node.current === null) { const step = cloneValue(node.step as LambdaNode); step.args[node.stateName] = cloneValue(node.state); node.current = step; }
      const child = itemRef(node as unknown as Record<string, Value>, 'current', node.type.s, env, `${ref.path}/step/${node.iteration}`);
      const out = await this.trigger(child);
      if (out.kind !== 'done') return this.quiesce(ref, node, `step ${node.iteration} ${out.kind}: ${out.detail}`);
      node.state = node.current; node.current = null; node.iteration++;
      const digest = hash(dump(node.state));
      if (node.seenHashes.includes(digest)) return this.quiesce(ref, node, 'degenerate: the state repeated an earlier state');
      node.seenHashes.push(digest); node.recent = [...node.recent, cloneValue(node.state as Value) as Value].slice(-3);
      const check = cloneValue(node.check as LambdaNode);
      if (node.checkName) check.args[node.checkName] = cloneValue(node.state);
      else { check.args.recent = cloneValue(node.recent as Value); check.args.iteration = node.iteration; }
      const box: Record<string, Value> = { value: check };
      const chk = itemRef(box, 'value', check.type.kind === 'lambda' ? check.type.returns : parseType('boolean'), env,
        `${ref.path}/check/${node.iteration}`);
      const checked = await this.trigger(chk);
      if (checked.kind !== 'done') return this.quiesce(ref, node, `check ${checked.kind}: ${checked.detail}`);
      const verdict = node.checkName ? { verdict: box.value === true ? 'done' : 'continue', reason: '' } : box.value as Record<string, Value>;
      if (verdict.verdict === 'done') return this.done(ref, node, node.state);
      if (verdict.verdict === 'degenerate') return this.quiesce(ref, node, `degenerate: ${String(verdict.reason)}`);
    }
    return this.quiesce(ref, node, `max iterations reached (${node.max})`);
  }
}

export class NativeSession {
  completed = false;
  actions = 0;
  toolCalls = 0;
  surfaceName = 'scope-eval-v1';
  readonly env: TypeEnv;
  private scopeCallSequence = 0;
  private readonly scopeLocalMutability = new Map<string, boolean>();
  private readonly explainedNaturalFunctions = new Set<string>();
  constructor(readonly runtime: NativeRuntime, readonly lam: LambdaNode, readonly outerEnv: TypeEnv,
    readonly path = '') {
    this.env = outerEnv.child(lam.types);
  }
  finish(): boolean {
    if (this.lam.return === MISSING || this.lam.type.kind !== 'lambda') return false;
    if (pendingProgramLines(this.lam.originalBody ?? this.lam.body, this.lam.marks).length) return false;
    const p = problems(this.lam.return, this.lam.type.returns, this.env, 'return');
    if (p.holes.length || p.pending.length) return false;
    const tx = this.lam.projectTransaction;
    if (tx?.open) {
      const delta = tx.folder.diffSync();
      const selected = this.lam.reducerMode === 'apply' ?
        tx.commitSync(this.lam.commitInclude, this.lam.commitExclude) :
        (tx.abort(), delta);
      this.runtime.trace.emit('folder', { call_id: this.runtime.currentCallId ?? null,
        phase: this.lam.reducerMode === 'apply' ? 'installed' : 'discarded', mode: this.lam.reducerMode,
        changes: selected.changes.map(change => ({ path: change.path, kind: change.kind })) });
    }
    this.completed = true; this.lam.body = ''; return true;
  }
  private summary(): string {
    if (this.lam.type.kind !== 'lambda') return 'problems: 0 blocking · 0 holes';
    if (this.lam.return === MISSING) return 'problems: 0 blocking · 0 holes';
    const issues = problems(this.lam.return, this.lam.type.returns, this.env, 'return');
    return `problems: 0 blocking · ${issues.holes.length} holes`;
  }
  private progress(): string {
    const size = (value: Value): string => {
      if (value === MISSING) return 'not finished';
      if (pending(value)) return 'not finished';
      if (Array.isArray(value)) return `${value.length} items`;
      if (typeof value === 'string') return `${value.trim().split(/\s+/).filter(Boolean).length} words`;
      if (value && typeof value === 'object') return 'record';
      return JSON.stringify(value).slice(0, 40);
    };
    const locals = Object.keys(this.lam.letTypes).filter(name => Object.hasOwn(this.lam.let, name))
      .map(name => `${name} (${size(this.lam.let[name]!)})`).join(', ') || 'none';
    let result = 'written';
    if (this.lam.return === MISSING) result = 'not written yet';
    else if (this.lam.type.kind === 'lambda') {
      const type = this.env.resolve(this.lam.type.returns);
      if (type.kind === 'record' && this.lam.return && typeof this.lam.return === 'object' &&
          !Array.isArray(this.lam.return) && !pending(this.lam.return)) {
        const value = this.lam.return as Record<string, Value>;
        const missing = type.fields.filter(field => !field.optional && !Object.hasOwn(value, field.name))
          .map(field => field.name);
        result = missing.length ? `has ${Object.keys(value).join(', ') || 'nothing'}; still missing ${missing.join(', ')}` : 'complete';
      } else if (pending(this.lam.return)) result = 'not finished';
    }
    return `locals: ${locals}\nreturn: ${result}`;
  }
  private record(name: string, args: Record<string, unknown>, result: NativeResult): NativeResult {
    if (['write', 'edit'].includes(name) && result.kind === 'ok') result.text = `ok   ${this.summary()}`;
    if ((name === 'write' || name === 'call') && args.done !== undefined && ['ok', 'done'].includes(result.kind)) {
      const mark = this.doneRange(args.done);
      const marked = this.applyNow('mark_done', mark);
      result.text = result.text.trimEnd() + '\n' + marked.text.slice(marked.text.indexOf('\n') + 1);
    }
    if (['write', 'edit', 'call'].includes(name) && ['ok', 'done', 'quiesced'].includes(result.kind))
      result.text = result.text.trimEnd() + '\n' + this.progress();
    this.runtime.trace.emit('action', { call_id: this.runtime.currentCallId ?? null,
      surface: this.surfaceName, name, arguments: args,
      outcome: result.kind, result_text: result.text, diagnostics: result.codes ?? [] });
    this.runtime.observeState('after-action');
    return result;
  }
  private doneRange(raw: unknown): { start: number; end: number } {
    const values = Array.isArray(raw) ? raw : [raw];
    if (values.length < 1 || values.length > 2 || values.some(value => !Number.isInteger(value)))
      throw new Reject([{ path: 'done', code: 'bad-range', expected: 'a line number, or [first, last]' }]);
    const start = Number(values[0]), end = Number(values.at(-1));
    this.validateMark(start, end);
    return { start, end };
  }
  private validateMark(start: unknown, end: unknown): void {
    const lines = (this.lam.originalBody ?? this.lam.body).replace(/^\n+|\n+$/g, '').split('\n');
    if (!Number.isInteger(start) || !Number.isInteger(end) || Number(start) < 1 ||
        Number(end) < Number(start) || Number(end) > lines.length)
      throw new Reject([{ path: 'start', code: 'bad-range',
        expected: `line numbers between 1 and ${lines.length}, start <= end`, got: `${start}..${end}` }]);
  }
  private actionLimitReached(): boolean {
    return (this.runtime.options.maxActions !== undefined && this.actions >= this.runtime.options.maxActions) ||
      (this.runtime.options.maxToolCalls !== undefined && this.toolCalls >= this.runtime.options.maxToolCalls);
  }
  apply(name: string, args: Record<string, unknown>): NativeResult {
    this.runtime.checkInterruption();
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    if (!['read_value', 'mark_lines', 'commit', 'report_blocker', 'report_error'].includes(name))
      return this.record(name, args, rejected(new Reject([{ path: name, code: 'bad-action',
        expected: 'a current scope-eval tool' }])));
    if (this.actionLimitReached())
      return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.toolCalls++;
    if (name !== 'mark_done' && name !== 'mark_lines') this.actions++;
    if (name !== 'mark_done' && name !== 'mark_lines') this.lam.steps++;
    if (name === 'write' && args.done !== undefined) try { this.doneRange(args.done); }
    catch (error) { if (error instanceof Reject) return this.record(name, args, rejected(error)); throw error; }
    return this.record(name, args, this.applyNow(name, args));
  }
  private applyNow(name: string, args: Record<string, unknown>): NativeResult {
    try {
      if (name === 'mark_lines') return this.applyNow('mark_done', args);
      if (name === 'read_value') {
        const path = this.scopePath(String(args.expression ?? ''));
        if (args.start !== undefined || args.end !== undefined) {
          const value = this.resolve(path).get();
          // String ranges are character slices, matching both JS slicing and the
          // ``(N chars)`` preview.  Lists continue to use item ranges.
          const sequence = Array.isArray(value) || typeof value === 'string' ? value : null;
          if (!sequence) throw new Reject([{ path, code: 'bad-range', expected: 'a list or text value' }]);
          let start = Number(args.start ?? 0), end = Number(args.end ?? sequence.length);
          if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end < 0)
            throw new Reject([{ path, code: 'bad-range', expected: 'non-negative JavaScript slice offsets' }]);
          start = Math.min(start, sequence.length); end = Math.min(end, sequence.length);
          if (end < start) end = start;
          const slice = sequence.slice(start, end);
          if (typeof value === 'string') {
            return { kind: 'ok', text: slice as string, value: slice as string };
          }
          return { kind: 'ok', text: (slice as Value[]).map((item, index) =>
            `${start + index}: ${typeof item === 'string' ? item : JSON.stringify(dump(item))}`).join('\n'),
            value: slice as Value[] };
        }
        return this.applyNow('read', { path });
      }
      if (name === 'commit') {
        if (this.lam.subtype !== 'directory-reducer' || !this.lam.projectTransaction)
          throw new Reject([{ path: 'commit', code: 'bad-action', expected: 'a running directory reducer' }]);
        const include = args.include, exclude = args.exclude;
        for (const [key, selectors] of [['include', include], ['exclude', exclude]] as const)
          if (selectors !== undefined && (!Array.isArray(selectors) || selectors.some(item =>
            typeof item !== 'string' || !item || item.startsWith('/') || item.startsWith('project/') || item.startsWith('codebase/'))))
            throw new Reject([{ path: key, code: 'bad-action', expected: 'relative project glob patterns' }]);
        if (this.lam.type.kind !== 'lambda')
          throw new Reject([{ path: 'commit', code: 'bad-action', expected: 'a typed directory reducer result' }]);
        const result = this.applyNow('write', { path: 'return', type: formatType(this.lam.type.returns), value: args.value });
        this.lam.commitInclude = include === undefined ? undefined : [...include as string[]];
        this.lam.commitExclude = exclude === undefined ? undefined : [...exclude as string[]];
        return result;
      }
      if (name === 'report_blocker' || name === 'report_error') {
        const message = String(args[name === 'report_blocker' ? 'missing' : 'message'] ?? '').trim();
        if (message.length < 8) throw new Reject([{ path: name === 'report_blocker' ? 'missing' : 'message',
          code: 'bad-action', expected: name === 'report_blocker' ?
            'a sentence saying what is missing' : 'a sentence explaining the error' }]);
        return { kind: 'blocked', text: `${name === 'report_blocker' ? 'blocked' : 'error'}: ${message}` };
      }
      if (name === 'mark_done') {
        const start = args.start, end = args.end ?? args.start;
        this.validateMark(start, end);
        const lines = (this.lam.originalBody ?? this.lam.body).replace(/^\n+|\n+$/g, '').split('\n');
        for (let i = Number(start); i <= Number(end); i++) {
          const text = lines[i - 1]!.trim();
          if (text && !text.startsWith('#') && !text.startsWith('function '))
            this.lam.marks[i] = args.skipped === true ? 'skipped' : 'done';
        }
        return { kind: 'ok', text: `ok\n${programListing(this.lam.originalBody ?? this.lam.body, this.lam.marks)}` };
      }
      if (name === 'write') {
        const path = String(args.path ?? '');
        if (path.startsWith('args/')) throw new Reject([{ path, code: 'not-writable' }]);
        const functionCopy = /^Function<([A-Za-z_][A-Za-z0-9_]*)>$/.exec(String(args.type ?? ''));
        if (functionCopy) {
          const local = /^let\/([A-Za-z_][A-Za-z0-9_]*)$/.exec(path)?.[1];
          const def = this.lam.codebase[functionCopy[1]!] as Record<string, unknown> | undefined;
          if (!local || !def) throw new Reject([{ path, code: !local ? 'not-writable' : 'no-such-function' }]);
          const params = Object.entries(def.args as Record<string, string> ?? {}).map(([key, type]) => `${key}: ${type}`).join(', ');
          const type = parseType(`(${params}) => ${String(def.returns)}`);
          const kind = Object.hasOwn(def, 'code') ? 'code' : 'instructions';
          const copied = buildPending({ $lambda: { type: formatType(type), [kind]: def[kind],
            engine: def.engine, types: def.types ?? {}, effects: def.effects ?? [],
            codebase: def.codebase ?? {}, function: functionCopy[1] } }, this.env);
          if (copied.nodeKind !== 'lambda') throw new Error('internal function copy error');
          this.checkEffects(copied, path);
          this.lam.letTypes[local] = type;
          this.lam.let[local] = copied;
          this.lam.fnCopies[local] = def;
          return { kind: 'ok', text: `ok   ${path} is a copy of ${functionCopy[1]}` };
        }
        const typeText = String(args.type ?? '').trim();
        const stated = typeText ? parseType(typeText) : this.resolve(path, true).type;
        if (!stated) throw new Reject([{ path, code: 'type-mismatch', expected: 'a type for the new local' }]);
        if (args.source === undefined &&
            (['lambda', 'map', 'fold', 'iterate'].includes(stated.kind) ||
              (args.value && typeof args.value === 'object' && !Array.isArray(args.value) &&
                Object.keys(args.value as object).some(key => ['$lambda', '$map', '$fold', '$iterate'].includes(key)))))
          throw new Reject([{ path: 'type', code: 'anonymous-lambda', expected: 'call with a checked function' }]);
        const local = /^let\/([A-Za-z_][A-Za-z0-9_]*)$/.exec(path)?.[1];
        const created = !!local && !Object.hasOwn(this.lam.letTypes, local);
        if (created) this.lam.letTypes[local!] = stated;
        try {
          const ref = this.resolve(path, true);
          if (ref.deny) throw new Reject([{ path, code: ref.deny }]);
          if (!ref.type || !fitsType(stated, ref.type, ref.env))
            throw new Reject([{ path, code: 'type-does-not-fit-slot', expected: ref.type ? formatType(ref.type) : '' }]);
          const raw = args.source !== undefined ? this.resolve(String(args.source)).get() : args.value;
          if (raw === undefined || raw === MISSING) throw new Reject([{ path, code: 'bad-action', expected: 'a value or source' }]);
          let value: Value;
          try { value = coerce(raw, stated, ref.env, path); }
          catch (first) {
            if (typeof raw !== 'string') throw first;
            try { value = coerce(JSON.parse(raw), stated, ref.env, path); }
            catch { throw first; }
          }
          this.checkEffects(value, path);
          ref.set(value);
          return { kind: 'ok', text: `ok   ${path}`, value };
        } catch (error) {
          if (created && !Object.hasOwn(this.lam.let, local!)) delete this.lam.letTypes[local!];
          throw error;
        }
      }
      if (name === 'read') {
        const path = String(args.path ?? '');
        const meta = /^(.+)@(status|note|effects|problems|origin|dist)$/.exec(path);
        if (meta) {
          const ref = this.resolve(meta[1]!); const value = ref.get();
          let item: string;
          if (meta[2] === 'status') item = pending(value) ? value.status : 'done';
          else if (meta[2] === 'note') item = pending(value) ? value.note : '';
          else if (meta[2] === 'effects') {
            const lam = pending(value) && value.nodeKind === 'lambda' ? value : this.lam;
            item = lam.journal.length ? YAML.stringify(lam.journal) : '(none)';
          } else if (meta[2] === 'problems') {
            if (pending(value) || !ref.type) item = '(not a value)';
            else { const issues = problems(value, ref.type, ref.env, ref.path);
              item = [...issues.holes.map(d => `${d.path}: ${d.code}`),
                ...issues.pending.map(at => `${at}: pending`)].join('\n') || 'no problems'; }
          } else if (meta[2] === 'origin') {
            const origin = this.runtime.origins.get(ref.path);
            item = origin ? YAML.stringify(dump(origin)) : '(no lambda origin)';
          } else item = '(not recorded)';
          return { kind: 'ok', text: item, value: item };
        }
        const selected = args.start !== undefined || args.end !== undefined ?
          `${path}[${args.start ?? args.end}..${args.end ?? args.start}]` : path;
        const ranged = this.ranged(selected);
        if (ranged) {
          const first = Number(/\[(\d+)\.\./.exec(selected)?.[1] ?? 0);
          const text = typeof ranged.value === 'string' ? ranged.value :
            (ranged.value as Value[]).map((item, index) => `${first + index}: ${typeof item === 'string' ? item : JSON.stringify(dump(item))}`).join('\n');
          return { kind: 'ok', text, value: ranged.value };
        }
        if (path === 'codebase') return { kind: 'ok', text: Object.keys(this.lam.codebase).join('\n') || '(no functions)' };
        if (path.startsWith('codebase/')) {
          const key = path.slice(9), fn = this.lam.codebase[key] as Record<string, unknown> | undefined;
          if (!fn) throw new Reject([{ path, code: 'no-such-path' }]);
          return { kind: 'ok', text: `${key}(${Object.entries(fn.args as Record<string, string> ?? {}).map(([n,t]) => `${n}: ${t}`).join(', ')}) -> ${fn.returns}\n${fn.description ?? ''}\n\n${fn.code ?? fn.instructions ?? ''}` };
        }
        const ref = this.resolve(path);
        const value = ref.get();
        if (isLazyDict(value)) return { kind: 'ok', text: value.entries().map(entry =>
          `${entry.kind === 'branch' ? 'dir ' : 'leaf'}  ${entry.name}`).join('\n') || '(empty)' };
        return { kind: 'ok', text: value === MISSING ? `${ref.path}: not supplied (missing value; not empty text)` :
          typeof value === 'string' ? value : JSON.stringify(dump(value), null, 1), value };
      }
      if (name === 'edit') {
        const ref = this.resolve(String(args.path ?? ''));
        if (ref.deny) throw new Reject([{ path: ref.path, code: ref.deny }]);
        const value = ref.get(), remembered = String(args.old ?? ''), replacement = String(args.new ?? '');
        if (typeof value !== 'string') throw new Reject([{ path: ref.path, code: 'type-mismatch', expected: 'a text' }]);
        const old = value.split(remembered).length - 1 === 1 ? remembered :
          args.fuzzy === true ? fuzzyEditSpan(value, remembered) ?? remembered : remembered;
        const occurrences = old ? value.split(old).length - 1 : 0;
        if (occurrences !== 1) throw new Reject([{ path: ref.path,
          code: occurrences ? 'old-not-unique' : 'old-not-found',
          expected: '`old` copied exactly from the text, occurring once', got: `${occurrences} occurrences` }]);
        ref.set(value.replace(old, replacement));
        return { kind: 'ok', text: `ok   ${ref.path}` };
      }
      if (name === 'delete') {
        const ref = this.resolve(String(args.path ?? ''));
        if (ref.deny) throw new Reject([{ path: ref.path, code: ref.deny }]);
        ref.del();
        return { kind: 'ok', text: `ok   ${ref.path}` };
      }
      if (name === 'copy') {
        const from = String(args.from ?? ''), to = String(args.to ?? '');
        const ranged = this.ranged(from);
        const src = ranged?.ref ?? this.resolve(from), dst = this.resolve(to, true), original = ranged?.value ?? src.get();
        const sourceType = ranged?.type ?? src.type;
        if (original === MISSING || !sourceType) throw new Reject([{ path: from, code: 'no-such-path' }]);
        if (dst.deny) throw new Reject([{ path: to, code: dst.deny }]);
        if (!dst.type || !fitsType(sourceType, dst.type, dst.env))
          throw new Reject([{ path: to, code: 'type-does-not-fit-slot', expected: dst.type ? formatType(dst.type) : '' }]);
        const value = cloneValue(original);
        if (pending(value)) value.status = 'unreduced';
        this.checkEffects(value, to);
        dst.set(value);
        return { kind: 'ok', text: `ok   ${to}`, value };
      }
      if (name === 'retry') {
        const path = String(args.path ?? '');
        const ref = this.resolve(path), value = ref.get();
        if (ref.deny) throw new Reject([{ path, code: ref.deny }]);
        if (value === MISSING || pending(value) || !ref.type)
          throw new Reject([{ path, code: 'no-such-path', expected: 'a completed value' }]);
        const origin = this.runtime.origins.get(ref.path);
        if (!origin || origin.kind !== 'instructions')
          throw new Reject([{ path, code: 'no-origin', expected: 'a value that a natural-language lambda produced' }]);
        const retry = cloneValue(origin);
        const feedback = String(args.feedback ?? '').replace(/^\n+|\n+$/g, '');
        retry.body = (feedback || origin.originalBody || '').replace(/\n+$/, '') + '\n';
        retry.return = cloneValue(value);
        retry.status = 'unreduced'; retry.note = '';
        ref.set(retry);
        return { kind: 'ok', text: `ok   ${path}: draft return prefilled`, value: retry };
      }
      throw new Reject([{ path: name, code: 'bad-action' }]);
    } catch (error) {
      if (error instanceof Reject) return rejected(error);
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes('NATLANG:effect-undeclared')) return { kind: 'rejected', text: message, codes: ['effect-undeclared'] };
      return { kind: 'error', text: message };
    }
  }

  private ranged(path: string): { ref: Ref; type: Type; value: Value } | undefined {
    const match = /^(.*?)\[(\d+)\.\.(\d+)\]$/.exec(path);
    if (!match) return;
    const ref = this.resolve(match[1]!);
    const start = Number(match[2]), end = Number(match[3]);
    if (start > end) throw new Reject([{ path, code: 'bad-range' }]);
    const value = ref.get(), type = ref.env.resolve(ref.type!);
    if (Array.isArray(value) && type.kind === 'list') {
      if (end >= value.length) throw new Reject([{ path, code: 'bad-range' }]);
      return { ref, type: ref.type!, value: value.slice(start, end + 1) };
    }
    if (typeof value === 'string' && type.kind === 'prim' && type.name === 'string') {
      const lines = value.match(/[^\n]*\n|[^\n]+$/g) ?? [];
      if (start < 1 || end > lines.length) throw new Reject([{ path, code: 'bad-range' }]);
      return { ref, type: ref.type!, value: lines.slice(start - 1, end).join('') };
    }
    throw new Reject([{ path, code: 'bad-range' }]);
  }

  private inferScopeType(value: unknown): string {
    if (value instanceof Folder || value instanceof FolderHandle) return 'Folder';
    if (value instanceof FileHandle) return 'FileHandle';
    if (value === null) return 'null';
    if (typeof value === 'boolean') return 'boolean';
    if (typeof value === 'number' && Number.isFinite(value)) return 'number';
    if (typeof value === 'string') return 'string';
    if (Array.isArray(value)) {
      if (!value.length) throw new Reject([{ path: 'as_type', code: 'type-mismatch', expected: 'as_type for an empty list' }]);
      const types = [...new Set(value.map(item => this.inferScopeType(item)))];
      if (types.length !== 1) throw new Reject([{ path: 'as_type', code: 'type-mismatch', expected: 'as_type for a heterogeneous list' }]);
      return `(${types[0]})[]`;
    }
    if (value && typeof value === 'object') {
      const entries = Object.entries(value);
      if (!entries.length) throw new Reject([{ path: 'as_type', code: 'type-mismatch', expected: 'as_type for an empty record' }]);
      return `{ ${entries.map(([key, item]) => `${key}: ${this.inferScopeType(item)}`).join(', ')} }`;
    }
    throw new Reject([{ path: 'value', code: 'type-mismatch', expected: 'a portable literal' }]);
  }

  private scopePath(expression: string): string {
    const match = /^([A-Za-z_$][\w$]*)(.*)$/.exec(expression.trim());
    if (!match) throw new Reject([{ path: 'expression', code: 'bad-action', expected: 'a variable and field/index selections' }]);
    const root = match[1]!, base = root === 'args' ? 'args' : Object.hasOwn(this.lam.let, root) ? `let/${root}` :
      Object.hasOwn(this.lam.args, root) ? `args/${root}` : '';
    if (!base) throw new Reject([{ path: root, code: 'no-such-path', expected: 'a scope variable' }]);
    let path = base, tail = match[2]!;
    while (tail) {
      const member = /^\.([A-Za-z_$][\w$]*)(.*)$/s.exec(tail);
      const index = /^\[(?:"([^"\\]+)"|'([^'\\]+)'|(\d+))\](.*)$/s.exec(tail);
      if (member) { path += `/${member[1]}`; tail = member[2]!; }
      else if (index) { path += `/${index[1] ?? index[2] ?? index[3]}`; tail = index[4]!; }
      else throw new Reject([{ path: 'expression', code: 'bad-action', expected: 'field and index selection without computation' }]);
    }
    return path;
  }

  private scopeInitializerType(expression: string, locals: Record<string, Type>): Type | undefined {
    const source = expression.trim();
    // A reduce result has the accumulator's type, which need not match its list.
    if (/\.(?:reduce|reduceRight)\s*\(/.test(source)) return;
    const directCall = /^(?:await\s+)?([A-Za-z_$][\w$]*)\s*\(/.exec(source);
    if (directCall && Object.hasOwn(this.lam.codebase, directCall[1]!)) {
      const definition = this.lam.codebase[directCall[1]!] as Record<string, unknown>;
      if (typeof definition.returns === 'string') return parseType(definition.returns);
    }
    const mappedCall = /^await\s+Promise\.all\([\s\S]*\.map\([\s\S]*?([A-Za-z_$][\w$]*)\s*\(/.exec(source);
    if (mappedCall && Object.hasOwn(this.lam.codebase, mappedCall[1]!)) {
      const definition = this.lam.codebase[mappedCall[1]!] as Record<string, unknown>;
      if (typeof definition.returns === 'string') return parseType(`(${definition.returns})[]`);
    }
    const collection = /^(.*)\.(find|filter|slice)\s*\(.*\)$/s.exec(source);
    const projected = /^(.*)\.map\s*\(\s*(?:\(\s*)?([A-Za-z_$][\w$]*)(?:\s*\))?\s*=>\s*\2((?:\.[A-Za-z_$][\w$]*|\[(?:\d+|"[^"]+"|'[^']+')\])+)\s*\)$/s.exec(source);
    const receiver = (collection?.[1] ?? projected?.[1] ?? source).trim();
    const selected = (base: Type, tail: string): Type | undefined => {
      let type = base;
      while (tail) {
        const member = /^\.([A-Za-z_$][\w$]*)(.*)$/s.exec(tail);
        const index = /^\[(?:"([^"\\]+)"|'([^'\\]+)'|(\d+))\](.*)$/s.exec(tail);
        const resolved = this.env.resolve(type);
        if (member && resolved.kind === 'record') {
          const field = resolved.fields.find(item => item.name === member[1]); if (!field) return;
          type = field.type; tail = member[2]!;
        } else if (index && resolved.kind === 'list') { type = resolved.element; tail = index[4]!; }
        else if (index && resolved.kind === 'dict') { type = resolved.element; tail = index[4]!; }
        else if (index && resolved.kind === 'record') {
          const field = resolved.fields.find(item => item.name === (index[1] ?? index[2] ?? index[3])); if (!field) return;
          type = field.type; tail = index[4]!;
        } else return;
      }
      return type;
    };
    try {
      const root = /^[A-Za-z_$][\w$]*/.exec(receiver)?.[0]; if (!root) return;
      const type = root in locals ? selected(locals[root]!, receiver.slice(root.length)) : this.resolve(this.scopePath(receiver)).type;
      if (!type) return;
      const resolved = this.env.resolve(type);
      if (collection) return resolved.kind === 'list' ? (collection[2] === 'find' ? resolved.element : type) : undefined;
      if (projected) return resolved.kind === 'list' ?
        (() => { const item = selected(resolved.element, projected[3]!); return item ? { kind: 'list', element: item } as Type : undefined; })() : undefined;
      return type;
    } catch { return; }
  }

  private scopeView(): Record<string, unknown> {
    const locals = Object.fromEntries(Object.entries(this.lam.let).filter(([, value]) => !pending(value)));
    if (!Object.hasOwn(locals, 'result') && this.lam.return !== MISSING) locals.result = this.lam.return;
    return { args: inlineEvalView(this.lam.args as Value), inputs: inlineEvalView(this.lam.args as Value),
      locals: inlineEvalView(locals as Value) };
  }

  scopeBridge(operation: string, raw: unknown[]): unknown {
    if (operation === 'fs') return this.scopeFsBridge(String(raw[0] ?? ''), raw.slice(1));
    if (operation === 'handle') return this.scopeHandleBridge(raw[0], String(raw[1] ?? ''), raw.slice(2));
    if (operation === 'directory') return this.scopeDirectoryBridge(String(raw[0] ?? ''), raw[1],
      String(raw[2] ?? ''), Array.isArray(raw[3]) ? raw[3] : []);
    if (operation !== 'call')
      throw new Reject([{ path: 'code', code: 'bad-call', expected: 'a checked natlang call', got: operation }]);
    const functionName = String(raw[0] ?? '');
    const rawPositional = raw[1];
    if (!Array.isArray(rawPositional))
      throw new Reject([{ path: 'code', code: 'bad-call', expected: 'positional function arguments' }]);
    const definition = functionName.split('.').reduce<Record<string, unknown> | undefined>((definition, part) =>
      (definition?.codebase as Record<string, unknown> | undefined ?? definition)?.[part] as Record<string, unknown> | undefined,
      this.lam.codebase as Record<string, unknown>);
    if (!definition)
      throw new Reject([{ path: functionName, code: 'no-such-function' }]);
    if (definition.subtype === 'directory-reducer') {
      if (this.lam.subtype !== 'directory-reducer' || !this.lam.projectTransaction)
        throw new Reject([{ path: functionName, code: 'bad-call', expected: 'directory reducers may only be called by a directory reducer' }]);
      if (!rawPositional.length || !this.isScopeHandle(rawPositional[0]))
        throw new Reject([{ path: functionName, code: 'bad-call',
          expected: 'a Folder handle as the first argument to a directly called directory reducer' }]);
      return this.scopeDirectoryBridge('direct', rawPositional[0], functionName, rawPositional.slice(1));
    }
    const positional = rawPositional.map(value => this.materializeScopeValue(value));
    const signature = definition.args as Record<string, string> ?? {};
    const declared = Object.keys(signature);
    const required = declared.filter(name => !name.endsWith('?')).length;
    if (positional.length < required || positional.length > declared.length)
      throw new Reject([{ path: functionName, code: 'bad-call',
        expected: `${required} to ${declared.length} positional arguments`, got: String(positional.length) }]);
    const values = Object.fromEntries(positional.flatMap((value, index) => {
      const parameter = declared[index]!;
      if (value === undefined) {
        if (!parameter.endsWith('?')) throw new Reject([{ path: functionName, code: 'bad-call',
          expected: `${parameter} is required` }]);
        return [];
      }
      return [[parameter.replace(/\?$/, ''), value]];
    }));
    if (Object.hasOwn(definition, 'code') && definition.async === false &&
        !/\bawait\b/.test(String(definition.code))) {
      const names = declared.map(name => name.replace(/\?$/, ''));
      const children = Object.keys(definition.codebase as Record<string, unknown> ?? {});
      const helpers = children.map(name => `const ${name} = Object.assign((...args: unknown[]) => ` +
        `fx.natlang.scope(self.__natlangScopeToken, "call", ` +
        `[${JSON.stringify(`${functionName}.${name}`)}, args]), ` +
        `{ __natlangFunction: ${JSON.stringify(`${functionName}.${name}`)} });`).join('\n');
      const source = `${names.length ? `let { ${names.join(', ')} } = self.inputs;` : ''}\n${helpers}\n` +
        `return (() => {\n${String(definition.code)}\n})();`;
      const result = this.runtime.evalScopeForSync(this.lam, source, `function/${functionName}`,
        { inputs: jsView(values as Value) }, (op, args) => this.scopeBridge(op, args));
      const named = Object.fromEntries(Object.entries(definition.types as Record<string, string> ?? {})
        .map(([name, type]) => [name, parseType(type)]));
      return dump(coerce(result.result, parseType(String(definition.returns)), this.env.child(named),
        `function/${functionName}`));
    }
    const hidden = `__scope_call_${++this.scopeCallSequence}`;
    return (async () => { try {
      const outcome = await this.applyAsync('call', { function: functionName, to: `let/${hidden}`, values });
      if (outcome.kind !== 'done') throw new Error(`${functionName}: ${outcome.kind}: ${outcome.text}`);
      return dump(this.lam.let[hidden]!) as Value;
    } finally {
      delete this.lam.let[hidden];
      delete this.lam.letTypes[hidden];
    } })();
  }

  private isScopeHandle(value: unknown): value is { __natlangHandle: Record<string, unknown> } {
    return !!value && typeof value === 'object' && !!(value as Record<string, unknown>).__natlangHandle;
  }

  private materializeScopeValue(value: unknown): unknown {
    if (this.isScopeHandle(value)) return this.resolveScopeHandle(value);
    if (Array.isArray(value)) return value.map(item => this.materializeScopeValue(item));
    if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value as Record<string, unknown>)
      .map(([key, item]) => [key, this.materializeScopeValue(item)]));
    return value;
  }

  private materializeLazyDict(value: import('./host-tree.js').LazyDict): Record<string, unknown> {
    return materializeLazyDictFlat(value);
  }

  private resolveScopeHandle(value: unknown): FolderHandle | FileHandle {
    if (!this.isScopeHandle(value))
      throw new Reject([{ path: 'code', code: 'type-mismatch', expected: 'a folder or file handle' }]);
    const descriptor = value.__natlangHandle, source = String(descriptor.source ?? ''),
      name = String(descriptor.name ?? ''), path = String(descriptor.path ?? ''), kind = String(descriptor.kind ?? 'folder');
    const base = source === 'project' ? this.lam.projectTransaction?.folder :
      source === 'codebase' ? this.editableCodebase() :
      source === 'arg' ? this.lam.args[name] : source === 'local' ? this.lam.let[name] : undefined;
    if (!(base instanceof Folder) && !(base instanceof FolderHandle) && !(base instanceof FileHandle))
      throw new Reject([{ path: name || source, code: 'type-mismatch', expected: 'a folder or file handle' }]);
    const folder = base instanceof Folder ? base : base.folder;
    const basePath = base instanceof Folder ? '' : base.path;
    const joined = path ? folder.join(basePath, path) : basePath;
    return kind === 'file' ? folder.file(joined) : folder.dir(joined);
  }

  private scopePortableHandle(value: unknown, descriptor?: Record<string, unknown>): unknown {
    if (value instanceof FileHandle) return { __natlangHandle: { ...(descriptor ?? {}), path: value.path, kind: 'file' } };
    if (value instanceof FolderHandle) return { __natlangHandle: { ...(descriptor ?? {}), path: value.path, kind: 'folder' } };
    if (Array.isArray(value)) return value.map(item => this.scopePortableHandle(item, descriptor));
    return value;
  }

  private async scopeHandleBridge(rawHandle: unknown, method: string, rawArgs: unknown[]): Promise<unknown> {
    const handle = this.resolveScopeHandle(rawHandle), owner = handle.folder;
    if (['remove', 'moveTo'].includes(method) && owner === this.editableCodebase())
      throw new Reject([{ path: 'code', code: 'not-writable', expected: 'codebase files cannot be moved or deleted' }]);
    const callable = (handle as unknown as Record<string, unknown>)[method];
    if (typeof callable !== 'function')
      throw new Reject([{ path: 'code', code: 'bad-call', expected: 'a Folder or FileHandle method', got: method }]);
    const args = rawArgs.map(value => this.isScopeHandle(value) ? this.resolveScopeHandle(value) : value);
    const codebaseWrite = handle instanceof FileHandle && owner === this.editableCodebase() &&
      ['writeText', 'writeBytes', 'writeJson', 'editText'].includes(method);
    if (codebaseWrite && !owner.isFile(handle.path))
      throw new Reject([{ path: `codebase/${handle.path}`, code: 'not-writable', expected: 'an existing codebase file' }]);
    const before = codebaseWrite ? owner.readBytesSync(handle.path) : undefined;
    let value: unknown;
    try { value = await (callable as Function).apply(handle, args); if (codebaseWrite) this.refreshCodebaseFile(handle.path); }
    catch (error) { if (before) owner.writeBytes(handle.path, before); throw error; }
    return this.scopePortableHandle(value, this.isScopeHandle(rawHandle) ? rawHandle.__natlangHandle : undefined);
  }

  private async scopeDirectoryBridge(mode: string, rawFolder: unknown, functionName: string,
    positional: unknown[]): Promise<unknown> {
    if (this.lam.subtype !== 'directory-reducer' || !this.lam.projectTransaction)
      throw new Reject([{ path: functionName, code: 'bad-call', expected: 'directory reducers may only be called by a directory reducer' }]);
    const handle = this.resolveScopeHandle(rawFolder);
    if (!(handle instanceof FolderHandle))
      throw new Reject([{ path: 'code', code: 'type-mismatch', expected: 'a folder handle' }]);
    const definition = this.lam.codebase[functionName] as Record<string, unknown> | undefined;
    if (!definition || definition.subtype !== 'directory-reducer')
      throw new Reject([{ path: functionName, code: 'bad-call', expected: 'a directory-reducer function' }]);
    const signature = definition.args as Record<string, string> ?? {}, declared = Object.keys(signature);
    const required = declared.filter(name => !name.endsWith('?')).length;
    if (positional.length < required || positional.length > declared.length)
      throw new Reject([{ path: functionName, code: 'bad-call', expected: `${required} to ${declared.length} positional arguments` }]);
    const values = Object.fromEntries(positional.map((value, index) =>
      [declared[index]!.replace(/\?$/, ''), this.materializeScopeValue(value)]));
    let tx: FolderTransaction;
    try { tx = await handle.beginTransaction(false); }
    catch (error) {
      if (error instanceof FolderBusyError)
        throw new Reject([{ path: 'code', code: 'folder-busy', expected: 'the folder writer to become available' }]);
      throw error;
    }
    const hidden = `__scope_directory_${++this.scopeCallSequence}`;
    try {
      const outcome = await this.applyAsync('call', { function: functionName, to: `let/${hidden}`, values,
        project_transaction: tx, reducer_mode: mode === 'apply' ? 'apply' : 'direct' });
      if (outcome.kind !== 'done') throw new Error(`${functionName}: ${outcome.kind}: ${outcome.text}`);
      return dump(this.lam.let[hidden]!) as Value;
    } catch (error) { if (tx.open) tx.abort(); throw error; }
    finally { delete this.lam.let[hidden]; delete this.lam.letTypes[hidden]; }
  }

  private async scopeFsBridge(method: string, values: unknown[]): Promise<unknown> {
    const rawPath = values[0];
    if ((!values.length && !['list', 'diff', 'exists'].includes(method)) ||
        (values.length && typeof rawPath !== 'string'))
      throw new Reject([{ path: 'code', code: 'bad-call', expected: `fs.${method}(path, ...)` }]);
    const path = typeof rawPath === 'string' ? rawPath : '';
    if (path.startsWith('/') || path === '..' || path.startsWith('../') || path.includes('/../') ||
        (!path && !['list', 'diff', 'exists'].includes(method)))
      throw new Reject([{ path, code: 'no-such-path', expected: 'a relative path in the current folder' }]);
    const folder = this.lam.projectTransaction?.folder;
    if (!folder) throw new Reject([{ path, code: 'bad-action', expected: 'a directory reducer folder' }]);
    if (method === 'exists') return folder.exists(path);
    if (method === 'list') {
      const options = values[1] && typeof values[1] === 'object' ? values[1] as Record<string, unknown> : {};
      return folder.listFiles(path, options.pattern === undefined ? undefined : String(options.pattern))
        .map(item => ({ ...item, path: item.path }));
    }
    if (method === 'readText' || method === 'readJson') {
      const options = values[1] && typeof values[1] === 'object' ? values[1] as Record<string, unknown> : {};
      const content = await folder.readText(path, options.startLine === undefined ? undefined : Number(options.startLine),
        options.endLine === undefined ? undefined : Number(options.endLine));
      return method === 'readJson' ? JSON.parse(content) : content;
    }
    if (method === 'writeText' || method === 'writeJson') {
      if (values.length !== 2) throw new Reject([{ path: 'code', code: 'bad-call', expected: `fs.${method}(path, value)` }]);
      const before = folder.isFile(path) ? folder.readBytesSync(path) : undefined;
      folder.writeText(path, method === 'writeText' ? String(values[1]) : `${JSON.stringify(values[1], null, 2)}\n`);
      return null;
    }
    if (method === 'editText') {
      if (values.length !== 2 || !values[1] || typeof values[1] !== 'object')
        throw new Reject([{ path: 'code', code: 'bad-call', expected: 'fs.editText(path, { find, replaceWith, fuzzy? })' }]);
      const options = values[1] as Record<string, unknown>;
      const result = await folder.editText(path, String(options.find ?? ''), String(options.replaceWith ?? ''), options.fuzzy === true);
      return result;
    }
    if (method === 'diff') return folder.diffSync(path).changes.map(change =>
      ({ path: change.path, kind: change.kind }));
    if (method === 'remove') {
      folder.remove(path); return null;
    }
    if (method === 'move') {
      if (values.length !== 2 || typeof values[1] !== 'string')
        throw new Reject([{ path: 'code', code: 'bad-call', expected: 'fs.move(source, destination)' }]);
      const destination = values[1];
      if (!destination || destination.startsWith('/') || destination === '..' || destination.startsWith('../') || destination.includes('/../'))
        throw new Reject([{ path: destination, code: 'no-such-path', expected: 'a relative path in the current folder' }]);
      folder.move(path, destination); return null;
    }
    throw new Reject([{ path: 'code', code: 'bad-call', expected: 'an fs method', got: method }]);
  }

  private async scopeEvalNative(code: string): Promise<NativeResult> {
    if (!code.trim())
      return rejected(new Reject([{ path: 'code', code: 'bad-action', expected: 'a TypeScript statement or expression' }]));
    const opaqueInputs = Object.entries(this.lam.args)
      .filter(([, value]) => isLazyDict(value) || value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle);
    const opaqueInputNames = new Set(opaqueInputs.map(([name]) => name));
    const inputNames = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields
      .map(field => field.name).filter(name => !opaqueInputNames.has(name)) : [];
    const locals = Object.entries(this.lam.let)
      .filter(([, value]) => !pending(value) && !isLazyDict(value) && !(value instanceof Folder) &&
        !(value instanceof FolderHandle) && !(value instanceof FileHandle));
    const opaqueLocals = Object.entries(this.lam.let)
      .filter(([, value]) => isLazyDict(value) || value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle);
    const localBindings = locals.map(([name]) => ({ name,
      mutable: this.scopeLocalMutability.get(name) ?? true,
      annotation: this.lam.letTypes[name] ? formatType(this.lam.letTypes[name]!) : undefined }));
    const helperNames = Object.keys(this.lam.codebase);
    const opaqueNames = [...opaqueInputs, ...opaqueLocals].map(([name]) => name);
    if (this.lam.projectTransaction) opaqueNames.push('folder', 'fs');
    const compiled = compileScopeSnippet(code, { inputBindings: inputNames, localBindings, helperBindings: helperNames,
      opaqueBindings: opaqueNames, resultBinding: true,
      ...this.runtime.environment.scopeCapabilities });
    if (!compiled.ok || !compiled.program) {
      const text = compiled.diagnostics.map(item =>
        `${item.line}:${item.column} ${item.code}: ${item.message}`).join('\n');
      return { kind: 'rejected', text, codes: [...new Set(compiled.diagnostics.map(item => item.code))] };
    }
    const handleFactory = `const __makeHandle = (descriptor: Record<string, unknown>) => {\n` +
      `  const value: Record<string, unknown> = { __natlangHandle: descriptor };\n` +
      `  const child = (kind: string, path: string) => __makeHandle({ ...descriptor, kind, path: ` +
      `[String(descriptor.path ?? ""), path].filter(Boolean).join("/") });\n` +
      `  Object.defineProperties(value, {\n` +
      `    path: { get: () => String(descriptor.path ?? "") }, relativePath: { get: () => String(descriptor.path ?? "") },\n` +
      `    name: { get: () => String(descriptor.path ?? "").split("/").at(-1) ?? "" },\n` +
      `    dir: { value: (path: string) => child("folder", path) }, file: { value: (path: string) => child("file", path) },\n` +
      `    entry: { value: (path: string) => child("folder", path) },\n` +
      `    apply: { value: (fn: { __natlangFunction?: string }, ...args: unknown[]) => ` +
      `fx.natlang.scope(self.__natlangScopeToken, "directory", ["apply", value, fn.__natlangFunction, args]) },\n` +
      `  });\n` +
      `  for (const method of ["exists","stat","remove","moveTo","readText","readBytes","readJson",` +
      `"writeText","writeBytes","writeJson","editText","entries","files","folders","diff"]) ` +
      `Object.defineProperty(value, method, { value: (...args: unknown[]) => ` +
      `fx.natlang.scope(self.__natlangScopeToken, "handle", [value, method, ...args]) });\n` +
      `  return value;\n};\n`;
    const handleBindings = [
      ...opaqueInputs.map(([name, value]) => isLazyDict(value) ?
        `const ${name} = ${JSON.stringify(this.materializeLazyDict(value))};` :
        `const ${name} = __makeHandle(${JSON.stringify({ source: 'arg', name,
          path: '', kind: value instanceof FileHandle ? 'file' : 'folder' })});`),
      ...opaqueLocals.map(([name, value]) => isLazyDict(value) ?
        `const ${name} = ${JSON.stringify(this.materializeLazyDict(value))};` :
        `const ${name} = __makeHandle(${JSON.stringify({ source: 'local', name,
          path: '', kind: value instanceof FileHandle ? 'file' : 'folder' })});`),
      ...(this.lam.projectTransaction ? [`const folder = __makeHandle({ source: "project", path: "", kind: "folder" });`] : []),
    ].join('\n');
    const fsBinding = this.lam.projectTransaction ? `\nconst fs = new Proxy({}, { get: (_, method) => (...args: unknown[]) => ` +
      `fx.natlang.scope(self.__natlangScopeToken, "fs", [String(method), ...args]) });\n` : '\n';
    const source = handleFactory + handleBindings + fsBinding + importInvokePrelude(this.lam.codebase) +
      `${compiled.program}\n` +
      `return await ${compiled.entrypoint}(self.inputs, self.locals, __invoke);`;
    try {
      const evaluated = await this.runtime.evalScopeFor(this.lam, source, 'eval', this.scopeView(),
        (operation, args) => this.scopeBridge(operation, args));
      const output = evaluated.result as { result?: unknown; inputs?: Record<string, unknown>;
        bindings?: Record<string, unknown> };
      if (!output || typeof output !== 'object' || !output.bindings || typeof output.bindings !== 'object')
        throw new Reject([{ path: 'code', code: 'bad-action', expected: 'an atomic scope transaction result' }]);
      const annotations = new Map(compiled.bindings.map(binding => [binding.name, binding.annotation]));
      const initializers = new Map(compiled.bindings.map(binding => [binding.name, binding.initializer]));
      const mutability = new Map(compiled.bindings.map(binding => [binding.name, binding.mutable]));
      const staged: [string, Type, Value][] = [];
      const stagedInputs: [string, Value][] = [];
      const inferred: Record<string, Type> = { ...this.lam.letTypes };
      if (output.inputs && typeof output.inputs === 'object') for (const [name, value] of Object.entries(output.inputs)) {
        const field = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.find(item => item.name === name) : undefined;
        if (!field || !Object.hasOwn(this.lam.args, name))
          throw new Reject([{ path: name, code: 'no-such-path', expected: 'a function parameter' }]);
        stagedInputs.push([name, coerce(value, field.type, this.env, `args/${name}`)]);
      }
      for (const [name, value] of Object.entries(output.bindings)) {
        if (Object.hasOwn(this.lam.args, name) || Object.hasOwn(this.lam.codebase, name))
          throw new Reject([{ path: name, code: 'not-writable', expected: 'a local variable' }]);
        const materialized = this.isScopeHandle(value) ? this.resolveScopeHandle(value) : value;
        if (name === 'result' && this.lam.type.kind === 'lambda') {
          try { coerce(materialized, this.lam.type.returns, this.env, 'return'); }
          catch { continue; /* An intermediate observation must not change the typed result slot. */ }
        }
        let type = this.lam.letTypes[name];
        const annotation = annotations.get(name);
        if (annotation) type = parseType(annotation);
        if (!type && initializers.get(name)) type = this.scopeInitializerType(initializers.get(name)!, inferred);
        if (!type) {
          try { type = parseType(this.inferScopeType(materialized)); }
          catch (error) {
            if (this.lam.type.kind !== 'lambda' || !compiled.resultBindings?.includes(name)) throw error;
            type = this.lam.type.returns;
          }
        }
        inferred[name] = type;
        staged.push([name, type, coerce(materialized, type, this.env, `let/${name}`)]);
      }
      let functionResult: Value | undefined;
      if (compiled.producesResult && this.lam.type.kind === 'lambda') try {
        functionResult = coerce(output.result, this.lam.type.returns, this.env, 'return');
      } catch { /* An intermediate expression of another type is still a useful eval result. */ }
      if (functionResult === undefined && !compiled.producesResult && this.lam.type.kind === 'lambda') {
        const resultBinding = staged.find(([name]) => name === 'result');
        if (resultBinding) try {
          functionResult = coerce(resultBinding[2], this.lam.type.returns, this.env, 'return');
        } catch { /* A local named result may still be an intermediate value. */ }
      }
      for (const [name, value] of stagedInputs) this.lam.args[name] = value;
      for (const [name, type, value] of staged) {
        this.lam.letTypes[name] = type;
        this.lam.let[name] = value;
        if (mutability.has(name)) this.scopeLocalMutability.set(name, mutability.get(name)!);
      }
      if (functionResult !== undefined) {
        this.lam.return = functionResult;
        if (this.lam.subtype === 'directory-reducer') {
          this.lam.commitInclude = undefined; this.lam.commitExclude = undefined;
        }
      }
      const text = JSON.stringify(output.result ?? null);
      const rendered = text.length <= 400 ? text : `${text.slice(0, 400)} … (${text.length} chars)`;
      const open = functionResult === undefined ? [] : pendingProgramLines(this.lam.originalBody ?? this.lam.body, this.lam.marks);
      const status = functionResult === undefined ? '' : open.length ?
        `\nFunction result set; lines still open: ${open.join(', ')}.` : '\nFunction result set.';
      const stored = [
        ...staged.map(([name, , value]) => `local ${name} = ${oneLine(value)}`),
      ];
      const storedStatus = stored.length ? `\nStored ${stored.join('; ')}.` : '';
      const logStatus = evaluated.logs?.length ? `console:\n${evaluated.logs.join('\n')}\n` : '';
      return { kind: 'ok', text: logStatus + rendered + storedStatus + status,
        value: (output.result ?? null) as Value,
        ...(compiled.repairs.length ? { codes: ['coerced-redundant-self-alias'] } : {}) };
    } catch (error) {
      if (error instanceof Reject) return rejected(error);
      return { kind: 'error', text: error instanceof Error ? error.message : String(error) };
    }
  }

  private editableDefinitionSource(name: string, definition: Record<string, unknown>): string {
    const kind = Object.hasOwn(definition, 'code') ? 'code' : 'instructions';
    const meta: Record<string, unknown> = { description: definition.description ?? '',
      args: definition.args ?? {}, returns: definition.returns };
    if (definition.types && Object.keys(definition.types as object).length) meta.types = definition.types;
    if (Array.isArray(definition.effects) && definition.effects.length) meta.effects = definition.effects;
    if (definition.subtype === 'directory-reducer') meta.kind = definition.subtype;
    if (kind === 'code' && definition.engine && definition.engine !== 'typescript-host') meta.engine = definition.engine;
    const front = YAML.stringify(meta).trimEnd(), body = String(definition[kind] ?? '').replace(/^\n+|\n+$/g, '') + '\n';
    if (kind !== 'code') return `---\n${front}\n---\n${body}`;
    const parameters = Object.entries(definition.args as Record<string, string> ?? {}).map(([raw, type]) =>
      `${raw.replace(/\?$/, '')}${raw.endsWith('?') ? '?' : ''}: ${type}`).join(', ');
    const isAsync = definition.async === true || /\bawait\b/.test(body), returns = String(definition.returns);
    return `export default ${isAsync ? 'async ' : ''}function ${name}(${parameters}): ` +
      `${isAsync ? `Promise<${returns}>` : returns} {\n${body}}\n`;
  }

  private editableCodebase(): Folder {
    if (!this.lam.codebaseFolder) {
      const files: Record<string, string> = {};
      const add = (definitions: Record<string, unknown>, prefix: string[] = [], ancestors = new Set<object>()): void => {
        for (const [name, raw] of Object.entries(definitions)) {
          const definition = raw as Record<string, unknown>, extension = Object.hasOwn(definition, 'code') ? '.ts' : '.nl';
          const key = [...prefix, name].join('/'), path = [...prefix, `${name}${extension}`].join('/');
          const nested = definition.codebase as Record<string, unknown> | undefined;
          this.lam.codebasePaths[key] = path;
          this.lam.codebaseFiles[path] = definition;
          this.lam.codebaseImports[path] = Object.fromEntries(Object.entries(nested ?? {}).map(([childName, childRaw]) => {
            const child = childRaw as Record<string, unknown>, childExtension = Object.hasOwn(child, 'code') ? '.ts' : '.nl';
            return [childName, [...prefix, name, `${childName}${childExtension}`].join('/')];
          }));
          files[path] = this.editableDefinitionSource(name, definition);
          if (!ancestors.has(definition)) {
            if (nested) add(nested, [...prefix, name], new Set([...ancestors, definition]));
          }
        }
      };
      add(this.lam.codebase);
      this.lam.codebaseFolder = Folder.fromFiles(files, 'overlay');
    }
    return this.lam.codebaseFolder;
  }

  private functionSource(name: string): { key: string; path: string; folder: Folder } {
    const folder = this.editableCodebase();
    let key = name.trim().replace(/\./g, '/');
    if (!this.lam.codebasePaths[key] && !key.includes('/')) {
      const matches = Object.keys(this.lam.codebasePaths).filter(candidate => candidate.endsWith(`/${key}`));
      if (matches.length === 1) key = matches[0]!;
    }
    const path = this.lam.codebasePaths[key];
    if (!path) throw new Reject([{ path: name, code: 'no-such-function',
      expected: Object.keys(this.lam.codebasePaths).map(item => item.replace(/\//g, '.')).join(', ') }]);
    return { key, path, folder };
  }

  private refreshCodebaseFile(path: string): void {
    const entry = Object.entries(this.lam.codebasePaths).find(([, source]) => source === path);
    if (!entry) throw new Reject([{ path: `codebase/${path}`, code: 'no-such-path', expected: 'an imported function source' }]);
    const [binding] = entry, parts = binding.split('/');
    const previous = this.lam.codebaseFiles[path] as Record<string, unknown>;
    const source = new TextDecoder().decode(this.editableCodebase().readBytesSync(path));
    const isCode = path.endsWith('.ts');
    let meta: Record<string, unknown>, body: string;
    if (isCode) {
      const module = parseCrispModule(source, path, specifier => {
        const packages = (this.runtime.environment as EvalEnvironment & { packages?: {
          validateImportSpecifier(specifier: string): void } }).packages;
        if (!packages) throw new Error('Package imports require a project package.json');
        packages.validateImportSpecifier(specifier);
      });
      meta = { args: module.args, returns: module.returns, effects: module.effects,
        kind: previous.subtype ?? 'function', async: module.async };
      body = module.code;
    } else {
      if (/^\s*import\s/m.test(source)) throw new Reject([{ path: `codebase/${path}`, code: 'bad-import',
        expected: 'application subfunctions in the companion folder; no source imports' }]);
      const match = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?([\s\S]*)$/.exec(source);
      if (!match) throw new Reject([{ path: `codebase/${path}`, code: 'type-mismatch', expected: 'natural-language frontmatter between --- lines' }]);
      meta = YAML.parse(match[1]!) as Record<string, unknown> ?? {};
      body = match[2]!.replace(/^\n+|\n+$/g, '') + '\n';
    }
    if (typeof meta.returns !== 'string') throw new Reject([{ path: `codebase/${path}`, code: 'type-mismatch', expected: 'returns' }]);
    const subtype = String(meta.kind ?? 'function');
    if (!['function', 'directory-reducer'].includes(subtype))
      throw new Reject([{ path: `codebase/${path}/kind`, code: 'type-mismatch', expected: 'function or directory-reducer' }]);
    const updated: Record<string, unknown> = { description: String(meta.description ?? ''),
      args: meta.args ?? {}, returns: meta.returns, [isCode ? 'code' : 'instructions']:
        body, types: meta.types ?? previous.types ?? {},
      effects: meta.effects ?? [], subtype, codebase: {} };
    if (isCode) updated.engine = String(meta.engine ?? 'typescript-host');
    if (isCode) updated.async = meta.async === true;
    // Parse all declared types before making the edited binding live.
    const env = this.env.child(Object.fromEntries(Object.entries(updated.types as Record<string, string>)
      .map(([key, value]) => [key, parseType(value)])));
    env.checkNames(parseType(`(${Object.entries(updated.args as Record<string, string>).map(([key, value]) =>
      `${key.replace(/\?$/, '')}${key.endsWith('?') ? '?' : ''}: ${value}`).join(', ')}) => ${updated.returns}`));
    this.lam.codebaseFiles[path] = updated;
    const memo = new Map<string, Record<string, unknown>>();
    const link = (sourcePath: string): Record<string, unknown> => {
      const existing = memo.get(sourcePath); if (existing) return existing;
      const clone = { ...(this.lam.codebaseFiles[sourcePath] as Record<string, unknown>), codebase: {} };
      memo.set(sourcePath, clone);
      clone.codebase = Object.fromEntries(Object.entries(this.lam.codebaseImports[sourcePath] ?? {})
        .map(([alias, target]) => [alias, link(target)]));
      return clone;
    };
    this.lam.codebase = Object.fromEntries(Object.keys(this.lam.codebase)
      .filter(name => this.lam.codebasePaths[name]).map(name => [name, link(this.lam.codebasePaths[name]!) ]));
  }

  private checkEffects(value: Value, path: string): void {
    if (pending(value)) {
      if (value.nodeKind === 'lambda') {
        const extra = value.effects.filter(effect => !this.lam.effects.includes(effect));
        if (extra.length) throw new Reject([{ path, code: 'effect-wider-than-parent', got: extra.join(', ') }]);
        for (const [key, child] of Object.entries(value.args)) this.checkEffects(child, `${path}/args/${key}`);
        this.checkEffects(value.return, `${path}/return`);
      } else for (const part of ['over', 'fn', 'init', 'step', 'check'] as const) {
        const child = (value as unknown as Record<string, Value>)[part];
        if (child !== undefined) this.checkEffects(child, `${path}/${part}`);
      }
    } else if (Array.isArray(value)) value.forEach((child, index) => this.checkEffects(child, `${path}/${index}`));
    else if (value && typeof value === 'object') for (const [key, child] of Object.entries(value))
      this.checkEffects(child, `${path}/${key}`);
  }

  async applyAsync(name: string, args: Record<string, unknown>): Promise<NativeResult> {
    this.runtime.checkInterruption();
    if (['read_function', 'edit_function', 'diff_functions'].includes(name)) {
      if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
      if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
      this.actions++; this.toolCalls++; this.lam.steps++;
      try {
        if (name === 'diff_functions') {
          const folder = this.editableCodebase();
          const changes = folder.diffSync().changes.map(change => ({
            function: (Object.entries(this.lam.codebasePaths).find(([, path]) => path === change.path)?.[0] ?? change.path)
              .replace(/\//g, '.'), kind: change.kind,
          }));
          return this.record(name, args, { kind: 'ok', text: JSON.stringify(changes, null, 2), value: changes as Value });
        }
        const source = this.functionSource(String(args.name ?? ''));
        if (name === 'read_function') {
          const content = await source.folder.readText(source.path);
          let explanation = '';
          if (source.path.endsWith('.nl') && !this.explainedNaturalFunctions.has(source.key)) {
            this.explainedNaturalFunctions.add(source.key);
            explanation = 'Natural-language function source: its frontmatter declares parameter and return types; the body contains instructions executed line by line.\n\n';
          }
          return this.record(name, args, { kind: 'ok', text: explanation + content, value: content });
        }
        const before = source.folder.readBytesSync(source.path);
        const result = await source.folder.editText(source.path, String(args.find ?? ''),
          String(args.replace_with ?? ''), args.fuzzy === true);
        try { this.refreshCodebaseFile(source.path); }
        catch (error) { source.folder.writeBytes(source.path, before); throw error; }
        return this.record(name, args, { kind: 'ok', text: JSON.stringify(result), value: result as Value });
      } catch (error) {
        if (error instanceof Reject) return this.record(name, args, rejected(error));
        return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
      }
    }
    if (['list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files'].includes(name)) {
      if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
      if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
      this.actions++; this.toolCalls++; this.lam.steps++;
      try {
        const raw = String(args.path ?? '');
        if (raw.startsWith('/') || raw === '..' || raw.startsWith('../') || raw.includes('/../'))
          throw new Reject([{ path: raw, code: 'no-such-path', expected: 'a relative path in the current folder' }]);
        const folder = this.lam.projectTransaction?.folder;
        if (!folder) throw new Reject([{ path: raw, code: 'bad-action', expected: 'a directory reducer folder' }]);
        const path = raw;
        if (['read_file', 'write_file', 'edit_file'].includes(name) && !path)
          throw new Reject([{ path: raw, code: 'no-such-path', expected: 'a file in the current folder' }]);
        let value: unknown;
        if (name === 'list_files') value = folder.listFiles(path, args.pattern === undefined ? undefined : String(args.pattern))
          .map(item => ({ ...item, path: item.path }));
        else if (name === 'search_files') value = (await folder.search(String(args.query ?? ''), path,
          args.pattern === undefined ? undefined : String(args.pattern), args.regex === true))
          .map(item => ({ ...item, path: item.path }));
        else if (name === 'read_file') value = await folder.readText(path,
          args.start_line === undefined ? undefined : Number(args.start_line),
          args.end_line === undefined ? undefined : Number(args.end_line));
        else if (name === 'write_file') {
          const before = folder.isFile(path) ? folder.readBytesSync(path) : undefined;
          folder.writeText(path, String(args.content ?? ''));
          value = { path: raw, changed: true };
        }
        else if (name === 'edit_file') {
          const before = folder.readBytesSync(path);
          value = await folder.editText(path, String(args.find ?? ''), String(args.replace_with ?? ''), args.fuzzy === true);
        }
        else value = folder.diffSync(path);
        const text = typeof value === 'string' ? value : JSON.stringify(value, null, 1);
        return this.record(name, args, { kind: 'ok', text, value: value as Value });
      } catch (error) {
        if (error instanceof Reject) return this.record(name, args, rejected(error));
        return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
      }
    }
    if (name === 'eval') {
      if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
      if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
      this.actions++; this.toolCalls++; this.lam.steps++;
      return this.record(name, args, await this.scopeEvalNative(String(args.code ?? '')));
    }
    if (name !== 'call') return this.apply(name, args);
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.actions++; this.toolCalls++; this.lam.steps++;
    if (args.done !== undefined) try { this.doneRange(args.done); }
    catch (error) { if (error instanceof Reject) return this.record(name, args, rejected(error)); throw error; }
    let newLocal: string | undefined;
    try {
      const label = String(args.function ?? '');
      const copyName = /^let\/([A-Za-z_][A-Za-z0-9_]*)$/.exec(label)?.[1];
      const template = copyName ? this.lam.let[copyName] : undefined;
      const original = copyName ? this.lam.fnCopies[copyName] as Record<string, unknown> | undefined : undefined;
      if (!copyName && this.lam.codebaseFolder && this.lam.codebasePaths[label])
        this.refreshCodebaseFile(this.lam.codebasePaths[label]!);
      const definition = copyName && template !== undefined && pending(template) && template.nodeKind === 'lambda' && original ?
        { ...original, [template.kind]: template.body, codebase: template.codebase } :
        this.lam.codebase[label] as Record<string, unknown> | undefined;
      if (!definition) throw new Reject([{ path: 'function', code: 'no-such-function', got: label }]);
      const functionName = copyName && template !== undefined && pending(template) && template.nodeKind === 'lambda' ?
        template.functionName : label;
      const path = String(args.to ?? '');
      if (['inputs', 'values', 'over', 'init', 'until', 'max'].every(key => args[key] === undefined)) {
        try {
          const existingRef = this.resolve(path);
          const existing = existingRef.get();
          if (pending(existing) && ['unreduced', 'quiesced'].includes(existing.status) &&
              (existing.nodeKind === 'lambda' ? existing.functionName :
                existing.nodeKind === 'map' && pending(existing.fn) && existing.fn.nodeKind === 'lambda' ? existing.fn.functionName : '') ===
                functionName) {
            const outcome = await this.runtime.trigger(existingRef);
            return this.record(name, args, { kind: outcome.kind, text: `${path}: ${outcome.kind}  ${outcome.detail}`, value: outcome.value });
          }
        } catch (error) { if (!(error instanceof Reject)) throw error; }
      }
      const signature = definition.args as Record<string, string> ?? {};
      const params = Object.entries(signature).map(([name, type]) => `${name.replace(/\?$/, '')}${name.endsWith('?') ? '?' : ''}: ${type}`).join(', ');
      const typeText = `(${params}) => ${String(definition.returns)}`;
      const kind = Object.hasOwn(definition, 'code') ? 'code' : 'instructions';
      const inputPaths = (args.inputs ?? {}) as Record<string, string>;
      const values: Record<string, unknown> = { ...args.values as Record<string, unknown> ?? {} };
      const overlap = Object.keys(inputPaths).filter(name => Object.hasOwn(values, name));
      if (overlap.length) throw new Reject([{ path: 'values', code: 'bad-call', expected: 'parameters bound once', got: overlap.join(', ') }]);
      const named = Object.fromEntries(Object.entries(definition.types as Record<string, string> ?? {})
        .map(([name, text]) => [name, parseType(text)]));
      const callEnv = this.env.child(named);
      const declared = Object.fromEntries(Object.entries(signature).map(([name, text]) =>
        [name.replace(/\?$/, ''), parseType(text)]));
      const declaredNames = Object.keys(signature).map(name => name.replace(/\?$/, ''));
      for (const name of [...Object.keys(inputPaths), ...Object.keys(values)])
        if (!(name in declared)) throw new Reject([{ path: `inputs/${name}`, code: 'unknown-field' }]);
      for (const [name, path] of Object.entries(inputPaths)) {
        const source = this.resolve(path), expected = declared[name]!;
        if (!source.type || !fitsType(source.type, expected, callEnv))
          throw new Reject([{ path, code: 'type-does-not-fit-slot', expected: formatType(expected) }]);
        values[name] = cloneValue(source.get());
      }
      const required = Object.keys(signature).filter(name => !name.endsWith('?')).map(name => name.replace(/\?$/, ''));
      const overRef = args.over === undefined ? undefined : this.resolve(String(args.over));
      const over = overRef?.get();
      const init = args.init === undefined ? undefined : typeof args.init === 'string' ?
        (() => { try { return this.resolve(args.init as string).get(); } catch { return args.init; } })() : args.init;
      const later = over === undefined ? args.until === undefined ? [] : [required.find(n => !(n in values))] :
        init === undefined ? [required.find(n => !(n in values))] : declaredNames.slice(0, 2);
      if (required.some(name => !(name in values) && !later.includes(name)))
        throw new Reject([{ path: 'inputs', code: 'bad-call', expected: required.join(', ') }]);
      if (overRef && (overRef.env.resolve(overRef.type!).kind !== 'list' || !Array.isArray(over)))
        throw new Reject([{ path: overRef.path, code: 'type-does-not-fit-slot', expected: 'a list' }]);
      if (overRef && overRef.type) {
        const itemType = signature[init === undefined ? later[0]! : later[1]!];
        if (itemType && !fitsType(overRef.type, parseType(`${itemType}[]`), callEnv))
          throw new Reject([{ path: overRef.path, code: 'type-does-not-fit-slot', expected: `${itemType}[]` }]);
      }
      if (over === undefined && init === undefined && args.until === undefined && args.max === undefined) {
        try {
          const existing = this.resolve(path).get();
          const source = String(definition[kind] ?? '').replace(/\n+$/, '') + '\n';
          if (pending(existing) && existing.nodeKind === 'lambda' && existing.status === 'quiesced' &&
              existing.functionName === functionName && source === (existing.originalBody ?? existing.body) &&
              JSON.stringify(dump(existing.args as Value)) === JSON.stringify(dump(values as Value))) {
            throw new Reject([{ path, code: 'unchanged-retry',
              expected: 'change the function source, inputs, or state; or explicitly retry the pending local' }]);
          }
        } catch (error) {
          // A missing destination is the normal first-call case.  Preserve the
          // intentional unchanged-retry rejection.
          if (error instanceof Reject && error.diagnostics.some(diagnostic => diagnostic.code === 'unchanged-retry')) throw error;
          if (!(error instanceof Reject)) throw error;
        }
      }
      const leaf = { type: typeText, [kind]: definition[kind],
        ...(kind === 'code' && definition.engine && definition.engine !== 'typescript-host' ?
          { engine: definition.engine } : {}), args: values, types: definition.types ?? {},
        effects: definition.effects ?? [], codebase: definition.codebase ?? {}, function: functionName,
        subtype: definition.subtype ?? 'function' };
      let raw: Record<string, unknown> = { $lambda: leaf };
      let destination = String(definition.returns);
      if (args.until !== undefined) {
        const stateName = later[0];
        const check = this.lam.codebase[String(args.until)] as Record<string, unknown> | undefined;
        if (!stateName || !check) throw new Reject([{ path: 'until', code: 'bad-call' }]);
        const stateType = signature[stateName]!;
        const checkArgs = check.args as Record<string, string> ?? {};
        const checkName = Object.keys(checkArgs)[0];
        if (!checkName) throw new Reject([{ path: 'until', code: 'bad-call', expected: 'a one-parameter check' }]);
        const checkKind = Object.hasOwn(check, 'code') ? 'code' : 'instructions';
        raw = { $iterate: { type: `Iterate<${stateType}>`, init, max: args.max,
          state_name: stateName, check_name: checkName,
          step: { $lambda: leaf }, check: { $lambda: { type: `(${checkName}: ${checkArgs[checkName]}) => ${check.returns}`,
            [checkKind]: check[checkKind],
            ...(checkKind === 'code' && check.engine && check.engine !== 'typescript-host' ?
              { engine: check.engine } : {}), function: String(args.until) } } } };
        destination = stateType;
      } else if (over !== undefined && init !== undefined) {
        const [accName, itemName] = declaredNames;
        if (!accName || !itemName) throw new Reject([{ path: 'inputs', code: 'bad-call' }]);
        raw = { $fold: { type: `Fold<${signature[itemName]}, ${signature[accName]}>`, over, init,
          acc_name: accName, item_name: itemName, step: { $lambda: leaf } } };
        destination = signature[accName]!;
      } else if (over !== undefined) {
        const itemName = later[0];
        if (!itemName) throw new Reject([{ path: 'inputs', code: 'bad-call' }]);
        raw = { $map: { type: `Map<${signature[itemName]}, ${definition.returns}>`, over,
          item_name: itemName, fn: { $lambda: leaf } } };
        destination = `${definition.returns}[]`;
      }
      const child = buildPending(raw, this.env);
      if (child.nodeKind === 'lambda' && args.project_transaction instanceof Object) {
        child.projectTransaction = args.project_transaction as FolderTransaction;
        child.reducerMode = args.reducer_mode === 'apply' ? 'apply' : 'direct';
      }
      if (path.startsWith('let/')) {
        const local = path.slice(4);
        if (!this.lam.letTypes[local]) {
          if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(local))
            throw new Reject([{ path, code: 'no-such-path', expected: 'let/<name>' }]);
          this.lam.letTypes[local] = parseType(destination); newLocal = local;
        }
      }
      const ref = this.resolve(path, true);
      if (ref.deny) throw new Reject([{ path, code: ref.deny }]);
      if (!ref.type || !fitsType(child.type, ref.type, this.env))
        throw new Reject([{ path, code: 'type-does-not-fit-slot', expected: ref.type ? formatType(ref.type) : '' }]);
      this.checkEffects(child, path); ref.set(child);
      const outcome = await this.runtime.trigger(ref);
      return this.record(name, args, { kind: outcome.kind, text: `${path}: ${outcome.kind}  ${outcome.detail}`, value: outcome.value });
    } catch (error) {
      if (newLocal && !Object.hasOwn(this.lam.let, newLocal)) delete this.lam.letTypes[newLocal];
      if (error instanceof Reject) return this.record(name, args, rejected(error));
      return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
    }
  }

  private resolve(path: string, create = false): Ref {
    if (path.startsWith('/') || path.split('/').some(part => part === '..' || part === '.'))
      throw new Reject([{ path, code: 'out-of-scope' }]);
    const parts = path.split('/').filter(Boolean);
    if (!parts.length) throw new Reject([{ path, code: 'no-such-path' }]);
    const first = parts.shift()!;
    let container: Record<string, Value>, type: Type, deny = '';
    if (first === 'args') { container = this.lam.args; type = this.lam.type.kind === 'lambda' ? this.lam.type.params : parseType('{}'); deny = 'not-writable'; }
    else if (first === 'return') { container = this.lam as unknown as Record<string, Value>; type = this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseType('null'); }
    else if (first === 'let') { container = this.lam.let; type = parseType('{}'); }
    else if (first === this.lam.kind) {
      if (parts.length) throw new Reject([{ path, code: 'no-such-path' }]);
      return itemRef(this.lam as unknown as Record<string, Value>, 'body', parseType('string'), this.env, first);
    }
    else throw new Reject([{ path, code: 'no-such-path' }]);
    if (first === 'return' && parts.length === 0) return itemRef(container, 'return', type, this.env, path, deny);
    if (first === 'args' && parts.length === 0) return { path, type, env: this.env, deny, get: () => this.lam.args as Value,
      set: () => { throw new Reject([{ path, code: 'not-writable' }]); }, del: () => {} };
    const name = parts.shift();
    if (!name) throw new Reject([{ path, code: 'no-such-path' }]);
    if (first === 'let') {
      type = this.lam.letTypes[name] ?? parseType('null');
      if (!this.lam.letTypes[name]) throw new Reject([{ path, code: 'no-such-path' }]);
    } else if (first === 'args') {
      if (type.kind !== 'record') throw new Reject([{ path, code: 'no-such-path' }]);
      const field = type.fields.find(f => f.name === name); if (!field) throw new Reject([{ path, code: 'unknown-field' }]);
      type = field.type;
    } else {
      const parent = itemRef(container, 'return', type, this.env, 'return');
      return this.descend(parent, [name, ...parts], create, deny);
    }
    return this.descend(itemRef(container, name, type, this.env, `${first}/${name}`, deny), parts, create, deny);
  }

  private descend(ref: Ref, parts: string[], create: boolean, deny: string): Ref {
    for (const part of parts) {
      const current = ref.get();
      if (pending(current)) {
        ref = this.pendingChild(ref, current, part);
        continue;
      }
      let type = ref.env.resolve(ref.type!);
      if (type.kind === 'union') {
        const selected = type.members.map(member => ref.env.resolve(member)).find(member =>
          member.kind === 'record' ? member.fields.some(field => field.name === part) :
            member.kind === 'dict' ? current !== null && typeof current === 'object' && !Array.isArray(current) :
              member.kind === 'list' ? Array.isArray(current) : false);
        if (selected) type = selected;
      }
      if (type.kind === 'dict' && isLazyDict(current)) {
        let child: Value;
        try {
          const raw = current.child(part);
          child = isLazyDict(raw) ? raw : coerce(raw, type.element, ref.env, `${ref.path}/${part}`);
        } catch (error) {
          throw new Reject([{ path: `${ref.path}/${part}`, code: 'no-such-path',
            expected: error instanceof Error ? error.message : String(error) }]);
        }
        const childType = isLazyDict(child) ? type : type.element;
        const parent = ref;
        ref = { path: `${parent.path}/${part}`, type: childType, env: parent.env,
          deny: parent.deny || deny || 'not-writable', get: () => child,
          set: () => { throw new Reject([{ path: `${parent.path}/${part}`, code: 'not-writable' }]); }, del: () => {} };
        continue;
      }
      let childType: Type;
      if (type.kind === 'record') {
        const field = type.fields.find(f => f.name === part);
        if (!field) throw new Reject([{ path: `${ref.path}/${part}`, code: 'unknown-field' }]);
        childType = field.type;
      } else if (type.kind === 'dict') childType = type.element;
      else if (type.kind === 'list') {
        if (part !== '+' && !/^\d+$/.test(part)) throw new Reject([{ path: `${ref.path}/${part}`, code: 'no-such-path' }]);
        const list = ref.get();
        if (Array.isArray(list) && Number(part) > list.length) throw new Reject([{ path: `${ref.path}/${part}`, code: 'no-such-path' }]);
        childType = type.element;
      } else throw new Reject([{ path: `${ref.path}/${part}`, code: 'no-such-path' }]);
      const parent = ref, key = part;
      ref = { path: `${parent.path}/${part}`, type: childType, env: parent.env, deny: parent.deny || deny,
        get: () => {
          const value = parent.get();
          return value !== MISSING && value !== null && typeof value === 'object' &&
            key in value ? (value as Record<string, Value>)[key]! : MISSING;
        },
        set: value => {
          let object = parent.get();
          if (object === MISSING || object === null) { object = type.kind === 'list' ? [] : {}; parent.set(object); }
          if (Array.isArray(object) && key === '+') object.push(value);
          else (object as Record<string, Value>)[key] = value;
        },
        del: () => { const object = parent.get(); if (Array.isArray(object)) object.splice(Number(key), 1);
          else if (object && typeof object === 'object') delete (object as Record<string, Value>)[key]; } };
    }
    return ref;
  }

  private pendingChild(parent: Ref, node: Pending, part: string): Ref {
    const at = `${parent.path}/${part}`;
    const env = parent.env.child(node.types);
    const deny = parent.deny || (node.status === 'running' && node !== this.lam ? 'frozen' : '');
    const field = (key: string, type: Type, denied = deny): Ref => itemRef(node as unknown as Record<string, Value>, key, type, env, at, denied);
    if (node.nodeKind === 'lambda' && node.type.kind === 'lambda') {
      if (part === 'args') return { path: at, type: node.type.params, env,
        deny: node === this.lam ? 'not-writable' : deny,
        get: () => node.args as Value, set: () => { throw new Reject([{ path: at, code: 'not-writable' }]); }, del: () => {} };
      if (part === 'return') return field('return', node.type.returns);
      if (part === node.kind) return field('body', parseType('string'));
      if (part === 'let' && node === this.lam) return { path: at, env, deny,
        get: () => node.let as Value, set: () => { throw new Reject([{ path: at, code: 'not-writable' }]); }, del: () => {} };
    }
    if (node.nodeKind === 'map' && node.type.kind === 'map') {
      if (part === 'over' || part === 'fn') return field(part, partType(node, part));
      if (/^\d+$/.test(part) && node.slots && Number(part) < node.slots.length)
        return itemRef(node.slots, Number(part), node.type.b, env, at, deny);
    }
    if (node.nodeKind === 'fold' && node.type.kind === 'fold') {
      if (['over', 'init', 'step'].includes(part)) return field(part, partType(node, part));
      if (part === 'acc' || part === 'at') return field(part, part === 'acc' ? node.type.s : parseType('number'), 'not-writable');
      if (part === 'current' && node.current !== null) return field(part, partType(node, 'step'));
    }
    if (node.nodeKind === 'iterate' && node.type.kind === 'iterate') {
      if (['init', 'step', 'check', 'max'].includes(part)) return field(part, partType(node, part));
      if (part === 'state' || part === 'iteration') return field(part, part === 'state' ? node.type.s : parseType('number'), 'not-writable');
      if (part === 'current' && node.current !== null) return field(part, partType(node, 'step'));
    }
    throw new Reject([{ path: at, code: 'no-such-path' }]);
  }
}
