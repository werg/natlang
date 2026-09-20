import { createHash } from 'node:crypto';
import YAML from 'yaml';
import { TypeScriptEnvironment, type HostEvent } from '../environment.js';
import { TypeEnv, fitsType, formatType, parseType, resultType, type Type } from './types.js';
import { MISSING, Reject, buildPending, cloneValue, coerce, dump, dumpState, isPending, loadProgram,
  partType, problems, unboundParts, type LambdaNode, type Pending, type Value } from './values.js';
import { changes, NativeTraceRecorder } from './trace.js';

export type NativeOutcome = { path: string; kind: 'done' | 'quiesced' | 'waiting' | 'replaced'; detail: string; value?: Value };
export type NativeResult = { kind: string; text: string; value?: Value; codes?: string[] };
export type NativeAgent = (session: NativeSession) => Promise<string | void> | string | void;
export type NativePoll = { kind: 'item'; value: unknown } | { kind: 'empty' } | { kind: 'closed' } | { kind: 'failed'; detail: string };
export interface NativeStream { poll(): Promise<NativePoll> | NativePoll }

type Ref = { path: string; type?: Type; env: TypeEnv; deny?: string;
  get(): Value; set(value: Value): void; del(): void };
const pending = (value: Value): value is Pending => isPending(value);
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const q = (value: unknown) => JSON.stringify(value);
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
  'not-writable': 'args are read-only. Write into return, or into the args of a sub-task you defined.',
  frozen: 'That sub-task is running; its args cannot change now.',
  'type-mismatch': 'Pass the value itself with the type shown as expected, not wrapped in another object: for Bool `true`, for Num `42.5`, for Text a string, for a record an object with exactly its fields.',
  'type-does-not-fit-slot': 'That slot needs the type shown as expected.',
  'unknown-field': 'Use one of the fields listed as expected.',
  'unbound-param': 'Give the sub-task its inputs first: copy a value into the path shown, or define it with args_from.',
  'unbound-part': 'The sub-task is missing the part shown: for a Map or Fold, pass the list with over_from or copy it to .../over.',
  'no-such-path': 'Use a path that appears in the state.',
  'old-not-found': 'Copy `old` exactly from the text, including punctuation.',
  'old-not-unique': 'Make `old` longer so that it occurs only once.',
  'no-origin': 'Only a result that a sub-task produced can be retried. Write the value again instead.',
  'too-deep': 'Sub-tasks are nested too deeply. Do this step directly.',
  'too-large': 'Read a part of it with `from` and `to`, or define a Map over it so that each sub-task sees one item.',
  'stuck-dependency': 'An input of this sub-task could not be produced: read its note, fix it, run it again.',
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
function jsView(value: Value): unknown {
  if (value === MISSING) return null;
  if (pending(value)) return { $pending: formatType(value.type), status: value.status };
  if (Array.isArray(value)) return value.map(jsView);
  if (value !== null && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, jsView(v)]));
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
  readonly options: { maxEpisodes: number; maxDepth: number; runId: string; mapWorkers: number; parallelModelSafe: boolean };
  readonly seedPolicy: { mode: 'compatibility' | 'derived' | 'backend'; root?: number };
  readonly environment: TypeScriptEnvironment;
  readonly agent?: NativeAgent;
  readonly capabilities: Record<string, (args: unknown[]) => unknown>;
  readonly episodeBudget: { limit: number; used: number };
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

  constructor(options: { environment?: TypeScriptEnvironment; host?: object; agent?: NativeAgent;
    capabilities?: Record<string, (args: unknown[]) => unknown>; maxEpisodes?: number;
    maxDepth?: number; runId?: string; stream?: NativeStream;
    signal?: AbortSignal; timeoutMs?: number;
    sourceRevision?: string; parentCallId?: string;
    sharedEpisodeBudget?: { limit: number; used: number };
    mapWorkers?: number; parallelModelSafe?: boolean;
    seedPolicy?: { mode: 'compatibility' | 'derived' | 'backend'; root?: number } } = {}) {
    this.options = { maxEpisodes: options.maxEpisodes ?? 256, maxDepth: options.maxDepth ?? 8,
      runId: options.runId ?? 'native-run', mapWorkers: options.mapWorkers ?? 1,
      parallelModelSafe: options.parallelModelSafe ?? false };
    if (!Number.isInteger(this.options.mapWorkers) || this.options.mapWorkers < 1)
      throw new RangeError('mapWorkers must be positive');
    this.episodeBudget = options.sharedEpisodeBudget ?? { limit: this.options.maxEpisodes, used: 0 };
    this.seedPolicy = options.seedPolicy ?? { mode: 'compatibility' };
    if (this.seedPolicy.mode === 'derived' && !Number.isInteger(this.seedPolicy.root))
      throw new TypeError('derived seed policy requires an integer root');
    this.trace = new NativeTraceRecorder({ run_id: this.options.runId, tool_schema: 'tools-v3',
      ...(options.sourceRevision ? { source_revision: options.sourceRevision } : {}),
      ...(options.parentCallId ? { parent_call_id: options.parentCallId } : {}),
      engines: ['typescript-host'], engine_contracts: { 'typescript-host': {
        environment_mode: options.environment?.mode ?? 'fresh', authority: 'shared-node-host', native_state_replayable: false } },
      seed_policy: this.seedPolicy, coverage: 'natlang-state-and-observed-host-effects' });
    this.agent = options.agent;
    this.capabilities = options.capabilities ?? {};
    this.environment = options.environment ?? new TypeScriptEnvironment({ mode: 'fresh', host: options.host });
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
      this.events.push(...result.events);
      return result;
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
      this.checkInterruption();
      this.events.push(...result.events);
      return result;
    } finally { this.acting = previous; this.currentCallId = previousCallId; }
  }

  private acting?: LambdaNode;
  private effect(cap: string, fn: string, args: unknown[]): unknown {
    this.checkInterruption();
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
    if (!this.trace.events.some(event => event.kind === 'state')) {
      this.trace.emit('state', { phase: 'initial', value: before });
      this.lastObserved = before;
    }
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
    return { path: ref.path, kind: 'done', detail: q(dump(value)), value };
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
      const result = await this.evalForAsync(node, node.body, ref.path,
        { args: jsView(node.args as Value), return: jsView(node.return) });
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
    if (this.depth >= this.options.maxDepth) return this.quiesce(ref, node, `run budget: episodes nested deeper than ${this.options.maxDepth}`);
    if (this.localEpisodesStarted >= this.options.maxEpisodes || this.episodeBudget.used >= this.episodeBudget.limit)
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
    } finally { this.trace.emit('invocation', { phase: 'end', call_id: callId });
      this.stack.pop(); this.invocationPaths.pop(); this.depth--; this.currentCallId = previousCallId; }
    if (session.completed) return this.done(ref, node, node.return);
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
        const child = new NativeRuntime({ environment: new TypeScriptEnvironment({ mode: 'fresh' }),
          agent: this.agent, maxEpisodes: this.options.maxEpisodes, maxDepth: this.options.maxDepth,
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
          if (polled.kind === 'empty') { node.status = 'waiting'; node.note = 'waiting for stream input'; return { path: ref.path, kind: 'waiting', detail: node.note }; }
          if (polled.kind === 'closed') return this.done(ref, node, node.acc);
          if (polled.kind === 'failed') return this.quiesce(ref, node, `stream failed: ${polled.detail}`);
          this.streamPosition++;
          this.streamCurrent = coerce(polled.value, node.type.a, env, `${ref.path}/over/${node.at}`);
        }
        item = this.streamCurrent;
      } else {
        if (!Array.isArray(node.over)) return this.quiesce(ref, node, 'invalid Fold input');
        if (node.at >= node.over.length) return this.done(ref, node, node.acc);
        item = node.over[node.at]!;
      }
      if (node.current === null) {
        const step = cloneValue(node.step as LambdaNode);
        step.args.acc = cloneValue(node.acc); step.args.item = cloneValue(item);
        node.current = step;
      }
      const child = itemRef(node as unknown as Record<string, Value>, 'current', node.type.s, env, `${ref.path}/step/${node.at}`);
      const out = await this.trigger(child);
      if (out.kind !== 'done') return this.quiesce(ref, node, `step ${node.at} ${out.kind}: ${out.detail}`);
      node.acc = node.current; node.current = null; node.at++;
      if (this.stream && !ref.path) this.streamCurrent = undefined;
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
      const chk = itemRef(box, 'value', check.type.kind === 'lambda' ? check.type.returns : parseType('Bool'), env,
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
  private textSet = false;
  readonly env: TypeEnv;
  constructor(readonly runtime: NativeRuntime, readonly lam: LambdaNode, readonly outerEnv: TypeEnv,
    readonly path = '') {
    this.env = outerEnv.child(lam.types);
  }
  finish(): boolean {
    if (this.lam.return === MISSING || this.lam.type.kind !== 'lambda') return false;
    const p = problems(this.lam.return, this.lam.type.returns, this.env, 'return');
    if (p.holes.length || p.pending.length) return false;
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
      surface: 'tools-v3', name, arguments: args,
      outcome: result.kind, diagnostics: result.codes ?? [] });
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
  /** Legacy one-header text action surface used by the conformance harness. */
  async act(source: string): Promise<NativeResult> {
    const text = source.replace(/^\s*<\|tool_call_start\|>/, '').replace(/<\|tool_call_end\|>\s*$/, '').replace(/^\n+|\n+$/g, '');
    const [header, ...body] = text.split('\n');
    const command = header?.trim() ?? '';
    const joined = body.join('\n');
    let result: NativeResult;
    try {
      const set = /^set\s+(\S+)\s+:\s+(.+)$/.exec(command);
      const copy = /^copy\s+(\S+)\s+to\s+(\S+)$/.exec(command);
      const simple = /^(read|edit|unset|reopen)\s+(\S+)$/.exec(command);
      if (set) {
        const type = set[2]!.trim(), parsed = parseType(type);
        const raw = this.env.resolve(parsed).kind === 'prim' && formatType(parsed) === 'Text' ? joined : YAML.parse(joined);
        const wrapper = ['lambda', 'map', 'fold', 'iterate'].includes(parsed.kind) ? `$${parsed.kind}` : '';
        this.textSet = true;
        try { result = this.apply('write', { path: set[1], type, value: wrapper ? { [wrapper]: { type, ...raw as object } } : raw }); }
        finally { this.textSet = false; }
      } else if (copy) result = this.apply('copy', { from: copy[1], to: copy[2] });
      else if (command.startsWith('reduce ')) result = await this.applyAsync('run', { paths: command.slice(7).trim().split(/\s+/) });
      else if (command === 'eval') result = this.apply('run_code', { code: joined, engine: 'typescript-host' });
      else if (simple) {
        const [, tool, path] = simple;
        if (tool === 'read') result = this.apply('read', { path });
        else if (tool === 'edit') result = this.legacyEdit(path!, joined);
        else if (tool === 'unset') result = this.apply('delete', { path });
        else if (tool === 'reopen') result = this.apply('retry', { path, feedback: joined });
        else throw new Reject([{ path: path!, code: 'bad-action' }]);
      } else throw new Reject([{ path: command, code: 'bad-action' }]);
    } catch (error) {
      result = error instanceof Reject ? { kind: 'rejected', text: error.message, codes: error.diagnostics.map(d => d.code) } :
        { kind: 'error', text: error instanceof Error ? error.message : String(error) };
    }
    this.runtime.trace.emit('action', { call_id: this.runtime.currentCallId ?? null,
      surface: 'text', action: source, outcome: result.kind, diagnostics: result.codes ?? [] });
    this.runtime.observeState('after-action');
    return result;
  }
  private legacyEdit(path: string, body: string): NativeResult {
    try {
      const match = /^(.*?)\[(\d+)\.\.(\d+)\]$/.exec(path);
      const ref = this.resolve(match?.[1] ?? path);
      if (ref.deny) throw new Reject([{ path: ref.path, code: ref.deny }]);
      const old = ref.get();
      if (typeof old !== 'string') throw new Reject([{ path: ref.path, code: 'type-mismatch',
        expected: 'a Text node', got: ref.type ? formatType(ref.type) : '' }]);
      const lines = old.match(/[^\n]*\n|[^\n]+$/g) ?? [];
      const replacement = body && !body.endsWith('\n') ? body + '\n' : body;
      let next = replacement;
      if (match) {
        const start = Number(match[2]), end = Number(match[3]);
        if (start < 1 || end < start || end > lines.length)
          throw new Reject([{ path, code: 'bad-range', expected: `lines 1..${lines.length}` }]);
        next = lines.slice(0, start - 1).join('') + replacement + lines.slice(end).join('');
      }
      if (ref.path === this.lam.kind && !next.trim()) {
        if (this.lam.type.kind !== 'lambda') throw new Reject([{ path, code: 'type-mismatch' }]);
        const issues = problems(this.lam.return, this.lam.type.returns, this.env, 'return');
        if (issues.holes.length || issues.pending.length) {
          const diagnostics = [...issues.holes.map(item => ({ path: item.path, code: 'commit-holes',
            expected: item.expected })), ...issues.pending.map(item => ({ path: item, code: 'commit-pending' }))];
          const text = `refused\n${diagnostics.map(item => `${item.path}: ${item.code}${'expected' in item && item.expected ? `, expected ${item.expected}` : ''}`).join('\n')}`;
          return { kind: 'refused', text, codes: [...new Set(diagnostics.map(item => item.code))] };
        }
        ref.set(''); this.completed = true;
        return { kind: 'completed', text: 'completed', value: this.lam.return };
      }
      ref.set(next);
      return { kind: 'ok', text: `ok   ${this.summary()}` };
    } catch (error) {
      if (error instanceof Reject) return { kind: 'rejected', text: `rejected\n${error.message}`,
        codes: error.diagnostics.map(item => item.code) };
      throw error;
    }
  }
  apply(name: string, args: Record<string, unknown>): NativeResult {
    this.runtime.checkInterruption();
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    if (this.actions >= 40 || this.toolCalls >= 128)
      return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.toolCalls++;
    if (name !== 'mark_done') this.actions++;
    if (name !== 'mark_done') this.lam.steps++;
    if (name === 'write' && args.done !== undefined) try { this.doneRange(args.done); }
    catch (error) { if (error instanceof Reject) return this.record(name, args, rejected(error)); throw error; }
    return this.record(name, args, this.applyNow(name, args));
  }
  private applyNow(name: string, args: Record<string, unknown>): NativeResult {
    try {
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
          const type = parseType(`Lambda<{ ${params} }, ${String(def.returns)}>`);
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
        if (!this.textSet && args.source === undefined &&
            (['lambda', 'map', 'fold', 'iterate'].includes(stated.kind) ||
              (args.value && typeof args.value === 'object' && !Array.isArray(args.value) &&
                Object.keys(args.value as object).some(key => ['$lambda', '$map', '$fold', '$iterate'].includes(key)))))
          throw new Reject([{ path: 'type', code: 'anonymous-lambda', expected: 'call with a checked function' }]);
        const local = /^let\/([A-Za-z_][A-Za-z0-9_]*)$/.exec(path)?.[1];
        const created = !!local && !Object.hasOwn(this.lam.letTypes, local);
        if (created && Object.keys(this.lam.letTypes).length >= 16)
          throw new Reject([{ path, code: 'too-many-locals', expected: 'at most 16 locals' }]);
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
        if (ref.deny) throw new Reject([{ path: ref.path, code: ref.deny }]);
        const value = ref.get();
        return { kind: 'ok', text: value === MISSING ? `${ref.path}: not supplied (missing value; not empty text)` :
          typeof value === 'string' ? value : JSON.stringify(dump(value), null, 1), value };
      }
      if (name === 'edit') {
        const ref = this.resolve(String(args.path ?? ''));
        if (ref.deny) throw new Reject([{ path: ref.path, code: ref.deny }]);
        const value = ref.get(), old = String(args.old ?? ''), replacement = String(args.new ?? '');
        if (typeof value !== 'string') throw new Reject([{ path: ref.path, code: 'type-mismatch', expected: 'a text' }]);
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
      if (name === 'run_code') {
        if (args.engine === undefined) throw new Reject([{ path: 'engine', code: 'bad-action', expected: 'an explicit available engine' }]);
        const engine = String(args.engine);
        if (engine !== 'typescript-host') return { kind: 'error', text: `engine ${engine} unavailable` };
        const result = this.runtime.evalFor(this.lam, String(args.code ?? ''), false, 'eval',
          { instructions: this.lam.body,
            args: jsView(this.lam.args as Value), return: jsView(this.lam.return), let: jsView(this.lam.let as Value) });
        return { kind: 'ok', text: JSON.stringify(result.result), value: result.result as Value };
      }
      throw new Reject([{ path: name, code: 'bad-action' }]);
    } catch (error) {
      if (error instanceof Reject) return rejected(error);
      const message = error instanceof Error ? error.message : String(error);
      if (message.includes('NATLANG:effect-undeclared')) return { kind: 'rejected', text: message, codes: ['effect-undeclared'] };
      if (name === 'run_code' && /read only|Cannot assign|not extensible/i.test(message))
        return { kind: 'rejected', text: message, codes: ['eval-cannot-write'] };
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
    if (typeof value === 'string' && type.kind === 'prim' && type.name === 'Text') {
      const lines = value.match(/[^\n]*\n|[^\n]+$/g) ?? [];
      if (start < 1 || end > lines.length) throw new Reject([{ path, code: 'bad-range' }]);
      return { ref, type: ref.type!, value: lines.slice(start - 1, end).join('') };
    }
    throw new Reject([{ path, code: 'bad-range' }]);
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
    if (name === 'run_code' && /\bawait\b/.test(String(args.code ?? ''))) {
      if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
      if (this.actions >= 40 || this.toolCalls >= 128)
        return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
      this.actions++; this.toolCalls++; this.lam.steps++;
      try {
        if (args.engine !== 'typescript-host') throw new Reject([{ path: 'engine', code: 'bad-action', expected: 'typescript-host' }]);
        const expression = String(args.code ?? '');
        const result = await this.runtime.evalForAsync(this.lam, `(async () => (${expression}))()`, 'eval',
          { instructions: this.lam.body, args: jsView(this.lam.args as Value),
            return: jsView(this.lam.return), let: jsView(this.lam.let as Value) }, false);
        return this.record(name, args, { kind: 'ok', text: JSON.stringify(result.result), value: result.result as Value });
      } catch (error) {
        if (error instanceof Reject) return this.record(name, args, rejected(error));
        return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
      }
    }
    if (name === 'run') {
      if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
      if (this.actions >= 40 || this.toolCalls >= 128) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
      this.actions++; this.toolCalls++; this.lam.steps++;
      try {
        const paths = Array.isArray(args.paths) ? args.paths : [args.paths];
        const outcomes = [];
        for (const path of paths) {
          const ref = this.resolve(String(path));
          if (ref.deny) throw new Reject([{ path: ref.path, code: ref.deny }]);
          const value = ref.get();
          if (pending(value)) {
            const missing = unboundParts(value, ref.env, ref.path);
            if (missing.length) return this.record(name, args, { kind: 'refused',
              text: missing.map(d => `${d.path}: ${d.code}`).join('\n'), codes: [...new Set(missing.map(d => d.code))] });
          }
          outcomes.push(await this.runtime.trigger(ref));
        }
        return this.record(name, args, { kind: outcomes.length === 1 ? outcomes[0]!.kind : 'ok',
          text: outcomes.map(o => `${o.path}: ${o.kind}  ${o.detail}`).join('\n'), value: outcomes.length === 1 ? outcomes[0]!.value : undefined });
      } catch (error) {
        if (error instanceof Reject) return this.record(name, args, rejected(error));
        return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
      }
    }
    if (name !== 'call') return this.apply(name, args);
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    if (this.actions >= 40 || this.toolCalls >= 128) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.actions++; this.toolCalls++; this.lam.steps++;
    if (args.done !== undefined) try { this.doneRange(args.done); }
    catch (error) { if (error instanceof Reject) return this.record(name, args, rejected(error)); throw error; }
    let newLocal: string | undefined;
    try {
      const label = String(args.function ?? '');
      const copyName = /^let\/([A-Za-z_][A-Za-z0-9_]*)$/.exec(label)?.[1];
      const template = copyName ? this.lam.let[copyName] : undefined;
      const original = copyName ? this.lam.fnCopies[copyName] as Record<string, unknown> | undefined : undefined;
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
      const typeText = `Lambda<{ ${params} }, ${String(definition.returns)}>`;
      const kind = Object.hasOwn(definition, 'code') ? 'code' : 'instructions';
      const inputPaths = (args.inputs ?? {}) as Record<string, string>;
      const values: Record<string, unknown> = { ...args.values as Record<string, unknown> ?? {} };
      const named = Object.fromEntries(Object.entries(definition.types as Record<string, string> ?? {})
        .map(([name, text]) => [name, parseType(text)]));
      const callEnv = this.env.child(named);
      const declared = Object.fromEntries(Object.entries(signature).map(([name, text]) =>
        [name.replace(/\?$/, ''), parseType(text)]));
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
        init === undefined ? [required.find(n => !(n in values))] : ['acc', 'item'];
      if (required.some(name => !(name in values) && !later.includes(name)))
        throw new Reject([{ path: 'inputs', code: 'bad-call', expected: required.join(', ') }]);
      if (overRef && (overRef.env.resolve(overRef.type!).kind !== 'list' || !Array.isArray(over)))
        throw new Reject([{ path: overRef.path, code: 'type-does-not-fit-slot', expected: 'a list' }]);
      if (overRef && overRef.type) {
        const itemType = signature[init === undefined ? later[0]! : 'item'];
        if (itemType && !fitsType(overRef.type, parseType(`${itemType}[]`), callEnv))
          throw new Reject([{ path: overRef.path, code: 'type-does-not-fit-slot', expected: `${itemType}[]` }]);
      }
      const leaf = { type: typeText, [kind]: definition[kind],
        engine: definition.engine ?? 'typescript-host', args: values, types: definition.types ?? {},
        effects: definition.effects ?? [], codebase: definition.codebase ?? {}, function: functionName };
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
          step: { $lambda: leaf }, check: { $lambda: { type: `Lambda<{ ${checkName}: ${checkArgs[checkName]} }, ${check.returns}>`,
            [checkKind]: check[checkKind], engine: check.engine ?? 'typescript-host', function: String(args.until) } } } };
        destination = stateType;
      } else if (over !== undefined && init !== undefined) {
        if (!('acc' in signature) || !('item' in signature)) throw new Reject([{ path: 'inputs', code: 'bad-call' }]);
        raw = { $fold: { type: `Fold<${signature.item}, ${signature.acc}>`, over, init, step: { $lambda: leaf } } };
        destination = signature.acc!;
      } else if (over !== undefined) {
        const itemName = later[0];
        if (!itemName) throw new Reject([{ path: 'inputs', code: 'bad-call' }]);
        raw = { $map: { type: `Map<${signature[itemName]}, ${definition.returns}>`, over,
          item_name: itemName, fn: { $lambda: leaf } } };
        destination = `${definition.returns}[]`;
      }
      const child = buildPending(raw, this.env);
      if (path.startsWith('let/')) {
        const local = path.slice(4);
        if (!this.lam.letTypes[local]) {
          if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(local))
            throw new Reject([{ path, code: 'no-such-path', expected: 'let/<name>' }]);
          if (Object.keys(this.lam.letTypes).length >= 16)
            throw new Reject([{ path, code: 'too-many-locals', expected: 'at most 16 locals' }]);
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
    else if (first === 'return') { container = this.lam as unknown as Record<string, Value>; type = this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseType('Null'); }
    else if (first === 'let') { container = this.lam.let; type = parseType('{}'); }
    else if (first === this.lam.kind) {
      if (parts.length) throw new Reject([{ path, code: 'no-such-path' }]);
      return itemRef(this.lam as unknown as Record<string, Value>, 'body', parseType('Text'), this.env, first);
    }
    else throw new Reject([{ path, code: 'no-such-path' }]);
    if (first === 'return' && parts.length === 0) return itemRef(container, 'return', type, this.env, path, deny);
    if (first === 'args' && parts.length === 0) return { path, type, env: this.env, deny, get: () => this.lam.args as Value,
      set: () => { throw new Reject([{ path, code: 'not-writable' }]); }, del: () => {} };
    const name = parts.shift();
    if (!name) throw new Reject([{ path, code: 'no-such-path' }]);
    if (first === 'let') {
      type = this.lam.letTypes[name] ?? parseType('Null');
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
      const type = ref.env.resolve(ref.type!);
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
      if (part === node.kind) return field('body', parseType('Text'));
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
      if (part === 'acc' || part === 'at') return field(part, part === 'acc' ? node.type.s : parseType('Num'), 'not-writable');
      if (part === 'current' && node.current !== null) return field(part, partType(node, 'step'));
    }
    if (node.nodeKind === 'iterate' && node.type.kind === 'iterate') {
      if (['init', 'step', 'check', 'max'].includes(part)) return field(part, partType(node, part));
      if (part === 'state' || part === 'iteration') return field(part, part === 'state' ? node.type.s : parseType('Num'), 'not-writable');
      if (part === 'current' && node.current !== null) return field(part, partType(node, 'step'));
    }
    throw new Reject([{ path: at, code: 'no-such-path' }]);
  }
}
