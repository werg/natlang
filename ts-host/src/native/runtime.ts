import { createHash } from 'node:crypto';
import { TypeScriptEnvironment, type HostEvent } from '../environment.js';
import { TypeEnv, fitsType, formatType, parseType, resultType, type Type } from './types.js';
import { MISSING, Reject, buildPending, cloneValue, coerce, dump, isPending, loadProgram,
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
  readonly options: { maxEpisodes: number; maxDepth: number; runId: string };
  readonly seedPolicy: { mode: 'compatibility' | 'derived' | 'backend'; root?: number };
  readonly environment: TypeScriptEnvironment;
  readonly agent?: NativeAgent;
  readonly capabilities: Record<string, (args: unknown[]) => unknown>;
  private releaseEffect: () => void;
  private stack: string[] = [];
  private depth = 0;
  episodesStarted = 0;
  private root?: { value: Value };
  private stream?: NativeStream;
  private streamCurrent: Value | undefined;

  constructor(options: { environment?: TypeScriptEnvironment; host?: object; agent?: NativeAgent;
    capabilities?: Record<string, (args: unknown[]) => unknown>; maxEpisodes?: number;
    maxDepth?: number; runId?: string; stream?: NativeStream;
    seedPolicy?: { mode: 'compatibility' | 'derived' | 'backend'; root?: number } } = {}) {
    this.options = { maxEpisodes: options.maxEpisodes ?? 256, maxDepth: options.maxDepth ?? 8,
      runId: options.runId ?? 'native-run' };
    this.seedPolicy = options.seedPolicy ?? { mode: 'backend' };
    this.trace = new NativeTraceRecorder({ run_id: this.options.runId, tool_schema: 'tools-v3',
      engines: ['typescript-host'], engine_contracts: { 'typescript-host': {
        environment_mode: options.environment?.mode ?? 'fresh', authority: 'shared-node-host', native_state_replayable: false } },
      seed_policy: this.seedPolicy, coverage: 'natlang-state-and-observed-host-effects' });
    this.agent = options.agent;
    this.capabilities = options.capabilities ?? {};
    this.environment = options.environment ?? new TypeScriptEnvironment({ mode: 'fresh', host: options.host });
    this.releaseEffect = this.environment.bindEffect((cap, fn, args) => this.effect(cap, fn, args));
    this.stream = options.stream;
  }

  close(): void { this.releaseEffect(); }

  private acting?: LambdaNode;
  private effect(cap: string, fn: string, args: unknown[]): unknown {
    const name = `${cap}.${fn}`, node = this.acting;
    if (!node || !node.effects.includes(name)) throw new Error('effect-undeclared');
    const entry = { seq: node.journal.length + 1, capability: name, function: fn, args_preview: q(args).slice(0, 80), status: 'pending' };
    node.journal.push(entry);
    this.events.push({ operation: 'effect.requested', capability: name, args });
    this.trace.emit('effect', { phase: 'requested', capability: name, sequence: entry.seq, args });
    try {
      const value = name === 'out.emit' ? (this.emitted.push(args[0]), null) : this.capabilities[name]?.(args);
      if (value === undefined && name !== 'out.emit') throw new Error('effect-unavailable');
      entry.status = 'ok'; this.events.push({ operation: 'effect.completed', capability: name, result: value });
      this.trace.emit('effect', { phase: 'completed', capability: name, sequence: entry.seq, result: value });
      return value;
    } catch (error) {
      entry.status = 'error'; this.events.push({ operation: 'effect.failed', capability: name });
      this.trace.emit('effect', { phase: 'failed', capability: name, sequence: entry.seq });
      throw error;
    }
  }

  async runRoot(root: Pending | Record<string, unknown>): Promise<{ outcome: NativeOutcome; value: Value; events: HostEvent[]; emitted: unknown[] }> {
    const source = isPending(root) ? root : loadProgram(root);
    if (!this.root) this.root = { value: source };
    else this.root.value = source;
    const box = this.root;
    const before = dump(box.value);
    if (!this.trace.events.some(event => event.kind === 'state')) this.trace.emit('state', { phase: 'initial', value: before });
    const ref: Ref = { path: '', env: new TypeEnv(), get: () => box.value,
      set: value => { box.value = value; }, del: () => { box.value = MISSING; } };
    const outcome = await this.trigger(ref);
    const after = dump(box.value);
    const delta = changes(before, after);
    if (delta.length) this.trace.emit('reduction', { phase: 'final', changes: delta });
    this.trace.emit('state', { phase: 'final', value: after, outcome: outcome.kind });
    return { outcome, value: box.value, events: this.events, emitted: this.emitted };
  }

  private done(ref: Ref, node: Pending, value: Value): NativeOutcome {
    node.status = 'done'; ref.set(value);
    this.trace.emit('node', { path: ref.path, transition: 'done', node_type: node.nodeKind });
    if (node.nodeKind === 'lambda') this.origins.set(ref.path, node);
    return { path: ref.path, kind: 'done', detail: q(dump(value)), value };
  }
  private quiesce(ref: Ref, node: Pending, detail: string): NativeOutcome {
    node.status = 'quiesced'; node.note = detail;
    this.trace.emit('node', { path: ref.path, transition: 'quiesced', node_type: node.nodeKind, detail });
    return { path: ref.path, kind: 'quiesced', detail };
  }

  async trigger(ref: Ref): Promise<NativeOutcome> {
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

  private crisp(ref: Ref, node: LambdaNode, env: TypeEnv): NativeOutcome {
    node.status = 'running';
    this.acting = node;
    this.trace.emit('eval', { phase: 'start', path: ref.path, mode: 'body', engine: node.engine,
      code: node.body, effectful: node.effects.length > 0 });
    try {
      if (node.type.kind !== 'lambda') throw new Error('invalid lambda type');
      const result = this.environment.execute({ code: node.body, body: true, path: ref.path,
        effectful: node.effects.length > 0, scope: { args: jsView(node.args as Value), return: jsView(node.return) } });
      this.events.push(...result.events);
      const value = coerce(result.result, node.type.returns, env, ref.path);
      const missing = problems(value, node.type.returns, env, ref.path);
      if (missing.holes.length) return this.quiesce(ref, node, 'returned value is incomplete');
      this.trace.emit('eval', { phase: 'completed', path: ref.path, value: dump(value) });
      return this.done(ref, node, value);
    } catch (error) {
      this.trace.emit('eval', { phase: 'failed', path: ref.path, error: error instanceof Error ? error.message : String(error) });
      return this.quiesce(ref, node, `code error: ${error instanceof Error ? error.message : String(error)}`);
    }
    finally { this.acting = undefined; }
  }

  private async episode(ref: Ref, node: LambdaNode): Promise<NativeOutcome> {
    if (this.depth >= this.options.maxDepth) return this.quiesce(ref, node, `run budget: episodes nested deeper than ${this.options.maxDepth}`);
    if (this.episodesStarted >= this.options.maxEpisodes) return this.quiesce(ref, node, `run budget: more than ${this.options.maxEpisodes} episodes`);
    const key = hash({ body: node.body, args: dump(node.args as Value), type: formatType(node.type) });
    if (this.stack.includes(key)) return this.quiesce(ref, node, 'identical to a lambda already being reduced above it');
    if (!this.agent) return this.quiesce(ref, node, 'no native model or agent driver supplied');
    node.status = 'running'; node.note = ''; node.attempts++;
    node.originalBody ??= node.body;
    this.episodesStarted++; this.depth++; this.stack.push(key);
    const callId = `${ref.path || '$root'}@${node.attempts}`;
    this.trace.emit('invocation', { phase: 'start', call_id: callId, path: ref.path, attempt: node.attempts });
    const session = new NativeSession(this, node, ref.env, ref.path);
    try {
      const note = await this.agent(session);
      if (session.completed) return this.done(ref, node, node.return);
      return this.quiesce(ref, node, String(note || 'budget exhausted'));
    } finally { this.trace.emit('invocation', { phase: 'end', call_id: callId }); this.stack.pop(); this.depth--; }
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
  private record(name: string, args: Record<string, unknown>, result: NativeResult): NativeResult {
    this.runtime.trace.emit('action', { surface: 'tools-v3', name, arguments: args,
      outcome: result.kind, codes: result.codes ?? [] });
    return result;
  }
  apply(name: string, args: Record<string, unknown>): NativeResult {
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    this.lam.steps++;
    return this.record(name, args, this.applyNow(name, args));
  }
  private applyNow(name: string, args: Record<string, unknown>): NativeResult {
    try {
      if (name === 'report_blocker' || name === 'report_error') {
        const message = String(args[name === 'report_blocker' ? 'missing' : 'message'] ?? '').trim();
        if (message.length < 8) throw new Reject([{ path: name, code: 'bad-action' }]);
        return { kind: 'blocked', text: message };
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
        const stated = parseType(String(args.type ?? ''));
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
        if (!old || value.split(old).length !== 2) throw new Reject([{ path: ref.path, code: value.includes(old) ? 'old-not-unique' : 'old-not-found' }]);
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
        const src = this.resolve(from), dst = this.resolve(to, true), original = src.get();
        if (original === MISSING || !src.type) throw new Reject([{ path: from, code: 'no-such-path' }]);
        if (dst.deny) throw new Reject([{ path: to, code: dst.deny }]);
        if (!dst.type || !fitsType(src.type, dst.type, dst.env))
          throw new Reject([{ path: to, code: 'type-does-not-fit-slot', expected: dst.type ? formatType(dst.type) : '' }]);
        const value = cloneValue(original);
        if (pending(value)) value.status = 'unreduced';
        this.checkEffects(value, to);
        dst.set(value);
        return { kind: 'ok', text: `ok   ${to}`, value };
      }
      if (name === 'run_code') {
        if (args.engine === undefined) throw new Reject([{ path: 'engine', code: 'bad-action', expected: 'an explicit available engine' }]);
        const engine = String(args.engine);
        if (engine !== 'typescript-host') return { kind: 'error', text: `engine ${engine} unavailable` };
        const result = this.runtime.environment.execute({ code: String(args.code ?? ''), body: false, path: 'eval',
          effectful: this.lam.effects.length > 0, scope: { instructions: this.lam.body,
            args: jsView(this.lam.args as Value), return: jsView(this.lam.return), let: jsView(this.lam.let as Value) } });
        this.runtime.events.push(...result.events);
        return { kind: 'ok', text: JSON.stringify(result.result), value: result.result as Value };
      }
      throw new Reject([{ path: name, code: 'bad-action' }]);
    } catch (error) {
      if (error instanceof Reject) return { kind: 'rejected', text: error.message, codes: error.diagnostics.map(d => d.code) };
      return { kind: 'error', text: error instanceof Error ? error.message : String(error) };
    }
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
    if (name === 'run') {
      try {
        const paths = Array.isArray(args.paths) ? args.paths : [args.paths];
        const outcomes = [];
        for (const path of paths) {
          const ref = this.resolve(String(path));
          if (ref.deny) throw new Reject([{ path: ref.path, code: ref.deny }]);
          outcomes.push(await this.runtime.trigger(ref));
        }
        return this.record(name, args, { kind: outcomes.length === 1 ? outcomes[0]!.kind : 'ok',
          text: outcomes.map(o => `${o.path}: ${o.kind}  ${o.detail}`).join('\n'), value: outcomes.length === 1 ? outcomes[0]!.value : undefined });
      } catch (error) {
        if (error instanceof Reject) return this.record(name, args, { kind: 'rejected', text: error.message, codes: error.diagnostics.map(d => d.code) });
        return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
      }
    }
    if (name !== 'call') return this.apply(name, args);
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
      for (const [name, path] of Object.entries(inputPaths)) values[name] = cloneValue(this.resolve(path).get());
      const required = Object.keys(signature).filter(name => !name.endsWith('?')).map(name => name.replace(/\?$/, ''));
      const over = args.over === undefined ? undefined : this.resolve(String(args.over)).get();
      const init = args.init === undefined ? undefined : typeof args.init === 'string' ?
        (() => { try { return this.resolve(args.init as string).get(); } catch { return args.init; } })() : args.init;
      const later = over === undefined ? args.until === undefined ? [] : [required.find(n => !(n in values))] :
        init === undefined ? [required.find(n => !(n in values))] : ['acc', 'item'];
      if (required.some(name => !(name in values) && !later.includes(name)))
        throw new Reject([{ path: 'inputs', code: 'bad-call', expected: required.join(', ') }]);
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
        if (!this.lam.letTypes[local]) this.lam.letTypes[local] = parseType(destination);
      }
      const ref = this.resolve(path, true);
      if (ref.deny) throw new Reject([{ path, code: ref.deny }]);
      if (!ref.type || !fitsType(child.type, ref.type, this.env))
        throw new Reject([{ path, code: 'type-does-not-fit-slot', expected: ref.type ? formatType(ref.type) : '' }]);
      this.checkEffects(child, path); ref.set(child);
      const outcome = await this.runtime.trigger(ref);
      return this.record(name, args, { kind: outcome.kind, text: `${path}: ${outcome.kind}  ${outcome.detail}`, value: outcome.value });
    } catch (error) {
      if (error instanceof Reject) return this.record(name, args, { kind: 'rejected', text: error.message, codes: error.diagnostics.map(d => d.code) });
      return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
    }
  }

  private resolve(path: string, create = false): Ref {
    const parts = path.split('/').filter(Boolean);
    if (!parts.length) throw new Reject([{ path, code: 'no-such-path' }]);
    const first = parts.shift()!;
    let container: Record<string, Value>, type: Type, deny = '';
    if (first === 'args') { container = this.lam.args; type = this.lam.type.kind === 'lambda' ? this.lam.type.params : parseType('{}'); deny = 'not-writable'; }
    else if (first === 'return') { container = this.lam as unknown as Record<string, Value>; type = this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseType('Null'); }
    else if (first === 'let') { container = this.lam.let; type = parseType('{}'); }
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
