import { createHash } from 'node:crypto';
import { TypeScriptEnvironment, type HostEvent } from '../environment.js';
import { TypeEnv, fitsType, formatType, parseType, resultType, type Type } from './types.js';
import { MISSING, Reject, buildPending, cloneValue, coerce, dump, isPending, loadProgram,
  partType, problems, unboundParts, type LambdaNode, type Pending, type Value } from './values.js';

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
  readonly emitted: unknown[] = [];
  readonly origins = new Map<string, LambdaNode>();
  readonly options: { maxEpisodes: number; maxDepth: number; runId: string };
  readonly environment: TypeScriptEnvironment;
  readonly agent?: NativeAgent;
  readonly capabilities: Record<string, (args: unknown[]) => unknown>;
  private stack: string[] = [];
  private depth = 0;
  episodesStarted = 0;
  private root?: { value: Value };
  private stream?: NativeStream;

  constructor(options: { environment?: TypeScriptEnvironment; host?: object; agent?: NativeAgent;
    capabilities?: Record<string, (args: unknown[]) => unknown>; maxEpisodes?: number;
    maxDepth?: number; runId?: string; stream?: NativeStream } = {}) {
    this.options = { maxEpisodes: options.maxEpisodes ?? 256, maxDepth: options.maxDepth ?? 8,
      runId: options.runId ?? 'native-run' };
    this.agent = options.agent;
    this.capabilities = options.capabilities ?? {};
    this.environment = options.environment ?? new TypeScriptEnvironment({ mode: 'fresh', host: options.host,
      observe: event => this.events.push(event), effect: (cap, fn, args) => this.effect(cap, fn, args) });
    this.stream = options.stream;
  }

  private acting?: LambdaNode;
  private effect(cap: string, fn: string, args: unknown[]): unknown {
    const name = `${cap}.${fn}`, node = this.acting;
    if (!node || !node.effects.includes(name)) throw new Error('effect-undeclared');
    const entry = { seq: node.journal.length + 1, capability: name, function: fn, args_preview: q(args).slice(0, 80), status: 'pending' };
    node.journal.push(entry);
    this.events.push({ operation: 'effect.requested', capability: name, args });
    try {
      const value = name === 'out.emit' ? (this.emitted.push(args[0]), null) : this.capabilities[name]?.(args);
      if (value === undefined && name !== 'out.emit') throw new Error('effect-unavailable');
      entry.status = 'ok'; this.events.push({ operation: 'effect.completed', capability: name, result: value });
      return value;
    } catch (error) {
      entry.status = 'error'; this.events.push({ operation: 'effect.failed', capability: name });
      throw error;
    }
  }

  async runRoot(root: Pending | Record<string, unknown>): Promise<{ outcome: NativeOutcome; value: Value; events: HostEvent[]; emitted: unknown[] }> {
    const source = isPending(root) ? root : loadProgram(root);
    if (!this.root) this.root = { value: source };
    else this.root.value = source;
    const box = this.root;
    const ref: Ref = { path: '', env: new TypeEnv(), get: () => box.value,
      set: value => { box.value = value; }, del: () => { box.value = MISSING; } };
    const outcome = await this.trigger(ref);
    return { outcome, value: box.value, events: this.events, emitted: this.emitted };
  }

  private done(ref: Ref, node: Pending, value: Value): NativeOutcome {
    node.status = 'done'; ref.set(value);
    if (node.nodeKind === 'lambda') this.origins.set(ref.path, node);
    return { path: ref.path, kind: 'done', detail: q(dump(value)), value };
  }
  private quiesce(ref: Ref, node: Pending, detail: string): NativeOutcome {
    node.status = 'quiesced'; node.note = detail;
    return { path: ref.path, kind: 'quiesced', detail };
  }

  private async trigger(ref: Ref): Promise<NativeOutcome> {
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
    const unbound = unboundParts(node, ref.env, ref.path);
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
    try {
      if (node.type.kind !== 'lambda') throw new Error('invalid lambda type');
      const result = this.environment.execute({ code: node.body, body: true, path: ref.path,
        effectful: node.effects.length > 0, scope: { args: jsView(node.args as Value), return: jsView(node.return) } });
      const value = coerce(result.result, node.type.returns, env, ref.path);
      const missing = problems(value, node.type.returns, env, ref.path);
      if (missing.holes.length) return this.quiesce(ref, node, 'returned value is incomplete');
      return this.done(ref, node, value);
    } catch (error) { return this.quiesce(ref, node, `code error: ${error instanceof Error ? error.message : String(error)}`); }
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
    const session = new NativeSession(this, node, ref.env);
    try {
      const note = await this.agent(session);
      if (session.completed) return this.done(ref, node, node.return);
      return this.quiesce(ref, node, String(note || 'budget exhausted'));
    } finally { this.stack.pop(); this.depth--; }
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
        const polled = await this.stream.poll();
        if (polled.kind === 'empty') { node.status = 'waiting'; node.note = 'waiting for stream input'; return { path: ref.path, kind: 'waiting', detail: node.note }; }
        if (polled.kind === 'closed') return this.done(ref, node, node.acc);
        if (polled.kind === 'failed') return this.quiesce(ref, node, `stream failed: ${polled.detail}`);
        item = coerce(polled.value, node.type.a, env, `${ref.path}/over/${node.at}`);
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
  constructor(readonly runtime: NativeRuntime, readonly lam: LambdaNode, readonly outerEnv: TypeEnv) {
    this.env = outerEnv.child(lam.types);
  }
  finish(): boolean {
    if (this.lam.return === MISSING || this.lam.type.kind !== 'lambda') return false;
    const p = problems(this.lam.return, this.lam.type.returns, this.env, 'return');
    if (p.holes.length || p.pending.length) return false;
    this.completed = true; this.lam.body = ''; return true;
  }
  apply(name: string, args: Record<string, unknown>): NativeResult {
    try {
      if (name === 'report_blocker' || name === 'report_error') {
        const message = String(args[name === 'report_blocker' ? 'missing' : 'message'] ?? '').trim();
        if (message.length < 8) throw new Reject([{ path: name, code: 'bad-action' }]);
        return { kind: 'blocked', text: message };
      }
      if (name === 'write') {
        const path = String(args.path ?? '');
        if (path.startsWith('args/')) throw new Reject([{ path, code: 'not-writable' }]);
        if (path.startsWith('let/')) {
          const local = path.split('/')[1]!;
          if (!this.lam.letTypes[local]) this.lam.letTypes[local] = parseType(String(args.type));
        }
        const ref = this.resolve(path, true);
        const raw = args.source ? this.resolve(String(args.source)).get() : args.value;
        if (raw === undefined) throw new Reject([{ path, code: 'bad-action', expected: 'a value or source' }]);
        let value: Value;
        try { value = coerce(raw, ref.type!, ref.env, path); }
        catch (first) {
          if (typeof raw !== 'string') throw first;
          try { value = coerce(JSON.parse(raw), ref.type!, ref.env, path); }
          catch { throw first; }
        }
        ref.set(value);
        return { kind: 'ok', text: `ok   ${path}`, value };
      }
      if (name === 'read') {
        const ref = this.resolve(String(args.path ?? ''));
        const value = ref.get();
        return { kind: 'ok', text: value === MISSING ? `${ref.path}: not supplied (missing value; not empty text)` :
          typeof value === 'string' ? value : JSON.stringify(dump(value), null, 1), value };
      }
      if (name === 'edit') {
        const ref = this.resolve(String(args.path ?? ''));
        const value = ref.get(), old = String(args.old ?? ''), replacement = String(args.new ?? '');
        if (typeof value !== 'string') throw new Reject([{ path: ref.path, code: 'type-mismatch', expected: 'a text' }]);
        if (!old || value.split(old).length !== 2) throw new Reject([{ path: ref.path, code: value.includes(old) ? 'old-not-unique' : 'old-not-found' }]);
        ref.set(value.replace(old, replacement));
        return { kind: 'ok', text: `ok   ${ref.path}` };
      }
      if (name === 'run_code') {
        const engine = String(args.engine ?? 'typescript-host');
        if (engine !== 'typescript-host') return { kind: 'error', text: `engine ${engine} unavailable` };
        const result = this.runtime.environment.execute({ code: String(args.code ?? ''), body: false, path: 'eval',
          effectful: this.lam.effects.length > 0, scope: { instructions: this.lam.body,
            args: jsView(this.lam.args as Value), return: jsView(this.lam.return), let: jsView(this.lam.let as Value) } });
        return { kind: 'ok', text: JSON.stringify(result.result), value: result.result as Value };
      }
      throw new Reject([{ path: name, code: 'bad-action' }]);
    } catch (error) {
      if (error instanceof Reject) return { kind: 'rejected', text: error.message, codes: error.diagnostics.map(d => d.code) };
      return { kind: 'error', text: error instanceof Error ? error.message : String(error) };
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
      ref = { path: `${parent.path}/${part}`, type: childType, env: parent.env, deny,
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
}
