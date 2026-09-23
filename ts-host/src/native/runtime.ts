/**
 * The natural-language function interpreter: runs one invocation of a `.nl` or inline `nl`
 * function as a model-driven session over a persistent TypeScript scope.
 *
 * Portable data reaches eval as a frozen snapshot; live values (host objects, functions, folder
 * handles, captured bindings, callables, services) arrive by reference through `__live`.
 */
import { EvalFailure, type EvalEnvironment, type HostEvent } from './evaluator.js';
import { TypeEnv, formatType, parseType, type Type } from './types.js';
import { MISSING, Reject, coerce, dump, dumpState, isLive, isPending, liveLabel, problems, unboundParts,
  type LambdaNode, type Value } from './values.js';
import { changes, NativeTraceRecorder } from './trace.js';
import { FileHandle, Folder, FolderHandle, editTextContent } from './scoped-fs.js';
import { compileScopeSnippet, SCOPE_RUNTIME_PRELUDE } from '../scope-compiler.js';
import { livePreview } from './agent.js';
import type { InlineLambdaPlan, NatlangDiagnostic } from '../compiler/inline.js';
import { runInFrame, type Frame } from '../runtime/context.js';
import { PATH_ONLY, parseModule, parseNatlang, type ItemRecord } from '../runtime/loader.js';

/** Services the invocation kernel provides to an interpreter run. */
export type NativeRuntimeHooks = {
  /** Callable tree for a codebase record tree (functions with child attributes and `iterateOn`). */
  callables(codebase: Record<string, unknown>, session: NativeSession): Record<string, unknown>;
  /** Create an inline natlang callable from an eval plan. */
  inline(session: NativeSession, plan: InlineLambdaPlan, values: unknown[], accessors: Record<string, unknown>): unknown;
  iterateOn(session: NativeSession, step: unknown, initial: unknown, ...args: unknown[]): unknown;
  finite(source: unknown): unknown;
  guard(id: string, fn: () => unknown): unknown;
  /** Type-checked analysis of `nl` in eval snippets. */
  analyze(session: NativeSession, source: string): { plans: InlineLambdaPlan[]; diagnostics: NatlangDiagnostic[] };
};
export type NativeOutcome = { kind: 'done' | 'quiesced'; detail: string; value?: Value };
export type NativeResult = { kind: string; text: string; value?: Value; codes?: string[] };
export type NativeAgent = (session: NativeSession) => Promise<string | void> | string | void;
export type NativeRuntimeOptions = { environment: EvalEnvironment; hooks: NativeRuntimeHooks; agent?: NativeAgent;
  maxActions?: number; maxToolCalls?: number;
  runId?: string; signal?: AbortSignal; timeoutMs?: number;
  sourceRevision?: string; parentCallId?: string;
  /** Task frame of this invocation (task, caller chain, parent call). */
  frame?: Frame;
  /** Extra manifest fields recorded for this invocation. */
  manifest?: Record<string, unknown>;
  /** Host services (already wrapped for effect recording), injected into eval as named bindings. */
  services?: Record<string, object>;
  sharedEpisodeBudget?: { limit?: number; used: number };
  seedPolicy?: { mode: 'compatibility' | 'derived' | 'backend'; root?: number } };

type Ref = { path: string; type?: Type; env: TypeEnv; deny?: string;
  get(): Value; set(value: Value): void; del(): void };

function oneLine(value: unknown): string {
  if (isLive(value)) return livePreview(value as object);
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
const DIAGNOSTIC_HINTS: Record<string, string> = {
  'type-mismatch': 'Pass the value itself with the type shown as expected, not wrapped in another object: for boolean use `true`, for number use `42.5`, for string use text, and for a record use an object with exactly its fields.',
  'unknown-field': 'Use one of the fields listed as expected.',
  'no-such-path': 'Use a variable or field that exists in the scope.',
  'capture-conflict': 'Another caller changed that captured variable; run the eval again with its current value.',
};
function rejected(error: Reject): NativeResult {
  const hint = error.diagnostics.map(diagnostic => DIAGNOSTIC_HINTS[diagnostic.code]).find(Boolean);
  return { kind: 'rejected', text: `rejected\n${error.message}${hint ? `\nhint: ${hint}` : ''}`,
    codes: error.diagnostics.map(diagnostic => diagnostic.code) };
}
const markable = (line: string) => { const text = line.trim(); return !!text && !text.startsWith('#') && !text.startsWith('function '); };
export function programListing(body: string, marks: Record<number, string>, window = 3): string {
  const lines = body.replace(/^\n+|\n+$/g, '').split('\n');
  const width = String(lines.length).length, output: string[] = [], closed: number[] = [];
  const flush = () => {
    if (!closed.length) return;
    const kinds = new Set(closed.map(number => marks[number]).filter(Boolean));
    const box = kinds.size === 1 && kinds.has('done') ? '[x]' : kinds.size === 1 && kinds.has('skipped') ? '[-]' : '[x/-]';
    const first = closed[0]!, last = closed.at(-1)!;
    output.push(`${String(first === last ? first : `${first}-${last}`).padStart(width)} ${box}`);
    closed.length = 0;
  };
  for (const [index, raw] of lines.entries()) {
    const number = index + 1, text = raw.trimEnd();
    if (Object.hasOwn(marks, number) || (!markable(text) && closed.length)) { closed.push(number); continue; }
    flush();
    const box = markable(text) ? marks[number] === 'done' ? '[x]' : marks[number] === 'skipped' ? '[-]' : '[ ]' : '   ';
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
export function pendingProgramLines(body: string, marks: Record<number, string>): number[] {
  return body.replace(/^\n+|\n+$/g, '').split('\n').flatMap((raw, index) =>
    markable(raw) && !Object.hasOwn(marks, index + 1) ? [index + 1] : []);
}
const isHandle = (value: unknown) => value instanceof Folder || value instanceof FolderHandle || value instanceof FileHandle;
/** True when a value is, or contains, something that must be passed by reference. */
function containsLive(value: unknown, seen = new Set<object>()): boolean {
  if (isLive(value) || isHandle(value)) return true;
  if (!value || typeof value !== 'object' || seen.has(value)) return false;
  seen.add(value);
  return Array.isArray(value) ? value.some(item => containsLive(item, seen)) : Object.values(value).some(item => containsLive(item, seen));
}
/** Split scope values into portable snapshot data and live references. */
function splitScope(values: Record<string, Value>): { portable: Record<string, unknown>; live: Record<string, unknown> } {
  const portable: Record<string, unknown> = {}, live: Record<string, unknown> = {};
  for (const [name, value] of Object.entries(values)) {
    if (value === MISSING || isPending(value)) continue;
    if (containsLive(value)) live[name] = value;
    else portable[name] = dump(value);
  }
  return { portable, live };
}
/** Natlang type text for a live value whose type was not declared. */
function liveTypeText(value: object): string {
  if (typeof value === 'function') return `Live<${JSON.stringify(`function ${(value as Function).name || ''}`.trim())}, "function", "">`;
  const tag = Object.prototype.toString.call(value).slice(8, -1), label = liveLabel(value);
  return tag !== 'Object' ? `Live<${JSON.stringify(tag)}, "tag", ${JSON.stringify(tag)}>` :
    `Live<${JSON.stringify(label)}, "class", ${JSON.stringify(label)}>`;
}
/** Rebuild plain data in this realm (eval values come from the sandbox realm); live values keep identity. */
function hostCopy(value: unknown, seen = new Map<object, unknown>()): unknown {
  if (!value || typeof value !== 'object' || isLive(value) || isHandle(value)) return value;
  if (seen.has(value)) return seen.get(value);
  if (Array.isArray(value)) { const out: unknown[] = []; seen.set(value, out); for (const item of value) out.push(hostCopy(item, seen)); return out; }
  const out: Record<string, unknown> = {}; seen.set(value, out);
  for (const [key, item] of Object.entries(value)) out[key] = hostCopy(item, seen);
  return out;
}
const itemRef = (container: Record<string, Value>, key: string, type: Type, env: TypeEnv, path: string, deny = ''): Ref =>
  ({ path, type, env, deny,
    get: () => Object.hasOwn(container, key) ? container[key]! : MISSING,
    set: value => { container[key] = value; },
    del: () => { delete container[key]; } });

/** Find a codebase item by dotted path; module exports resolve to their module's source. */
export function findCodebaseItem(codebase: Record<string, unknown>, name: string):
  { path: string[]; record: ItemRecord } | undefined {
  const parts = name.trim().replace(/\//g, '.').split('.').filter(Boolean);
  const walk = (level: Record<string, ItemRecord>, index: number, trail: string[]): { path: string[]; record: ItemRecord } | undefined => {
    const item = level[parts[index]!];
    if (!item) return;
    const path = [...trail, parts[index]!];
    if (index === parts.length - 1) return { path, record: item };
    if (item.kind === 'module' && Object.hasOwn(item.exports, parts[index + 1]!) && index + 1 === parts.length - 1) return { path, record: item };
    return walk(item.codebase, index + 1, path);
  };
  const direct = parts.length ? walk(codebase as Record<string, ItemRecord>, 0, []) : undefined;
  if (direct || parts.length !== 1) return direct;
  const matches: { path: string[]; record: ItemRecord }[] = [];
  const search = (level: Record<string, ItemRecord>, trail: string[]) => {
    for (const [key, item] of Object.entries(level)) {
      if (key === parts[0]) matches.push({ path: [...trail, key], record: item });
      search(item.codebase, [...trail, key]);
    }
  };
  search(codebase as Record<string, ItemRecord>, []);
  return matches.length === 1 ? matches[0] : undefined;
}
function listCodebase(codebase: Record<string, unknown>, prefix = ''): string[] {
  return Object.entries(codebase as Record<string, ItemRecord>).flatMap(([name, item]) =>
    [`${prefix}${name}`, ...listCodebase(item.codebase, `${prefix}${name}.`)]);
}
function replaceCodebaseItem(codebase: Record<string, unknown>, path: string[], record: ItemRecord): Record<string, unknown> {
  const [head, ...rest] = path;
  const current = (codebase as Record<string, ItemRecord>)[head!]!;
  return { ...codebase, [head!]: rest.length ? { ...current, codebase: replaceCodebaseItem(current.codebase, rest, record) } : record };
}
/** Declared return type of a callable path in a codebase, for typing eval locals. */
function returnTypeOf(codebase: Record<string, unknown>, name: string): string | undefined {
  const found = findCodebaseItem(codebase, name);
  if (!found || found.path.join('.') !== name.replace(/\//g, '.') && found.record.kind !== 'module') return;
  const record = found.record;
  if (record.kind === 'natlang') return record.returns;
  if (record.kind === 'module') {
    const exportName = name.split('.').at(-1)!;
    const spec = record.exports[exportName === record.name ? 'default' : exportName] ?? record.exports.default;
    return spec?.kind === 'function' ? spec.returns : undefined;
  }
  return;
}

export class NativeRuntime {
  readonly events: HostEvent[] = [];
  readonly trace: NativeTraceRecorder;
  readonly options: { maxActions?: number; maxToolCalls?: number; runId: string };
  readonly seedPolicy: { mode: 'compatibility' | 'derived' | 'backend'; root?: number };
  readonly environment: EvalEnvironment;
  readonly agent?: NativeAgent;
  readonly episodeBudget: { limit?: number; used: number };
  readonly frame?: Frame;
  readonly hooks: NativeRuntimeHooks;
  readonly services: Record<string, object>;
  currentCallId?: string;
  private root?: LambdaNode;
  private lastObserved?: unknown;
  private readonly signal?: AbortSignal;
  private readonly deadline?: number;

  constructor(options: NativeRuntimeOptions) {
    this.options = { maxActions: options.maxActions, maxToolCalls: options.maxToolCalls, runId: options.runId ?? 'native-run' };
    for (const [name, value] of Object.entries({ maxActions: this.options.maxActions, maxToolCalls: this.options.maxToolCalls }))
      if (value !== undefined && (!Number.isInteger(value) || value < 1)) throw new RangeError(`${name} must be a positive integer`);
    this.episodeBudget = options.sharedEpisodeBudget ?? { used: 0 };
    this.seedPolicy = options.seedPolicy ?? { mode: 'compatibility' };
    if (this.seedPolicy.mode === 'derived' && !Number.isInteger(this.seedPolicy.root))
      throw new TypeError('derived seed policy requires an integer root');
    this.trace = new NativeTraceRecorder({ run_id: this.options.runId, tool_schema: 'scope-eval-v2',
      ...(options.sourceRevision ? { source_revision: options.sourceRevision } : {}),
      ...(options.parentCallId ? { parent_call_id: options.parentCallId } : {}),
      environment: { mode: options.environment.mode, authority: options.environment.authority, native_state_replayable: false },
      seed_policy: this.seedPolicy, coverage: 'natlang-state-and-observed-host-effects', ...(options.manifest ?? {}) });
    this.frame = options.frame;
    this.hooks = options.hooks;
    this.services = options.services ?? {};
    this.agent = options.agent;
    this.environment = options.environment;
    this.signal = options.signal;
    if (options.timeoutMs !== undefined) {
      if (!Number.isFinite(options.timeoutMs) || options.timeoutMs < 1) throw new RangeError('timeoutMs must be positive');
      this.deadline = Date.now() + options.timeoutMs;
    }
  }

  checkInterruption(): void {
    if (this.signal?.aborted) throw new Error('natlang run aborted; external effects may have occurred');
    if (this.deadline !== undefined && Date.now() >= this.deadline)
      throw new Error('natlang run timed out; external effects may have occurred');
  }

  async evaluate(node: LambdaNode, code: string, scope: Record<string, unknown>, live: Record<string, unknown>) {
    this.checkInterruption();
    try {
      const request = { code, body: true, path: 'eval', scope, live };
      const result = await (this.frame ? runInFrame(this.frame, () => this.environment.executeAsync(request)) :
        this.environment.executeAsync(request));
      this.recordHostEvents(result.events);
      this.checkInterruption();
      return result;
    } catch (error) {
      if (error instanceof EvalFailure) this.recordHostEvents(error.events);
      throw error;
    }
    void node;
  }

  private recordHostEvents(events: HostEvent[]): void {
    this.events.push(...events);
    for (const event of events) if (event.operation !== 'typescript.eval') this.trace.emit('host', { event });
  }

  /** Run one invocation to completion or quiescence. */
  async run(node: LambdaNode): Promise<{ outcome: NativeOutcome; value: Value }> {
    this.checkInterruption();
    this.root = node;
    const before = dumpState(node);
    this.trace.emit('state', { phase: 'initial', value: before });
    this.lastObserved = before;
    const env = new TypeEnv().child(node.types);
    env.classes = node.hostClasses;
    const outcome = await this.episode(node, env);
    this.observeState('final', outcome.kind);
    return { outcome, value: outcome.kind === 'done' ? node.return : MISSING };
  }

  observeState(phase: string, outcome?: string): void {
    if (!this.root) return;
    const value = dumpState(this.root);
    const delta = changes(this.lastObserved, value);
    if (delta.length) this.trace.emit('reduction', { phase, changes: delta });
    this.trace.emit('state', { phase, value, ...(outcome ? { outcome } : {}) });
    this.lastObserved = value;
  }

  private quiesce(node: LambdaNode, detail: string): NativeOutcome {
    node.status = 'quiesced'; node.note = detail;
    this.trace.emit('node', { transition: 'quiesced', detail });
    if (node.projectTransaction?.open) {
      node.projectTransaction.abort();
      this.trace.emit('folder', { call_id: this.currentCallId ?? null, phase: 'discarded', mode: node.reducerMode, reason: detail });
    }
    return { kind: 'quiesced', detail };
  }

  private async episode(node: LambdaNode, env: TypeEnv): Promise<NativeOutcome> {
    const unbound = unboundParts(node, env, '');
    if (unbound.length) return this.quiesce(node, `unbound: ${unbound.map(d => d.path).join(', ')}`);
    if (this.episodeBudget.limit !== undefined && this.episodeBudget.used >= this.episodeBudget.limit)
      return this.quiesce(node, `run budget: more than ${this.episodeBudget.limit} episodes`);
    if (!this.agent) return this.quiesce(node, 'no model or agent driver supplied');
    node.status = 'running'; node.note = ''; node.attempts++;
    node.originalBody ??= node.body;
    this.episodeBudget.used++;
    const callId = this.options.runId;
    this.currentCallId = callId;
    this.trace.emit('invocation', { phase: 'start', call_id: callId, attempt: node.attempts });
    const session = new NativeSession(this, node, env);
    let note: string | void;
    try {
      note = await this.agent(session);
      this.checkInterruption();
    } catch (error) {
      this.quiesce(node, `interpreter exception: ${error instanceof Error ? error.message : String(error)}`);
      throw error;
    } finally { this.trace.emit('invocation', { phase: 'end', call_id: callId }); }
    // Ending the interpreter turn is the completion signal: the runtime validates the typed result
    // and line marks, then commits any directory-reducer transaction.
    if (session.completed || session.finish()) {
      node.status = 'done';
      this.trace.emit('node', { transition: 'done' });
      return { kind: 'done', detail: oneLine(dump(node.return)), value: node.return };
    }
    return this.quiesce(node, String(note || 'budget exhausted'));
  }
}

export type ScopeFailureDebug = {
  version: 'natlang.scope_failure/1'; serial: number; kind: 'compile' | 'runtime' | 'boundary';
  message: string; code: string; scope: Record<string, unknown>; trace: Record<string, unknown>[];
  diagnostics: Record<string, unknown>[]; logs: string[]; stack?: string;
};

export class NativeSession {
  completed = false;
  actions = 0;
  toolCalls = 0;
  readonly surfaceName = 'scope-eval-v2';
  failureSerial = 0;
  failureDebug?: ScopeFailureDebug;
  private readonly scopeLocalMutability = new Map<string, boolean>();
  private readonly explainedNaturalFunctions = new Set<string>();
  private readonly originalSources = new Map<string, string>();
  private callableCache?: { codebase: Record<string, unknown>; tree: Record<string, unknown> };
  constructor(readonly runtime: NativeRuntime, readonly lam: LambdaNode, readonly env: TypeEnv) {}

  get failureBinding(): string | undefined {
    if (!this.failureDebug) return;
    const used = new Set([...Object.keys(this.lam.args), ...Object.keys(this.lam.let),
      ...Object.keys(this.lam.codebase), ...Object.keys(this.lam.captures ?? {}), ...Object.keys(this.runtime.services),
      'result', 'folder']);
    for (const name of ['debug', '__natlangDebug']) if (!used.has(name)) return name;
    let suffix = 2;
    while (used.has(`__natlangDebug${suffix}`)) suffix++;
    return `__natlangDebug${suffix}`;
  }

  private captureScopeFailure(kind: ScopeFailureDebug['kind'], code: string, scope: Record<string, unknown>,
    message: string, diagnostics: Record<string, unknown>[] = [], error?: unknown): string {
    const trace = this.runtime.trace.events.filter(event =>
      ['action', 'eval', 'effect', 'host', 'node', 'invocation', 'folder'].includes(event.kind)).slice(-24)
      .map(event => {
        const fields: [string, unknown][] = [];
        for (const key of ['seq', 'kind', 'phase', 'transition', 'operation', 'capability', 'status', 'outcome',
          'error', 'result_text', 'call_id']) {
          const value = event[key];
          if (typeof value === 'string') fields.push([key, value.slice(0, 1200)]);
          else if (typeof value === 'number' || typeof value === 'boolean') fields.push([key, value]);
        }
        return Object.fromEntries(fields);
      });
    const sourceStack = error instanceof EvalFailure ? error.debug.sourceStack : error instanceof Error ? error.stack : undefined;
    const logs = error instanceof EvalFailure ? error.debug.logs ?? [] : [];
    this.failureDebug = { version: 'natlang.scope_failure/1', serial: ++this.failureSerial,
      kind, message, code: code.slice(0, 16000), scope, trace, diagnostics,
      logs: logs.slice(0, 32).map(line => line.slice(0, 2000)),
      ...(sourceStack ? { stack: sourceStack.split('\n').slice(0, 12).join('\n').slice(0, 4000) } : {}) };
    this.runtime.trace.emit('scope_failure', { failure_kind: kind, serial: this.failureSerial,
      message: message.slice(0, 1200), diagnostic_count: diagnostics.length });
    return `\nDebug snapshot available as immutable ${this.failureBinding} in the next eval. ` +
      'Inspect its scope, diagnostics, logs, stack, and trace before revising the code. Host effects may already have happened.';
  }

  /** Complete the invocation if the typed result is set and every instruction line is closed. */
  finish(): boolean {
    if (this.lam.return === MISSING || this.lam.type.kind !== 'lambda') return false;
    if (pendingProgramLines(this.lam.originalBody ?? this.lam.body, this.lam.marks).length) return false;
    if (problems(this.lam.return, this.lam.type.returns, this.env, 'return').holes.length) return false;
    const tx = this.lam.projectTransaction;
    if (tx?.open) {
      const delta = tx.folder.diffSync();
      const selected = this.lam.reducerMode === 'apply' ? tx.commitSync(this.lam.commitInclude, this.lam.commitExclude) :
        (tx.abort(), delta);
      this.runtime.trace.emit('folder', { call_id: this.runtime.currentCallId ?? null,
        phase: this.lam.reducerMode === 'apply' ? 'installed' : 'discarded', mode: this.lam.reducerMode,
        changes: selected.changes.map(change => ({ path: change.path, kind: change.kind })) });
    }
    this.completed = true;
    return true;
  }

  private record(name: string, args: Record<string, unknown>, result: NativeResult): NativeResult {
    this.runtime.trace.emit('action', { call_id: this.runtime.currentCallId ?? null, surface: this.surfaceName, name,
      arguments: args, outcome: result.kind, result_text: result.text, diagnostics: result.codes ?? [] });
    this.runtime.observeState('after-action');
    return result;
  }
  private validateMark(start: unknown, end: unknown): void {
    const lines = (this.lam.originalBody ?? this.lam.body).replace(/^\n+|\n+$/g, '').split('\n');
    if (!Number.isInteger(start) || !Number.isInteger(end) || Number(start) < 1 || Number(end) < Number(start) || Number(end) > lines.length)
      throw new Reject([{ path: 'start', code: 'bad-range',
        expected: `line numbers between 1 and ${lines.length}, start <= end`, got: `${start}..${end}` }]);
  }
  private actionLimitReached(): boolean {
    return (this.runtime.options.maxActions !== undefined && this.actions >= this.runtime.options.maxActions) ||
      (this.runtime.options.maxToolCalls !== undefined && this.toolCalls >= this.runtime.options.maxToolCalls);
  }

  /** The session's callable tree, rebuilt when a codebase edit publishes a new revision. */
  callables(): Record<string, unknown> {
    if (this.callableCache?.codebase !== this.lam.codebase)
      this.callableCache = { codebase: this.lam.codebase, tree: this.runtime.hooks.callables(this.lam.codebase, this) };
    return this.callableCache.tree;
  }

  /** Apply one tool call. */
  async applyAsync(name: string, args: Record<string, unknown>): Promise<NativeResult> {
    this.runtime.checkInterruption();
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    const tools = ['eval', 'read_value', 'mark_lines', 'commit', 'report_blocker', 'report_error',
      'read_function', 'edit_function', 'diff_functions',
      'list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files'];
    if (!tools.includes(name))
      return this.record(name, args, rejected(new Reject([{ path: name, code: 'bad-action', expected: 'a scope-eval tool' }])));
    if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.toolCalls++;
    if (name !== 'mark_lines') { this.actions++; this.lam.steps++; }
    try {
      if (name === 'eval') return this.record(name, args, await this.evaluate(String(args.code ?? '')));
      if (['read_function', 'edit_function', 'diff_functions'].includes(name)) return this.record(name, args, this.functionTool(name, args));
      if (['list_files', 'search_files', 'read_file', 'write_file', 'edit_file', 'diff_files'].includes(name))
        return this.record(name, args, await this.fileTool(name, args));
      return this.record(name, args, this.scopeTool(name, args));
    } catch (error) {
      if (error instanceof Reject) return this.record(name, args, rejected(error));
      return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
    }
  }
  /** Synchronous tools (no eval). */
  apply(name: string, args: Record<string, unknown>): NativeResult {
    this.runtime.checkInterruption();
    if (this.completed) return this.record(name, args, { kind: 'error', text: 'the task has already finished' });
    if (!['read_value', 'mark_lines', 'commit', 'report_blocker', 'report_error', 'read_function', 'edit_function', 'diff_functions'].includes(name))
      return this.record(name, args, rejected(new Reject([{ path: name, code: 'bad-action', expected: 'a synchronous scope-eval tool' }])));
    if (this.actionLimitReached()) return this.record(name, args, { kind: 'budget', text: 'action or tool-call budget exhausted' });
    this.toolCalls++;
    if (name !== 'mark_lines') { this.actions++; this.lam.steps++; }
    try {
      return this.record(name, args, name.endsWith('_function') || name === 'diff_functions' ? this.functionTool(name, args) : this.scopeTool(name, args));
    } catch (error) {
      if (error instanceof Reject) return this.record(name, args, rejected(error));
      return this.record(name, args, { kind: 'error', text: error instanceof Error ? error.message : String(error) });
    }
  }

  private scopeTool(name: string, args: Record<string, unknown>): NativeResult {
    if (name === 'read_value') {
      const expression = String(args.expression ?? '').trim();
      const debugName = this.failureBinding;
      if (debugName && (expression === debugName || expression.startsWith(`${debugName}.`))) {
        let inspected: unknown = this.failureDebug;
        for (const field of expression.slice(debugName.length).split('.').filter(Boolean)) {
          if (!inspected || typeof inspected !== 'object' || !Object.hasOwn(inspected, field))
            throw new Reject([{ path: expression, code: 'no-such-path', expected: 'a debug snapshot field' }]);
          inspected = (inspected as Record<string, unknown>)[field];
        }
        const page = this.readValuePage(expression, inspected as Value, args.start, args.end);
        if (page) return page;
        if (args.start !== undefined || args.end !== undefined)
          throw new Reject([{ path: expression, code: 'bad-range', expected: 'a list, text, or record value' }]);
        return { kind: 'ok', text: typeof inspected === 'string' ? inspected : JSON.stringify(inspected), value: inspected as Value };
      }
      const ref = this.resolve(this.scopePath(expression));
      const value = ref.get();
      const page = this.readValuePage(ref.path, value, args.start, args.end);
      if (page) return page;
      if (args.start !== undefined || args.end !== undefined)
        throw new Reject([{ path: ref.path, code: 'bad-range', expected: 'a list, text, or record value' }]);
      if (isLive(value) || isHandle(value)) return { kind: 'ok', text: livePreview(value as object), value };
      return { kind: 'ok', text: value === MISSING ? `${expression}: not supplied (missing value; not empty text)` :
        typeof value === 'string' ? value : JSON.stringify(dump(value), null, 1), value };
    }
    if (name === 'commit') {
      if (this.lam.subtype !== 'directory-reducer' || !this.lam.projectTransaction)
        throw new Reject([{ path: 'commit', code: 'bad-action', expected: 'a running directory reducer' }]);
      const include = args.include, exclude = args.exclude;
      for (const [key, selectors] of [['include', include], ['exclude', exclude]] as const)
        if (selectors !== undefined && (!Array.isArray(selectors) || selectors.some(item =>
          typeof item !== 'string' || !item || item.startsWith('/'))))
          throw new Reject([{ path: key, code: 'bad-action', expected: 'relative glob patterns' }]);
      if (this.lam.type.kind !== 'lambda') throw new Reject([{ path: 'commit', code: 'bad-action', expected: 'a typed result' }]);
      let value: Value;
      try { value = coerce(args.value, this.lam.type.returns, this.env, 'return'); }
      catch (first) {
        if (typeof args.value !== 'string') throw first;
        try { value = coerce(JSON.parse(args.value), this.lam.type.returns, this.env, 'return'); } catch { throw first; }
      }
      this.lam.return = value;
      this.lam.commitInclude = include === undefined ? undefined : [...include as string[]];
      this.lam.commitExclude = exclude === undefined ? undefined : [...exclude as string[]];
      return { kind: 'ok', text: 'ok   return', value };
    }
    if (name === 'report_blocker' || name === 'report_error') {
      const message = String(args[name === 'report_blocker' ? 'missing' : 'message'] ?? '').trim();
      if (message.length < 8) throw new Reject([{ path: name === 'report_blocker' ? 'missing' : 'message',
        code: 'bad-action', expected: name === 'report_blocker' ? 'a sentence saying what is missing' : 'a sentence explaining the error' }]);
      return { kind: 'blocked', text: `${name === 'report_blocker' ? 'blocked' : 'error'}: ${message}` };
    }
    // mark_lines
    const start = args.start, end = args.end ?? args.start;
    this.validateMark(start, end);
    const lines = (this.lam.originalBody ?? this.lam.body).replace(/^\n+|\n+$/g, '').split('\n');
    for (let i = Number(start); i <= Number(end); i++)
      if (markable(lines[i - 1]!)) this.lam.marks[i] = args.skipped === true ? 'skipped' : 'done';
    return { kind: 'ok', text: `ok\n${programListing(this.lam.originalBody ?? this.lam.body, this.lam.marks)}` };
  }

  /** read_function, edit_function, diff_functions over the codebase record tree. */
  private functionTool(name: string, args: Record<string, unknown>): NativeResult {
    if (name === 'diff_functions') {
      const changed = [...this.originalSources.keys()].map(source => ({ function: source, kind: 'modified' }));
      return { kind: 'ok', text: JSON.stringify(changed, null, 2), value: changed as Value };
    }
    const requested = String(args.name ?? '');
    const found = findCodebaseItem(this.lam.codebase, requested);
    if (!found) throw new Reject([{ path: requested, code: 'no-such-function', expected: listCodebase(this.lam.codebase).join(', ') }]);
    const record = found.record;
    if (record.kind === 'namespace')
      throw new Reject([{ path: requested, code: 'no-such-function', expected: `an item inside ${requested}: ${Object.keys(record.codebase).join(', ')}` }]);
    if (name === 'read_function') {
      let explanation = '';
      if (record.kind === 'natlang' && !this.explainedNaturalFunctions.has(record.id)) {
        this.explainedNaturalFunctions.add(record.id);
        explanation = 'Natural-language function source: its frontmatter declares parameter and return types; the body contains instructions executed line by line.\n\n';
      }
      return { kind: 'ok', text: explanation + record.text, value: record.text };
    }
    const text = editTextContent(record.text, String(args.find ?? ''), String(args.replace_with ?? ''), args.fuzzy === true);
    const updated = record.kind === 'natlang' ? parseNatlang(record.source, text, record.types, PATH_ONLY) :
      parseModule(record.source, text, record.types, PATH_ONLY);
    updated.codebase = record.codebase;
    if (!this.originalSources.has(record.source)) this.originalSources.set(record.source, record.text);
    this.lam.codebase = replaceCodebaseItem(this.lam.codebase, found.path, updated);
    return { kind: 'ok', text: JSON.stringify({ function: found.path.join('.'), source: record.source, changed: true }),
      value: { function: found.path.join('.'), changed: true } };
  }

  /** File tools for directory reducers, over the private folder copy. */
  private async fileTool(name: string, args: Record<string, unknown>): Promise<NativeResult> {
    const path = String(args.path ?? '');
    if (path.startsWith('/') || path === '..' || path.startsWith('../') || path.includes('/../'))
      throw new Reject([{ path, code: 'no-such-path', expected: 'a relative path in the current folder' }]);
    const folder = this.lam.projectTransaction?.folder;
    if (!folder) throw new Reject([{ path, code: 'bad-action', expected: 'a directory reducer folder' }]);
    if (['read_file', 'write_file', 'edit_file'].includes(name) && !path)
      throw new Reject([{ path, code: 'no-such-path', expected: 'a file in the current folder' }]);
    let value: unknown;
    if (name === 'list_files') value = folder.listFiles(path, args.pattern === undefined ? undefined : String(args.pattern));
    else if (name === 'search_files') value = await folder.search(String(args.query ?? ''), path,
      args.pattern === undefined ? undefined : String(args.pattern), args.regex === true);
    else if (name === 'read_file') value = await folder.readText(path,
      args.start_line === undefined ? undefined : Number(args.start_line), args.end_line === undefined ? undefined : Number(args.end_line));
    else if (name === 'write_file') { folder.writeText(path, String(args.content ?? '')); value = { path, changed: true }; }
    else if (name === 'edit_file') value = await folder.editText(path, String(args.find ?? ''), String(args.replace_with ?? ''), args.fuzzy === true);
    else value = folder.diffSync(path);
    return { kind: 'ok', text: typeof value === 'string' ? value : JSON.stringify(value, null, 1), value: value as Value };
  }

  private inferScopeType(value: unknown): string {
    if (isLive(value)) return liveTypeText(value as object);
    if (value instanceof Folder || value instanceof FolderHandle) return 'Folder';
    if (value instanceof FileHandle) return 'FileHandle';
    if (value === null) return 'null';
    if (typeof value === 'boolean') return 'boolean';
    if (typeof value === 'number' && Number.isFinite(value)) return 'number';
    if (typeof value === 'string') return 'string';
    if (Array.isArray(value)) {
      if (!value.length) throw new Reject([{ path: 'value', code: 'type-mismatch', expected: 'an annotation for an empty list' }]);
      if (containsLive(value)) return 'Live<"array", "tag", "Array">';
      const types = [...new Set(value.map(item => this.inferScopeType(item)))];
      if (types.length !== 1) throw new Reject([{ path: 'value', code: 'type-mismatch', expected: 'an annotation for a heterogeneous list' }]);
      return `(${types[0]})[]`;
    }
    if (value && typeof value === 'object') {
      const entries = Object.entries(value);
      if (!entries.length) throw new Reject([{ path: 'value', code: 'type-mismatch', expected: 'an annotation for an empty record' }]);
      if (containsLive(value)) return 'Live<"object", "any", "">';
      return `{ ${entries.map(([key, item]) => `${key}: ${this.inferScopeType(item)}`).join(', ')} }`;
    }
    throw new Reject([{ path: 'value', code: 'type-mismatch', expected: 'a portable value' }]);
  }

  private scopePath(expression: string): string {
    const match = /^([A-Za-z_$][\w$]*)(.*)$/.exec(expression.trim());
    if (!match) throw new Reject([{ path: 'expression', code: 'bad-action', expected: 'a variable and field/index selections' }]);
    const root = match[1]!;
    const base = Object.hasOwn(this.lam.let, root) ? `let/${root}` : Object.hasOwn(this.lam.args, root) ? `args/${root}` :
      root === 'result' && this.lam.return !== MISSING ? 'return' : '';
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

  private readValuePage(path: string, value: Value, startRaw?: unknown, endRaw?: unknown): NativeResult | undefined {
    const explicit = startRaw !== undefined || endRaw !== undefined;
    if (typeof value !== 'string' && !Array.isArray(value) &&
        !(value && typeof value === 'object' && !isPending(value) && !isHandle(value) && !isLive(value))) return;
    const entries = typeof value === 'string' || Array.isArray(value) ? undefined : Object.entries(value);
    const size = typeof value === 'string' || Array.isArray(value) ? value.length : entries!.length;
    const rendered = typeof value === 'string' ? value : JSON.stringify(dump(value), null, 1);
    if (!explicit && rendered.length <= 4000 && size <= 20) return;
    const requestedStart = Number(startRaw ?? 0);
    const pageSize = typeof value === 'string' ? 2000 : 12;
    const requestedEnd = Number(endRaw ?? requestedStart + pageSize);
    if (!Number.isInteger(requestedStart) || !Number.isInteger(requestedEnd) || requestedStart < 0 || requestedEnd < 0)
      throw new Reject([{ path, code: 'bad-range', expected: 'non-negative JavaScript slice offsets' }]);
    const start = Math.min(size, requestedStart);
    const end = Math.min(size, Math.max(start, requestedEnd), start + pageSize);
    const footer = `[${path}: ${typeof value === 'string' ? 'characters' : Array.isArray(value) ? 'items' : 'fields'} ` +
      `[${start}, ${end}) of ${size}; ${end < size ? `next start=${end}` : 'end of value'}]`;
    if (typeof value === 'string') {
      const slice = value.slice(start, end);
      return { kind: 'ok', text: (explicit || end === size) && size <= 4000 ? slice : `${slice}\n${footer}`, value: slice };
    }
    if (Array.isArray(value)) {
      const slice = value.slice(start, end);
      const body = slice.map((item, index) => `${start + index}: ${JSON.stringify(dump(item))}`).join('\n');
      return { kind: 'ok', text: `${body.slice(0, 4000)}${body.length > 4000 ? '\n[CUT OFF: inspect an item by index]' : ''}\n` + footer, value: slice };
    }
    const selected = entries!.slice(start, end);
    const body = selected.map(([key, item]) => `${key}: ${JSON.stringify(dump(item))}`).join('\n');
    return { kind: 'ok', text: `${body.slice(0, 4000)}${body.length > 4000 ? '\n[CUT OFF: inspect a field by name]' : ''}\n` + footer,
      value: Object.fromEntries(selected) as Value };
  }

  /** Static type of an initializer expression, for locals declared without an annotation. */
  private scopeInitializerType(expression: string, locals: Record<string, Type>): Type | undefined {
    const source = expression.trim();
    // A reduce result has the accumulator's type, which need not match its list.
    if (/\.(?:reduce|reduceRight)\s*\(/.test(source)) return;
    const directCall = /^(?:await\s+)?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(/.exec(source);
    const directReturns = directCall ? returnTypeOf(this.lam.codebase, directCall[1]!) : undefined;
    if (directReturns) return parseType(directReturns);
    const mappedCall = /^await\s+Promise\.all\([\s\S]*\.map\([\s\S]*?([A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*)\s*\(/.exec(source);
    const mappedReturns = mappedCall ? returnTypeOf(this.lam.codebase, mappedCall[1]!) : undefined;
    if (mappedReturns) return parseType(`(${mappedReturns})[]`);
    const lastCollectionMethod = [...source.matchAll(/\.(find|filter|slice|map|flatMap)\s*\(/g)].at(-1)?.[1];
    const collection = ['find', 'filter', 'slice'].includes(lastCollectionMethod ?? '') ?
      /^(.*)\.(find|filter|slice)\s*\(.*\)$/s.exec(source) : null;
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
        } else if (index && (resolved.kind === 'list' || resolved.kind === 'dict')) { type = resolved.element; tail = index[4]!; }
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

  /** Portable snapshot of the scope for failure debugging. */
  private scopeSnapshot(): Record<string, unknown> {
    const locals: Record<string, Value> = { ...this.lam.let };
    if (!Object.hasOwn(locals, 'result') && this.lam.return !== MISSING) locals.result = this.lam.return;
    const view = (values: Record<string, Value>) => Object.fromEntries(Object.entries(values).map(([name, value]) =>
      [name, containsLive(value) ? oneLine(value) : dump(value)]));
    return { inputs: view(this.lam.args), locals: view(locals) };
  }

  /** Execute one eval action as an atomic scope transaction. */
  private async evaluate(code: string): Promise<NativeResult> {
    if (!code.trim()) throw new Reject([{ path: 'code', code: 'bad-action', expected: 'a TypeScript statement or expression' }]);
    const scopeBefore = this.scopeSnapshot();
    const inputNames = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.map(field => field.name) : [];
    const localNames = Object.keys(this.lam.let).filter(name => !isPending(this.lam.let[name]!));
    const localBindings = localNames.map(name => ({ name, mutable: this.scopeLocalMutability.get(name) ?? true,
      annotation: this.lam.letTypes[name] && this.lam.letTypes[name]!.kind !== 'host' ? formatType(this.lam.letTypes[name]!) : undefined }));
    const callableNames = Object.keys(this.lam.codebase);
    const opaqueNames: string[] = [];
    if (this.lam.projectTransaction) opaqueNames.push('folder');
    const debugBinding = this.failureBinding;
    if (debugBinding) opaqueNames.push(debugBinding);
    const captureCells = this.lam.captures ?? {};
    const captureRead = Object.fromEntries(Object.entries(captureCells).map(([name, cell]) => [name, cell.get()]));
    const serviceNames = Object.keys(this.runtime.services).filter(name => !inputNames.includes(name) &&
      !callableNames.includes(name) && !Object.hasOwn(captureCells, name));
    const hooks = this.runtime.hooks;
    const compiled = compileScopeSnippet(code, { inputBindings: inputNames, localBindings, helperBindings: callableNames,
      opaqueBindings: opaqueNames, resultBinding: true,
      captureBindings: Object.values(captureCells).map(cell => ({ name: cell.name, mutable: cell.mutable })),
      serviceBindings: serviceNames, analyze: source => hooks.analyze(this, source),
      guardPrefix: `eval:${this.runtime.options.runId}`, ...this.runtime.environment.scopeCapabilities });
    if (!compiled.ok || !compiled.program) {
      const text = compiled.diagnostics.map(item => `${item.line}:${item.column} ${item.code}: ${item.message}`).join('\n');
      return { kind: 'rejected', text: text + this.captureScopeFailure('compile', code, scopeBefore, text, compiled.diagnostics),
        codes: [...new Set(compiled.diagnostics.map(item => item.code))] };
    }
    const inputs = splitScope(this.lam.args);
    const locals = splitScope(Object.fromEntries(localNames.map(name => [name, this.lam.let[name]!])));
    if (this.lam.return !== MISSING && !Object.hasOwn(this.lam.let, 'result')) {
      const result = splitScope({ result: this.lam.return });
      Object.assign(locals.portable, result.portable); Object.assign(locals.live, result.live);
    }
    let finished: unknown;
    const plans = compiled.plans ?? [];
    const live = { inputs: inputs.live, locals: locals.live, captures: captureRead, callables: this.callables(),
      services: this.runtime.services, folder: this.lam.projectTransaction?.folder.root(),
      inline: (index: number, values: unknown[], accessors: Record<string, unknown>) => {
        const plan = plans[index];
        if (!plan) throw new Error('internal error: unknown inline plan');
        return hooks.inline(this, plan, values, accessors);
      },
      finite: hooks.finite, guard: hooks.guard,
      iterateOn: (step: unknown, initial: unknown, ...args: unknown[]) => hooks.iterateOn(this, step, initial, ...args),
      finish: (value: unknown) => { finished = value; } };
    const prologue = [
      ...(debugBinding ? [`const ${debugBinding} = self.debug;`] : []),
      ...(this.lam.projectTransaction ? ['const folder = __live.folder;'] : []),
    ].join('\n');
    const source = `${SCOPE_RUNTIME_PRELUDE}${prologue}\n${compiled.program}\n` +
      `return await ${compiled.entrypoint}(Object.assign({}, self.inputs, __live.inputs), ` +
      `Object.assign({}, self.locals, __live.locals), __live.captures);`;
    try {
      const evaluated = await this.runtime.evaluate(this.lam, source,
        { inputs: inputs.portable, locals: locals.portable, ...(this.failureDebug ? { debug: this.failureDebug } : {}) }, live);
      const raw = finished as { result?: unknown; inputs?: Record<string, unknown>;
        bindings?: Record<string, unknown>; captures?: Record<string, unknown> } | undefined;
      if (!raw || typeof raw !== 'object' || !raw.bindings || typeof raw.bindings !== 'object')
        throw new Reject([{ path: 'code', code: 'bad-action', expected: 'an atomic scope transaction result' }]);
      // Plain data is rebuilt in this realm; captured values keep their identity for write-back.
      const output = { ...hostCopy({ result: raw.result, inputs: raw.inputs, bindings: raw.bindings }) as
        { result?: unknown; inputs?: Record<string, unknown>; bindings: Record<string, unknown> }, captures: raw.captures };
      const thenable = (value: unknown) => !!value && (typeof value === 'object' || typeof value === 'function') &&
        typeof (value as PromiseLike<unknown>).then === 'function';
      if ([output.result, ...Object.values(output.bindings), ...Object.values(output.captures ?? {})].some(thenable))
        throw new Error('await the asynchronous function call before using its value');
      const annotations = new Map(compiled.bindings.map(binding => [binding.name, binding.annotation]));
      const initializers = new Map(compiled.bindings.map(binding => [binding.name, binding.initializer]));
      const mutability = new Map(compiled.bindings.map(binding => [binding.name, binding.mutable]));
      const staged: [string, Type, Value][] = [];
      const stagedInputs: [string, Value][] = [];
      const inferred: Record<string, Type> = { ...this.lam.letTypes };
      for (const [name, value] of Object.entries(output.inputs ?? {})) {
        const field = this.lam.type.kind === 'lambda' ? this.lam.type.params.fields.find(item => item.name === name) : undefined;
        if (!field || !Object.hasOwn(this.lam.args, name)) continue;
        stagedInputs.push([name, coerce(value, field.type, this.env, `args/${name}`)]);
      }
      for (const [name, value] of Object.entries(output.bindings)) {
        if (Object.hasOwn(this.lam.args, name) || Object.hasOwn(this.lam.codebase, name))
          throw new Reject([{ path: name, code: 'not-writable', expected: 'a local variable' }]);
        if (name === 'result' && this.lam.type.kind === 'lambda') {
          try { coerce(value, this.lam.type.returns, this.env, 'return'); }
          catch { continue; /* An intermediate observation must not change the typed result slot. */ }
        }
        let type = name === 'result' && this.lam.type.kind === 'lambda' ? this.lam.type.returns : this.lam.letTypes[name];
        const annotation = annotations.get(name);
        if (annotation) type = parseType(annotation);
        if (!type && initializers.get(name)) type = this.scopeInitializerType(initializers.get(name)!, inferred);
        if (!type) {
          try { type = parseType(this.inferScopeType(value)); }
          catch (error) {
            if (this.lam.type.kind !== 'lambda' || !compiled.resultBindings?.includes(name)) throw error;
            type = this.lam.type.returns;
          }
        }
        inferred[name] = type;
        staged.push([name, type, coerce(value, type, this.env, `let/${name}`)]);
      }
      let functionResult: Value | undefined;
      if (compiled.producesResult && this.lam.type.kind === 'lambda') try {
        functionResult = coerce(output.result, this.lam.type.returns, this.env, 'return');
      } catch { /* An intermediate expression of another type is still a useful eval result. */ }
      if (functionResult === undefined && !compiled.producesResult && this.lam.type.kind === 'lambda') {
        const resultBinding = staged.find(([name]) => name === 'result');
        if (resultBinding) try { functionResult = coerce(resultBinding[2], this.lam.type.returns, this.env, 'return'); }
        catch { /* A local named result may still be an intermediate value. */ }
      }
      const captureWrites: [string, unknown][] = [];
      for (const [name, value] of Object.entries(output.captures ?? {})) {
        const cell = captureCells[name];
        if (!cell?.mutable || Object.is(value, captureRead[name])) continue;
        if (!Object.is(cell.get(), captureRead[name]))
          throw new Reject([{ path: name, code: 'capture-conflict',
            expected: `${name} was changed by another caller while this eval ran; run the eval again with its current value` }]);
        captureWrites.push([name, value]);
      }
      for (const [name, value] of captureWrites) captureCells[name]!.set!(value);
      for (const [name, value] of stagedInputs) this.lam.args[name] = value;
      for (const [name, type, value] of staged) {
        this.lam.letTypes[name] = type;
        this.lam.let[name] = value;
        if (mutability.has(name)) this.scopeLocalMutability.set(name, mutability.get(name)!);
      }
      if (functionResult !== undefined) {
        this.lam.return = functionResult;
        this.failureDebug = undefined;
        if (this.lam.subtype === 'directory-reducer') { this.lam.commitInclude = undefined; this.lam.commitExclude = undefined; }
      }
      const text = isLive(output.result) || isHandle(output.result) ? livePreview(output.result as object) :
        JSON.stringify(output.result ?? null) ?? 'null';
      const rendered = text.length <= 400 ? text : `${text.slice(0, 400)} … (${text.length} chars)`;
      const open = functionResult === undefined ? [] : pendingProgramLines(this.lam.originalBody ?? this.lam.body, this.lam.marks);
      const status = functionResult === undefined ? '' : open.length ?
        `\nFunction result set; lines still open: ${open.join(', ')}.` : '\nFunction result set.';
      const stored = staged.map(([name, , value]) => `local ${name} = ${oneLine(value)}`);
      const storedStatus = stored.length ? `\nStored ${stored.join('; ')}.` : '';
      const logStatus = evaluated.logs?.length ? `console:\n${evaluated.logs.join('\n')}\n` : '';
      return { kind: 'ok', text: logStatus + rendered + storedStatus + status, value: (output.result ?? null) as Value,
        ...(compiled.repairs.length ? { codes: ['coerced-redundant-self-alias'] } : {}) };
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      const note = this.captureScopeFailure(error instanceof EvalFailure ? 'runtime' : 'boundary',
        code, scopeBefore, message, error instanceof Reject ? error.diagnostics : [], error);
      if (error instanceof Reject) { const result = rejected(error); return { ...result, text: result.text + note }; }
      return { kind: 'error', text: message + note };
    }
  }

  private resolve(path: string): Ref {
    const parts = path.split('/').filter(Boolean);
    const first = parts.shift();
    if (first === 'return') {
      const ref = itemRef(this.lam as unknown as Record<string, Value>, 'return',
        this.lam.type.kind === 'lambda' ? this.lam.type.returns : parseType('null'), this.env, 'return');
      return this.descend(ref, parts);
    }
    const name = parts.shift();
    if (!name) throw new Reject([{ path, code: 'no-such-path' }]);
    if (first === 'let') {
      const type = this.lam.letTypes[name];
      if (!type) throw new Reject([{ path, code: 'no-such-path' }]);
      return this.descend(itemRef(this.lam.let, name, type, this.env, `let/${name}`), parts);
    }
    if (first === 'args' && this.lam.type.kind === 'lambda') {
      const field = this.lam.type.params.fields.find(f => f.name === name);
      if (!field) throw new Reject([{ path, code: 'unknown-field' }]);
      return this.descend(itemRef(this.lam.args, name, field.type, this.env, `args/${name}`, 'not-writable'), parts);
    }
    throw new Reject([{ path, code: 'no-such-path' }]);
  }

  private descend(ref: Ref, parts: string[]): Ref {
    for (const part of parts) {
      const current = ref.get();
      let type = ref.env.resolve(ref.type!);
      if (type.kind === 'union') {
        const selected = type.members.map(member => ref.env.resolve(member)).find(member =>
          member.kind === 'record' ? member.fields.some(field => field.name === part) :
            member.kind === 'dict' ? current !== null && typeof current === 'object' && !Array.isArray(current) :
              member.kind === 'list' ? Array.isArray(current) : false);
        if (selected) type = selected;
      }
      let childType: Type;
      if (type.kind === 'record') {
        const field = type.fields.find(f => f.name === part);
        if (!field) throw new Reject([{ path: `${ref.path}/${part}`, code: 'unknown-field' }]);
        childType = field.type;
      } else if (type.kind === 'dict') childType = type.element;
      else if (type.kind === 'list') {
        if (!/^\d+$/.test(part)) throw new Reject([{ path: `${ref.path}/${part}`, code: 'no-such-path' }]);
        childType = type.element;
      } else throw new Reject([{ path: `${ref.path}/${part}`, code: 'no-such-path' }]);
      const parent = ref, key = part;
      ref = { path: `${parent.path}/${part}`, type: childType, env: parent.env, deny: parent.deny,
        get: () => {
          const value = parent.get();
          return value !== MISSING && value !== null && typeof value === 'object' && key in value ? (value as Record<string, Value>)[key]! : MISSING;
        },
        set: () => { throw new Reject([{ path: `${parent.path}/${key}`, code: 'not-writable' }]); }, del: () => {} };
    }
    return ref;
  }
}
