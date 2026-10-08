import { ProgramView } from '../adaptation/program-view.js';
import { isAdaptationBinding, bindAdaptation } from '../adaptation/compatibility.js';
import { fingerprint } from '../adaptation/identity.js';
import type { ProgramDescriptor, AdaptationBinding, ExecutorIdentity } from '../adaptation/types.js';
import { declarationNamespace } from '../native/external.js';
import type { EvalEnvironment } from '../native/evaluator.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import type { NativeReviewOptions } from '../native/agent.js';
import { TOOLS_PROMPT } from '../native/prompt.js';
import { NatlangContextError, currentFrame, runInFrame, type DecisionReadout, type Frame } from './context.js';
import type { IterationStatisticsStore, ProgressJudgeFunction } from './iterate.js';
import type { CallStoreLike } from '../calls/recorder.js';
import { CompilationCache } from '../calls/compilations.js';

/** How far stored compilations may serve calls: not at all, compared in the background only, or served. */
export type SpecializationMode = 'off' | 'shadow' | 'on';
const MODE_RANK: Record<SpecializationMode, number> = { off: 0, shadow: 1, on: 2 };

export type ModelDriver = (request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn> | ModelTurn;
export type ModelConfig = { driver: ModelDriver;
  /** Model identity and revision, recorded in execution-graph manifests (spec/NEURALESE_GRAPH.md). */
  id?: string; revision?: string;
  maxTurns?: number; maxTokens?: number; turnTokens?: number;
  temperature?: number; maxSeconds?: number;
  /** Context budget in prompt tokens before old tool outputs are elided (default: the context window the model server reports, less an eighth for the reply; 16384 when it does not say; null never compacts). */
  contextTokens?: number | null;
  /** Failed evals or rejected tool calls in a row before the call stops; unlimited unless set. */
  maxFailureRepairs?: number;
  /** `finite-returns`: every call with a finite result type uses the decision readout when the driver can score
   * replies; `declared` (default): only functions whose frontmatter says `readout: decision`. */
  decisionReadout?: 'declared' | 'finite-returns';
  /** Guided generation on natlang's own servers (NativeToolAgent's `guidance`); off by default. */
  guidance?: boolean | { repeat?: number; syntax?: boolean; retries?: number; tools?: string[] };
  review?: NativeReviewOptions };

/** One natlang invocation's trace, delivered to the runtime's trace sink when the invocation ends. */
export type InvocationTrace = { callId: string; parentCallId: string | null; taskId: string;
  adaptation?: Record<string, unknown>; definitionId: string; name: string; outcome: string; detail: string; events: Record<string, unknown>[] };
export type TraceSink = (trace: InvocationTrace) => void;
/** A decision with its distribution (`runtime.decide`): every allowed value's probability, highest is `value`. */
export type Decision<T> = { value: T; probabilities: { value: T; probability: number }[]; confidence: number; scored: boolean };

export type NatlangLimits = { maxEpisodes?: number; maxDepth?: number; maxActions?: number; maxToolCalls?: number;
  /** Wall-clock limit for one task. */
  timeoutMs?: number };

export type Services = Record<string, object>;

export type NatlangRuntimeOptions = {
  /** Account for speculative source edits in the current task. */
  onFolderProposal?: () => void;
  program?: ProgramDescriptor;
  adaptation?: AdaptationBinding | null;
  executorIdentity?: ExecutorIdentity;
  isolateModules?: boolean;
  signal?: AbortSignal;
  evaluation?: { id: string; caseId: string; replicate: number };
  codeEdits?: 'allow' | 'deny';
  model?: ModelDriver | ModelConfig;
  /**
   * Further models by name. A named function whose frontmatter says `model: NAME` runs on this one (a small model
   * for judgments, a large one for authoring); without an entry of that name it runs on `model`.
   */
  models?: Record<string, ModelDriver | ModelConfig>;
  /** Drive interpreter sessions directly instead of through a model (fixtures, replay, tests). */
  agent?: import('../native/runtime.js').NativeAgent;
  /** Typed host services, available to callable-folder code via `natlang:services` and to eval as named bindings. */
  services?: Services;
  /**
   * What the model is shown of each service: TypeScript declarations of its members with doc comments (a `.d.ts`
   * body of `export` declarations, or a whole `declare namespace name { … }`). The model can read them with
   * read_code; without one a service is listed by its method names only.
   */
  serviceDeclarations?: Record<string, string>;
  /** Services only some functions may use, by each function's source path (`review/assess.nl`); see SPEC. */
  serviceScopes?: Record<string, string[]>;
  trace?: TraceSink;
  /**
   * Exact host-only invocation I/O capture. Inputs use the named source allowlist;
   * captureAllOutputs explicitly includes arbitrary inline/named return values. Disabled
   * unless configured; no model-visible context or execution state changes.
   */
  exactHostTraceCapture?: { definitionSources: string[]; inputArguments: string[]; captureOutput?: boolean; captureAllOutputs?: boolean; maxBytes: number };
  limits?: NatlangLimits;
  seed?: { mode: 'derived' | 'backend'; root?: number };
  /** Evaluator factory; each platform installs a default. */
  environment?: () => EvalEnvironment;
  /** Directory whose node_modules eval imports resolve from (Node; default: the nearest package.json above cwd). */
  workspace?: string;
  /** Allow `fetch` in eval (default: true). */
  network?: boolean;
  /** Extra system prompt text appended for every invocation. */
  /** Application instructions appended to the shared runtime prompt. */
  systemPrompt?: string | (() => string);
  statistics?: IterationStatisticsStore;
  progressJudge?: ProgressJudgeFunction;
  /** Tensor store and write port for Neuralese values (S0 §3). Without it, soft literals fail the call. */
  neuralese?: import('../native/neuralese.js').NeuraleseRuntimeOptions;
  /** Law-based combinator rewrites and their measurements (spec/NEURALESE_REWRITES.md); every rule is off without it. */
  rewrites?: import('../compiler/rewrites.js').RewriteGate;
  /**
   * The call record store (plans/TRACE_SPECIALIZATION.md): every call is recorded there and served from its
   * compilations. Default: the machine's store on Node (`$NATLANG_CALL_STORE`); `false` records nothing.
   */
  calls?: CallStoreLike | false;
  /** The program's directory on this machine, recorded with each call so offline work can reload its definitions. */
  programRoot?: string;
  /** The program's `recording` settings (`natlang.json`): definitions (`dir/*.nl`) or arguments (`name.arg`) recorded by type only. */
  recording?: { exclude?: string[] };
  /** How far compilations may serve this runtime's calls; the machine setting bounds it (default: the machine setting). */
  specialization?: SpecializationMode;
};

export type TaskOptions = { program?: ProgramDescriptor; adaptation?: AdaptationBinding | null; services?: Services; serviceDeclarations?: Record<string, string>;
  serviceScopes?: Record<string, string[]>; signal?: AbortSignal; trace?: TraceSink; name?: string;
  /** Compilations for this task's calls (an audit or replay runs with `off`). */
  specialization?: SpecializationMode;
  /** Set on an audit run: the call whose inputs it re-runs through the agent. */
  auditOf?: string };

let defaultEnvironment: ((options: NatlangRuntimeOptions) => EvalEnvironment) | undefined;
let defaultCallStore: (() => CallStoreLike | undefined) | undefined;
/** Installed by the Node entry point: the machine's call record store. */
export function setDefaultCallStoreFactory(factory: () => CallStoreLike | undefined): void { defaultCallStore = factory; }
/** Installed by the Node and browser entry points. */
export function setDefaultEnvironmentFactory(factory: (options: NatlangRuntimeOptions) => EvalEnvironment): void { defaultEnvironment = factory; }

let taskSequence = 0;
const activeTasks = new Set<NatlangTask>();

/** An ordinary TypeScript task in which natlang functions can be called. */
export class NatlangTask {
  readonly id: string;
  readonly services: Services;
  readonly serviceDeclarations: Record<string, string>;
  readonly serviceScopes: Record<string, string[]>;
  readonly signal: AbortSignal;
  readonly episodeBudget: { limit?: number; used: number };
  readonly traces: InvocationTrace[] = [];
  readonly programView: ProgramView;
  readonly specialization?: SpecializationMode;
  readonly auditOf?: string;
  readonly moduleInstances = new WeakMap<import('./loader.js').ModuleRecord, { exports: Record<string, unknown>; ready: boolean }>();
  private readonly pending = new Set<Promise<unknown>>();
  private readonly pendingChildren = new Map<string, Set<Promise<unknown>>>();
  track<T>(call: Promise<T>, parentCallId?: string): Promise<T> {
    this.pending.add(call);
    if (parentCallId) {
      let children = this.pendingChildren.get(parentCallId);
      if (!children) this.pendingChildren.set(parentCallId, children = new Set());
      children.add(call);
    }
    call.finally(() => {
      this.pending.delete(call);
      if (parentCallId) {
        const children = this.pendingChildren.get(parentCallId);
        children?.delete(call);
        if (children?.size === 0) this.pendingChildren.delete(parentCallId);
      }
    }).catch(() => {});
    return call;
  }
  /** The child calls of an invocation that are still running. */
  pendingChildCalls(parentCallId: string): Promise<unknown>[] { return [...this.pendingChildren.get(parentCallId) ?? []]; }
  async drainChildren(parentCallId: string): Promise<void> {
    while (true) {
      const children = this.pendingChildren.get(parentCallId);
      if (!children?.size) return;
      await Promise.allSettled([...children]);
    }
  }
  hasPendingChildren(parentCallId: string): boolean {
    return (this.pendingChildren.get(parentCallId)?.size ?? 0) > 0;
  }
  async drain(): Promise<void> { while (this.pending.size) await Promise.allSettled([...this.pending]); }
  private readonly abort = new AbortController();
  private callSequence = 0;
  private readonly definitionCalls = new Map<string, number>();
  private readonly taskOrdinal: number;
  definitionSeedId(key: string): string { const ordinal = (this.definitionCalls.get(key) ?? 0) + 1; this.definitionCalls.set(key, ordinal); return key + '/' + ordinal; }
  private closed = false;
  private timer?: ReturnType<typeof setTimeout>;
  private appPromptSnapshot?: string;

  constructor(readonly runtime: NatlangRuntime, options: TaskOptions = {}) {
    let binding = options.adaptation === undefined ? runtime.options.adaptation : options.adaptation;
    const program = options.program ?? runtime.options.program ?? binding?.program;
    if (binding) {
      const artifact = binding.artifact;
      // A package can be loaded through multiple module URLs. Revalidate foreign
      // bindings from their artifact; never trust their supplied candidate map.
      if (!isAdaptationBinding(binding)) {
        if (!program || !runtime.options.executorIdentity) throw new Error('adaptation must be created by bindAdaptation');
        binding = bindAdaptation(artifact, program, runtime.options.executorIdentity);
      }
      if (program?.buildHash !== binding.program.buildHash || program?.id !== binding.program.id) throw new Error('task program and adaptation differ');
      if (!runtime.options.executorIdentity || fingerprint(runtime.options.executorIdentity) !== fingerprint(binding.executor))
        throw new Error('runtime executor identity does not match adaptation');
      const configured = typeof runtime.options.model === 'function' ? {} : runtime.options.model ?? {};
      const { driver: _driver, ...allSettings } = configured as Partial<ModelConfig>;
      const settings = Object.fromEntries(Object.entries(allSettings).filter(([_key, value]) => value !== undefined));
      if (fingerprint(settings) !== fingerprint(binding.artifact.policy.settings))
        throw new Error('runtime inference settings differ from evaluated artifact policy; configure the evaluated profile or revalidate');
      if (runtime.options.codeEdits && runtime.options.codeEdits !== binding.artifact.policy.codeEdits)
        throw new Error('runtime code-edit policy differs from evaluated artifact');
      if (fingerprint(runtime.options.limits ?? {}) !== fingerprint(binding.artifact.policy.limits ?? {}))
        throw new Error('runtime inference limits differ from evaluated artifact policy');
      const extra = runtime.options.systemPrompt;
      this.appPromptSnapshot = typeof extra === 'function' ? extra() : extra ?? '';
      if (binding.artifact.policy.systemPromptHash !== undefined) {
        if (fingerprint(this.appPromptSnapshot, 'natlang.system-prompt/v1') !== binding.artifact.policy.systemPromptHash)
          throw new Error('application system prompt differs from evaluated artifact policy');
      } else if (this.appPromptSnapshot) throw new Error('declare evaluated systemPromptHash or use program guidance');
    }
    this.programView = new ProgramView(program, binding);
    this.taskOrdinal = ++taskSequence;
    this.id = `${options.name ?? 'task'}-${this.taskOrdinal}-${Math.random().toString(36).slice(2, 8)}`;
    this.services = options.services ?? runtime.options.services ?? {};
    this.serviceDeclarations = Object.fromEntries(Object.entries(options.serviceDeclarations ?? runtime.options.serviceDeclarations ?? {})
      .map(([name, text]) => [name, /^\s*declare (?:namespace|const) /.test(text) ? text.trim() : declarationNamespace(name, text)]));
    this.serviceScopes = options.serviceScopes ?? runtime.options.serviceScopes ?? {};
    if (binding) {
      const expected = program?.services ?? { declarations: {}, scopes: {} };
      if (fingerprint({ declarations: this.serviceDeclarations, scopes: this.serviceScopes }, 'natlang.services/v1') !== fingerprint(expected, 'natlang.services/v1'))
        throw new Error('runtime service declarations/scopes differ from evaluated program contract');
      if (Object.keys(this.services).some(name => !Object.hasOwn(expected.declarations, name)))
        throw new Error('adapted tasks cannot add undeclared service capabilities');
    }
    const externalSignal = options.signal ?? runtime.options.signal;
    this.signal = externalSignal ? AbortSignal.any([externalSignal, this.abort.signal]) : this.abort.signal;
    this.episodeBudget = { limit: runtime.options.limits?.maxEpisodes, used: 0 };
    const timeout = runtime.options.limits?.timeoutMs;
    if (timeout !== undefined) this.timer = setTimeout(() => this.cancel(new Error('natlang task timed out')), timeout);
    this.traceSink = options.trace;
    this.specialization = options.specialization;
    this.auditOf = options.auditOf;
  }
  private readonly traceSink?: TraceSink;
  /** How far compilations may serve this task's calls: the machine, runtime and task settings, whichever is lowest. */
  specializationMode(): SpecializationMode {
    const store = this.runtime.callStore();
    if (!store) return 'off';
    let mode: SpecializationMode = 'on';
    try { mode = store.settings().specialization; } catch { return 'off'; }
    for (const limit of [this.runtime.options.specialization, this.specialization])
      if (limit && MODE_RANK[limit] < MODE_RANK[mode]) mode = limit;
    return mode;
  }

  get frame(): Frame { return { task: this, chain: [], signal: this.signal }; }
  nextCallId(): string { return `${this.id}/${++this.callSequence}`; }
  checkOpen(): void {
    if (this.closed) throw new Error('this natlang task has finished');
    if (this.signal.aborted) throw this.signal.reason instanceof Error ? this.signal.reason : new Error('natlang task aborted');
  }
  record(trace: InvocationTrace): void {
    this.traces.push(trace);
    this.traceSink?.(trace);
    this.runtime.options.trace?.(trace);
  }
  environment(): EvalEnvironment {
    if (this.runtime.options.environment) return this.runtime.options.environment();
    if (!defaultEnvironment) throw new Error('no natlang evaluator is installed for this platform');
    return defaultEnvironment(this.runtime.options);
  }
  model(name?: string): ModelConfig | undefined {
    const model = (name && this.runtime.options.models?.[name]) || this.runtime.options.model;
    return typeof model === 'function' ? { driver: model } : model;
  }
  systemPrompt(): string {
    const extra = this.runtime.options.systemPrompt;
    return TOOLS_PROMPT + (this.appPromptSnapshot ?? (typeof extra === 'function' ? extra() : extra ?? ''));
  }
  cancel(reason: unknown = new Error('natlang task cancelled')): void { this.abort.abort(reason); }
  get isClosed(): boolean { return this.closed; }
  close(): void {
    if (this.closed) return;
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
    // NatlangRuntime.run drains in-flight work before closing the task. Abort
    // its derived signal now so external-signal subscriptions for this task
    // are released promptly instead of waiting for garbage collection.
    this.abort.abort(new Error('natlang task finished'));
    activeTasks.delete(this);
  }
}

/** Project-level natlang runtime: model, services, trace sink, limits. Create tasks with `run`. */
export class NatlangRuntime {
  private closed = false;
  private store?: CallStoreLike | null;
  private compiled?: CompilationCache;
  /** The call record store this runtime records to, if any. */
  callStore(): CallStoreLike | undefined {
    if (this.store === undefined) {
      if (this.options.calls === false) this.store = null;
      else if (this.options.calls) this.store = this.options.calls;
      else {
        try { this.store = defaultCallStore?.() ?? null; }
        catch (error) {
          console.warn(`natlang: the call record store could not be opened; calls are not recorded: ${error instanceof Error ? error.message : String(error)}`);
          this.store = null;
        }
      }
    }
    return this.store ?? undefined;
  }
  /** Loaded compilations of this runtime's store. */
  compilations(): CompilationCache | undefined {
    const store = this.callStore();
    return store ? this.compiled ??= new CompilationCache(store) : undefined;
  }
  constructor(readonly options: NatlangRuntimeOptions = {}) {
    const capture = options.exactHostTraceCapture;
    if (capture && (!Number.isSafeInteger(capture.maxBytes) || capture.maxBytes < 1 || capture.maxBytes > 8_000_000 ||
        !Array.isArray(capture.definitionSources) || (!capture.definitionSources.length && capture.captureAllOutputs !== true) ||
        (capture.captureAllOutputs !== undefined && typeof capture.captureAllOutputs !== 'boolean') ||
        capture.definitionSources.some(source => typeof source !== 'string' || !source.endsWith('.nl')) ||
        !Array.isArray(capture.inputArguments) || capture.inputArguments.some(name => typeof name !== 'string' || !name)))
      throw new RangeError('exact host trace capture requires source allowlist, argument names, and a bounded positive byte budget');
  }

  /** Run `fn` as a natlang task. Natlang functions called anywhere inside it use this runtime. */
  async run<T>(fn: () => T | Promise<T>, options: TaskOptions = {}): Promise<T> {
    if (this.closed) throw new Error('natlang runtime is closed');
    const task = new NatlangTask(this, options);
    activeTasks.add(task);
    try { return await runInFrame(task.frame, fn); }
    finally { await task.drain(); task.close(); }
  }

  /**
   * Call a natural-language function with a finite result type and return its answer with the probability of every
   * allowed value, so code can act on confidence (escalate below a floor, ask before an irreversible step). The
   * probabilities come from the decision readout (`readout: decision`, or `decisionReadout: 'finite-returns'`); when
   * the call did not score (no readout, or a backend that cannot score), the answer has probability 1 and `scored`
   * is false.
   */
  async decide<A extends unknown[], T>(fn: (...args: A) => Promise<T>, ...args: A): Promise<Decision<T>> {
    return this.run(() => decideInFrame(currentFrame()!, fn, args));
  }

  /** Bind a callback to the current task so it can call natlang functions when it runs later. */
  bind<F extends (...args: never[]) => unknown>(fn: F): F {
    const frame = currentFrame();
    if (!frame) throw new NatlangContextError('runtime.bind(fn) must be called inside runtime.run(...).');
    return ((...args: Parameters<F>) => runInFrame(frame, () => fn(...args))) as F;
  }

  close(): void { this.closed = true; }
}

export function createNatlangRuntime(options: NatlangRuntimeOptions = {}): NatlangRuntime {
  return new NatlangRuntime(options);
}

/**
 * Call fn in frame and return its answer with the distribution its decision readout scored: runtime.decide on the
 * host, decide(...) in eval. Without a scored readout the answer has probability 1 and `scored` is false.
 */
export async function decideInFrame<A extends unknown[], T>(frame: Frame, fn: (...args: A) => Promise<T>, args: A): Promise<Decision<T>> {
  if (typeof fn !== 'function') throw new TypeError('decide(fn, ...args) needs a function to call, such as a natural-language function with a finite result type');
  let readout: DecisionReadout | undefined;
  const value = await runInFrame({ ...frame, readout: scored => { readout ??= scored; } }, () => fn(...args));
  const scored = readout as DecisionReadout | undefined;
  const probabilities = scored
    ? scored.options.map((option, index) => ({ value: JSON.parse(option) as T, probability: scored.probabilities[index]! }))
    : [{ value, probability: 1 }];
  return { value, probabilities, confidence: Math.max(...probabilities.map(item => item.probability)), scored: scored !== undefined };
}

/** Resolve the frame for a natlang call: the propagated frame, the creation frame, or the only active task. */
export function resolveFrame(created?: Frame): Frame {
  const frame = currentFrame();
  if (frame && !frameClosed(frame)) return frame;
  if (created && !frameClosed(created)) return created;
  if (activeTasks.size === 1) return [...activeTasks][0]!.frame;
  throw new NatlangContextError(activeTasks.size > 1 ?
    'Several natlang tasks are active and this call cannot tell which one it belongs to. ' +
    'Wrap the callback with runtime.bind(fn) inside the task that should own it.' : undefined);
}
const frameClosed = (frame: Frame) => frame.task.isClosed;

export { recordingServices } from '../native/effects.js';
