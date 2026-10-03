import { ProgramView } from '../adaptation/program-view.js';
import { isAdaptationBinding, bindAdaptation } from '../adaptation/compatibility.js';
import { fingerprint } from '../adaptation/identity.js';
import type { ProgramDescriptor, AdaptationBinding, ExecutorIdentity } from '../adaptation/types.js';
import { declarationNamespace } from '../native/external.js';
import type { EvalEnvironment } from '../native/evaluator.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import type { NativeReviewOptions } from '../native/agent.js';
import { TOOLS_PROMPT } from '../native/prompt.js';
import { NatlangContextError, currentFrame, runInFrame, type Frame } from './context.js';
import type { IterationStatisticsStore, ProgressJudgeFunction } from './iterate.js';

export type ModelDriver = (request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn> | ModelTurn;
export type ModelConfig = { driver: ModelDriver;
  /** Model identity and revision, recorded in execution-graph manifests (spec/NEURALESE_GRAPH.md). */
  id?: string; revision?: string;
  maxTurns?: number; maxTokens?: number; turnTokens?: number;
  temperature?: number; maxSeconds?: number;
  /** Context budget in prompt tokens before old tool outputs are elided (default 16384; null never compacts). */
  contextTokens?: number | null;
  /** Failed evals or rejected tool calls in a row before the call stops; unlimited unless set. */
  maxFailureRepairs?: number;
  review?: NativeReviewOptions };

/** One natlang invocation's trace, delivered to the runtime's trace sink when the invocation ends. */
export type InvocationTrace = { callId: string; parentCallId: string | null; taskId: string;
  adaptation?: Record<string, unknown>; definitionId: string; name: string; outcome: string; detail: string; events: Record<string, unknown>[] };
export type TraceSink = (trace: InvocationTrace) => void;

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
};

export type TaskOptions = { program?: ProgramDescriptor; adaptation?: AdaptationBinding | null; services?: Services; serviceDeclarations?: Record<string, string>;
  serviceScopes?: Record<string, string[]>; signal?: AbortSignal; trace?: TraceSink; name?: string };

let defaultEnvironment: ((options: NatlangRuntimeOptions) => EvalEnvironment) | undefined;
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
  }
  private readonly traceSink?: TraceSink;

  get frame(): Frame { return { task: this, chain: [] }; }
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
  model(): ModelConfig | undefined {
    const model = this.runtime.options.model;
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
  constructor(readonly options: NatlangRuntimeOptions = {}) {}

  /** Run `fn` as a natlang task. Natlang functions called anywhere inside it use this runtime. */
  async run<T>(fn: () => T | Promise<T>, options: TaskOptions = {}): Promise<T> {
    if (this.closed) throw new Error('natlang runtime is closed');
    const task = new NatlangTask(this, options);
    activeTasks.add(task);
    try { return await runInFrame(task.frame, fn); }
    finally { await task.drain(); task.close(); }
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
