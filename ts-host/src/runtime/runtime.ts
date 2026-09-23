import type { EvalEnvironment } from '../native/evaluator.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import type { NativeReviewOptions } from '../native/agent.js';
import { NatlangContextError, currentFrame, runInFrame, type Frame } from './context.js';
import type { IterationStatisticsStore, ProgressJudgeFunction } from './iterate.js';

export type ModelDriver = (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;
export type ModelConfig = { driver: ModelDriver; maxTurns?: number; maxTokens?: number; turnTokens?: number;
  temperature?: number; maxSeconds?: number; segmentTurns?: number | null; segmentMessages?: number | null;
  review?: NativeReviewOptions; validationFeedback?: 'caller' | 'local' };

/** One natlang invocation's trace, delivered to the runtime's trace sink when the invocation ends. */
export type InvocationTrace = { callId: string; parentCallId: string | null; taskId: string;
  definitionId: string; name: string; outcome: string; detail: string; events: Record<string, unknown>[] };
export type TraceSink = (trace: InvocationTrace) => void;

export type NatlangLimits = { maxEpisodes?: number; maxDepth?: number; maxActions?: number; maxToolCalls?: number;
  /** Wall-clock limit for one task. */
  timeoutMs?: number };

export type Services = Record<string, object>;

export type NatlangRuntimeOptions = {
  model?: ModelDriver | ModelConfig;
  /** Drive interpreter sessions directly instead of through a model (fixtures, replay, tests). */
  agent?: import('../native/runtime.js').NativeAgent;
  /** Typed host services, available to callable-folder code via `natlang:services` and to eval as named bindings. */
  services?: Services;
  trace?: TraceSink;
  limits?: NatlangLimits;
  seed?: { mode: 'compatibility' | 'derived' | 'backend'; root?: number };
  /** Evaluator factory; each platform installs a default. */
  environment?: () => EvalEnvironment;
  /** Application directory whose package.json declares importable packages (Node; default: nearest to cwd). */
  workspace?: string;
  /** Allow `fetch` in eval (Node default: true when a workspace exists). */
  network?: boolean;
  /** Extra system prompt text appended for every invocation. */
  systemPrompt?: string | (() => string);
  statistics?: IterationStatisticsStore;
  progressJudge?: ProgressJudgeFunction;
};

export type TaskOptions = { services?: Services; signal?: AbortSignal; trace?: TraceSink; name?: string };

let defaultEnvironment: ((options: NatlangRuntimeOptions) => EvalEnvironment) | undefined;
let defaultPrompt: (environment: EvalEnvironment) => string = () => '';
/** Installed by the Node and browser entry points. */
export function setDefaultEnvironmentFactory(factory: (options: NatlangRuntimeOptions) => EvalEnvironment): void { defaultEnvironment = factory; }
export function setDefaultSystemPrompt(prompt: (environment: EvalEnvironment) => string): void { defaultPrompt = prompt; }

let taskSequence = 0;
const activeTasks = new Set<NatlangTask>();

/** An ordinary TypeScript task in which natlang functions can be called. */
export class NatlangTask {
  readonly id: string;
  readonly services: Services;
  readonly signal: AbortSignal;
  readonly episodeBudget: { limit?: number; used: number };
  readonly traces: InvocationTrace[] = [];
  private readonly abort = new AbortController();
  private callSequence = 0;
  private closed = false;
  private timer?: ReturnType<typeof setTimeout>;

  constructor(readonly runtime: NatlangRuntime, options: TaskOptions = {}) {
    this.id = `${options.name ?? 'task'}-${++taskSequence}-${Math.random().toString(36).slice(2, 8)}`;
    this.services = options.services ?? runtime.options.services ?? {};
    this.signal = options.signal ? AbortSignal.any([options.signal, this.abort.signal]) : this.abort.signal;
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
  systemPrompt(environment: EvalEnvironment): string {
    const extra = this.runtime.options.systemPrompt;
    return defaultPrompt(environment) + (typeof extra === 'function' ? extra() : extra ?? '');
  }
  cancel(reason: unknown = new Error('natlang task cancelled')): void { this.abort.abort(reason); }
  get isClosed(): boolean { return this.closed; }
  close(): void {
    if (this.closed) return;
    this.closed = true;
    if (this.timer) clearTimeout(this.timer);
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
    finally { task.close(); }
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

/**
 * Wrap services so each method call is recorded as an effect. Objects are wrapped one level deep
 * per property access; the original receiver is preserved.
 */
export function recordingServices(services: Services,
  emit: (event: { phase: 'requested' | 'completed' | 'failed'; service: string; method: string; args?: unknown; result?: unknown; error?: string }) => void): Services {
  const preview = (value: unknown): unknown => {
    try {
      const text = JSON.stringify(value);
      return text === undefined ? String(value) : text.length > 400 ? `${text.slice(0, 400)} … (${text.length} chars)` : JSON.parse(text);
    } catch { return String(value); }
  };
  const wrap = (target: object, label: string): object => new Proxy(target, {
    get(object, property, receiver) {
      const value = Reflect.get(object, property, receiver);
      if (typeof property === 'symbol') return value;
      if (typeof value === 'function') return (...args: unknown[]) => {
        emit({ phase: 'requested', service: label, method: property, args: preview(args) });
        try {
          const result = (value as Function).apply(object, args);
          if (result && typeof (result as PromiseLike<unknown>).then === 'function')
            return Promise.resolve(result).then(resolved => { emit({ phase: 'completed', service: label, method: property, result: preview(resolved) }); return resolved; },
              error => { emit({ phase: 'failed', service: label, method: property, error: String(error?.message ?? error) }); throw error; });
          emit({ phase: 'completed', service: label, method: property, result: preview(result) });
          return result;
        } catch (error) {
          emit({ phase: 'failed', service: label, method: property, error: String((error as Error)?.message ?? error) });
          throw error;
        }
      };
      return value;
    },
  });
  return Object.freeze(Object.fromEntries(Object.entries(services).map(([name, service]) =>
    [name, service && typeof service === 'object' ? wrap(service, name) : service])));
}
