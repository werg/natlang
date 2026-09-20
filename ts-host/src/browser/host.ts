import { TypeScriptEnvironment } from './environment.js';
import { NativeRuntime, type NativeStream } from '../native/runtime.js';
import { NativeToolAgent, type NativeReviewOptions } from '../native/agent.js';
import { checkedDefinitions, type NativeDefinition } from '../native/codebase.js';
import { buildPending, coerce, dump, type Pending } from '../native/values.js';
import { TypeEnv } from '../native/types.js';
import type { ModelTurn, ModelTurnRequest, RunOptions } from '../runtime.js';

export type BrowserRunRequest = {
  source: { kind: 'program'; program: Record<string, unknown> } |
    { kind: 'definitions'; entries: Record<string, NativeDefinition>; root: string };
  inputs?: Record<string, unknown>;
  streams?: { over?: AsyncIterable<unknown> | Iterable<unknown> };
  modelTurn?: (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;
  review?: NativeReviewOptions;
  validationFeedback?: 'caller' | 'local';
  capabilities?: Record<string, (args: unknown[]) => unknown>;
  options?: RunOptions;
  mapWorkers?: number;
  parallelMapSafe?: boolean;
  signal?: AbortSignal;
  timeoutMs?: number;
};

/** Browser host for in-memory natlang programs, using the same reducer and tool agent as Node. */
export class BrowserNatlangHost {
  readonly environment: TypeScriptEnvironment;
  private running = false;
  private closed = false;
  private readonly ownsEnvironment: boolean;

  constructor(options: { environment?: TypeScriptEnvironment; host?: object; mode?: 'fresh' | 'retained' } = {}) {
    this.ownsEnvironment = !options.environment;
    this.environment = options.environment ?? new TypeScriptEnvironment({ host: options.host, mode: options.mode });
  }

  async run(request: BrowserRunRequest): Promise<{ outcome: { kind: string; path: string; detail: string };
    value: unknown; emitted: unknown[]; trace: Record<string, unknown>[]; run_id: string }> {
    if (this.closed) throw new Error('natlang host is disposed');
    if (this.running) throw new Error('concurrent runs cannot share a native eval environment');
    if (request.signal?.aborted) throw new Error('natlang run aborted');
    if (request.mapWorkers !== undefined && (!Number.isInteger(request.mapWorkers) || request.mapWorkers < 1))
      throw new RangeError('mapWorkers must be positive');
    this.running = true;
    let runtime: NativeRuntime | undefined;
    try {
      const root: Pending = request.source.kind === 'program' ? buildPending(request.source.program) :
        checkedDefinitions(request.source.entries, request.source.root).instantiate(request.inputs);
      if (request.source.kind === 'program' && request.inputs) {
        if (root.nodeKind !== 'lambda' || root.type.kind !== 'lambda') throw new TypeError('inputs require a Lambda program');
        const env = new TypeEnv().child(root.types);
        for (const [name, value] of Object.entries(request.inputs)) {
          const field = root.type.params.fields.find(item => item.name === name);
          if (!field) throw new TypeError(`${name} is not a program parameter`);
          root.args[name] = coerce(value, field.type, env, `args/${name}`);
        }
      }
      if (request.streams?.over && root.nodeKind !== 'fold') throw new TypeError('streams require a root Fold');
      const source = request.streams?.over;
      const iterator = source && (Symbol.asyncIterator in source ? source[Symbol.asyncIterator]() :
        (async function* () { yield* source as Iterable<unknown>; })()[Symbol.asyncIterator]());
      const stream: NativeStream | undefined = iterator && { async poll() {
        try { const step = await iterator.next(); return step.done || step.value === '$close' ?
          { kind: 'closed' } : { kind: 'item', value: step.value }; }
        catch (error) { return { kind: 'failed', detail: error instanceof Error ? error.message : String(error) }; }
      } };
      const agent = request.modelTurn ? new NativeToolAgent(request.modelTurn, {
        maxTurns: request.options?.model?.max_turns, maxTokens: request.options?.model?.max_tokens,
        turnTokens: request.options?.model?.turn_tokens, temperature: request.options?.model?.temperature,
        maxSeconds: request.options?.model?.max_seconds, validationFeedback: request.validationFeedback,
        review: request.review }) : undefined;
      const runId = request.options?.run_id ?? globalThis.crypto?.randomUUID?.() ?? String(Date.now());
      runtime = new NativeRuntime({ environment: this.environment as never, stream,
        agent: agent ? session => agent.run(session) : undefined,
        capabilities: request.capabilities, maxEpisodes: request.options?.max_episodes,
        maxDepth: request.options?.max_depth, mapWorkers: request.mapWorkers,
        parallelModelSafe: request.parallelMapSafe, runId, signal: request.signal,
        timeoutMs: request.timeoutMs, seedPolicy: request.options?.seed?.mode ?
          { mode: request.options.seed.mode, root: request.options.seed.root } : undefined });
      let timer: ReturnType<typeof setTimeout> | undefined;
      let abortListener: (() => void) | undefined;
      const interruption = new Promise<never>((_, reject) => {
        if (request.timeoutMs !== undefined)
          timer = setTimeout(() => reject(new Error('natlang run timed out; external effects may have occurred')),
            request.timeoutMs);
        if (request.signal) {
          abortListener = () => reject(new Error('natlang run aborted; external effects may have occurred'));
          request.signal.addEventListener('abort', abortListener, { once: true });
          if (request.signal.aborted) abortListener();
        }
      });
      let result;
      try { result = await Promise.race([runtime.runRoot(root), interruption]); }
      finally {
        if (timer) clearTimeout(timer);
        if (abortListener) request.signal?.removeEventListener('abort', abortListener);
      }
      return { outcome: { kind: result.outcome.kind, path: result.outcome.path, detail: result.outcome.detail },
        value: dump(result.value), emitted: result.emitted,
        trace: runtime.trace.events as Record<string, unknown>[], run_id: runId };
    } finally { runtime?.close(); this.running = false; }
  }

  close(): void { this.closed = true; if (this.ownsEnvironment) this.environment.close(); }
}
