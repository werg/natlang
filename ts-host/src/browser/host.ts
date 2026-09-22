import { TypeScriptEnvironment } from './environment.js';
import { NativeRuntime, type NativeStream } from '../native/runtime.js';
import { NativeToolAgent } from '../native/agent.js';
import { checkedDefinitions, type NativeDefinition } from '../native/codebase.js';
import { buildPending, coerce, dump, type Pending } from '../native/values.js';
import { TypeEnv } from '../native/types.js';
import type { BrowserLocalModel } from './local-model.js';
import { loadFunctionFiles } from './source.js';

export type BrowserModelTurnRequest = { messages: unknown[]; tools: unknown[]; temperature: number;
  seed: number | null; max_tokens: number | null };
export type BrowserModelTurn = { calls?: [string, Record<string, unknown>][]; text?: string;
  raw_calls?: unknown[]; completion_tokens?: number;
  value_confidence?: (number | { geometric_mean?: number } | null)[];
  raw_response?: Record<string, unknown> };
export type BrowserRunOptions = { seed?: { mode?: 'compatibility' | 'derived' | 'backend'; root?: number;
  version?: string }; model?: { temperature?: number; max_turns?: number; max_tokens?: number;
  max_seconds?: number; turn_tokens?: number; segment_turns?: number | null;
  segment_messages?: number | null; tool_schema?: 'tools-v3' | 'tools-v4' }; world_seed?: number;
  max_episodes?: number; max_depth?: number; max_actions?: number;
  max_tool_calls?: number; run_id?: string };
export type BrowserReviewOptions = { driver?: (request: BrowserModelTurnRequest) =>
    Promise<BrowserModelTurn> | BrowserModelTurn; threshold?: number;
  scope?: 'values' | 'actions'; withdrawalPolicy?: 'caller' | 'retry';
  prompt?: 'baseline' | 'repeat_instructions' | 'checklist';
  order?: 'reason_first' | 'decision_first' };

export type BrowserRunRequest = {
  source: { kind: 'program'; program: Record<string, unknown> } |
    { kind: 'definitions'; entries: Record<string, NativeDefinition>; root: string } |
    { kind: 'files'; root: string; files: Record<string, string> };
  inputs?: Record<string, unknown>;
  streams?: { over?: AsyncIterable<unknown> | Iterable<unknown> };
  modelTurn?: (request: BrowserModelTurnRequest) => Promise<BrowserModelTurn> | BrowserModelTurn;
  review?: BrowserReviewOptions;
  validationFeedback?: 'caller' | 'local';
  capabilities?: Record<string, (args: unknown[]) => unknown>;
  options?: BrowserRunOptions;
  mapWorkers?: number;
  parallelMapSafe?: boolean;
  signal?: AbortSignal;
  timeoutMs?: number;
};

/** Browser host for natlang programs, using the same reducer and tool agent as Node. */
export class BrowserNatlangHost {
  readonly environment: TypeScriptEnvironment;
  readonly model?: BrowserLocalModel;
  private running = false;
  private closed = false;
  private readonly ownsEnvironment: boolean;

  constructor(options: { environment?: TypeScriptEnvironment; host?: object; mode?: 'fresh' | 'retained';
    model?: BrowserLocalModel } = {}) {
    this.ownsEnvironment = !options.environment;
    this.environment = options.environment ?? new TypeScriptEnvironment({ host: options.host, mode: options.mode });
    this.model = options.model;
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
    const modelAbort = new AbortController();
    const abortModel = () => modelAbort.abort();
    request.signal?.addEventListener('abort', abortModel, { once: true });
    try {
      const root: Pending = request.source.kind === 'program' ? buildPending(request.source.program) :
        request.source.kind === 'files' ? loadFunctionFiles(request.source.root, request.source.files) :
          checkedDefinitions(request.source.entries, request.source.root).instantiate(request.inputs);
      if (request.source.kind !== 'definitions' && request.inputs) {
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
      const modelTurn = request.modelTurn ?? (this.model ?
        (turn: BrowserModelTurnRequest) => this.model!.turn(turn, modelAbort.signal) : undefined);
      const agent = modelTurn ? new NativeToolAgent(modelTurn, {
        maxTurns: request.options?.model?.max_turns, maxTokens: request.options?.model?.max_tokens,
        turnTokens: request.options?.model?.turn_tokens, temperature: request.options?.model?.temperature,
        segmentTurns: request.options?.model?.segment_turns,
        segmentMessages: request.options?.model?.segment_messages,
        maxSeconds: request.options?.model?.max_seconds, validationFeedback: request.validationFeedback,
        review: request.review, toolSchema: request.options?.model?.tool_schema }) : undefined;
      const runId = request.options?.run_id ?? globalThis.crypto?.randomUUID?.() ?? String(Date.now());
      runtime = new NativeRuntime({ environment: this.environment as never, stream,
        agent: agent ? session => agent.run(session) : undefined,
        capabilities: request.capabilities, maxEpisodes: request.options?.max_episodes,
        maxDepth: request.options?.max_depth, maxActions: request.options?.max_actions,
        maxToolCalls: request.options?.max_tool_calls, mapWorkers: request.mapWorkers,
        parallelModelSafe: request.parallelMapSafe, runId, signal: request.signal,
        timeoutMs: request.timeoutMs, seedPolicy: request.options?.seed?.mode ?
          { mode: request.options.seed.mode, root: request.options.seed.root } : undefined });
      let timer: ReturnType<typeof setTimeout> | undefined;
      let abortListener: (() => void) | undefined;
      const interruption = new Promise<never>((_, reject) => {
        if (request.timeoutMs !== undefined)
          timer = setTimeout(() => { abortModel();
            reject(new Error('natlang run timed out; external effects may have occurred')); }, request.timeoutMs);
        if (request.signal) {
          abortListener = () => { abortModel();
            reject(new Error('natlang run aborted; external effects may have occurred')); };
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
    } finally { abortModel(); request.signal?.removeEventListener('abort', abortModel);
      runtime?.close(); this.running = false; }
  }

  close(): void { this.closed = true; if (this.ownsEnvironment) this.environment.close(); }
}
