import * as wllamaRuntime from '@wllama/wllama/esm/index.js';
import type { DecisionRequest, DecisionScorer, DecisionScores, ModelTurn, ModelTurnRequest } from '../contracts.js';
import { chatCompletionModelTurn, modelTools, type ChatTransport, type ChatTurnStats } from '../model/chat-completion.js';
import { probeBrowserGpu, type BrowserGpuCapability } from './gpu.js';
import { perRequestLora } from '../model/lora-scales.js';
import { nativeStateMessages } from '../model/openai-compatible.js';
import { coalescingScorer } from '../model/scoring.js';

type ModelMessage = { role: 'system' | 'user' | 'assistant' | 'tool'; content?: string | null;
  tool_calls?: unknown[]; tool_call_id?: string };
type LoadParams = { n_ctx: number; n_gpu_layers?: number; n_threads?: number; jinja: boolean;
  reasoning: boolean; chat_template?: string;
  lora_adapters?: { blob: Blob; scale: number }[]; lora_init_without_apply?: boolean;
  native_states?: { blob: Blob }[];
  progressCallback?: (progress: { loaded: number; total: number }) => void;
  signal?: AbortSignal };
/** The subset needed by natlang; applications may inject a preloaded Wllama instance. */
export type BrowserInferenceEngine = {
  isSupportWebGPU(): boolean;
  setCompat(compat: { wasm: string; worker: string }, mode?: 'safari' | 'firefox_safari'): void;
  isModelLoaded(): boolean;
  loadModelFromHF(model: { repo: string; file?: string; quant?: string }, params: LoadParams): Promise<void>;
  loadModelFromUrl(url: string, params: LoadParams): Promise<void>;
  loadModel(files: Blob[], params: LoadParams): Promise<void>;
  /** OpenAI-style chat completion; with `stream: true` an async iterator of chunks. */
  createChatCompletion(request: Record<string, unknown>): Promise<AsyncIterable<Record<string, unknown>> | Record<string, unknown>>;
  /** Decision readout: each reply's log-probability as the closed final assistant message (llama.cpp fork
   * `/v1/natlang/score`, in the vendored wllama). */
  createNatlangScore?(request: { items: Record<string, unknown>[] }): Promise<{ results: ({ log_probs: number[]; tokens: number[] } | { error: string })[] }>;
  getLoadedContextInfo?(): { n_ctx: number; n_layer: number };
  getModelMetadata?(): Record<string, unknown>;
  getWorkerResources?(): { compat: boolean; noWebGPU?: boolean };
  /** Feature detection of the vendored wllama build (ts-host/vendor/wllama). */
  readonly capabilities?: { readonly loraBlobs?: boolean; readonly nativeStates?: boolean; readonly natlangScore?: boolean };
  exit(): Promise<void>;
};

const Wllama = (wllamaRuntime as unknown as { Wllama: new (paths: { default: string },
  options: { allowOffline: boolean }) => BrowserInferenceEngine }).Wllama;

export type BrowserModelLoadOptions = {
  contextTokens?: number;
  /** Official model tool-call template, when the GGUF embeds a reduced template. */
  chatTemplate?: string;
  /** Auto-selects all layers on a capable adapter, otherwise CPU. Set 0 for CPU. */
  gpuLayers?: number;
  threads?: number;
  /** Tuned LoRA adapters (a release's GGUF exports, `loadBrowserRelease`), loaded at scale 0 with the model; each
   * request applies only its own call's stack (ts-host/vendor/wllama). */
  loraAdapters?: readonly { readonly id: string; readonly blob: Blob }[];
  /** Tuned soft instruction/skill states (a release's `.state.gguf` exports, `loadBrowserRelease`): exact native state
   * spans the model consumes wherever a message carries the state's sentinel (ts-host/vendor/wllama). */
  nativeStates?: readonly { readonly id: string; readonly blob: Blob }[];
  onProgress?: (progress: { loaded: number; total: number }) => void;
  signal?: AbortSignal;
};

export type BrowserModelDiagnostics = {
  loaded: boolean; supportsWebGPU: boolean; requestedGpuLayers: number | null;
  contextTokens: number | null; modelLayers: number | null;
  workerCompatibility: boolean | null; workerNoWebGPU: boolean | null;
  gpuCapability: BrowserGpuCapability | null; gpuSelectionReason: string | null;
  /** Wllama does not expose the actual number of offloaded layers. */
  actualGpuLayers: null;
};

function loadParams(options: BrowserModelLoadOptions) {
  if (options.gpuLayers !== undefined && (!Number.isInteger(options.gpuLayers) || options.gpuLayers < 0))
    throw new RangeError('gpuLayers must be a nonnegative integer');
  return { n_ctx: options.contextTokens ?? 4096, n_gpu_layers: options.gpuLayers ?? 99999,
    n_threads: options.threads, jinja: true, reasoning: false, chat_template: options.chatTemplate,
    ...(options.loraAdapters?.length ? { lora_adapters: options.loraAdapters.map(adapter => ({ blob: adapter.blob, scale: 0 })),
      lora_init_without_apply: true } : {}),
    ...(options.nativeStates?.length ? { native_states: options.nativeStates.map(state => ({ blob: state.blob })) } : {}),
    progressCallback: options.onProgress, signal: options.signal };
}

/** Remove natlang-only schema hints before sending the public JSON Schema to llama.cpp. */
function chatMessages(messages: unknown[]): ModelMessage[] {
  return messages.map((raw, index) => {
    if (!raw || typeof raw !== 'object') throw new TypeError(`model message ${index} must be an object`);
    const message = raw as Record<string, unknown>;
    if (!['system', 'user', 'assistant', 'tool'].includes(String(message.role)))
      throw new TypeError(`model message ${index} has an invalid role`);
    if (message.role !== 'assistant' || !Array.isArray(message.tool_calls))
      return message as ModelMessage;
    // The official LFM tool template expects argument mappings in prior assistant calls.
    // OpenAI-style responses carry JSON strings, so normalize history at the adapter edge.
    return { ...message, tool_calls: message.tool_calls.map((call: unknown) => {
      if (!call || typeof call !== 'object') return call;
      const record = call as Record<string, unknown>;
      const fn = record.function as Record<string, unknown> | undefined;
      if (!fn || typeof fn.arguments !== 'string') return call;
      let args: unknown;
      try { args = JSON.parse(fn.arguments); }
      catch { return call; }
      return { ...record, function: { ...fn, arguments: args } };
    }) } as ModelMessage;
  });
}

/** Keep the scope-eval surface and names intact; remove only host-private schema annotations. */
/** Tool definitions as the model sees them (shared with every transport). */
export const compileBrowserTools = modelTools;

/**
 * The in-page transport: wllama applies the chat template itself, so earlier tool calls carry argument objects
 * (llama-server does this conversion on its side).
 */
export function wllamaChatTransport(engine: BrowserInferenceEngine, tuned?: Record<string, unknown>,
  nativeStates?: ReadonlySet<string>): ChatTransport {
  return async (body, signal) => {
    const messages = body.messages as unknown[];
    try {
      return await engine.createChatCompletion({ ...body, ...tuned,
        messages: chatMessages(nativeStateMessages(messages, nativeStates ?? new Set())),
        cache_prompt: true, stream: true, abortSignal: signal });
    } catch (error) {
      if (String(error).includes('kv_cache_full'))
        throw new Error('model context is full; reduce prompt or tool schemas, or reload with a larger context', { cause: error });
      throw error;
    }
  };
}

/** GGUF inference in a browser worker; WebGPU offloads all layers when available. */
export class BrowserLocalModel {
  readonly engine: BrowserInferenceEngine;
  private readonly ownsEngine: boolean;
  private closed = false;
  private requestedGpuLayers: number | null = null;
  private requestedContextTokens: number | null = null;
  private gpuCapability: BrowserGpuCapability | null = null;
  private gpuSelectionReason: string | null = null;
  private readonly gpuProbe: () => Promise<BrowserGpuCapability>;
  private turnQueue: Promise<void> = Promise.resolve();
  /** Loaded tuned adapters: artifact ID -> adapter index (fixed for the loaded model). */
  private readonly loraIndex = new Map<string, number>();
  /** Loaded native state spans by ID (fixed for the loaded model). */
  private readonly nativeStates = new Set<string>();
  lastLoadMs: number | null = null;
  lastTurn: { durationMs: number; promptTokens: number | null; cachedTokens: number | null;
    completionTokens: number | null; toolSchemaBytes: number; retries: number; tokensPerSecond: number | null } | null = null;
  readonly turnHistory: NonNullable<BrowserLocalModel['lastTurn']>[] = [];

  constructor(options: { wasmUrl?: string; compatWasmUrl?: string; compatWorkerUrl?: string;
    firefoxGpuCompatibility?: boolean; engine?: BrowserInferenceEngine; allowOffline?: boolean;
    gpuProbe?: () => Promise<BrowserGpuCapability> } = {}) {
    this.gpuProbe = options.gpuProbe ?? (() => probeBrowserGpu({
      firefoxCompatibility: options.firefoxGpuCompatibility }));
    this.ownsEngine = !options.engine;
    this.engine = options.engine ?? new Wllama({ default: options.wasmUrl ??
      new URL('./wllama.wasm', import.meta.url).href },
      { allowOffline: options.allowOffline ?? true });
    if (this.ownsEngine) this.engine.setCompat({
      wasm: options.compatWasmUrl ?? new URL('./wllama-compat.wasm', import.meta.url).href,
      worker: options.compatWorkerUrl ?? new URL('./wllama-compat.js', import.meta.url).href,
    }, options.firefoxGpuCompatibility ? 'firefox_safari' : 'safari');
  }

  get loaded(): boolean { return !this.closed && this.engine.isModelLoaded(); }
  /** Browser capability, not a promise that a particular model fits in VRAM. */
  get supportsWebGPU(): boolean { return !this.closed && this.engine.isSupportWebGPU(); }

  get diagnostics(): BrowserModelDiagnostics {
    const info = this.loaded ? this.engine.getLoadedContextInfo?.() : undefined;
    const worker = this.engine.getWorkerResources?.();
    return { loaded: this.loaded, supportsWebGPU: this.supportsWebGPU,
      requestedGpuLayers: this.requestedGpuLayers,
      contextTokens: info?.n_ctx ?? this.requestedContextTokens, modelLayers: info?.n_layer ?? null,
      workerCompatibility: worker?.compat ?? null, workerNoWebGPU: worker?.noWebGPU ?? null,
      gpuCapability: this.gpuCapability, gpuSelectionReason: this.gpuSelectionReason,
      actualGpuLayers: null };
  }

  private async load(options: BrowserModelLoadOptions, action: (params: LoadParams) => Promise<void>): Promise<void> {
    if (this.closed) throw new Error('local model is closed');
    const gpuLayers = options.gpuLayers;
    const params = loadParams(options);
    if (options.loraAdapters?.length && !this.engine.capabilities?.loraBlobs)
      throw new Error('weights-unavailable: this wllama build cannot load LoRA adapters; use the vendored build (ts-host/vendor/wllama)');
    if (options.nativeStates?.length && !this.engine.capabilities?.nativeStates)
      throw new Error('weights-unavailable: this wllama build cannot load native states; use the vendored build (ts-host/vendor/wllama)');
    this.gpuCapability = await this.gpuProbe();
    if (gpuLayers === undefined) {
      params.n_gpu_layers = this.gpuCapability.usable ? 99999 : 0;
      this.gpuSelectionReason = this.gpuCapability.usable ?
        'Automatic full GPU offload requested' : `Automatic CPU fallback: ${this.gpuCapability.reason}`;
    } else {
      this.gpuSelectionReason = gpuLayers === 0 ? 'CPU selected explicitly' :
        `${gpuLayers} GPU layers requested explicitly`;
      if (gpuLayers > 0 && !this.gpuCapability.usable)
        throw new Error(`Cannot request GPU layers: ${this.gpuCapability.reason}`);
    }
    this.requestedGpuLayers = params.n_gpu_layers ?? null;
    this.requestedContextTokens = params.n_ctx;
    const started = performance.now();
    this.loraIndex.clear();
    this.nativeStates.clear();
    await action(params);
    for (const adapter of options.loraAdapters ?? []) this.loraIndex.set(adapter.id, this.loraIndex.size);
    for (const state of options.nativeStates ?? []) this.nativeStates.add(state.id);
    this.lastLoadMs = Math.round(performance.now() - started);
  }

  async loadFromHuggingFace(model: { repo: string; file?: string; quant?: string },
    options: BrowserModelLoadOptions = {}): Promise<void> {
    await this.load(options, params => this.engine.loadModelFromHF(model, params));
  }

  async loadFromUrl(url: string, options: BrowserModelLoadOptions = {}): Promise<void> {
    await this.load(options, params => this.engine.loadModelFromUrl(url, params));
  }

  async loadFiles(files: Blob[], options: BrowserModelLoadOptions = {}): Promise<void> {
    await this.load(options, params => this.engine.loadModel(files, params));
  }

  /** Run `work` after the turns and readouts queued before it (the in-page engine serves one request at a time). */
  private async queued<T>(work: () => Promise<T>): Promise<T> {
    const previous = this.turnQueue;
    let release!: () => void;
    this.turnQueue = new Promise<void>(resolve => { release = resolve; });
    await previous;
    try { return await work(); } finally { release(); }
  }

  /** Score decision items in one engine request; each carries its own call's adapter scales (`tuned`). */
  private readonly scoreItems = async (items: (DecisionRequest & { tuned?: Record<string, unknown> })[],
    signal?: AbortSignal): Promise<PromiseSettledResult<DecisionScores>[]> => {
    if (!this.loaded) throw new Error('load a local GGUF model before scoring a decision');
    if (!this.engine.createNatlangScore || !this.engine.capabilities?.natlangScore)
      throw new Error('decision-unsupported: this wllama build cannot score replies; use the vendored build (ts-host/vendor/wllama)');
    const body = { items: items.map(item => ({ ...item.tuned, continuations: item.options,
      messages: chatMessages(nativeStateMessages(item.messages, this.nativeStates)) })) };
    const response = await this.queued(async () => { signal?.throwIfAborted(); return this.engine.createNatlangScore!(body); });
    return items.map((item, index): PromiseSettledResult<DecisionScores> => {
      const result = response.results[index];
      if (!result || 'error' in result || result.log_probs.length !== item.options.length)
        return { status: 'rejected', reason: new Error(result && 'error' in result ? result.error : 'batched scoring returned no scores for an item') };
      return { status: 'fulfilled', value: { log_probs: result.log_probs, tokens: result.tokens } };
    });
  };

  /**
   * Decision readout (natlang `decide`): readouts arriving together are scored in one request; each item's adapter
   * scales are its own call's, read before coalescing. Carried by `turn`, so a runtime given `model.turn` scores.
   */
  readonly decide: DecisionScorer = (() => {
    const tag = (item: DecisionRequest) => {
      const tuned = perRequestLora(this.loraIndex, 'the in-page model');
      return tuned ? { ...item, tuned } : item;
    };
    const many = (items: DecisionRequest[], signal?: AbortSignal) => this.scoreItems(items, signal);
    const coalesced = coalescingScorer(Object.assign(async (item: DecisionRequest, signal?: AbortSignal) => {
      const [result] = await many([item], signal);
      if (result!.status === 'rejected') throw result!.reason;
      return result!.value;
    }, { scoreMany: many }));
    return Object.assign((item: DecisionRequest, signal?: AbortSignal) => coalesced(tag(item), signal),
      { scoreMany: (items: DecisionRequest[], signal?: AbortSignal) => many(items.map(tag), signal) });
  })();

  readonly turn = Object.assign(async (request: ModelTurnRequest, signal?: AbortSignal): Promise<ModelTurn> => {
    if (!this.loaded) throw new Error('load a local GGUF model before running a natural-language lambda');
    // The calling invocation's own stack, read before queueing.
    const tuned = perRequestLora(this.loraIndex, 'the in-page model');
    return this.queued(async () => {
      const toolSchemaBytes = new TextEncoder().encode(JSON.stringify(modelTools(request.tools))).length;
      let stats: ChatTurnStats | undefined;
      const turn = await chatCompletionModelTurn(wllamaChatTransport(this.engine, tuned, this.nativeStates),
        { onTurn: value => { stats = value; } })(request, signal);
      if (stats) {
        this.lastTurn = { ...stats, toolSchemaBytes, tokensPerSecond: stats.completionTokens !== null && stats.durationMs > 0 ?
          Math.round(stats.completionTokens * 1000 / stats.durationMs * 10) / 10 : null };
        this.turnHistory.push(this.lastTurn);
      }
      return turn;
    });
  }, { decide: this.decide });

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.ownsEngine) await this.engine.exit();
  }
}

export type BrowserModelSource =
  | { kind: 'url'; url: string; id?: string; templateUrl?: string; chatTemplate?: string }
  | { kind: 'files'; files: Blob[]; id?: string; templateUrl?: string; chatTemplate?: string }
  | { kind: 'huggingface'; repo: string; file?: string; quant?: string; id?: string; templateUrl?: string; chatTemplate?: string };
export type BrowserModelStatus = { id: string; diagnostics: BrowserModelDiagnostics; loadMs: number | null; gpuFallbackReason: string | null };
export type LoadedBrowserModel = { model: BrowserLocalModel; status: BrowserModelStatus };

/**
 * Load a local GGUF model: fetch its chat template when one is published separately, run
 * single-threaded when the page is not cross-origin isolated, and fall back to CPU when an automatic
 * GPU load fails (unless `cpuFallback` is false or GPU layers were chosen explicitly).
 */
export async function loadBrowserLocalModel(source: BrowserModelSource,
  options: BrowserModelLoadOptions & { cpuFallback?: boolean; engine?: ConstructorParameters<typeof BrowserLocalModel>[0] } = {}): Promise<LoadedBrowserModel> {
  if (source.kind === 'files' && !source.files.length) throw new Error('select at least one GGUF file');
  const template = source.chatTemplate ?? (source.templateUrl ? await fetch(source.templateUrl).then(response => {
    if (!response.ok) throw new Error(`model template unavailable: ${source.templateUrl}`);
    return response.text();
  }) : undefined);
  const { cpuFallback = true, engine, ...given } = options;
  const loadOptions: BrowserModelLoadOptions = { ...given, ...(template ? { chatTemplate: template } : {}),
    ...(globalThis.crossOriginIsolated ? {} : { threads: given.threads ?? 1 }) };
  if (!Number.isInteger(loadOptions.contextTokens ?? 4096) || (loadOptions.contextTokens ?? 4096) < 512)
    throw new RangeError('contextTokens must be an integer of at least 512');
  const attempt = async (override: Partial<BrowserModelLoadOptions> = {}): Promise<{ model: BrowserLocalModel | null; reason: string | null }> => {
    const candidate = new BrowserLocalModel(engine);
    try {
      const params = { ...loadOptions, ...override };
      if (source.kind === 'url') await candidate.loadFromUrl(source.url, params);
      else if (source.kind === 'files') await candidate.loadFiles(source.files, params);
      else await candidate.loadFromHuggingFace({ repo: source.repo, file: source.file, quant: source.quant }, params);
      return { model: candidate, reason: null };
    } catch (error) {
      const gpuAttempted = candidate.diagnostics.requestedGpuLayers === 99999;
      try { await candidate.close(); } catch { /* preserve the load error */ }
      if (cpuFallback && loadOptions.gpuLayers === undefined && gpuAttempted && override.gpuLayers === undefined)
        return { model: null, reason: String(error) };
      throw error;
    }
  };
  let first = await attempt(), gpuFallbackReason: string | null = null;
  if (!first.model) {
    gpuFallbackReason = first.reason;
    first = await attempt({ gpuLayers: 0 });
    if (!first.model) throw new Error('CPU model load failed');
  }
  const model = first.model;
  const id = source.id ?? (source.kind === 'url' ? source.url : source.kind === 'files' ?
    source.files.map(file => (file as File).name ?? 'GGUF').join(',') : `${source.repo}/${source.file ?? source.quant ?? ''}`);
  return { model, status: { id, diagnostics: model.diagnostics, loadMs: model.lastLoadMs, gpuFallbackReason } };
}
