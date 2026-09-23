import * as wllamaRuntime from '@wllama/wllama/esm/index.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { probeBrowserGpu, type BrowserGpuCapability } from './gpu.js';

type ModelMessage = { role: 'system' | 'user' | 'assistant' | 'tool'; content?: string | null;
  tool_calls?: unknown[]; tool_call_id?: string };
type ModelTool = { type: 'function'; function: { name: string; description?: string;
  parameters?: Record<string, unknown> } };
type ModelResponse = { choices: Array<{ finish_reason?: string | null; message: {
  content?: string | null; tool_calls?: Array<{ id?: string; type: string; function: { name: string; arguments: string } }> } }>;
  usage?: { completion_tokens?: number; prompt_tokens?: number;
    prompt_tokens_details?: { cached_tokens?: number } }; [key: string]: unknown };
type LoadParams = { n_ctx: number; n_gpu_layers?: number; n_threads?: number; jinja: boolean;
  reasoning: boolean; chat_template?: string;
  progressCallback?: (progress: { loaded: number; total: number }) => void;
  signal?: AbortSignal };
type ModelCompletionRequest = { messages: ModelMessage[]; tools: ModelTool[]; tool_choice: 'auto';
  max_tokens?: number; temperature?: number; seed?: number; abortSignal?: AbortSignal;
  cache_prompt?: boolean };

/** The subset needed by natlang; applications may inject a preloaded Wllama instance. */
export type BrowserInferenceEngine = {
  isSupportWebGPU(): boolean;
  setCompat(compat: { wasm: string; worker: string }, mode?: 'safari' | 'firefox_safari'): void;
  isModelLoaded(): boolean;
  loadModelFromHF(model: { repo: string; file?: string; quant?: string }, params: LoadParams): Promise<void>;
  loadModelFromUrl(url: string, params: LoadParams): Promise<void>;
  loadModel(files: Blob[], params: LoadParams): Promise<void>;
  createChatCompletion(request: ModelCompletionRequest): Promise<ModelResponse>;
  getLoadedContextInfo?(): { n_ctx: number; n_layer: number };
  getModelMetadata?(): Record<string, unknown>;
  getWorkerResources?(): { compat: boolean; noWebGPU?: boolean };
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
    progressCallback: options.onProgress, signal: options.signal };
}

/** Remove natlang-only schema hints before sending the public JSON Schema to llama.cpp. */
function publicSchema(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(publicSchema);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !key.startsWith('x-natlang') && key !== 'x-optional')
    .map(([key, item]) => [key, publicSchema(item)]));
  return value;
}

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
export function compileBrowserTools(rawTools: unknown[]): ModelTool[] {
  const tools: ModelTool[] = [];
  for (const [toolIndex, raw] of rawTools.entries()) {
    const tool = raw as ModelTool;
    if (tool?.type !== 'function' || typeof tool.function?.name !== 'string')
      throw new TypeError(`model tool ${toolIndex} is invalid`);
    tools.push(publicSchema(tool) as ModelTool);
  }
  return tools;
}

/** Adapt one local model response to the host's template-independent model turn. */
export function localModelTurn(response: ModelResponse): ModelTurn {
  const choice = response.choices[0];
  if (!choice) throw new Error('local model returned no choice');
  if (choice.finish_reason === 'length') throw new Error('local model reached its token limit before finishing the turn');
  const rawCalls = choice.message.tool_calls ?? [];
  const calls = rawCalls.map((call, index): [string, Record<string, unknown>] => {
    if (call.type !== 'function' || !call.function?.name) throw new Error(`local model tool call ${index} has no function`);
    let args: unknown;
    try { args = JSON.parse(call.function.arguments); }
    catch { throw new Error(`local model tool call ${index} has invalid JSON arguments`); }
    if (!args || typeof args !== 'object' || Array.isArray(args))
      throw new Error(`local model tool call ${index} needs object arguments`);
    return [call.function.name, args as Record<string, unknown>];
  });
  return { calls, text: choice.message.content ?? '', raw_calls: rawCalls,
    completion_tokens: response.usage?.completion_tokens,
    raw_response: response as unknown as Record<string, unknown> };
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
    await action(params);
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

  readonly turn = async (request: ModelTurnRequest, signal?: AbortSignal): Promise<ModelTurn> => {
    if (!this.loaded) throw new Error('load a local GGUF model before running a natural-language lambda');
    const previous = this.turnQueue;
    let release!: () => void;
    this.turnQueue = new Promise<void>(resolve => { release = resolve; });
    await previous;
    try {
      if (signal?.aborted) throw new Error('local model turn aborted');
      const tools = compileBrowserTools(request.tools);
      const schemaBytes = new TextEncoder().encode(JSON.stringify(tools)).length;
      const baseMessages = chatMessages(request.messages);
      let messages = baseMessages;
      let retries = 0, promptTokens = 0, completionTokens = 0, cachedTokens = 0;
      let hasPrompt = false, hasCompletion = false, hasCached = false;
      const started = performance.now();
      while (true) {
        let response: ModelResponse;
        try {
          response = await this.engine.createChatCompletion({ messages,
            tools, tool_choice: 'auto', cache_prompt: true,
            ...(request.max_tokens === null ? {} : { max_tokens: request.max_tokens }),
            ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
            ...(request.seed === null ? {} : { seed: request.seed }),
            abortSignal: signal });
        } catch (error) {
          if (String(error).includes('kv_cache_full'))
            throw new Error('model context is full; reduce prompt or tool schemas, or reload with a larger context',
              { cause: error });
          throw error;
        }
        if (response.usage?.prompt_tokens !== undefined) {
          promptTokens += response.usage.prompt_tokens; hasPrompt = true;
        }
        if (response.usage?.completion_tokens !== undefined) {
          completionTokens += response.usage.completion_tokens; hasCompletion = true;
        }
        if (response.usage?.prompt_tokens_details?.cached_tokens !== undefined) {
          cachedTokens += response.usage.prompt_tokens_details.cached_tokens; hasCached = true;
        }
        try {
          const turn = localModelTurn(response);
          const durationMs = Math.round(performance.now() - started);
          this.lastTurn = { durationMs, promptTokens: hasPrompt ? promptTokens : null,
            completionTokens: hasCompletion ? completionTokens : null,
            cachedTokens: hasCached ? cachedTokens : null, toolSchemaBytes: schemaBytes, retries,
            tokensPerSecond: hasCompletion && durationMs > 0 ?
              Math.round(completionTokens * 1000 / durationMs * 10) / 10 : null };
          this.turnHistory.push(this.lastTurn);
          return { ...turn, prompt_tokens: hasPrompt ? promptTokens : undefined,
            completion_tokens: hasCompletion ? completionTokens : undefined };
        } catch (error) {
          if (retries >= 1 || signal?.aborted || response.choices[0]?.finish_reason === 'length') throw error;
          retries++;
          messages = [...baseMessages, { role: 'assistant', content: response.choices[0]?.message.content ?? '' },
            { role: 'user', content: 'The last tool call was malformed. Call one offered tool with valid JSON object arguments. Do not change the task or invent a new tool.' }];
        }
      }
    } finally { release(); }
  };

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
