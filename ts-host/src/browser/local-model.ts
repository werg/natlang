import * as wllamaRuntime from '@wllama/wllama/esm/index.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

type ModelMessage = { role: 'system' | 'user' | 'assistant' | 'tool'; content?: string | null;
  tool_calls?: unknown[]; tool_call_id?: string };
type ModelTool = { type: 'function'; function: { name: string; description?: string;
  parameters?: Record<string, unknown> } };
type ModelResponse = { choices: Array<{ finish_reason?: string | null; message: {
  content?: string | null; tool_calls?: Array<{ type: string; function: { name: string; arguments: string } }> } }>;
  usage?: { completion_tokens?: number }; [key: string]: unknown };
type LoadParams = { n_ctx: number; n_gpu_layers?: number; n_threads?: number; jinja: boolean;
  reasoning: boolean; progressCallback?: (progress: { loaded: number; total: number }) => void;
  signal?: AbortSignal };
type ModelCompletionRequest = { messages: ModelMessage[]; tools: ModelTool[]; tool_choice: 'auto';
  max_tokens: number; temperature: number; seed?: number; abortSignal?: AbortSignal };

/** The subset needed by natlang; applications may inject a preloaded Wllama instance. */
export type BrowserInferenceEngine = {
  isModelLoaded(): boolean;
  loadModelFromHF(model: { repo: string; file?: string; quant?: string }, params: LoadParams): Promise<void>;
  loadModelFromUrl(url: string, params: LoadParams): Promise<void>;
  loadModel(files: Blob[], params: LoadParams): Promise<void>;
  createChatCompletion(request: ModelCompletionRequest): Promise<ModelResponse>;
  exit(): Promise<void>;
};

const Wllama = (wllamaRuntime as unknown as { Wllama: new (paths: { default: string },
  options: { allowOffline: boolean }) => BrowserInferenceEngine }).Wllama;

export type BrowserModelLoadOptions = {
  contextTokens?: number;
  gpuLayers?: number;
  threads?: number;
  onProgress?: (progress: { loaded: number; total: number }) => void;
  signal?: AbortSignal;
};

function loadParams(options: BrowserModelLoadOptions) {
  return { n_ctx: options.contextTokens ?? 4096, n_gpu_layers: options.gpuLayers,
    n_threads: options.threads, jinja: true, reasoning: false,
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
    return message as ModelMessage;
  });
}

function chatTools(tools: unknown[]): ModelTool[] {
  return tools.map((raw, index) => {
    const tool = publicSchema(raw) as ModelTool;
    if (tool?.type !== 'function' || typeof tool.function?.name !== 'string')
      throw new TypeError(`model tool ${index} is invalid`);
    return tool;
  });
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

/** GGUF inference in a browser worker; model weights stay in browser storage. */
export class BrowserLocalModel {
  readonly engine: BrowserInferenceEngine;
  private readonly ownsEngine: boolean;
  private closed = false;

  constructor(options: { wasmUrl?: string; engine?: BrowserInferenceEngine; allowOffline?: boolean } = {}) {
    this.ownsEngine = !options.engine;
    this.engine = options.engine ?? new Wllama({ default: options.wasmUrl ??
      new URL('./wllama.wasm', import.meta.url).href },
      { allowOffline: options.allowOffline ?? true });
  }

  get loaded(): boolean { return !this.closed && this.engine.isModelLoaded(); }

  async loadFromHuggingFace(model: { repo: string; file?: string; quant?: string },
    options: BrowserModelLoadOptions = {}): Promise<void> {
    if (this.closed) throw new Error('local model is closed');
    await this.engine.loadModelFromHF(model, loadParams(options));
  }

  async loadFromUrl(url: string, options: BrowserModelLoadOptions = {}): Promise<void> {
    if (this.closed) throw new Error('local model is closed');
    await this.engine.loadModelFromUrl(url, loadParams(options));
  }

  async loadFiles(files: Blob[], options: BrowserModelLoadOptions = {}): Promise<void> {
    if (this.closed) throw new Error('local model is closed');
    await this.engine.loadModel(files, loadParams(options));
  }

  readonly turn = async (request: ModelTurnRequest, signal?: AbortSignal): Promise<ModelTurn> => {
    if (!this.loaded) throw new Error('load a local GGUF model before running a natural-language lambda');
    const response = await this.engine.createChatCompletion({ messages: chatMessages(request.messages),
      tools: chatTools(request.tools), tool_choice: 'auto', max_tokens: request.max_tokens,
      temperature: request.temperature, ...(request.seed === null ? {} : { seed: request.seed }),
      abortSignal: signal });
    return localModelTurn(response);
  };

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    if (this.ownsEngine) await this.engine.exit();
  }
}
