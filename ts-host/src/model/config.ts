import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { defaultNatlangConfigDirectory } from '../package/store.js';
import { DEFAULT_MODEL_RELEASE } from '../model-default.js';
import { fingerprint } from '../adaptation/identity.js';
import type { ModelConfig, NatlangRuntimeOptions } from '../runtime/runtime.js';
import { nativeEndpoint, validateNativeSettings, type NativeModelSettings } from './native-managed.js';

export type BatchingSettings = {
  /** server-continuous (default), explicit-batch or serial. */
  mode?: 'server-continuous' | 'explicit-batch' | 'serial';
  maxConcurrent?: number;
  /** Coalescing window in ms (default 2 for explicit-batch, 0 otherwise). */
  coalesceMs?: number;
  /** Serve turns of running calls before new calls (default true). */
  priority?: boolean;
  /** Optional explicit-batch scoring endpoint: a URL, or true for `{endpoint}/v1/natlang/score`. Explicit-batch mode only. */
  scoreEndpoint?: string | boolean;
};

export type ModelProfile = {
  /** Match the prompt rendering declared by the trained weights; unadapted models use full. */
  promptRendering?: import('../native/system-prompts.js').PromptRendering;
  endpoint?: string; chatCompletionsUrl?: string; provider?: string; model?: string; apiKeyEnv?: string;
  /** The endpoint is a native Neuralese reference server: tuned weights, native state and soft instruction/skill
   * states run through its exact native path (docs/TUNING.md). */
  neuraleseServer?: boolean;
  /** A natlang-owned native reference server for a pinned native checkpoint (model/native-managed.ts): started on the
   * first turn, or reused when one already serves that checkpoint. Written by `natlang setup --native SOURCE`. */
  native?: NativeModelSettings;
  /** Profile that answers a call this profile's model fails (`natlang run|call|ask`; calls/tiers.ts tier 0). */
  escalate?: string;
  /** The endpoint is a vLLM server with LoRA enabled: each tuned call is routed to its whole-stack LoRA by name. */
  vllm?: boolean;
  headers?: Record<string, string>; request?: Record<string, unknown>;
  piOptions?: Record<string, unknown>; piPayload?: Record<string, unknown>;
  piMode?: 'native' | 'simple'; modelOptions?: Record<string, unknown>;
  /**
   * Managed server settings. `contextTokens` is per slot. `parallel` is the slot count; unset, it is sized from memory
   * (model/server-slots.ts), using `memoryBudgetMiB` (default half of available memory) and `kvBytesPerToken`.
   */
  local?: { contextTokens?: number; gpuLayers?: number; parallel?: number; cacheRamMiB?: number; args?: string[];
    memoryBudgetMiB?: number; kvBytesPerToken?: number;
    /** A GGUF to serve instead of the pinned default, its chat template and model ID (`natlang setup --native SOURCE
     * --gguf TYPE` writes these); NATLANG_MODEL_PATH and NATLANG_TEMPLATE still override them. */
    modelPath?: string; template?: string; modelId?: string;
    /** llama-server executable for this profile (e.g. a CUDA build); NATLANG_LLAMA_SERVER still overrides it. */
    server?: string };
  /** Model requests in flight at once from this process (turns and decision scoring together). A managed local server
   * defaults to its slot count, other backends to unlimited. `batching.maxConcurrent` takes precedence. */
  concurrency?: number;
  /** How this backend turns concurrent requests into batches (model/scheduler.ts). */
  batching?: BatchingSettings;
  runtime?: Omit<ModelConfig, 'driver'> & { seed?: NatlangRuntimeOptions['seed'] };
};

type Common = { promptRendering?: ModelProfile['promptRendering']; headers?: Record<string, string>; runtime?: ModelProfile['runtime']; concurrency?: number; batching?: BatchingSettings };
export type ResolvedModelChoice =
  | (Common & { kind: 'managed-local'; model: string; request?: Record<string, unknown>; local?: ModelProfile['local'] })
  | (Common & { kind: 'external'; endpoint: string; chatCompletionsUrl?: string; model: string; apiKeyEnv: string; request?: Record<string, unknown>;
    neuraleseServer?: true; vllm?: true; managedNative?: NativeModelSettings })
  | (Common & { kind: 'pi-provider'; provider: string; model: string; apiKeyEnv?: string;
    piOptions?: Record<string, unknown>; piPayload?: Record<string, unknown>;
    piMode: 'native' | 'simple'; modelOptions?: Record<string, unknown> });

export type ModelSelectionOverrides = { provider?: string; model?: string };
export type LoadedModelConfiguration = { name: string; configPath: string; profile: ModelProfile; choice: ResolvedModelChoice;
  /** Named checkpoint pointers for `natlang setup --native NAME` (config `nativeSources`). */
  nativeSources: Record<string, string> };

export function resolveModelChoice(profile: ModelProfile): ResolvedModelChoice {
  if (profile.promptRendering !== undefined && !['full', 'program-header'].includes(profile.promptRendering))
    throw new Error('promptRendering must be full or program-header');
  if (profile.runtime && (typeof profile.runtime !== 'object' || Array.isArray(profile.runtime)))
    throw new TypeError('profile.runtime must be an object');
  if (profile.runtime?.review && 'driver' in profile.runtime.review)
    throw new Error('profile.runtime.review.driver is programmatic only');
  if (profile.provider && profile.endpoint) throw new Error('model profile cannot set both provider and endpoint');
  if (profile.escalate !== undefined && (typeof profile.escalate !== 'string' || !profile.escalate))
    throw new Error('escalate must name another model profile');
  if (profile.native) {
    if (profile.provider || profile.endpoint || profile.chatCompletionsUrl || profile.local || profile.vllm)
      throw new Error('a native profile cannot also set provider, endpoint, chatCompletionsUrl, local or vllm');
    validateNativeSettings(profile.native);
  }
  if (profile.chatCompletionsUrl !== undefined && !profile.endpoint)
    throw new Error('chatCompletionsUrl requires an external endpoint metadata root');
  if (profile.chatCompletionsUrl !== undefined) {
    let target: URL;
    try { target = new URL(profile.chatCompletionsUrl); }
    catch { throw new Error('chatCompletionsUrl must be an absolute HTTP(S) URL'); }
    if (!['http:', 'https:'].includes(target.protocol) || target.username || target.password || target.hash || target.search)
      throw new Error('chatCompletionsUrl must be an absolute HTTP(S) URL without credentials or query parameters');
  }
  if (profile.provider && !profile.model) throw new Error('Pi provider profile needs a model ID');
  if (profile.provider && profile.request && Object.keys(profile.request).length)
    throw new Error('Pi provider profiles do not support raw request fields; use an endpoint profile for guided wire controls');
  for (const key of ['apiKey', 'signal', 'fetch', 'onPayload', 'onResponse', 'transformHeaders'])
    if (profile.piOptions && Object.hasOwn(profile.piOptions, key))
      throw new Error(`piOptions.${key} cannot be set in a model profile; use an environment variable or the programmatic Pi backend`);
  if (!profile.provider && (profile.piOptions || profile.piPayload || profile.modelOptions || profile.piMode))
    throw new Error('piMode, piOptions, piPayload and modelOptions require a Pi provider profile');
  if (profile.piMode && !['native', 'simple'].includes(profile.piMode))
    throw new Error('piMode must be native or simple');
  if (profile.local && (profile.provider || profile.endpoint || profile.chatCompletionsUrl))
    throw new Error('local server options require a managed-local profile');
  if (profile.local) {
    for (const [key, minimum] of [['contextTokens', 1024], ['gpuLayers', 0], ['parallel', 1], ['cacheRamMiB', 0],
      ['memoryBudgetMiB', 1], ['kvBytesPerToken', 1]] as const) {
      const value = profile.local[key];
      if (value !== undefined && (!Number.isSafeInteger(value) || value < minimum))
        throw new Error(`local.${key} must be an integer >= ${minimum}`);
    }
    for (const key of ['modelPath', 'template', 'modelId', 'server'] as const)
      if (profile.local[key] !== undefined && (typeof profile.local[key] !== 'string' || !profile.local[key]))
        throw new Error(`local.${key} must be a nonempty string`);
    if ((profile.local.template || profile.local.modelId) && !profile.local.modelPath)
      throw new Error('local.template and local.modelId describe local.modelPath; set it too');
    if (profile.local.args && (!Array.isArray(profile.local.args) ||
      profile.local.args.some(value => typeof value !== 'string')))
      throw new Error('local.args must be an array of strings');
    const ownedFlags = new Set(['-m', '--model', '--model-url', '--host', '--port', '--chat-template-file',
      '-c', '--ctx-size', '-ngl', '--n-gpu-layers', '--parallel', '-np', '--cache-ram']);
    for (const arg of profile.local.args ?? []) if (ownedFlags.has(arg.split('=')[0]!))
      throw new Error(`local.args cannot override managed flag ${arg}; use the dedicated local setting where available`);
  }
  if (profile.concurrency !== undefined && (!Number.isSafeInteger(profile.concurrency) || profile.concurrency < 1))
    throw new Error('concurrency must be an integer >= 1');
  const batching = profile.batching;
  if (batching !== undefined) {
    if (!batching || typeof batching !== 'object' || Array.isArray(batching)) throw new TypeError('batching must be an object');
    if (batching.mode !== undefined && !['server-continuous', 'explicit-batch', 'serial'].includes(batching.mode))
      throw new Error('batching.mode must be server-continuous, explicit-batch or serial');
    if (batching.maxConcurrent !== undefined && (!Number.isSafeInteger(batching.maxConcurrent) || batching.maxConcurrent < 1))
      throw new Error('batching.maxConcurrent must be an integer >= 1');
    if (batching.coalesceMs !== undefined && !(Number.isFinite(batching.coalesceMs) && batching.coalesceMs >= 0))
      throw new Error('batching.coalesceMs must be a number >= 0');
    if (batching.scoreEndpoint && batching.mode !== 'explicit-batch')
      throw new Error('batching.scoreEndpoint requires batching.mode explicit-batch');
  }
  const common = { promptRendering: profile.promptRendering ?? 'full', headers: profile.headers, runtime: profile.runtime,
    ...(profile.concurrency === undefined ? {} : { concurrency: profile.concurrency }),
    ...(batching === undefined ? {} : { batching }) };
  if (profile.provider) return { ...common, kind: 'pi-provider', provider: profile.provider, model: profile.model!,
    apiKeyEnv: profile.apiKeyEnv, piOptions: profile.piOptions, piPayload: profile.piPayload,
    piMode: profile.piMode ?? 'native', modelOptions: profile.modelOptions };
  if (profile.native) return { ...common, kind: 'external', endpoint: nativeEndpoint(profile.native), model: profile.model ?? 'native',
    apiKeyEnv: profile.apiKeyEnv ?? 'NATLANG_API_KEY', request: profile.request, neuraleseServer: true, managedNative: profile.native };
  if (profile.endpoint) return { ...common, kind: 'external', endpoint: profile.endpoint,
    ...(profile.chatCompletionsUrl ? { chatCompletionsUrl: profile.chatCompletionsUrl } : {}),
    model: profile.model ?? DEFAULT_MODEL_RELEASE.id, apiKeyEnv: profile.apiKeyEnv ?? 'NATLANG_API_KEY',
    request: profile.request, ...(profile.neuraleseServer ? { neuraleseServer: true as const } : {}),
    ...(profile.vllm ? { vllm: true as const } : {}) };
  if (profile.model) throw new Error('a configured model ID needs an endpoint; omit both to use the managed local default');
  return { ...common, kind: 'managed-local', model: profile.local?.modelId ?? (profile.local?.modelPath ? 'local-gguf' : DEFAULT_MODEL_RELEASE.id),
    request: profile.request, local: profile.local };
}

export function loadModelConfiguration(name?: string, overrides: ModelSelectionOverrides = {},
  environment: NodeJS.ProcessEnv = process.env): LoadedModelConfiguration {
  const configPath = join(defaultNatlangConfigDirectory(environment), 'config.json');
  let config: { defaultProfile?: string; profiles?: Record<string, ModelProfile>; nativeSources?: Record<string, string> } = {};
  if (existsSync(configPath)) config = JSON.parse(readFileSync(configPath, 'utf8')) as typeof config;
  const selected = name ?? environment.NATLANG_PROFILE ?? config.defaultProfile ?? 'default';
  const configured = config.profiles?.[selected] ?? {};
  const provider = overrides.provider ?? environment.NATLANG_PROVIDER;
  const endpoint = environment.NATLANG_SERVER;
  const profile: ModelProfile = { ...configured,
    ...(provider ? { provider, endpoint: undefined, chatCompletionsUrl: undefined, local: undefined, request: undefined, native: undefined } : {}),
    ...(endpoint && !provider ? { endpoint, provider: undefined, piOptions: undefined, native: undefined,
      chatCompletionsUrl: undefined, piPayload: undefined, piMode: undefined, modelOptions: undefined, local: undefined } : {}),
    ...(overrides.model ?? environment.NATLANG_MODEL ? { model: overrides.model ?? environment.NATLANG_MODEL } : {}),
    ...(environment.NATLANG_CONCURRENCY ? { concurrency: Number(environment.NATLANG_CONCURRENCY) } : {}) };
  return { name: selected, configPath, profile, choice: resolveModelChoice(profile), nativeSources: config.nativeSources ?? {} };
}

/** Explicit compatibility identity without authentication material. Profiles are caller-declared metadata. */
export function executorIdentityForChoice(choice: ResolvedModelChoice): import('../adaptation/types.js').ExecutorIdentity {
  const credential = /(?:api.?key|authorization|password|credential|secret|access.?token|refresh.?token)/i;
  const sanitize = (input: unknown): unknown => {
    if (Array.isArray(input)) return input.map(sanitize);
    if (input && typeof input === 'object') return Object.fromEntries(Object.entries(input).filter(([key, value]) =>
      value !== undefined && key.toLowerCase() !== 'headers' && !credential.test(key)).map(([key, value]) => {
        if (typeof value === 'string' && /(?:endpoint|baseurl|url)$/i.test(key)) {
          try { const url = new URL(value); return [key, url.origin + url.pathname]; } catch { /* Non-URL configuration remains explicit metadata. */ }
        }
        return [key, sanitize(value)];
      }));
    return input;
  };
  const configuration = sanitize(choice) as Record<string, unknown>;
  delete configuration.concurrency; delete configuration.batching;
  // A managed native server is identified by the checkpoint it serves, not by how it is launched.
  if (choice.kind === 'external' && choice.managedNative) configuration.managedNative = { sha256: choice.managedNative.sha256 };   // how many requests run at once does not change what the model does
  const behaviorHeaders = Object.fromEntries(Object.entries(choice.headers ?? {}).filter(([key]) => !credential.test(key)));
  if (Object.keys(behaviorHeaders).length) configuration.headersHash = fingerprint(behaviorHeaders, 'natlang.model-headers/v1');
  if (choice.kind === 'external') {
    const url = new URL(choice.endpoint); configuration.endpoint = url.origin + url.pathname;
  }
  return { id: choice.kind + ':' + ('provider' in choice ? choice.provider + ':' : '') + choice.model, configuration };
}
