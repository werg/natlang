import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { defaultNatlangConfigDirectory } from '../package/store.js';
import { DEFAULT_MODEL_RELEASE } from '../model-default.js';
import { fingerprint } from '../adaptation/identity.js';
import type { ModelConfig, NatlangRuntimeOptions } from '../runtime/runtime.js';

export type ModelProfile = {
  endpoint?: string; provider?: string; model?: string; apiKeyEnv?: string;
  headers?: Record<string, string>; request?: Record<string, unknown>;
  piOptions?: Record<string, unknown>; piPayload?: Record<string, unknown>;
  piMode?: 'native' | 'simple'; modelOptions?: Record<string, unknown>;
  local?: { contextTokens?: number; gpuLayers?: number; parallel?: number; cacheRamMiB?: number; args?: string[] };
  runtime?: Omit<ModelConfig, 'driver'> & { seed?: NatlangRuntimeOptions['seed'] };
};

type Common = { headers?: Record<string, string>; runtime?: ModelProfile['runtime'] };
export type ResolvedModelChoice =
  | (Common & { kind: 'managed-local'; model: string; request?: Record<string, unknown>; local?: ModelProfile['local'] })
  | (Common & { kind: 'external'; endpoint: string; model: string; apiKeyEnv: string; request?: Record<string, unknown> })
  | (Common & { kind: 'pi-provider'; provider: string; model: string; apiKeyEnv?: string;
    piOptions?: Record<string, unknown>; piPayload?: Record<string, unknown>;
    piMode: 'native' | 'simple'; modelOptions?: Record<string, unknown> });

export type ModelSelectionOverrides = { provider?: string; model?: string };
export type LoadedModelConfiguration = { name: string; configPath: string; profile: ModelProfile; choice: ResolvedModelChoice };

export function resolveModelChoice(profile: ModelProfile): ResolvedModelChoice {
  if (profile.runtime && (typeof profile.runtime !== 'object' || Array.isArray(profile.runtime)))
    throw new TypeError('profile.runtime must be an object');
  if (profile.runtime?.review && 'driver' in profile.runtime.review)
    throw new Error('profile.runtime.review.driver is programmatic only');
  if (profile.provider && profile.endpoint) throw new Error('model profile cannot set both provider and endpoint');
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
  if (profile.local && (profile.provider || profile.endpoint))
    throw new Error('local server options require a managed-local profile');
  if (profile.local) {
    for (const [key, minimum] of [['contextTokens', 1024], ['gpuLayers', 0], ['parallel', 1], ['cacheRamMiB', 0]] as const) {
      const value = profile.local[key];
      if (value !== undefined && (!Number.isSafeInteger(value) || value < minimum))
        throw new Error(`local.${key} must be an integer >= ${minimum}`);
    }
    if (profile.local.args && (!Array.isArray(profile.local.args) ||
      profile.local.args.some(value => typeof value !== 'string')))
      throw new Error('local.args must be an array of strings');
    const ownedFlags = new Set(['-m', '--model', '--model-url', '--host', '--port', '--chat-template-file',
      '-c', '--ctx-size', '-ngl', '--n-gpu-layers', '--parallel', '--cache-ram']);
    for (const arg of profile.local.args ?? []) if (ownedFlags.has(arg.split('=')[0]!))
      throw new Error(`local.args cannot override managed flag ${arg}; use the dedicated local setting where available`);
  }
  const common = { headers: profile.headers, runtime: profile.runtime };
  if (profile.provider) return { ...common, kind: 'pi-provider', provider: profile.provider, model: profile.model!,
    apiKeyEnv: profile.apiKeyEnv, piOptions: profile.piOptions, piPayload: profile.piPayload,
    piMode: profile.piMode ?? 'native', modelOptions: profile.modelOptions };
  if (profile.endpoint) return { ...common, kind: 'external', endpoint: profile.endpoint,
    model: profile.model ?? DEFAULT_MODEL_RELEASE.id, apiKeyEnv: profile.apiKeyEnv ?? 'NATLANG_API_KEY',
    request: profile.request };
  if (profile.model) throw new Error('a configured model ID needs an endpoint; omit both to use the managed local default');
  return { ...common, kind: 'managed-local', model: DEFAULT_MODEL_RELEASE.id,
    request: profile.request, local: profile.local };
}

export function loadModelConfiguration(name?: string, overrides: ModelSelectionOverrides = {},
  environment: NodeJS.ProcessEnv = process.env): LoadedModelConfiguration {
  const configPath = join(defaultNatlangConfigDirectory(environment), 'config.json');
  let config: { defaultProfile?: string; profiles?: Record<string, ModelProfile> } = {};
  if (existsSync(configPath)) config = JSON.parse(readFileSync(configPath, 'utf8')) as typeof config;
  const selected = name ?? environment.NATLANG_PROFILE ?? config.defaultProfile ?? 'default';
  const configured = config.profiles?.[selected] ?? {};
  const provider = overrides.provider ?? environment.NATLANG_PROVIDER;
  const endpoint = environment.NATLANG_SERVER;
  const profile: ModelProfile = { ...configured,
    ...(provider ? { provider, endpoint: undefined, local: undefined, request: undefined } : {}),
    ...(endpoint && !provider ? { endpoint, provider: undefined, piOptions: undefined,
      piPayload: undefined, piMode: undefined, modelOptions: undefined, local: undefined } : {}),
    ...(overrides.model ?? environment.NATLANG_MODEL ? { model: overrides.model ?? environment.NATLANG_MODEL } : {}) };
  return { name: selected, configPath, profile, choice: resolveModelChoice(profile) };
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
  const behaviorHeaders = Object.fromEntries(Object.entries(choice.headers ?? {}).filter(([key]) => !credential.test(key)));
  if (Object.keys(behaviorHeaders).length) configuration.headersHash = fingerprint(behaviorHeaders, 'natlang.model-headers/v1');
  if (choice.kind === 'external') {
    const url = new URL(choice.endpoint); configuration.endpoint = url.origin + url.pathname;
  }
  return { id: choice.kind + ':' + ('provider' in choice ? choice.provider + ':' : '') + choice.model, configuration };
}
