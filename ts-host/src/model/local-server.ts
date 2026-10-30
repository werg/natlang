import { activeNativeAdapters } from '../native/model-recording.js';
import { perRequestLora } from './lora-scales.js';
import { stackServingName } from '../adaptation/weights/binding.js';
import { resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import { DEFAULT_LOCAL_MODEL, type LocalModelRelease } from '../model-catalog.js';
import { openAICompatibleModelTurn, serverContextWindow, type OpenAICompatibleOptions } from './openai-compatible.js';
import { neuraleseServerModelTurn } from './neuralese-server.js';
import { ensureNativeServer, stopNativeServer } from './native-managed.js';
import { schedulerForSettings } from './scheduler.js';
import type { Scheduler } from './scheduler.js';
import type { SlotPlan } from '../llama/slots.js';
import type { WarmPrefix } from '../package/warm.js';
import type { DecisionRequest, DecisionScores, ModelTurn, ModelTurnRequest, SessionTurnOptions } from '../contracts.js';
import type { LlamaRuntimeDiscovery, LlamaServerInspection } from '../llama/runtime.js';
import * as models from '../llama/models.js';
import { createLlamaServer, supportsNativeStates as supportsStates } from '../llama/server.js';
import { nodeLlamaHost } from '../llama/node.js';
import { resolveModelChoice, storedApiKey, type ModelProfile, type ResolvedModelChoice } from './config.js';

export { DEFAULT_LOCAL_MODEL, STOCK_MODELS, stockModel, type LocalModelRelease } from '../model-catalog.js';
export { runtimeLabel, warmTag } from '../llama/models.js';

/**
 * The Node binding of the managed llama.cpp machinery (llama/): the model session the CLI and `@natlang/node` use, and the
 * environment-based entry points over a Node host (`nodeLlamaHost`). Hosts with their own supervisor use `@natlang/node/llama`.
 */
export type { ModelProfile } from './config.js';
export type ManagedModelStatus = { source: 'external' | 'managed-local' | 'pi-provider'; endpoint: string | null;
  model: string; executable: string | null; modelPath: string | null; running: boolean;
  /** A managed server's stock model and llama.cpp build (undefined for other backends and configured GGUFs). */
  release?: LocalModelRelease; runtimeBuild?: number | null;
  /** The llama.cpp build and commit, git-describe style (`b12067ga6401bb`): fork builds all report build 1. */
  runtime?: string | null;
  /** The prompt openings this session warmed (prefilled and saved, or restored) on a managed server. */
  warm?: WarmPrefix[];
  /** The session uses a server another process keeps loaded (model/shared-server.ts). */
  shared?: boolean };
export type ManagedModelSession = { prepare(): Promise<ManagedModelStatus>;
  /** A native reference server's artifact store, for uploading tuned artifacts and states; undefined otherwise. */
  nativeArtifacts?(): import('./neuralese-server.js').HttpNeuraleseStore | undefined;
  /** Requests carry native content parts (native-state imports); wrappers must keep this marker. */
  readonly neuralese?: boolean;
  /** The backend is one of natlang's servers (the native server or the managed llama.cpp fork): it applies guided
   * generation (the `guidance` request field). */
  readonly guidance?: boolean;
  neuraleseRuntime?(): Promise<import('../native/neuralese.js').NeuraleseRuntimeOptions | undefined>;
  vllmStacks?(names: Iterable<string>): void;
  /** One turn; `options` carry the driver's turn options (`onDelta`) and, for a Pi provider backend, `onProgress`. */
  turn(request: ModelTurnRequest, signal?: AbortSignal, options?: SessionTurnOptions): Promise<ModelTurn>;
  /** Score finite replies (decision readout); fails with `decision-unsupported` where the backend cannot. */
  decide(request: DecisionRequest, signal?: AbortSignal): Promise<DecisionScores>;
  /** The model's context window as its server reports it (undefined when it does not say). */
  contextWindow(): Promise<number | undefined>;
  status(): ManagedModelStatus; close(): Promise<void>;
  /** The session's model scheduler (undefined for Pi provider backends): queue, batch records and occupancy. */
  scheduler: Scheduler | undefined;
  /** The slot sizing of a managed local server once it has started (null otherwise). */
  slotPlan(): SlotPlan | null };
export type ManagedModelRuntimeOptions = { ensureRuntime?:
  (discovery: LlamaRuntimeDiscovery) => Promise<LlamaServerInspection | null>;
  /** Tuned weight adapters (GGUF LoRA exports of the active release) to load at server start, keyed by artifact ID.
   * Each request then applies only its own call's stack through per-request `lora` scales. */
  loraAdapters?: () => readonly { readonly id: string; readonly path: string }[];
  /** Native state span files (tuned soft states) the managed server loads with `--native-state` (the llama.cpp fork). */
  nativeStates?: () => readonly { readonly id: string; readonly path: string }[];
  /** Native reference server: request exact sampling receipts (`x_natlang_rl_capture`) on every turn, for RL rollouts. */
  rlCapture?: boolean;
  /** With rlCapture: also record the sampler's top-k log-probs at each sampled position (`x_natlang_rl_top_k`,
   * score-centered RL). */
  rlTopK?: number };


/** Where a stock model is downloaded to (the checkout's models/ directory is used first when it holds the file). */
export const stockModelPath = (release: LocalModelRelease, environment: NodeJS.ProcessEnv = process.env): string =>
  models.stockModelPath(nodeLlamaHost(environment), release);

export const localModelPrerequisites = (environment: NodeJS.ProcessEnv = process.env, release: LocalModelRelease = DEFAULT_LOCAL_MODEL) =>
  models.localModelPrerequisites(nodeLlamaHost(environment), release);

/** The slot count a managed local server would start with, and why (`natlang doctor`). */
export const localSlotPlan = (local: ModelProfile['local'], environment: NodeJS.ProcessEnv = process.env,
  modelPath?: string | null, release: LocalModelRelease = DEFAULT_LOCAL_MODEL): SlotPlan =>
  models.localSlotPlan(nodeLlamaHost(environment), local, modelPath, release);

export const managedModelSha256 = (environment: NodeJS.ProcessEnv = process.env, release: LocalModelRelease = DEFAULT_LOCAL_MODEL): Promise<string> =>
  models.managedModelSha256(nodeLlamaHost(environment), release);

/** The download of `release` another live process is writing now (its `.partial-<pid>` file), if any. */
export const stockModelDownload = (release: LocalModelRelease, environment: NodeJS.ProcessEnv = process.env) =>
  models.stockModelDownload(nodeLlamaHost(environment), release);

/** The stock model's verified local file, downloaded first when absent (with progress on `error`). */
export const ensureStockModel = (release: LocalModelRelease, environment: NodeJS.ProcessEnv = process.env,
  error: NodeJS.WritableStream = process.stderr, onProgress?: (received: number, total: number) => void): Promise<string> =>
  models.ensureStockModel(nodeLlamaHost(environment, error), release, onProgress);

/** Whether a llama-server binary imports native state spans exactly (`--native-state`, the llama.cpp fork). */
export const supportsNativeStates = (executable: string, environment: NodeJS.ProcessEnv = process.env): boolean =>
  supportsStates(nodeLlamaHost(environment), executable);

export function createManagedModelSession(profile: ModelProfile,
  environment: NodeJS.ProcessEnv = process.env, error: NodeJS.WritableStream = process.stderr,
  runtimeOptions: ManagedModelRuntimeOptions = {}): ManagedModelSession {
  return createResolvedModelSession(resolveModelChoice(profile), environment, error, runtimeOptions);
}

/** The environment a managed local profile runs in: its own GGUF, template and server stand where NATLANG_MODEL_PATH,
 * NATLANG_TEMPLATE and NATLANG_LLAMA_SERVER would; those still override. */
export function profileEnvironment(choice: ResolvedModelChoice, environment: NodeJS.ProcessEnv = process.env): NodeJS.ProcessEnv {
  const profileModel = choice.kind === 'managed-local' ? choice.local : undefined;
  if (profileModel?.modelPath && !environment.NATLANG_MODEL_PATH) environment = { ...environment, NATLANG_MODEL_PATH: profileModel.modelPath,
    ...(profileModel.template && !environment.NATLANG_TEMPLATE ? { NATLANG_TEMPLATE: profileModel.template } : {}) };
  if (profileModel?.server && !environment.NATLANG_LLAMA_SERVER) environment = { ...environment, NATLANG_LLAMA_SERVER: profileModel.server };
  return environment;
}

export function createResolvedModelSession(choice: ResolvedModelChoice,
  environment: NodeJS.ProcessEnv = process.env, error: NodeJS.WritableStream = process.stderr,
  runtimeOptions: ManagedModelRuntimeOptions = {}): ManagedModelSession {
  environment = profileEnvironment(choice, environment);
  const external = choice.kind === 'external' ? choice : null;
  const managed = choice.kind === 'managed-local' ? choice : null;
  type PiBackend = ReturnType<(typeof import('./pi-provider.js'))['createPiModelBackend']>;
  let pi: Promise<PiBackend> | null = null;
  const piBackend = () => {
    if (choice.kind !== 'pi-provider') throw new Error('Pi backend requires a Pi provider choice');
    return pi ??= import('./pi-provider.js').then(module =>
      module.createPiModelBackend(choice.provider, choice.model, environment, choice.apiKeyEnv,
        choice.headers, choice.piOptions, choice.modelOptions, choice.piMode, choice.piPayload));
  };
  let closed = false;
  const release = managed?.release ?? DEFAULT_LOCAL_MODEL;
  // The managed llama.cpp server (llama/server.ts), over the Node host.
  const llama = managed ? createLlamaServer(nodeLlamaHost(environment, error), { release, local: managed.local,
    ...(runtimeOptions.ensureRuntime ? { ensureRuntime: runtimeOptions.ensureRuntime } : {}),
    ...(runtimeOptions.loraAdapters ? { loraAdapters: runtimeOptions.loraAdapters } : {}),
    ...(runtimeOptions.nativeStates ? { nativeStates: runtimeOptions.nativeStates } : {}) }) : null;
  // One scheduler over every request of the session; a server found unable to score is not asked again. A managed
  // server's concurrency is its slot count unless configured, and is set once the server has started.
  const scheduler = choice.kind === 'pi-provider' ? undefined : schedulerForSettings(choice);
  const tunedRequest = () => perRequestLora(llama!.loraIndex, 'the managed model server');
  // A native reference server: one driver whose requests carry the exact native adapter stack and state parts.
  let native: ReturnType<typeof neuraleseServerModelTurn> | undefined;
  const nativeDriver = () => native ??= neuraleseServerModelTurn({ endpoint: external!.endpoint, model: external!.model,
    headers: external!.headers, apiKey: storedApiKey(external!.apiKeyEnv, environment), owner: randomUUID(),
    request: runtimeOptions.rlCapture ? { ...external!.request, x_natlang_rl_capture: true,
      ...(runtimeOptions.rlTopK !== undefined ? { x_natlang_rl_top_k: runtimeOptions.rlTopK } : {}) } : external!.request });
  // vLLM: a call's whole stack is one loaded LoRA, selected by name; the base model name serves untuned calls.
  const vllmLoaded = new Set<string>();
  const vllmTunedRequest = () => {
    const active = activeNativeAdapters();
    if (!active.length) return undefined;
    const name = stackServingName(active);
    if (!vllmLoaded.has(name)) throw new Error(`weights-unavailable: stack ${name} is not loaded by the vLLM server; rerun natlang tune or restart`);
    return { x_natlang_model_override: name };
  };
  // The managed llama.cpp fork scores decisions at /v1/natlang/score (single readouts and per-call LoRA scales too);
  // an upstream server answers 404 there and decisions fall back to prompt_logprobs, which it does not serve.
  const openai = (options: OpenAICompatibleOptions) => openAICompatibleModelTurn({ ...options, concurrency: scheduler,
    scoreEndpoint: choice.kind === 'pi-provider' ? undefined : choice.batching?.scoreEndpoint ?? (choice.kind === 'managed-local' || undefined),
    ...(choice.kind === 'managed-local' ? { scoreAll: true } : {}),
    ...(choice.kind === 'managed-local' ? { tunedRequest, nativeStates: () => llama!.nativeStatesLoaded } : external?.vllm ? { tunedRequest: vllmTunedRequest } : {}) });
  // One driver per server: its scheduler, request limit and decision coalescer are what turns and readouts that
  // arrive together share, so they are not rebuilt per request.
  const drivers = new Map<string, ReturnType<typeof openai> | ReturnType<typeof nativeDriver>>();
  const driver = (options: OpenAICompatibleOptions) => {
    const key = `${options.endpoint}\n${options.model}`;
    let built = drivers.get(key);
    if (!built) drivers.set(key, built = external?.neuraleseServer ? Object.assign(nativeDriver(),
      { contextWindow: () => serverContextWindow(options) }) : openai(options));
    return built as ReturnType<typeof openai>;
  };
  let unscored: string | null = null, sized = false;

  const start = async (): Promise<OpenAICompatibleOptions> => {
    if (closed) throw new Error('model session is closed');
    if (external?.managedNative) await ensureNativeServer(external.managedNative, environment, error);
    if (external) return { endpoint: external.endpoint, chatCompletionsUrl: external.chatCompletionsUrl,
      model: external.model, headers: external.headers,
      request: external.request, apiKey: storedApiKey(external.apiKeyEnv, environment) };
    const endpoint = await llama!.start();
    if (!sized) {
      sized = true;
      if (scheduler && managed!.batching?.maxConcurrent === undefined && managed!.concurrency === undefined) scheduler.setMaxConcurrent(llama!.plan()!.slots);
    }
    // Prefix caching: a slot keeps the previous prompt's KV, so calls that share a prefix skip its prefill.
    return { endpoint, model: environment.NATLANG_MODEL ?? choice.model, headers: choice.headers,
      request: { cache_prompt: true, ...managed!.request } };
  };

  let window: Promise<number | undefined> | undefined;
  return {
    async prepare() { if (choice.kind === 'pi-provider') await (await piBackend()).prepare(); else await start(); return this.status(); },
    async turn(request, signal, turnOptions) {
      signal?.throwIfAborted();
      if (choice.kind === 'pi-provider') return (await piBackend()).turn(request, signal, turnOptions);
      const options = await start(); signal?.throwIfAborted();
      const admitted = llama ? await llama.admit(request, signal) : undefined;
      // The HTTP driver streams deltas; aggregate progress is a Pi backend's report.
      try { return await (turnOptions?.onDelta ? driver(options)(request, signal, { onDelta: turnOptions.onDelta }) : driver(options)(request, signal)); }
      finally { admitted?.release(); }
    },
    async decide(request, signal) {
      signal?.throwIfAborted();
      if (choice.kind === 'pi-provider') throw new Error('decision-unsupported: Pi provider backends cannot score replies');
      if (unscored) throw new Error(unscored);
      const options = await start();
      try { return await driver(options).decide(request, signal); }
      catch (error) { if (String((error as Error)?.message).startsWith('decision-unsupported')) unscored = (error as Error).message; throw error; }
    },
    contextWindow() {
      if (choice.kind === 'pi-provider') return Promise.resolve(undefined);
      return window ??= start().then(options => driver(options).contextWindow(), () => undefined);
    },
    status() {
      if (choice.kind === 'pi-provider') return { source: 'pi-provider', endpoint: null, model: `${choice.provider}/${choice.model}`,
        executable: null, modelPath: null, running: false };
      if (external) return { source: 'external', endpoint: external.endpoint, model: external.model,
        executable: null, modelPath: null, running: false };
      const server = llama!.status();
      return { source: 'managed-local', endpoint: server.endpoint, model: environment.NATLANG_MODEL ?? choice.model, executable: server.executable,
        modelPath: environment.NATLANG_MODEL_PATH ? resolve(environment.NATLANG_MODEL_PATH) : null, running: server.running,
        ...(environment.NATLANG_MODEL_PATH ? {} : { release }), runtimeBuild: server.runtimeBuild, runtime: server.runtime, warm: server.warm,
        ...(server.shared ? { shared: true } : {}) };
    },
    scheduler,
    /** The native reference server's artifact store (uploads of tuned artifacts and states), when this is one. */
    nativeArtifacts() { return external?.neuraleseServer ? nativeDriver().artifacts : undefined; },
    neuralese: !!external?.neuraleseServer,
    guidance: !!external?.neuraleseServer || choice.kind === 'managed-local',
    async neuraleseRuntime() {
      if (!external?.neuraleseServer) return undefined;
      await start(); return nativeDriver().neuraleseRuntime();
    },
    /** Record stacks the vLLM server has loaded (by serving name). */
    vllmStacks(names: Iterable<string>) { for (const name of names) vllmLoaded.add(name); },
    slotPlan() { return llama?.plan() ?? null; },
    async close() {
      closed = true;
      scheduler?.close();
      if (pi) await pi.then(backend => backend.close(), () => undefined);
      await llama?.close();
      if (external?.managedNative) await stopNativeServer(external.endpoint);
    },
  };
}
