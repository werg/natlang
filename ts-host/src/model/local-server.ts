import { activeNativeAdapters } from '../native/model-recording.js';
import { perRequestLora } from './lora-scales.js';
import { stackServingName } from '../adaptation/weights/binding.js';
import { createReadStream, createWriteStream, existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:net';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { defaultNatlangCacheDirectory } from '../package/store.js';
import { DEFAULT_LOCAL_MODEL, type LocalModelRelease } from '../model-catalog.js';
import { openAICompatibleModelTurn, serverContextWindow, type OpenAICompatibleOptions } from './openai-compatible.js';
import { neuraleseServerModelTurn } from './neuralese-server.js';
import { ensureNativeServer, stopNativeServer } from './native-managed.js';
import { randomUUID } from 'node:crypto';
import { schedulerForSettings, type Scheduler } from './scheduler.js';
import { planServerSlots, type SlotPlan } from './server-slots.js';
import { openingOf, pruneWarmStates, warmOpening, type WarmPrefix } from '../package/warm.js';
import { findSharedServer } from './shared-server.js';
import type { DecisionRequest, DecisionScores, ModelTurn, ModelTurnRequest, SessionTurnOptions } from '../contracts.js';
import { describeLlamaRuntime, discoverLlamaRuntime, type LlamaRuntimeDiscovery,
  type LlamaServerInspection } from './llama-runtime.js';
import { resolveModelChoice, storedApiKey, type ModelProfile, type ResolvedModelChoice } from './config.js';

export { DEFAULT_LOCAL_MODEL, STOCK_MODELS, stockModel, type LocalModelRelease } from '../model-catalog.js';

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
export function stockModelPath(release: LocalModelRelease, environment: NodeJS.ProcessEnv = process.env): string {
  return join(defaultNatlangCacheDirectory(environment), 'models', `${release.sha256.slice(0, 12)}-${release.file}`);
}

export function localModelPrerequisites(environment: NodeJS.ProcessEnv = process.env, release: LocalModelRelease = DEFAULT_LOCAL_MODEL) {
  const runtime = discoverLlamaRuntime(environment), found = runtime.selected?.path ?? null;
  const explicitModel = environment.NATLANG_MODEL_PATH ? resolve(environment.NATLANG_MODEL_PATH) : null;
  const checkoutModel = resolve('models', release.file);
  const modelPath = (explicitModel ? [explicitModel] : [checkoutModel, stockModelPath(release, environment)])
    .find((value): value is string => Boolean(value && existsSync(value))) ?? null;
  const template = templateCandidates(environment, release).find(existsSync) ?? null;
  const modelAvailable = Boolean(modelPath || (!explicitModel && release.downloadUrl));
  // A configured GGUF without a template uses the one it carries.
  const templateAvailable = Boolean(template || (explicitModel && !environment.NATLANG_TEMPLATE));
  return { executable: found, available: Boolean(found && modelAvailable && templateAvailable),
    command: environment.NATLANG_LLAMA_SERVER ?? 'llama-server', runtime,
    model: explicitModel ? basename(explicitModel) : release.id, modelPath, template,
    downloadable: !explicitModel && Boolean(release.downloadUrl), downloadBytes: explicitModel ? null : release.bytes };
}

/** The slot count a managed local server would start with, and why (`natlang doctor`). */
export function localSlotPlan(local: ModelProfile['local'], environment: NodeJS.ProcessEnv = process.env,
  modelPath: string | null = localModelPrerequisites(environment).modelPath, release: LocalModelRelease = DEFAULT_LOCAL_MODEL): SlotPlan {
  const modelBytes = modelPath && existsSync(modelPath) ? statSync(modelPath).size : release.bytes;
  // A configured GGUF is another model: only the stock file's measured cache size applies to it.
  const stock = !environment.NATLANG_MODEL_PATH;
  return planServerSlots({ local, modelBytes, defaultContextTokens: release.contextTokens, environment,
    ...(stock ? { defaultKvBytesPerToken: release.kvBytesPerToken } : {}) });
}

async function fileHash(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

/**
 * The sha256 of the base GGUF the managed server runs: the pinned default's declared hash, or the configured file's
 * content hash (cached beside it by size and mtime). Tuned GGUF adapters must bind exactly this file.
 */
export async function managedModelSha256(environment: NodeJS.ProcessEnv = process.env, release: LocalModelRelease = DEFAULT_LOCAL_MODEL): Promise<string> {
  if (!environment.NATLANG_MODEL_PATH) return release.sha256;
  const path = resolve(environment.NATLANG_MODEL_PATH), stat = statSync(path);
  const cache = join(defaultNatlangCacheDirectory(environment), 'model-hashes', createHash('sha256').update(path).digest('hex').slice(0, 24));
  const stamp = `${stat.size}:${stat.mtimeMs}`;
  try { const [seen, hash] = readFileSync(cache, 'utf8').split('\n'); if (seen === stamp && hash && /^[0-9a-f]{64}$/.test(hash)) return hash; } catch { /* not cached */ }
  const hash = await fileHash(path);
  mkdirSync(dirname(cache), { recursive: true }); writeFileSync(cache, `${stamp}\n${hash}`);
  return hash;
}

const serverHelp = new Map<string, string>();
/** Whether a llama-server binary has an option (its `--help`, read once per executable). */
function serverSupports(executable: string, option: string): boolean {
  let help = serverHelp.get(executable);
  if (help === undefined) {
    const result = spawnSync(executable, ['--help'], { encoding: 'utf8', timeout: 20000 });
    serverHelp.set(executable, help = `${result.stdout ?? ''}${result.stderr ?? ''}`);
  }
  return help.includes(option);
}
/** Whether a llama-server binary batches different per-request LoRA stacks (`--lora-per-sequence`). */
const supportsPerSequenceLora = (executable: string) => serverSupports(executable, '--lora-per-sequence');
/** Whether a llama-server binary imports native state spans exactly (`--native-state`, the llama.cpp fork). */
export const supportsNativeStates = (executable: string) => serverSupports(executable, '--native-state');

/** A verified file is remembered by size and mtime beside it, so a 2 GB model is hashed once, not at every start. */
async function verifiedRelease(path: string, release: LocalModelRelease): Promise<boolean> {
  if (!existsSync(path)) return false;
  const stat = statSync(path), stamp = `${path}.verified`, expected = `${stat.size}:${stat.mtimeMs}:${release.sha256}`;
  if (stat.size !== release.bytes) return false;
  try { if (readFileSync(stamp, 'utf8') === expected) return true; } catch { /* not verified yet */ }
  if (await fileHash(path) !== release.sha256) return false;
  try { writeFileSync(stamp, expected); } catch { /* a read-only location is verified again next time */ }
  return true;
}

/** The download of `release` another live process is writing now (its `.partial-<pid>` file), if any. */
export function stockModelDownload(release: LocalModelRelease, environment: NodeJS.ProcessEnv = process.env):
  { received: number; total: number; pid: number } | null {
  const destination = stockModelPath(release, environment), prefix = `${basename(destination)}.partial-`;
  let names: string[];
  try { names = readdirSync(dirname(destination)); } catch { return null; }
  for (const name of names) {
    const pid = name.startsWith(prefix) ? Number(name.slice(prefix.length)) : NaN;
    if (!Number.isInteger(pid) || pid <= 0 || pid === process.pid) continue;
    try { process.kill(pid, 0); } catch (error) { if ((error as NodeJS.ErrnoException).code !== 'EPERM') continue; }
    try { return { received: statSync(join(dirname(destination), name)).size, total: release.bytes, pid }; } catch { continue; }
  }
  return null;
}

/** The stock model's verified local file, downloaded first when absent (with progress on `error`). A download another
 * process is already making is awaited instead of repeated. */
export async function ensureStockModel(release: LocalModelRelease, environment: NodeJS.ProcessEnv = process.env,
  error: NodeJS.WritableStream = process.stderr, onProgress?: (received: number, total: number) => void): Promise<string> {
  const checkout = resolve('models', release.file);
  if (await verifiedRelease(checkout, release)) return checkout;
  const destination = stockModelPath(release, environment);
  if (await verifiedRelease(destination, release)) return destination;
  for (let other = stockModelDownload(release, environment), announced = false; other; other = stockModelDownload(release, environment)) {
    if (!announced) error.write(`natlang: waiting for the ${release.label} download already running in process ${other.pid}\n`);
    announced = true; onProgress?.(other.received, other.total);
    await new Promise(wake => setTimeout(wake, 1000));
  }
  if (await verifiedRelease(destination, release)) return destination;
  mkdirSync(dirname(destination), { recursive: true });
  const temporary = `${destination}.partial-${process.pid}`;
  error.write(`natlang: downloading ${release.label} from ${release.source.repository} (${Math.ceil(release.bytes / 1_000_000)} MB, once)\n`);
  try {
    const response = await fetch(release.downloadUrl, { redirect: 'follow' });
    if (!response.ok || !response.body) throw new Error(`model download failed: HTTP ${response.status}`);
    let received = 0, reported = 0;
    const counted = async function* (body: AsyncIterable<Uint8Array>) {
      for await (const chunk of body) {
        received += chunk.length; onProgress?.(received, release.bytes);
        // A terminal gets one updating line; logs (the dashboard's jobs, pipes) a line every 5%.
        const terminal = 'isTTY' in error && error.isTTY === true;
        if (!onProgress && received - reported > (terminal ? 16_000_000 : release.bytes / 20)) {
          reported = received;
          const line = `natlang: downloaded ${Math.round(received / 1_000_000)} of ${Math.ceil(release.bytes / 1_000_000)} MB (${Math.floor(100 * received / release.bytes)}%)`;
          error.write(terminal ? `\r\x1b[2K${line}` : `${line}\n`);
        }
        yield chunk;
      }
    };
    await pipeline(Readable.from(counted(response.body as AsyncIterable<Uint8Array>)), createWriteStream(temporary, { flags: 'wx' }));
    if (reported && 'isTTY' in error && error.isTTY === true) error.write('\r\x1b[2K');
    if (!await verifiedRelease(temporary, release)) throw new Error(`downloaded ${release.file} failed its size or SHA-256 check`);
    try { renameSync(temporary, destination); renameSync(`${temporary}.verified`, `${destination}.verified`); }
    catch (failure) { if (!await verifiedRelease(destination, release)) throw failure; }
    return destination;
  } finally { rmSync(temporary, { force: true }); rmSync(`${temporary}.verified`, { force: true }); }
}

async function ensureModel(environment: NodeJS.ProcessEnv, error: NodeJS.WritableStream, release: LocalModelRelease): Promise<string> {
  if (environment.NATLANG_MODEL_PATH) {
    const path = resolve(environment.NATLANG_MODEL_PATH);
    if (!existsSync(path)) throw new Error(`NATLANG_MODEL_PATH does not exist: ${path}`);
    return path;
  }
  return ensureStockModel(release, environment, error);
}

function templateCandidates(environment: NodeJS.ProcessEnv, release: LocalModelRelease): string[] {
  if (environment.NATLANG_TEMPLATE) return [resolve(environment.NATLANG_TEMPLATE)];
  if (environment.NATLANG_MODEL_PATH) return [];
  return [resolve('models', 'templates', release.template),
    fileURLToPath(new URL(`../../../models/templates/${release.template}`, import.meta.url)),
    fileURLToPath(new URL(`../../model-assets/${release.template}`, import.meta.url))];
}

/** The chat template file to pass, or null for a configured GGUF without one (it is served with its own). */
function templatePath(environment: NodeJS.ProcessEnv, release: LocalModelRelease): string | null {
  const candidates = templateCandidates(environment, release);
  if (!candidates.length) return null;
  const selected = candidates.find(existsSync);
  if (!selected) throw new Error(`missing the chat template ${candidates[0]} for ${environment.NATLANG_TEMPLATE ? 'NATLANG_TEMPLATE' : release.id}`);
  return selected;
}

async function freePort(): Promise<number> {
  return await new Promise((resolvePort, reject) => {
    const server = createServer();
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => {
      const address = server.address();
      if (!address || typeof address === 'string') { server.close(); reject(new Error('could not allocate a local model port')); return; }
      server.close(error => error ? reject(error) : resolvePort(address.port));
    });
  });
}

async function stopChild(child: ChildProcess): Promise<void> {
  if (child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise<void>(resolveExit => child.once('exit', () => resolveExit()));
  child.kill('SIGTERM');
  let timer: NodeJS.Timeout | undefined;
  await Promise.race([exited, new Promise<void>(resolveTimeout => { timer = setTimeout(resolveTimeout, 5000); })]);
  if (timer) clearTimeout(timer);
  if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
}

/**
 * The key part of a warm state for this model file, llama.cpp build and slot layout: a saved slot state restores only
 * into a server with the same number of slots and context per slot.
 */
/** A llama.cpp binary's identity as `b<build>g<commit>`: the build number alone does not tell fork commits apart. */
export const runtimeLabel = (runtime: { build: number | null; commit: string | null } | null | undefined) =>
  runtime ? `b${runtime.build ?? 'x'}${runtime.commit ? `g${runtime.commit.slice(0, 7)}` : ''}` : 'bx';
export const warmTag = (modelSha256: string, runtime: string, plan: Pick<SlotPlan, 'slots' | 'contextPerSlot'>) =>
  `${modelSha256.slice(0, 12)}-${runtime}-p${plan.slots}x${plan.contextPerSlot}`;

/** Whether no slot of a llama.cpp server is processing (its /slots listing). */
async function serverIdle(endpoint: string): Promise<boolean> {
  try {
    const response = await fetch(endpoint + '/slots', { signal: AbortSignal.timeout(2000) });
    const slots = await response.json() as { is_processing?: boolean }[];
    return response.ok && Array.isArray(slots) && slots.every(slot => !slot.is_processing);
  } catch { return false; }
}

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
  const localSettings = choice.kind === 'managed-local' ? choice.local : undefined;
  type PiBackend = ReturnType<(typeof import('./pi-provider.js'))['createPiModelBackend']>;
  let pi: Promise<PiBackend> | null = null;
  const piBackend = () => {
    if (choice.kind !== 'pi-provider') throw new Error('Pi backend requires a Pi provider choice');
    return pi ??= import('./pi-provider.js').then(module =>
      module.createPiModelBackend(choice.provider, choice.model, environment, choice.apiKeyEnv,
        choice.headers, choice.piOptions, choice.modelOptions, choice.piMode, choice.piPayload));
  };
  let child: ChildProcess | null = null, local: OpenAICompatibleOptions | null = null;
  let starting: Promise<OpenAICompatibleOptions> | null = null, recentError = '', closed = false;
  const release = (choice.kind === 'managed-local' ? choice.release : undefined) ?? DEFAULT_LOCAL_MODEL;
  let prerequisites = localModelPrerequisites(environment, release);
  // One scheduler over every request of the session; a server found unable to score is not asked again. A managed
  // server's concurrency is its slot count unless configured, and is set once the server has started.
  const scheduler = choice.kind === 'pi-provider' ? undefined : schedulerForSettings(choice);
  let plan: SlotPlan | null = null;
  // Loaded tuned adapters: artifact ID -> llama-server adapter index (fixed for the server's lifetime).
  const loraIndex = new Map<string, number>();
  const tunedRequest = () => perRequestLora(loraIndex, 'the managed model server');
  // Native state spans loaded by the managed server (--native-state): soft-state sentinels become native_state parts.
  const nativeStatesLoaded = new Set<string>();
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
    ...(choice.kind === 'managed-local' ? { tunedRequest, nativeStates: () => nativeStatesLoaded } : external?.vllm ? { tunedRequest: vllmTunedRequest } : {}) });
  const driver = (options: OpenAICompatibleOptions) => external?.neuraleseServer ? Object.assign(nativeDriver(),
    { contextWindow: () => serverContextWindow(options) }) : openai(options);
  let unscored: string | null = null;
  // Warm openings: each distinct system prompt and tool list is prefilled once per machine and model, then restored.
  const warmDirectory = join(defaultNatlangCacheDirectory(environment), 'warm-prefixes');
  let warming = false, inflight = 0, sharedEndpoint: string | null = null;
  const openings = new Map<string, Promise<WarmPrefix | undefined>>(), warmed: WarmPrefix[] = [];
  const warmFor = async (request: ModelTurnRequest): Promise<void> => {
    if (choice.kind !== 'managed-local') return;
    const opening = openingOf(request);
    if (!opening) return;
    let pending = openings.get(opening.key);
    if (!pending) {
      if (inflight > 0) return; // every slot must be idle to receive a state; a later idle moment warms it
      pending = (async () => {
        const options = await start();
        if (!warming) return undefined;
        // A shared server also serves other processes: warm it only while none of its slots is working.
        if (sharedEndpoint && !await serverIdle(sharedEndpoint)) { openings.delete(opening.key); return undefined; }
        const sha = await managedModelSha256(environment, release);
        const state = await warmOpening(options.endpoint, warmDirectory, opening, warmTag(sha, runtimeLabel(prerequisites.runtime.selected), plan!), plan?.slots ?? 1);
        if (state) {
          warmed.push(state);
          error.write(state.created ? `natlang: prefilled the ${state.tokens}-token prompt opening once for this model; later calls and runs restore it\n` :
            `natlang: restored the warm ${state.tokens}-token prompt opening into ${state.restoredSlots} slot${state.restoredSlots === 1 ? '' : 's'}\n`);
          pruneWarmStates(warmDirectory);
        }
        return state;
      })().catch(failure => { error.write(`natlang: prompt warming skipped: ${(failure as Error).message.split('\n')[0]!.slice(0, 200)}\n`); return undefined; });
      openings.set(opening.key, pending);
    }
    await pending;
  };

  const start = async (): Promise<OpenAICompatibleOptions> => {
    if (closed) throw new Error('model session is closed');
    if (external?.managedNative) await ensureNativeServer(external.managedNative, environment, error);
    if (external) return { endpoint: external.endpoint, chatCompletionsUrl: external.chatCompletionsUrl,
      model: external.model, headers: external.headers,
      request: external.request, apiKey: storedApiKey(external.apiKeyEnv, environment) };
    if (local) return local;
    if (starting) return starting;
    starting = (async () => {
      if (!prerequisites.executable && runtimeOptions.ensureRuntime) {
        const installed = await runtimeOptions.ensureRuntime(prerequisites.runtime);
        if (installed?.compatible) prerequisites = { ...localModelPrerequisites(environment, release), executable: installed.path };
      }
      if (!prerequisites.executable) {
        const explicit = prerequisites.runtime.candidates.find(candidate => candidate.source === 'explicit');
        if (explicit) throw new Error(`NATLANG_LLAMA_SERVER is incompatible: ${describeLlamaRuntime(prerequisites.runtime)}`);
        throw new Error(`managed model server is unavailable: ${describeLlamaRuntime(prerequisites.runtime)}; run natlang --setup or natlang --runtime install`);
      }
      // A server another natlang process keeps loaded for exactly this model file and build (model/shared-server.ts).
      const adapterless = !(runtimeOptions.loraAdapters?.() ?? []).length && !(runtimeOptions.nativeStates?.() ?? []).length;
      const shared = adapterless && choice.kind === 'managed-local' ?
        await findSharedServer(await managedModelSha256(environment, release), runtimeLabel(prerequisites.runtime.selected), environment) : undefined;
      if (shared) {
        plan = { budgetSource: 'configured-parallel', slots: shared.slots, contextPerSlot: shared.contextPerSlot,
          totalContext: shared.slots * shared.contextPerSlot, source: 'configured', budgetBytes: null, modelBytes: 0, kvBytesPerToken: 0,
          formula: `shared model server (pid ${shared.pid})` };
        if (scheduler && choice.batching?.maxConcurrent === undefined && choice.concurrency === undefined) scheduler.setMaxConcurrent(shared.slots);
        sharedEndpoint = shared.endpoint;
        warming = environment.NATLANG_WARM_PREFIXES !== 'off' && !!shared.warm;
        error.write(`natlang: using the ${shared.model} model kept loaded at ${shared.endpoint} (${shared.slots} slots)\n`);
        local = { endpoint: shared.endpoint, model: environment.NATLANG_MODEL ?? choice.model, headers: choice.headers,
          request: choice.kind === 'managed-local' ? { cache_prompt: true, ...choice.request } : undefined };
        return local;
      }
      const modelPath = await ensureModel(environment, error, release);
      if (closed) throw new Error('model session is closed');
      const port = await freePort();
      const endpoint = `http://127.0.0.1:${port}`;
      // `-c` is shared among the slots, so it is the per-slot context times the slot count.
      plan = localSlotPlan(localSettings, environment, modelPath, release);
      if (scheduler && choice.kind === 'managed-local' && choice.batching?.maxConcurrent === undefined && choice.concurrency === undefined)
        scheduler.setMaxConcurrent(plan.slots);
      const template = templatePath(environment, release);
      const args = ['-m', modelPath, '--host', '127.0.0.1', '--port', String(port), '--parallel', String(plan.slots),
        '-c', String(plan.totalContext), '-ngl',
        String(localSettings?.gpuLayers ?? 99), '--cache-ram', String(localSettings?.cacheRamMiB ?? 256), '--no-webui',
        '--jinja', ...(template ? ['--chat-template-file', template] : []), ...(localSettings?.args ?? [])];
      loraIndex.clear();
      const adapters = runtimeOptions.loraAdapters?.() ?? [];
      if (adapters.length) {
        // Loaded off; each request sets its own stack's scales. A server with per-sequence scales (the llama.cpp
        // fork) also batches requests whose stacks differ; others serialize them.
        args.push('--lora-init-without-apply', ...(supportsPerSequenceLora(prerequisites.executable) ? ['--lora-per-sequence'] : []));
        for (const adapter of adapters) { loraIndex.set(adapter.id, loraIndex.size); args.push('--lora', adapter.path); }
      }
      // Warm prompt openings (package/warm.ts) are saved and restored here; not with adapters, whose KV differs.
      warming = environment.NATLANG_WARM_PREFIXES !== 'off' && !adapters.length && !(runtimeOptions.nativeStates?.() ?? []).length;
      if (warming) { mkdirSync(warmDirectory, { recursive: true }); args.push('--slot-save-path', warmDirectory); }
      nativeStatesLoaded.clear();
      const states = runtimeOptions.nativeStates?.() ?? [];
      if (states.length) {
        if (!supportsNativeStates(prerequisites.executable))
          throw new Error(`weights-unavailable: ${prerequisites.executable} cannot import native state spans (tuned soft states need the llama.cpp fork's --native-state)`);
        for (const state of states) { nativeStatesLoaded.add(state.id); args.push('--native-state', state.path); }
      }
      error.write(`natlang: starting managed model ${basename(modelPath)} with ${plan.slots} slot${plan.slots === 1 ? '' : 's'} (${plan.contextPerSlot} tokens each; ${plan.budgetSource === 'configured-parallel' ? 'local.parallel' : `${plan.budgetSource === 'default' ? 'default' : plan.budgetSource} KV budget ${Math.round((plan.budgetBytes ?? 0) / 2 ** 20)} MiB`})\n`);
      child = spawn(prerequisites.executable, args, { stdio: ['ignore', 'ignore', 'pipe'] });
      child.stderr?.on('data', chunk => { recentError = (recentError + String(chunk)).slice(-16000); });
      child.once('error', failure => { recentError = `${recentError}\n${failure.message}`.slice(-16000); });
      const startedAt = Date.now(), deadline = startedAt + Number(environment.NATLANG_MODEL_START_TIMEOUT_MS ?? 120000);
      while (Date.now() < deadline) {
        if (child.exitCode !== null || child.signalCode !== null)
          throw new Error(`managed model server exited during startup\n${recentError.trim()}`);
        try { const response = await fetch(endpoint + '/health'); if (response.ok) {
          error.write(`natlang: model ready in ${((Date.now() - startedAt) / 1000).toFixed(1)} s\n`);
          local = { endpoint, model: environment.NATLANG_MODEL ?? choice.model,
            // Prefix caching: a slot keeps the previous prompt's KV, so calls that share a prefix skip its prefill.
            headers: choice.headers, request: choice.kind === 'managed-local' ? { cache_prompt: true, ...choice.request } : undefined }; return local;
        } } catch { /* server is still loading */ }
        await new Promise(resolveWait => setTimeout(resolveWait, 250));
      }
      await stopChild(child);
      throw new Error(`managed model server did not become ready within the startup timeout\n${recentError.trim()}`);
    })();
    try { return await starting; }
    catch (failure) { if (child) await stopChild(child); child = null; throw failure; }
    finally { starting = null; }
  };

  // Runs before the host's own handlers: the server stops with the process, and the signal is raised again (its default
  // action ends the process) only when nothing else handles it. A host with a handler (natlang dashboard) closes first.
  const terminate = (signal: NodeJS.Signals) => {
    // llama-server finishes the batch in progress on its first signal (a long prefill takes many seconds on a CPU) and
    // exits at once on a second; a different signal, since a pending signal of the same kind can merge with the first.
    child?.kill('SIGTERM'); child?.kill('SIGINT');
    process.removeListener(signal, terminate);
    if (!process.listenerCount(signal)) process.kill(process.pid, signal);
  };
  process.prependOnceListener('SIGINT', terminate); process.prependOnceListener('SIGTERM', terminate);
  const onExit = () => { if (child && child.exitCode === null && child.signalCode === null) child.kill('SIGTERM'); };
  process.once('exit', onExit);
  let window: Promise<number | undefined> | undefined;
  return {
    async prepare() { if (choice.kind === 'pi-provider') await (await piBackend()).prepare(); else await start(); return this.status(); },
    async turn(request, signal, turnOptions) {
      signal?.throwIfAborted();
      if (choice.kind === 'pi-provider') return (await piBackend()).turn(request, signal, turnOptions);
      const options = await start(); signal?.throwIfAborted();
      await warmFor(request); signal?.throwIfAborted();
      inflight++;
      // The HTTP driver streams deltas; aggregate progress is a Pi backend's report.
      try { return await (turnOptions?.onDelta ? driver(options)(request, signal, { onDelta: turnOptions.onDelta }) : driver(options)(request, signal)); }
      finally { inflight--; }
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
    status() { return choice.kind === 'pi-provider' ? { source: 'pi-provider', endpoint: null, model: `${choice.provider}/${choice.model}`,
      executable: null, modelPath: null, running: false } : external ? { source: 'external', endpoint: external.endpoint, model: external.model,
      executable: null, modelPath: null, running: false } : { source: 'managed-local', endpoint: local?.endpoint ?? null,
      model: environment.NATLANG_MODEL ?? choice.model, executable: prerequisites.executable,
      modelPath: environment.NATLANG_MODEL_PATH ? resolve(environment.NATLANG_MODEL_PATH) : null, running: Boolean(child || sharedEndpoint),
      ...(environment.NATLANG_MODEL_PATH ? {} : { release }), runtimeBuild: prerequisites.runtime.selected?.build ?? null,
      runtime: prerequisites.runtime.selected ? runtimeLabel(prerequisites.runtime.selected) : null, warm: [...warmed],
      ...(sharedEndpoint ? { shared: true } : {}) }; },
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
    slotPlan() { return plan; },
    async close() {
      closed = true;
      scheduler?.close();
      if (pi) await pi.then(backend => backend.close(), () => undefined);
      process.removeListener('SIGINT', terminate); process.removeListener('SIGTERM', terminate);
      process.removeListener('exit', onExit);
      if (child) await stopChild(child);
      if (starting) await starting.catch(() => undefined);
      if (child) await stopChild(child);
      if (external?.managedNative) await stopNativeServer(external.endpoint);
      child = null; local = null;
    },
  };
}
