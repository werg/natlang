import { createReadStream, createWriteStream, existsSync, mkdirSync, renameSync, rmSync, statSync } from 'node:fs';
import { spawn, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createServer } from 'node:net';
import { basename, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { Readable } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { defaultNatlangCacheDirectory } from '../package/store.js';
import { DEFAULT_MODEL_RELEASE } from '../model-default.js';
import { openAICompatibleModelTurn, type OpenAICompatibleOptions } from './openai-compatible.js';
import { schedulerForSettings, type Scheduler } from './scheduler.js';
import { planServerSlots, type SlotPlan } from './server-slots.js';
import type { DecisionRequest, DecisionScores, ModelStreamProgressSink, ModelTurn, ModelTurnRequest } from '../contracts.js';
import { describeLlamaRuntime, discoverLlamaRuntime, type LlamaRuntimeDiscovery,
  type LlamaServerInspection } from './llama-runtime.js';
import { resolveModelChoice, type ModelProfile, type ResolvedModelChoice } from './config.js';

export const DEFAULT_LOCAL_MODEL = DEFAULT_MODEL_RELEASE;

export type { ModelProfile } from './config.js';
export type ManagedModelStatus = { source: 'external' | 'managed-local' | 'pi-provider'; endpoint: string | null;
  model: string; executable: string | null; modelPath: string | null; running: boolean };
export type ManagedModelSession = { prepare(): Promise<ManagedModelStatus>;
  turn(request: ModelTurnRequest, signal?: AbortSignal, onProgress?: ModelStreamProgressSink): Promise<ModelTurn>;
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
  (discovery: LlamaRuntimeDiscovery) => Promise<LlamaServerInspection | null> };

export function localModelPrerequisites(environment: NodeJS.ProcessEnv = process.env) {
  const runtime = discoverLlamaRuntime(environment), found = runtime.selected?.path ?? null;
  const explicitModel = environment.NATLANG_MODEL_PATH ? resolve(environment.NATLANG_MODEL_PATH) : null;
  const checkoutModel = resolve('models', DEFAULT_LOCAL_MODEL.file);
  const cachedModel = join(defaultNatlangCacheDirectory(environment), 'models',
    `${DEFAULT_LOCAL_MODEL.sha256.slice(0, 12)}-${DEFAULT_LOCAL_MODEL.file}`);
  const modelPath = [explicitModel, checkoutModel, cachedModel].find((value): value is string => Boolean(value && existsSync(value))) ?? null;
  const template = templateCandidates(environment).find(existsSync) ?? null;
  const modelAvailable = Boolean(modelPath || DEFAULT_LOCAL_MODEL.downloadUrl);
  return { executable: found, available: Boolean(found && modelAvailable && template),
    command: environment.NATLANG_LLAMA_SERVER ?? 'llama-server', runtime,
    model: DEFAULT_LOCAL_MODEL.id, modelPath, template, downloadable: Boolean(DEFAULT_LOCAL_MODEL.downloadUrl) };
}

/** The slot count a managed local server would start with, and why (`natlang doctor`). */
export function localSlotPlan(local: ModelProfile['local'], environment: NodeJS.ProcessEnv = process.env,
  modelPath: string | null = localModelPrerequisites(environment).modelPath): SlotPlan {
  const modelBytes = modelPath && existsSync(modelPath) ? statSync(modelPath).size : DEFAULT_LOCAL_MODEL.bytes;
  return planServerSlots({ local, modelBytes, defaultContextTokens: DEFAULT_LOCAL_MODEL.contextTokens, environment });
}

async function fileHash(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

async function verifiedDefault(path: string): Promise<boolean> {
  if (!existsSync(path) || statSync(path).size !== DEFAULT_LOCAL_MODEL.bytes) return false;
  return await fileHash(path) === DEFAULT_LOCAL_MODEL.sha256;
}

async function ensureModel(environment: NodeJS.ProcessEnv, error: NodeJS.WritableStream): Promise<string> {
  if (environment.NATLANG_MODEL_PATH) {
    const path = resolve(environment.NATLANG_MODEL_PATH);
    if (!existsSync(path)) throw new Error(`NATLANG_MODEL_PATH does not exist: ${path}`);
    return path;
  }
  const checkout = resolve('models', DEFAULT_LOCAL_MODEL.file);
  if (await verifiedDefault(checkout)) return checkout;
  const directory = join(defaultNatlangCacheDirectory(environment), 'models');
  const destination = join(directory, `${DEFAULT_LOCAL_MODEL.sha256.slice(0, 12)}-${DEFAULT_LOCAL_MODEL.file}`);
  if (await verifiedDefault(destination)) return destination;
  mkdirSync(directory, { recursive: true });
  const temporary = `${destination}.partial-${process.pid}`;
  if (!DEFAULT_LOCAL_MODEL.downloadUrl) throw new Error(`default model ${DEFAULT_LOCAL_MODEL.id} is not installed locally and has no distribution URL`);
  error.write(`natlang: downloading default model ${DEFAULT_LOCAL_MODEL.id} (${Math.ceil(DEFAULT_LOCAL_MODEL.bytes / 1_000_000)} MB)\n`);
  try {
    const response = await fetch(DEFAULT_LOCAL_MODEL.downloadUrl, { redirect: 'follow' });
    if (!response.ok || !response.body) throw new Error(`model download failed: HTTP ${response.status}`);
    await pipeline(Readable.from(response.body as AsyncIterable<Uint8Array>),
      createWriteStream(temporary, { flags: 'wx' }));
    if (!await verifiedDefault(temporary)) throw new Error('downloaded default model failed its size or SHA-256 check');
    try { renameSync(temporary, destination); }
    catch (failure) { if (!await verifiedDefault(destination)) throw failure; }
    return destination;
  } finally { rmSync(temporary, { force: true }); }
}

function templateCandidates(environment: NodeJS.ProcessEnv): string[] {
  return [environment.NATLANG_TEMPLATE && resolve(environment.NATLANG_TEMPLATE),
    resolve('models', 'templates', DEFAULT_LOCAL_MODEL.template),
    fileURLToPath(new URL(`../../../models/templates/${DEFAULT_LOCAL_MODEL.template}`, import.meta.url)),
    fileURLToPath(new URL('../../model-assets/default.jinja', import.meta.url))]
    .filter((value): value is string => Boolean(value));
}

function templatePath(environment: NodeJS.ProcessEnv): string {
  const selected = templateCandidates(environment).find(existsSync);
  if (!selected) throw new Error(`missing the template for default model ${DEFAULT_LOCAL_MODEL.id}`);
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

export function createManagedModelSession(profile: ModelProfile,
  environment: NodeJS.ProcessEnv = process.env, error: NodeJS.WritableStream = process.stderr,
  runtimeOptions: ManagedModelRuntimeOptions = {}): ManagedModelSession {
  return createResolvedModelSession(resolveModelChoice(profile), environment, error, runtimeOptions);
}

export function createResolvedModelSession(choice: ResolvedModelChoice,
  environment: NodeJS.ProcessEnv = process.env, error: NodeJS.WritableStream = process.stderr,
  runtimeOptions: ManagedModelRuntimeOptions = {}): ManagedModelSession {
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
  let prerequisites = localModelPrerequisites(environment);
  // One scheduler over every request of the session; a server found unable to score is not asked again. A managed
  // server's concurrency is its slot count unless configured, and is set once the server has started.
  const scheduler = choice.kind === 'pi-provider' ? undefined : schedulerForSettings(choice);
  let plan: SlotPlan | null = null;
  const driver = (options: OpenAICompatibleOptions) => openAICompatibleModelTurn({ ...options, concurrency: scheduler,
    scoreEndpoint: choice.kind === 'pi-provider' ? undefined : choice.batching?.scoreEndpoint });
  let unscored: string | null = null;

  const start = async (): Promise<OpenAICompatibleOptions> => {
    if (closed) throw new Error('model session is closed');
    if (external) return { endpoint: external.endpoint, chatCompletionsUrl: external.chatCompletionsUrl,
      model: external.model, headers: external.headers,
      request: external.request, apiKey: environment[external.apiKeyEnv] };
    if (local) return local;
    if (starting) return starting;
    starting = (async () => {
      if (!prerequisites.executable && runtimeOptions.ensureRuntime) {
        const installed = await runtimeOptions.ensureRuntime(prerequisites.runtime);
        if (installed?.compatible) prerequisites = { ...localModelPrerequisites(environment), executable: installed.path };
      }
      if (!prerequisites.executable) {
        const explicit = prerequisites.runtime.candidates.find(candidate => candidate.source === 'explicit');
        if (explicit) throw new Error(`NATLANG_LLAMA_SERVER is incompatible: ${describeLlamaRuntime(prerequisites.runtime)}`);
        throw new Error(`managed model server is unavailable: ${describeLlamaRuntime(prerequisites.runtime)}; run natlang --setup or natlang --runtime install`);
      }
      const modelPath = await ensureModel(environment, error);
      if (closed) throw new Error('model session is closed');
      const port = await freePort();
      const endpoint = `http://127.0.0.1:${port}`;
      // `-c` is shared among the slots, so it is the per-slot context times the slot count.
      plan = localSlotPlan(localSettings, environment, modelPath);
      if (scheduler && choice.kind === 'managed-local' && choice.batching?.maxConcurrent === undefined && choice.concurrency === undefined)
        scheduler.setMaxConcurrent(plan.slots);
      const args = ['-m', modelPath, '--host', '127.0.0.1', '--port', String(port), '--parallel', String(plan.slots),
        '-c', String(plan.totalContext), '-ngl',
        String(localSettings?.gpuLayers ?? 99), '--cache-ram', String(localSettings?.cacheRamMiB ?? 256), '--no-webui',
        '--jinja', '--chat-template-file', templatePath(environment), ...(localSettings?.args ?? [])];
      error.write(`natlang: starting managed model ${basename(modelPath)} with ${plan.slots} slot${plan.slots === 1 ? '' : 's'} (${plan.contextPerSlot} tokens each; ${plan.budgetSource === 'configured-parallel' ? 'local.parallel' : `${plan.budgetSource === 'default' ? 'default' : plan.budgetSource} KV budget ${Math.round((plan.budgetBytes ?? 0) / 2 ** 20)} MiB`})\n`);
      child = spawn(prerequisites.executable, args, { stdio: ['ignore', 'ignore', 'pipe'] });
      child.stderr?.on('data', chunk => { recentError = (recentError + String(chunk)).slice(-16000); });
      child.once('error', failure => { recentError = `${recentError}\n${failure.message}`.slice(-16000); });
      const deadline = Date.now() + Number(environment.NATLANG_MODEL_START_TIMEOUT_MS ?? 120000);
      while (Date.now() < deadline) {
        if (child.exitCode !== null || child.signalCode !== null)
          throw new Error(`managed model server exited during startup\n${recentError.trim()}`);
        try { const response = await fetch(endpoint + '/health'); if (response.ok) {
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

  const terminate = (signal: NodeJS.Signals) => {
    child?.kill(signal);
    process.removeListener(signal, terminate);
    process.kill(process.pid, signal);
  };
  process.once('SIGINT', terminate); process.once('SIGTERM', terminate);
  const onExit = () => { if (child && child.exitCode === null && child.signalCode === null) child.kill('SIGTERM'); };
  process.once('exit', onExit);
  let window: Promise<number | undefined> | undefined;
  return {
    async prepare() { if (choice.kind === 'pi-provider') await (await piBackend()).prepare(); else await start(); return this.status(); },
    async turn(request, signal, onProgress) { signal?.throwIfAborted(); if (choice.kind === 'pi-provider') return (await piBackend()).turn(request, signal, onProgress); const options = await start(); signal?.throwIfAborted(); return driver(options)(request, signal); },
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
      modelPath: environment.NATLANG_MODEL_PATH ? resolve(environment.NATLANG_MODEL_PATH) : null, running: Boolean(child) }; },
    scheduler,
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
      child = null; local = null;
    },
  };
}
