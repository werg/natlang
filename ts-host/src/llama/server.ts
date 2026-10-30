/**
 * One managed llama.cpp server: find or install its runtime, resolve its model and chat template, size its slots, start
 * it (or use a server another natlang process keeps loaded, or one the host supervises itself), keep the warm prompt
 * openings (llama/warm.ts) saved and restored in its slots, and stop it. Mechanism only; what to ask the model, how
 * requests are queued and how replies are scored belong to the caller, which reads `start()`'s endpoint.
 *
 * Who runs the process is the host's choice (llama/host.ts):
 * - by default the package spawns llama-server through `host.processes` and waits for its health check (`spawnLauncher`);
 * - a host with its own supervisor passes `launch`, which receives the exact command, arguments and endpoint and returns
 *   once the server answers (the package still plans slots and warm states around it);
 * - a host that already runs a server passes `attach` and nothing is started.
 */
import { DEFAULT_LOCAL_MODEL, type LocalModelRelease } from '../model-catalog.js';
import { basename } from 'node:path';
import { mkdirSync } from 'node:fs';
import type { ModelTurnRequest } from '../contracts.js';
import type { WarmPrefix } from '../package/warm.js';
import type { LlamaHost, LlamaProcess } from './host.js';
import { describeLlamaRuntime, type LlamaRuntimeDiscovery, type LlamaServerInspection } from './runtime.js';
import { ensureModel, localModelPrerequisites, localSlotPlan, managedModelSha256, runtimeLabel, templatePath, warmTag } from './models.js';
import { findSharedServer } from './shared.js';
import type { LocalServerSettings, SlotPlan } from './slots.js';
import { openingOf, pruneWarmStates, warmOpening } from './warm.js';

/** A llama-server invocation the package has decided on. */
export type LlamaLaunch = { command: string; args: readonly string[]; port: number; endpoint: string };
/** A launched server: stop it. Resolves `launch` only once the server answers its health check. */
export type LlamaLaunched = { stop(): Promise<void> };
/** Start a llama-server. `signal` aborts when the server is closed while starting; honor it by stopping what was started. */
export type LlamaLauncher = (launch: LlamaLaunch, signal: AbortSignal) => Promise<LlamaLaunched>;

/** A server the host already runs, to plan around without starting it. */
export type LlamaAttachment = { endpoint: string; slots: number; contextPerSlot: number;
  /** The GGUF it serves and its llama-server build as `runtimeLabel` writes it: a warm state restores only into the same pair. */
  modelSha256: string; runtime: string | null;
  /** It was started with `--slot-save-path` set to `host.directories.warmPrefixes`, so warm openings can be saved and restored. */
  warm?: boolean };

export type LlamaServerOptions = {
  release?: LocalModelRelease;
  local?: LocalServerSettings;
  /** Tuned weight adapters (GGUF LoRA exports) to load at start, keyed by artifact ID; see `LlamaServer.loraIndex`. */
  loraAdapters?: () => readonly { readonly id: string; readonly path: string }[];
  /** Native state span files (tuned soft states) to load with `--native-state` (the llama.cpp fork). */
  nativeStates?: () => readonly { readonly id: string; readonly path: string }[];
  /** Called when no compatible runtime is found; may install one and return it (a host's consent decides). */
  ensureRuntime?: (discovery: LlamaRuntimeDiscovery) => Promise<LlamaServerInspection | null>;
  /** Sent with the package's own requests to the server (health, slots, warm states). */
  headers?: Record<string, string>;
  /** A server another natlang process keeps loaded for exactly this model file and build may be used (default true). */
  useSharedServer?: boolean;
  launch?: LlamaLauncher;
  attach?: LlamaAttachment;
};

export type LlamaServerStatus = { endpoint: string | null; executable: string | null; running: boolean;
  /** It is a server another process keeps loaded. */
  shared: boolean;
  /** The llama.cpp build number and its `runtimeLabel` (`b<build>g<commit>`). */
  runtimeBuild: number | null; runtime: string | null;
  /** The prompt openings this server warmed (prefilled and saved, or restored). */
  warm: WarmPrefix[] };

export type LlamaServer = {
  /** Start the server if it is not running (once, however many callers) and return its endpoint. Throws when it is closed. */
  start(): Promise<string>;
  /**
   * The gate of one generated turn: start the server, restore or prefill the request's warm opening in its slots, and
   * count the turn as in flight (a warm state needs every slot idle). Call the returned `release` when the turn ends.
   */
  admit(request: ModelTurnRequest, signal?: AbortSignal): Promise<{ endpoint: string; release(): void }>;
  /** The slot sizing once the server has started (null before). */
  plan(): SlotPlan | null;
  /** Adapter artifact ID -> llama-server adapter index, for the adapters loaded at the last start. */
  readonly loraIndex: ReadonlyMap<string, number>;
  /** The native state IDs loaded at the last start. */
  readonly nativeStatesLoaded: ReadonlySet<string>;
  status(): LlamaServerStatus;
  close(): Promise<void>;
};

const serverHelp = new Map<string, string>();
/** Whether a llama-server binary has an option (its `--help`, read once per executable). */
function serverSupports(host: LlamaHost, executable: string, option: string): boolean {
  let help = serverHelp.get(executable);
  if (help === undefined) {
    const result = host.processes.run(executable, ['--help'], { timeoutMs: 20000 });
    serverHelp.set(executable, help = `${result.stdout}${result.stderr}`);
  }
  return help.includes(option);
}
/** Whether a llama-server binary batches different per-request LoRA stacks (`--lora-per-sequence`). */
const supportsPerSequenceLora = (host: LlamaHost, executable: string) => serverSupports(host, executable, '--lora-per-sequence');
/** Whether a llama-server binary imports native state spans exactly (`--native-state`, the llama.cpp fork). */
export const supportsNativeStates = (host: LlamaHost, executable: string) => serverSupports(host, executable, '--native-state');

async function stopProcess(child: LlamaProcess): Promise<void> {
  if (child.hasExited()) return;
  child.kill('SIGTERM');
  let timer: NodeJS.Timeout | undefined;
  await Promise.race([child.exited, new Promise<void>(resolveTimeout => { timer = setTimeout(resolveTimeout, 5000); })]);
  if (timer) clearTimeout(timer);
  if (!child.hasExited()) child.kill('SIGKILL');
}

/** Whether no slot of a llama.cpp server is processing (its /slots listing). */
async function serverIdle(host: LlamaHost, endpoint: string, headers: Record<string, string> | undefined): Promise<boolean> {
  try {
    const response = await host.fetch(endpoint + '/slots', { signal: AbortSignal.timeout(2000), ...(headers ? { headers } : {}) });
    const slots = await response.json() as { is_processing?: boolean }[];
    return response.ok && Array.isArray(slots) && slots.every(slot => !slot.is_processing);
  } catch { return false; }
}

/**
 * The default launcher: spawn llama-server through the host and wait for `/health`. The server stops with the process
 * (SIGINT and SIGTERM stop it first, and the signal is raised again only when nothing else handles it, so a host with a
 * handler of its own, such as `natlang dashboard`, closes first), and a server that exits or does not answer within
 * `NATLANG_MODEL_START_TIMEOUT_MS` (default 120 s) fails the start with its error output.
 */
export function spawnLauncher(host: LlamaHost): LlamaLauncher {
  return async (launch, signal) => {
    let recentError = '';
    const child = host.processes.spawn(launch.command, launch.args, {
      onStderr: text => { recentError = (recentError + text).slice(-16000); },
      onError: failure => { recentError = `${recentError}\n${failure.message}`.slice(-16000); } });
    // llama-server finishes the batch in progress on its first signal (a long prefill takes many seconds on a CPU) and
    // exits at once on a second; a different signal, since a pending signal of the same kind can merge with the first.
    const terminate = (signalName: NodeJS.Signals) => {
      child.kill('SIGTERM'); child.kill('SIGINT');
      process.removeListener(signalName, terminate);
      if (!process.listenerCount(signalName)) process.kill(process.pid, signalName);
    };
    const onExit = () => { if (!child.hasExited()) child.kill('SIGTERM'); };
    process.prependOnceListener('SIGINT', terminate); process.prependOnceListener('SIGTERM', terminate);
    process.once('exit', onExit);
    const stop = async () => {
      process.removeListener('SIGINT', terminate); process.removeListener('SIGTERM', terminate); process.removeListener('exit', onExit);
      await stopProcess(child);
    };
    try {
      const deadline = Date.now() + Number(host.environment.NATLANG_MODEL_START_TIMEOUT_MS ?? 120000);
      while (Date.now() < deadline) {
        signal.throwIfAborted();
        if (child.hasExited()) throw new Error(`managed model server exited during startup\n${recentError.trim()}`);
        try { const response = await host.fetch(launch.endpoint + '/health'); if (response.ok) return { stop }; } catch { /* server is still loading */ }
        await new Promise(resolveWait => setTimeout(resolveWait, 250));
      }
      throw new Error(`managed model server did not become ready within the startup timeout\n${recentError.trim()}`);
    } catch (failure) { await stop(); throw failure; }
  };
}

export function createLlamaServer(host: LlamaHost, options: LlamaServerOptions = {}): LlamaServer {
  const { environment, log } = host;
  const release = options.release ?? DEFAULT_LOCAL_MODEL, headers = options.headers;
  let prerequisites = localModelPrerequisites(host, release);
  let launched: LlamaLaunched | null = null, endpoint: string | null = null, starting: Promise<string> | null = null;
  const stopping = new AbortController();
  let closed = false, plan: SlotPlan | null = null;
  // Loaded tuned adapters: artifact ID -> llama-server adapter index (fixed for the server's lifetime).
  const loraIndex = new Map<string, number>(), nativeStatesLoaded = new Set<string>();
  // Warm openings: each distinct system prompt and tool list is prefilled once per machine and model, then restored.
  let warming = false, inflight = 0, contended = false, sharedServer = false, modelSha256: string | null = null, runtime: string | null = null;
  const openings = new Map<string, Promise<WarmPrefix | undefined>>(), warmed: WarmPrefix[] = [];

  const warmFor = async (request: ModelTurnRequest): Promise<void> => {
    const opening = openingOf(request);
    if (!opening) return;
    let pending = openings.get(opening.key);
    if (!pending) {
      if (inflight > 0) return; // every slot must be idle to receive a state; a later idle moment warms it
      pending = (async () => {
        const base = await start();
        if (!warming) return undefined;
        // A server that also serves other processes is warmed only while none of its slots is working.
        if (contended && !await serverIdle(host, base, headers)) { openings.delete(opening.key); return undefined; }
        const sha = modelSha256 ?? await managedModelSha256(host, release);
        const state = await warmOpening(host, { endpoint: base, ...(headers ? { headers } : {}) }, host.directories.warmPrefixes, opening,
          warmTag(sha, runtime ?? runtimeLabel(prerequisites.runtime.selected), plan!), plan?.slots ?? 1);
        if (state) {
          warmed.push(state);
          log.write(state.created ? `natlang: prefilled the ${state.tokens}-token prompt opening once for this model; later calls and runs restore it\n` :
            `natlang: restored the warm ${state.tokens}-token prompt opening into ${state.restoredSlots} slot${state.restoredSlots === 1 ? '' : 's'}\n`);
          pruneWarmStates(host.directories.warmPrefixes);
        }
        return state;
      })().catch(failure => { log.write(`natlang: prompt warming skipped: ${(failure as Error).message.split('\n')[0]!.slice(0, 200)}\n`); return undefined; });
      openings.set(opening.key, pending);
    }
    await pending;
  };

  const startServer = async (): Promise<string> => {
    const { attach } = options;
    if (attach) {
      plan = { budgetSource: 'configured-parallel', slots: attach.slots, contextPerSlot: attach.contextPerSlot,
        totalContext: attach.slots * attach.contextPerSlot, source: 'configured', budgetBytes: null, modelBytes: 0, kvBytesPerToken: 0,
        formula: 'server supervised by the host' };
      warming = environment.NATLANG_WARM_PREFIXES !== 'off' && attach.warm === true;
      contended = true; modelSha256 = attach.modelSha256; runtime = attach.runtime;
      return attach.endpoint;
    }
    if (!prerequisites.executable && options.ensureRuntime) {
      const installed = await options.ensureRuntime(prerequisites.runtime);
      if (installed?.compatible) prerequisites = { ...localModelPrerequisites(host, release), executable: installed.path };
    }
    if (!prerequisites.executable) {
      const explicit = prerequisites.runtime.candidates.find(candidate => candidate.source === 'explicit');
      if (explicit) throw new Error(`NATLANG_LLAMA_SERVER is incompatible: ${describeLlamaRuntime(prerequisites.runtime)}`);
      throw new Error(`managed model server is unavailable: ${describeLlamaRuntime(prerequisites.runtime)}; run natlang --setup or natlang --runtime install`);
    }
    // A server another natlang process keeps loaded for exactly this model file and build (llama/shared.ts).
    const adapterless = !(options.loraAdapters?.() ?? []).length && !(options.nativeStates?.() ?? []).length;
    const shared = adapterless && options.useSharedServer !== false ?
      await findSharedServer(host, await managedModelSha256(host, release), runtimeLabel(prerequisites.runtime.selected)) : undefined;
    if (shared) {
      plan = { budgetSource: 'configured-parallel', slots: shared.slots, contextPerSlot: shared.contextPerSlot,
        totalContext: shared.slots * shared.contextPerSlot, source: 'configured', budgetBytes: null, modelBytes: 0, kvBytesPerToken: 0,
        formula: `shared model server (pid ${shared.pid})` };
      contended = true; sharedServer = true;
      warming = environment.NATLANG_WARM_PREFIXES !== 'off' && !!shared.warm;
      log.write(`natlang: using the ${shared.model} model kept loaded at ${shared.endpoint} (${shared.slots} slots)\n`);
      return shared.endpoint;
    }
    const modelPath = await ensureModel(host, release);
    if (closed) throw new Error('model session is closed');
    const port = await host.processes.freePort();
    const base = `http://127.0.0.1:${port}`;
    // `-c` is shared among the slots, so it is the per-slot context times the slot count.
    plan = localSlotPlan(host, options.local, modelPath, release);
    const template = templatePath(host, release);
    const args = ['-m', modelPath, '--host', '127.0.0.1', '--port', String(port), '--parallel', String(plan.slots),
      '-c', String(plan.totalContext), '-ngl',
      String(options.local?.gpuLayers ?? 99), '--cache-ram', String(options.local?.cacheRamMiB ?? 256), '--no-webui',
      '--jinja', ...(template ? ['--chat-template-file', template] : []), ...(options.local?.args ?? [])];
    loraIndex.clear();
    const adapters = options.loraAdapters?.() ?? [];
    if (adapters.length) {
      // Loaded off; each request sets its own stack's scales. A server with per-sequence scales (the llama.cpp
      // fork) also batches requests whose stacks differ; others serialize them.
      args.push('--lora-init-without-apply', ...(supportsPerSequenceLora(host, prerequisites.executable) ? ['--lora-per-sequence'] : []));
      for (const adapter of adapters) { loraIndex.set(adapter.id, loraIndex.size); args.push('--lora', adapter.path); }
    }
    // Warm prompt openings (llama/warm.ts) are saved and restored here; not with adapters, whose KV differs.
    warming = environment.NATLANG_WARM_PREFIXES !== 'off' && !adapters.length && !(options.nativeStates?.() ?? []).length;
    if (warming) { mkdirSync(host.directories.warmPrefixes, { recursive: true }); args.push('--slot-save-path', host.directories.warmPrefixes); }
    nativeStatesLoaded.clear();
    const states = options.nativeStates?.() ?? [];
    if (states.length) {
      if (!supportsNativeStates(host, prerequisites.executable))
        throw new Error(`weights-unavailable: ${prerequisites.executable} cannot import native state spans (tuned soft states need the llama.cpp fork's --native-state)`);
      for (const state of states) { nativeStatesLoaded.add(state.id); args.push('--native-state', state.path); }
    }
    log.write(`natlang: starting managed model ${basename(modelPath)} with ${plan.slots} slot${plan.slots === 1 ? '' : 's'} (${plan.contextPerSlot} tokens each; ${plan.budgetSource === 'configured-parallel' ? 'local.parallel' : `${plan.budgetSource === 'default' ? 'default' : plan.budgetSource} KV budget ${Math.round((plan.budgetBytes ?? 0) / 2 ** 20)} MiB`})\n`);
    stopping.signal.throwIfAborted();
    const startedAt = Date.now();
    launched = await (options.launch ?? spawnLauncher(host))({ command: prerequisites.executable, args, port, endpoint: base }, stopping.signal);
    log.write(`natlang: model ready in ${((Date.now() - startedAt) / 1000).toFixed(1)} s\n`);
    return base;
  };

  const start = async (): Promise<string> => {
    if (closed) throw new Error('model session is closed');
    if (endpoint) return endpoint;
    if (starting) return starting;
    starting = startServer();
    try { return endpoint = await starting; }
    finally { starting = null; }
  };

  return {
    start,
    async admit(request, signal) {
      signal?.throwIfAborted();
      const base = await start(); signal?.throwIfAborted();
      await warmFor(request); signal?.throwIfAborted();
      inflight++;
      let released = false;
      return { endpoint: base, release() { if (!released) { released = true; inflight--; } } };
    },
    plan: () => plan,
    loraIndex, nativeStatesLoaded,
    status: () => ({ endpoint, executable: prerequisites.executable, running: Boolean(launched || sharedServer || options.attach && endpoint),
      shared: sharedServer, runtimeBuild: prerequisites.runtime.selected?.build ?? null,
      runtime: runtime ?? (prerequisites.runtime.selected ? runtimeLabel(prerequisites.runtime.selected) : null), warm: [...warmed] }),
    async close() {
      closed = true;
      stopping.abort();
      if (starting) await starting.catch(() => undefined);
      await launched?.stop();
      launched = null; endpoint = null;
    },
  };
}
