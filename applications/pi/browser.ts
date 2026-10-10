/**
 * pi in the browser: the same harness as `natlang run applications/pi` (index.ts openPi: pi-durable with the
 * natural-language task kinds, pi's coding tools, prompt and subagent), on the browser host's parts:
 *
 * - the workspace is a natlang Folder at /workspace (host/folder-env.ts): read, write and edit change it, and bash runs
 *   natlang's folder shell (just-bash) over it;
 * - the session is pi-durable's SQLite storage in the origin private file system (sqlite-wasm through ts-host's
 *   `openOpfsSqlite`, host/sqlite-wasm.ts), which needs a dedicated worker; elsewhere it is kept in memory;
 * - the executor is the page's natlang runtime (`@natlang/browser`), and the agent model is reached through natlang's
 *   model transport (host/agent-models.ts): an HTTP endpoint, or the in-page WebAssembly Neuralese engine's endpoint
 *   (`startBrowserNeuralese`), whose blocks are archived in the OPFS block store.
 *
 * Built by ts-host/scripts/build-application-browser.mjs under the browser policy (no `node:*` module in its graph);
 * the runtime is the page's `@natlang/browser` module (an import map names it). See README.md "In the browser".
 */
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import { NatlangRuntime, Folder, openBrowserNeuraleseStore, openOpfsSqlite, opfsSqliteAvailable, type NeuraleseStore,
  type OpfsStorage } from '@natlang/browser';
import { Harness } from './vendor/durable/src/harness/harness.ts';
import type { Storage } from './vendor/durable/src/types.ts';
import { MemoryStorage } from './vendor/durable/src/storage/memory.ts';
import { openPi, runPiTask, type Implementations, type RunResult } from './index.ts';
import { agentModels, type AgentModels, type AgentTransport } from './host/agent-models.ts';
import { declareReader, type AgentReader } from './host/natlang-provider.ts';
import { FolderExecutionEnv, WORKSPACE } from './host/folder-env.ts';
import { openWasmSqliteStorage, type WasmSqliteDb } from './host/sqlite-wasm.ts';
import { codingRegistry, ExecutionEnvs } from './extensions/index.ts';
import { companion, CompanionDoc } from './extensions/companion/index.ts';

// The parts, for pages that compose their own harness (pi-durable's Harness and MemoryStorage, as the tests do).
export { Folder, FolderExecutionEnv, WORKSPACE, openWasmSqliteStorage, agentModels, codingRegistry, ExecutionEnvs, runPiTask, Harness,
  MemoryStorage, companion, CompanionDoc, BACKGROUND_CONTEXT };
export type { RunResult };

/** The agent model's server, as the CLI's --agent-* flags name it. */
export type BrowserAgent = {
  /** An OpenAI-compatible or Neuralese server, or the in-page engine's endpoint (`startBrowserNeuralese().endpoint`). */
  endpoint: string;
  model: string;
  apiKey?: string;
  /** `natlang` (default here: natlang's own transport, which speaks the Neuralese wire standard) or `pi-ai`. */
  transport?: AgentTransport;
  /** What the agent model reads: `text` (default) or a Neuralese dialect, checked against the server's info. */
  reader?: string;
  contextWindow?: number;
  maxTokens?: number;
};

export type BrowserPiOptions = {
  /** The executor: the page's runtime, which runs pi's natural-language functions. */
  natlang: NatlangRuntime;
  /** The agent model: its server, or pi-ai models with the agent's ref (a custom provider). */
  agent: BrowserAgent | AgentModels;
  /** The agent's thinking level (default off). */
  thinking?: string;
  /** The workspace; the agent's tools and shell see it at /workspace. */
  folder: Folder;
  /** The agent's working directory under /workspace (default /workspace). */
  cwd?: string;
  /**
   * The session: a Storage, a sqlite-wasm database, or the name of a database in the origin private file system
   * (default `pi-session`). OPFS needs a dedicated worker; anywhere else the session is kept in memory.
   */
  session?: Storage | WasmSqliteDb | string;
  /**
   * Where the blocks of a Neuralese agent are archived: a block store, or the OPFS to keep them in (default the
   * runtime's store, else `navigator.storage`, else memory). Unused for a text reader.
   */
  blocks?: NeuraleseStore | OpfsStorage;
  /** Network access for the agent's shell commands (default off). */
  network?: boolean;
  /** Run the companion beside the agent (COMPANION.md). */
  companion?: boolean;
  /** Pluggable hot paths: `crisp` (the default), `nl` or `shadow` per point. */
  implementations?: Partial<Implementations>;
  /** pi's package directory, as the prompt's docs section names it (default /pi). */
  packageDir?: string;
  onReport?: (error: unknown) => void;
  onPhase?: Parameters<typeof openPi>[0]['onPhase'];
};

export type BrowserPi = {
  harness: Harness;
  /** Where the session is kept: `opfs` (the named database), `sqlite` (a database given), `memory`, or `given`. */
  session: 'opfs' | 'sqlite' | 'memory' | 'given';
  /** The environments of the workspace's directories. */
  envs: ExecutionEnvs<FolderExecutionEnv>;
  /** The executor, with the block archive when the agent reads Neuralese. */
  natlang: NatlangRuntime;
  /** Run one task to its answer. */
  run(task: string, signal?: AbortSignal): Promise<RunResult>;
  close(): Promise<void>;
};

const isStorage = (value: unknown): value is Storage => typeof (value as Storage).commit === 'function';
const isStore = (value: unknown): value is NeuraleseStore => typeof (value as NeuraleseStore).put === 'function';
const isModels = (agent: BrowserAgent | AgentModels): agent is AgentModels => 'models' in agent;

/**
 * The session's storage: the given one, pi-durable's SQLite storage in the given sqlite-wasm database or in the named
 * OPFS database, or memory when OPFS cannot hold one here (the reason is reported).
 */
export async function openBrowserSession(session: Storage | WasmSqliteDb | string = 'pi-session', onReport?: (error: unknown) => void):
    Promise<{ storage: Storage; kind: BrowserPi['session'] }> {
  if (typeof session !== 'string') return isStorage(session) ? { storage: session, kind: 'given' } :
    { storage: await openWasmSqliteStorage(session), kind: 'sqlite' };
  if (opfsSqliteAvailable()) {
    try { return { storage: await openWasmSqliteStorage(await openOpfsSqlite(session, { pool: 'natlang-pi' })), kind: 'opfs' }; }
    catch (error) { onReport?.(new Error(`the session is kept in memory: OPFS refused the database (${error instanceof Error ? error.message : String(error)})`)); }
  } else onReport?.(new Error('the session is kept in memory: OPFS databases need a dedicated worker'));
  return { storage: new MemoryStorage(), kind: 'memory' };
}

/** Open pi on the browser host. */
export async function openBrowserPi(options: BrowserPiOptions): Promise<BrowserPi> {
  const context = BACKGROUND_CONTEXT;
  const cwd = options.cwd ?? WORKSPACE;
  let natlang = options.natlang;
  let models: AgentModels;
  if (isModels(options.agent)) models = options.agent;
  else {
    const agent = options.agent;
    const transport = agent.transport ?? 'natlang';
    const root = agent.endpoint.replace(/\/+$/, '').replace(/\/v1$/, '');
    const reader: AgentReader = transport === 'natlang' ? await declareReader(agent.reader ?? 'text', root, { apiKey: agent.apiKey }) :
      { kind: 'text' };
    // A Neuralese agent's blocks are archived with the runtime, so a view outlives the engine that wrote it.
    let store = natlang.options.neuralese?.store;
    if (reader.kind === 'neuralese' && (options.blocks || !store)) {
      store = options.blocks && isStore(options.blocks) ? options.blocks :
        await openBrowserNeuraleseStore(options.blocks ? { storage: options.blocks } : {});
      natlang = new NatlangRuntime({ ...natlang.options, neuralese: { ...natlang.options.neuralese, store } });
    }
    models = agentModels({ endpoint: root, modelId: agent.model, ...(agent.apiKey === undefined ? {} : { apiKey: agent.apiKey }),
      transport, reader, ...(agent.contextWindow === undefined ? {} : { contextWindow: agent.contextWindow }),
      ...(agent.maxTokens === undefined ? {} : { maxTokens: agent.maxTokens }), ...(store ? { store } : {}) });
  }
  const { storage, kind } = await openBrowserSession(options.session, options.onReport);
  const envs = new ExecutionEnvs(cwd, dir => new FolderExecutionEnv({ folder: options.folder, cwd: dir, network: options.network ?? false }));
  const registry = codingRegistry(natlang, { prompt: { cwd, packageDir: options.packageDir ?? '/pi' } });
  let opened: Harness | undefined;
  if (options.companion) registry.install(companion(natlang, { harness: () => opened!, onReport: options.onReport }));
  const harness = await openPi({ storage, natlang, models: models.models, registry, env: envs.env,
    ...(options.implementations ? { implementations: options.implementations } : {}),
    ...(options.onReport ? { onReport: options.onReport } : {}), ...(options.onPhase ? { onPhase: options.onPhase } : {}) }, context);
  opened = harness;
  return { harness, session: kind, envs, natlang,
    run: (task, signal) => runPiTask(harness, { model: models.ref, thinkingLevel: options.thinking ?? 'off', cwd }, task, context, signal),
    async close() { await harness.close(context); await envs.cleanup(context); } };
}
