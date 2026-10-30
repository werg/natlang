/** @natlang/browser: natural-language functions in browser applications. */
import { SlotContextStore, markRaces, propagateSlotFrames } from '../runtime/context.js';
import { platformDefaults } from '../platform-defaults.js';
import type { Platform } from '../runtime/platform.js';
import { BrowserCallStore, browserCallStoreAvailable } from './call-store.js';
import { ModuleUnavailableError } from '../runtime/modules.js';
import { neuraleseModule } from '../neuralese/combinators.js';
import { TypeScriptEnvironment } from './environment.js';
import { webWorkerWatchdog } from '../native/python-watchdog.js';
import type { runFolderPython as runPython } from '../native/folder-python.js';
import { buildStandardLibrary as buildLibrary } from '../neuralese/combinators.js';
import type { OpfsSqliteOptions } from './opfs-sqlite.js';
import { opfsAvailable } from './neuralese-opfs-store.js';

/**
 * The browser platform: what `createNatlangRuntime` uses here when no `platform` is passed. Its first construction sets
 * up the page's realm: promise callbacks and timers keep the task frame they were scheduled in (as AsyncLocalStorage
 * gives Node), `Promise.race` and `Promise.any` record the calls they race, and compiled code reaches the await hook
 * as `globalThis.__natlang_bindAwait`. A page has one context slot, so the platform is created once.
 */
export function browserPlatform(): Platform { return defaultBrowserPlatform ??= createBrowserPlatform(); }
let defaultBrowserPlatform: Platform | undefined;

function createBrowserPlatform(): Platform {
  const slot = new SlotContextStore();
  propagateSlotFrames(slot);
  markRaces(Promise);
  Object.defineProperty(globalThis, '__natlang_bindAwait', { value: slot.bindAwait, configurable: true });
  // Recording is on by default (§3.2): one call store per origin, in its private file system, shared by the page's runtimes.
  let originStore: BrowserCallStore | undefined;
  return {
    name: 'browser',
    contextStore: slot,
    environment: () => new TypeScriptEnvironment({ bindAwait: slot.bindAwait }),
    moduleRealm: () => new TypeScriptEnvironment({ mode: 'retained', bindAwait: slot.bindAwait }),
    callStore: () => browserCallStoreAvailable() ? originStore ??= new BrowserCallStore() : undefined,
    builtinModules: {
      // The combinators run against the task's Neuralese server as on Node.
      'natlang:neuralese': () => neuraleseModule,
      // Learning (grad, optimizers, adapter and system-prompt training) needs Node: per-task scopes on AsyncLocalStorage
      // and content hashing on node:crypto.
      'natlang:learning': () => { throw new ModuleUnavailableError('natlang:learning', 'gradients, optimizers and ' +
        'adapter training run on Node (@natlang/node) against a learning service. Train there and give this program the ' +
        'native states or named adapter tensors (a typed .nz file through loadStandardLibrary, or a native artifact store); natlang:neuralese ' +
        'combinators and Neuralese calls do work in the browser.'); },
    },
    runFolderBash,
    runFolderPython,
  };
}

export const { createNatlangRuntime, compileVirtualProject, projectEntry, loadVirtualNatlang, loadVirtualCallables } = platformDefaults(browserPlatform);
export function buildStandardLibrary(options: Omit<Parameters<typeof buildLibrary>[0], 'platform'> & { platform?: Platform }) {
  return buildLibrary({ platform: browserPlatform(), ...options });
}

export * from '../runtime/index.js';
export { compileProject, formatDiagnostics, NEUTRAL_RUNTIME_SPECIFIER } from '../compiler/project.js';
export type { BuildOptions, BuildResult, ProjectFiles, DefinitionManifest } from '../compiler/project.js';
export { virtualProjectFiles, virtualSourceFiles } from '../runtime/virtual-project.js';
export type { VirtualProject, CompiledProject } from '../runtime/virtual-project.js';
export { loadNamedFunction, loadCallableFolder, NatlangSourceError } from '../runtime/loader.js';
export type { ItemRecord, NatlangRecord, SourceFiles } from '../runtime/loader.js';
export { EventLoop, EventQueue, KeyedEventLoop } from '../app/event-loop.js';
export type { AppEvent, Transition, Commit, Failure, StepContext, EventLoopOptions, KeyedEventLoopOptions, WakeEvent } from '../app/event-loop.js';
export { BrowserDomRenderer } from './dom.js';
export { BrowserCallStore, browserCallStoreAvailable, type BrowserCallStoreOptions } from './call-store.js';
export { wasmCallStore, wasmDatabase, DatabaseMedium } from '../calls/store-wasm.js';
export { CallStore } from '../calls/store-core.js';
export type { UiNode, UiAction } from './dom.js';
export { BrowserLocalModel, compileBrowserTools, loadBrowserLocalModel, wllamaChatTransport } from './local-model.js';
export { loadBrowserRelease, type BrowserRelease } from './tuned-release.js';
export { chatCompletionModelTurn, httpChatTransport, assembleChatCompletion, fetchModel, requestLimit, limitedTransport } from '../model/chat-completion.js';
export type { ChatTransport, ChatCompletionOptions, ChatTurnStats, HttpChatOptions, RequestLimit } from '../model/chat-completion.js';
export { openAICompatibleModelTurn } from '../model/openai-compatible.js';
export type { BrowserModelLoadOptions, BrowserInferenceEngine, BrowserModelDiagnostics, BrowserModelSource,
  BrowserModelStatus, LoadedBrowserModel } from './local-model.js';
export { BROWSER_MODEL_CATALOG, loadBrowserModelCatalog, checkModelStorage } from './models.js';
export type { BrowserModelManifest, BrowserModelCatalog, BrowserStorageStatus } from './models.js';
export { probeBrowserGpu } from './gpu.js';
export { cachedModelFile, removeCachedModelFiles, type ModelFileRef } from './model-files.js';
export { neuraleseServerModelTurn, HttpNeuraleseStore, restoreNativeArtifacts, withRestoredNativeArtifacts, OWNER_HEADER } from '../model/neuralese-server.js';
export { MemoryNeuraleseStore } from '../native/neuralese-store.js';
export { OpfsNeuraleseStore, openBrowserNeuraleseStore, opfsAvailable, type OpfsNeuraleseStoreOptions, type OpfsStorage,
  type OpfsDirectoryHandle } from './neuralese-opfs-store.js';
export type { NeuraleseServerOptions } from '../model/neuralese-server.js';
export { neuraleseServerInfo, checkNeuraleseReader, type NeuraleseServerInfo } from '../model/neuralese-info.js';
export { supportsNeuralese, NeuraleseUnsupportedError, textToParts, hasNeuraleseSentinel, nativeStateConsumerFromInfo,
  type NativeStateConsumer } from '../native/neuralese.js';
export { serverViewer, VIEW_TYPE, type Viewer, type ViewSite, type NativeViewTurn } from '../neuralese/view.js';
// The standard library from and to bytes (Node's variants also take a path).
export { COMBINATORS, loadStandardLibrary, createNeuraleseLibrary } from '../neuralese/combinators.js';
export type { StandardLibrary, CombinatorName } from '../neuralese/combinators.js';
export type { NeuraleseStore, NeuraleseBlockMeta } from '../native/neuralese-store.js';
export type { BrowserGpuCapability } from './gpu.js';
export { newPlaygroundProject, assertPlaygroundProject, editPlaygroundProject, validatePlaygroundProject,
  validProjectPath, runPlaygroundProject, projectSignature, traceFrame, admitPlaygroundRun } from '../app/playground.js';
export type { PlaygroundProject, PlaygroundRun, PlaygroundDiagnostic, TraceFrame } from '../app/playground.js';
export { TypeScriptEnvironment } from './environment.js';
export { Folder, FolderSnapshot, FolderHandle, FileHandle } from '../native/scoped-fs.js';
export { folderFromData, folderToData } from '../native/data-layout.js';
export type { FolderDataLayout, FolderDataResult } from '../native/data-layout.js';
export { openArchive } from '../native/archive.js';
export type { ArchiveFormat } from '../native/archive.js';
export async function runFolderBash(folder: Parameters<typeof import('../native/folder-shell.js').runFolderBash>[0], command: string,
    options: import('../native/folder-shell.js').FolderBashOptions = {}) {
  return (await import('../native/folder-shell.js')).runFolderBash(folder, command, { ...options, python: { watchdog: webWorkerWatchdog, ...options.python } });
}
export type { FolderBashResult, FolderBashOptions } from '../native/folder-shell.js';
/**
 * A SQLite database in the origin private file system (opfs-sqlite.ts), loaded on first use. It needs a dedicated
 * worker: OPFS sync access handles exist only there.
 */
export async function openOpfsSqlite(name: string, options: Omit<OpfsSqliteOptions, 'wasmUrl'> = {}) {
  return (await import('./opfs-sqlite.js')).openOpfsSqlite(name, { ...options, wasmUrl: new URL('./sqlite3.wasm', import.meta.url).href });
}
export type { OpfsSqliteOptions } from './opfs-sqlite.js';
/** Whether `openOpfsSqlite` can open a database here: a dedicated worker of a browser with OPFS sync access handles. */
export function opfsSqliteAvailable(scope: unknown = globalThis): boolean {
  const global = scope as { FileSystemFileHandle?: { prototype?: { createSyncAccessHandle?: unknown } }; navigator?: { storage?: unknown } };
  return typeof global.FileSystemFileHandle?.prototype?.createSyncAccessHandle === 'function' && opfsAvailable(global.navigator?.storage);
}
export type { WasmDb } from '../calls/store-wasm.js';
export { ONCE_EFFECTS } from '../native/effects.js';
export async function runFolderPython(folder: Parameters<typeof runPython>[0], source: string, host: Parameters<typeof runPython>[2] = {}, timeoutMs?: number) {
  return (await import('../native/folder-python.js')).runFolderPython(folder, source, { watchdog: webWorkerWatchdog, ...host }, timeoutMs);
}
export { TypeEnv, parseType, formatType, fitsType } from '../native/types.js';
export type { ModelTurn, ModelTurnRequest, ModelTurnDelta, ModelTurnOptions, ModelContentPart } from '../contracts.js';

export * from '../adaptation/index.js';

export type { FolderProposal } from '../native/scoped-fs.js';

export { nativeStateWritePort, nativeStateRuntimeOptions, nativeTextOperation, type NativeClientOptions, type NativeSourceCapture, type NativeTextResult } from '../model/native-state-client.js';
