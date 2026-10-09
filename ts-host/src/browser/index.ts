/** @natlang/browser: natural-language functions in browser applications. */
import { SlotContextStore, bindAwait, markRaces, propagateSlotFrames, setContextStore } from '../runtime/context.js';
import { setDefaultCallStoreFactory, setDefaultEnvironmentFactory } from '../runtime/runtime.js';
import { BrowserCallStore, browserCallStoreAvailable } from './call-store.js';
import { ModuleUnavailableError, registerBuiltinModule, setModuleRealm, setModuleTarget } from '../runtime/modules.js';
import { neuraleseModule } from '../neuralese/combinators.js';
import { setDefaultLibProvider } from '../compiler/host.js';
import { TypeScriptEnvironment } from './environment.js';

declare const __NATLANG_TS_LIBS__: Record<string, string>;
const slot = new SlotContextStore();
setContextStore(slot);
// Promise callbacks and timers keep the task frame they were scheduled in, as AsyncLocalStorage gives Node.
propagateSlotFrames(slot);
// Eval code runs in the page realm: Promise.race and Promise.any record the calls they race (see markRaces).
markRaces(Promise);
setDefaultEnvironmentFactory(() => new TypeScriptEnvironment());
// Recording is on by default (§3.2): one call store per origin, in its private file system, shared by the page's runtimes.
let originStore: BrowserCallStore | undefined;
setDefaultCallStoreFactory(() => browserCallStoreAvailable() ? originStore ??= new BrowserCallStore() : undefined);
setModuleRealm(() => new TypeScriptEnvironment({ mode: 'retained' }));
setModuleTarget('browser');
setDefaultLibProvider(name => __NATLANG_TS_LIBS__[name]);
// The combinators run against the task's Neuralese service (a server, or startBrowserNeuralese) as on Node.
registerBuiltinModule('natlang:neuralese', () => neuraleseModule);
// Learning (grad, optimizers, adapter and system-prompt training) needs Node: per-task scopes on AsyncLocalStorage
// and content hashing on node:crypto.
registerBuiltinModule('natlang:learning', () => { throw new ModuleUnavailableError('natlang:learning', 'gradients, optimizers and ' +
  'adapter training run on Node (@natlang/node) against a learning service. Train there and give this program the ' +
  'trained blocks or adapters (a .nz file through loadStandardLibrary, or a Neuralese block store); natlang:neuralese ' +
  'combinators and Neuralese calls do work in the browser.'); });
// Compiled browser code restores the natlang task after each await through this hook.
Object.defineProperty(globalThis, '__natlang_bindAwait', { value: bindAwait, configurable: true });

export * from '../runtime/index.js';
export { compileProject, formatDiagnostics } from '../compiler/project.js';
export type { BuildOptions, BuildResult, ProjectFiles, DefinitionManifest } from '../compiler/project.js';
export { compileVirtualProject, virtualProjectFiles, virtualSourceFiles, loadVirtualNatlang, loadVirtualCallables } from '../runtime/virtual-project.js';
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
export { chatCompletionModelTurn, httpChatTransport, assembleChatCompletion, fetchModel, requestLimit, limitedTransport } from '../model/chat-completion.js';
export type { ChatTransport, ChatCompletionOptions, ChatTurnStats, HttpChatOptions, RequestLimit } from '../model/chat-completion.js';
export { openAICompatibleModelTurn } from '../model/openai-compatible.js';
export type { BrowserModelLoadOptions, BrowserInferenceEngine, BrowserModelDiagnostics, BrowserModelSource,
  BrowserModelStatus, LoadedBrowserModel } from './local-model.js';
export { BROWSER_MODEL_CATALOG, loadBrowserModelCatalog, checkModelStorage } from './models.js';
export type { BrowserModelManifest, BrowserModelCatalog, BrowserStorageStatus } from './models.js';
export { probeBrowserGpu } from './gpu.js';
export { startBrowserNeuralese, NeuraleseWasmService, chooseNeuraleseBuild } from './neuralese-wasm.js';
export { cachedModelFile, startNeuraleseModel, type ModelFileRef, type NeuraleseModelManifest } from './model-files.js';
export type { NeuraleseWasmOptions, NeuraleseWasmModule, NeuraleseWasmFactory, StartedNeuralese } from './neuralese-wasm.js';
export { neuraleseServerModelTurn, HttpNeuraleseStore, referenceAdapterLoras } from '../model/neuralese-server.js';
export { MemoryNeuraleseStore, neuraleseContentId } from '../native/neuralese-store.js';
export { encodeBlockBody, decodeBlockBody } from '../model/neuralese-server.js';
export type { NeuraleseServerOptions } from '../model/neuralese-server.js';
export { neuraleseServerInfo, checkNeuraleseReader, type NeuraleseServerInfo } from '../model/neuralese-info.js';
export { supportsNeuralese, NeuraleseUnsupportedError, textToParts, hasNeuraleseSentinel } from '../native/neuralese.js';
export { serverDigester, DIGEST_TYPE, type Digester, type DigestSite } from '../neuralese/digest.js';
// The standard library from and to bytes (Node's variants also take a path).
export { COMBINATORS, buildStandardLibrary, loadStandardLibrary, createNeuraleseLibrary } from '../neuralese/combinators.js';
export type { StandardLibrary, CombinatorName } from '../neuralese/combinators.js';
export { registerBuiltinModule, ModuleUnavailableError } from '../runtime/modules.js';
export type { NeuraleseStore, NeuraleseBlock, NeuraleseBlockMeta, NeuraleseBlockInput, NeuraleseDtype } from '../native/neuralese-store.js';
export type { BrowserGpuCapability } from './gpu.js';
export { newPlaygroundProject, assertPlaygroundProject, editPlaygroundProject, validatePlaygroundProject,
  validProjectPath, runPlaygroundProject, projectEntry, projectSignature, traceFrame, admitPlaygroundRun } from '../app/playground.js';
export type { PlaygroundProject, PlaygroundRun, PlaygroundDiagnostic, TraceFrame } from '../app/playground.js';
export { TypeScriptEnvironment } from './environment.js';
export { Folder, FolderSnapshot, FolderHandle, FileHandle } from '../native/scoped-fs.js';
export { folderFromData, folderToData } from '../native/data-layout.js';
export type { FolderDataLayout, FolderDataResult } from '../native/data-layout.js';
export { openArchive } from '../native/archive.js';
export type { ArchiveFormat } from '../native/archive.js';
export async function runFolderBash(...args: Parameters<typeof import('../native/folder-shell.js').runFolderBash>) {
  return (await import('../native/folder-shell.js')).runFolderBash(...args);
}
export async function runFolderPython(...args: Parameters<typeof import('../native/folder-python.js').runFolderPython>) {
  return (await import('../native/folder-python.js')).runFolderPython(...args);
}
export { TypeEnv, parseType, formatType, fitsType } from '../native/types.js';
export type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export * from '../adaptation/index.js';

export type { FolderProposal } from '../native/scoped-fs.js';
