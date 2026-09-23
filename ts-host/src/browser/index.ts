/** @natlang/browser: natural-language functions in browser applications. */
import { SlotContextStore, bindAwait, setContextStore } from '../runtime/context.js';
import { setDefaultEnvironmentFactory, setDefaultSystemPrompt } from '../runtime/runtime.js';
import { setModuleRealm, setModuleTarget } from '../runtime/modules.js';
import { setDefaultLibProvider } from '../compiler/host.js';
import { TOOLS_PROMPT } from '../native/prompt.js';
import { TypeScriptEnvironment } from './environment.js';

declare const __NATLANG_TS_LIBS__: Record<string, string>;
setContextStore(new SlotContextStore());
setDefaultEnvironmentFactory(() => new TypeScriptEnvironment());
setModuleRealm(() => new TypeScriptEnvironment({ mode: 'retained' }));
setModuleTarget('browser');
setDefaultSystemPrompt(() => TOOLS_PROMPT);
setDefaultLibProvider(name => __NATLANG_TS_LIBS__[name]);
// Compiled browser code restores the natlang task after each await through this hook.
Object.defineProperty(globalThis, '__natlang_bindAwait', { value: bindAwait, configurable: true });

export * from '../runtime/index.js';
export { compileProject, formatDiagnostics } from '../compiler/project.js';
export type { BuildOptions, BuildResult, ProjectFiles, DefinitionManifest } from '../compiler/project.js';
export { compileVirtualProject, virtualProjectFiles, virtualSourceFiles, loadVirtualNatlang, loadVirtualCallables } from '../runtime/virtual-project.js';
export type { VirtualProject, CompiledProject } from '../runtime/virtual-project.js';
export { loadNamedFunction, loadCallableFolder, NatlangSourceError } from '../runtime/loader.js';
export type { ItemRecord, NatlangRecord, SourceFiles } from '../runtime/loader.js';
export { EventLoop, EventQueue } from '../app/event-loop.js';
export type { AppEvent, Transition, Commit, Failure, StepContext, EventLoopOptions } from '../app/event-loop.js';
export { BrowserDomRenderer } from './dom.js';
export type { UiNode, UiAction } from './dom.js';
export { BrowserLocalModel, compileBrowserTools, loadBrowserLocalModel } from './local-model.js';
export type { BrowserModelLoadOptions, BrowserInferenceEngine, BrowserModelDiagnostics, BrowserModelSource,
  BrowserModelStatus, LoadedBrowserModel } from './local-model.js';
export { BROWSER_MODEL_CATALOG, loadBrowserModelCatalog, checkModelStorage } from './models.js';
export type { BrowserModelManifest, BrowserModelCatalog, BrowserStorageStatus } from './models.js';
export { probeBrowserGpu } from './gpu.js';
export type { BrowserGpuCapability } from './gpu.js';
export { newPlaygroundProject, assertPlaygroundProject, editPlaygroundProject, validatePlaygroundProject,
  validProjectPath, runPlaygroundProject, projectEntry, projectSignature, traceFrame, admitPlaygroundRun } from '../app/playground.js';
export type { PlaygroundProject, PlaygroundRun, PlaygroundDiagnostic, TraceFrame } from '../app/playground.js';
export { TypeScriptEnvironment } from './environment.js';
export { Folder, FolderHandle, FileHandle } from '../native/scoped-fs.js';
export { TypeEnv, parseType, formatType, fitsType } from '../native/types.js';
export type { ModelTurn, ModelTurnRequest } from '../contracts.js';
