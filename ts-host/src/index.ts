/** @natlang/node: natural-language functions in TypeScript applications on Node. */
export * from './runtime/node.js';
export { compileProject, formatDiagnostics, natlangDeclaration } from './compiler/project.js';
export { buildProject, checkProject, nodeProjectFiles } from './compiler/node-project.js';
export type { BuildOptions, BuildResult, DefinitionManifest, ProjectFiles } from './compiler/project.js';
export { analyzeInlineLambdas } from './compiler/inline.js';
export type { InlineLambdaPlan, NatlangDiagnostic, CapturePlan } from './compiler/inline.js';
export { loadNamedFunction, loadCallableFolder, NatlangSourceError } from './runtime/loader.js';
export type { ItemRecord, NatlangRecord, ModuleRecord, NamespaceRecord } from './runtime/loader.js';
export { EventLoop, EventQueue } from './app/event-loop.js';
export { newPlaygroundProject, editPlaygroundProject, validatePlaygroundProject, assertPlaygroundProject,
  validProjectPath, runPlaygroundProject, projectEntry, projectSignature, traceFrame, admitPlaygroundRun } from './app/playground.js';
export type { PlaygroundProject, PlaygroundRun, PlaygroundDiagnostic, TraceFrame } from './app/playground.js';
export type { AppEvent, Transition, Commit, Failure, StepContext, EventLoopOptions } from './app/event-loop.js';
export { TerminalSessionStore, renderTerminalView, runTerminalShell } from './terminal/index.js';
export type { TerminalCheckpoint, TerminalBlock, TerminalView, TerminalRenderOptions, TerminalShellOptions } from './terminal/index.js';
export { Folder, FolderHandle, FileHandle, EntryHandle, FolderTransaction, FolderBusyError, FolderConflictError } from './native/scoped-fs.js';
export type { FolderSource, FolderAccess, FileContents, EntryStat, SearchMatch, Change, ChangeSet, ChangeKind, EntryKind } from './native/scoped-fs.js';
export { openFolder, saveFolder } from './native/node-files.js';
export type { SavedChange } from './native/node-files.js';
export { DesktopBindings } from './desktop.js';
export type { JobState } from './desktop.js';
export { WorkspaceModules, findPackageWorkspace } from './workspace-modules.js';
export { TypeEnv, TypeSyntaxError, parseType, formatType, fitsType } from './native/types.js';
export type { Type as NatlangType } from './native/types.js';
export type { ModelTurn, ModelTurnRequest } from './contracts.js';
export type { NativeReviewOptions } from './native/agent.js';
export { fetchModel, openAICompatibleModelTurn, chatCompletionModelTurn, httpChatTransport, assembleChatCompletion,
  modelTools, createManagedModelSession, localModelPrerequisites,
  DEFAULT_LOCAL_MODEL, LLAMA_RUNTIME_RELEASE, defaultNatlangRuntimeDirectory,
  discoverLlamaRuntime, inspectLlamaServer, installManagedLlamaRuntime,
  isCompatibleLlamaVersion, describeLlamaRuntime } from './model/index.js';
export type { OpenAICompatibleOptions, OpenAICompatibleExchange, ChatTransport, ChatCompletionOptions, ChatTurnStats,
  HttpChatOptions, ManagedModelSession,
  ManagedModelStatus, ManagedModelRuntimeOptions, ModelProfile, LlamaRuntimeArtifact, LlamaRuntimeRelease,
  LlamaServerInspection, LlamaRuntimeDiscovery } from './model/index.js';
export * from './package/index.js';
export { compileVirtualProject, virtualProjectFiles, virtualSourceFiles, loadVirtualNatlang, loadVirtualCallables } from './runtime/virtual-project.js';
export type { VirtualProject, CompiledProject } from './runtime/virtual-project.js';
