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
export { Folder, FolderSnapshot, FolderHandle, FileHandle, EntryHandle, FolderTransaction, FolderBusyError, FolderConflictError } from './native/scoped-fs.js';
export type { FolderSource, FolderAccess, FileContents, EntryStat, SearchMatch, Change, ChangeSet, ChangeKind, EntryKind } from './native/scoped-fs.js';
export { folderFromData, folderToData } from './native/data-layout.js';
export type { FolderDataLayout, FolderDataResult } from './native/data-layout.js';
export { FolderFs, checkBashPolicy, runFolderBash } from './native/folder-shell.js';
export type { FolderBashResult } from './native/folder-shell.js';
export { runFolderPython } from './native/folder-python.js';
export type { PythonHost, PythonResult } from './native/folder-python.js';
export { openFolder, openArchiveFile, saveFolder } from './native/node-files.js';
export { openArchive } from './native/archive.js';
export type { ArchiveFormat } from './native/archive.js';
export type { SavedChange } from './native/node-files.js';
export { DesktopBindings } from './desktop.js';
export type { JobState } from './desktop.js';
export { WorkspaceModules, findPackageWorkspace } from './workspace-modules.js';
export { TypeEnv, TypeSyntaxError, parseType, formatType, fitsType } from './native/types.js';
export type { Type as NatlangType } from './native/types.js';
export type { ModelTurn, ModelTurnRequest } from './contracts.js';
export type { NativeReviewOptions } from './native/agent.js';
export { fetchModel, openAICompatibleModelTurn, chatCompletionModelTurn, httpChatTransport, assembleChatCompletion,
  modelTools, createManagedModelSession, createResolvedModelSession, loadModelConfiguration,
  resolveModelChoice, localModelPrerequisites,
  DEFAULT_LOCAL_MODEL, LLAMA_RUNTIME_RELEASE, defaultNatlangRuntimeDirectory,
  discoverLlamaRuntime, inspectLlamaServer, installManagedLlamaRuntime,
  isCompatibleLlamaVersion, describeLlamaRuntime } from './model/index.js';
export type { OpenAICompatibleOptions, OpenAICompatibleExchange, ChatTransport, ChatCompletionOptions, ChatTurnStats,
  HttpChatOptions, ManagedModelSession,
  ManagedModelStatus, ManagedModelRuntimeOptions, ModelProfile, ResolvedModelChoice,
  LoadedModelConfiguration, ModelSelectionOverrides, LlamaRuntimeArtifact, LlamaRuntimeRelease,
  LlamaServerInspection, LlamaRuntimeDiscovery } from './model/index.js';
export * from './package/index.js';
export { compileVirtualProject, virtualProjectFiles, virtualSourceFiles, loadVirtualNatlang, loadVirtualCallables } from './runtime/virtual-project.js';
export type { VirtualProject, CompiledProject } from './runtime/virtual-project.js';

export * from './adaptation/index.js';

export * from './skills/index.js';
export { directorySkillSource } from './skills/node.js';

export type { FolderProposal } from './native/scoped-fs.js';

export * from './improvement/index.js';
export { learningService, isLearningService, createLearning, objectives, stopGradient, Loss, LearningError, LAW_NAMES, type LawName } from './neuralese/learning.js';
export type { LearningService, Gradient, Optimizer, OptimizerState, Trajectory } from './neuralese/learning.js';
export { COMBINATORS, buildStandardLibrary, loadStandardLibrary, createNeuraleseLibrary } from './neuralese/combinators.js';
export type { StandardLibrary, CombinatorName } from './neuralese/combinators.js';
export { registerBuiltinModule } from './runtime/modules.js';
