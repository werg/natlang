export { TypeScriptEnvironment, portable } from './environment.js';
export type { EnvironmentMode, EvalEnvironment, EvalRequest, EvalResult, HostEvent } from './environment.js';
export { compileScopeSnippet, SCOPE_COMPILE_VERSION } from './scope-compiler.js';
export type { ScopeBinding, ScopeCompileDiagnostic, ScopeCompileOptions,
  ScopeCompileResult, ScopeExistingBinding, ScopeSourceSpan } from './scope-compiler.js';
export type { ModelTurn, ModelTurnRequest, RunOptions, RunRequest, RunResult, Source } from './contracts.js';
export { resolveApplicationInputs } from './application-inputs.js';
export type { ApplicationInputs } from './application-inputs.js';
export { DesktopBindings } from './desktop.js';
export type { JobState } from './desktop.js';
export { TypeEnv, TypeSyntaxError, LOOP_VERDICT, parseType, formatType, fitsType, resultType } from './native/types.js';
export type { Type as NatlangType } from './native/types.js';
export { NativeNatlangHost, NativeNatlangHost as NatlangHost } from './native/host.js';
export type { NativeRunRequest } from './native/host.js';
export type { NativeReviewOptions } from './native/agent.js';
export { NodeNativeRuntime, NodeNativeRuntime as NativeRuntime } from './node-runtime.js';
export { NativeSession } from './native/runtime.js';
export type { NativeRuntimeOptions } from './native/runtime.js';
export type { NodeNativeRuntimeOptions } from './node-runtime.js';
export { NativeToolAgent } from './native/agent.js';
export { checkedDefinitions } from './native/codebase.js';
export { loadAnonymousInstruction, loadCodebaseDirectory, loadFunctionFile } from './native/source.js';
export { dumpState as dumpNativeState, loadProgram as loadNativeProgram } from './native/values.js';
export { NativeSourceWorkspace } from './native/workspace.js';
export { LazyDict, MemoryTreeProvider, lazyDict } from './native/host-tree.js';
export type { TreeEntry, TreeProvider } from './native/host-tree.js';
export { textFileTree, validateFileWrites, FILE_TREE_LEAF_TYPE, FILE_WRITE_TYPE } from './native/file-tree.js';
export type { FileTreeLeaf, FileWrite } from './native/file-tree.js';
export { NodeFileTree, commitFileWrites } from './native/node-files.js';
export type { FileWriteReceipt } from './native/node-files.js';
export { Folder, FolderHandle, FileHandle, EntryHandle, FolderTransaction, WriterLock, FolderBusyError, FolderConflictError } from './native/scoped-fs.js';
export type { FolderAccess, FileContents, EntryStat, SearchMatch, Change, ChangeSet, ChangeKind, EntryKind } from './native/scoped-fs.js';
export { admitNativeTrace } from './native/scenario.js';
export type { NativeScenarioContract } from './native/scenario.js';
export { TerminalNatlangApplication, createTerminalHost, TerminalSessionStore,
  renderTerminalView, runTerminalShell, TerminalEventQueue } from './terminal/index.js';
export type { TerminalEvent, TerminalSource, TerminalTransition, TerminalFailure,
  TerminalCommit, TerminalRunner, TerminalApplicationOptions, TerminalCheckpoint,
  TerminalBlock, TerminalView, TerminalRenderOptions, TerminalShellOptions } from './terminal/index.js';
export { openAICompatibleModelTurn, createManagedModelSession, localModelPrerequisites,
  DEFAULT_LOCAL_MODEL, LLAMA_RUNTIME_RELEASE, defaultNatlangRuntimeDirectory,
  discoverLlamaRuntime, inspectLlamaServer, installManagedLlamaRuntime,
  isCompatibleLlamaVersion, describeLlamaRuntime } from './model/index.js';
export type { OpenAICompatibleOptions, OpenAICompatibleExchange, ManagedModelSession,
  ManagedModelStatus, ManagedModelRuntimeOptions, ModelProfile, LlamaRuntimeArtifact, LlamaRuntimeRelease,
  LlamaServerInspection, LlamaRuntimeDiscovery } from './model/index.js';
export * from './package/index.js';
