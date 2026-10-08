/** Node wiring: AsyncLocalStorage context propagation, the vm evaluator, and module package loading. */
import { AsyncLocalStorage } from 'node:async_hooks';
import { TypeScriptEnvironment } from '../environment.js';
import { WorkspaceModules, findPackageWorkspace } from '../workspace-modules.js';
import { currentFrame, setContextStore, type Frame } from './context.js';
import { setDefaultCallStoreFactory, setDefaultEnvironmentFactory } from './runtime.js';
import { CallStore, machineStoreRoot } from '../calls/store.js';
import { registerBuiltinModule, setModuleRealm, setPackageLoader } from './modules.js';
import { learningModule } from '../neuralese/learning.js';
import { neuraleseModule } from '../neuralese/combinators.js';

const storage = new AsyncLocalStorage<Frame | undefined>();
setContextStore({ current: () => storage.getStore(), run: (frame, fn) => storage.run(frame, fn) });
setDefaultEnvironmentFactory(options => new TypeScriptEnvironment({ workspace: options.workspace, network: options.network }));
setModuleRealm(() => new TypeScriptEnvironment({ mode: 'retained' }));
// Every runtime records to the machine's call store unless it says otherwise (NATLANG_CALL_STORE=off disables it).
setDefaultCallStoreFactory(() => { const root = machineStoreRoot(); return root ? CallStore.open(root) : undefined; });

registerBuiltinModule('natlang:learning', () => learningModule);
registerBuiltinModule('natlang:neuralese', () => neuraleseModule);

// Callable-folder modules load packages from the calling task's workspace.
const packages = new Map<string, WorkspaceModules>();
setPackageLoader(specifier => {
  const workspace = currentFrame()?.task.runtime.options.workspace ?? findPackageWorkspace(process.cwd());
  if (!workspace) throw new Error(`package imports need an application package.json: ${specifier}`);
  let loader = packages.get(workspace);
  if (!loader) packages.set(workspace, loader = new WorkspaceModules(workspace));
  return loader.load(specifier);
});

export * from './index.js';
export * from '../calls/index.js';
export { CallStore, machineStoreRoot } from '../calls/store.js';
export type { CallSummary, CallFilter, HotDefinition, AuditJob } from '../calls/store.js';
export { loadNatlang, loadCallables, applicationContextRecords, nodeSourceFiles, fileTraceSink } from './node-files.js';
