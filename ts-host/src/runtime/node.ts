/** Node wiring: AsyncLocalStorage context propagation, the vm evaluator, and module package loading. */
import { AsyncLocalStorage } from 'node:async_hooks';
import { Worker } from 'node:worker_threads';
import { TypeScriptEnvironment } from '../environment.js';
import type { EvalEnvironment, EnvironmentMode } from '../native/evaluator.js';
import type { NatlangRuntimeOptions } from './runtime.js';
import { WorkspaceModules, findPackageWorkspace } from '../workspace-modules.js';
import { currentFrame, setContextStore, type Frame } from './context.js';
import { setDefaultCallStoreFactory, setDefaultEnvironmentFactory } from './runtime.js';
import { CallStore, machineStoreRoot } from '../calls/store.js';
import { registerBuiltinModule, setModuleRealm, setPackageLoader } from './modules.js';
import { learningModule } from '../neuralese/learning.js';
import { neuraleseModule } from '../neuralese/combinators.js';
import { setPythonWatchdog, WATCHDOG_SCRIPT } from '../native/folder-python.js';
import { gepaModule } from '../gepa/index.js';

const storage = new AsyncLocalStorage<Frame | undefined>();
setContextStore({ current: () => storage.getStore(), run: (frame, fn) => storage.run(frame, fn) });
setDefaultEnvironmentFactory(options => new TypeScriptEnvironment({ workspace: options.workspace, network: options.network }));
setModuleRealm(() => new TypeScriptEnvironment({ mode: 'retained' }));

/**
 * Explicit, process-wide opt-in for an *owned* guest interpreter.
 *
 * Installs both natlang evaluation seams. The factory is supplied by the
 * embedding application (for example tracing-js-self/natlang), so the
 * natlang distribution never silently imports an untraced fallback runtime.
 * This does NOT claim general callback/package/module compatibility: a
 * bounded interpreter must reject any unsupported natlang-generated code.
 *
 * Call before starting tasks or loading callable-folder modules. This mutates
 * the platform defaults; already-created module instances are not migrated.
 */
export function installOwnedTracingEnvironment(
  factory: (mode: EnvironmentMode, options?: NatlangRuntimeOptions) => EvalEnvironment
): void {
  if (typeof factory !== 'function') throw new TypeError('Owned tracing evaluator factory required');
  function instantiate(mode: EnvironmentMode, options?: NatlangRuntimeOptions): EvalEnvironment {
    const evaluator = factory(mode, options);
    if (!evaluator || evaluator.mode !== mode ||
        typeof evaluator.authority !== 'string' ||
        !evaluator.authority.startsWith('owned-') ||
        typeof evaluator.execute !== 'function' ||
        typeof evaluator.executeAsync !== 'function' ||
        typeof evaluator.evaluateModule !== 'function' ||
        typeof evaluator.close !== 'function' ||
        typeof evaluator.fork !== 'function')
      throw new TypeError('Owned tracing factory returned an incompatible evaluator; native fallback forbidden');
    if (evaluator.scopeCapabilities?.allowNetwork)
      throw new TypeError('An owned tracing evaluator cannot declare ambient network access');
    return evaluator;
  }
  setDefaultEnvironmentFactory(options => instantiate('fresh', options));
  setModuleRealm(() => instantiate('retained'));
}
/**
 * Compatibility-first integration. Unlike installOwnedTracingEnvironment this
 * deliberately permits the ordinary Node evaluator. A native selection has
 * NO owned instruction trace and MUST NOT enter replay-verified training data.
 *
 * Routing happens before evaluation, never as a catch-and-retry after partially
 * executed guest code or committed external side effects.
 */
export function installHybridTracingEnvironment(
  factory: (mode: EnvironmentMode, options?: NatlangRuntimeOptions) => EvalEnvironment,
  options: {
    route?: (request: {kind: 'execute' | 'executeAsync' | 'evaluateModule';
      mode: EnvironmentMode; code: string; path?: string}) => 'owned' | 'native';
    onNativeExecution?: (event: {operation: 'execution.provenance';
      authority: 'native-untraced/1'; kind: string; mode: EnvironmentMode;
      path?: string}) => void;
  } = {}
): void {
  if (typeof factory !== 'function') throw new TypeError('Hybrid owned evaluator factory required');
  if (!options || typeof options !== 'object' ||
      (options.route !== undefined && typeof options.route !== 'function') ||
      (options.onNativeExecution !== undefined && typeof options.onNativeExecution !== 'function'))
    throw new TypeError('Invalid hybrid tracing options');
  // Compatibility-first is the intentional default. Applications must select
  // the owned route for any invocation whose output may be admitted as a trace.
  const route = options.route ?? (() => 'owned' as const);
  function instantiate(mode: EnvironmentMode, runtimeOptions?: NatlangRuntimeOptions): EvalEnvironment {
    const owned = factory(mode, runtimeOptions);
    if (!owned || owned.mode !== mode || !owned.authority?.startsWith('owned-') ||
        typeof owned.execute !== 'function' || typeof owned.executeAsync !== 'function' ||
        typeof owned.evaluateModule !== 'function' ||
        typeof owned.close !== 'function' || typeof owned.fork !== 'function')
      throw new TypeError('Hybrid tracing factory returned an incompatible owned evaluator');
    const native = new TypeScriptEnvironment({
      mode, workspace: runtimeOptions?.workspace, network: runtimeOptions?.network
    });
    let closed = false;
    const choose = (kind: 'execute' | 'executeAsync' | 'evaluateModule', code: string, path?: string) => {
      if (closed) throw new Error('Hybrid tracing environment is closed');
      const decision = route({kind, mode, code, path});
      if (decision !== 'owned' && decision !== 'native')
        throw new TypeError('Hybrid tracing route must return owned or native');
      if (decision === 'native')
        throw new Error('Trace required: native execution has no owned semantic archive');
      return decision;
    };
    const mark = (kind: 'execute' | 'executeAsync' | 'evaluateModule', path?: string) => {
      const event = {operation: 'execution.provenance' as const,
        authority: 'native-untraced/1' as const, kind, mode, path};
      options.onNativeExecution?.(event);
      return event;
    };
    return {
      authority: 'owned-trace-required/1',
      mode,
      scopeCapabilities: {allowModules: false, allowNetwork: false},
      execute(request) {
        if (choose('execute', request.code, request.path) === 'owned')
          return owned.execute(request);
        const marker = mark('execute', request.path);
        const result = native.execute(request);
        return {...result, events: [marker, ...result.events]};
      },
      async executeAsync(request) {
        if (choose('executeAsync', request.code, request.path) === 'owned')
          return owned.executeAsync(request);
        const marker = mark('executeAsync', request.path);
        const result = await native.executeAsync(request);
        return {...result, events: [marker, ...result.events]};
      },
      evaluateModule(code, bindings) {
        if (choose('evaluateModule', code) === 'owned')
          return owned.evaluateModule(code, bindings);
        mark('evaluateModule');
        return native.evaluateModule(code, bindings);
      },
      declarationOf(name) { return owned.declarationOf?.(name) ?? native.declarationOf?.(name); },
      fork() { return instantiate('fresh', runtimeOptions); },
      close() {
        if (closed) return;
        closed = true;
        try { owned.close(); } finally { native.close(); }
      }
    };
  }
  setDefaultEnvironmentFactory(options => instantiate('fresh', options));
  setModuleRealm(() => instantiate('retained'));
}

// Every runtime records to the machine's call store unless it says otherwise (NATLANG_CALL_STORE=off disables it).
setDefaultCallStoreFactory(() => { const root = machineStoreRoot(); return root ? CallStore.open(root) : undefined; });

// Python cells time out through a worker thread that sets Pyodide's interrupt flag.
setPythonWatchdog((buffer, ms) => new Worker(`const { workerData } = require('worker_threads'); const buffer = workerData.buffer, ms = workerData.ms; ${WATCHDOG_SCRIPT}`,
  { eval: true, workerData: { buffer, ms } }));

registerBuiltinModule('natlang:learning', () => learningModule);
registerBuiltinModule('natlang:neuralese', () => neuraleseModule);
registerBuiltinModule('natlang:gepa', () => gepaModule);

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
export type { CallSummary, CallFilter, HotDefinition, AuditJob, FindingInput, FindingRow, SavingsRow } from '../calls/store.js';
export * from '../calls/replay.js';
export * from '../calls/judge.js';
export * from '../calls/mining.js';
export * from '../calls/offline.js';
export * from '../calls/specializer.js';
export * from '../calls/groups.js';
export * from '../calls/findings.js';
export * from '../calls/corpus.js';
export { tracesService, TRACES_DECLARATIONS } from '../calls/service.js';
export type { TraceCall } from '../calls/service.js';
export { loadNatlang, loadCallables, applicationContextRecords, nodeSourceFiles, fileTraceSink } from './node-files.js';
