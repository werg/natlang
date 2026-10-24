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
  // Every accepted evaluation uses the owned interpreter.
  const route = options.route ?? (() => 'owned' as const);
  function instantiate(mode: EnvironmentMode, runtimeOptions?: NatlangRuntimeOptions): EvalEnvironment {
    const owned = factory(mode, runtimeOptions);
    if (!owned || owned.mode !== mode || !owned.authority?.startsWith('owned-') ||
        typeof owned.execute !== 'function' || typeof owned.executeAsync !== 'function' ||
        typeof owned.evaluateModule !== 'function' ||
        typeof owned.close !== 'function' || typeof owned.fork !== 'function')
      throw new TypeError('Hybrid tracing factory returned an incompatible owned evaluator');
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
    return {
      authority: 'owned-trace-required/1',
      mode,
      scopeCapabilities: {allowModules: false, allowNetwork: false},
      execute(request) {
        if (choose('execute', request.code, request.path) === 'owned')
          return owned.execute(request);
        throw new Error('Trace required: native execution has no owned semantic archive');
      },
      async executeAsync(request) {
        if (choose('executeAsync', request.code, request.path) === 'owned')
          return owned.executeAsync(request);
        throw new Error('Trace required: native execution has no owned semantic archive');
      },
      evaluateModule(code, bindings) {
        if (choose('evaluateModule', code) === 'owned')
          return owned.evaluateModule(code, bindings);
        throw new Error('Trace required: native execution has no owned semantic archive');
      },
      declarationOf(name) { return owned.declarationOf?.(name); },
      fork() { return instantiate('fresh', runtimeOptions); },
      close() {
        if (closed) return;
        closed = true;
        owned.close();
      }
    };
  }
  setDefaultEnvironmentFactory(options => instantiate('fresh', options));
  setModuleRealm(() => instantiate('retained'));
}

/**
 * Native observation mode for unsupported JS/TS. Records execution boundaries
 * but does NOT invent VM operation events or an exact-replayable owned archive.
 * Retained callable exports can execute after module installation: those calls
 * are outside this bootstrap-level observer.
 */
export function installObservedNativeTracingEnvironment(
  onTrace: (record: {
    schema: 'natlang-native-observation/1';
    phase: 'start' | 'completed' | 'failed';
    kind: 'execute' | 'executeAsync' | 'evaluateModule';
    mode: EnvironmentMode;
    code: string;
    path?: string;
    error?: string;
  }) => void,
): void {
  if (typeof onTrace !== 'function') throw new TypeError('Native trace observer required');
  function instantiate(mode: EnvironmentMode, runtimeOptions?: NatlangRuntimeOptions): EvalEnvironment {
    const native = new TypeScriptEnvironment({
      mode, workspace: runtimeOptions?.workspace, network: runtimeOptions?.network
    });
    let closed = false;
    const emit = (phase: 'start'|'completed'|'failed',
      kind: 'execute'|'executeAsync'|'evaluateModule',code: string,path?: string,error?: unknown) => {
      const record = {
        schema:'natlang-native-observation/1' as const,phase,kind,mode,code,path,
        ...(error === undefined ? {} : {error: error instanceof Error ? error.message : String(error)})
      };
      onTrace(record);
    };
    const start = (kind:'execute'|'executeAsync'|'evaluateModule',code:string,path?:string) => {
      if (closed) throw new Error('Observed native environment is closed');
      emit('start',kind,code,path);
    };
    return {
      authority:'native-observed/1',
      mode,
      scopeCapabilities:{allowModules:true,allowNetwork:Boolean(runtimeOptions?.network)},
      execute(request) {
        start('execute',request.code,request.path);
        try {
          const result=native.execute(request);
          emit('completed','execute',request.code,request.path);
          return {...result,events:[
            {operation:'execution.provenance',authority:'native-observed/1',
              traceSchema:'natlang-native-observation/1'},...result.events
          ]};
        } catch (error) {
          emit('failed','execute',request.code,request.path,error);
          throw error;
        }
      },
      async executeAsync(request) {
        start('executeAsync',request.code,request.path);
        try {
          const result=await native.executeAsync(request);
          emit('completed','executeAsync',request.code,request.path);
          return {...result,events:[
            {operation:'execution.provenance',authority:'native-observed/1',
              traceSchema:'natlang-native-observation/1'},...result.events
          ]};
        } catch (error) {
          emit('failed','executeAsync',request.code,request.path,error);
          throw error;
        }
      },
      evaluateModule(code,bindings) {
        start('evaluateModule',code);
        try {
          const result=native.evaluateModule(code,bindings);
          emit('completed','evaluateModule',code);
          return result;
        } catch(error) {
          emit('failed','evaluateModule',code,undefined,error);
          throw error;
        }
      },
      declarationOf(name) {return native.declarationOf(name);},
      fork() {return instantiate('fresh',runtimeOptions);},
      close() {if(!closed){closed=true;native.close();}}
    };
  }
  setDefaultEnvironmentFactory(options=>instantiate('fresh',options));
  setModuleRealm(()=>instantiate('retained'));
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
