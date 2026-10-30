/** Node platform: AsyncLocalStorage context propagation, the vm evaluator, the machine's call store and module package loading. */
import { AsyncLocalStorage } from 'node:async_hooks';
import { Worker } from 'node:worker_threads';
import { TypeScriptEnvironment } from '../environment.js';
import { EvalFailure, type EvalEnvironment, type EnvironmentMode } from '../native/evaluator.js';
import type { NatlangRuntimeOptions } from './runtime.js';
import { platformDefaults } from '../platform-defaults.js';
import { WorkspaceModules, findPackageWorkspace } from '../workspace-modules.js';
import type { ReturnErrorText } from '../native/return-errors.js';
import type { ContextStore, Frame } from './context.js';
import type { Platform } from './platform.js';
import { machineStoreRoot, openMachineStore } from '../machine-store.js';
import { learningModule } from '../neuralese/learning.js';
import { neuraleseModule } from '../neuralese/combinators.js';
import { runFolderPython as runPython, type PythonHost } from '../native/folder-python.js';
import { WATCHDOG_SCRIPT, type PythonWatchdog } from '../native/python-watchdog.js';
import { runFolderBash as runBash, type FolderBashOptions, type FolderBashResult } from '../native/folder-shell.js';
import type { Folder } from '../native/scoped-fs.js';
import { gepaModule } from '../gepa/index.js';
import { selectedServiceSurface } from '../native/effects.js';
import { selectedNativeIdentity, selectedNativeSurface, selectedNativeCleanup } from '../native/selected-surface.js';

const storage = new AsyncLocalStorage<Frame | undefined>();
const nodeContextStore: ContextStore = { current: () => storage.getStore(), run: (frame, fn) => storage.run(frame, fn) };

// Python cells time out through a worker thread that sets Pyodide's interrupt flag.
const nodePythonWatchdog: PythonWatchdog = (buffer, ms) => new Worker(`const { workerData } = require('worker_threads'); const buffer = workerData.buffer, ms = workerData.ms; ${WATCHDOG_SCRIPT}`,
  { eval: true, workerData: { buffer, ms } });

/**
 * The Node platform: what `createNatlangRuntime` uses here when no `platform` is passed. It reads the process
 * environment (`NATLANG_CALL_STORE`, `NATLANG_SPECIALIZATION`, `NATLANG_RETURN_ERROR_TEXT`) where the kernel, which reads
 * none, is configured. Build a variant with an object spread (`{ ...nodePlatform(), builtinModules }`): module state belongs
 * to the platform value, so a variant has its own module instances.
 */
export function nodePlatform(): Platform { return defaultNodePlatform ??= createNodePlatform(); }
let defaultNodePlatform: Platform | undefined;

function createNodePlatform(): Platform {
  // Callable-folder modules load packages from the calling task's workspace.
  const packages = new Map<string, WorkspaceModules>();
  return {
    name: 'node',
    contextStore: nodeContextStore,
    environment: options => new TypeScriptEnvironment({ workspace: options.workspace, network: options.network }),
    moduleRealm: () => new TypeScriptEnvironment({ mode: 'retained' }),
    // Every runtime records to the machine's call store unless it says otherwise (NATLANG_CALL_STORE=off disables it).
    callStore: () => { const root = machineStoreRoot(); return root ? openMachineStore(root) : undefined; },
    builtinModules: { 'natlang:learning': () => learningModule, 'natlang:neuralese': () => neuraleseModule, 'natlang:gepa': () => gepaModule },
    packageLoader: (specifier, task) => {
      const workspace = task?.runtime.options.workspace ?? findPackageWorkspace(process.cwd());
      if (!workspace) throw new Error(`package imports need an application package.json: ${specifier}`);
      let loader = packages.get(workspace);
      if (!loader) packages.set(workspace, loader = new WorkspaceModules(workspace));
      return loader.load(specifier);
    },
    runFolderBash,
    runFolderPython,
    get returnErrorText() { return process.env.NATLANG_RETURN_ERROR_TEXT as ReturnErrorText | undefined; },
  };
}

/** Run a Python cell over `folder`; its timeout is enforced by the Node watchdog unless `host` brings another. */
export function runFolderPython(folder: Folder, source: string, host: PythonHost = {}, timeoutMs?: number) {
  return runPython(folder, source, { watchdog: nodePythonWatchdog, ...host }, timeoutMs);
}

/** Run a shell command over `folder`; its Python cells are bounded by the Node watchdog unless `options.python` brings another. */
export function runFolderBash(folder: Folder, command: string, options: FolderBashOptions = {}): Promise<FolderBashResult> {
  return runBash(folder, command, { ...options, python: { watchdog: nodePythonWatchdog, ...options.python } });
}

export const { createNatlangRuntime, compileVirtualProject, projectEntry, loadVirtualNatlang, loadVirtualCallables } = platformDefaults(nodePlatform);

/**
 * Explicit opt-in for an *owned* guest interpreter: a platform (`base`, the Node platform by default) whose per-call
 * and module evaluators both come from `factory`. The factory is supplied by the embedding application (for example
 * tracing-js-self/natlang), so the natlang distribution never silently imports an untraced fallback runtime.
 * This does NOT claim general callback/package/module compatibility: a bounded interpreter must reject any
 * unsupported natlang-generated code.
 *
 * Pass the result as `platform` to `createNatlangRuntime`. Module instances belong to the platform value, so an
 * owned platform never serves exports created by another authority.
 */
export function ownedTracingPlatform(
  factory: (mode: EnvironmentMode, options?: NatlangRuntimeOptions) => EvalEnvironment,
  base: Platform = nodePlatform(),
): Platform {
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
  return { ...base, environment: options => instantiate('fresh', options), moduleRealm: () => instantiate('retained') };
}
/**
 * A platform (`base`, the Node platform by default) with an asynchronous, selected-semantic executor at both Natlang seams.
 *
 * Synchronous entry and module APIs deliberately throw: falling back to the
 * process V8 evaluator would misrepresent semantic execution as model-owned.
 * The driver must implement its actual selection, tool and observation loop;
 * this installation only wires and checks its authority.
 */
export function selectedSemanticPlatform(
  factory: (mode: EnvironmentMode, options?: NatlangRuntimeOptions) => EvalEnvironment,
  base: Platform = nodePlatform(),
): Platform {
  if (typeof factory !== 'function') throw new TypeError('Selected semantic evaluator factory required');
  const instantiate = (mode: EnvironmentMode, options?: NatlangRuntimeOptions): EvalEnvironment => {
    const driver = factory(mode, options);
    if (!driver || driver.mode !== mode ||
        typeof driver.authority !== 'string' ||
        !driver.authority.startsWith('selected-semantic/') ||
        typeof driver.executeAsync !== 'function' ||
        typeof driver.evaluateModuleAsync !== 'function' ||
        typeof driver.fork !== 'function' || typeof driver.close !== 'function')
      throw new TypeError('Selected semantic driver needs async entry and async module execution');
    if (driver.scopeCapabilities?.allowNetwork)
      throw new TypeError('Ambient network access is not a selected capability grant');
    driver.configureNativeBindings?.({ adaptValue: value => selectedNativeSurface(selectedServiceSurface(value)),
      resourceIdentity: selectedNativeIdentity, resourceCleanup: selectedNativeCleanup });
    let closed = false;
    const check = () => {
      if (closed) throw new EvalFailure('Selected semantic environment is closed', []);
    };
    const evaluate = async <T>(work: () => Promise<T>): Promise<T> => {
      check();
      try {
        const result = await work();
        check();
        return result;
      } catch (error) {
        if (error instanceof EvalFailure) throw error;
        const failure = error as { message?: unknown; events?: unknown; debug?: unknown; stack?: unknown };
        throw new EvalFailure(typeof failure?.message === 'string' ? failure.message : String(error),
          Array.isArray(failure?.events) ? failure.events : [],
          { ...(failure?.debug && typeof failure.debug === 'object' ? failure.debug : {}),
            ...(typeof failure?.stack === 'string' ? { sourceStack: failure.stack } : {}) });
      }
    };
    return {
      authority: driver.authority,
      mode,
      scopeCapabilities: { allowModules: true, allowNetwork: false },
      execute() { throw new Error('Selected semantic execution is asynchronous; use executeAsync'); },
      executeAsync(request) { return evaluate(() => driver.executeAsync(request)); },
      evaluateModule() {
        throw new Error('Selected semantic module initialization is asynchronous; use moduleInstanceAsync');
      },
      evaluateModuleAsync(code, bindings) { return evaluate(() => driver.evaluateModuleAsync!(code, bindings)); },
      declarationOf(name) { return driver.declarationOf?.(name); },
      fork() { check(); return instantiate('fresh', options); },
      close() { if (!closed) { closed = true; driver.close(); } },
    };
  };
  return { ...base, environment: options => instantiate('fresh', options), moduleRealm: () => instantiate('retained') };
}
/**
 * Compatibility-first integration. Unlike ownedTracingPlatform this
 * deliberately permits the ordinary Node evaluator. A native selection has
 * NO owned instruction trace and MUST NOT enter replay-verified training data.
 *
 * Routing happens before evaluation, never as a catch-and-retry after partially
 * executed guest code or committed external side effects.
 */
export function hybridTracingPlatform(
  factory: (mode: EnvironmentMode, options?: NatlangRuntimeOptions) => EvalEnvironment,
  options: {
    route?: (request: {kind: 'execute' | 'executeAsync' | 'evaluateModule';
      mode: EnvironmentMode; code: string; path?: string}) => 'owned' | 'native';
    onNativeExecution?: (event: {operation: 'execution.provenance';
      authority: 'native-untraced/1'; kind: string; mode: EnvironmentMode;
      path?: string}) => void;
  } = {},
  base: Platform = nodePlatform(),
): Platform {
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
  return { ...base, environment: options => instantiate('fresh', options), moduleRealm: () => instantiate('retained') };
}

/**
 * Native observation mode for unsupported JS/TS. Records execution boundaries
 * but does NOT invent VM operation events or an exact-replayable owned archive.
 * Retained callable exports can execute after module installation: those calls
 * are outside this bootstrap-level observer.
 */
export function observedNativeTracingPlatform(
  onTrace: (record: {
    schema: 'natlang-native-observation/1';
    phase: 'start' | 'completed' | 'failed' | 'settled';
    kind: 'execute' | 'executeAsync' | 'evaluateModule' | 'exportCall';
    mode: EnvironmentMode;
    code: string;
    path?: string;
    exportName?: string;
    invocationId?: number;
    completion?: 'returned' | 'fulfilled' | 'rejected';
    error?: string;
  }) => void,
  base: Platform = nodePlatform(),
): Platform {
  if (typeof onTrace !== 'function') throw new TypeError('Native trace observer required');
  function instantiate(mode: EnvironmentMode, runtimeOptions?: NatlangRuntimeOptions): EvalEnvironment {
    const native = new TypeScriptEnvironment({
      mode, workspace: runtimeOptions?.workspace, network: runtimeOptions?.network
    });
    let closed = false;
    let invocationSequence = 0;
    const observeExports = (bindings: Record<string, unknown>) => {
      // Decorate only own data-property callables; never evaluate host getters.
      // This is deliberately a boundary trace, not an owned VM instruction trace.
      const exportsValue = (bindings.module as {exports?:unknown}|undefined)?.exports ?? bindings.exports;
      if (!exportsValue || typeof exportsValue !== 'object') return;
      for (const key of Reflect.ownKeys(exportsValue)) {
        const descriptor = Reflect.getOwnPropertyDescriptor(exportsValue,key);
        if (!descriptor || !('value' in descriptor) || typeof descriptor.value !== 'function' ||
            !descriptor.configurable || !descriptor.writable) continue;
        const name = typeof key === 'symbol' ? key.toString() : key;
        const original = descriptor.value as Function;
        const wrap = new Proxy(original, {
          apply(target,thisArg,args) {
            const id = ++invocationSequence;
            onTrace({schema:'natlang-native-observation/1',phase:'start',
              kind:'exportCall',mode,code:'',exportName:name,invocationId:id});
            try {
              const value=Reflect.apply(target,thisArg,args);
              onTrace({schema:'natlang-native-observation/1',phase:'completed',
                kind:'exportCall',mode,code:'',exportName:name,invocationId:id,
                completion:'returned'});
              // Observe later promise settlement while returning the ORIGINAL value.
              // Never replace it with a new promise: callers may depend on identity.
              if (value instanceof Promise) {
                value.then(
                  () => { try { onTrace({schema:'natlang-native-observation/1',
                    phase:'settled',kind:'exportCall',mode,code:'',exportName:name,
                    invocationId:id,completion:'fulfilled'}); } catch {} },
                  (error:unknown) => { try { onTrace({schema:'natlang-native-observation/1',
                    phase:'settled',kind:'exportCall',mode,code:'',exportName:name,
                    invocationId:id,completion:'rejected',
                    error:error instanceof Error ? error.message : String(error)}); } catch {} }
                );
              }
              return value;
            } catch(error) {
              onTrace({schema:'natlang-native-observation/1',phase:'failed',
                kind:'exportCall',mode,code:'',exportName:name,invocationId:id,
                error:error instanceof Error ? error.message : String(error)});
              throw error;
            }
          },
          construct(target,args,newTarget) {
            const id=++invocationSequence;
            onTrace({schema:'natlang-native-observation/1',phase:'start',
              kind:'exportCall',mode,code:'',exportName:name,invocationId:id});
            try {
              const value=Reflect.construct(target,args,newTarget);
              onTrace({schema:'natlang-native-observation/1',phase:'completed',
                kind:'exportCall',mode,code:'',exportName:name,invocationId:id,
                completion:'returned'});
              return value;
            } catch(error) {
              onTrace({schema:'natlang-native-observation/1',phase:'failed',
                kind:'exportCall',mode,code:'',exportName:name,invocationId:id,
                error:error instanceof Error ? error.message : String(error)});
              throw error;
            }
          }
        });
        Reflect.defineProperty(exportsValue,key,{...descriptor,value:wrap});
      }
    };
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
          observeExports(bindings);
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
  return { ...base, environment: options => instantiate('fresh', options), moduleRealm: () => instantiate('retained') };
}

export * from './index.js';
export * from '../calls/index.js';
export { CallStore } from '../calls/store.js';
export { machineStoreRoot, openMachineStore } from '../machine-store.js';
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
