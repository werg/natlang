import {recordingEffectObserver} from './native/effects.js';
import {registerTrace,releaseTrace,producerScopeForTrace} from './native/graph.js';
import { TypeScriptEnvironment } from './environment.js';
import { NativeRuntime as PlatformRuntime, runIdentity, type NativeOutcome, type NativeRuntimeOptions } from './native/runtime.js';
import { CallCapture, definitionKey, interfaceHash, type CallStoreLike } from './calls/recorder.js';
import type { DefinitionIdentity } from './calls/types.js';
import { formatType } from './native/types.js';
import type { LambdaNode, Value } from './native/values.js';
import { toHost } from './runtime/kernel.js';
import { kernelHooks } from './runtime/hooks.js';
import { NatlangRuntime, NatlangTask, type NatlangRuntimeOptions } from './runtime/runtime.js';
import { nodePlatform } from './runtime/node.js';
import { runInFrame } from './runtime/context.js';

export type NodeNativeRuntimeOptions = Partial<NativeRuntimeOptions> & Pick<NatlangRuntimeOptions,
  'neuralese' | 'nativeProgramGuidance' | 'program' | 'adaptation' | 'executorIdentity' | 'traceStepStates' | 'specialization' |
  'agentDriver'> & {
  /** The call store the root run is recorded in: the machine's by default, `false` for none (§3.2). */
  calls?: CallStoreLike | false;
  /** The model the agent runs, for the root's call record (its children name their own). */
  modelId?: string;
};

/**
 * The lambda interpreter bound to the invocation kernel and a standalone task. Used by fixtures and
 * tests that drive one lambda directly; applications call natlang functions inside `runtime.run`.
 * The root run is recorded in the call store like a kernel call, with the run ID as its call ID, so its children link to
 * it (the teacher collector's runs are in the store with the rest). The run ID is unique per run (a fresh one when none
 * is given); the run's logical name (`runLabel`) is kept on the record as `run_label`.
 */
export class NodeNativeRuntime extends PlatformRuntime {
  private readonly recording: { capture?: CallCapture };
  private readonly store?: CallStoreLike;
  private readonly rootOptions: NodeNativeRuntimeOptions;
  constructor(options: NodeNativeRuntimeOptions = {}) {
    options = { ...options, ...runIdentity(options) };
    // Calls this one makes run in the task, so the task holds what they are shown of the services too.
    const taskFrame = options.frame ?? new NatlangTask(new NatlangRuntime({ platform: nodePlatform(), services: options.services, agent: options.agent,
      checkEvalBindings: options.checkEvalBindings,
      checkAwaitedFields: options.checkAwaitedFields,
      settleNativeEffects: options.settleNativeEffects,
      ...(options.agentDriver ? { agentDriver: options.agentDriver } : {}),
      exactHostTraceCapture: options.exactHostTraceCapture, neuralese: options.neuralese,
      ...(options.traceStepStates ? { traceStepStates: true } : {}),
      seed: options.seedPolicy,
      nativeProgramGuidance: options.nativeProgramGuidance, program: options.program, adaptation: options.adaptation,
      executorIdentity: options.executorIdentity,
      ...(options.specialization ? { specialization: options.specialization } : {}),
      ...(options.calls !== undefined ? { calls: options.calls } : {}) }),
      { serviceDeclarations: options.declarations, serviceScopes: options.serviceScopes }).frame;
    // This standalone invocation is the caller of its kernel children. Without its
    // identity the trace incorrectly records every first-level child as another root.
    const frame = options.frame ? taskFrame : { ...taskFrame, parentCallId: options.runId! };
    const recording: { capture?: CallCapture } = {};
    super({ ...options, environment: options.environment ?? new TypeScriptEnvironment({ mode: 'fresh' }),
      frame, hooks: options.hooks ?? kernelHooks, services: options.services ?? frame.task.services,
      observeEffect: event => { const observed={...event,...(event.phase==='requested'?{graph_order:frame.task.nextGraphEffectOrder()}:{})};options.observeEffect?.(observed); return recording.capture?.effect(observed, 'agent'); } });
    this.recording = recording;
    this.rootOptions = options;
    // A runtime given a frame runs inside a task that records its own calls.
    if (!options.frame) { try { this.store = frame.task.runtime.callStore(); } catch { /* not recorded */ } }
  }

  override async run(node: LambdaNode): Promise<{ outcome: NativeOutcome; value: Value }> {
    const capture = this.store ? this.openRootCapture(this.store, node) : undefined;
    if(!this.rootOptions.frame){this.frame!.task.beginInvocation(this.options.runId,!!capture);registerTrace(this.options.runId,this.trace,producerScopeForTrace(this.trace));}
    this.recording.capture = capture;
    if(!this.rootOptions.frame){const observe=recordingEffectObserver(this.services);if(observe)this.frame!.task.setInvocationEffectObserver(this.options.runId,observe);}
    capture?.watch(() => this.trace.events as Record<string, unknown>[]);
    let outcome = 'failed', detail = '', output: unknown, hasOutput = false;
    try {
      const result = await super.run(node);
      outcome = result.outcome.kind; detail = result.outcome.detail ?? '';
      if (outcome === 'done') { output = toHost(result.value); hasOutput = true; }
      return result;
    } catch (error) {
      detail = error instanceof Error ? error.message : String(error);
      throw error;
    } finally {
      await this.nativeEffects.drain();
      if(!this.rootOptions.frame){releaseTrace(this.options.runId);this.frame!.task.endInvocation(this.options.runId);}
      this.recording.capture = undefined;
      capture?.finish({ outcome, detail, output, hasOutput, events: this.trace.events as Record<string, unknown>[] });
    }
  }

  /** A manually driven NativeSession also owns its actual evaluation scope,
   * without pretending that the standalone root was durably captured by run(). */
  override async evaluate(...args: Parameters<PlatformRuntime['evaluate']>) {
    return this.asInvocation(() => super.evaluate(...args));
  }

  /** Run host code (a collector's application controller) as this standalone invocation: the Natlang calls it
   * makes are live children of the run. Entering `frame` without this names an invocation that never began. */
  runHosted<T>(fn: () => T | Promise<T>): Promise<T> {
    return this.asInvocation(() => runInFrame(this.frame!, fn));
  }

  private async asInvocation<T>(fn: () => T | Promise<T>): Promise<T> {
    const task=this.frame!.task,id=this.options.runId;
    const own=!this.rootOptions.frame&&!task.isInvocationActive(id);
    if(own)task.beginInvocation(id);
    try { return await fn(); }
    finally { if(own)task.endInvocation(id); }
  }

  /** Open the root's call record; recording never fails the run. */
  private openRootCapture(store: CallStoreLike, node: LambdaNode): CallCapture | undefined {
    try {
      const type = node.type.kind === 'lambda' ? node.type : undefined;
      const params = type ? type.params.fields.map(field => ({ name: field.name, type: formatType(field.type) })) : [];
      const returns = type ? formatType(type.returns) : 'unknown';
      const manifest = (this.rootOptions.manifest ?? {}) as { definition_id?: string; definition_source?: string };
      const id = manifest.definition_id ?? `native:${node.functionName || 'root'}`;
      const body = node.originalBody ?? node.body;
      const identity: DefinitionIdentity = { id, name: node.functionName || 'root', source: manifest.definition_source ?? null,
        key: definitionKey({ id, body, params, returns, types: node.typesSrc, subtype: node.subtype, readout: node.readout }),
        interface: interfaceHash(node.codebase), site: 'named', subtype: node.subtype, params, returns,
        instructions: { complete: false, reason: 'excluded' }, types: node.typesSrc, ...(node.readout ? { readout: node.readout } : {}) };
      const task = this.frame?.task;
      const capture = new CallCapture(store, store.settings(), { callId: this.options.runId, runLabel: this.options.runLabel ?? null,
        parentCallId: this.rootOptions.parentCallId ?? null, parentActionIndex: null, taskId: task?.id ?? 'native',
        programId: null, buildHash: null, programRoot: task?.runtime.options.programRoot ?? null, definition: identity,
        model: { id: this.rootOptions.modelId ?? 'undeclared:native-run', revision: null } });
      identity.instructions = capture.ref(body);
      capture.setInputs(Object.fromEntries(Object.entries(node.args).map(([name, value]) => [name, toHost(value)])));
      if (node.projectTransaction) capture.setFolderInput(node.projectTransaction.folder);
      capture.announce();
      return capture;
    } catch (error) {
      console.warn(`natlang: call recording failed for the native run: ${error instanceof Error ? error.message : String(error)}`);
      return undefined;
    }
  }
}
