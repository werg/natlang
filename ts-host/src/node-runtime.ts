import { TypeScriptEnvironment } from './environment.js';
import { NativeRuntime as PlatformRuntime, type NativeRuntimeOptions } from './native/runtime.js';
import { kernelHooks } from './runtime/hooks.js';
import { NatlangRuntime, NatlangTask } from './runtime/runtime.js';
import './runtime/node.js';

export type NodeNativeRuntimeOptions = Partial<NativeRuntimeOptions>;

/**
 * The lambda interpreter bound to the invocation kernel and a standalone task. Used by fixtures and
 * tests that drive one lambda directly; applications call natlang functions inside `runtime.run`.
 */
export class NodeNativeRuntime extends PlatformRuntime {
  constructor(options: NodeNativeRuntimeOptions = {}) {
    // Calls this one makes run in the task, so the task holds what they are shown of the services too.
    const taskFrame = options.frame ?? new NatlangTask(new NatlangRuntime({ services: options.services, agent: options.agent }),
      { serviceDeclarations: options.declarations, serviceScopes: options.serviceScopes }).frame;
    // This standalone invocation is the caller of its kernel children. Without its
    // identity the trace incorrectly records every first-level child as another root.
    const frame = options.frame ? taskFrame : { ...taskFrame, parentCallId: options.runId ?? 'native-run' };
    super({ ...options, environment: options.environment ?? new TypeScriptEnvironment({ mode: 'fresh' }),
      frame, hooks: options.hooks ?? kernelHooks, services: options.services ?? frame.task.services });
  }
}
