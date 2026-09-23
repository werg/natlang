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
    const frame = options.frame ?? new NatlangTask(new NatlangRuntime({ services: options.services, agent: options.agent })).frame;
    super({ ...options, environment: options.environment ?? new TypeScriptEnvironment({ mode: 'fresh' }),
      frame, hooks: options.hooks ?? kernelHooks, services: options.services ?? frame.task.services });
  }
}
