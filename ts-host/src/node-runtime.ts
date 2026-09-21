import { TypeScriptEnvironment } from './environment.js';
import { NativeRuntime as PlatformRuntime, type NativeRuntimeOptions } from './native/runtime.js';

export type NodeNativeRuntimeOptions = Omit<NativeRuntimeOptions, 'environment'> & {
  environment?: TypeScriptEnvironment;
  host?: object;
};

/** Node convenience runtime. The platform-neutral reducer itself requires an explicit evaluator. */
export class NodeNativeRuntime extends PlatformRuntime {
  constructor(options: NodeNativeRuntimeOptions = {}) {
    super({ ...options, environment: options.environment ?? new TypeScriptEnvironment({ mode: 'fresh', host: options.host }) });
  }
}
