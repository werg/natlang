import type { ModelDriver, NatlangRuntime } from '../runtime/runtime.js';
import type { NatlangTarget } from './manifest.js';
import type { AdaptationBinding, ExecutorIdentity, ProgramDescriptor } from '../adaptation/types.js';

export type TargetIO = { input: NodeJS.ReadableStream; output: NodeJS.WritableStream;
  error: NodeJS.WritableStream; color: boolean };
/** What `natlang run` and package launches pass to an application's entry function. */
export type TargetContext = {
  package?: { name: string; version: string; digest: string; root: string };
  dependencies: Record<string, { name: string; version: string; digest: string; root: string }>;
  targetName: string;
  target: NatlangTarget;
  /** Validated portable program metadata and adaptation supplied by a host integration. */
  program?: ProgramDescriptor;
  adaptation?: AdaptationBinding | null;
  executorIdentity?: ExecutorIdentity;
  workspace: string;
  stateDirectory: string;
  traceDirectory: string;
  args: string[];
  io: TargetIO;
  /** Model driver configured by the launcher (managed local model or a profile endpoint). */
  model: ModelDriver;
  /**
   * The launcher's OpenAI-compatible endpoint when its model is one (a profile's external endpoint), for targets that
   * call the same server through another client; absent for managed local models and provider backends.
   */
  modelEndpoint?: { endpoint: string; model: string; apiKeyEnv?: string };
  /** A natlang runtime using that model, with traces written to `traceDirectory`. */
  runtime: NatlangRuntime;
};
export type TargetExecutable = { run(): Promise<number | void> | number | void; close?(): Promise<void> | void };
/** An entry function may run directly (returning an exit code) or return an executable. */
export type TargetMain = (context: TargetContext) => Promise<number | void | TargetExecutable> | number | void | TargetExecutable;
