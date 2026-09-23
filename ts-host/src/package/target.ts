import type { ModelDriver, NatlangRuntime } from '../runtime/runtime.js';
import type { NatlangTarget } from './manifest.js';

export type TargetIO = { input: NodeJS.ReadableStream; output: NodeJS.WritableStream;
  error: NodeJS.WritableStream; color: boolean };
/** What `natlang run` and package launches pass to an application's entry function. */
export type TargetContext = {
  package?: { name: string; version: string; digest: string; root: string };
  dependencies: Record<string, { name: string; version: string; digest: string; root: string }>;
  targetName: string;
  target: NatlangTarget;
  workspace: string;
  stateDirectory: string;
  traceDirectory: string;
  args: string[];
  io: TargetIO;
  /** Model driver configured by the launcher (managed local model or a profile endpoint). */
  model: ModelDriver;
  /** A natlang runtime using that model, with traces written to `traceDirectory`. */
  runtime: NatlangRuntime;
};
export type TargetExecutable = { run(): Promise<number | void> | number | void; close?(): Promise<void> | void };
/** An entry function may run directly (returning an exit code) or return an executable. */
export type TargetMain = (context: TargetContext) => Promise<number | void | TargetExecutable> | number | void | TargetExecutable;
