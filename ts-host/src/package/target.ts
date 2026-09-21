import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import type { NatlangTarget } from './manifest.js';

export type PackageTargetIO = { input: NodeJS.ReadableStream; output: NodeJS.WritableStream;
  error: NodeJS.WritableStream; color: boolean };
export type PackageTargetContext = {
  package: { name: string; version: string; digest: string; root: string };
  targetName: string;
  target: NatlangTarget;
  workspace: string;
  stateDirectory: string;
  traceDirectory: string;
  args: string[];
  io: PackageTargetIO;
  modelTurn?: (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;
  runtime: Record<string, unknown>;
};
export type PackageExecutable = { run(): Promise<number | void> | number | void; close?(): Promise<void> | void };
export type PackageTargetFactory = (context: PackageTargetContext) => PackageExecutable | Promise<PackageExecutable>;
