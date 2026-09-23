export type EnvironmentMode = 'fresh' | 'retained';
export type EvalRequest = { code: string; scope: Record<string, unknown>; body: boolean;
  path: string;
  /** Values passed by reference (live objects, functions, capture cells); visible to code as `__live`. */
  live?: Record<string, unknown> };
export type HostEvent = { operation: string; [key: string]: unknown };
export type EvalResult = { result: unknown; events: HostEvent[]; logs?: string[] };

export class EvalFailure extends Error {
  constructor(message: string, readonly events: HostEvent[],
    readonly debug: { sourceStack?: string; logs?: string[] } = {}) {
    super(message); this.name = 'EvalFailure';
  }
}

/** The deliberately small boundary between the interpreter and a TypeScript evaluator. */
export interface EvalEnvironment {
  readonly authority: string;
  readonly mode: EnvironmentMode;
  readonly scopeCapabilities?: { allowModules?: boolean; allowNetwork?: boolean };
  execute(request: EvalRequest): EvalResult;
  executeAsync(request: EvalRequest): Promise<EvalResult>;
  /** Evaluate a module body once; `bindings` are passed by reference. Returns the completion value. */
  evaluateModule(code: string, bindings: Record<string, unknown>): unknown;
  fork(): EvalEnvironment;
  close(): void;
}
