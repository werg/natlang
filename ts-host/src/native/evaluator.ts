export type EnvironmentMode = 'fresh' | 'retained';
export type EvalRequest = { code: string; scope: Record<string, unknown>; body: boolean;
  path: string; effectful: boolean };
export type HostEvent = { operation: string; [key: string]: unknown };
export type EvalResult = { result: unknown; events: HostEvent[]; logs?: string[] };

export class EvalFailure extends Error {
  constructor(message: string, readonly events: HostEvent[]) { super(message); this.name = 'EvalFailure'; }
}

/** The deliberately small boundary between reduction and a crisp evaluator. */
export interface EvalEnvironment {
  readonly authority: string;
  readonly mode: EnvironmentMode;
  readonly host: object;
  readonly scopeCapabilities?: { allowModules?: boolean; allowNetwork?: boolean };
  execute(request: EvalRequest): EvalResult;
  executeAsync(request: EvalRequest): Promise<EvalResult>;
  bindEffect(handler: (capability: string, operation: string, args: unknown[]) => unknown): () => void;
  fork(): EvalEnvironment;
  close(): void;
}
