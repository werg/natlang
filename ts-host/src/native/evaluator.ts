export type EnvironmentMode = 'fresh' | 'retained';
export type EvalRequest = { code: string; scope: Record<string, unknown>; body: boolean;
  path: string;
  /** Fail the eval if it has not finished within this many milliseconds (the eval tool's timeout_ms). */
  timeoutMs?: number;
  /** Values passed by reference (live objects, functions, capture cells); visible to code as `__live`. */
  live?: Record<string, unknown> };
export type HostEvent = { operation: string; [key: string]: unknown };
export type EvalResult = { result: unknown; events: HostEvent[]; logs?: string[] };

/** How much of one output a tool result shows; read_page shows the rest. */
export const PAGE_CHARS = 2000;
/** A console writer for one eval; the runtime pages long output for the model. */
export function consoleWriter(logs: string[], show: (value: unknown) => string): (...values: unknown[]) => void {
  return (...values: unknown[]) => {
    logs.push(values.map(value => typeof value === 'string' ? value : value === undefined ? 'undefined' : show(value)).join(' '));
  };
}

/** Settle with `work`, or fail once `timeoutMs` passes; work already started is not stopped. */
export async function withinTimeout<T>(work: Promise<T>, timeoutMs: number | undefined): Promise<T> {
  if (timeoutMs === undefined) return work;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const expired = new Promise<never>((_, reject) => { timer = setTimeout(() =>
    reject(new Error(`eval timed out after ${timeoutMs} ms; effects it already started may still complete`)), timeoutMs); });
  try { return await Promise.race([work, expired]); } finally { clearTimeout(timer); }
}

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
