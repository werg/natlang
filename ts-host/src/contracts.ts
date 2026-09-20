/** Public requests and results for the native TypeScript host. */
export type Source =
  | { kind: 'program'; program: Record<string, unknown> }
  | { kind: 'definitions'; entries: Record<string, Record<string, unknown>>; root: string }
  | { kind: 'file'; path: string };

export type RunOptions = { seed?: { mode?: 'compatibility' | 'derived' | 'backend'; root?: number;
  version?: string }; model?: { temperature?: number; max_turns?: number; max_tokens?: number;
  max_seconds?: number; turn_tokens?: number }; world_seed?: number;
  max_episodes?: number; max_depth?: number; max_actions?: number;
  max_tool_calls?: number; run_id?: string };
export type ModelTurnRequest = { messages: unknown[]; tools: unknown[]; temperature: number;
  seed: number | null; max_tokens: number | null };
export type ModelTurn = { calls?: [string, Record<string, unknown>][]; text?: string;
  raw_calls?: unknown[]; completion_tokens?: number; prompt_tokens?: number;
  value_confidence?: (number | { geometric_mean?: number } | null)[];
  raw_response?: Record<string, unknown> };
export type RunRequest = { source: Source; inputs?: Record<string, unknown>;
  options?: RunOptions; mapWorkers?: number; tracePath?: string;
  modelTurn?: (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;
  streams?: Record<string, AsyncIterable<unknown> | Iterable<unknown>>;
  capabilities?: Record<string, (args: unknown[]) => Promise<unknown> | unknown>;
  signal?: AbortSignal; timeoutMs?: number };
export type RunResult = { outcome: { kind: string; path: string; detail: string };
  value: unknown; emitted: unknown[]; trace: Record<string, unknown>[] | null; run_id: string };
