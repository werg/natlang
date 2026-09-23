/** Model-turn contract shared by model transports and the interpreter's tool agent. */
export type ModelTurnRequest = { messages: unknown[]; tools: unknown[]; temperature: number;
  seed: number | null; max_tokens: number | null };
/** Calls are an ordered, non-atomic batch. Dependent calls belong in later turns. */
export type ModelTurn = { calls?: [string, Record<string, unknown>][]; text?: string;
  raw_calls?: unknown[]; completion_tokens?: number; prompt_tokens?: number;
  value_confidence?: (number | { geometric_mean?: number } | null)[];
  raw_response?: Record<string, unknown> };
