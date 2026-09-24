/** Model-turn contract shared by model transports and the interpreter's tool agent. */
/** temperature is omitted unless the application configures one; the model's server keeps its own sampling. */
export type ModelTurnRequest = { messages: unknown[]; tools: unknown[]; temperature?: number;
  seed: number | null; max_tokens: number | null;
  /** "required": the reply must be a tool call (a turn that offers exactly the tool it must use). Default "auto". */
  tool_choice?: 'auto' | 'required' };
/** Calls are an ordered, non-atomic batch. Dependent calls belong in later turns. */
export type ModelTurn = { calls?: [string, Record<string, unknown>][]; text?: string;
  raw_calls?: unknown[]; completion_tokens?: number; prompt_tokens?: number;
  value_confidence?: (number | { geometric_mean?: number } | null)[];
  raw_response?: Record<string, unknown>;
  /** The model's reasoning before this turn's reply, when the backend returns it. */
  reasoning?: string;
  /** The reply stopped at max_tokens. */
  truncated?: boolean };
