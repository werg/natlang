/** Model-turn contract shared by model transports and the interpreter's tool agent. */
/** temperature is omitted unless the application configures one; the model's server keeps its own sampling. */
/** Invocation identity is collector metadata; transports do not send it as model input. */
export type ModelTurnRequest = { invocation_id?: string; messages: unknown[]; tools: unknown[]; temperature?: number;
  seed: number | null; max_tokens: number | null;
  /** "required": the reply must be a tool call (a turn that offers exactly the tool it must use). Default "auto". */
  tool_choice?: 'auto' | 'required';
  /** Weight adapters bound in the calling function's context (Neuralese servers only; others refuse the call). */
  adapters?: { id: string; scale: number }[];
  /**
   * Template readout (Neuralese servers only): the reply is forced to a call of `call` with `arguments`, cut at its
   * `value` argument; `write` makes the value a written block and closes the call, `decode` decodes the value.
   */
  template?: { call: string; arguments: Record<string, unknown>; value: 'write' | 'decode';
    /** Optional size hint for a written value: exactly this many vectors, no stop decision; `passes` writes the block
     * block-wise in that many parallel passes. Never required: without it the stop head decides. */
    length?: number; passes?: number };
  /** Guided generation (natlang's servers only): see the agent's `guidance` option. */
  guidance?: boolean | Record<string, unknown> };
/**
 * Neuralese content (S0 §10, S4 §3). In a request, a message's `content` and a tool call's `function.arguments` may be
 * an array of parts instead of a string, where they carry soft values; in a reply, `text` and string call arguments
 * may be such arrays, where the server wrote a block. A transport that carries parts sets `neuralese: true` on its
 * driver function; the runtime fails a call with `neuralese-unsupported-backend` rather than send blocks elsewhere.
 */
export type ModelContentPart = { type: 'text'; text: string } | { type: 'neuralese'; id: string };
/** Calls are an ordered, non-atomic batch. Dependent calls belong in later turns. */
export type ModelTurn = { calls?: [string, Record<string, unknown>][]; text?: string;
  raw_calls?: unknown[]; completion_tokens?: number; prompt_tokens?: number;
  value_confidence?: (number | { geometric_mean?: number } | null)[];
  raw_response?: Record<string, unknown>;
  /** The model's reasoning before this turn's reply, when the backend returns it. */
  reasoning?: string;
  /** A synthetic plan elicited in a separate required-tool turn before this reply. */
  /** null means planning was requested but the provider did not produce a valid plan. */
  execution_plan?: string | null;
  /** The reply stopped at max_tokens. */
  truncated?: boolean };


/**
 * Decision readout (single pass over the prompt): score a finite set of assistant replies after `messages`. A
 * driver that can do this carries a `decide` function. `log_probs[i]` is the total log-probability of `options[i]`
 * as the whole reply, end of message included, over the tokens where the options differ; `tokens[i]` counts them.
 * A server that cannot score replies fails with an error whose message starts with `decision-unsupported`.
 */
export type DecisionRequest = { messages: unknown[]; options: string[]; adapters?: { id: string; scale: number }[] };
export type DecisionScores = { log_probs: number[]; tokens?: number[] };
export type DecisionScorer = (request: DecisionRequest, signal?: AbortSignal) => Promise<DecisionScores>;

/** Aggregate-only progress from a streaming provider turn. Contains no generated content. */
export type ModelStreamProgress = { status: 'progress' | 'completed' | 'failed';
  deltaEvents: number; deltaBytes: number; textDeltaEvents: number; textDeltaBytes: number;
  thinkingDeltaEvents: number; thinkingDeltaBytes: number; toolCallDeltaEvents: number;
  toolCallDeltaBytes: number; startedAt: string; observedAt: string; elapsedMs: number };
export type ModelStreamProgressSink = (progress: ModelStreamProgress) => void;
