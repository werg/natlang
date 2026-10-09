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
    /** Native value syntax in the model prefix; opaque placeholders remain strings on the wire. */
    value_type?: 'string' | 'unknown';
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
export type ModelContentPart = { type: 'text'; text: string } | { type: 'neuralese'; id: string; value_type?: 'string' | 'unknown' };
/** Calls are an ordered, non-atomic batch. Dependent calls belong in later turns. */
export type ModelTurn = { calls?: [string, Record<string, unknown>][]; text?: string;
  raw_calls?: unknown[]; completion_tokens?: number; prompt_tokens?: number;
  /** Exact rendered-provider-context and explicit text-channel emulation provenance, when an adapter supplies it. */
  transport_provenance?: Record<string, unknown>;
  value_confidence?: (number | { geometric_mean?: number } | null)[];
  raw_response?: Record<string, unknown>;
  /** The model's reasoning before this turn's reply, when the backend returns it. */
  reasoning?: string;
  /** A synthetic plan elicited in a separate required-tool turn before this reply. */
  /** null means planning was requested but the provider did not produce a valid plan. */
  execution_plan?: string | null;
  /** The reply stopped at max_tokens. */
  truncated?: boolean;
  /** How the model scheduler batched this turn's request (batch id and size, requests in flight, queue wait). */
  scheduling?: { batch_id: string; batch_size: number; in_flight: number; queue_wait_ms: number; priority: 'running' | 'new' } };

/**
 * A piece of a model turn while it streams (plans/STREAMING.md §1.1), in the order the model produced it:
 * - `text`, `reasoning`: the next characters of the reply or of the reasoning before it;
 * - `tool_call`: a fragment of call `index` — its `id` and `name` when they arrive (names already as the runtime
 *   knows them, tool aliases undone), and the next `arguments` text (JSON text, possibly with Neuralese parts);
 * - `neuralese`: a block the server wrote into the reply, once complete (`block`: its metadata, when sent);
 * - `reset`: the turn's request is sent again (a malformed-call retry, or a retry after restoring blocks); deltas so
 *   far belong to an abandoned attempt.
 * Deltas are hints for showing and preparing: the turn the driver returns stays authoritative (on a Neuralese server,
 * the final message it sends). A driver emits deltas only for replies it receives as a stream; a whole reply emits
 * none. They come from the one place streamed chunks are assembled (`assembleChatCompletion`), so the deltas of a
 * turn concatenate to the content of the turn assembled from them.
 */
export type ModelTurnDelta = { type: 'text'; text: string } | { type: 'reasoning'; text: string } |
  { type: 'tool_call'; index: number; id?: string; name?: string; arguments?: string } |
  { type: 'neuralese'; part: Extract<ModelContentPart, { type: 'neuralese' }>; block?: Record<string, unknown> } |
  { type: 'reset' };
/**
 * Per-call options of a model-turn driver, its optional third argument: `(request, signal?, options?)`. They are not
 * part of `ModelTurnRequest`, which is data (recorded, cloned, sent to workers). `onDelta` receives the turn's
 * deltas as they arrive; it must not throw (an observer never affects the turn) and is not awaited.
 */
export type ModelTurnOptions = { onDelta?: (delta: ModelTurnDelta) => void };


/**
 * Decision readout (single pass over the prompt): score a finite set of assistant replies after `messages`. A
 * driver that can do this carries a `decide` function. `log_probs[i]` is the total log-probability of `options[i]`
 * as the whole reply, end of message included, over the tokens where the options differ; `tokens[i]` counts them.
 * A server that cannot score replies fails with an error whose message starts with `decision-unsupported`.
 */
export type DecisionRequest = { messages: unknown[]; options: string[]; adapters?: { id: string; scale: number }[] };
export type DecisionScores = { log_probs: number[]; tokens?: number[];
  /** Set by a coalescing scorer: the batch this readout was scored in. */
  batch?: { batch_id: string; batch_size: number } };
export type DecisionScorer = ((request: DecisionRequest, signal?: AbortSignal) => Promise<DecisionScores>) & {
  /**
   * Score many requests at once, so the backend can batch them (plans/BATCHED_EXECUTION.md §3.3). One entry per item,
   * in order; an item that cannot be scored fails alone. Optional: use `scoreMany` from native/decision.ts, which
   * falls back to issuing the items concurrently.
   */
  scoreMany?: (items: DecisionRequest[], signal?: AbortSignal) => Promise<PromiseSettledResult<DecisionScores>[]> };

/** Aggregate-only progress from a streaming provider turn. Contains no generated content. */
export type ModelStreamProgress = { status: 'progress' | 'completed' | 'failed';
  deltaEvents: number; deltaBytes: number; textDeltaEvents: number; textDeltaBytes: number;
  thinkingDeltaEvents: number; thinkingDeltaBytes: number; toolCallDeltaEvents: number;
  toolCallDeltaBytes: number; startedAt: string; observedAt: string; elapsedMs: number };
export type ModelStreamProgressSink = (progress: ModelStreamProgress) => void;
