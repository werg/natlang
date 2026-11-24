/**
 * Scoring finite replies with a model's own log-probabilities, as pure functions over the request and response bodies.
 * natlang's OpenAI-compatible driver and every other host that talks to such a server (the Vibestudio model gateway)
 * build the same requests and read the same responses, so there is one implementation of both ends:
 *
 *   - per reply: the prompt plus the reply as the closed final assistant message, `prompt_logprobs: 0`, one token to
 *     generate (`promptLogprobRequest`); `scoresFromPromptLogprobs` sums each reply's token log-probabilities after the
 *     prefix all replies share;
 *   - the explicit-batch endpoint (`scoring.ts`): `batchScoreItem` writes an item, `scoresFromBatchResult` reads its result;
 *   - `open_reply` (a field-by-field step of a record decision) scores each reply as the start of the message: continued
 *     rather than closed on the chat-completions route, the item's `open_reply` on the batch endpoint, echoed back.
 */
import type { DecisionRequest, DecisionScorer, DecisionScores } from '../contracts.js';

type Json = Record<string, unknown>;

/** The softmax of log-probabilities: normalized probabilities in the same order. */
export function softmax(logProbs: readonly number[]): number[] {
  const max = Math.max(...logProbs);
  const weights = logProbs.map(value => Math.exp(value - max));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map(value => value / total);
}

/** The chat-completions body that evaluates `reply` as the closed final assistant message after `messages`, or with
 * `open` as the start of that message (continued, so its end is not part of the prompt); after `reasoning` when given
 * (the message's reasoning, both aliases as for history, chat-completion.ts). */
export function promptLogprobRequest(messages: readonly unknown[], reply: string, extra: Json = {}, open = false, reasoning?: string): Json {
  return { ...extra, messages: [...messages, { role: 'assistant', content: reply,
    ...(typeof reasoning === 'string' ? { reasoning, reasoning_content: reasoning } : {}) }],
    add_generation_prompt: false, continue_final_message: open, prompt_logprobs: 0, max_tokens: 1 };
}

/** What one reply's response is read for: the server's `prompt_logprobs` rows and the prompt tokens it evaluated. */
export type PromptLogprobResponse = { prompt_logprobs?: unknown; usage?: { prompt_tokens?: number } };

/**
 * The scores of replies from their responses (in reply order). Each response's rows map token ID to its entry; with
 * `prompt_logprobs: 0` the prompt's own token is the only (first) key. A reply's score is the sum of its tokens'
 * log-probabilities after the prefix all replies share. A server that returns no `prompt_logprobs` cannot score.
 */
export function scoresFromPromptLogprobs(responses: readonly PromptLogprobResponse[]): DecisionScores {
  const sequences = responses.map(body => {
    const rows = body.prompt_logprobs;
    if (!Array.isArray(rows)) throw new Error('decision-unsupported: the server returned no prompt_logprobs');
    const promptTokens = body.usage?.prompt_tokens;
    const tokens = rows.map(row => {
      if (!row || typeof row !== 'object') return null;
      const [id, entry] = Object.entries(row as Record<string, { logprob?: number }>)[0] ?? [];
      return id === undefined || typeof entry?.logprob !== 'number' ? null : { id, logprob: entry.logprob };
    });
    return { tokens, promptTokens: typeof promptTokens === 'number' ? promptTokens : undefined };
  });
  const rowsOf = sequences.map(item => item.tokens);
  // Every reply's request evaluates the whole prompt again, and the sum is what the readout used (unknown if any request does not say).
  const promptTokens = sequences.every(item => item.promptTokens !== undefined) ? sequences.reduce((sum, item) => sum + item.promptTokens!, 0) : undefined;
  let shared = 0;
  const shortest = Math.min(...rowsOf.map(sequence => sequence.length));
  while (shared < shortest && rowsOf.every(sequence => sequence[shared]?.id === rowsOf[0]![shared]?.id)) shared++;
  // With a single reply nothing differs; its score is the whole reply after the shared prompt.
  if (responses.length === 1) shared = 0;
  return { log_probs: rowsOf.map(sequence => sequence.slice(shared).reduce((sum, row) => sum + (row?.logprob ?? 0), 0)),
    tokens: rowsOf.map(sequence => sequence.length - shared),
    ...(promptTokens !== undefined ? { prompt_tokens: promptTokens } : {}) };
}

/**
 * Whether score items ask for both tokenizations of each option (`split_tokenization`): the engine sums the option's
 * canonical tokenization and its split one (the options' shared leading and trailing characters tokenized apart, so
 * `"spam"` is also `"`,`spam`,`"` where the canonical BPE writes `"sp`,`am"`), since the model may write either.
 * Default on; NATLANG_DECISION_TOKENIZATIONS=canonical scores the canonical tokenization alone. An engine that does
 * not serve the field scores the canonical one (its result has no `split_tokenization` echo).
 */
export function splitTokenization(): boolean {
  return (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NATLANG_DECISION_TOKENIZATIONS !== 'canonical';
}

/**
 * Whether a reasoned item carries its reasoning's sampled ids (`reasoning_ids`), which the engine scores in place of the
 * reasoning text re-encoded. Default on; NATLANG_DECISION_REASONING_IDS=rendered scores after the re-encoded text.
 */
export function sendReasoningIds(): boolean {
  return (globalThis as { process?: { env?: Record<string, string | undefined> } }).process?.env?.NATLANG_DECISION_REASONING_IDS !== 'rendered';
}

/** One item of the explicit-batch endpoint's request: the prompt and its replies as continuations. */
export function batchScoreItem(item: Pick<DecisionRequest, 'messages' | 'options'> & Partial<Pick<DecisionRequest, 'adapters' | 'native_state_modes' | 'native_recording' | 'cache_messages' | 'open_reply' | 'reasoning' | 'reasoning_ids'>>,
  extra?: Json): Json {
  return { ...extra, messages: item.messages, continuations: item.options,
    ...(item.cache_messages ? { cache_messages: item.cache_messages } : {}),
    ...(item.open_reply ? { open_reply: true } : {}),
    ...(item.options.length >= 2 && splitTokenization() ? { split_tokenization: true } : {}),
    ...(typeof item.reasoning === 'string' ? { reasoning: item.reasoning } : {}),
    ...(typeof item.reasoning === 'string' && item.reasoning && Array.isArray(item.reasoning_ids) && item.reasoning_ids.length
      && sendReasoningIds() ? { reasoning_ids: [...item.reasoning_ids] } : {}),
    ...(item.adapters?.length ? { adapters: item.adapters } : {}),
    ...(item.native_state_modes ? { native_state_modes: item.native_state_modes } : {}),
    ...(item.native_recording ? { native_recording: true } : {}) };
}

/** One result of the explicit-batch endpoint's response, for an item of `replyCount` replies. */
export type BatchScoreResult = { log_probs?: number[]; tokens?: number[]; prompt_tokens?: number; error?: string; open_reply?: boolean; reasoning?: boolean;
  native_decision_capture?: Record<string, unknown> } | null | undefined;

/** The scores of a batch result for `item`, or the failure it reports. An item scored as open replies must say so: a
 * server that ignored `open_reply` scored them closed, which is refused as `decision-open-unsupported`; likewise an
 * item with reasoning (`decision-reasoning-unsupported`: a server that ignored it scored the replies without it). */
export function scoresFromBatchResult(result: BatchScoreResult, item: Pick<DecisionRequest, 'options' | 'open_reply' | 'reasoning'>): DecisionScores {
  if (!result || typeof result.error === 'string' || !Array.isArray(result.log_probs) || result.log_probs.length !== item.options.length)
    throw new Error(result?.error ?? 'batched scoring returned no scores for an item');
  if (item.open_reply === true && result.open_reply !== true)
    throw new Error('decision-open-unsupported: the server scored the replies of a field-by-field decision as closed (open_reply not served)');
  if (typeof item.reasoning === 'string' && result.reasoning !== true)
    throw new Error('decision-reasoning-unsupported: the server scored the replies without the reasoning before them (reasoning not served)');
  return { log_probs: result.log_probs, ...(result.tokens ? { tokens: result.tokens } : {}),
    ...(Number.isSafeInteger(result.prompt_tokens) ? { prompt_tokens: result.prompt_tokens } : {}),
    ...(result.native_decision_capture ? { native_decision_capture: result.native_decision_capture } : {}) };
}

/** Score `items` together on `scorer` (its `scoreMany` when it batches, else concurrently), in order; the first item
 * that fails fails the whole (an auto-framed readout needs every score: decision-frame.ts `autoFramedScores`). */
export async function scoreAll(scorer: DecisionScorer, items: DecisionRequest[], signal?: AbortSignal): Promise<DecisionScores[]> {
  if (items.length === 1) return [await scorer(items[0]!, signal)];
  const settled = scorer.scoreMany ? await scorer.scoreMany(items, signal) : await Promise.allSettled(items.map(item => scorer(item, signal)));
  const failed = settled.find(result => result.status === 'rejected');
  if (failed) throw (failed as PromiseRejectedResult).reason;
  return settled.map(result => (result as PromiseFulfilledResult<DecisionScores>).value);
}
