/**
 * Scoring finite replies with a model's own log-probabilities, as pure functions over the request and response bodies.
 * natlang's OpenAI-compatible driver and every other host that talks to such a server (the Vibestudio model gateway)
 * build the same requests and read the same responses, so there is one implementation of both ends:
 *
 *   - per reply: the prompt plus the reply as the closed final assistant message, `prompt_logprobs: 0`, one token to
 *     generate (`promptLogprobRequest`); `scoresFromPromptLogprobs` sums each reply's token log-probabilities after the
 *     prefix all replies share;
 *   - the explicit-batch endpoint (`scoring.ts`): `batchScoreItem` writes an item, `scoresFromBatchResult` reads its result.
 */
import type { DecisionRequest, DecisionScores } from '../contracts.js';

type Json = Record<string, unknown>;

/** The softmax of log-probabilities: normalized probabilities in the same order. */
export function softmax(logProbs: readonly number[]): number[] {
  const max = Math.max(...logProbs);
  const weights = logProbs.map(value => Math.exp(value - max));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map(value => value / total);
}

/** The chat-completions body that evaluates `reply` as the closed final assistant message after `messages`. */
export function promptLogprobRequest(messages: readonly unknown[], reply: string, extra: Json = {}): Json {
  return { ...extra, messages: [...messages, { role: 'assistant', content: reply }],
    add_generation_prompt: false, continue_final_message: false, prompt_logprobs: 0, max_tokens: 1 };
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

/** One item of the explicit-batch endpoint's request: the prompt and its replies as continuations. */
export function batchScoreItem(item: Pick<DecisionRequest, 'messages' | 'options'> & Partial<Pick<DecisionRequest, 'adapters' | 'native_state_modes' | 'native_recording'>>,
  extra?: Json): Json {
  return { ...extra, messages: item.messages, continuations: item.options,
    ...(item.adapters?.length ? { adapters: item.adapters } : {}),
    ...(item.native_state_modes ? { native_state_modes: item.native_state_modes } : {}),
    ...(item.native_recording ? { native_recording: true } : {}) };
}

/** One result of the explicit-batch endpoint's response, for an item of `replyCount` replies. */
export type BatchScoreResult = { log_probs?: number[]; tokens?: number[]; prompt_tokens?: number; error?: string; native_decision_capture?: Record<string, unknown> } | null | undefined;

/** The scores of a batch result, or the failure it reports. */
export function scoresFromBatchResult(result: BatchScoreResult, replyCount: number): DecisionScores {
  if (!result || typeof result.error === 'string' || !Array.isArray(result.log_probs) || result.log_probs.length !== replyCount)
    throw new Error(result?.error ?? 'batched scoring returned no scores for an item');
  return { log_probs: result.log_probs, ...(result.tokens ? { tokens: result.tokens } : {}),
    ...(Number.isSafeInteger(result.prompt_tokens) ? { prompt_tokens: result.prompt_tokens } : {}),
    ...(result.native_decision_capture ? { native_decision_capture: result.native_decision_capture } : {}) };
}
