/**
 * Batched decision readouts (plans/BATCHED_EXECUTION.md §3.3): the default `scoreMany`, the coalescing wrapper that
 * routes decision readouts arriving together into one `scoreMany`, and the optional explicit-batch endpoint that
 * scores a prefix with K continuations in one request.
 */
import type { DecisionRequest, DecisionScorer, DecisionScores } from '../contracts.js';
import type { Scheduler } from './scheduler.js';
import { fetchModel } from './chat-completion.js';

type Settled = PromiseSettledResult<DecisionScores>;
type ScoreOne = (request: DecisionRequest, signal?: AbortSignal) => Promise<DecisionScores>;
type ScoreMany = (items: DecisionRequest[], signal?: AbortSignal) => Promise<Settled[]>;

/** The default `scoreMany`: every item is issued at once, so the scheduler and the server batch them. */
export function concurrentScoreMany(score: ScoreOne): ScoreMany {
  return (items, signal) => Promise.allSettled(items.map(item => score(item, signal)));
}

/**
 * Wrap a scorer so that decision readouts that arrive within `windowMs` (and the same tick) are scored by one
 * `scoreMany`. Callers keep the one-request interface, so `Promise.all` over decision-readout calls, list refinements
 * and classification fan-outs batch with no change in the program. Each caller's own failure and abort stay its own.
 */
export function coalescingScorer(base: DecisionScorer, options: { windowMs?: number; maxItems?: number } = {}): DecisionScorer {
  const windowMs = options.windowMs ?? 0, maxItems = options.maxItems ?? 256;
  const many: ScoreMany = base.scoreMany ?? concurrentScoreMany(base);
  type Pending = { item: DecisionRequest; signal?: AbortSignal; resolve: (value: DecisionScores) => void; reject: (reason: unknown) => void };
  let pending: Pending[] = [], timer: ReturnType<typeof setTimeout> | undefined;
  const flush = () => {
    if (timer !== undefined) { clearTimeout(timer); timer = undefined; }
    const batch = pending.filter(entry => !entry.signal?.aborted); pending = [];
    if (!batch.length) return;
    // The batch has no signal of its own: one caller's abort must not cancel the others. An aborted caller is
    // rejected at once (below) and its score is dropped.
    many(batch.map(entry => entry.item)).then(results => {
      batch.forEach((entry, index) => {
        const result = results[index];
        if (!result) entry.reject(new Error('scoreMany returned too few results'));
        else if (result.status === 'fulfilled') entry.resolve(result.value);
        else entry.reject(result.reason);
      });
    }, failure => batch.forEach(entry => entry.reject(failure)));
  };
  const scorer: ScoreOne = (request, signal) => {
    try { signal?.throwIfAborted(); } catch (error) { return Promise.reject(error); }
    return new Promise<DecisionScores>((resolve, reject) => {
      signal?.addEventListener('abort', () => reject(signal.reason), { once: true });
      pending.push({ item: request, signal, resolve, reject });
      if (pending.length >= maxItems) flush();
      else if (timer === undefined) timer = setTimeout(flush, windowMs);
    });
  };
  return Object.assign(scorer, { scoreMany: many });
}

/**
 * The explicit-batch scoring contract (optional endpoint, capability-gated; plans/BATCHED_EXECUTION.md step 2):
 *
 *   POST {url}  { items: [{ messages, continuations: string[], adapters? }], ...extra }
 *   200         { results: [{ log_probs: number[], tokens?: number[] } | { error: string }] }
 *
 * `continuations[k]` is the k-th option written as the closed final assistant message after `messages`. The server
 * prefills each distinct prefix once and scores the K continuations from it. `log_probs[k]` and `tokens[k]` mean what
 * they mean in `DecisionScores`: total log-probability over the tokens where the options differ, end of message
 * included. 404, 405 and 501 mean the server has no such endpoint; the caller then falls back to concurrent scoring.
 */
export type ExplicitBatchOptions = { url: string; headers?: Record<string, string>; extra?: Record<string, unknown>;
  scheduler?: Scheduler; maxItems?: number };
export class ScoreEndpointMissing extends Error {}

export async function postExplicitBatch(options: ExplicitBatchOptions, items: DecisionRequest[], signal?: AbortSignal): Promise<Settled[]> {
  const release = options.scheduler ? await options.scheduler.acquire({}, signal) : () => {};
  try {
    const response = await fetchModel(options.url, { method: 'POST', signal,
      headers: { 'content-type': 'application/json', ...options.headers },
      body: JSON.stringify({ ...options.extra, items: items.map(item => ({ messages: item.messages, continuations: item.options,
        ...(item.adapters?.length ? { adapters: item.adapters } : {}) })) }) });
    if ([404, 405, 501].includes(response.status)) throw new ScoreEndpointMissing(`no scoring endpoint at ${options.url}`);
    if (!response.ok) throw new Error(`batched scoring HTTP ${response.status}: ${(await response.text()).slice(0, 2000)}`);
    const body = await response.json() as { results?: ({ log_probs?: number[]; tokens?: number[]; error?: string } | null)[] };
    if (!Array.isArray(body.results) || body.results.length !== items.length)
      throw new Error('batched scoring returned a different number of results than items');
    return body.results.map((result, index): Settled => {
      if (!result || typeof result.error === 'string' || !Array.isArray(result.log_probs) ||
          result.log_probs.length !== items[index]!.options.length)
        return { status: 'rejected', reason: new Error(result?.error ?? 'batched scoring returned no scores for an item') };
      return { status: 'fulfilled', value: { log_probs: result.log_probs, ...(result.tokens ? { tokens: result.tokens } : {}) } };
    });
  } finally { release(); }
}

/**
 * `scoreMany` over an optional explicit-batch endpoint, falling back to `fallback` (concurrent items) when the server
 * does not have it. Once the endpoint is found missing it is not asked again.
 */
export function explicitBatchScoreMany(options: ExplicitBatchOptions, fallback: ScoreMany): ScoreMany {
  let missing = false;
  const maxItems = options.maxItems ?? 64;
  return async (items, signal) => {
    if (missing || items.length < 2) return fallback(items, signal);
    const out: Settled[] = [];
    for (let start = 0; start < items.length; start += maxItems) {
      const chunk = items.slice(start, start + maxItems);
      try { out.push(...await postExplicitBatch(options, chunk, signal)); }
      catch (error) {
        if (error instanceof ScoreEndpointMissing) {
          missing = true;
          out.push(...await fallback(items.slice(start), signal));
          return out;
        }
        if (signal?.aborted) throw error;
        out.push(...chunk.map((): Settled => ({ status: 'rejected', reason: error })));
      }
    }
    return out;
  };
}
