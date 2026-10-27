/** Chat completions over HTTP: the shared adapter (chat-completion.ts) with the HTTP transport. */
import { chatCompletionModelTurn, fetchModel, httpChatTransport, limitedTransport, openAIEndpointRoot, promptLogprobDecider, requestLimit, type ChatExchange, type HttpChatOptions,
  type RequestLimit } from './chat-completion.js';
import type { ModelTurnRequest } from '../contracts.js';
import type { Scheduler } from './scheduler.js';
import { coalescingScorer, concurrentScoreMany, explicitBatchScoreMany } from './scoring.js';
import { nativeStateSentinelIds, runtimeTextToParts } from '../native/neuralese.js';
export { fetchModel } from './chat-completion.js';

export type OpenAICompatibleExchange = ChatExchange;
export type OpenAICompatibleOptions = HttpChatOptions & {
  toolAliases?: Record<string, string>; request?: Record<string, unknown>;
  onExchange?: (exchange: OpenAICompatibleExchange) => void | Promise<void>;
  onRequestStart?: (request: ModelTurnRequest, retryIndex: number) => void | Promise<void>;
  /**
   * Requests in flight at once, turns and decision scoring together: a number, a limit shared with other drivers, or
   * a model scheduler (model/scheduler.ts), which also orders the queue and records each request's batch.
   */
  concurrency?: number | RequestLimit | Scheduler;
  /** Optional explicit-batch scoring endpoint (see model/scoring.ts): a URL, or true for `{endpoint}/v1/natlang/score`. */
  scoreEndpoint?: string | boolean;
  /** Decision readouts arriving within this many ms are scored together (default 0: the same tick). */
  scoreCoalesceMs?: number;
  /**
   * The score endpoint also scores single readouts and carries each item's tuned request fields (the llama.cpp fork,
   * whose chat completions have no prompt_logprobs and whose score items take `lora`).
   */
  scoreAll?: boolean;
  /**
   * Extra request fields selecting the current call's tuned weights (llama.cpp's per-request `lora` scales). Read in
   * the calling invocation's context, before any queueing or coalescing, so each request carries its own stack.
   */
  tunedRequest?: () => Record<string, unknown> | undefined;
  /**
   * Native state spans the server has loaded by ID (the llama.cpp fork's `--native-state`, e.g. tuned soft
   * instructions). Message text holding a state's sentinel is sent as content parts with `{type: "native_state", id}`;
   * a sentinel naming a state the server has not loaded is `weights-unavailable`.
   */
  nativeStates?: () => ReadonlySet<string>;
};

/** Messages whose text carries native state sentinels, as content-part arrays the llama.cpp fork consumes exactly. */
export function nativeStateMessages(messages: readonly unknown[], loaded: ReadonlySet<string>): unknown[] {
  return messages.map(message => {
    const content = (message as { content?: unknown })?.content;
    if (typeof content !== 'string' || !nativeStateSentinelIds(content).length) return message;
    return { ...(message as object), content: runtimeTextToParts(content).map(part => {
      if (part.type === 'text') return { type: 'text', text: part.text };
      if (!loaded.has(part.id)) throw new Error(`weights-unavailable: native state ${part.id} is not loaded by the model server; rerun natlang tune or restart`);
      return { type: 'native_state', id: part.id };
    }) };
  });
}
const TUNED: unique symbol = Symbol('natlang.tuned-request');
const isScheduler = (value: unknown): value is Scheduler => typeof (value as Scheduler | undefined)?.transport === 'function';

/**
 * Configurable OpenAI chat-completions transport; natlang policy stays in the runtime. Its `decide` scores finite
 * replies through `prompt_logprobs` (servers without them fail with `decision-unsupported`); its `contextWindow` asks
 * the server how many tokens the model's context holds (undefined when the server does not say).
 */
export function openAICompatibleModelTurn(options: OpenAICompatibleOptions) {
  const { toolAliases, request, onExchange, onRequestStart, concurrency, scoreEndpoint, scoreCoalesceMs, scoreAll, tunedRequest, nativeStates, ...http } = options;
  // A bare number keeps its plain FIFO limit; a scheduler adds priority, adjacency and batch records.
  const scheduler = isScheduler(concurrency) ? concurrency : undefined;
  const limit: RequestLimit | undefined = typeof concurrency === 'number' ? requestLimit(concurrency) : scheduler ? undefined : concurrency as RequestLimit | undefined;
  const transport = (settings: typeof http) => {
    const plain = httpChatTransport(settings);
    return scheduler ? scheduler.transport(plain) : limit ? limitedTransport(plain, limit) : plain;
  };
  let window: Promise<number | undefined> | undefined;
  // Decision readouts that arrive together are scored together: by the explicit-batch endpoint where the backend has
  // declared one, otherwise as concurrent requests that the scheduler and the server batch.
  type Tuned = { [TUNED]?: Record<string, unknown> };
  const single = promptLogprobDecider(transport({ ...http, stream: false }), { request,
    ...(tunedRequest ? { extraFor: (item: object) => (item as Tuned)[TUNED] } : {}) });
  const url = scoreEndpoint === true ? openAIEndpointRoot(http.endpoint) + '/v1/natlang/score' : scoreEndpoint || undefined;
  const concurrent = concurrentScoreMany(single);
  const explicit = url ? explicitBatchScoreMany({ url, headers: { ...(http.apiKey ? { authorization: `Bearer ${http.apiKey}` } : {}), ...http.headers },
    extra: { model: http.model, ...(request ? { request } : {}) }, scheduler,
    ...(scoreAll ? { minItems: 1, itemExtra: (item: object) => (item as Tuned)[TUNED] } : {}) }, concurrent) : concurrent;
  // Without per-item fields on the explicit-batch endpoint, tuned items are scored as individual requests.
  const many: typeof explicit = items => !scoreAll && items.some(item => (item as Tuned)[TUNED]) ? concurrent(items) : explicit(items);
  const coalesced = coalescingScorer(Object.assign((...args: Parameters<typeof single>) => single(...args), { scoreMany: many }),
    { windowMs: scoreCoalesceMs ?? 0 });
  const withStates = <T extends { messages?: unknown }>(item: T): T => {
    const loaded = nativeStates?.();
    return Array.isArray(item.messages) ? { ...item, messages: nativeStateMessages(item.messages, loaded ?? new Set()) } : item;
  };
  const prepareDecision = <T extends { messages?: unknown }>(item: T): T => {
    const tuned = tunedRequest?.();
    const states = withStates(item);
    return tuned ? Object.assign({ ...states }, { [TUNED]: tuned }) : states;
  };
  const decide: typeof coalesced = Object.assign(
    (item: Parameters<typeof coalesced>[0], signal?: AbortSignal) => coalesced(prepareDecision(item), signal),
    { scoreMany: (items: Parameters<NonNullable<typeof coalesced.scoreMany>>[0], signal?: AbortSignal) =>
      coalesced.scoreMany!(items.map(prepareDecision), signal) });
  const turns = transport(http);
  const tunedTurns: typeof turns = (body, signal, meta) => {
    const tuned = tunedRequest?.();
    const states = withStates(body as { messages?: unknown }) as typeof body;
    return turns(tuned ? { ...states, ...tuned } : states, signal, meta);
  };
  return Object.assign(chatCompletionModelTurn(tunedTurns, { toolAliases, request, onExchange, onRequestStart }),
    { model: http.model, decide, contextWindow: () => window ??= serverContextWindow(http) });
}

const tokens = (value: unknown) => typeof value === 'number' && Number.isInteger(value) && value > 0 ? value : undefined;

/** The model's context length as the server reports it: vLLM's max_model_len, llama.cpp's n_ctx, or a context_length field. */
export async function serverContextWindow(http: HttpChatOptions): Promise<number | undefined> {
  if (!http.endpoint) return undefined;
  const base = openAIEndpointRoot(http.endpoint);
  const get = async (path: string): Promise<Record<string, unknown> | undefined> => {
    try {
      const response = await fetchModel(base + path, { signal: AbortSignal.timeout(5000),
        headers: { ...(http.apiKey ? { authorization: `Bearer ${http.apiKey}` } : {}), ...http.headers } });
      return response.ok ? await response.json() as Record<string, unknown> : undefined;
    } catch { return undefined; }
  };
  const listed = (await get('/v1/models'))?.data;
  const models = Array.isArray(listed) ? listed as Record<string, unknown>[] : [];
  const model = models.find(item => item.id === http.model) ?? (models.length === 1 ? models[0] : undefined);
  const meta = model?.meta as Record<string, unknown> | undefined;
  const reported = tokens(model?.max_model_len) ?? tokens(model?.context_length) ?? tokens(model?.context_window);
  if (reported) return reported;
  const settings = (await get('/props'))?.default_generation_settings as Record<string, unknown> | undefined;
  return tokens(settings?.n_ctx) ?? tokens(meta?.n_ctx_train);
}
