/** Chat completions over HTTP: the shared adapter (chat-completion.ts) with the HTTP transport. */
import { chatCompletionModelTurn, fetchModel, httpChatTransport, limitedTransport, openAIEndpointRoot, promptLogprobDecider, requestLimit, type ChatExchange, type HttpChatOptions,
  type RequestLimit } from './chat-completion.js';
import type { ModelTurnRequest } from '../contracts.js';
export { fetchModel } from './chat-completion.js';

export type OpenAICompatibleExchange = ChatExchange;
export type OpenAICompatibleOptions = HttpChatOptions & {
  toolAliases?: Record<string, string>; request?: Record<string, unknown>;
  onExchange?: (exchange: OpenAICompatibleExchange) => void | Promise<void>;
  onRequestStart?: (request: ModelTurnRequest, retryIndex: number) => void | Promise<void>;
  /** Requests in flight at once, turns and decision scoring together: a number, or a limit shared with other drivers. */
  concurrency?: number | RequestLimit;
};

/**
 * Configurable OpenAI chat-completions transport; natlang policy stays in the runtime. Its `decide` scores finite
 * replies through `prompt_logprobs` (servers without them fail with `decision-unsupported`); its `contextWindow` asks
 * the server how many tokens the model's context holds (undefined when the server does not say).
 */
export function openAICompatibleModelTurn(options: OpenAICompatibleOptions) {
  const { toolAliases, request, onExchange, onRequestStart, concurrency, ...http } = options;
  const limit = typeof concurrency === 'number' ? requestLimit(concurrency) : concurrency;
  const transport = (settings: typeof http) => limit ? limitedTransport(httpChatTransport(settings), limit) : httpChatTransport(settings);
  let window: Promise<number | undefined> | undefined;
  return Object.assign(chatCompletionModelTurn(transport(http), { toolAliases, request, onExchange, onRequestStart }),
    { model: http.model, decide: promptLogprobDecider(transport({ ...http, stream: false }), { request }),
      contextWindow: () => window ??= serverContextWindow(http) });
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
