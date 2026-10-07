/** Chat completions over HTTP: the shared adapter (chat-completion.ts) with the HTTP transport. */
import { chatCompletionModelTurn, httpChatTransport, limitedTransport, promptLogprobDecider, requestLimit, type ChatExchange, type HttpChatOptions,
  type RequestLimit } from './chat-completion.js';
export { fetchModel } from './chat-completion.js';

export type OpenAICompatibleExchange = ChatExchange;
export type OpenAICompatibleOptions = HttpChatOptions & {
  toolAliases?: Record<string, string>; request?: Record<string, unknown>;
  onExchange?: (exchange: OpenAICompatibleExchange) => void | Promise<void>;
  /** Requests in flight at once, turns and decision scoring together: a number, or a limit shared with other drivers. */
  concurrency?: number | RequestLimit;
};

/**
 * Configurable OpenAI chat-completions transport; natlang policy stays in the runtime. Its `decide` scores finite
 * replies through `prompt_logprobs` (servers without them fail with `decision-unsupported`).
 */
export function openAICompatibleModelTurn(options: OpenAICompatibleOptions) {
  const { toolAliases, request, onExchange, concurrency, ...http } = options;
  const limit = typeof concurrency === 'number' ? requestLimit(concurrency) : concurrency;
  const transport = (settings: typeof http) => limit ? limitedTransport(httpChatTransport(settings), limit) : httpChatTransport(settings);
  return Object.assign(chatCompletionModelTurn(transport(http), { toolAliases, request, onExchange }),
    { decide: promptLogprobDecider(transport({ ...http, stream: false }), { request }) });
}
