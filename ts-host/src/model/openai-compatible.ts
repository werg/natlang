/** Chat completions over HTTP: the shared adapter (chat-completion.ts) with the HTTP transport. */
import { chatCompletionModelTurn, httpChatTransport, promptLogprobDecider, type ChatExchange, type HttpChatOptions } from './chat-completion.js';
export { fetchModel } from './chat-completion.js';

export type OpenAICompatibleExchange = ChatExchange;
export type OpenAICompatibleOptions = HttpChatOptions & {
  toolAliases?: Record<string, string>; request?: Record<string, unknown>;
  onExchange?: (exchange: OpenAICompatibleExchange) => void | Promise<void>;
};

/**
 * Configurable OpenAI chat-completions transport; natlang policy stays in the runtime. Its `decide` scores finite
 * replies through `prompt_logprobs` (servers without them fail with `decision-unsupported`).
 */
export function openAICompatibleModelTurn(options: OpenAICompatibleOptions) {
  const { toolAliases, request, onExchange, ...http } = options;
  return Object.assign(chatCompletionModelTurn(httpChatTransport(http), { toolAliases, request, onExchange }),
    { decide: promptLogprobDecider(httpChatTransport({ ...http, stream: false }), { request }) });
}
