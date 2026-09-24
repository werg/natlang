import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export type OpenAICompatibleExchange = { request: ModelTurnRequest;
  wireRequest: Record<string, unknown>; wireResponse: Record<string, unknown> };
export type OpenAICompatibleOptions = {
  endpoint: string; model: string; apiKey?: string; headers?: Record<string, string>;
  toolAliases?: Record<string, string>; request?: Record<string, unknown>;
  onExchange?: (exchange: OpenAICompatibleExchange) => void | Promise<void>;
};

/**
 * Under Node, fetch (undici) fails a request whose response headers take more than five minutes. A model server
 * with several slots can queue a request behind others and then reason for minutes before its first byte, so model
 * requests go through a dispatcher without header or body timeouts; callers bound calls with their own deadlines.
 * Elsewhere this is plain fetch.
 */
let longDispatcher: Promise<unknown> | undefined;
export async function fetchModel(url: string, init: RequestInit = {}): Promise<Response> {
  if (typeof process !== 'undefined' && process.versions?.node) {
    longDispatcher ??= import('undici').then(({ Agent }) => new Agent({ headersTimeout: 0, bodyTimeout: 0 }), () => undefined);
    const dispatcher = await longDispatcher;
    if (dispatcher) return fetch(url, { ...init, dispatcher } as RequestInit);
  }
  return fetch(url, init);
}

function decode(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>;
  if (typeof value !== 'string') throw new Error('tool arguments are neither JSON text nor an object');
  const parsed = JSON.parse(value);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('tool arguments must be an object');
  return parsed as Record<string, unknown>;
}

/** Configurable OpenAI chat-completions transport; natlang policy stays in the runtime. */
export function openAICompatibleModelTurn(options: OpenAICompatibleOptions) {
  if (!options.endpoint || !options.model) throw new Error('model endpoint and ID are required');
  const forward = options.toolAliases ?? {};
  const reverse = Object.fromEntries(Object.entries(forward).map(([source, target]) => [target, source]));
  return async (request: ModelTurnRequest): Promise<ModelTurn> => {
    const tools = structuredClone(request.tools) as Array<{ function?: { name?: string } }>;
    for (const tool of tools) if (tool.function?.name && forward[tool.function.name])
      tool.function.name = forward[tool.function.name];
    const messages = structuredClone(request.messages) as Array<Record<string, unknown> &
      { tool_calls?: Array<{ function?: { name?: string } }> }>;
    for (const message of messages) for (const call of message.tool_calls ?? [])
      if (call.function?.name && forward[call.function.name]) call.function.name = forward[call.function.name];
    const originalMessages = messages;
    let attemptMessages = originalMessages, retries = 0;
    let promptTokens = 0, completionTokens = 0, hasPromptTokens = false, hasCompletionTokens = false;
    while (true) {
      const wireRequest: Record<string, unknown> = { ...options.request, model: options.model,
        messages: attemptMessages, tools, tool_choice: 'auto' };
      if (request.temperature !== undefined) wireRequest.temperature = request.temperature;
      if (request.seed !== null) wireRequest.seed = request.seed;
      if (request.max_tokens !== null) wireRequest.max_tokens = request.max_tokens;
      const url = options.endpoint.replace(/\/$/, '') + '/v1/chat/completions';
      let response: Response;
      try {
        response = await fetchModel(url, { method: 'POST', headers: { 'content-type': 'application/json',
          ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}), ...options.headers },
          body: JSON.stringify(wireRequest) });
      } catch (error) {
        const cause = error instanceof Error && error.cause instanceof Error ? error.cause : error;
        throw new Error(`model request to ${url} failed: ${cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause)}`, { cause: error });
      }
      const body = await response.json() as Record<string, unknown>;
      await options.onExchange?.({ request, wireRequest, wireResponse: body });
      if (!response.ok) throw new Error(`model HTTP ${response.status}: ${JSON.stringify(body).slice(0, 2000)}`);
      const choices = body.choices as Array<Record<string, unknown>> | undefined;
      const choice = choices?.[0];
      const message = choice?.message as Record<string, unknown> | undefined;
      const rawCalls = message?.tool_calls as Array<Record<string, unknown>> | undefined;
      const usage = body.usage as Record<string, unknown> | undefined;
      if (typeof usage?.prompt_tokens === 'number') { promptTokens += usage.prompt_tokens; hasPromptTokens = true; }
      if (typeof usage?.completion_tokens === 'number') { completionTokens += usage.completion_tokens; hasCompletionTokens = true; }
      try {
        const calls: [string, Record<string, unknown>][] = (rawCalls ?? []).map(raw => {
          const fn = raw.function as Record<string, unknown> | undefined;
          const wireName = String(fn?.name ?? '');
          return [reverse[wireName] ?? wireName, decode(fn?.arguments)];
        });
        return { calls, text: String(message?.content ?? ''), raw_calls: rawCalls,
          ...(choice?.finish_reason === 'length' ? { truncated: true } : {}),
          completion_tokens: hasCompletionTokens ? completionTokens : undefined,
          prompt_tokens: hasPromptTokens ? promptTokens : undefined,
          raw_response: body };
      } catch (error) {
        // A reply cut off mid tool call is a truncated turn, not a transport failure.
        if (choice?.finish_reason === 'length')
          return { calls: [], text: String(message?.content ?? ''), truncated: true, raw_response: body,
            completion_tokens: hasCompletionTokens ? completionTokens : undefined,
            prompt_tokens: hasPromptTokens ? promptTokens : undefined };
        if (retries >= 1)
          throw new Error(`model returned malformed tool arguments after ${retries + 1} attempt${retries ? 's' : ''}: ${error instanceof Error ? error.message : String(error)}`,
            { cause: error });
        retries++;
        attemptMessages = [...originalMessages,
          { role: 'assistant', content: String(message?.content ?? '') },
          { role: 'user', content: 'The last tool call was malformed. Call one offered tool with valid JSON object arguments. Do not change the task or invent a new tool.' }];
      }
    }
  };
}
