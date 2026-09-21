import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

export type OpenAICompatibleExchange = { request: ModelTurnRequest;
  wireRequest: Record<string, unknown>; wireResponse: Record<string, unknown> };
export type OpenAICompatibleOptions = {
  endpoint: string; model: string; apiKey?: string; headers?: Record<string, string>;
  toolAliases?: Record<string, string>; request?: Record<string, unknown>;
  onExchange?: (exchange: OpenAICompatibleExchange) => void | Promise<void>;
};

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
    const messages = structuredClone(request.messages) as Array<{ tool_calls?: Array<{ function?: { name?: string } }> }>;
    for (const message of messages) for (const call of message.tool_calls ?? [])
      if (call.function?.name && forward[call.function.name]) call.function.name = forward[call.function.name];
    const wireRequest: Record<string, unknown> = { ...options.request, model: options.model,
      messages, tools, tool_choice: 'auto', temperature: request.temperature };
    if (request.seed !== null) wireRequest.seed = request.seed;
    if (request.max_tokens !== null) wireRequest.max_tokens = request.max_tokens;
    const response = await fetch(options.endpoint.replace(/\/$/, '') + '/v1/chat/completions', {
      method: 'POST', headers: { 'content-type': 'application/json',
        ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}), ...options.headers },
      body: JSON.stringify(wireRequest),
    });
    const body = await response.json() as Record<string, unknown>;
    await options.onExchange?.({ request, wireRequest, wireResponse: body });
    if (!response.ok) throw new Error(`model HTTP ${response.status}: ${JSON.stringify(body).slice(0, 2000)}`);
    const choices = body.choices as Array<Record<string, unknown>> | undefined;
    const message = choices?.[0]?.message as Record<string, unknown> | undefined;
    const rawCalls = message?.tool_calls as Array<Record<string, unknown>> | undefined;
    const calls: [string, Record<string, unknown>][] = (rawCalls ?? []).map(raw => {
      const fn = raw.function as Record<string, unknown> | undefined;
      const wireName = String(fn?.name ?? '');
      return [reverse[wireName] ?? wireName, decode(fn?.arguments)];
    });
    const usage = body.usage as Record<string, unknown> | undefined;
    return { calls, text: String(message?.content ?? ''), raw_calls: rawCalls,
      completion_tokens: typeof usage?.completion_tokens === 'number' ? usage.completion_tokens : undefined,
      prompt_tokens: typeof usage?.prompt_tokens === 'number' ? usage.prompt_tokens : undefined,
      raw_response: body };
  };
}
