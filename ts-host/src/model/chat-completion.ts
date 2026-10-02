/**
 * The one chat-completions adapter between the runtime's model turns and an OpenAI-style chat endpoint, shared by
 * every host. A transport only moves a request body to a model and yields what comes back: streamed chunks, or one
 * whole response from an endpoint that does not stream. Everything else lives here: request building, tool-name
 * aliases, assembling streamed deltas into one response, decoding tool calls, truncation, the malformed-call retry,
 * and token accounting.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';

type Json = Record<string, unknown>;
/** Streamed chunks (`chat.completion.chunk`), or a single complete `chat.completion` body. */
export type ChatTransport = (body: Json, signal?: AbortSignal) => Promise<AsyncIterable<Json> | Json>;

export type ChatExchange = { request: ModelTurnRequest; wireRequest: Json; wireResponse: Json };
export type ChatTurnStats = { durationMs: number; promptTokens: number | null; completionTokens: number | null;
  cachedTokens: number | null; retries: number };
export type ChatCompletionOptions = {
  /** Extra request fields (sampling settings, template arguments). */
  request?: Json;
  /** Runtime tool name -> name the model is shown. */
  toolAliases?: Record<string, string>;
  /** Called with each request and its (assembled) response. */
  onExchange?: (exchange: ChatExchange) => void | Promise<void>;
  /** Called once per turn with its token counts and duration. */
  onTurn?: (stats: ChatTurnStats) => void;
};

/** Tool-call markup (Qwen-style XML or <tool_call> tags) left in a reply's text. */
export const TOOL_MARKUP = /<tool_call>|<\/tool_call>|<function=|<\/function>|<parameter=|<\/parameter>/;
class LeakedCall extends Error {}

/**
 * The one offered tool whose arguments a reply's whole text is: a JSON object (optionally in a code fence) whose keys
 * include every required parameter of exactly one tool and name no parameter it lacks. Some models write a call out
 * like this instead of calling; a reply that could be prose, or several tools' arguments, is left as text.
 */
export function callWrittenAsText(text: string, tools: Array<{ function: { name: string } & Json }>): [string, Json] | undefined {
  const body = text.trim().replace(/^```(?:json)?\s*\n([\s\S]*?)\n?```$/, '$1').trim();
  if (!body.startsWith('{')) return undefined;
  let value: unknown;
  try { value = JSON.parse(body); } catch { return undefined; }
  if (!value || typeof value !== 'object' || Array.isArray(value)) return undefined;
  const keys = Object.keys(value);
  const matches = tools.filter(tool => {
    const parameters = tool.function.parameters as { properties?: Json; required?: string[] } | undefined;
    const properties = Object.keys(parameters?.properties ?? {});
    return keys.length > 0 && keys.every(key => properties.includes(key)) &&
      (parameters?.required ?? []).every(key => keys.includes(key));
  });
  return matches.length === 1 ? [matches[0]!.function.name, value as Json] : undefined;
}

const MALFORMED_RETRY = 'The last tool call was malformed. Call one offered tool with valid JSON object arguments. ' +
  'Do not change the task or invent a new tool.';

function isStream(value: AsyncIterable<Json> | Json): value is AsyncIterable<Json> {
  return typeof (value as AsyncIterable<Json>)[Symbol.asyncIterator] === 'function';
}

/**
 * Fold streamed chunks into the `chat.completion` body a non-streaming request would have returned: content,
 * reasoning, and tool calls are concatenated per index; usage and timings come from the chunk that carries them.
 */
export async function assembleChatCompletion(chunks: AsyncIterable<Json>): Promise<Json> {
  let head: Json = {}, finish: unknown = null, content = '', reasoning = '', sawContent = false, sawReasoning = false;
  const calls: Array<{ id?: string; type: string; function: { name: string; arguments: string } }> = [];
  const extra: Json = {};
  for await (const chunk of chunks) {
    if (!Object.keys(head).length) head = { id: chunk.id, model: chunk.model, created: chunk.created };
    for (const key of ['usage', 'timings', 'system_fingerprint'] as const) if (chunk[key] != null) extra[key] = chunk[key];
    const choice = (chunk.choices as Json[] | undefined)?.[0];
    if (!choice) continue;
    if (choice.finish_reason != null) finish = choice.finish_reason;
    const delta = (choice.delta ?? {}) as Json;
    if (typeof delta.content === 'string') { content += delta.content; sawContent = true; }
    const reasoningDelta = delta.reasoning_content ?? delta.reasoning;
    if (typeof reasoningDelta === 'string') { reasoning += reasoningDelta; sawReasoning = true; }
    for (const part of (delta.tool_calls ?? []) as Json[]) {
      const index = typeof part.index === 'number' ? part.index : calls.length;
      const call = calls[index] ??= { type: 'function', function: { name: '', arguments: '' } };
      if (typeof part.id === 'string') call.id = part.id;
      if (typeof part.type === 'string') call.type = part.type;
      const fn = (part.function ?? {}) as Json;
      if (typeof fn.name === 'string') call.function.name += fn.name;
      if (typeof fn.arguments === 'string') call.function.arguments += fn.arguments;
    }
  }
  const message: Json = { role: 'assistant', content: sawContent ? content : null,
    ...(sawReasoning ? { reasoning_content: reasoning } : {}),
    ...(calls.length ? { tool_calls: calls.filter(Boolean) } : {}) };
  return { ...head, object: 'chat.completion', choices: [{ index: 0, message, finish_reason: finish }], ...extra };
}

/** Tool schemas as the model sees them: the runtime's private annotations (`x-natlang…`, `x-optional`) removed. */
export function publicSchema(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(publicSchema);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value)
    .filter(([key]) => !key.startsWith('x-natlang') && key !== 'x-optional')
    .map(([key, item]) => [key, publicSchema(item)]));
  return value;
}

/** Tool definitions checked and reduced to what the model sees. */
export function modelTools(rawTools: unknown[]): Array<{ type: 'function'; function: { name: string } & Json }> {
  return rawTools.map((raw, index) => {
    const tool = raw as { type?: unknown; function?: { name?: unknown } };
    if (tool?.type !== 'function' || typeof tool.function?.name !== 'string') throw new TypeError(`model tool ${index} is invalid`);
    return publicSchema(tool) as { type: 'function'; function: { name: string } & Json };
  });
}

function decodeArguments(value: unknown): Json {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as Json;
  if (typeof value !== 'string') throw new Error('tool arguments are neither JSON text nor an object');
  const parsed = JSON.parse(value);
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) throw new Error('tool arguments must be an object');
  return parsed as Json;
}

/** A model-turn driver over a chat transport. */
export function chatCompletionModelTurn(transport: ChatTransport, options: ChatCompletionOptions = {}) {
  const forward = options.toolAliases ?? {};
  const reverse = Object.fromEntries(Object.entries(forward).map(([source, target]) => [target, source]));
  return async (request: ModelTurnRequest, signal?: AbortSignal): Promise<ModelTurn> => {
    const tools = modelTools(request.tools);
    for (const tool of tools) tool.function.name = forward[tool.function.name] ?? tool.function.name;
    const messages = structuredClone(request.messages) as Array<Json & { tool_calls?: Array<{ function?: { name?: string } }> }>;
    for (const message of messages) for (const call of message.tool_calls ?? [])
      if (call.function?.name && forward[call.function.name]) call.function.name = forward[call.function.name];
    let attemptMessages: unknown[] = messages, retries = 0;
    let promptTokens = 0, completionTokens = 0, cachedTokens = 0;
    let hasPrompt = false, hasCompletion = false, hasCached = false;
    const started = Date.now();
    const counts = () => ({ ...(hasCompletion ? { completion_tokens: completionTokens } : {}),
      ...(hasPrompt ? { prompt_tokens: promptTokens } : {}) });
    const finish = (turn: ModelTurn): ModelTurn => {
      options.onTurn?.({ durationMs: Date.now() - started, promptTokens: hasPrompt ? promptTokens : null,
        completionTokens: hasCompletion ? completionTokens : null, cachedTokens: hasCached ? cachedTokens : null, retries });
      return turn;
    };
    while (true) {
      if (signal?.aborted) throw new Error('model turn aborted');
      const wireRequest: Json = { ...options.request, messages: attemptMessages, tools, tool_choice: request.tool_choice ?? 'auto' };
      if (request.temperature !== undefined) wireRequest.temperature = request.temperature;
      if (request.seed !== null) wireRequest.seed = request.seed;
      if (request.max_tokens !== null) wireRequest.max_tokens = request.max_tokens;
      const reply = await transport(wireRequest, signal);
      const body = isStream(reply) ? await assembleChatCompletion(reply) : reply;
      await options.onExchange?.({ request, wireRequest, wireResponse: body });
      const choice = (body.choices as Json[] | undefined)?.[0];
      const message = choice?.message as Json | undefined;
      const rawCalls = message?.tool_calls as Json[] | undefined;
      const usage = body.usage as Json | undefined;
      if (typeof usage?.prompt_tokens === 'number') { promptTokens += usage.prompt_tokens; hasPrompt = true; }
      if (typeof usage?.completion_tokens === 'number') { completionTokens += usage.completion_tokens; hasCompletion = true; }
      const cached = (usage?.prompt_tokens_details as Json | undefined)?.cached_tokens ?? (body.timings as Json | undefined)?.cache_n;
      if (typeof cached === 'number') { cachedTokens += cached; hasCached = true; }
      const truncated = choice?.finish_reason === 'length';
      try {
        const calls: [string, Json][] = (rawCalls ?? []).map((raw, index) => {
          const fn = raw.function as Json | undefined;
          const wireName = String(fn?.name ?? '');
          if (!wireName) throw new Error(`tool call ${index} has no function name`);
          return [reverse[wireName] ?? wireName, decodeArguments(fn?.arguments)];
        });
        // A tool call the server could not parse arrives as markup in the text: it is a malformed call, not a reply.
        if (!calls.length && TOOL_MARKUP.test(String(message?.content ?? ''))) throw new LeakedCall('the tool call came back as text');
        // A reply that is nothing but one offered tool's arguments, written out as JSON, is that call.
        const written = calls.length ? undefined : callWrittenAsText(String(message?.content ?? ''), tools);
        if (written) calls.push([reverse[written[0]] ?? written[0], written[1]]);
        const reasoning = message?.reasoning_content ?? message?.reasoning;
        return finish({ calls, text: String(message?.content ?? ''), raw_calls: rawCalls,
          ...(typeof reasoning === 'string' && reasoning.trim() ? { reasoning } : {}),
          ...(truncated ? { truncated: true } : {}), ...counts(), raw_response: body });
      } catch (error) {
        // A reply cut off mid tool call is a truncated turn, not a transport failure.
        if (truncated) return finish({ calls: [], text: String(message?.content ?? ''), truncated: true,
          ...counts(), raw_response: body });
        // A second leaked call ends the turn as an empty reply: its markup must never read as the answer.
        if (retries >= 1 && error instanceof LeakedCall) return finish({ calls: [], text: '', ...counts(), raw_response: body });
        if (retries >= 1 || signal?.aborted)
          throw new Error(`model returned malformed tool arguments after ${retries + 1} attempt${retries ? 's' : ''}: ` +
            `${error instanceof Error ? error.message : String(error)}`, { cause: error });
        retries++;
        attemptMessages = [...messages, { role: 'assistant', content: String(message?.content ?? '') },
          { role: 'user', content: MALFORMED_RETRY }];
      }
    }
  };
}

/**
 * Under Node, fetch (undici) fails a request whose response headers take more than five minutes, and a model server
 * with several slots can queue a request and then reason for minutes. Model requests there go through a dispatcher
 * without header or body timeouts; callers bound calls with their own deadlines. Elsewhere this is plain fetch.
 */
let longDispatcher: Promise<unknown> | undefined;
export async function fetchModel(url: string, init: RequestInit = {}): Promise<Response> {
  if (typeof process !== 'undefined' && process.versions?.node) {
    longDispatcher ??= import(/* @vite-ignore */ 'undici').then(({ Agent }) =>
      new Agent({ headersTimeout: 0, bodyTimeout: 0 }), () => undefined);
    const dispatcher = await longDispatcher;
    if (dispatcher) return fetch(url, { ...init, dispatcher } as RequestInit);
  }
  return fetch(url, init);
}

/** Server-sent events of a streaming chat completion, one parsed chunk per `data:` line. */
async function* serverSentChunks(body: ReadableStream<Uint8Array>): AsyncIterable<Json> {
  const decoder = new TextDecoder();
  let buffer = '';
  const reader = body.getReader();
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let newline: number;
      while ((newline = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, newline).replace(/\r$/, '');
        buffer = buffer.slice(newline + 1);
        if (!line.startsWith('data:')) continue;
        const data = line.slice(5).trim();
        if (data === '[DONE]') return;
        const chunk = JSON.parse(data) as Json;
        if (chunk.error) throw new Error(`model stream error: ${JSON.stringify(chunk.error).slice(0, 2000)}`);
        yield chunk;
      }
      if (done) return;
    }
  } finally { reader.releaseLock(); }
}

export type HttpChatOptions = { endpoint: string; model: string; apiKey?: string; headers?: Record<string, string>;
  /** Stream responses (default). Streaming sends headers at once, so no idle timeout on the way can fire. */
  stream?: boolean };

/** Chat completions over HTTP (`POST {endpoint}/v1/chat/completions`), in Node and in browsers. */
export function httpChatTransport(options: HttpChatOptions): ChatTransport {
  if (!options.endpoint || !options.model) throw new Error('model endpoint and ID are required');
  const url = options.endpoint.replace(/\/$/, '') + '/v1/chat/completions';
  const stream = options.stream ?? true;
  return async (body, signal) => {
    let response: Response;
    try {
      response = await fetchModel(url, { method: 'POST', signal, headers: { 'content-type': 'application/json',
        ...(options.apiKey ? { authorization: `Bearer ${options.apiKey}` } : {}), ...options.headers },
        body: JSON.stringify({ ...body, model: options.model,
          ...(stream ? { stream: true, stream_options: { include_usage: true } } : {}) }) });
    } catch (error) {
      const cause = error instanceof Error && error.cause instanceof Error ? error.cause : error;
      throw new Error(`model request to ${url} failed: ${cause instanceof Error ? `${cause.name}: ${cause.message}` : String(cause)}`,
        { cause: error });
    }
    if (!response.ok) {
      const text = await response.text();
      throw new Error(`model HTTP ${response.status}: ${text.slice(0, 2000)}`);
    }
    // A server that ignores `stream` answers with one JSON body.
    if (stream && response.body && /text\/event-stream/.test(response.headers.get('content-type') ?? ''))
      return serverSentChunks(response.body);
    return await response.json() as Json;
  };
}
