/**
 * Loopback-only OpenAI Chat Completions envelope for the official OpenCode
 * prompt-directed JSON text action backend. Completion generation is non-incremental;
 * stream=true receives a buffered JSON completion, which the repository's
 * Chat Completions transport explicitly accepts. This is not native provider
 * tool-call or incremental streaming fidelity.
 */
import { createServer } from 'node:http';
import { randomUUID } from 'node:crypto';
import { createOpenCodeStructuredTurnBackend } from './opencode-structured-turn.mjs';

const BRIDGE_ID = 'natlang-opencode-loopback-chat-adapter/3';
const DEFAULTS = Object.freeze({ host: '127.0.0.1', port: 0, maxConcurrency: 1,
  maxBodyBytes: 2 * 1024 * 1024, maxRequestMs: 180_000, cleanupTimeoutMs: 2_000 });

function jsonResponse(response, status, body) {
  if (response.destroyed || response.writableEnded) return;
  response.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  response.end(JSON.stringify(body));
}

function errorBody(message, code, metadata = {}) {
  return { error: { message, type: 'server_error', code, ...metadata } };
}

async function readJson(request, maxBodyBytes) {
  let size = 0;
  const chunks = [];
  for await (const chunk of request) {
    size += chunk.length;
    if (size > maxBodyBytes) {
      request.resume();
      const error = new Error(`request body exceeds ${maxBodyBytes} bytes`);
      error.code = 'BODY_TOO_LARGE';
      throw error;
    }
    chunks.push(chunk);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); }
  catch (cause) { throw new TypeError('request body must be valid JSON', { cause }); }
}

function validateRequest(body, modelAlias) {
  if (!body || typeof body !== 'object' || Array.isArray(body)) throw new TypeError('request JSON must be an object');
  if (body.model !== undefined && body.model !== modelAlias)
    throw Object.assign(new Error('request model does not match immutable adapter configuration'), { code: 'MODEL_MISMATCH' });
  if (!Array.isArray(body.messages) || !Array.isArray(body.tools))
    throw new TypeError('request must include messages and tools arrays');
  if (body.tool_choice !== undefined && !['auto', 'required'].includes(body.tool_choice) &&
      !(body.tool_choice && typeof body.tool_choice === 'object'))
    throw new TypeError('tool_choice must be auto, required, or a named-tool object');
  if (body.tool_choice && typeof body.tool_choice === 'object')
    throw Object.assign(new Error('named-tool tool_choice is unsupported by the structured-action bridge'),
      { code: 'UNSUPPORTED_TOOL_CHOICE' });
  if (body.stream !== undefined && typeof body.stream !== 'boolean')
    throw new TypeError('stream must be a boolean');
  if (body.prompt_logprobs !== undefined || body.logprobs !== undefined || body.logit_bias !== undefined)
    throw Object.assign(new Error('decision scoring/logprob controls are unsupported by the OpenCode structured-action bridge'),
      { code: 'UNSUPPORTED_SCORING' });
  if (body.n !== undefined && body.n !== 1)
    throw Object.assign(new Error('the OpenCode structured-action bridge supports one completion only'), { code: 'UNSUPPORTED_N' });
  return {
    messages: body.messages,
    tools: body.tools,
    tool_choice: body.tool_choice === 'required' || (body.tool_choice && typeof body.tool_choice === 'object') ? 'required' : 'auto',
    ...(body.temperature === undefined ? {} : { temperature: body.temperature }),
    ...(body.seed === undefined ? {} : { seed: body.seed }),
    ...(body.max_tokens === undefined ? {} : { max_tokens: body.max_tokens })
  };
}

function makeCompletion(turn, modelAlias, streamRequested) {
  const id = `chatcmpl-${randomUUID()}`;
  const created = Math.floor(Date.now() / 1000);
  const calls = (turn.calls ?? []).map(([name, args]) => ({
    id: `call_${randomUUID()}`,
    type: 'function',
    function: { name, arguments: JSON.stringify(args) }
  }));
  const usage = Number.isFinite(turn.prompt_tokens) && Number.isFinite(turn.completion_tokens) ? {
    prompt_tokens: turn.prompt_tokens,
    completion_tokens: turn.completion_tokens,
    total_tokens: turn.prompt_tokens + turn.completion_tokens
  } : undefined;
  const bridge = {
    id: BRIDGE_ID,
    transport: 'OpenCode SDK session.prompt with prompt-directed JSON text',
    native_provider_tool_calls: false,
    native_opencode_tool_policy: 'official SDK tool.ids inventory plus session.prompt wildcard deny; only pure invalid rejection is enabled',
    incremental_token_streaming: false,
    stream_requested: streamRequested,
    stream_honored: false,
    response_delivery: 'buffered JSON completion',
    call_ids: 'loopback adapter generated correlation IDs; not provider returned IDs',
    generation_controls: turn.raw_response?.generation_controls ?? {}
  };
  return {
    id, object: 'chat.completion', created, model: modelAlias,
    choices: [{ index: 0, message: { role: 'assistant', content: turn.text ?? null,
      ...(calls.length ? { tool_calls: calls } : {}) }, finish_reason: calls.length ? 'tool_calls' : 'stop' }],
    ...(usage ? { usage } : {}),
    raw_response: turn.raw_response,
    x_natlang_bridge: bridge
  };
}

function statusFor(error) {
  if (error?.code === 'BODY_TOO_LARGE') return 413;
  if (error?.code === 'UNSUPPORTED_SCORING' || error?.code === 'UNSUPPORTED_N') return 501;
  if (error?.code === 'UNSUPPORTED_TOOL_CHOICE') return 501;
  if (error?.code === 'MODEL_MISMATCH') return 409;
  if (error?.code === 'OVERLOADED') return 429;
  if (error?.code === 'REQUEST_TIMEOUT') return 504;
  if (error?.code === 'OPENCODE_PROVIDER_ERROR' && [429, 503].includes(error.providerStatusCode))
    return error.providerStatusCode;
  if (error instanceof TypeError) return 400;
  return 502;
}

/** Start a bounded, loopback-only HTTP adapter; no credential or mutable provider fields enter its public config. */
export async function createOpenCodeLoopbackChatAdapter(options = {}) {
  const host = options.host ?? DEFAULTS.host;
  if (host !== '127.0.0.1' && host !== '::1') throw new TypeError('OpenCode Chat adapter must bind to an IP loopback address');
  const port = options.port ?? DEFAULTS.port;
  const maxConcurrency = options.maxConcurrency ?? DEFAULTS.maxConcurrency;
  const maxBodyBytes = options.maxBodyBytes ?? DEFAULTS.maxBodyBytes;
  const maxRequestMs = options.maxRequestMs ?? DEFAULTS.maxRequestMs;
  const cleanupTimeoutMs = options.cleanupTimeoutMs ?? DEFAULTS.cleanupTimeoutMs;
  for (const [name, value, min] of [['port', port, 0], ['maxConcurrency', maxConcurrency, 1],
    ['maxBodyBytes', maxBodyBytes, 1024], ['maxRequestMs', maxRequestMs, 1], ['cleanupTimeoutMs', cleanupTimeoutMs, 1]]) {
    if (!Number.isSafeInteger(value) || value < min) throw new RangeError(`${name} is outside its supported range`);
  }
  if (!options.client) throw new TypeError('an official OpenCode SDK client is required');
  if (typeof options.providerID !== 'string' || !options.providerID || typeof options.modelID !== 'string' || !options.modelID)
    throw new TypeError('providerID and modelID are required');
  if (typeof options.directory !== 'string' || !options.directory.startsWith('/'))
    throw new TypeError('an absolute isolated OpenCode scratch directory is required');
  const modelAlias = options.modelAlias ?? `${options.providerID}/${options.modelID}`;
  const backend = createOpenCodeStructuredTurnBackend({ client: options.client, providerID: options.providerID,
    modelID: options.modelID, agent: options.agent, directory: options.directory, cleanupTimeoutMs });
  let active = 0, closing = false;
  const activeControllers = new Set();
  const server = createServer(async (request, response) => {
    if (request.method === 'GET' && request.url === '/health') {
      jsonResponse(response, 200, { ok: true, adapter: BRIDGE_ID, providerChecked: false });
      return;
    }
    if (request.method !== 'POST' || request.url !== '/v1/chat/completions') {
      jsonResponse(response, 404, errorBody('route not found', 'not_found'));
      return;
    }
    if (closing) { jsonResponse(response, 503, errorBody('adapter is closing', 'closing')); return; }
    if (active >= maxConcurrency) {
      jsonResponse(response, 429, errorBody('adapter concurrency limit reached', 'overloaded'));
      return;
    }
    active++;
    const abortController = new AbortController();
    activeControllers.add(abortController);
    const cancel = () => { if (!abortController.signal.aborted) abortController.abort(new Error('client disconnected')); };
    request.once('aborted', cancel);
    response.once('close', () => { if (!response.writableEnded) cancel(); });
    const timeout = setTimeout(() => {
      if (!abortController.signal.aborted)
        abortController.abort(Object.assign(new Error('OpenCode turn timed out'), { code: 'REQUEST_TIMEOUT' }));
    }, maxRequestMs);
    timeout.unref?.();
    try {
      const body = await readJson(request, maxBodyBytes);
      const modelRequest = validateRequest(body, modelAlias);
      const turn = await backend(modelRequest, abortController.signal);
      if (abortController.signal.aborted) throw abortController.signal.reason;
      const completion = makeCompletion(turn, modelAlias, body.stream === true);
      jsonResponse(response, 200, completion);
    } catch (error) {
      if (!response.destroyed && !response.writableEnded) {
        const code = error?.code === 'REQUEST_TIMEOUT' ? 'request_timeout' :
          error?.code === 'UNSUPPORTED_SCORING' ? 'decision_unsupported' :
            error?.code === 'UNSUPPORTED_TOOL_CHOICE' ? 'unsupported_tool_choice' :
            error?.code === 'OVERLOADED' ? 'overloaded' :
              error?.code === 'OPENCODE_PROVIDER_ERROR' ? 'provider_error' : 'opencode_bridge_error';
        const providerError = error?.code === 'OPENCODE_PROVIDER_ERROR' ? {
          provider_status_code: Number.isSafeInteger(error.providerStatusCode) ? error.providerStatusCode : null,
          provider_retryable: typeof error.providerRetryable === 'boolean' ? error.providerRetryable : null
        } : {};
        const diagnostics = error?.transportDiagnostic && typeof error.transportDiagnostic === 'object' ?
          { transport_diagnostic: error.transportDiagnostic } : {};
        jsonResponse(response, statusFor(error), errorBody(error instanceof Error ? error.message : String(error), code,
          { ...providerError, ...diagnostics }));
      }
    } finally {
      clearTimeout(timeout);
      request.removeListener('aborted', cancel);
      activeControllers.delete(abortController);
      active--;
    }
  });
  server.requestTimeout = maxRequestMs;
  server.headersTimeout = Math.min(maxRequestMs, 60_000);

  await new Promise((resolve, reject) => {
    const onError = error => { server.removeListener('listening', onListening); reject(error); };
    const onListening = () => { server.removeListener('error', onError); resolve(); };
    server.once('error', onError);
    server.once('listening', onListening);
    server.listen(port, host);
  });
  const address = server.address();
  if (!address || typeof address === 'string') throw new Error('loopback server did not expose a TCP address');
  const actualHost = host === '::1' ? '[::1]' : host;
  const config = Object.freeze({ bridge: BRIDGE_ID, host, port: address.port, providerID: options.providerID,
    modelID: options.modelID, modelAlias, agent: options.agent ?? null, maxConcurrency, maxBodyBytes, maxRequestMs,
    nativeOpenCodeToolPolicy: 'official SDK tool.ids inventory plus session.prompt wildcard deny; only pure invalid rejection is enabled',
    cleanupTimeoutMs, responseMode: 'buffered JSON completion even when stream=true; no incremental generation',
    scoring: 'unsupported', providerAvailability: 'not-probed' });
  let closed = false;
  const close = async () => {
    if (closed) return;
    closed = true;
    closing = true;
    for (const controller of activeControllers)
      if (!controller.signal.aborted) controller.abort(new Error('adapter closing'));
    await new Promise(resolve => {
      const timer = setTimeout(() => { server.closeAllConnections(); resolve(); }, cleanupTimeoutMs);
      timer.unref?.();
      server.close(() => { clearTimeout(timer); resolve(); });
    });
  };
  return Object.freeze({ url: `http://${actualHost}:${address.port}`, config, close });
}

export const openCodeLoopbackBridgeId = BRIDGE_ID;
