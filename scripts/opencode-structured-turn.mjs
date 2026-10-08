/**
 * An experimental OpenCode SDK adapter for Natlang's ModelTurn contract.
 *
 * OpenCode owns the provider request and may run its own agent loop. This
 * adapter asks for a JSON text action envelope, validates it locally, and maps
 * it to Natlang calls. It does not claim provider-enforced JSON Schema or
 * native provider tool-call fidelity.
 * Natlang remains responsible for executing those mapped calls.
 */

import { createHash, randomUUID } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const BRIDGE_ID = 'opencode-session-prompt-json-text-action-bridge/6';
const FAILURE_DIAGNOSTIC_VERSION = 'natlang.opencode_transport_failure/1';
const FAILURE_TEXT_PREVIEW_BYTES = 16 * 1024;
const FAILURE_INLINE_TEXT_PREVIEW_BYTES = 256;
const FAILURE_PART_PREVIEW_COUNT = 4;

function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

function redactCredentialLikeText(value) {
  return value
    .replace(/\bBearer\s+[^\s"'`,;]+/gi, 'Bearer [REDACTED]')
    .replace(/\b(?:sk|rk|xox[baprs])-?[A-Za-z0-9_-]{16,}\b/g, '[REDACTED_CREDENTIAL]')
    .replace(/\bAIza[0-9A-Za-z_-]{24,}\b/g, '[REDACTED_CREDENTIAL]')
    .replace(/((?:api[_-]?key|authorization|access[_-]?token|refresh[_-]?token|password|secret)\s*[:=]\s*)("[^"]*"|'[^']*'|[^\s,;]+)/gi,
      '$1[REDACTED]');
}

function partDigest(part) {
  let serialized;
  try { serialized = JSON.stringify(part); }
  catch { serialized = String(part); }
  if (typeof serialized !== 'string') serialized = String(part);
  return { sha256: sha256(serialized), bytes: Buffer.byteLength(serialized) };
}

function failureDiagnostic({ classification, providerID, modelID, sessionID, data, error }) {
  const info = data?.info && typeof data.info === 'object' ? data.info : {};
  const parts = Array.isArray(data?.parts) ? data.parts : [];
  const text = parts.filter(part => part?.type === 'text' && typeof part.text === 'string')
    .map(part => part.text).join('');
  const redacted = redactCredentialLikeText(text);
  const excerpt = Buffer.from(redacted, 'utf8').subarray(0, FAILURE_TEXT_PREVIEW_BYTES).toString('utf8');
  const partRows = parts.slice(0, FAILURE_PART_PREVIEW_COUNT).map(part => ({
    type: typeof part?.type === 'string' ? part.type.slice(0, 64) : 'unknown',
    ...(typeof part?.tool === 'string' ? { tool: part.tool.slice(0, 64) } : {}),
    ...(typeof part?.state?.status === 'string' ? { status: part.state.status.slice(0, 32) } : {}),
    ...partDigest(part)
  }));
  return {
    version: FAILURE_DIAGNOSTIC_VERSION,
    classification,
    ...(error?.transportFailurePhase ? { failure_phase: error.transportFailurePhase } : {}),
    provider_id: typeof providerID === 'string' ? providerID.slice(0, 128) : null,
    model_id: typeof modelID === 'string' ? modelID.slice(0, 128) : null,
    session_id: typeof sessionID === 'string' ? sessionID.slice(0, 128) : null,
    assistant_message_id: typeof info.id === 'string' ? info.id.slice(0, 128) : null,
    assistant_provider_id: typeof info.providerID === 'string' ? info.providerID.slice(0, 128) : null,
    assistant_model_id: typeof info.modelID === 'string' ? info.modelID.slice(0, 128) : null,
    assistant_text: {
      sha256: sha256(text),
      bytes: Buffer.byteLength(text),
      redacted_preview: excerpt,
      preview_bytes: Buffer.byteLength(excerpt),
      preview_truncated: Buffer.byteLength(redacted) > Buffer.byteLength(excerpt),
      redacted: redacted !== text
    },
    parts_total_count: parts.length,
    parts_omitted_count: Math.max(0, parts.length - partRows.length),
    parts: partRows,
    parts_preview_bytes: partRows.reduce((sum, part) => sum + part.bytes, 0),
    ...(error?.transportUpstreamError ? { upstream_error: error.transportUpstreamError } : {}),
    ...(error?.transportRetrySchedule ? { retry_schedule: error.transportRetrySchedule } : {}),
    ...(error?.transportEventStream ? { event_stream: error.transportEventStream } : {}),
    error: redactCredentialLikeText((error instanceof Error ? `${error.name}: ${error.message}` : String(error)).slice(0, 128))
  };
}

function safeSessionApiError(event, context) {
  if (event?.type !== 'session.error' || event.properties?.sessionID !== context.sessionID) return undefined;
  const upstream = event.properties?.error;
  if (upstream?.name !== 'APIError' || !upstream.data || typeof upstream.data !== 'object') return undefined;
  const data = upstream.data;
  const rawMessage = typeof data.message === 'string' ? data.message : 'provider API request failed';
  const message = redactCredentialLikeText(rawMessage).slice(0, 512);
  const rawBody = typeof data.responseBody === 'string' ? data.responseBody : '';
  const redactedBody = redactCredentialLikeText(rawBody);
  const preview = Buffer.from(redactedBody, 'utf8').subarray(0, 4096).toString('utf8');
  const retryHeader = Object.entries(data.responseHeaders ?? {}).find(([name]) => name.toLowerCase() === 'retry-after')?.[1];
  const retrySeconds = typeof retryHeader === 'string' ? Number(retryHeader) : NaN;
  const retryDate = typeof retryHeader === 'string' ? Date.parse(retryHeader) : NaN;
  const retryAfterMs = Number.isFinite(retrySeconds) && retrySeconds >= 0 ? Math.ceil(retrySeconds * 1000) :
    Number.isFinite(retryDate) ? Math.max(0, retryDate - Date.now()) : undefined;
  const error = new Error(`OpenCode provider request failed${Number.isSafeInteger(data.statusCode) ? ` (HTTP ${data.statusCode})` : ''}: ${message}`);
  error.code = 'OPENCODE_PROVIDER_ERROR';
  if (Number.isSafeInteger(data.statusCode)) error.providerStatusCode = data.statusCode;
  if (typeof data.isRetryable === 'boolean') error.providerRetryable = data.isRetryable;
  if (Number.isSafeInteger(retryAfterMs)) error.providerRetryAfterMs = retryAfterMs;
  error.transportUpstreamError = {
    event_type: 'session.error',
    event_session_id: event.properties.sessionID.slice(0, 128),
    name: 'APIError',
    ...(Number.isSafeInteger(data.statusCode) ? { status_code: data.statusCode } : {}),
    ...(typeof data.isRetryable === 'boolean' ? { retryable: data.isRetryable } : {}),
    ...(Number.isSafeInteger(retryAfterMs) ? { retry_after_ms: retryAfterMs } : {}),
    message,
    ...(rawBody ? { response_body: {
      sha256: sha256(rawBody), bytes: Buffer.byteLength(rawBody), redacted_preview: preview,
      preview_bytes: Buffer.byteLength(preview), preview_truncated: Buffer.byteLength(redactedBody) > Buffer.byteLength(preview),
      redacted: redactedBody !== rawBody
    } } : {})
  };
  return error;
}

function safeSessionStatus(event, sessionID) {
  if (event?.type !== 'session.status' || event.properties?.sessionID !== sessionID) return undefined;
  const status = event.properties?.status;
  if (!status || !['idle', 'busy', 'retry'].includes(status.type)) return undefined;
  if (status.type !== 'retry') return { type: status.type };
  const action = status.action && typeof status.action === 'object' ? status.action : undefined;
  return {
    type: 'retry',
    ...(Number.isSafeInteger(status.attempt) ? { attempt: status.attempt } : {}),
    ...(Number.isSafeInteger(status.next) ? { next_at_ms: status.next } : {}),
    ...(typeof status.message === 'string' ? { message: redactCredentialLikeText(status.message).slice(0, 512) } : {}),
    ...(action ? { action: Object.fromEntries(['reason', 'provider', 'title', 'message', 'label']
      .filter(key => typeof action[key] === 'string')
      .map(key => [key, redactCredentialLikeText(action[key]).slice(0, 160)])) } : {})
  };
}

function scheduledRetryError(status, sessionID) {
  if (status?.type !== 'retry') return undefined;
  const delay = Number.isSafeInteger(status.next_at_ms) ? Math.max(0, status.next_at_ms - Date.now()) : undefined;
  const error = new Error('OpenCode scheduled an internal provider retry; the bridge stopped it to preserve the configured request retry budget');
  error.code = 'OPENCODE_RETRY_SCHEDULED';
  error.providerRetryable = true;
  error.sdkRetrySuppressed = false;
  if (Number.isSafeInteger(delay)) error.providerRetryAfterMs = delay;
  error.transportRetrySchedule = {
    event_type: 'session.status', session_id: sessionID,
    ...(Number.isSafeInteger(status.attempt) ? { attempt: status.attempt } : {}),
    ...(Number.isSafeInteger(status.next_at_ms) ? { next_at_ms: status.next_at_ms } : {}),
    ...(Number.isSafeInteger(delay) ? { delay_ms: delay } : {}),
    ...(typeof status.message === 'string' ? { message: status.message } : {}),
    ...(status.action ? { action: status.action } : {}),
    upstream_http_status: null,
    sdk_retry_policy: {
      source: 'OpenCode session processor retry policy',
      server_commit: '53d1eabb61e21162157817bf677da0a4ad3332e3',
      source_file: 'packages/opencode/src/session/retry.ts',
      max_scheduled_retries: 5,
      canceled_before_scheduled_retry: false,
      note: 'The provider request that produced this status has already occurred. The bridge attempts to abort the scheduled retry; collector retries are a separate budget.'
    },
    retry_cancellation: { attempted: false, succeeded: false }
  };
  return error;
}

function watchSessionErrors(client, sessionID, directory, fetchImpl = globalThis.fetch) {
  const controller = new AbortController();
  let resolveMatch;
  const matchingError = new Promise(resolve => { resolveMatch = resolve; });
  let resolveReady;
  const ready = new Promise(resolve => { resolveReady = resolve; });
  let readySettled = false;
  let readyResult;
  let streamState = 'connecting';
  let streamHandshake;
  let latestSessionStatus;
  const permissionReplies = [];
  const settleReady = result => {
    if (readySettled) return;
    readySettled = true;
    readyResult = result;
    resolveReady(result);
  };
  const task = (async () => {
    try {
      // The pinned SDK's legacy /event endpoint takes directory in the first
      // argument and request options (including signal/fetch) in the second.
      // Its SSE result is lazy: subscribe() returns before the first fetch.
      // Mark ready only after that fetch has returned a valid event-stream.
      const eventFetch = async request => {
        const response = await fetchImpl(request);
        const contentType = response.headers.get('content-type') ?? '';
        streamHandshake = { status: response.status, contentType: contentType.slice(0, 96), directory };
        if (response.ok && response.body && contentType.toLowerCase().includes('text/event-stream')) {
          streamState = 'established';
          settleReady({ available: true, status: response.status, contentType: contentType.slice(0, 96) });
        } else {
          streamState = 'unavailable';
          settleReady({ available: false, reason: `event endpoint returned HTTP ${response.status} (${contentType.slice(0, 64) || 'no content type'})` });
        }
        return response;
      };
      const onSseError = error => {
        streamState = 'error';
        settleReady({ available: false,
          reason: error instanceof Error ? error.message.slice(0, 160) : 'event stream connection failed' });
      };
      const subscription = await client.event.subscribe({ directory }, {
        signal: controller.signal,
        fetch: eventFetch,
        sseMaxRetryAttempts: 1,
        sseDefaultRetryDelay: 0,
        sseSleepFn: async () => {},
        onSseError
      });
      if (!subscription?.stream || typeof subscription.stream[Symbol.asyncIterator] !== 'function') {
        streamState = 'unavailable';
        settleReady({ available: false, reason: 'event subscription returned no async stream' });
        return;
      }
      for await (const event of subscription.stream) {
        if (controller.signal.aborted) return;
        if ((event?.type === 'permission.asked' || event?.type === 'permission.v2.asked') &&
            event.properties?.sessionID === sessionID) {
          const request = event.properties;
          const v2 = event.type === 'permission.v2.asked';
          const row = { request_id: typeof request.id === 'string' ? request.id.slice(0, 128) : null,
            permission: typeof (request.permission ?? request.action) === 'string' ? (request.permission ?? request.action).slice(0, 128) : null,
            patterns: Array.isArray(request.patterns ?? request.resources) ? (request.patterns ?? request.resources)
              .filter(value => typeof value === 'string').slice(0, 16) : [] };
          try {
            const reply = v2 ? client.session?.permission?.reply : client.permission?.reply;
            if (!row.request_id || typeof reply !== 'function')
              throw new Error('official SDK permission reply API unavailable');
            const result = await reply.call(v2 ? client.session.permission : client.permission, {
              ...(v2 ? { sessionID } : {}), requestID: row.request_id, directory, reply: 'reject',
              message: 'Natlang bridge rejects every requested OpenCode tool except its pre-authorized action MCP tool.' });
            const replyData = unwrapSdkResult(result, 'permission rejection');
            if (!v2 && replyData !== true) throw new Error('official SDK did not confirm permission rejection');
            permissionReplies.push({ ...row, reply: 'reject', succeeded: true });
          } catch (error) {
            permissionReplies.push({ ...row, reply: 'reject', succeeded: false,
              error: redactCredentialLikeText(error instanceof Error ? error.message : String(error)).slice(0, 160) });
            const fatal = new Error('OpenCode permission request could not be rejected by the official SDK');
            fatal.code = 'OPENCODE_PERMISSION_REJECTION_FAILED';
            resolveMatch(fatal);
            return;
          }
        }
        latestSessionStatus = safeSessionStatus(event, sessionID) ?? latestSessionStatus;
        if (latestSessionStatus?.type === 'retry') {
          const retry = scheduledRetryError(latestSessionStatus, sessionID);
          if (retry) {
            retry.transportEventStream = { directory: streamHandshake?.directory ?? directory,
              status: streamHandshake?.status ?? null, content_type: streamHandshake?.contentType ?? null };
            resolveMatch(retry);
            return;
          }
        }
        const error = safeSessionApiError(event, { sessionID });
        if (error) {
          error.transportUpstreamError.event_stream = {
            directory: streamHandshake?.directory ?? directory,
            status: streamHandshake?.status ?? null,
            content_type: streamHandshake?.contentType ?? null
          };
          resolveMatch(error);
          return;
        }
      }
      if (!controller.signal.aborted) streamState = 'closed';
    } catch (error) {
      streamState = 'error';
      // The prompt response remains authoritative if the optional diagnostic
      // stream disconnects. It must not turn an SSE problem into a model error.
      settleReady({ available: false, reason: error instanceof Error ? error.message.slice(0, 160) : 'event subscription failed' });
    }
  })();
  return {
    matchingError,
    ready,
    async stop() {
      controller.abort();
      streamState = 'aborted';
      await withTimeout(task.catch(() => {}), 500, 'OpenCode event stream stop').catch(() => {});
    },
    snapshot() {
      return {
        state: streamState,
        ready: readyResult?.available ?? false,
        ...(readyResult?.reason ? { reason: redactCredentialLikeText(String(readyResult.reason)).slice(0, 160) } : {}),
        ...(streamHandshake ? { handshake: {
          status: streamHandshake.status,
          content_type: streamHandshake.contentType,
          directory: streamHandshake.directory
        } } : {}),
        ...(latestSessionStatus ? { latest_session_status: latestSessionStatus } : {}),
        permission_rejections: permissionReplies
      };
    }
  };
}

function failWithDiagnostic(message, context, classification, cause) {
  const error = new Error(message, cause ? { cause } : undefined);
  error.transportDiagnosticFull = failureDiagnostic({ ...context, classification, error });
  return error;
}

function classifyFailure(error, phase) {
  if (error?.code === 'OPENCODE_RETRY_SCHEDULED') return 'provider_retry_scheduled';
  if (error?.code === 'OPENCODE_PROVIDER_ERROR') return 'provider_request_failure';
  if (error?.code === 'REQUEST_TIMEOUT') return 'request_timeout';
  if (phase === 'tool_inventory' || phase === 'session_create') return 'opencode_setup_failure';
  if (phase === 'event_stream_bootstrap') return 'event_stream_bootstrap_failure';
  if (phase === 'session_prompt') return 'provider_prompt_failure';
  if (phase === 'assistant_validation') return 'assistant_response_validation_failure';
  if (phase === 'session_history_audit') return 'session_history_audit_failure';
  return 'bridge_failure';
}

async function persistFailureDiagnostic(error, directory) {
  const full = error.transportDiagnosticFull ?? failureDiagnostic({
    providerID: undefined, modelID: undefined, sessionID: undefined, classification: 'bridge_failure', error
  });
  const body = `${JSON.stringify(full)}\n`;
  const bodyBytes = Buffer.byteLength(body);
  const id = String(full.session_id ?? randomUUID()).replace(/[^A-Za-z0-9._-]/g, '_').slice(0, 128);
  const path = join(directory, '.natlang-transport-failures', `failure-${id}-${randomUUID()}.json`);
  let persisted;
  try {
    await mkdir(join(directory, '.natlang-transport-failures'), { recursive: true, mode: 0o700 });
    await writeFile(path, body, { flag: 'wx', mode: 0o600 });
    persisted = { path, bytes: bodyBytes, sha256: sha256(body) };
  } catch (writeError) {
    persisted = { write_error: redactCredentialLikeText((writeError instanceof Error ? writeError.message : String(writeError)).slice(0, 128)) };
  }
  const textPreview = Buffer.from(full.assistant_text.redacted_preview, 'utf8')
    .subarray(0, FAILURE_INLINE_TEXT_PREVIEW_BYTES).toString('utf8');
  error.transportDiagnostic = {
    version: full.version,
    classification: full.classification,
    provider_id: full.provider_id,
    model_id: full.model_id,
    session_id: full.session_id,
    assistant_message_id: full.assistant_message_id,
    assistant_text: {
      sha256: full.assistant_text.sha256,
      bytes: full.assistant_text.bytes,
      redacted_preview: textPreview,
      preview_bytes: Buffer.byteLength(textPreview),
      preview_truncated: full.assistant_text.preview_truncated || Buffer.byteLength(full.assistant_text.redacted_preview) > Buffer.byteLength(textPreview),
      redacted: full.assistant_text.redacted
    },
    parts_total_count: full.parts_total_count,
    parts_omitted_count: full.parts_omitted_count,
    parts: full.parts,
    parts_bytes: full.parts_bytes,
    ...(full.upstream_error ? { upstream_error: full.upstream_error } : {}),
    ...(full.retry_schedule ? { retry_schedule: full.retry_schedule } : {}),
    ...(full.event_stream ? { event_stream: full.event_stream } : {}),
    ...(full.failure_phase ? { failure_phase: full.failure_phase } : {}),
    error: full.error,
    ...(persisted.path ? { evidence_path: persisted.path, evidence_bytes: persisted.bytes, evidence_sha256: persisted.sha256 } : {}),
    ...(persisted.write_error ? { evidence_write_error: persisted.write_error } : {})
  };
  delete error.transportDiagnosticFull;
}

const SYSTEM_INSTRUCTIONS = [
  'You are returning one response for a Natlang model turn.',
  'The user message contains the complete serialized Natlang request, including its ordered message history and tool schemas.',
  'Treat that serialized history as the conversation context. Do not use built-in OpenCode tools. When a declared Natlang action is needed, call the natlang_action_bridge submit_action MCP tool with the exact Natlang function name and object arguments. This tool records a candidate action for the Natlang host and does not execute it. Do not invent or execute any other action.',
  'Return exactly one JSON text object matching the response_schema included in the user payload. Do not add markdown, fences, commentary, or extra keys.',
  'The object contains content, which is the assistant text, and toolCalls, which must be an empty array. The host obtains declared action calls only from the isolated submit_action MCP tool record.',
  'This is prompt-directed JSON text, not provider-enforced JSON Schema output. The Natlang host validates the recorded tool name and arguments, executes the action, and owns the next turn.'
].join(' ');

function isPlainRecord(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const proto = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

function jsonText(value, label) {
  let serialized;
  try { serialized = JSON.stringify(value); }
  catch (error) { throw new TypeError(`${label} must be JSON serializable`, { cause: error }); }
  if (typeof serialized !== 'string') throw new TypeError(`${label} must be JSON serializable`);
  return serialized;
}

function withTimeout(promise, milliseconds, label) {
  let timer;
  const timeout = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new Error(`${label} timed out after ${milliseconds} ms`)), milliseconds);
  });
  return Promise.race([Promise.resolve(promise), timeout]).finally(() => clearTimeout(timer));
}

function withAbort(promise, signal, label) {
  if (!signal) return Promise.resolve(promise);
  if (signal.aborted) {
    // The SDK call is already evaluated before this helper receives its promise.
    // Consume a simultaneous rejection when the signal was aborted synchronously.
    Promise.resolve(promise).catch(() => {});
    return Promise.reject(signal.reason ?? new Error(`${label} aborted`));
  }
  let onAbort;
  const aborted = new Promise((_, reject) => {
    onAbort = () => reject(signal.reason ?? new Error(`${label} aborted`));
    signal.addEventListener('abort', onAbort, { once: true });
  });
  return Promise.race([Promise.resolve(promise), aborted]).finally(() => signal.removeEventListener('abort', onAbort));
}

function toolNames(tools) {
  if (!Array.isArray(tools)) throw new TypeError('Natlang request tools must be an array');
  return tools.map((tool, index) => {
    const name = tool?.function?.name ?? tool?.name;
    if (typeof name !== 'string' || !name) throw new TypeError(`Natlang tool ${index} has no name`);
    return name;
  });
}

/**
 * Verify the official server kept its complete default inventory and the
 * configured action MCP tool. The session asks before tools except the exact
 * bridge tool; the SDK event watcher rejects observed asks. This is not a
 * universal execution barrier: handlers that skip Permission.ask could run
 * before history audit, so the caller must not claim native execution was
 * prevented without a separately reviewed containment boundary.
 * Pinned official SDK/client, tag commit
 * 53d1eabb61e21162157817bf677da0a4ad3332e3:
 * https://github.com/anomalyco/opencode/blob/53d1eabb61e21162157817bf677da0a4ad3332e3/packages/opencode/src/session/prompt.ts
 * https://github.com/anomalyco/opencode/blob/53d1eabb61e21162157817bf677da0a4ad3332e3/packages/opencode/src/permission/index.ts
 */
export function buildOpenCodeToolPolicy(ids) {
  if (!Array.isArray(ids) || ids.some(id => typeof id !== 'string' || !id))
    throw new TypeError('OpenCode tool inventory must be an array of nonempty IDs');
  const unique = [...new Set(ids)].sort();
  if (unique.length !== ids.length) throw new TypeError('OpenCode tool inventory contains duplicate IDs');
  if (unique.includes('*')) throw new TypeError('OpenCode tool inventory uses the reserved wildcard ID');
  if (!unique.includes('natlang_action_bridge_submit_action'))
    throw new TypeError('OpenCode default inventory omits the configured Natlang action MCP tool');
  return { wildcard: 'ask', action_tool: 'natlang_action_bridge_submit_action', retained_default_inventory: unique };
}

export function buildOpenCodeSessionPermissions(toolIds) {
  const rows = [{ permission: '*', pattern: '*', action: 'ask' }];
  rows.push({ permission: 'natlang_action_bridge_submit_action', pattern: '*',
    action: toolIds.length ? 'allow' : 'deny' });
  return rows;
}

/** Build the official SDK session.prompt input; default tools stay visible and gated by session permissions. */
export function buildOpenCodeStructuredPrompt(request, { providerID, modelID, agent } = {}) {
  if (!request || typeof request !== 'object' || Array.isArray(request))
    throw new TypeError('Natlang request must be an object');
  if (!Array.isArray(request.messages)) throw new TypeError('Natlang request messages must be an array');
  const names = toolNames(request.tools ?? []);
  const uniqueNames = [...new Set(names)];
  if (uniqueNames.length !== names.length) throw new TypeError('Natlang request contains duplicate tool names');
  if (request.tool_choice !== undefined && !['auto', 'required'].includes(request.tool_choice))
    throw new TypeError('Natlang tool_choice must be auto or required');

  const payload = {
    protocol: BRIDGE_ID,
    messages: request.messages,
    tools: request.tools ?? [],
    ...(request.tool_choice ? { tool_choice: request.tool_choice } : {}),
    ...(request.temperature === undefined ? {} : { temperature: request.temperature }),
    ...(request.seed === undefined ? {} : { seed: request.seed }),
    ...(request.max_tokens === undefined ? {} : { max_tokens: request.max_tokens }),
    response_schema: {
      type: 'object',
      additionalProperties: false,
      properties: {
        content: { type: 'string' },
        toolCalls: {
          type: 'array',
          items: {
            type: 'object',
            additionalProperties: false,
            properties: {
              name: { type: 'string', ...(uniqueNames.length ? { enum: uniqueNames } : {}) },
              arguments: { type: 'object' }
            },
            required: ['name', 'arguments']
          },
          ...(uniqueNames.length ? {} : { maxItems: 0 }),
          ...(request.tool_choice === 'required' ? { minItems: 1 } : {})
        }
      },
      required: ['content', 'toolCalls']
    }
  };
  const body = {
    ...(providerID && modelID ? { model: { providerID, modelID } } : {}),
    ...(agent ? { agent } : {}),
    system: SYSTEM_INSTRUCTIONS,
    parts: [{ type: 'text', text: jsonText(payload, 'serialized Natlang request') }]
  };
  return { body, responseSchema: payload.response_schema, toolNames: uniqueNames };
}

function unwrapSdkResult(result, operation) {
  if (!result || typeof result !== 'object') throw new Error(`OpenCode ${operation} returned no result`);
  if (result.error) {
    const name = typeof result.error.name === 'string' ? result.error.name : 'SDK error';
    const error = new Error(`OpenCode ${operation} failed (${name})`);
    if (name === 'APIError') {
      const data = result.error.data;
      if (Number.isSafeInteger(data?.statusCode)) error.providerStatusCode = data.statusCode;
      if (typeof data?.isRetryable === 'boolean') error.providerRetryable = data.isRetryable;
      error.code = 'OPENCODE_PROVIDER_ERROR';
    }
    throw error;
  }
  if (!Object.hasOwn(result, 'data'))
    throw new Error(`OpenCode ${operation} returned no data`);
  return result.data;
}

function parseStructuredTurn(value, allowedNames, toolChoice) {
  if (!isPlainRecord(value)) throw new TypeError('OpenCode JSON text output must be an object');
  const keys = Object.keys(value).sort();
  if (keys.join(',') !== 'content,toolCalls')
    throw new TypeError('OpenCode JSON text output must contain exactly content and toolCalls');
  if (typeof value.content !== 'string') throw new TypeError('OpenCode JSON text content must be a string');
  if (!Array.isArray(value.toolCalls)) throw new TypeError('OpenCode structured toolCalls must be an array');
  const calls = value.toolCalls.map((call, index) => {
    if (!isPlainRecord(call) || Object.keys(call).sort().join(',') !== 'arguments,name')
      throw new TypeError(`OpenCode structured tool call ${index} must contain exactly name and arguments`);
    if (typeof call.name !== 'string' || !allowedNames.has(call.name))
      throw new TypeError(`OpenCode returned an unavailable Natlang tool at index ${index}`);
    if (!isPlainRecord(call.arguments))
      throw new TypeError(`OpenCode structured tool call ${index} arguments must be an object`);
    return [call.name, call.arguments];
  });
  if (allowedNames.size === 0 && calls.length)
    throw new TypeError('OpenCode returned a tool call when Natlang offered no tools');
  return { calls, text: value.content };
}

function exactRecordKeys(value, keys) {
  return isPlainRecord(value) && Object.keys(value).sort().join(',') === [...keys].sort().join(',');
}

// OpenCode v1.18.35's pinned repair hook in session/llm.ts routes unavailable
// provider tools to tool/invalid.ts. That handler only returns an error result;
// it performs no I/O. Match its exact completed record and the Natlang tool
// name it rejected. Pinned upstream source (tag commit 53d1eabb61e21162157817bf677da0a4ad3332e3):
// https://github.com/anomalyco/opencode/blob/53d1eabb61e21162157817bf677da0a4ad3332e3/packages/opencode/src/session/llm.ts
// https://github.com/anomalyco/opencode/blob/53d1eabb61e21162157817bf677da0a4ad3332e3/packages/opencode/src/tool/invalid.ts
function rejectedNatlangToolAttempt(part, allowedNames) {
  const state = part?.state;
  const input = state?.input;
  if (part?.tool !== 'invalid' || state?.status !== 'completed' || state.title !== 'Invalid Tool' ||
      !exactRecordKeys(input, ['tool', 'error']) || typeof input.tool !== 'string' || !allowedNames.has(input.tool) ||
      typeof input.error !== 'string' || !input.error.startsWith(`Model tried to call unavailable tool '${input.tool}'.`) ||
      state.output !== `The arguments provided to the tool are invalid: ${input.error}` ||
      !exactRecordKeys(state.metadata, [])) return undefined;
  return { rejected_tool_name: input.tool, rejection: input.error, handler: 'OpenCode InvalidTool', status: 'completed',
    protocol_record: part };
}

const NATLANG_ACTION_TOOL = 'natlang_action_bridge_submit_action';

function auditOpenCodeTools(messages, allowedNames) {
  const parts = (Array.isArray(messages) ? messages : [])
    .flatMap(message => Array.isArray(message?.parts) ? message.parts : [])
    .filter(part => part?.type === 'tool');
  const rejectedNatlangAttempts = [];
  const bridgeActions = [];
  const externalActions = [];
  for (const part of parts) {
    if (part?.tool === NATLANG_ACTION_TOOL && part?.state?.status === 'completed' &&
        part.state.output === 'ACTION_RECORDED' && exactRecordKeys(part.state.input, ['name', 'arguments']) &&
        typeof part.state.input.name === 'string' && allowedNames.has(part.state.input.name) &&
        isPlainRecord(part.state.input.arguments)) {
      bridgeActions.push([part.state.input.name, part.state.input.arguments]);
      continue;
    }
    const rejected = rejectedNatlangToolAttempt(part, allowedNames);
    if (rejected) rejectedNatlangAttempts.push(rejected);
    else externalActions.push(part);
  }
  return { externalActions, bridgeActions, rejectedNatlangAttempts, toolParts: parts };
}

function assistantUsageAudit(messages, { finalMessageId, finalStructured } = {}) {
  const assistants = messages.filter(message => message?.info?.role === 'assistant');
  const sum = key => {
    const values = assistants.map(message => message.info?.tokens?.[key]).filter(Number.isFinite);
    return values.length ? values.reduce((total, value) => total + value, 0) : undefined;
  };
  const rows = assistants.map(message => {
    const info = message.info;
    const tokens = {};
    for (const key of ['total', 'input', 'output', 'reasoning'])
      if (Number.isFinite(info.tokens?.[key])) tokens[key] = info.tokens[key];
    if (info.tokens?.cache && typeof info.tokens.cache === 'object') {
      const cache = {};
      for (const key of ['read', 'write'])
        if (Number.isFinite(info.tokens.cache[key])) cache[key] = info.tokens.cache[key];
      if (Object.keys(cache).length) tokens.cache = cache;
    }
    return {
      message_id: typeof info.id === 'string' ? info.id : null,
      provider_id: typeof info.providerID === 'string' ? info.providerID : null,
      model_id: typeof info.modelID === 'string' ? info.modelID : null,
      agent: typeof info.agent === 'string' ? info.agent : null,
      ...(Number.isFinite(info.cost) ? { cost: info.cost } : {}),
      ...(Object.keys(tokens).length ? { tokens } : {}),
      ...(typeof info.finish === 'string' ? { finish: info.finish } : {}),
      ...(Object.hasOwn(info, 'structured') ? { structured_output: info.structured } :
        info.id === finalMessageId ? { parsed_json_text: finalStructured } : {}),
      tool_parts: (Array.isArray(message.parts) ? message.parts : []).filter(part => part?.type === 'tool')
        .map(part => ({ name: String(part.tool ?? 'unknown'), status: part.state?.status ?? 'unknown' }))
    };
  });
  return { rows, inputTokens: sum('input'), outputTokens: sum('output') };
}

function extractAssistantResult(data, allowedNames, context) {
  const info = data?.info;
  if (!info || typeof info !== 'object')
    throw failWithDiagnostic('OpenCode session prompt returned no assistant message', { ...context, data }, 'assistant_message_missing');
  if (typeof info.id !== 'string' || !info.id)
    throw failWithDiagnostic('OpenCode session prompt returned an assistant message without an id', { ...context, data }, 'assistant_message_missing');
  if (info.error) {
    const name = typeof info.error.name === 'string' ? info.error.name : 'assistant error';
    const error = failWithDiagnostic(`OpenCode assistant turn failed (${name})`, { ...context, data }, 'provider_or_assistant_error');
    if (name === 'APIError') {
      const data = info.error.data;
      if (Number.isSafeInteger(data?.statusCode)) error.providerStatusCode = data.statusCode;
      if (typeof data?.isRetryable === 'boolean') error.providerRetryable = data.isRetryable;
      error.code = 'OPENCODE_PROVIDER_ERROR';
    }
    throw error;
  }
  const parts = Array.isArray(data.parts) ? data.parts : [];
  const toolAudit = auditOpenCodeTools([{ parts }], allowedNames);
  const { externalActions } = toolAudit;
  if (externalActions.length) {
    const names = [...new Set(externalActions.map(part => String(part.tool ?? 'unknown')))];
    throw failWithDiagnostic(`OpenCode session contains non-bridge tool part(s) during a Natlang turn: ${names.join(', ')}`,
      { ...context, data }, 'native_tool_refusal');
  }
  const text = parts.filter(part => part?.type === 'text').map(part => part.text).join('');
  if (!text) throw failWithDiagnostic('OpenCode assistant message has no JSON text output', { ...context, data }, 'empty_assistant_text');
  let structured;
  try { structured = JSON.parse(text); }
  catch (cause) {
    throw failWithDiagnostic('OpenCode assistant text is not one valid JSON action object', { ...context, data },
      'invalid_json_text', cause);
  }
  return { structured, info, parts, text, bridgeActions: toolAudit.bridgeActions };
}

/**
 * Create an experimental single-turn backend around an official OpenCode v2 SDK
 * client. Each request gets a new session; cleanup runs on success, provider
 * error, and abort. OpenCode's default tool inventory stays present; session
 * permissions ask for every tool except the action-recording MCP tool, and
 * observed permission requests are rejected and logged through the SDK.
 */
export function createOpenCodeStructuredTurnBackend({ client, providerID, modelID, agent, directory, eventFetchImpl,
  cleanupTimeoutMs = 2_000 } = {}) {
  if (!client?.tool?.ids || !client?.session?.create || !client?.session?.prompt || !client?.session?.messages || !client?.session?.delete)
    throw new TypeError('an OpenCode SDK v2 client with tool inventory and session create, prompt, messages, and delete is required');
  if (typeof providerID !== 'string' || !providerID || typeof modelID !== 'string' || !modelID)
    throw new TypeError('providerID and modelID are required');
  if (typeof directory !== 'string' || !directory.startsWith('/'))
    throw new TypeError('an absolute isolated OpenCode scratch directory is required');
  if (!Number.isSafeInteger(cleanupTimeoutMs) || cleanupTimeoutMs < 1)
    throw new RangeError('cleanupTimeoutMs must be a positive integer');
  if (eventFetchImpl !== undefined && typeof eventFetchImpl !== 'function')
    throw new TypeError('eventFetchImpl must be a function when supplied');

  return async function turn(request, signal) {
    signal?.throwIfAborted();
    const prompt = buildOpenCodeStructuredPrompt(request, { providerID, modelID, agent });
    let failurePhase = 'tool_inventory';
    const toolInventoryResult = await withAbort(client.tool.ids({ directory }, { ...(signal ? { signal } : {}) }),
      signal, 'OpenCode tool inventory');
    const openCodeToolIds = unwrapSdkResult(toolInventoryResult, 'tool inventory');
    const openCodeTools = buildOpenCodeToolPolicy(openCodeToolIds);
    failurePhase = 'session_create';
    const createResult = await withAbort(client.session.create({
      model: { providerID, id: modelID }, ...(agent ? { agent } : {}), directory,
      title: 'Natlang isolated action turn', permission: buildOpenCodeSessionPermissions(prompt.toolNames)
    }, { ...(signal ? { signal } : {}) }), signal, 'OpenCode session create');
    const session = unwrapSdkResult(createResult, 'session create');
    if (typeof session.id !== 'string' || !session.id) throw new Error('OpenCode session create returned no session id');

    let primaryError;
    let assistantData;
    let errorWatch;
    let promptController;
    let removeExternalAbort;
    let errorEventMode = 'SDK event subscription unavailable';
    try {
      signal?.throwIfAborted();
      if (typeof client.event?.subscribe === 'function') {
        failurePhase = 'event_stream_bootstrap';
        // Preserve the fetch configured on the official SDK client (headers,
        // auth, and local routing); the test seam only replaces that transport.
        const sdkFetch = client.event.client?.getConfig?.().fetch;
        errorWatch = watchSessionErrors(client, session.id, directory,
          eventFetchImpl ?? sdkFetch ?? globalThis.fetch);
        try {
          const ready = await withAbort(withTimeout(errorWatch.ready, cleanupTimeoutMs,
            'OpenCode event stream bootstrap'), signal, 'OpenCode event stream bootstrap');
          if (ready.available) errorEventMode = `official SDK SSE stream established before prompt (HTTP ${ready.status}; ${ready.contentType}); matching-session errors and retry status monitored concurrently`;
          else {
            errorEventMode = `prompt-authoritative fallback; event stream unavailable (${ready.reason})`;
            await errorWatch.stop();
            errorWatch = undefined;
          }
        } catch (error) {
          if (signal?.aborted) throw error;
          errorEventMode = 'prompt-authoritative fallback; event stream bootstrap timed out';
          await errorWatch.stop();
          errorWatch = undefined;
        }
      }
      promptController = new AbortController();
      if (signal) {
        const forwardAbort = () => promptController.abort(signal.reason);
        if (signal.aborted) forwardAbort();
        else {
          signal.addEventListener('abort', forwardAbort, { once: true });
          removeExternalAbort = () => signal.removeEventListener('abort', forwardAbort);
        }
      }
      failurePhase = 'session_prompt';
      const promptPromise = withAbort(client.session.prompt({
        sessionID: session.id, directory, model: { providerID, modelID }, agent,
        system: prompt.body.system, parts: prompt.body.parts
      }, { signal: promptController.signal }), promptController.signal, 'OpenCode session prompt')
        .then(result => ({ kind: 'prompt', result }), error => ({ kind: 'prompt_error', error }));
      const outcome = errorWatch
        ? await Promise.race([promptPromise, errorWatch.matchingError.then(error => ({
          kind: error?.code === 'OPENCODE_RETRY_SCHEDULED' ? 'session_retry_scheduled' : 'session_error', error
        }))])
        : await promptPromise;
      if (outcome.kind === 'session_error') {
        promptController.abort(outcome.error);
        if (client.session.abort) {
          try { await withTimeout(client.session.abort({ sessionID: session.id, directory }), cleanupTimeoutMs,
            'OpenCode failed-session abort'); } catch {}
        }
        throw outcome.error;
      }
      if (outcome.kind === 'session_retry_scheduled') {
        promptController.abort(outcome.error);
        const retrySchedule = outcome.error.transportRetrySchedule;
        if (client.session.abort) {
          retrySchedule.retry_cancellation.attempted = true;
          try {
            const abortResult = await withTimeout(client.session.abort({ sessionID: session.id, directory }),
              cleanupTimeoutMs, 'OpenCode scheduled-retry abort');
            const aborted = unwrapSdkResult(abortResult, 'scheduled-retry abort') === true;
            retrySchedule.retry_cancellation.succeeded = aborted;
            if (!aborted) retrySchedule.retry_cancellation.result = 'server returned false';
            outcome.error.sdkRetrySuppressed = aborted;
            retrySchedule.sdk_retry_policy.canceled_before_scheduled_retry = aborted;
          } catch (abortError) {
            retrySchedule.retry_cancellation.error = redactCredentialLikeText(
              (abortError instanceof Error ? `${abortError.name}: ${abortError.message}` : String(abortError)).slice(0, 160));
          }
        } else {
          retrySchedule.retry_cancellation.error = 'session.abort endpoint unavailable';
        }
        throw outcome.error;
      }
      if (outcome.kind === 'prompt_error') throw outcome.error;
      const promptResult = outcome.result;
      const data = unwrapSdkResult(promptResult, 'session prompt');
      assistantData = data;
      const allowedNames = new Set(prompt.toolNames);
      const context = { providerID, modelID, sessionID: session.id };
      failurePhase = 'assistant_validation';
      const response = extractAssistantResult(data, allowedNames, context);
      let parsed;
      try { parsed = parseStructuredTurn(response.structured, new Set(prompt.toolNames), request.tool_choice); }
      catch (error) {
        throw failWithDiagnostic(error instanceof Error ? error.message : String(error), { ...context, data },
          'schema_invalid_json_text', error);
      }
      if (parsed.calls.length)
        throw failWithDiagnostic('JSON text output must leave toolCalls empty; native action calls are accepted only through submit_action MCP',
          { ...context, data }, 'schema_invalid_json_text');
      failurePhase = 'session_history_audit';
      const historyResult = await withAbort(client.session.messages({
        sessionID: session.id, directory
      }, { ...(signal ? { signal } : {}) }), signal, 'OpenCode session messages');
      const history = unwrapSdkResult(historyResult, 'session messages');
      if (!Array.isArray(history))
        throw new Error('OpenCode session message audit returned an invalid response');
      if (!history.length) throw new Error('OpenCode session message audit is empty; final assistant cannot be verified');
      const assistantHistory = history.filter(message => message?.info?.role === 'assistant');
      const finalHistoryMessage = assistantHistory.at(-1);
      if (!finalHistoryMessage || finalHistoryMessage.info?.id !== response.info.id)
        throw new Error('OpenCode session history does not contain the exact final assistant message');
      const historyAudit = auditOpenCodeTools(history, allowedNames);
      const externalActions = historyAudit.externalActions;
      if (externalActions.length) {
        const names = [...new Set(externalActions.map(part => String(part.tool ?? 'unknown')))];
        throw failWithDiagnostic(`OpenCode session contains non-bridge tool part(s): ${names.join(', ')}`,
          { ...context, data: { info: response.info, parts: history.flatMap(message => Array.isArray(message?.parts) ? message.parts : []) } },
          'native_tool_refusal');
      }
      const recordedCalls = historyAudit.bridgeActions;
      if (request.tool_choice === 'required' && recordedCalls.length === 0)
        throw failWithDiagnostic('Natlang required a tool call, but the action MCP did not record one',
          { ...context, data }, 'schema_invalid_json_text');
      const usageAudit = assistantUsageAudit(history, { finalMessageId: response.info.id, finalStructured: response.structured });
      return {
        ...(recordedCalls.length ? { calls: recordedCalls } : {}),
        ...(parsed.text ? { text: parsed.text } : {}),
        ...(usageAudit.inputTokens === undefined ? {} : { prompt_tokens: usageAudit.inputTokens }),
        ...(usageAudit.outputTokens === undefined ? {} : { completion_tokens: usageAudit.outputTokens }),
        raw_response: {
          transport: BRIDGE_ID,
          provider_id: providerID,
          model_id: modelID,
          session_id: session.id,
          provider_error_events: errorEventMode,
          fidelity: 'prompt-directed-strict-json-text; not provider-enforced JSON Schema or native provider tool-call output',
          output_contract: 'exact JSON text parsed and validated by the bridge',
          open_code_tool_policy: {
            control: 'official SDK session.create permission rules; prompt default inventory retained',
            inventory: 'official SDK tool.ids endpoint, queried for this request',
            inventory_ids: openCodeTools.retained_default_inventory,
            wildcard_action: 'ask',
            allowed_tool_ids: prompt.toolNames.length ? [NATLANG_ACTION_TOOL] : [],
            rejected_permission_requests: errorWatch?.snapshot().permission_rejections ?? [],
            session_history_audited: true,
            non_bridge_tool_parts_observed: false,
            native_execution_prevention: 'not established by permission policy and post-turn history audit'
          },
          generation_controls: {
            temperature: { requested: request.temperature ?? null, enforced_by_sdk: false },
            seed: { requested: request.seed ?? null, enforced_by_sdk: false },
            max_tokens: { requested: request.max_tokens ?? null, enforced_by_sdk: false },
            note: 'These controls are preserved in the serialized request for provenance; this SDK prompt API does not apply them as provider generation settings.'
          },
          structured_output: response.structured,
          upstream_message: { info: response.info, parts: response.parts },
          session_messages_audited: history.length,
          rejected_native_tool_attempts: historyAudit.rejectedNatlangAttempts,
          assistant_steps: usageAudit.rows,
          assistant_total_cost: assistantHistory.map(message => message.info?.cost).filter(Number.isFinite)
            .reduce((total, value) => total + value, 0),
          open_code_tool_parts: historyAudit.toolParts.map(part => ({
            name: String(part.tool ?? 'unknown'), status: part.state?.status ?? 'unknown'
          }))
        }
      };
    } catch (error) {
      primaryError = error;
      // Session history is removed in finally, so preserve bounded, secret-redacted
      // response evidence on the transport error before cleanup completes.
      if (error && typeof error.message === 'string') {
        error.transportFailurePhase = failurePhase;
        error.transportEventStream = {
          mode: redactCredentialLikeText(String(errorEventMode)).slice(0, 256),
          failure_phase: failurePhase,
          ...(errorWatch?.snapshot() ?? { state: 'unavailable', ready: false })
        };
        if (!error.transportDiagnosticFull) error.transportDiagnosticFull = failureDiagnostic({ providerID, modelID,
          sessionID: session.id, data: assistantData, classification: classifyFailure(error, failurePhase), error });
        await persistFailureDiagnostic(error, directory);
      }
      if (signal?.aborted && client.session.abort) {
        try { await withTimeout(client.session.abort({ sessionID: session.id, directory }),
          cleanupTimeoutMs, 'OpenCode abort cleanup'); }
        catch { /* preserve the original provider/abort failure */ }
      }
      throw error;
    } finally {
      removeExternalAbort?.();
      await errorWatch?.stop();
      try {
        const deleted = await withTimeout(client.session.delete({ sessionID: session.id, directory }),
          cleanupTimeoutMs, 'OpenCode session cleanup');
        if (!primaryError) unwrapSdkResult(deleted, 'session delete');
      } catch (error) {
        if (!primaryError) throw error;
      }
    }
  };
}

export const openCodeStructuredTurnBridgeId = BRIDGE_ID;
