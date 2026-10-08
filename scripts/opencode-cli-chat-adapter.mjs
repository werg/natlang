/**
 * Loopback Chat Completions adapter backed by the official OpenCode CLI.
 * The official CLI owns each provider request; the local MCP tool only records
 * declared Natlang action candidates. This is buffered and prompt-directed JSON,
 * not provider-enforced schema or native provider tool-call fidelity.
 */
import { appendFileSync, readFileSync, statSync } from 'node:fs';
import { createHash, randomUUID } from 'node:crypto';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import { buildOpenCodeStructuredPrompt } from './opencode-structured-turn.mjs';

const ID = 'natlang-opencode-cli-chat-adapter/1';
const MAX_BODY_BYTES = 2 * 1024 * 1024;

const sha256 = value => createHash('sha256').update(value).digest('hex');
const fileSize = path => { try { return statSync(path).size; } catch { return 0; } };
const safeError = value => String(value?.message ?? value).replace(/(?:Bearer\s+)[^\s,;]+/gi, 'Bearer [redacted]').replace(/\b(?:sk|rk|tok)[-_][A-Za-z0-9_-]{12,}\b/g, '[redacted]').slice(0, 500);
const isObject = value => Boolean(value) && typeof value === 'object' && !Array.isArray(value);
const jsonResponse = (res, status, value) => {
  if (res.destroyed || res.writableEnded) return;
  res.writeHead(status, { 'content-type': 'application/json; charset=utf-8', 'cache-control': 'no-store' });
  res.end(JSON.stringify(value));
};

async function readJson(req) {
  let size = 0;
  const chunks = [];
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw Object.assign(new Error('request body too large'), { status: 413 });
    chunks.push(chunk);
  }
  try { return JSON.parse(Buffer.concat(chunks).toString('utf8')); }
  catch { throw Object.assign(new Error('request body must be JSON'), { status: 400 }); }
}

function validateRequest(body, modelAlias) {
  if (!isObject(body) || !Array.isArray(body.messages) || !Array.isArray(body.tools))
    throw Object.assign(new Error('request must include messages and tools arrays'), { status: 400 });
  if (body.model !== undefined && body.model !== modelAlias)
    throw Object.assign(new Error('model does not match immutable adapter configuration'), { status: 409 });
  if (body.n !== undefined && body.n !== 1)
    throw Object.assign(new Error('only one completion is supported'), { status: 501 });
  if (body.stream !== undefined && typeof body.stream !== 'boolean')
    throw Object.assign(new Error('stream must be boolean'), { status: 400 });
  if (body.prompt_logprobs !== undefined || body.logprobs !== undefined || body.logit_bias !== undefined)
    throw Object.assign(new Error('scoring controls are unsupported'), { status: 501 });
  if (body.tool_choice && typeof body.tool_choice === 'object')
    throw Object.assign(new Error('named tool choice is unsupported'), { status: 501 });
  return body;
}

export function extractSessionText(messages) {
  if (!Array.isArray(messages)) return '';
  const latest = messages.filter(row => row?.info?.role === 'assistant').at(-1);
  if (!Array.isArray(latest?.parts)) return '';
  return latest.parts.filter(part => part?.type === 'text' && typeof part.text === 'string')
    .map(part => part.text).join('');
}

export async function readSessionText(client, sessionID, directory) {
  if (!client.session?.messages) return '';
  const result = await client.session.messages({ path: { id: sessionID }, query: { directory } });
  return !result || result.error ? '' : extractSessionText(result.data);
}

const canonicalJson = value => Array.isArray(value) ? `[${value.map(canonicalJson).join(',')}]` :
  isObject(value) ? `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}` : JSON.stringify(value);

function readActionRows(logPath, startBytes) {
  return readFileSync(logPath).subarray(startBytes).toString('utf8').split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
}

function validateActionRows(rows, allowedNames) {
  const allowed = new Set(allowedNames);
  for (const [index, row] of rows.entries())
    if (!isObject(row) || typeof row.name !== 'string' || !allowed.has(row.name) || !isObject(row.arguments))
      throw new Error(`MCP action ${index} did not match a declared Natlang tool`);
  return rows;
}

async function withBoundedTimeout(promise, timeoutMs, label) {
  let timer;
  try {
    return await Promise.race([Promise.resolve(promise), new Promise((_, reject) => {
      timer = setTimeout(() => reject(new Error(`${label} exceeded ${timeoutMs} ms`)), timeoutMs);
      timer.unref?.();
    })]);
  } finally { clearTimeout(timer); }
}

/** Abort and confirm one observed session through the official SDK's v1 path/query shape. */
export async function abortOpenCodeSession(client, sessionID, directory, { timeoutMs = 2000, pollMs = 50 } = {}) {
  if (typeof sessionID !== 'string' || !sessionID) return { ok: false, status: 'missing_session_id' };
  if (typeof client?.session?.abort !== 'function') return { ok: false, status: 'abort_unavailable' };
  const deadline = Date.now() + timeoutMs;
  if (typeof client.session.status === 'function') {
    const before = await withBoundedTimeout(client.session.status({ query: { directory } }),
      Math.max(1, Math.min(500, deadline - Date.now())), 'OpenCode session status');
    if (before?.error) return { ok: false, status: 'status_error', error: safeError(before.error) };
    if (!['busy', 'retry'].includes(before?.data?.[sessionID]?.type)) return { ok: true, status: 'not_busy' };
  }
  const abortResponse = await withBoundedTimeout(client.session.abort({ path: { id: sessionID }, query: { directory } }),
    Math.max(1, deadline - Date.now()), 'OpenCode session abort');
  if (abortResponse?.error) return { ok: false, status: 'abort_error', error: safeError(abortResponse.error) };
  if (abortResponse?.data !== true) return { ok: false, status: 'abort_not_confirmed', data: abortResponse?.data ?? null };
  if (typeof client.session.status !== 'function') return { ok: true, status: 'abort_confirmed_status_unavailable' };
  while (Date.now() < deadline) {
    const statusResponse = await withBoundedTimeout(client.session.status({ query: { directory } }),
      Math.max(1, Math.min(500, deadline - Date.now())), 'OpenCode session status');
    if (statusResponse?.error) return { ok: false, status: 'status_error', error: safeError(statusResponse.error) };
    if (!['busy', 'retry'].includes(statusResponse?.data?.[sessionID]?.type)) return { ok: true, status: 'idle', abort_confirmed: true };
    await new Promise(resolve => setTimeout(resolve, Math.min(pollMs, Math.max(1, deadline - Date.now()))));
  }
  return { ok: false, status: 'still_busy', abort_confirmed: true };
}

async function readActionRowsAfterWrite(logPath, startBytes, expectedCount, timeoutMs = 300) {
  const deadline = Date.now() + timeoutMs;
  let rows = [];
  do {
    try { rows = readActionRows(logPath, startBytes); } catch { rows = []; }
    if (rows.length >= expectedCount) return rows;
    await new Promise(resolve => setTimeout(resolve, 10));
  } while (Date.now() < deadline);
  return rows;
}

export function parseOpenCodeEnvelope(text, names, recordedActions) {
  let value;
  try { value = JSON.parse(text); }
  catch { throw new Error('OpenCode CLI returned invalid JSON text'); }
  if (!isObject(value) || Object.keys(value).sort().join(',') !== 'content,toolCalls' ||
      typeof value.content !== 'string' || !Array.isArray(value.toolCalls))
    throw new Error('OpenCode CLI response must have exactly content and toolCalls fields');
  if (value.toolCalls.length) {
    const echoedActions = value.toolCalls.map(call => ({ name: call?.name, arguments: call?.arguments }));
    if (echoedActions.length !== recordedActions.length || canonicalJson(echoedActions) !== canonicalJson(recordedActions))
      throw new Error('OpenCode CLI response toolCalls did not match audited MCP action records');
  }
  const allowed = new Set(names);
  return { content: value.content, allowed };
}

/**
 * Build the collector response from the action MCP audit when actions exist.
 * OpenCode's post-tool assistant text is not authoritative for those calls;
 * with no audited actions, retain the strict JSON envelope requirement.
 */
export function buildAuditedCompletion({ responseText, names, recordedActions, cliExitCode = 0, nonBridgeToolUses = 0 }) {
  if (cliExitCode !== 0) throw new Error(`OpenCode CLI exited ${cliExitCode}`);
  if (nonBridgeToolUses > 0) throw new Error('OpenCode CLI emitted non-bridge tool-use event(s)');
  if (!Array.isArray(recordedActions)) throw new Error('audited MCP actions must be an array');
  const allowed = new Set(names);
  const calls = recordedActions.map((row, index) => {
    if (!isObject(row) || typeof row.name !== 'string' || !allowed.has(row.name) || !isObject(row.arguments))
      throw new Error(`MCP action ${index} did not match a declared Natlang tool`);
    return { id: `call_${randomUUID()}`, type: 'function', function: { name: row.name, arguments: JSON.stringify(row.arguments) } };
  });
  if (!calls.length) {
    const envelope = parseOpenCodeEnvelope(responseText, names, recordedActions);
    return { content: envelope.content || null, calls, finalTextStatus: 'strict_json_envelope_no_actions' };
  }

  let finalTextStatus = 'empty_final_text_ignored';
  if (typeof responseText === 'string' && responseText.length) {
    try {
      const envelope = JSON.parse(responseText);
      if (!isObject(envelope) || Object.keys(envelope).sort().join(',') !== 'content,toolCalls' ||
          typeof envelope.content !== 'string' || !Array.isArray(envelope.toolCalls)) {
        finalTextStatus = 'non_envelope_final_text_ignored';
      } else {
        const echoed = envelope.toolCalls.map(call => ({ name: call?.name, arguments: call?.arguments }));
        finalTextStatus = echoed.length === recordedActions.length && canonicalJson(echoed) === canonicalJson(recordedActions)
          ? 'matching_echo_ignored' : 'inconsistent_echo_ignored';
      }
    } catch { finalTextStatus = 'invalid_json_final_text_ignored'; }
  }
  return { content: null, calls, finalTextStatus };
}

function parseJsonLines(stdout) {
  const events = [];
  for (const line of stdout.split(/\r?\n/).filter(Boolean)) {
    try { events.push(JSON.parse(line)); } catch { /* retain raw output in evidence */ }
  }
  return events;
}

function extractText(events) {
  const rows = events.filter(event => event?.type === 'text' && typeof event.part?.text === 'string');
  if (!rows.length) return '';
  // CLI JSON events may include text from more than one assistant message
  // around MCP tool calls. Select only the final assistant message when IDs
  // are available; otherwise retain the final complete text event.
  const finalPart = rows.at(-1).part;
  const messageID = finalPart.messageID ?? finalPart.messageId;
  if (typeof messageID !== 'string') return finalPart.text;
  return rows.filter(event => (event.part.messageID ?? event.part.messageId) === messageID)
    .map(event => event.part.text).join('');
}

function auditCliEvents(events, actionToolNames) {
  const toolUses = [];
  const steps = { started: 0, finished: 0, finish_events_with_usage: 0, tokens: { total: 0, input: 0, output: 0, reasoning: 0, cache_read: 0, cache_write: 0 } };
  for (const event of events) {
    const part = event?.part ?? {};
    if (event?.type === 'step_start' || part.type === 'step-start') steps.started++;
    if (event?.type === 'step_finish' || part.type === 'step-finish') {
      steps.finished++;
      const usage = part.tokens;
      if (isObject(usage)) {
        steps.finish_events_with_usage++;
        for (const [out, key] of [['total', 'total'], ['input', 'input'], ['output', 'output'], ['reasoning', 'reasoning']])
          if (Number.isFinite(usage[key])) steps.tokens[out] += usage[key];
        if (Number.isFinite(usage.cache?.read)) steps.tokens.cache_read += usage.cache.read;
        if (Number.isFinite(usage.cache?.write)) steps.tokens.cache_write += usage.cache.write;
      }
    }
    const isToolEvent = String(event?.type ?? '').toLowerCase().includes('tool') || part.type === 'tool';
    if (isToolEvent) {
      const name = part.tool ?? part.name ?? event?.tool ?? event?.name ?? null;
      const callID = part.callID ?? part.callId ?? event?.callID ?? event?.callId ?? null;
      toolUses.push({ event_type: event?.type ?? null, part_type: part.type ?? null, name, call_id: callID,
        bridge: typeof name === 'string' && actionToolNames.has(name) });
    }
  }
  steps.usage_available = steps.finish_events_with_usage > 0;
  return { steps, toolUses };
}

function boundedKill(child, firstSignal = 'SIGINT', graceMs = 1000) {
  if (!child || child.exitCode !== null || child.signalCode !== null) return Promise.resolve();
  return new Promise(resolve => {
    let done = false;
    let timer;
    const finish = () => { if (done) return; done = true; clearTimeout(timer); child.off('close', finish); resolve(); };
    child.once('close', finish);
    child.kill(firstSignal);
    timer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
      timer = setTimeout(() => {
        if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
        timer = setTimeout(finish, graceMs);
      }, graceMs);
    }, graceMs);
  });
}

function permissionDetails(event) {
  const v2 = event.type === 'permission.v2.asked';
  const p = event.properties ?? {};
  return { v2, sessionID: p.sessionID, requestID: p.id, permission: p.permission ?? p.action ?? null };
}

function createEventWatcher({ baseUrl, directory, client, onViolation }) {
  const controller = new AbortController();
  let response, reader;
  const permissionRejections = [];
  const retryEvents = [];
  const sessionErrors = [];
  const task = (async () => {
    try {
      response = await fetch(`${baseUrl}/event?directory=${encodeURIComponent(directory)}`, { signal: controller.signal });
      if (!response.ok || !response.headers.get('content-type')?.includes('text/event-stream') || !response.body)
        throw new Error(`event stream unavailable (HTTP ${response.status})`);
      reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      while (!controller.signal.aborted) {
        const { value, done } = await reader.read();
        if (done) return;
        buffer += decoder.decode(value, { stream: true });
        let split;
        while ((split = buffer.search(/\r?\n\r?\n/)) >= 0) {
          const block = buffer.slice(0, split);
          buffer = buffer.slice(split).replace(/^\r?\n\r?\n/, '');
          const data = block.split(/\r?\n/).filter(line => line.startsWith('data:'))
            .map(line => line.slice(5).trimStart()).join('\n');
          if (!data) continue;
          let event;
          try { event = JSON.parse(data); } catch { continue; }
          if (event.type === 'permission.asked' || event.type === 'permission.v2.asked') {
            const details = permissionDetails(event);
            const record = { ...details, reply: 'reject', succeeded: false };
            onViolation({ kind: 'native_permission', sessionID: details.sessionID });
            try {
              const method = details.v2 ? client.session.permission.reply : client.permission.reply;
              const reply = await method({ ...(details.v2 ? { sessionID: details.sessionID } : {}),
                requestID: details.requestID, directory, reply: 'reject',
                message: 'Natlang CLI bridge rejects native OpenCode tool permissions.' });
              if (reply?.error) throw new Error(String(reply.error.message ?? 'permission rejection failed').slice(0, 180));
              if (!details.v2 && reply?.data !== true) throw new Error('permission rejection was not confirmed');
              record.succeeded = true;
            } catch (error) { record.error = String(error?.message ?? error).slice(0, 180); }
            permissionRejections.push(record);
          } else if (event.type === 'session.status' && event.properties?.status?.type === 'retry') {
            const sessionID = event.properties.sessionID;
            const retry = { sessionID, attempt: event.properties.status.attempt ?? null,
              message: String(event.properties.status.message ?? '').slice(0, 300) };
            retryEvents.push(retry);
            try { retry.abort_result = await abortOpenCodeSession(client, sessionID, directory, { timeoutMs: 1000 }); }
            catch (error) { retry.abort_result = { ok: false, status: 'abort_error', error: safeError(error) }; }
            onViolation({ kind: 'provider_retry', sessionID });
          } else if (event.type === 'session.error') {
            sessionErrors.push({ sessionID: event.properties?.sessionID ?? null,
              name: event.properties?.error?.name ?? 'Error',
              message: String(event.properties?.error?.message ?? '').slice(0, 300) });
          }
        }
      }
    } catch (error) {
      if (!controller.signal.aborted) {
        sessionErrors.push({ kind: 'event_stream', message: String(error?.message ?? error).slice(0, 240) });
        onViolation({ kind: 'event_stream_error' });
      }
    }
  })();
  return {
    task, permissionRejections, retryEvents, sessionErrors,
    async close() {
      if (!controller.signal.aborted) controller.abort();
      try { await reader?.cancel(); } catch {}
      try { await response?.body?.cancel(); } catch {}
      await Promise.race([task.catch(() => {}), new Promise(resolve => setTimeout(resolve, 500))]);
    }
  };
}

/** Create the one-at-a-time OpenAI-compatible collector endpoint. */
export async function createOpenCodeCliChatAdapter(options = {}) {
  const { cliPath, client, baseUrl, directory, modelAlias, modelID, outputDirectory } = options;
  if (!cliPath || !client || !baseUrl || !directory?.startsWith('/') || !outputDirectory?.startsWith('/'))
    throw new TypeError('official CLI, SDK client, loopback server URL, isolated directory, and output directory are required');
  const maxRequestMs = options.maxRequestMs ?? 180_000;
  const maxCliTurns = options.maxCliTurns ?? options.maxModelRequests ?? 384;
  const contextTokens = options.contextTokens ?? 32768;
  if (!Number.isSafeInteger(contextTokens) || contextTokens < 1) throw new RangeError('contextTokens must be positive');
  if (!Number.isSafeInteger(maxCliTurns) || maxCliTurns < 1) throw new RangeError('maxCliTurns must be positive');
  const timeoutMs = options.timeoutMs ?? maxRequestMs;
  if (!Number.isSafeInteger(maxRequestMs) || maxRequestMs < 1 || !Number.isSafeInteger(timeoutMs) || timeoutMs < 1)
    throw new RangeError('timeouts must be positive integers');
  const providerID = options.providerID ?? 'opencode';
  const modelName = modelAlias ?? `${providerID}/${modelID}`;
  if (!client.mcp?.status || !client.mcp?.connect || !client.tool?.ids)
    throw new TypeError('official OpenCode SDK client must expose MCP status/connect and tool inventory');
  const unwrap = (result, operation) => {
    if (!result || result.error || !Object.hasOwn(result, 'data'))
      throw new Error(`official OpenCode ${operation} failed`);
    return result.data;
  };
  let mcpStatus = unwrap(await client.mcp.status({ directory }), 'MCP status');
  if (mcpStatus?.natlang_action_bridge?.status !== 'connected') {
    unwrap(await client.mcp.connect({ name: 'natlang_action_bridge', directory }), 'action MCP connect');
    mcpStatus = unwrap(await client.mcp.status({ directory }), 'MCP status');
  }
  if (mcpStatus?.natlang_action_bridge?.status !== 'connected')
    throw new Error('configured Natlang action MCP is not connected');
  const defaultToolIds = unwrap(await client.tool.ids({ directory }), 'default tool inventory');
  if (!Array.isArray(defaultToolIds) || defaultToolIds.length === 0 ||
      defaultToolIds.some(id => typeof id !== 'string' || !id))
    throw new Error('official default OpenCode tool inventory is absent or invalid');
  let active = false, closing = false, activeChild, cliTurnCount = 0;
  const server = createServer(async (req, res) => {
    if (req.method === 'GET' && req.url === '/health') {
      jsonResponse(res, 200, { ok: true, adapter: ID, providerChecked: false });
      return;
    }
    if (req.method !== 'POST' || req.url !== '/v1/chat/completions') {
      jsonResponse(res, 404, { error: { message: 'route not found', code: 'not_found' } });
      return;
    }
    if (closing) { jsonResponse(res, 503, { error: { message: 'adapter closing', code: 'closing' } }); return; }
    if (cliTurnCount >= maxCliTurns) { jsonResponse(res, 429, { error: { message: 'CLI turn budget exhausted', code: 'cli_turn_budget' } }); return; }
    if (active) { jsonResponse(res, 429, { error: { message: 'only one active CLI request is allowed', code: 'overloaded' } }); return; }
    active = true;
    let timedOut = false, clientCancelled = false, childStopPromise;
    const stopChild = () => !activeChild ? Promise.resolve() : (childStopPromise ??= boundedKill(activeChild));
    const cancelRequest = () => { clientCancelled = true; void stopChild(); };
    req.once('aborted', cancelRequest);
    res.once('close', () => { if (!res.writableEnded) cancelRequest(); });
    let watcher;
    let diagnostics = { request_id: randomUUID(), started_at: new Date().toISOString(), action_log_start_bytes: null };
    let stdout = '', stderr = '', exit = null, violation, invocationWritten = false;
    const timer = setTimeout(() => { timedOut = true; void stopChild(); }, timeoutMs);
    timer.unref?.();
    try {
      const body = validateRequest(await readJson(req), modelName);
      cliTurnCount++;
      const prompt = buildOpenCodeStructuredPrompt(body, { providerID, modelID: modelID ?? modelName });
      const userText = [
        'Return one response for a Natlang model turn. The following text contains the full serialized Natlang request with ordered message history and tool schemas. Use the configured `natlang_action_bridge` MCP `submit_action` tool to record each declared Natlang action; it only records candidates and does not execute them. Do not use built-in OpenCode tools. Return exactly one JSON object matching the included response_schema, with content and an empty toolCalls array. This is prompt-directed text, not provider-enforced JSON Schema.',
        prompt.body.parts[0].text
      ].join('\n\n');
      const allowedNames = body.tools.map(tool => tool?.function?.name ?? tool?.name);
      if (allowedNames.some(name => typeof name !== 'string' || !name) || new Set(allowedNames).size !== allowedNames.length)
        throw new TypeError('tool list has missing or duplicate function names');
      const logPath = options.actionLogPath;
      if (!logPath) throw new Error('action MCP log path is required');
      const startBytes = statSync(logPath).size;
      diagnostics.action_log_start_bytes = startBytes;
      diagnostics.user_text_sha256 = sha256(userText);
      diagnostics.user_text_bytes = Buffer.byteLength(userText);
      diagnostics.allowed_tool_names = allowedNames;
      const observedSessionIDs = new Set();
      const actionCallIDsBySession = new Map();
      const bridgeCallIDs = new Set();
      let nonBridgeToolUseCount = 0, streamBuffer = '', streamQueue = Promise.resolve(), terminalActionBoundary = null;
      const actionToolNames = new Set(['natlang_action_bridge_submit_action', 'submit_action']);
      const handleStreamEvent = async event => {
        const part = event?.part ?? {};
        const sessionID = event?.sessionID ?? part.sessionID;
        if (typeof sessionID === 'string' && sessionID) observedSessionIDs.add(sessionID);
        if ((event?.type === 'step_start' || part.type === 'step-start') && terminalActionBoundary?.session_id === sessionID)
          terminalActionBoundary.continued_after_submit = true;
        const isToolEvent = String(event?.type ?? '').toLowerCase().includes('tool') || part.type === 'tool';
        if (isToolEvent) {
          const name = part.tool ?? part.name ?? event?.tool ?? event?.name;
          const callID = part.callID ?? part.callId ?? event?.callID ?? event?.callId;
          const completed = part.state?.status === 'completed' || event?.state?.status === 'completed';
          if (typeof name === 'string' && !actionToolNames.has(name)) nonBridgeToolUseCount++;
          if (completed && actionToolNames.has(name) && typeof callID === 'string' && typeof sessionID === 'string') {
            bridgeCallIDs.add(callID);
            const ids = actionCallIDsBySession.get(sessionID) ?? new Set();
            ids.add(callID); actionCallIDsBySession.set(sessionID, ids);
          }
        }
        const stepFinished = event?.type === 'step_finish' || part.type === 'step-finish';
        const reason = part.reason ?? event?.reason;
        if (!stepFinished || reason !== 'tool-calls' || terminalActionBoundary || nonBridgeToolUseCount > 0 ||
            watcher?.sessionErrors?.length || typeof sessionID !== 'string') return;
        const sessionCalls = actionCallIDsBySession.get(sessionID);
        if (!sessionCalls?.size) return;
        const rows = await readActionRowsAfterWrite(logPath, startBytes, bridgeCallIDs.size);
        if (rows.length !== bridgeCallIDs.size || rows.length === 0) return;
        try { validateActionRows(rows, allowedNames); }
        catch (error) { diagnostics.terminal_action_boundary_error = safeError(error); return; }
        terminalActionBoundary = { session_id: sessionID, step_finish_reason: reason,
          bridge_call_ids: [...bridgeCallIDs], action_record_count: rows.length, status: 'abort_pending' };
        try {
          const result = await abortOpenCodeSession(client, sessionID, directory, { timeoutMs: 2000 });
          terminalActionBoundary.abort_result = result;
          terminalActionBoundary.status = result.ok ? (result.status === 'not_busy' ? 'already_idle' : 'aborted') : 'abort_not_confirmed';
        } catch (error) {
          terminalActionBoundary.abort_result = { ok: false, status: 'abort_error', error: safeError(error) };
          terminalActionBoundary.status = 'abort_error';
        }
      };
      const queueStreamChunk = chunk => {
        streamBuffer += chunk.toString();
        const lines = streamBuffer.split(/\r?\n/); streamBuffer = lines.pop() ?? '';
        for (const line of lines) if (line.trim()) streamQueue = streamQueue.then(() => {
          try { return handleStreamEvent(JSON.parse(line)); }
          catch (error) { diagnostics.stream_event_error = safeError(error); }
        });
      };
      watcher = createEventWatcher({ baseUrl, directory, client,
        onViolation: value => { violation ??= value; diagnostics.violation = value; void stopChild(); } });
      const args = ['run', '--format', 'json', '--model', modelName, '--dir', directory,
        '--pure', '--attach', baseUrl, userText];
      diagnostics.stdout_path = 'cli-stdout.raw'; diagnostics.stderr_path = 'cli-stderr.raw';
      diagnostics.stdout_start_bytes = fileSize(`${outputDirectory}/cli-stdout.raw`);
      diagnostics.stderr_start_bytes = fileSize(`${outputDirectory}/cli-stderr.raw`);
      const child = spawn(cliPath, args, { cwd: directory, env: options.env ?? process.env,
        stdio: ['ignore', 'pipe', 'pipe'] });
      activeChild = child;
      const stdoutPath = `${outputDirectory}/cli-stdout.raw`;
      const stderrPath = `${outputDirectory}/cli-stderr.raw`;
      child.stdout.on('data', chunk => { stdout += chunk.toString(); appendFileSync(stdoutPath, chunk); queueStreamChunk(chunk); });
      child.stderr.on('data', chunk => { stderr += chunk.toString(); appendFileSync(stderrPath, chunk); });
      if (clientCancelled || timedOut || violation) void stopChild();
      exit = await new Promise((resolve, reject) => {
        child.once('error', reject);
        child.once('close', (code, signal) => resolve({ code, signal }));
      });
      await streamQueue.catch(() => {});
      if (childStopPromise) await childStopPromise;
      diagnostics.stdout_end_bytes = fileSize(stdoutPath); diagnostics.stderr_end_bytes = fileSize(stderrPath);
      const events = parseJsonLines(stdout);
      const audit = auditCliEvents(events, new Set(['natlang_action_bridge_submit_action', 'submit_action']));
      diagnostics.cli_turn_number = cliTurnCount;
      diagnostics.cli_event_count = events.length;
      diagnostics.cli_session_ids = [...new Set([...observedSessionIDs,
        ...events.map(event => event?.sessionID ?? event?.part?.sessionID).filter(id => typeof id === 'string')])];
      diagnostics.provider_step_telemetry = audit.steps;
      diagnostics.cli_tool_use_audit = audit.toolUses;
      diagnostics.non_bridge_tool_use_count = audit.toolUses.filter(use => !use.bridge).length;
      diagnostics.terminal_action_boundary = terminalActionBoundary;
      const cleanupReason = timedOut ? 'timeout' : clientCancelled ? 'client_cancelled' : violation?.kind;
      if (cleanupReason && diagnostics.cli_session_ids.length) {
        diagnostics.server_session_cleanup = [];
        for (const sessionID of diagnostics.cli_session_ids) {
          try { diagnostics.server_session_cleanup.push({ session_id: sessionID,
            ...(await abortOpenCodeSession(client, sessionID, directory, { timeoutMs: 1500 })) }); }
          catch (error) { diagnostics.server_session_cleanup.push({ session_id: sessionID,
            ok: false, status: 'abort_error', error: safeError(error) }); }
        }
        diagnostics.server_session_cleanup_reason = cleanupReason;
      }
      const cliEventErrors = events.filter(event => event?.type === 'error').map(event => ({
        name: event.error?.name ?? 'Error', message: safeError(event.error?.message ?? event.error ?? 'OpenCode CLI error') }));
      diagnostics.cli_event_errors = cliEventErrors;
      if (timedOut) throw Object.assign(new Error(`OpenCode CLI request exceeded ${timeoutMs} ms`), { code: 'REQUEST_TIMEOUT' });
      if (clientCancelled) throw Object.assign(new Error('OpenCode CLI request was cancelled by the client'), { code: 'CLIENT_CANCELLED' });
      if (violation) throw Object.assign(new Error(`OpenCode CLI stopped after ${violation.kind}`), { code: violation.kind });
      if (exit.code !== 0) throw Object.assign(new Error(`OpenCode CLI exited ${exit.code ?? exit.signal}`), { code: 'CLI_EXIT' });
      diagnostics.cli_exit_code = exit.code; diagnostics.cli_signal = exit.signal;
      const nonBridgeToolUses = audit.toolUses.filter(use => !use.bridge);
      if (nonBridgeToolUses.length) throw Object.assign(new Error('OpenCode CLI emitted non-bridge tool-use event(s)'), {
        code: 'NON_BRIDGE_TOOL_USE', toolUses: audit.toolUses });
      if (cliEventErrors.length) throw Object.assign(new Error('OpenCode CLI emitted provider error event(s)'), {
        code: 'CLI_EVENT_ERROR', cliEventErrors });
      if (terminalActionBoundary?.continued_after_submit) throw Object.assign(new Error(
        'OpenCode session started another model step after a completed audited action'), { code: 'TERMINAL_BOUNDARY_LATE' });
      await watcher.close();
      const sessionErrors = watcher.sessionErrors.filter(error =>
        !(terminalActionBoundary?.status === 'aborted' &&
          error.sessionID === terminalActionBoundary.session_id && /^aborted$/i.test(String(error.message ?? ''))));
      diagnostics.session_errors = watcher.sessionErrors;
      if (sessionErrors.length) throw Object.assign(new Error('OpenCode CLI emitted session error event(s)'), {
        code: 'SESSION_ERROR', sessionErrors });
      const rawLog = readFileSync(logPath);
      const actionRows = rawLog.subarray(startBytes).toString('utf8').split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
      const recordedActions = actionRows.map(row => ({ name: row.name, arguments: row.arguments }));
      let responseText = extractText(events);
      if (!responseText && client.session?.messages) {
        for (const sessionID of [...diagnostics.cli_session_ids].reverse()) {
          responseText = await readSessionText(client, sessionID, directory);
          if (responseText) break;
        }
        diagnostics.response_text_source = responseText ? 'official_sdk_session_messages_fallback' : 'cli_json_events';
      }
      diagnostics.final_text_sha256 = sha256(responseText);
      diagnostics.final_text_bytes = Buffer.byteLength(responseText);
      const completion = buildAuditedCompletion({ responseText, names: allowedNames, recordedActions,
        cliExitCode: exit.code, nonBridgeToolUses: nonBridgeToolUses.length });
      diagnostics.final_text_status = completion.finalTextStatus;
      const calls = completion.calls;
      diagnostics.action_log_end_bytes = rawLog.length;
      diagnostics.action_record_count = calls.length;
      diagnostics.stdout_bytes = Buffer.byteLength(stdout); diagnostics.stderr_bytes = Buffer.byteLength(stderr);
      diagnostics.permission_rejections = watcher.permissionRejections;
      diagnostics.provider_retries = watcher.retryEvents;
      diagnostics.event_stream_errors = watcher.sessionErrors;
      diagnostics.finished_at = new Date().toISOString();
      appendFileSync(`${outputDirectory}/cli-invocations.jsonl`, JSON.stringify(diagnostics) + '\n', { mode: 0o600 });
      invocationWritten = true;
      jsonResponse(res, 200, {
        id: `chatcmpl-${randomUUID()}`, object: 'chat.completion', created: Math.floor(Date.now() / 1000), model: modelName,
        choices: [{ index: 0, message: { role: 'assistant', content: completion.content, ...(calls.length ? { tool_calls: calls } : {}) },
          finish_reason: calls.length ? 'tool_calls' : 'stop' }],
        raw_response: { cli_exit_code: exit.code, cli_signal: exit.signal, cli_events: events.map(e => e.type ?? null),
          event_stream_errors: watcher.sessionErrors, permission_rejections: watcher.permissionRejections,
          provider_retries: watcher.retryEvents, user_text_sha256: sha256(userText), user_text_bytes: Buffer.byteLength(userText),
          action_log_byte_range: [startBytes, rawLog.length], action_record_count: calls.length,
          route: 'official OpenCode CLI run --attach; default tool inventory; local action MCP',
          cli_turn_number: cliTurnCount, cli_turn_limit: maxCliTurns,
          provider_step_telemetry: audit.steps, cli_tool_use_audit: audit.toolUses,
          provider_request_count: null,
          provider_request_count_note: 'The adapter counts CLI turns. One CLI turn may contain multiple provider model steps; step_finish and token usage are recorded when emitted, but are not a complete provider-request meter.',
          training_admission: false }
      });
    } catch (error) {
      diagnostics.raw_error = safeError(error);
      diagnostics.timed_out = timedOut;
      diagnostics.stdout_end_bytes = fileSize(`${outputDirectory}/cli-stdout.raw`);
      diagnostics.stderr_end_bytes = fileSize(`${outputDirectory}/cli-stderr.raw`);
      diagnostics.action_log_end_bytes = fileSize(options.actionLogPath);
      if (Number.isSafeInteger(diagnostics.action_log_start_bytes)) {
        const failedLog = readFileSync(options.actionLogPath).subarray(diagnostics.action_log_start_bytes).toString('utf8').split(/\r?\n/).filter(Boolean);
        diagnostics.action_record_count = failedLog.length;
      }
      diagnostics.error_code = error?.code ?? null;
      diagnostics.cli_exit_code = exit?.code ?? null; diagnostics.cli_signal = exit?.signal ?? null;
      diagnostics.stdout_bytes = Buffer.byteLength(stdout); diagnostics.stderr_bytes = Buffer.byteLength(stderr);
      diagnostics.permission_rejections = watcher?.permissionRejections ?? [];
      if (error?.toolUses) diagnostics.cli_tool_use_audit = error.toolUses;
      diagnostics.session_errors = watcher?.sessionErrors ?? [];
      diagnostics.cli_turn_number = cliTurnCount;
      diagnostics.cli_turn_limit = maxCliTurns;
      diagnostics.provider_retries = watcher?.retryEvents ?? [];
      diagnostics.event_stream_errors = watcher?.sessionErrors ?? [];
      diagnostics.finished_at = new Date().toISOString();
      try { if (!invocationWritten) appendFileSync(`${outputDirectory}/cli-invocations.jsonl`, JSON.stringify(diagnostics) + '\n', { mode: 0o600 }); } catch {}
      const status = Number.isInteger(error?.status) ? error.status : timedOut || error?.code === 'REQUEST_TIMEOUT' ? 504 :
        error?.code === 'provider_retry' ? 503 : 502;
      jsonResponse(res, status, { error: { message: String(error?.message ?? error).slice(0, 500), code: error?.code ?? 'cli_bridge_error' } });
    } finally {
      clearTimeout(timer);
      req.removeListener('aborted', cancelRequest);
      await watcher?.close();
      activeChild = undefined;
      active = false;
    }
  });
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(options.port ?? 0, options.host ?? '127.0.0.1', resolve);
  });
  const address = server.address();
  let closed = false;
  return Object.freeze({
    url: `http://${options.host ?? '127.0.0.1'}:${address.port}/v1`,
    config: Object.freeze({ id: ID, host: options.host ?? '127.0.0.1', port: address.port,
      providerID, modelID, modelAlias: modelName, maxRequestMs, timeoutMs, maxCliTurns, contextTokens,
      retained_default_tool_ids: defaultToolIds,
      route: 'official OpenCode CLI run --attach', stream: 'buffered only', nativeProviderToolCalls: false,
      trainingAdmission: false }),
    async close() {
      if (closed) return;
      closed = true;
      closing = true;
      await boundedKill(activeChild);
      await new Promise(resolve => {
        const timer = setTimeout(() => { server.closeAllConnections(); resolve(); }, 1500);
        server.close(() => { clearTimeout(timer); resolve(); });
      });
    }
  });
}
