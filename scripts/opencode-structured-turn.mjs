/**
 * A contained OpenCode SDK adapter for Natlang's ModelTurn contract.
 *
 * OpenCode owns the provider request and may run its own agent loop. This
 * adapter asks for a JSON text action envelope, validates it locally, and maps
 * it to Natlang calls. It does not claim provider-enforced JSON Schema or
 * native provider tool-call fidelity.
 * Natlang remains responsible for executing those mapped calls.
 */

const BRIDGE_ID = 'opencode-session-prompt-json-text-action-bridge/3';

const SYSTEM_INSTRUCTIONS = [
  'You are returning one response for a Natlang model turn.',
  'The user message contains the complete serialized Natlang request, including its ordered message history and tool schemas.',
  'Treat that serialized history as the conversation context. Do not call OpenCode tools or take external actions. The SDK denies every OpenCode tool except its no-I/O invalid-call rejection handler.',
  'Return exactly one JSON text object matching the response_schema included in the user payload. Do not add markdown, fences, commentary, or extra keys.',
  'The object contains content, which is the assistant text, and toolCalls, which contains only declared Natlang tool calls for the host to execute.',
  'This is prompt-directed JSON text, not provider-enforced JSON Schema output and not native provider tool-call output. The Natlang host validates and executes toolCalls, and owns the next turn.'
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
 * Disable every tool in the official server inventory, except its pure rejection handler.
 * OpenCode v1.18.35 applies these booleans as session permission rules and
 * resolves `*` through its wildcard matcher, so tools registered after the
 * inventory lookup remain denied too. Pinned source, tag commit
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
  if (!unique.includes('invalid')) throw new TypeError('OpenCode tool inventory omits its invalid-call rejection handler');
  return Object.fromEntries([['*', false], ...unique.filter(id => id !== 'invalid').map(id => [id, false]), ['invalid', true]]);
}

/** Build the official SDK session.prompt input; native OpenCode tools are controlled separately by its tools map. */
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
  if (toolChoice === 'required' && calls.length === 0)
    throw new TypeError('Natlang required a tool call, but OpenCode returned none');
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

function auditOpenCodeTools(messages, allowedNames) {
  const parts = (Array.isArray(messages) ? messages : [])
    .flatMap(message => Array.isArray(message?.parts) ? message.parts : [])
    .filter(part => part?.type === 'tool');
  const rejectedNatlangAttempts = [];
  const externalActions = [];
  for (const part of parts) {
    const rejected = rejectedNatlangToolAttempt(part, allowedNames);
    if (rejected) rejectedNatlangAttempts.push(rejected);
    else externalActions.push(part);
  }
  return { externalActions, rejectedNatlangAttempts, toolParts: parts };
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

function extractAssistantResult(data, allowedNames) {
  const info = data.info;
  if (!info || typeof info !== 'object') throw new Error('OpenCode session prompt returned no assistant message');
  if (typeof info.id !== 'string' || !info.id)
    throw new Error('OpenCode session prompt returned an assistant message without an id');
  if (info.error) {
    const name = typeof info.error.name === 'string' ? info.error.name : 'assistant error';
    const error = new Error(`OpenCode assistant turn failed (${name})`);
    if (name === 'APIError') {
      const data = info.error.data;
      if (Number.isSafeInteger(data?.statusCode)) error.providerStatusCode = data.statusCode;
      if (typeof data?.isRetryable === 'boolean') error.providerRetryable = data.isRetryable;
      error.code = 'OPENCODE_PROVIDER_ERROR';
    }
    throw error;
  }
  const parts = Array.isArray(data.parts) ? data.parts : [];
  const { externalActions } = auditOpenCodeTools([{ parts }], allowedNames);
  if (externalActions.length) {
    const names = [...new Set(externalActions.map(part => String(part.tool ?? 'unknown')))];
    throw new Error(`OpenCode session contains non-bridge tool part(s) during a Natlang turn: ${names.join(', ')}`);
  }
  const text = parts.filter(part => part?.type === 'text').map(part => part.text).join('');
  if (!text) throw new Error('OpenCode assistant message has no JSON text output');
  let structured;
  try { structured = JSON.parse(text); }
  catch { throw new Error('OpenCode assistant text is not one valid JSON action object'); }
  return { structured, info, parts, text };
}

/**
 * Create an experimental single-turn backend around an official OpenCode v2 SDK
 * client. Each request gets an isolated session; cleanup runs on success,
 * provider error, and abort. Before prompting, the official SDK tool inventory
 * is read and every listed native tool is disabled except the pure `invalid`
 * rejection handler.
 */
export function createOpenCodeStructuredTurnBackend({ client, providerID, modelID, agent, directory,
  cleanupTimeoutMs = 2_000 } = {}) {
  if (!client?.tool?.ids || !client?.session?.create || !client?.session?.prompt || !client?.session?.messages || !client?.session?.delete)
    throw new TypeError('an OpenCode SDK v2 client with tool inventory and session create, prompt, messages, and delete is required');
  if (typeof providerID !== 'string' || !providerID || typeof modelID !== 'string' || !modelID)
    throw new TypeError('providerID and modelID are required');
  if (typeof directory !== 'string' || !directory.startsWith('/'))
    throw new TypeError('an absolute isolated OpenCode scratch directory is required');
  if (!Number.isSafeInteger(cleanupTimeoutMs) || cleanupTimeoutMs < 1)
    throw new RangeError('cleanupTimeoutMs must be a positive integer');

  return async function turn(request, signal) {
    signal?.throwIfAborted();
    const prompt = buildOpenCodeStructuredPrompt(request, { providerID, modelID, agent });
    const toolInventoryResult = await withAbort(client.tool.ids({ directory }, { ...(signal ? { signal } : {}) }),
      signal, 'OpenCode tool inventory');
    const openCodeToolIds = unwrapSdkResult(toolInventoryResult, 'tool inventory');
    const openCodeTools = buildOpenCodeToolPolicy(openCodeToolIds);
    const createResult = await withAbort(client.session.create({
      model: { providerID, id: modelID }, ...(agent ? { agent } : {}), directory,
      title: 'Natlang JSON text action turn'
    }, { ...(signal ? { signal } : {}) }), signal, 'OpenCode session create');
    const session = unwrapSdkResult(createResult, 'session create');
    if (typeof session.id !== 'string' || !session.id) throw new Error('OpenCode session create returned no session id');

    let primaryError;
    try {
      signal?.throwIfAborted();
      const promptResult = await withAbort(client.session.prompt({
        sessionID: session.id, directory, model: { providerID, modelID }, agent,
        system: prompt.body.system, parts: prompt.body.parts, tools: openCodeTools
      }, { ...(signal ? { signal } : {}) }), signal, 'OpenCode session prompt');
      const data = unwrapSdkResult(promptResult, 'session prompt');
      const allowedNames = new Set(prompt.toolNames);
      const response = extractAssistantResult(data, allowedNames);
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
        throw new Error(`OpenCode session contains non-bridge tool part(s): ${names.join(', ')}`);
      }
      const parsed = parseStructuredTurn(response.structured, new Set(prompt.toolNames), request.tool_choice);
      const usageAudit = assistantUsageAudit(history, { finalMessageId: response.info.id, finalStructured: response.structured });
      return {
        ...(parsed.calls.length ? { calls: parsed.calls } : {}),
        ...(parsed.text ? { text: parsed.text } : {}),
        ...(usageAudit.inputTokens === undefined ? {} : { prompt_tokens: usageAudit.inputTokens }),
        ...(usageAudit.outputTokens === undefined ? {} : { completion_tokens: usageAudit.outputTokens }),
        raw_response: {
          transport: BRIDGE_ID,
          provider_id: providerID,
          model_id: modelID,
          session_id: session.id,
          fidelity: 'prompt-directed-strict-json-text; not provider-enforced JSON Schema or native provider tool-call output',
          output_contract: 'exact JSON text parsed and validated by the bridge',
          open_code_tool_policy: {
            control: 'official SDK session.prompt tools boolean map',
            inventory: 'official SDK tool.ids endpoint, queried for this request',
            inventory_ids: openCodeToolIds,
            enabled_ids: ['invalid'],
            disabled_wildcard: '*',
            all_other_inventory_ids_disabled: true,
            future_tool_ids_remain_disabled: true
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
      if (signal?.aborted && client.session.abort) {
        try { await withTimeout(client.session.abort({ sessionID: session.id, directory }),
          cleanupTimeoutMs, 'OpenCode abort cleanup'); }
        catch { /* preserve the original provider/abort failure */ }
      }
      throw error;
    } finally {
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
