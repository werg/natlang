// Text the model actually received before this child decision. The native
// materializer keeps the current action in `turn.target`, separately from
// the durable opening and prior messages in `turn.messages`; preceding
// assistant tool calls are part of the actual conversation context.
export function visibleTurnText(turn) {
  const visibleToolCallText = call => {
    const raw = call.function?.arguments ?? '';
    if (typeof raw !== 'string' || !raw) return '';
    // Tool arguments are transported as JSON. Inspect both the exact wire
    // text and decoded argument values so captured source values embedded in
    // earlier eval code are checked as the model saw them.
    let decoded = '';
    try {
      const value = JSON.parse(raw);
      const leaves = [];
      const visit = child => {
        if (typeof child === 'string') leaves.push(child);
        else if (Array.isArray(child)) child.forEach(visit);
        else if (child && typeof child === 'object') Object.values(child).forEach(visit);
      };
      visit(value);
      decoded = leaves.join('\n');
    } catch {}
    return `${raw}\n${decoded}`;
  };
  return (turn.messages ?? []).map(message => `${message.content ?? ''}\n${(message.tool_calls ?? [])
    .map(visibleToolCallText).join('\n')}`).join('\n');
}

function decodedArguments(call) {
  const raw = call?.function?.arguments;
  if (typeof raw !== 'string') return raw;
  try { return JSON.parse(raw); } catch { return undefined; }
}

function sameActionCalls(step, turn) {
  const modelCalls = turn.target?.tool_calls ?? [];
  const recordedCalls = step.assistant?.calls ?? [];
  if (modelCalls.length === 0 && recordedCalls.length === 0)
    return turn.target?.role === (step.assistant?.role ?? 'assistant') &&
      typeof turn.target?.content === 'string' && turn.target.content === step.assistant?.content;
  return modelCalls.length > 0 && modelCalls.length === recordedCalls.length &&
    modelCalls.every((call, index) => {
      const recorded = recordedCalls[index];
      const args = decodedArguments(call);
      return args !== undefined && call?.function?.name === recorded?.tool &&
        canonical(args) === canonical(recorded?.arguments);
    });
}

// Bind visibility to the exact request rendered for the same recorded action.
// Materialized turn.messages can be reconstructed from pre-render context and
// therefore contain values that the provider request later omitted or clipped.
export function providerRenderedTurn(turn, row) {
  const invocationId = turn?.source_ref?.invocation_id;
  if (typeof invocationId !== 'string' || !invocationId)
    return { text: null, reason: 'missing_turn_invocation_id' };
  const matches = (row?.trajectory ?? []).filter(step => step.phase === 'action' &&
    step.invocation_id === invocationId && sameActionCalls(step, turn));
  if (matches.length !== 1)
    return { text: null, reason: matches.length ? 'ambiguous_same_invocation_provider_action' :
      'same_invocation_provider_action_not_found' };
  const provenance = matches[0].model_response?.transport_provenance;
  const renderedMessages = provenance?.rendered_messages;
  if (!Array.isArray(renderedMessages) || !renderedMessages.every(message =>
      message && typeof message === 'object' && typeof message.role === 'string'))
    return { text: null, reason: 'same_invocation_rendered_messages_missing_or_malformed' };
  return { text: visibleTurnText({ messages: renderedMessages }), renderedMessages,
    renderedRequestSha256: provenance.rendered_request_sha256 ?? null,
    requestSha256: matches[0].request_sha256 ?? null };
}

function compactJson(text) {
  let out = '', quoted = false, escaped = false;
  for (const char of text) {
    if (quoted) {
      out += char;
      if (escaped) escaped = false;
      else if (char === '\\') escaped = true;
      else if (char === '"') quoted = false;
    } else if (char === '"') {
      quoted = true;
      out += char;
    } else if (!/\s/.test(char)) out += char;
  }
  return out;
}

// Structured source fields are visible only when every value appears in the
// exact serialized structure. A compact display placeholder is insufficient.
export function visibleValue(value, text) {
  if (typeof text !== 'string') return false;
  if (typeof value === 'string') return text.includes(value) || text.includes(JSON.stringify(value));
  const serialized = JSON.stringify(value);
  return compactJson(text).includes(compactJson(serialized));
}
import { canonical } from '../../ts-host/dist/adaptation/identity.js';
