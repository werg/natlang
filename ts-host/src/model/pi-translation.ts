/**
 * natlang's model turn <-> Pi (pi-ai) translation, as pure functions over STRUCTURAL Pi types.
 *
 * Nothing here imports pi-ai or a platform module: the Pi types below declare only the fields natlang reads or writes,
 * so one translation serves every Pi instance a host brings. Upstream `@earendil-works/pi-ai` (the Node CLI's) and
 * Vibestudio's fork `@panticonic/pi-ai` both satisfy them (test/pi-translation.test.mjs checks both shapes; the fork
 * adds fields and one stream event, `prompt_progress`, which natlang ignores). The backend that owns credentials, model
 * selection and the stream (pi-provider.ts, a gateway) calls these and stays free of translation logic.
 *
 * Values returned for Pi (`PiContext`, `PiMessage`) are assignable to pi-ai's `Context` and `Message`, except that a
 * tool's `parameters` is a JSON schema where Pi types a TypeBox schema: the caller's one cast at the boundary.
 */
import { modelTools } from './chat-completion.js';
import type { ModelTurn, ModelTurnDelta, ModelTurnRequest } from '../contracts.js';

export type PiJsonValue = null | boolean | number | string | readonly PiJsonValue[] | { [key: string]: PiJsonValue };
export type PiJsonObject = { [key: string]: PiJsonValue };

export type PiTextBlock = { type: 'text'; text: string };
export type PiThinkingBlock = { type: 'thinking'; thinking: string };
export type PiToolCallBlock = { type: 'toolCall'; id: string; name: string; arguments: PiJsonObject; thoughtSignature?: string };
export type PiContentBlock = PiTextBlock | PiThinkingBlock | PiToolCallBlock;

/** Token usage in Pi's shape; natlang reads `input` and `output` and passes the whole value through to the trace. */
export type PiUsage = { input: number; output: number; cacheRead: number; cacheWrite: number; totalTokens: number;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number } };

export type PiMessage =
  | { role: 'system'; content: string; timestamp: number }
  | { role: 'user'; content: string; timestamp: number }
  | { role: 'assistant'; content: PiContentBlock[]; api: string; provider: string; model: string; usage: PiUsage;
    stopReason: 'stop' | 'toolUse'; timestamp: number }
  | { role: 'toolResult'; toolCallId: string; toolName: string; content: PiTextBlock[]; isError: false; timestamp: number };
export type PiTool = { name: string; description: string; parameters: object };
export type PiContext = { messages: PiMessage[]; tools: PiTool[] };

/** What natlang reads of a completed Pi assistant message (pi-ai's `AssistantMessage` is assignable to it). */
export type PiAssistantMessage = { content: readonly (PiContentBlock | { type: string })[]; stopReason: string;
  usage: { input: number; output: number }; errorMessage?: string };
/** The Pi stream events natlang reads (a subset of pi-ai's `AssistantMessageEvent`; other events are ignored). */
export type PiStreamEvent = { type: string; contentIndex?: number; delta?: string;
  partial?: { content: readonly (PiContentBlock | { type: string })[] } };

export const emptyPiUsage: PiUsage = Object.freeze({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
  cost: Object.freeze({ input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }) });

/**
 * A request's messages and tools as a Pi context for `model` (its API decides how history is carried: the Google APIs
 * take host-generated calls as user notes, since they do not accept calls the model never made).
 */
export function toPiContext(request: ModelTurnRequest, model: { api: string; provider: string; id: string }): PiContext {
  const messages: PiMessage[] = [];
  const names = new Map<string, string>();
  const hostGeneratedCalls = new Set<string>();
  const googleHistory = model.api === 'google-generative-ai' || model.api === 'google-vertex';
  for (const raw of request.messages as Record<string, unknown>[]) {
    const role = raw.role, content = String(raw.content ?? '');
    const timestamp = Date.now();
    if (role === 'system') messages.push({ role, content, timestamp });
    else if (role === 'user') messages.push({ role, content, timestamp });
    else if (role === 'assistant') {
      const calls = (raw.tool_calls ?? []) as Array<{ id: string; function: { name: string; arguments: string };
        thought_signature?: string }>;
      const hostGenerated = googleHistory && raw.natlang_host_generated === true;
      if (hostGenerated) {
        const notes: string[] = [];
        if (content) notes.push(content);
        const reasoning = raw.reasoning_content ?? raw.reasoning;
        if (typeof reasoning === 'string' && reasoning) notes.push(reasoning);
        for (const call of calls) {
          names.set(call.id, call.function.name);
          hostGeneratedCalls.add(call.id);
          notes.push(`Host-provided operation ${call.function.name} arguments: ${call.function.arguments || '{}'}`);
        }
        if (notes.length) messages.push({ role: 'user', content: notes.join('\n'), timestamp });
        continue;
      }
      const blocks: PiContentBlock[] = [];
      const reasoning = raw.reasoning_content ?? raw.reasoning;
      if (typeof reasoning === 'string' && reasoning) blocks.push({ type: 'thinking', thinking: reasoning });
      if (content) blocks.push({ type: 'text', text: content });
      for (const call of calls) {
        names.set(call.id, call.function.name);
        blocks.push({ type: 'toolCall', id: call.id, name: call.function.name,
          arguments: JSON.parse(call.function.arguments || '{}') as PiJsonObject,
          ...(typeof call.thought_signature === 'string' ? { thoughtSignature: call.thought_signature } : {}) });
      }
      messages.push({ role, content: blocks, api: model.api, provider: model.provider, model: model.id,
        usage: emptyPiUsage, stopReason: calls.length ? 'toolUse' : 'stop', timestamp });
    } else if (role === 'tool') {
      const callId = String(raw.tool_call_id);
      const toolName = names.get(callId) ?? 'unknown';
      if (googleHistory && hostGeneratedCalls.has(callId))
        messages.push({ role: 'user', content: `Host-provided result for ${toolName}: ${content}`, timestamp });
      else messages.push({ role: 'toolResult', toolCallId: callId,
        toolName, content: [{ type: 'text', text: content }], isError: false, timestamp });
    }
  }
  const tools: PiTool[] = modelTools(request.tools).map(tool => ({ name: tool.function.name,
    description: String(tool.function.description ?? ''), parameters: tool.function.parameters as object }));
  return { messages, tools };
}

/**
 * The Pi stream option that makes the reply a tool call (`tool_choice: 'required'`), or none: Pi's neutral
 * `toolChoice` knows only auto and none, so each API takes its own word for "must call a tool" (`any` or `required`).
 */
export function piToolChoiceOptions(api: string, request: Pick<ModelTurnRequest, 'tool_choice'>): { toolChoice?: 'any' | 'required' } {
  if (request.tool_choice !== 'required') return {};
  if (['anthropic-messages', 'bedrock-converse-stream', 'google-generative-ai', 'google-vertex'].includes(api))
    return { toolChoice: 'any' };
  if (['openai-completions', 'openai-responses', 'openai-codex-responses', 'azure-openai-responses',
    'mistral-conversations', 'pi-messages'].includes(api)) return { toolChoice: 'required' };
  return {};
}

/**
 * Translate a turn's Pi stream events into `ModelTurnDelta`s (contracts.ts): text, reasoning (a later thinking part
 * after a newline, as the turn joins them), and each tool call by its order among the calls. Stateful: use one
 * translator per turn. Returns undefined for events that carry nothing to show.
 */
export function piDeltaTranslator(): (event: PiStreamEvent) => ModelTurnDelta | undefined {
  const callOrder = new Map<number, number>();
  let thinkingParts = 0;
  return event => {
    let delta: ModelTurnDelta | undefined;
    if (event.type === 'text_delta') delta = { type: 'text', text: event.delta! };
    else if (event.type === 'thinking_start') delta = thinkingParts++ ? { type: 'reasoning', text: '\n' } : undefined;
    else if (event.type === 'thinking_delta') delta = { type: 'reasoning', text: event.delta! };
    else if (event.type === 'toolcall_start') {
      const index = callOrder.size;
      callOrder.set(event.contentIndex!, index);
      const block = event.partial?.content[event.contentIndex!];
      const call = block?.type === 'toolCall' ? block as PiToolCallBlock : undefined;
      return { type: 'tool_call', index, ...(call?.id ? { id: call.id } : {}), ...(call?.name ? { name: call.name } : {}) };
    } else if (event.type === 'toolcall_delta' && callOrder.has(event.contentIndex!))
      return { type: 'tool_call', index: callOrder.get(event.contentIndex!)!, arguments: event.delta! };
    return delta && (delta.type !== 'text' && delta.type !== 'reasoning' || delta.text) ? delta : undefined;
  };
}

/** The parts of a completed Pi assistant message: its tool calls (with the OpenAI-shaped `raw_calls`), text and reasoning. */
export function piAssistantContent(reply: PiAssistantMessage) {
  const calls = reply.content.filter((block): block is PiToolCallBlock => block.type === 'toolCall');
  const text = reply.content.filter((block): block is PiTextBlock => block.type === 'text').map(block => block.text).join('');
  const reasoning = reply.content.filter((block): block is PiThinkingBlock => block.type === 'thinking').map(block => block.thinking).join('\n');
  const rawCalls = calls.map(call => ({ id: call.id, type: 'function', function: { name: call.name,
    arguments: JSON.stringify(call.arguments) },
  ...(typeof call.thoughtSignature === 'string' ? { thought_signature: call.thoughtSignature } : {}) }));
  return { calls, text, reasoning, rawCalls };
}

/**
 * A completed Pi assistant message as the model turn natlang's interpreter reads. A reply that ended in `error` or
 * `aborted` is a failed turn: it throws, carrying Pi's message (`provider` names the backend in the fallback text).
 * `raw_response` holds Pi's stop reason and usage; a backend adds its own observations to it.
 */
export function piAssistantToTurn(reply: PiAssistantMessage, provider: string): ModelTurn {
  if (reply.stopReason === 'error' || reply.stopReason === 'aborted')
    throw new Error(reply.errorMessage ?? `provider ${provider} failed`);
  const { calls, text, reasoning, rawCalls } = piAssistantContent(reply);
  return { calls: calls.map(call => [call.name, call.arguments as Record<string, unknown>]), raw_calls: rawCalls, text,
    ...(reasoning ? { reasoning } : {}), prompt_tokens: reply.usage.input, completion_tokens: reply.usage.output,
    ...(reply.stopReason === 'length' ? { truncated: true } : {}),
    raw_response: { stop_reason: reply.stopReason, usage: reply.usage } };
}
