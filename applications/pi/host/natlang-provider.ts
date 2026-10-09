/**
 * The natlang transport for pi's agent model: a pi-ai provider (`ProviderStreams`) that reaches the model through
 * natlang's own model-turn drivers (ts-host chat-completion.ts, and neuralese-server.ts for a Neuralese server), so the
 * agent speaks the wire standard (spec/NEURALESE_PORT.md "Wire protocol", ts-host contracts.ts) exactly as natlang
 * programs do. pi-ai keeps its transcript hygiene (`transformMessages`, the replayed system prompt and tools); the
 * request, the content parts, tool-call decoding and token accounting are natlang's.
 *
 * Representation is chosen by use (plans/neuralese/DECISIONS.md 2026-10-09): the agent model declares its reader at
 * startup (`declareReader`: text, or a Neuralese dialect the server's `/v1/neuralese/info` confirms) on the model
 * object as `reader`, so code that produces a value for the agent picks its representation before making it. A
 * Neuralese part (types.ts `NeuraleseContent`) goes to the wire unchanged; sent to a driver that does not carry parts
 * it fails the request with neuralese-unsupported-backend. There is no text fallback.
 */
import { AssistantMessageEventStream, calculateCost, collapseSystemMessages, getCurrentTools, getSystemMessageText,
  lazyStream, uuidv7 } from '@earendil-works/pi-ai';
import type { Api, AssistantMessage, AssistantMessageEvent, Message, Model, ProviderStreams, SimpleStreamOptions,
  StreamOptions, TranscriptContext, ToolCall } from '@earendil-works/pi-ai';
import { transformMessages } from '@earendil-works/pi-ai/api/transform-messages';
import { checkNeuraleseReader, NeuraleseUnsupportedError, supportsNeuralese, textToParts, hasNeuraleseSentinel,
  type ModelContentPart, type ModelTurn, type ModelTurnRequest } from '@natlang/node';
import type { NeuraleseContent } from '../types.ts';

/** A natlang model-turn driver; one that carries content parts sets `neuralese: true` (ts-host contracts.ts). */
export type ModelDriver = ((request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn>) & { neuralese?: boolean };

/** What the agent model reads: text, or the blocks of one Neuralese dialect (at most `maxBlockLength` positions each). */
export type AgentReader = { kind: 'text' } | { kind: 'neuralese'; dialect: string; maxBlockLength?: number };

/** The driver for a model; `reasoning`: the request asks for thinking (pi's thinking level is not "off"). */
export type NatlangDrivers = (model: Model<Api>, options: { reasoning: boolean }) => ModelDriver;

/**
 * Tokens a Neuralese block is estimated at where its length is not known: the reference server's default maximum
 * block length (training/neuralese serve/engine.py `max_block`), an upper bound there.
 */
export const NEURALESE_BLOCK_TOKENS = 64;

/** Whether a content part is a Neuralese block. */
export function isNeuraleseContent(part: unknown): part is NeuraleseContent {
  const found = part as { type?: unknown; id?: unknown } | null;
  return found?.type === 'neuralese' && typeof found.id === 'string';
}

/** The Neuralese blocks a message carries. */
export function neuraleseBlocks(message: { content?: unknown }): number {
  return Array.isArray(message.content) ? message.content.filter(isNeuraleseContent).length : 0;
}

/** A content part in a transcript for a text reader: a block is named, never paraphrased. */
export function transcriptText(part: { type: string; text?: string; id?: string }): string {
  return part.type === 'text' ? part.text ?? '' : isNeuraleseContent(part) ? `[Neuralese block ${part.id}]` : `[${part.type}]`;
}

/** The reader a model declares; a model without a declaration reads text. */
export function modelReader(model: Model<Api> | undefined): AgentReader {
  return (model as { reader?: AgentReader } | undefined)?.reader ?? { kind: 'text' };
}

/**
 * The reader `spec` declares ("text", or a dialect name), checked against the server at `endpoint`: a dialect it does
 * not speak is a startup error (neuralese-dialect-mismatch), as is a server without Neuralese
 * (neuralese-unsupported-backend).
 */
export async function declareReader(spec: string, endpoint: string, options: { apiKey?: string } = {}): Promise<AgentReader> {
  if (spec === 'text') return { kind: 'text' };
  const info = await checkNeuraleseReader(endpoint, spec, options);
  return { kind: 'neuralese', dialect: spec,
    ...(typeof info.max_block_length === 'number' ? { maxBlockLength: info.max_block_length } : {}) };
}

type WireContent = string | ModelContentPart[];
type Part = { type: string; text?: string; thinking?: string; redacted?: boolean };

/**
 * Message content on the wire: text alone is a string (its parts joined by `separator`, as pi-ai's OpenAI provider
 * joins them); content with a Neuralese part is the part array, each block unchanged at its position.
 */
function wireContent(content: string | readonly Part[], separator: string): WireContent {
  if (typeof content === 'string') return content;
  for (const part of content) if (part.type !== 'text' && !isNeuraleseContent(part))
    throw new Error(`the natlang transport carries text and Neuralese content, not ${part.type} parts`);
  if (!content.some(isNeuraleseContent)) return content.map(part => part.text ?? '').join(separator);
  return content.flatMap((part): ModelContentPart[] => isNeuraleseContent(part) ? [part] :
    part.text ? [{ type: 'text', text: part.text }] : []);
}

const isEmpty = (content: WireContent) => content.length === 0;

/** pi-ai's transcript as a natlang model-turn request; `carriesBlocks`: some message holds a Neuralese part. */
export function modelTurnRequest(model: Model<Api>, context: TranscriptContext, options: SimpleStreamOptions = {}):
    { request: ModelTurnRequest; carriesBlocks: boolean } {
  if (options.toolChoice === 'none') throw new Error('the natlang transport has no tool choice "none": offer no tools instead');
  const collapsed = collapseSystemMessages(context);
  const tools = getCurrentTools(collapsed.messages).map(tool => ({ type: 'function',
    function: { name: tool.name, description: tool.description, parameters: tool.parameters } }));
  const messages: Record<string, unknown>[] = [];
  let carriesBlocks = false;
  for (const message of transformMessages(collapsed.messages, model) as Message[]) {
    if (neuraleseBlocks(message)) carriesBlocks = true;
    if (message.role === 'system') {
      const text = getSystemMessageText(message);
      if (text) messages.push({ role: 'system', content: text });
    } else if (message.role === 'user') {
      const content = wireContent(message.content, '\n');
      if (!isEmpty(content)) messages.push({ role: 'user', content });
    } else if (message.role === 'assistant') {
      const parts = message.content as readonly (Part | ToolCall)[];
      const reasoning = parts.flatMap(part => part.type === 'thinking' && !(part as Part).redacted &&
        (part as Part).thinking?.trim() ? [(part as Part).thinking!] : []).join('\n');
      const calls = parts.filter((part): part is ToolCall => part.type === 'toolCall');
      const content = wireContent(parts.filter(part => part.type === 'text' || isNeuraleseContent(part)) as Part[], '');
      if (isEmpty(content) && !calls.length) continue;
      messages.push({ role: 'assistant', content, ...(reasoning ? { reasoning_content: reasoning } : {}),
        ...(calls.length ? { tool_calls: calls.map(call => ({ id: call.id, type: 'function',
          function: { name: call.name, arguments: JSON.stringify(call.arguments) } })) } : {}) });
    } else if (message.role === 'toolResult') {
      const content = wireContent(message.content, '\n');
      messages.push({ role: 'tool', tool_call_id: message.toolCallId, content: isEmpty(content) ? '(no tool output)' : content });
    }
  }
  return { carriesBlocks, request: { messages, tools, tool_choice: 'auto', seed: null,
    max_tokens: options.maxTokens ?? (model.maxTokens || null),
    ...(options.temperature === undefined ? {} : { temperature: options.temperature }) } };
}

/** A reply's text as pi content: text, with each block a Neuralese part where the server wrote one. */
function replyContent(text: string): (AssistantMessage['content'][number] | NeuraleseContent)[] {
  if (!hasNeuraleseSentinel(text)) return text ? [{ type: 'text', text }] : [];
  return textToParts(text).filter(part => part.type !== 'text' || part.text)
    .map((part): AssistantMessage['content'][number] | NeuraleseContent =>
      part.type === 'neuralese' ? { type: 'neuralese', id: part.id } : { type: 'text', text: part.text });
}

/**
 * One model turn as pi-ai's event stream. natlang's drivers return the whole turn, so its parts are emitted in order
 * once it arrives: each text and thinking part as start, one delta, end; each call as start, its arguments as one
 * delta, end. A Neuralese part has no pi-ai event; it is in the partial and final message at its position.
 */
async function* turnEvents(model: Model<Api>, driver: ModelDriver, context: TranscriptContext,
    options: SimpleStreamOptions): AsyncGenerator<AssistantMessageEvent> {
  const output: AssistantMessage = { role: 'assistant', content: [], api: model.api, provider: model.provider, model: model.id,
    usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
    stopReason: 'stop', timestamp: Date.now() };
  yield { type: 'start', partial: output };
  try {
    const { request, carriesBlocks } = modelTurnRequest(model, context, options);
    if (carriesBlocks && !supportsNeuralese(driver))
      throw new NeuraleseUnsupportedError(`the conversation holds Neuralese blocks, but the transport of ` +
        `${model.provider}/${model.id} does not carry content parts (its reader is ${modelReader(model).kind})`);
    const turn = await driver(request, options.signal);
    const content = output.content as (AssistantMessage['content'][number] | NeuraleseContent)[];
    if (turn.reasoning) {
      const index = content.push({ type: 'thinking', thinking: '' }) - 1;
      yield { type: 'thinking_start', contentIndex: index, partial: output };
      (content[index] as { thinking: string }).thinking = turn.reasoning;
      yield { type: 'thinking_delta', contentIndex: index, delta: turn.reasoning, partial: output };
      yield { type: 'thinking_end', contentIndex: index, content: turn.reasoning, partial: output };
    }
    for (const part of replyContent(turn.text ?? '')) {
      if (part.type !== 'text') { content.push(part); continue; }
      const index = content.push({ type: 'text', text: '' }) - 1;
      yield { type: 'text_start', contentIndex: index, partial: output };
      (content[index] as { text: string }).text = part.text;
      yield { type: 'text_delta', contentIndex: index, delta: part.text, partial: output };
      yield { type: 'text_end', contentIndex: index, content: part.text, partial: output };
    }
    const raw = (turn.raw_calls ?? []) as { id?: unknown }[];
    for (const [i, [name, args]] of (turn.calls ?? []).entries()) {
      const id = typeof raw[i]?.id === 'string' && raw[i]!.id ? raw[i]!.id as string : `call_${uuidv7()}`;
      const call: ToolCall = { type: 'toolCall', id, name, arguments: {} };
      const index = content.push(call) - 1;
      yield { type: 'toolcall_start', contentIndex: index, partial: output };
      call.arguments = args as ToolCall['arguments'];
      yield { type: 'toolcall_delta', contentIndex: index, delta: JSON.stringify(args), partial: output };
      yield { type: 'toolcall_end', contentIndex: index, toolCall: call, partial: output };
    }
    output.usage.input = turn.prompt_tokens ?? 0;
    output.usage.output = turn.completion_tokens ?? 0;
    output.usage.totalTokens = output.usage.input + output.usage.output;
    output.usage.cost = calculateCost(model, output.usage);
    if (typeof turn.raw_response?.id === 'string') output.responseId = turn.raw_response.id;
    output.stopReason = turn.truncated ? 'length' : turn.calls?.length ? 'toolUse' : 'stop';
    yield { type: 'done', reason: output.stopReason as 'stop' | 'length' | 'toolUse', message: output };
  } catch (error) {
    // The driver's message stays whole: its standard code (neuralese-unsupported-backend, …) and the server's own
    // words, which pi-ai's overflow and retry classification read.
    output.stopReason = options.signal?.aborted ? 'aborted' : 'error';
    output.errorMessage = error instanceof Error ? error.message : String(error);
    yield { type: 'error', reason: output.stopReason, error: output };
  }
}

/** The natlang transport as pi-ai's chat API: `drivers` gives each model's driver. */
export function natlangApi(drivers: NatlangDrivers): ProviderStreams {
  const stream = (model: Model<Api>, context: TranscriptContext, options: SimpleStreamOptions = {}): AssistantMessageEventStream =>
    lazyStream(model, async () => turnEvents(model, drivers(model, { reasoning: Boolean(options.reasoning) }), context, options));
  return { stream: (model, context, options?: StreamOptions) => stream(model, context, options), streamSimple: stream };
}
