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
  type ModelContentPart, type ModelTurn, type ModelTurnDelta, type ModelTurnOptions, type ModelTurnRequest,
  type NeuraleseBlockMeta, type NeuraleseStore } from '@natlang/node';
import { parseStreamingJson } from '@earendil-works/pi-ai/utils/json-parse';
import type { NeuraleseContent } from '../types.ts';

/**
 * A natlang model-turn driver; one that carries content parts sets `neuralese: true` (ts-host contracts.ts). Its
 * `onDelta` option streams the turn's deltas.
 */
export type ModelDriver = ((request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => Promise<ModelTurn>) &
  { neuralese?: boolean };

/** What the agent model reads: text, or the blocks of one Neuralese dialect (at most `maxBlockLength` positions each). */
export type AgentReader = { kind: 'text' } | { kind: 'neuralese'; dialect: string; maxBlockLength?: number };

/** The driver for a model; `reasoning`: the request asks for thinking (pi's thinking level is not "off"). */
export type NatlangDrivers = (model: Model<Api>, options: { reasoning: boolean }) => ModelDriver;

/** Whether a content part is a Neuralese block. */
export function isNeuraleseContent(part: unknown): part is NeuraleseContent {
  const found = part as { type?: unknown; id?: unknown } | null;
  return found?.type === 'neuralese' && typeof found.id === 'string';
}

/** The Neuralese blocks a message carries. */
export function neuraleseBlocks(message: { content?: unknown }): number {
  return Array.isArray(message.content) ? message.content.filter(isNeuraleseContent).length : 0;
}

/** The IDs of the Neuralese blocks the messages carry, each once, in order. */
export function neuraleseBlockIds(messages: readonly { content?: unknown }[]): string[] {
  const ids = new Set<string>();
  for (const message of messages) if (Array.isArray(message.content))
    for (const part of message.content) if (isNeuraleseContent(part)) ids.add(part.id);
  return [...ids];
}

/**
 * The context positions a Neuralese block occupies: its length (the number of vectors), from its metadata in the
 * runtime's store (`peek`, no I/O). Metadata that is not there is an error naming the block and the fix, never a guess.
 */
export function blockLength(id: string, store: NeuraleseStore | undefined): number {
  if (!store?.peek) throw new Error(`neuralese-unknown-block-length: block ${id} cannot be measured: the runtime has no ` +
    'process-local Neuralese store to hold block metadata. Give the natlang runtime a Neuralese store (its neuralese.store ' +
    'option) so the blocks the agent model writes, and the metadata fetched for others, are recorded there.');
  const meta = store.peek(id);
  if (meta && Number.isSafeInteger(meta.length) && meta.length >= 0) return meta.length;
  throw new Error(`neuralese-unknown-block-length: the runtime's Neuralese store has no metadata for block ${id}, so the ` +
    'context positions it occupies are unknown. Blocks the agent model writes are recorded as they arrive, and context() ' +
    'fetches the metadata of the others from the agent model\'s server; for messages that did not come from context(), ' +
    'call await ai.blockMeta(messages) before estimating them. When ai.blockMeta returns this ID, neither the store nor ' +
    `the server (GET /v1/neuralese/blocks/${id}/meta) has the block: it is lost, and the message holding it must leave the ` +
    'conversation.');
}

/** The context positions of every Neuralese block a message carries (`blockLength` of each). */
export function blockPositions(message: { content?: unknown }, store: NeuraleseStore | undefined): number {
  if (!Array.isArray(message.content)) return 0;
  let positions = 0;
  for (const part of message.content) if (isNeuraleseContent(part)) positions += blockLength(part.id, store);
  return positions;
}

/** Metadata requests in flight, by server and block: concurrent callers share one request. */
const fetching = new Map<string, Promise<NeuraleseBlockMeta | undefined>>();

/**
 * Note in `store` the metadata of each block of `ids` it cannot `peek`, fetched once from the Neuralese server at
 * `root` (`GET /v1/neuralese/blocks/{id}/meta`, spec/NEURALESE_PORT.md). Returns the IDs the server does not have
 * either. A store that cannot note metadata (no `note`) is left as it is, and every block it lacks is returned.
 */
export async function noteBlockMeta(root: string, ids: readonly string[], store: NeuraleseStore,
    options: { signal?: AbortSignal; headers?: Record<string, string> } = {}): Promise<string[]> {
  const base = root.replace(/\/+$/, '').replace(/\/v1$/, '');
  const missing: string[] = [];
  await Promise.all([...new Set(ids)].map(async id => {
    if (store.peek?.(id)) return;
    if (!store.note) { missing.push(id); return; }
    const key = `${base}\0${id}`;
    let request = fetching.get(key);
    if (!request) {
      request = (async () => {
        const response = await fetch(`${base}/v1/neuralese/blocks/${encodeURIComponent(id)}/meta`,
          { headers: options.headers, signal: options.signal });
        if (response.status === 404) return undefined;
        if (!response.ok) throw new Error(`neuralese block metadata for ${id} failed (HTTP ${response.status}): ` +
          (await response.text()).slice(0, 500));
        return await response.json() as NeuraleseBlockMeta;
      })().finally(() => fetching.delete(key));
      fetching.set(key, request);
    }
    const meta = await request;
    if (meta && meta.id === id && Number.isSafeInteger(meta.length)) store.note(meta); else missing.push(id);
  }));
  return missing;
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

type Content = AssistantMessage['content'][number] | NeuraleseContent;

/**
 * The event a Neuralese part streams as: pi-ai's event union has no part type for it, so the transport adds one. The
 * block arrives whole, once the server has written it, and shows as a block reference (plans/STREAMING.md §1.5); a
 * consumer that does not know the event still finds the part in `partial` at `contentIndex`.
 */
export type NeuraleseEvent = { type: 'neuralese'; contentIndex: number; content: NeuraleseContent; partial: AssistantMessage };
/** What the natlang transport streams: pi-ai's events and the Neuralese part event. */
export type NatlangEvent = AssistantMessageEvent | NeuraleseEvent;

/** A turn's content as pi parts, in the order the final message holds them: thinking, the reply, the calls. */
function turnContent(turn: ModelTurn, streamedIds: (string | undefined)[]): Content[] {
  const content: Content[] = turn.reasoning ? [{ type: 'thinking', thinking: turn.reasoning }] : [];
  content.push(...replyContent(turn.text ?? ''));
  const raw = (turn.raw_calls ?? []) as { id?: unknown }[];
  for (const [i, [name, args]] of (turn.calls ?? []).entries()) {
    const id = typeof raw[i]?.id === 'string' && raw[i]!.id ? raw[i]!.id as string : streamedIds[i] || `call_${uuidv7()}`;
    content.push({ type: 'toolCall', id, name, arguments: args as ToolCall['arguments'] });
  }
  return content;
}

/**
 * The parts of one streamed turn in the partial message `output`, and the pi-ai events that build them.
 *
 * Deltas (ts-host contracts.ts `ModelTurnDelta`) map as they arrive: text to a text part and reasoning to a thinking
 * part (start, then a delta per piece; a part of another kind ends it), a tool-call fragment to the call of its index
 * (start at its first fragment, its argument text as deltas, the partial's arguments parsed as far as they go), a
 * Neuralese block to a `NeuraleseEvent`. Calls stay open until the turn ends.
 *
 * `reset` (the driver sends the request again: a malformed-call retry, or blocks restored on the server) replaces the
 * abandoned attempt within the same live attempt: its open parts end as they stand, the partial is emptied, and the
 * re-sent request's parts start again at content index 0. pi-ai's events cannot be retracted, and the abandoned
 * attempt is not a pi-durable attempt: that is the generation's retry (it counts against the retry policy, and
 * `convertPartial` would store the abandoned partial as an aborted assistant entry the model then reads). So pi.live
 * shows the re-sent request's output in place of the abandoned one at its next partial, and nothing of the abandoned
 * attempt is stored.
 *
 * `finish` takes the driver's final turn, which is authoritative: where the streamed parts are the start of its parts
 * (same kinds in order, an ended part equal, an open text or thinking part a prefix, a call of the same name), they
 * are completed and ended, and the rest follow whole; otherwise the streamed parts are replaced as on `reset`, and the
 * turn's parts follow whole. Either way the message `done` carries is exactly the final turn's.
 */
function streamedTurn(output: AssistantMessage, store?: NeuraleseStore) {
  const content = output.content as Content[];
  let current: number | undefined;
  /** Tool-call delta index -> content index, and each open call's argument text. */
  const calls = new Map<number, number>();
  const argumentText = new Map<number, string>();
  const ended = (index: number): NatlangEvent => {
    const part = content[index]!;
    if (part.type === 'text') return { type: 'text_end', contentIndex: index, content: part.text, partial: output };
    if (part.type === 'thinking') return { type: 'thinking_end', contentIndex: index, content: part.thinking, partial: output };
    argumentText.delete(index);
    return { type: 'toolcall_end', contentIndex: index, toolCall: part as ToolCall, partial: output };
  };
  /** End the open text or thinking part. */
  const endCurrent = (): NatlangEvent[] => {
    if (current === undefined) return [];
    const index = current;
    current = undefined;
    return [ended(index)];
  };
  /** End every open part, in content order. */
  const endAll = (): NatlangEvent[] => {
    const open = [...new Set([...(current === undefined ? [] : [current]), ...argumentText.keys()])].sort((a, b) => a - b);
    current = undefined;
    return open.map(ended);
  };
  const clear = (): NatlangEvent[] => {
    const events = endAll();
    content.length = 0;
    calls.clear();
    return events;
  };
  /** A part sent whole: its start, its content as one delta, its end. */
  const whole = (part: Content): NatlangEvent[] => {
    const index = content.push(part) - 1;
    if (part.type === 'text') return [{ type: 'text_start', contentIndex: index, partial: output },
      { type: 'text_delta', contentIndex: index, delta: part.text, partial: output }, ended(index)];
    if (part.type === 'thinking') return [{ type: 'thinking_start', contentIndex: index, partial: output },
      { type: 'thinking_delta', contentIndex: index, delta: part.thinking, partial: output }, ended(index)];
    if (part.type === 'toolCall') return [{ type: 'toolcall_start', contentIndex: index, partial: output },
      { type: 'toolcall_delta', contentIndex: index, delta: JSON.stringify(part.arguments), partial: output }, ended(index)];
    return [{ type: 'neuralese', contentIndex: index, content: part as NeuraleseContent, partial: output }];
  };
  /** The next piece of a text or thinking part. */
  const piece = (kind: 'text' | 'thinking', text: string): NatlangEvent[] => {
    const events: NatlangEvent[] = [];
    if (current === undefined || content[current]!.type !== kind) {
      events.push(...endCurrent());
      current = content.push(kind === 'text' ? { type: 'text', text: '' } : { type: 'thinking', thinking: '' }) - 1;
      events.push({ type: kind === 'text' ? 'text_start' : 'thinking_start', contentIndex: current, partial: output });
    }
    const part = content[current]!;
    if (part.type === 'text') part.text += text; else if (part.type === 'thinking') part.thinking += text;
    events.push({ type: kind === 'text' ? 'text_delta' : 'thinking_delta', contentIndex: current, delta: text, partial: output });
    return events;
  };
  /** Whether a streamed part is the start of the final part at its position. */
  const agrees = (index: number, part: Content, final: Content | undefined): boolean => {
    if (final?.type !== part.type) return false;
    const open = index === current || argumentText.has(index);
    if (part.type === 'text' || part.type === 'thinking') {
      const [streamed, wanted] = part.type === 'text' ? [part.text, (final as { text: string }).text] :
        [part.thinking, (final as { thinking: string }).thinking];
      return open ? wanted.startsWith(streamed) : wanted === streamed;
    }
    if (part.type === 'toolCall') return part.name === (final as ToolCall).name;
    return (part as NeuraleseContent).id === (final as NeuraleseContent).id;
  };
  return {
    delta(delta: ModelTurnDelta): NatlangEvent[] {
      if (delta.type === 'text') return piece('text', delta.text);
      if (delta.type === 'reasoning') return piece('thinking', delta.text);
      if (delta.type === 'reset') return clear();
      if (delta.type === 'neuralese') {
        // The block's metadata, when the server sends it, is recorded at once: the part stays a pure reference, and
        // its length (context positions) is read from the store.
        const meta = delta.block as NeuraleseBlockMeta | undefined;
        if (meta?.id === delta.part.id && Number.isSafeInteger(meta.length) && !store?.peek?.(meta.id)) store?.note?.(meta);
        return [...endCurrent(), ...whole({ ...delta.part })];
      }
      const events = endCurrent();
      let index = calls.get(delta.index);
      if (index === undefined) {
        index = content.push({ type: 'toolCall', id: '', name: '', arguments: {} }) - 1;
        calls.set(delta.index, index);
        argumentText.set(index, '');
        events.push({ type: 'toolcall_start', contentIndex: index, partial: output });
      }
      const call = content[index] as ToolCall;
      if (delta.id) call.id = delta.id;
      if (delta.name) call.name += delta.name;
      if (delta.arguments && argumentText.has(index)) {
        const text = argumentText.get(index)! + delta.arguments;
        argumentText.set(index, text);
        call.arguments = parseStreamingJson<ToolCall['arguments']>(text);
        events.push({ type: 'toolcall_delta', contentIndex: index, delta: delta.arguments, partial: output });
      }
      return events;
    },
    finish(turn: ModelTurn): NatlangEvent[] {
      const streamedIds = content.filter((part): part is ToolCall => part.type === 'toolCall').map(call => call.id);
      const final = turnContent(turn, streamedIds);
      if (!content.every((part, index) => agrees(index, part, final[index]))) return [...clear(), ...final.flatMap(whole)];
      const events: NatlangEvent[] = [];
      for (const [index, part] of content.entries()) {
        const wanted = final[index]!;
        if (index === current && (part.type === 'text' || part.type === 'thinking')) {
          const [streamed, text] = part.type === 'text' ? [part.text, (wanted as { text: string }).text] :
            [part.thinking, (wanted as { thinking: string }).thinking];
          if (text.length > streamed.length) events.push(...piece(part.type, text.slice(streamed.length)));
        }
        // The final part replaces the streamed one's fields: a call's id and decoded arguments, a block's metadata.
        if (part.type === 'toolCall' || part.type === 'neuralese') {
          for (const key of Object.keys(part)) delete (part as Record<string, unknown>)[key];
          Object.assign(part, wanted);
        }
      }
      events.push(...endAll());
      for (const part of final.slice(content.length)) events.push(...whole(part));
      return events;
    },
  };
}

/**
 * One model turn as pi-ai's event stream, built from the driver's deltas as they arrive (`streamedTurn`) and closed
 * from the final turn, which the `done` message equals. A driver that returns the whole turn sends no deltas: its parts
 * are then emitted in order once it arrives, each text and thinking part as start, one delta, end, each call as start,
 * its arguments as one delta, end, and each Neuralese part as its `NeuraleseEvent`.
 */
async function* turnEvents(model: Model<Api>, driver: ModelDriver, context: TranscriptContext,
    options: SimpleStreamOptions, store?: NeuraleseStore): AsyncGenerator<NatlangEvent> {
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
    const parts = streamedTurn(output, store);
    // Deltas arrive while the driver runs; its events queue until the consumer takes them.
    const queue: NatlangEvent[] = [];
    let wake: (() => void) | undefined;
    let outcome = undefined as { turn: ModelTurn } | { error: unknown } | undefined;
    const notify = () => { wake?.(); wake = undefined; };
    driver(request, options.signal, { onDelta: delta => { queue.push(...parts.delta(delta)); notify(); } })
      .then(turn => { outcome = { turn }; }, error => { outcome = { error }; }).finally(notify);
    while (true) {
      while (queue.length) yield queue.shift()!;
      if (outcome) break;
      await new Promise<void>(resolve => { wake = resolve; });
    }
    if ('error' in outcome) throw outcome.error;
    const { turn } = outcome;
    yield* parts.finish(turn);
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

/**
 * The natlang transport as pi-ai's chat API: `drivers` gives each model's driver. `store`: the runtime's Neuralese
 * store, where the metadata a streamed block arrives with is noted (the driver archives the block itself there).
 */
export function natlangApi(drivers: NatlangDrivers, store?: NeuraleseStore): ProviderStreams {
  const stream = (model: Model<Api>, context: TranscriptContext, options: SimpleStreamOptions = {}): AssistantMessageEventStream =>
    lazyStream(model, async () => turnEvents(model, drivers(model, { reasoning: Boolean(options.reasoning) }), context, options, store) as
      AsyncIterable<AssistantMessageEvent>);
  return { stream: (model, context, options?: StreamOptions) => stream(model, context, options), streamSimple: stream };
}
