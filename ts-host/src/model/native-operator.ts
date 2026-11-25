/**
 * The learned operators' in-context capture on engines that write states only from produce spans they sample
 * (OPERATORS_RESTORE_20261117.md §3.1); `native.capture` servers use model/native-state-client.ts `serverCapture`.
 */
import type { NativeCapture } from '../native/operator-frames.js';
import { nativeStateBlockMeta } from '../native/neuralese-store.js';
import { hexDigest } from '../native/hash.js';
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { fetchModel } from './chat-completion.js';
import { serves, type NeuraleseServerInfo } from './neuralese-info.js';
import { archiveNativeState } from './native-state-client.js';
import type { NeuralesePort } from '../native/neuralese-store.js';

/**
 * The in-context capture on an engine that produces states only from produce spans it samples (the managed
 * llama-server and the page, which serve neither `native.capture` nor `native.text-write`): the call's own messages
 * again, with a grammar that forces the exact reply ids up to its content, `produce_open`, the content and
 * `produce_close` (grammar token syntax `<[id]>`). The engine writes that span as it writes any produce span, so the
 * state is the same function of the same ids as `native.capture` on the Python engine and SGLang
 * (OPERATORS_RESTORE_20261117.md §3.1). The reply ids are the sampled ones when the call reported them, else the
 * tokenization of its reasoning and text.
 *
 * A capture of the reply itself (no `write`: a compaction operator's reply, a `compact_history` note) is stored as
 * `producer_kind` 'capture' when the server serves `native.producer-kind.capture` (`x_natlang_producer_kind`), so
 * readers splice it as stored, as on the Python engines (`serve/engine.py` `reads_in_context`); a write (a value's text
 * after its producing call) stays a 'span', read in context.
 */
export function grammarCapture(options: { endpoint: string; headers?: Record<string, string>; store: NeuraleseStore;
  info: NeuraleseServerInfo }): NativeCapture {
  const base = options.endpoint.replace(/\/$/, '');
  const headers = { 'content-type': 'application/json', ...options.headers };
  const markers = (options.info.native_state as { role_marker_ids?: number[] } | undefined)?.role_marker_ids;
  const capturesKind = serves(options.info, 'native.producer-kind.capture');
  const tokenize = async (content: string, special: boolean): Promise<number[]> => {
    const response = await fetchModel(`${base}/tokenize`, { method: 'POST', headers,
      body: JSON.stringify({ content, add_special: false, parse_special: special }) });
    if (!response.ok) throw new Error(`neuralese-native-capture: tokenize failed (${response.status})`);
    return ((await response.json()) as { tokens: (number | { id: number })[] }).tokens.map(t => typeof t === 'number' ? t : t.id);
  };
  const detokenize = async (tokens: number[]): Promise<string> => {
    const response = await fetchModel(`${base}/detokenize`, { method: 'POST', headers, body: JSON.stringify({ tokens }) });
    if (!response.ok) throw new Error(`neuralese-native-capture: detokenize failed (${response.status})`);
    return ((await response.json()) as { content: string }).content;
  };
  /** serve/engine.py `selected_span`: the whole tokens covering the one occurrence of `text` as the reply spells it. */
  const selectedSpan = async (ids: number[], text: string): Promise<[number, number]> => {
    const ends: number[] = [];
    for (let i = 0; i < ids.length; i++) ends.push((await detokenize(ids.slice(0, i + 1))).length);
    const whole = await detokenize(ids);
    for (const spelling of [...new Set([text, JSON.stringify(text).slice(1, -1)])]) {
      const first = whole.indexOf(spelling);
      if (first < 0 || whole.indexOf(spelling, first + 1) >= 0) continue;
      return [ends.findIndex(end => end > first), ends.findIndex(end => end >= first + spelling.length) + 1];
    }
    throw new Error('neuralese-native-capture: the selected text does not occur exactly once in the reply');
  };
  // The call's chat-template switches (an operator's `enable_thinking: false`): every re-rendering of the call uses them.
  const kwargs = (call: { chat_template_kwargs?: Record<string, unknown> }) =>
    call.chat_template_kwargs ? { chat_template_kwargs: call.chat_template_kwargs } : {};
  const applyTemplate = async (call: { messages: unknown[]; tools: unknown[]; chat_template_kwargs?: Record<string, unknown> }): Promise<string> => {
    const response = await fetchModel(`${base}/apply-template`, { method: 'POST', headers,
      body: JSON.stringify({ messages: call.messages, ...(call.tools.length ? { tools: call.tools } : {}), ...kwargs(call) }) });
    if (!response.ok) throw new Error(`neuralese-native-capture: apply-template failed (${response.status})`);
    return ((await response.json()) as { prompt: string }).prompt;
  };
  let specials: Promise<{ thinkOpen: number; thinkClose: number; ends: Set<number> }> | undefined;
  const single = (ids: number[], name: string) => {
    if (ids.length !== 1) throw new Error(`neuralese-native-capture: ${name} is not one token on this server`);
    return ids[0]!;
  };
  return async (call, extra = {}) => {
    if (!Array.isArray(markers) || markers.length !== 4)
      throw new Error('neuralese-native-capture-unavailable: the server declares no native role markers');
    if (!options.store.putNativeState || !options.store.getNativeState)
      throw new Error('neuralese-native-archive-unavailable: captures need a native-capable archive');
    specials ??= (async () => ({ thinkOpen: single(await tokenize('<think>', true), '<think>'),
      thinkClose: single(await tokenize('</think>', true), '</think>'),
      ends: new Set([single(await tokenize('<|im_end|>', true), '<|im_end|>')]) }))();
    const { thinkOpen, thinkClose, ends } = await specials;
    // The prompt ids: the call's own when it reported them, else the server's rendering of the same messages (a
    // captured rollout's sampling law refuses a grammar, so the forced request reports none itself).
    const client = { endpoint: options.endpoint, headers: options.headers, store: options.store };
    // A prompt that imported states (an operator call under its soft prompt state): its segments as recorded, each
    // state with its row count and identity, as the native.capture servers return them.
    const imported: Record<string, unknown>[] = [];
    let prompt: number[] = [];
    if (call.promptSegments?.length) {
      for (const segment of call.promptSegments) {
        if (segment.kind === 'tokens') { prompt.push(...segment.token_ids); continue; }
        const state = await archiveNativeState(client, segment.id);
        imported.push({ at: prompt.length, segment: { kind: 'native_state', id: segment.id, token_count: state.tensors.token_ids!.shape[0]!,
          native_identity: { checkpoint_sha256: state.metadata.checkpoint_sha256, tokenizer_fingerprint: state.metadata.tokenizer_fingerprint,
            embedding_weight_sha256: state.metadata.embedding_weight_sha256, architecture: state.metadata.architecture } } });
      }
    } else prompt = call.prompt?.length ? call.prompt : await tokenize(await applyTemplate(call), true);
    let lead: number[], span: number[];
    if (call.write !== undefined) {
      // A text write in its producing call (serve/engine.py capture_plan `text`): the whole reply up to its end of turn
      // is context (an empty or unfinished thought is closed first), then the text as the span.
      const reply = call.reply ?? [];
      const end = reply.findIndex(token => ends.has(token));
      lead = reply.slice(0, end < 0 ? reply.length : end);
      if ((prompt.at(-1) === thinkOpen || lead.includes(thinkOpen)) && !lead.includes(thinkClose)) lead.push(thinkClose);
      span = await tokenize(call.write, false);
      if (lead.some(token => markers.includes(token)))
        throw new Error('neuralese-native-capture: the producing reply carries role markers');
    } else {
      // Without its ids, the reply is its reasoning, `</think>`, then its text; a call rendered without reasoning
      // (`enable_thinking: false`: the prompt already closed an empty think block) is its text alone.
      const reply = call.reply?.length ? call.reply : call.chat_template_kwargs?.enable_thinking === false ?
        await tokenize(call.text ?? '', false) :
        [...await tokenize(call.reasoning ?? '', false), thinkClose, ...await tokenize(call.text ?? '', false)];
      const end = reply.findIndex(token => ends.has(token));
      let stop = end < 0 ? reply.length : end;
      let start = reply.slice(0, stop).lastIndexOf(thinkClose) + 1;
      if (call.select !== undefined) [start, stop] = await selectedSpan(reply.slice(0, stop), call.select);
      else if (!start && call.reply?.length && call.prompt?.at(-1) === thinkOpen)
        throw new Error('neuralese-native-capture: the reply ended inside its reasoning: no content to capture');
      lead = reply.slice(0, start);
      span = reply.slice(start, stop);
    }
    if (!span.length || span.some(token => markers.includes(token)))
      throw new Error('neuralese-native-capture: the reply has no content to capture, or carries role markers');
    const [open, close] = markers as [number, number];
    const forced = [...lead, open, ...span, close];
    // model-call-kind: native-forced-write -- training/model-call-kinds.json
    const response = await fetchModel(`${base}/v1/chat/completions`, { method: 'POST', headers, body: JSON.stringify({
      messages: call.messages, ...(call.tools.length ? { tools: call.tools } : {}), ...kwargs(call), temperature: 0,
      max_tokens: forced.length + 1, grammar: `root ::= ${forced.map(token => `<[${token}]>`).join(' ')}`,
      ...(call.write === undefined && capturesKind ? { x_natlang_producer_kind: 'capture' } : {}) }) });
    if (!response.ok) throw new Error(`neuralese-native-capture: forced span failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
    const result = await response.json() as { native_state?: { blocks?: { id?: string }[] } };
    const id = result.native_state?.blocks?.[0]?.id;
    if (typeof id !== 'string' || result.native_state!.blocks!.length !== 1)
      throw new Error('neuralese-native-capture: the forced reply did not produce exactly one state');
    const artifact = await archiveNativeState(client, id);
    const ids = artifact.tensors.token_ids!, view = new DataView(ids.data.buffer, ids.data.byteOffset, ids.data.byteLength);
    if (ids.shape[0] !== span.length || span.some((token, index) => BigInt(token) !== view.getBigInt64(index * 8, true)))
      throw new Error('neuralese-native-capture: the produced state does not hold the forced span');
    if (call.write === undefined && capturesKind && artifact.metadata.producer_kind !== 'capture')
      throw new Error('neuralese-native-capture: the server stored the capture without producer_kind capture');
    const prefix = [...prompt, ...lead, open];
    const text = typeof artifact.metadata.text === 'string' ? artifact.metadata.text : '';
    const meta = nativeStateBlockMeta(artifact);
    const segments: Record<string, unknown>[] = [];
    let last = 0, rows = 0;
    for (const { at, segment } of imported as { at: number; segment: { token_count: number } }[]) {
      if (at > last) segments.push({ kind: 'tokens', token_ids: prefix.slice(last, at), producing: prefix.slice(last, at).map(() => false) });
      segments.push(segment);
      rows += segment.token_count;
      last = at;
    }
    segments.push({ kind: 'tokens', token_ids: prefix.slice(last), producing: prefix.slice(last).map(() => false) });
    return { ...meta, ...(extra.type ? { type: extra.type } : {}), producer: {
      source_text_sha256: hexDigest(text), native_source_capture: {
        policy: { checkpoint_sha256: artifact.metadata.checkpoint_sha256, tokenizer_fingerprint: artifact.metadata.tokenizer_fingerprint,
          embedding_weight_sha256: artifact.metadata.embedding_weight_sha256, architecture: artifact.metadata.architecture },
        segments: [...segments, { kind: 'tokens', token_ids: span, producing: span.map(() => true) },
          { kind: 'tokens', token_ids: [close], producing: [false] }],
        payload_start: prefix.length + rows, payload_stop: prefix.length + rows + span.length } } };
  };
}

/**
 * The write port of an engine that writes states only from produce spans it samples (the managed llama-server, the
 * page): `grammarCapture` in write mode, the text forced as a produce span at the end of its producing call
 * (`NeuralesePort.write`'s `context`), or of `text_producer_messages`' call (the text as the user's message) for a value
 * with none: the same document `native.text-write` writes on the Python engine and SGLang.
 */
export function grammarWritePort(options: { endpoint: string; headers?: Record<string, string>; store: NeuraleseStore;
  info: NeuraleseServerInfo }): NeuralesePort {
  const architecture = (options.info.native_state as { architecture?: string } | undefined)?.architecture;
  if (!architecture) throw new Error('neuralese-native-text-write-unavailable: the server declares no native architecture');
  const capture = grammarCapture(options);
  return { dialect: `native-state/${architecture}`, forcedText: true, async write(text, extra = {}) {
    const context = extra.context;
    return capture({ messages: context?.messages ?? [{ role: 'user', content: text }], tools: context?.tools ?? [],
      ...(context?.prompt ? { prompt: context.prompt } : {}), ...(context?.promptSegments ? { promptSegments: context.promptSegments } : {}),
      reply: context?.reply ?? [], write: text }, { ...(extra.type ? { type: extra.type } : {}) });
  } };
}
