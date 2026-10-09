/**
 * Transport to a Neuralese-capable model server (spec/NEURALESE_PORT.md, "Wire protocol"; S4 §3–4).
 *
 * The reference server is `python -m natlang_neuralese.serve` (training/neuralese). Requests are ordinary chat
 * completions whose message content and tool-call arguments may be arrays of `{ type: "text" }` and
 * `{ type: "neuralese", id }` parts; the server writes blocks while decoding and returns them the same way, plus a
 * `neuralese.blocks` list with each written block's metadata and write record (temperature, seed, IDs of the payload
 * mean and log-scale, stop logits).
 *
 * Replies stream when the server declares `stream: true` in `/v1/neuralese/info` (plans/STREAMING.md §1.2): text and
 * each written block arrive as the turn's deltas (`ModelTurnOptions.onDelta`), and the final `x_natlang_message` is the
 * reply. A server that does not declare it (the llama.cpp fork until it serves SSE) gets whole requests.
 *
 * The driver keeps the runtime's store and the server's store in step: before a request it uploads any block the
 * conversation refers to that the server lacks, and after a reply it downloads the blocks the server wrote, so that
 * references the runtime holds resolve locally (for `.nz` files, saving, re-sending). Content IDs are computed the same
 * way on both sides, so a downloaded block lands under the ID the server reported.
 *
 * The runtime's store is also the archive that makes blocks outlive the server (plans/neuralese/DECISIONS.md,
 * "representation chosen by use"): `withRestoredBlocks` is the one place blocks move runtime → server, and when the
 * server answers `neuralese-unknown-block` (it restarted, or collected them) it re-uploads them from the archive and
 * retries once. A client names itself with `owner` (the `x-natlang-owner` header), so its pins and collections never
 * touch other sessions' blocks.
 */
import { assembleChatCompletion, chatCompletionModelTurn, fetchModel, httpChatTransport, isStream, type ChatCompletionOptions, type ChatTransport,
  type HttpChatOptions } from './chat-completion.js';
import type { DecisionScorer, DecisionScores, ModelTurn, ModelTurnDelta, ModelTurnOptions, ModelTurnRequest } from '../contracts.js';
import { activeAdapters, activeRecorder } from '../neuralese/recording.js';
import { isContentParts, partsToText, type ContentPart } from '../native/neuralese.js';
import { constantBlock, neuraleseContentId, type NeuraleseBlock, type NeuraleseBlockMeta, type NeuraleseBlockInput,
  type NeuraleseDtype, type NeuraleseStore } from '../native/neuralese-store.js';

type Json = Record<string, unknown>;

const SAFETENSORS_DTYPE: Record<NeuraleseDtype, string> = { f32: 'F32', f16: 'F16', bf16: 'BF16' };
const FROM_SAFETENSORS: Record<string, NeuraleseDtype> = { F32: 'f32', F16: 'f16', BF16: 'bf16' };

/** One block as a safetensors body: a `payload` tensor and the metadata under `natlang.block`. */
export function encodeBlockBody(meta: NeuraleseBlockMeta, data: Uint8Array): Uint8Array {
  const header: Json = { __metadata__: { 'natlang.block': JSON.stringify(meta) },
    payload: { dtype: SAFETENSORS_DTYPE[meta.dtype], shape: [meta.length, meta.width], data_offsets: [0, data.length] } };
  const json = JSON.stringify(header);
  const length = new TextEncoder().encode(json).length;
  const raw = new TextEncoder().encode(json + ' '.repeat((8 - length % 8) % 8));
  const out = new Uint8Array(8 + raw.length + data.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(raw.length), true);
  out.set(raw, 8); out.set(data, 8 + raw.length);
  return out;
}

export function decodeBlockBody(body: Uint8Array): NeuraleseBlock {
  if (body.length < 8) throw new Error('neuralese-bad-block: not a safetensors body');
  const size = Number(new DataView(body.buffer, body.byteOffset, body.byteLength).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(body.subarray(8, 8 + size))) as Json;
  const meta = JSON.parse(String((header.__metadata__ as Json | undefined)?.['natlang.block'] ?? '{}')) as NeuraleseBlockMeta;
  const entry = header.payload as { dtype: string; shape: [number, number]; data_offsets: [number, number] } | undefined;
  if (!entry) throw new Error('neuralese-bad-block: no payload tensor');
  const dtype = FROM_SAFETENSORS[entry.dtype];
  if (!dtype) throw new Error(`neuralese-bad-block: unsupported dtype ${entry.dtype}`);
  const data = body.slice(8 + size + entry.data_offsets[0], 8 + size + entry.data_offsets[1]);
  return { meta: { ...meta, length: entry.shape[0], width: entry.shape[1], dtype }, data };
}

/** Header naming the owner of requests (a session or runtime ID): its pins and collections are its own (serve/store.py). */
export const OWNER_HEADER = 'x-natlang-owner';

/** Whether `error` is a server's `neuralese-unknown-block`: the server lacks a block a request names. */
export function isUnknownBlockError(error: unknown): boolean {
  const found = error as { providerCode?: unknown; code?: unknown; message?: unknown } | null;
  return found?.providerCode === 'neuralese-unknown-block' || found?.code === 'neuralese-unknown-block' ||
    (typeof found?.message === 'string' && found.message.includes('neuralese-unknown-block'));
}

/**
 * Upload to the server every block of `ids` it lacks, from `archive` (the runtime's content-addressed store) or the
 * registered constants. Returns the IDs that neither has; the caller reports them in its own terms.
 */
export async function restoreBlocks(remote: NeuraleseStore, archive: NeuraleseStore | undefined,
    ids: Iterable<string>): Promise<string[]> {
  const missing: string[] = [];
  for (const id of new Set(ids)) {
    if (await remote.has(id)) continue;
    const block = (archive && await archive.get(id)) ?? constantBlock(id);
    if (!block) { missing.push(id); continue; }
    const { id: _, ...rest } = block.meta;
    await remote.put({ ...rest, data: block.data });
  }
  return missing;
}

/**
 * Run a server request that reads `ids`, with those blocks on the server first. `known` caches IDs already uploaded
 * (they are not checked again); when the server still answers `neuralese-unknown-block` (it restarted or collected
 * them since), every block is checked and restored from `archive`, and the request runs once more.
 */
export async function withRestoredBlocks<T>(remote: NeuraleseStore, archive: NeuraleseStore | undefined,
    ids: readonly string[], run: () => Promise<T>, known?: Set<string>): Promise<T> {
  const restore = async (all: boolean) => {
    if (all) for (const id of ids) known?.delete(id);
    const pending = known ? ids.filter(id => !known.has(id)) : ids;
    const missing = await restoreBlocks(remote, archive, pending);
    if (missing.length) throw new Error(`neuralese-unknown-block: ${missing.join(', ')} ${missing.length === 1 ? 'is' : 'are'} ` +
      `neither on the server nor in the runtime's store`);
    for (const id of pending) known?.add(id);
  };
  await restore(false);
  try {
    return await run();
  } catch (error) {
    if (!isUnknownBlockError(error)) throw error;
    await restore(true);
    return await run();
  }
}

/** The server's block endpoints as a `NeuraleseStore`. */
export class HttpNeuraleseStore implements NeuraleseStore {
  private readonly base: string;
  constructor(endpoint: string, private readonly headers: Record<string, string> = {}) {
    this.base = endpoint.replace(/\/$/, '') + '/v1/neuralese';
  }
  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    return fetchModel(this.base + path, { ...init, headers: { ...this.headers, ...(init.headers as Record<string, string> ?? {}) } });
  }
  async put(block: NeuraleseBlockInput): Promise<NeuraleseBlockMeta> {
    const { data, ...rest } = block;
    const id = neuraleseContentId(block);
    const response = await this.request(`/blocks/${id}`, { method: 'PUT', body: encodeBlockBody({ id, ...rest }, data) as BodyInit,
      headers: { 'content-type': 'application/octet-stream' } });
    if (!response.ok) throw new Error(`neuralese block upload failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
    return await response.json() as NeuraleseBlockMeta;
  }
  async get(id: string): Promise<NeuraleseBlock | undefined> {
    const response = await this.request(`/blocks/${id}`);
    if (response.status === 404) return undefined;
    if (!response.ok) throw new Error(`neuralese block fetch failed (${response.status})`);
    return decodeBlockBody(new Uint8Array(await response.arrayBuffer()));
  }
  async meta(id: string): Promise<NeuraleseBlockMeta | undefined> {
    const response = await this.request(`/blocks/${id}/meta`);
    if (response.status === 404) return undefined;
    if (!response.ok) throw new Error(`neuralese block metadata failed (${response.status})`);
    return await response.json() as NeuraleseBlockMeta;
  }
  async has(id: string): Promise<boolean> { return (await this.meta(id)) !== undefined; }
  async pin(id: string): Promise<void> {
    const response = await this.request(`/blocks/${id}/pin`, { method: 'POST' });
    if (!response.ok) throw new Error(`neuralese-unknown-block: ${id}`);
  }
  async unpin(id: string): Promise<void> { await this.request(`/blocks/${id}/unpin`, { method: 'POST' }); }
  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    const response = await this.request('/collect', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ referenced: [...referenced] }) });
    return ((await response.json()) as { removed: string[] }).removed;
  }
}

export type NeuraleseServerOptions = Omit<HttpChatOptions, 'stream'> & Pick<ChatCompletionOptions, 'onExchange' | 'onTurn'> & {
  /** The runtime's store: blocks are uploaded (and restored after a server restart) from it and written blocks are
   * downloaded into it. */
  store?: NeuraleseStore;
  /** This session's or runtime's ID on the server (`x-natlang-owner`): it holds the blocks it uploads or is sent, and
   * its pins and collections (`blocks.pin`, `blocks.collect`) leave other owners' blocks alone. */
  owner?: string;
  /** Neuralese temperature for written payloads (spec/NEURALESE_PORT.md, "Temperature"); 0, the default, is deterministic. */
  neuraleseTemperature?: number;
  /** Requested maximum block length; the server caps it by its hard maximum. */
  neuraleseMaxLength?: number;
  /** Extra request fields. */
  request?: Json;
  /**
   * Weight adapters as GGUF LoRAs, for servers that apply adapters as LoRAs (the llama.cpp fork, native or in the
   * browser: `info.adapters === 'lora'`): the driver uploads each bound adapter's LoRA once
   * (`PUT /v1/neuralese/adapters/{id}/lora`). `referenceAdapterLoras` exports them from a reference server; a browser
   * runtime passes the files its model manifest ships. Without a LoRA the fork refuses the request (409).
   */
  adapterLoras?: (id: string) => Promise<Uint8Array | null>;
};

/** Adapter LoRAs exported by a reference server (`GET /v1/neuralese/adapters/{id}/lora`), for `adapterLoras`. */
export function referenceAdapterLoras(endpoint: string, headers: Record<string, string> = {}): (id: string) => Promise<Uint8Array | null> {
  return async id => {
    const response = await fetchModel(`${endpoint.replace(/\/$/, '')}/v1/neuralese/adapters/${id}/lora`, { headers });
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`adapter LoRA ${id}: HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
    return new Uint8Array(await response.arrayBuffer());
  };
}

/** Block IDs a request's messages carry as parts. */
export function requestBlockIds(messages: readonly unknown[]): string[] {
  const ids = new Set<string>();
  const visit = (value: unknown) => {
    if (isContentParts(value)) { for (const part of value) if (part.type === 'neuralese') ids.add(part.id); return; }
    if (Array.isArray(value)) value.forEach(visit);
    else if (value && typeof value === 'object') Object.values(value).forEach(visit);
  };
  visit(messages);
  return [...ids];
}

/** A reply's part arrays (message content, strings inside tool-call arguments) as conversation text. */
function partsAsText(value: unknown): unknown {
  if (isContentParts(value)) return partsToText(value as ContentPart[]);
  if (Array.isArray(value)) return value.map(partsAsText);
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, partsAsText(item)]));
  return value;
}

function normalizeReply(body: Json): Json {
  const choice = (body.choices as Json[] | undefined)?.[0];
  const message = choice?.message as Json | undefined;
  if (!message) return body;
  if (isContentParts(message.content)) message.content = partsToText(message.content as ContentPart[]);
  for (const call of (message.tool_calls ?? []) as Array<{ function?: { arguments?: unknown } }>) {
    const fn = call.function;
    if (!fn) continue;
    const parsed = typeof fn.arguments === 'string' ? JSON.parse(fn.arguments) : fn.arguments;
    fn.arguments = JSON.stringify(partsAsText(parsed));
  }
  return body;
}

/** A model-turn driver for a Neuralese server. It advertises `neuralese: true`. */
export function neuraleseServerModelTurn(options: NeuraleseServerOptions):
    ((request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => Promise<ModelTurn>) & { neuralese: true; blocks: HttpNeuraleseStore; decide: DecisionScorer } {
  const { store, neuraleseTemperature, neuraleseMaxLength, request: extra, onExchange, onTurn, adapterLoras, owner,
    ...connection } = options;
  const http = owner === undefined ? connection : { ...connection, headers: { ...connection.headers, [OWNER_HEADER]: owner } };
  const remote = new HttpNeuraleseStore(http.endpoint, http.headers);
  const uploaded = new Set<string>();
  // What the server declares (`/v1/neuralese/info`), read once: whether it applies adapters as LoRAs, and whether it
  // streams (`stream: true`; a server that does not declare it gets whole requests, never a streamed try first).
  let info: Promise<Json | null> | undefined;
  const serverInfo = () => info ??= fetchModel(http.endpoint.replace(/\/$/, '') + '/v1/neuralese/info', { headers: http.headers })
    .then(async response => response.ok ? await response.json() as Json : null).catch(() => null);
  const whole = httpChatTransport({ ...http, stream: false });
  const streamed = httpChatTransport({ ...http, stream: true });
  // Adapter blocks are uploaded like message blocks; their IDs travel as request parts the server resolves.
  // Adapters of the dynamic scope (`withAdapters`) and those the calling function's context binds (request field).
  // A server that applies adapters as LoRAs (the fork) gets each adapter's LoRA once, from `adapterLoras`.
  const loras = new Set<string>();
  const uploadLoras = async (ids: readonly string[]) => {
    if (!adapterLoras || (await serverInfo())?.adapters !== 'lora') return;
    for (const id of ids) {
      if (loras.has(id)) continue;
      const bytes = await adapterLoras(id);
      if (!bytes) continue;
      const response = await fetchModel(`${http.endpoint.replace(/\/$/, '')}/v1/neuralese/adapters/${id}/lora`, { method: 'PUT',
        headers: { 'content-type': 'application/octet-stream', ...http.headers }, body: bytes as unknown as BodyInit });
      if (!response.ok) throw new Error(`adapter LoRA ${id}: HTTP ${response.status}: ${(await response.text()).slice(0, 500)}`);
      loras.add(id);
    }
  };
  const adapterParts = async (bound: readonly { id: string; scale: number }[] = []) => {
    const adapters = [...activeAdapters(), ...bound];
    if (adapters.length) await uploadLoras(adapters.map(item => item.id));
    return adapters;
  };
  const blockIds = (messages: unknown[], adapters: readonly { id: string }[]) =>
    [...requestBlockIds(messages), ...adapters.map(item => item.id)];
  const transport: ChatTransport = async (body, signal, meta) => {
    const adapters = await adapterParts((body.x_natlang_adapters ?? []) as { id: string; scale: number }[]);
    const inner = (await serverInfo())?.stream === true ? streamed : whole;
    // A streamed reply is assembled here, where its body is post-processed; its deltas go to the turn's observer, and a
    // request re-sent after restoring blocks first voids what the failed attempt streamed.
    let emitted = false;
    const onDelta = meta?.onDelta && ((delta: ModelTurnDelta) => { emitted = true; meta.onDelta!(delta); });
    const reply = await withRestoredBlocks(remote, store, blockIds(body.messages as unknown[], adapters), async () => {
      if (onDelta && emitted) { onDelta({ type: 'reset' }); emitted = false; }
      const answer = await inner(adapters.length ? { ...body, x_natlang_adapters: adapters } : body, signal);
      return isStream(answer) ? await assembleChatCompletion(answer, onDelta) : answer;
    }, uploaded);
    const recorder = activeRecorder();
    if (recorder) {
      const message = (reply.choices as Json[] | undefined)?.[0]?.message as Json | undefined;
      if (message) recorder.record({ messages: structuredClone(body.messages as unknown[]),
        tools: body.tools ? structuredClone(body.tools as unknown[]) : undefined, reply: structuredClone(message),
        blocks: structuredClone(((reply.neuralese as Json | undefined)?.blocks ?? []) as Json[]),
        ...(adapters.length ? { adapters: [...adapters] } : {}) });
    }
    if (store) for (const meta of ((reply.neuralese as Json | undefined)?.blocks ?? []) as NeuraleseBlockMeta[]) {
      if (!(await store.has(meta.id))) {
        const block = await remote.get(meta.id);
        if (!block) throw new Error(`neuralese-unknown-block: the server reported ${meta.id} but does not have it`);
        const { id: _, ...rest } = meta;
        const local = await store.put({ ...rest, dtype: block.meta.dtype, length: block.meta.length,
          width: block.meta.width, data: block.data });
        if (local.id !== meta.id) throw new Error(`neuralese block ${meta.id} arrived as ${local.id}: content IDs disagree`);
      }
      uploaded.add(meta.id);
    }
    return normalizeReply(reply);
  };
  const request: Json = { ...extra,
    ...(neuraleseTemperature === undefined ? {} : { neuralese_temperature: neuraleseTemperature }),
    ...(neuraleseMaxLength === undefined ? {} : { neuralese_max_length: neuraleseMaxLength }) };
  const driver = chatCompletionModelTurn(transport, { request, onExchange, onTurn });
  // Decision readout: one prompt pass, every option scored from its cache (serve/grad.py `decide`).
  const decide: DecisionScorer = async ({ messages, options: replies, adapters: bound }, signal) => {
    const adapters = await adapterParts(bound ?? []);
    activeRecorder()?.record({ messages: structuredClone(messages), reply: { role: 'assistant', content: null }, blocks: [],
      decision: { options: [...replies] }, ...(adapters.length ? { adapters: [...adapters] } : {}) });
    return await withRestoredBlocks(remote, store, blockIds(messages, adapters), async () => {
      const response = await fetchModel(http.endpoint.replace(/\/$/, '') + '/v1/neuralese/decide', { method: 'POST', signal,
        headers: { 'content-type': 'application/json', ...http.headers },
        body: JSON.stringify({ messages, options: replies, ...(adapters.length ? { adapters } : {}) }) });
      if (response.status === 404) throw new Error('decision-unsupported: the server has no /v1/neuralese/decide');
      if (!response.ok) throw new Error(`neuralese decide HTTP ${response.status}: ${(await response.text()).slice(0, 2000)}`);
      return await response.json() as DecisionScores;
    }, uploaded);
  };
  return Object.assign(driver, { neuralese: true as const, blocks: remote, decide });
}
