/**
 * Transport to a Neuralese-capable model server (spec/NEURALESE_PORT.md, "Wire protocol"; S4 §3–4).
 *
 * The reference server is `python -m natlang_neuralese.serve` (training/neuralese). Requests are ordinary chat
 * completions whose message content and tool-call arguments may be arrays of `{ type: "text" }` and
 * `{ type: "neuralese", id }` parts; the server writes blocks while decoding and returns them the same way, plus a
 * `neuralese.blocks` list with each written block's metadata and write record (temperature, seed, IDs of the payload
 * mean and log-scale, stop logits).
 *
 * The driver keeps the runtime's store and the server's store in step: before a request it uploads any block the
 * conversation refers to that the server lacks, and after a reply it downloads the blocks the server wrote, so that
 * references the runtime holds resolve locally (for `.nz` files, saving, re-sending). Content IDs are computed the same
 * way on both sides, so a downloaded block lands under the ID the server reported.
 */
import { chatCompletionModelTurn, fetchModel, httpChatTransport, type ChatCompletionOptions, type ChatTransport,
  type HttpChatOptions } from './chat-completion.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { isContentParts, partsToText, type ContentPart } from '../native/neuralese.js';
import { neuraleseContentId, type NeuraleseBlock, type NeuraleseBlockMeta, type NeuraleseBlockInput,
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
  /** The runtime's store: blocks are uploaded from it and written blocks are downloaded into it. */
  store?: NeuraleseStore;
  /** Neuralese temperature for written payloads (spec/NEURALESE_PORT.md, "Temperature"); 0, the default, is deterministic. */
  neuraleseTemperature?: number;
  /** Requested maximum block length; the server caps it by its hard maximum. */
  neuraleseMaxLength?: number;
  /** Extra request fields. */
  request?: Json;
};

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
    ((request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn>) & { neuralese: true; blocks: HttpNeuraleseStore } {
  const { store, neuraleseTemperature, neuraleseMaxLength, request: extra, onExchange, onTurn, ...http } = options;
  const remote = new HttpNeuraleseStore(http.endpoint, http.headers);
  const uploaded = new Set<string>();
  const inner = httpChatTransport({ ...http, stream: false });
  const transport: ChatTransport = async (body, signal) => {
    for (const id of requestBlockIds(body.messages as unknown[])) {
      if (uploaded.has(id)) continue;
      if (!(await remote.has(id))) {
        const block = store && await store.get(id);
        if (!block) throw new Error(`neuralese-unknown-block: ${id} is neither on the server nor in the runtime's store`);
        const { id: _, ...rest } = block.meta;
        await remote.put({ ...rest, data: block.data });
      }
      uploaded.add(id);
    }
    const reply = await inner(body, signal) as Json;
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
  return Object.assign(driver, { neuralese: true as const, blocks: remote });
}
