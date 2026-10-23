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
import { coalescingScorer, explicitBatchScoreMany } from './scoring.js';
import type { DecisionScorer, DecisionScores, ModelTurn, ModelTurnDelta, ModelTurnOptions, ModelTurnRequest } from '../contracts.js';
import { activeAdapters, activeNativeStates, activeRecorder, type AdapterBinding, type TurnRecorder } from '../neuralese/recording.js';
import { serves } from './neuralese-info.js';
import {validateCoefficientRuntime,requireCoefficientAttestation,coefficientBindings,coefficientStateModes} from './native-coefficient-policy.js';
import { isContentParts, nativeStateSentinel, nativeStateSentinelIds,
  isNativeStateId, runtimeTextToParts, type RuntimeContentPart } from '../native/neuralese.js';
import { nativeStateConsumerFromInfo } from '../native/neuralese.js';
import { constantBlock, decodeNativeState, encodeNativeState, nativeStateContentId, nativeStateBlockMeta,
  decodeNativeTensorArtifact, encodeNativeTensorArtifact, nativeTensorArtifactContentId, nativeTensorArtifactDependencies, nativeStateArtifactDependencies, archiveNativeArtifactFrom, archiveNativeTensorFrom,
  nativeTensorArtifactBlockMeta, neuraleseContentId, nativeModelIdentityEqual, type NeuraleseBlock, type NeuraleseBlockMeta, type NeuraleseBlockInput,
  type NativeStateArtifact, type NativeTensorArtifact, type NeuraleseDtype, type NeuraleseStore } from '../native/neuralese-store.js';

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
    found?.providerCode === 'neuralese-unknown-native-state' || found?.code === 'neuralese-unknown-native-state' ||
    found?.providerCode === 'neuralese-unknown-native-artifact' || found?.code === 'neuralese-unknown-native-artifact' ||
    (typeof found?.message === 'string' && /neuralese-unknown-(?:block|native-state|native-artifact)/.test(found.message));
}

/**
 * Upload to the server every block of `ids` it lacks, from `archive` (the runtime's content-addressed store) or the
 * registered constants. Returns the IDs that neither has; the caller reports them in its own terms.
 */
export async function restoreBlocks(remote: NeuraleseStore, archive: NeuraleseStore | undefined,
    ids: Iterable<string>): Promise<string[]> {
  const missing: string[] = [], visited = new Set<string>();
  const restore = async (id: string): Promise<void> => {
    if (visited.has(id)) return;
    visited.add(id);
    if (id.startsWith('nzs1_')) {
      const artifact=await archive?.getNativeState?.(id)??await remote.getNativeState?.(id);
      if(!artifact||!remote.putNativeState){missing.push(id);return;}
      const before=missing.length;
      for(const dependency of nativeStateArtifactDependencies(artifact))await restore(dependency);
      if(missing.length!==before)return;
      if(!(await remote.has(id))){
        const {id:_,...input}=artifact;
        if((await remote.putNativeState(input)).id!==id)throw new Error(`native state changed while restoring ${id}`);
      }
      return;
    }
    if (id.startsWith('nnt1_')) {
      const artifact = await archive?.getNativeTensorArtifact?.(id) ?? await remote.getNativeTensorArtifact?.(id);
      if (!artifact || !remote.putNativeTensorArtifact) { missing.push(id); return; }
      for (const dependency of nativeTensorArtifactDependencies(artifact)) await restore(dependency);
      if (!(await remote.has(id))) {
        const { id: _, ...input } = artifact;
        if ((await remote.putNativeTensorArtifact(input)).id !== id) throw new Error(`native artifact changed while restoring ${id}`);
      }
      return;
    }
    if (await remote.has(id)) return;
    const block = (archive && await archive.get(id)) ?? constantBlock(id);
    if (!block) { missing.push(id); return; }
    const { id: _, ...rest } = block.meta;
    await remote.put({ ...rest, data: block.data });
  };
  for (const id of new Set(ids)) await restore(id);
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
  async putNativeState(artifact: Omit<NativeStateArtifact, 'id'>): Promise<NativeStateArtifact> {
    const id = nativeStateContentId(artifact.metadata, artifact.tensors);
    const response = await this.request(`/native-states/${id}`, { method: 'PUT',
      body: encodeNativeState(artifact) as BodyInit, headers: { 'content-type': 'application/octet-stream' } });
    if (!response.ok) throw new Error(`native-state upload failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
    return { ...structuredClone(artifact), id };
  }
  async getNativeState(id: string): Promise<NativeStateArtifact | undefined> {
    const response = await this.request(`/native-states/${id}`);
    if (response.status === 404) return undefined;
    if (!response.ok) throw new Error(`native-state fetch failed (${response.status})`);
    const artifact = decodeNativeState(new Uint8Array(await response.arrayBuffer()));
    if (artifact.id !== id) throw new Error(`native-state identity mismatch: requested ${id}, received ${artifact.id}`);
    return artifact;
  }
  async hasNativeState(id: string): Promise<boolean> {
    const response = await this.request(`/native-states/${id}/meta`);
    if (response.status === 404) return false;
    if (!response.ok) throw new Error(`native-state metadata failed (${response.status})`);
    return true;
  }
  async putNativeTensorArtifact(artifact: Omit<NativeTensorArtifact, 'id'>): Promise<NativeTensorArtifact> {
    const id = nativeTensorArtifactContentId(artifact.metadata, artifact.tensors, artifact.canonicalMetadata);
    const response = await this.request(`/native-tensor-artifacts/${id}`, { method: 'PUT',
      body: encodeNativeTensorArtifact(artifact) as BodyInit, headers: { 'content-type': 'application/octet-stream' } });
    if (!response.ok) throw new Error(`native tensor artifact upload failed (${response.status}): ${(await response.text()).slice(0, 500)}`);
    return decodeNativeTensorArtifact(encodeNativeTensorArtifact(artifact));
  }
  async getNativeTensorArtifact(id: string): Promise<NativeTensorArtifact | undefined> {
    const response = await this.request(`/native-tensor-artifacts/${id}`);
    if (response.status === 404) return undefined;
    if (!response.ok) throw new Error(`native tensor artifact fetch failed (${response.status})`);
    const artifact = decodeNativeTensorArtifact(new Uint8Array(await response.arrayBuffer()));
    if (artifact.id !== id) throw new Error(`native tensor artifact identity mismatch: requested ${id}, received ${artifact.id}`);
    return artifact;
  }
  async hasNativeTensorArtifact(id: string): Promise<boolean> {
    const response = await this.request(`/native-tensor-artifacts/${id}/meta`);
    if (response.status === 404) return false;
    if (!response.ok) throw new Error(`native tensor artifact metadata failed (${response.status})`);
    return true;
  }
  async meta(id: string): Promise<NeuraleseBlockMeta | undefined> {
    if (isNativeStateId(id)) {
      const artifact = await this.getNativeState(id);
      return artifact && nativeStateBlockMeta(artifact);
    }
    if (/^nnt1_[a-z2-7]{52}$/.test(id)) {
      const response = await this.request(`/native-tensor-artifacts/${id}/meta`);
      if (response.status === 404) return undefined;
      if (!response.ok) throw new Error(`native tensor artifact metadata failed (${response.status})`);
      const record = await response.json() as { id: string; schema: NativeTensorArtifact['metadata']['schema'];
        kind: NativeTensorArtifact['metadata']['kind']; native_identity: NativeTensorArtifact['metadata']['native_identity'];
        source_artifact_id: string | null; provenance: Record<string, unknown>;
        tensors: NativeTensorArtifact['metadata']['tensors'] };
      return nativeTensorArtifactBlockMeta({ id: record.id, metadata: { schema: record.schema, kind: record.kind,
        native_identity: record.native_identity, source_artifact_id: record.source_artifact_id,
        provenance: record.provenance, tensors: record.tensors } });
    }
    const response = await this.request(`/blocks/${id}/meta`);
    if (response.status === 404) return undefined;
    if (!response.ok) throw new Error(`neuralese block metadata failed (${response.status})`);
    return await response.json() as NeuraleseBlockMeta;
  }
  async has(id: string): Promise<boolean> { return (await this.meta(id)) !== undefined; }
  async pin(id: string): Promise<void> {
    const family = id.startsWith('nzs1_') ? 'native-states' : id.startsWith('nnt1_') ? 'native-tensor-artifacts' : 'blocks';
    const response = await this.request(`/${family}/${id}/pin`, { method: 'POST' });
    if (!response.ok) throw new Error(`neuralese-unknown-block: ${id}`);
  }
  async unpin(id: string): Promise<void> {
    const family = id.startsWith('nzs1_') ? 'native-states' : id.startsWith('nnt1_') ? 'native-tensor-artifacts' : 'blocks';
    await this.request(`/${family}/${id}/unpin`, { method: 'POST' });
  }
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
  /** Explicit raw-state exchange between actors of the exact same native base. */
  nativeStateModes?: Record<string, 'same-base-adapter-state'>;
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

function malformedNativePartList(value: unknown): boolean {
  return Array.isArray(value) && value.length > 0 && value.every(part => part && typeof part === 'object' &&
    ['text', 'neuralese', 'native_state'].includes(part.type)) &&
    value.some(part => part.type === 'native_state' && !isNativeStateId(part.id));
}

/** Native artifacts carried by parts or the runtime's private-use transcript marker. */
export function requestNativeStateIds(messages: readonly unknown[]): string[] {
  const ids = new Set<string>();
  const visit = (value: unknown, key?: string) => {
    if (key === 'arguments' && typeof value === 'string') {
      try { visit(JSON.parse(value)); } catch { /* the transport will report malformed arguments */ }
      return;
    }
    if (malformedNativePartList(value)) return;
    if (Array.isArray(value) && value.length && value.every(part => part && typeof part === 'object' &&
        (part.type === 'text' && typeof part.text === 'string' ||
         part.type === 'neuralese' && typeof part.id === 'string' ||
         part.type === 'native_state' && isNativeStateId(part.id)))) {
      for (const part of value as RuntimeContentPart[]) if (part.type === 'native_state') ids.add(part.id);
      return;
    }
    if (typeof value === 'string') { nativeStateSentinelIds(value).forEach(id => ids.add(id)); return; }
    if (Array.isArray(value)) value.forEach(item => visit(item));
    else if (value && typeof value === 'object') {
      const part = value as { type?: unknown; id?: unknown };
      if (part.type === 'native_state' && isNativeStateId(part.id)) { ids.add(part.id); return; }
      for (const [name, item] of Object.entries(value)) visit(item, name);
    }
  };
  visit(messages);
  return [...ids];
}

function materializeNativeStateSentinels(value: unknown, key?: string): unknown {
  if (key === 'arguments' && typeof value === 'string') {
    try { return JSON.stringify(materializeNativeStateSentinels(JSON.parse(value))); }
    catch { return value; } // preserve malformed input for the server's existing argument error path
  }
  if (malformedNativePartList(value)) return value;
  if (typeof value === 'string') return nativeStateSentinelIds(value).length ? runtimeTextToParts(value) : value;
  if (Array.isArray(value) && value.length && value.every(part => part && typeof part === 'object' &&
      (part.type === 'text' && typeof part.text === 'string' ||
       part.type === 'neuralese' && typeof part.id === 'string' ||
       part.type === 'native_state' && isNativeStateId(part.id))))
    return value.flatMap(part => part.type === 'text' ? runtimeTextToParts(part.text) : [part]);
  if (Array.isArray(value)) return value.map(item => materializeNativeStateSentinels(item));
  if (value && typeof value === 'object') {
    const part = value as { type?: unknown; id?: unknown;stop_gradient?:unknown;source_occurrence?:unknown };
    if (part.type === 'native_state' && part.source_occurrence !== undefined && (typeof part.source_occurrence!=='string'||!part.source_occurrence)) throw new TypeError('native source occurrence must be a nonempty actual writer reference');
    if (part.type === 'native_state' && part.stop_gradient !== undefined && part.stop_gradient !== true) throw new TypeError('native state constant-use metadata must be true or absent');
    if (part.type === 'native_state' && isNativeStateId(part.id)) return [{ type: 'native_state', id: part.id,
      ...(part.stop_gradient===true?{stop_gradient:true}:{}),...(typeof part.source_occurrence==='string'?{source_occurrence:part.source_occurrence}:{}) }];
    return Object.fromEntries(Object.entries(value).map(([name, item]) =>
      [name, materializeNativeStateSentinels(item, name)]));
  }
  return value;
}

export async function restoreNativeStates(remote: HttpNeuraleseStore, archive: NeuraleseStore | undefined,
    ids: readonly string[], known?: Set<string>, all = false): Promise<void> {
  if (all) for (const id of ids) known?.delete(id);
  const missing: string[] = [];
  for (const id of ids) {
    const local = await archive?.getNativeState?.(id);
    if (local) {
      const absent = await restoreBlocks(remote, archive, nativeStateArtifactDependencies(local));
      if (absent.length) throw new Error(`neuralese-unknown-native-artifact: restore source adapter coefficients and bases ${absent.join(', ')}`);
    }
    if (known?.has(id)) continue;
    if (await remote.hasNativeState(id)) { known?.add(id); continue; }
    const artifact = local;
    if (!artifact) { missing.push(id); continue; }
    const { id: _, ...input } = artifact;
    const restored = await remote.putNativeState(input);
    if (restored.id !== id) throw new Error(`native-state identity mismatch while restoring ${id}`);
    known?.add(id);
  }
  if (missing.length) throw new Error(`neuralese-unknown-native-state: ${missing.join(', ')} neither on the server nor in the runtime store`);
}

async function withRestoredNativeStates<T>(remote: HttpNeuraleseStore, archive: NeuraleseStore | undefined,
    ids: readonly string[], run: () => Promise<T>, known?: Set<string>): Promise<T> {
  await restoreNativeStates(remote, archive, ids, known);
  try { return await run(); }
  catch (error) {
    if (!(error instanceof Error) || !/neuralese-unknown-native-state/.test(error.message)) throw error;
    await restoreNativeStates(remote, archive, ids, known, true);
    return await run();
  }
}

function normalizeReply(body: Json): Json {
  const choice = (body.choices as Json[] | undefined)?.[0];
  const message = choice?.message as Json | undefined;
  if (!message) return body;
  // The shared chat adapter derives conversation text. Keep actual wire parts
  // here so an assistant-only return retains its typed transport provenance.
  for (const call of (message.tool_calls ?? []) as Array<{ function?: { arguments?: unknown } }>) {
    const fn = call.function;
    if (!fn) continue;
    const parsed = typeof fn.arguments === 'string' ? JSON.parse(fn.arguments) : fn.arguments;
    // Tool argument leaves retain their actual wire parts until the typed
    // runtime boundary. Eval source and assistant text normalize separately.
    fn.arguments = JSON.stringify(parsed);
  }
  return body;
}

/** Archive the same exact reply closure for live turns and saved-turn replay. */
export async function archiveNeuraleseReplyArtifacts(reply: Json, remote: NeuraleseStore,
    store: NeuraleseStore | undefined, references: readonly unknown[] = []):
    Promise<{ blocks: Set<string>; native: Set<string> }> {
  const blocks = new Set<string>();
  if (store) for (const meta of ((reply.neuralese as Json | undefined)?.blocks ?? []) as NeuraleseBlockMeta[]) {
    if (!(await store.has(meta.id))) {
      const block = await remote.get(meta.id);
      if (!block) throw new Error(`neuralese-unknown-block: the server reported ${meta.id} but does not have it`);
      const { id: _, ...rest } = meta;
      const local = await store.put({ ...rest, dtype: block.meta.dtype, length: block.meta.length,
        width: block.meta.width, data: block.data });
      if (local.id !== meta.id) throw new Error(`neuralese block ${meta.id} arrived as ${local.id}: content IDs disagree`);
    }
    blocks.add(meta.id);
  }
  const nativeBlocks = ((reply.native_state as Json | undefined)?.blocks ?? []) as Array<{ id: string }>;
  const sampling = reply.x_natlang_rl_capture as { prompt?: Array<{ kind?: string; id?: string; source_state_id?: string }>;
    writes?: Array<{ state_id?: string; raw_state_id?: string }> } | undefined;
  const native = new Set([...nativeBlocks.map(meta => meta.id), ...requestNativeStateIds([reply, ...references])]);
  for (const observation of sampling?.prompt ?? [])
    if (observation.kind === 'native_state') for (const id of [observation.id, observation.source_state_id]) if (id) native.add(id);
  for (const write of sampling?.writes ?? [])
    for (const id of [write.state_id, write.raw_state_id]) if (id) native.add(id);
  for (const id of native) {
    if (!isNativeStateId(id)) throw new Error('neuralese-block-integrity: sampling receipt names an invalid native artifact');
    if (!store) continue;
    if (!store.putNativeState || !store.getNativeState)
      throw new Error('neuralese-native-archive-unavailable: configure a store that preserves native-state artifacts');
    await archiveNativeArtifactFrom(remote, store, id);
  }
  return { blocks, native };
}

/** A model-turn driver for a Neuralese server. It advertises `neuralese: true`. */
export function neuraleseServerModelTurn(options: NeuraleseServerOptions):
    ((request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => Promise<ModelTurn>) & { neuralese: true; blocks: HttpNeuraleseStore; decide: DecisionScorer } {
  const { store, neuraleseTemperature, neuraleseMaxLength, request: extra, onExchange, onTurn, owner, nativeStateModes,
    ...connection } = options;
  const http = { ...connection, headers: { ...connection.headers,
    ...(connection.apiKey ? { authorization: 'Bearer ' + connection.apiKey } : {}),
    ...(owner === undefined ? {} : { [OWNER_HEADER]: owner }) } };
  const remote = new HttpNeuraleseStore(http.endpoint, http.headers);
  const coefficientPolicy=extra?.x_natlang_native_coefficient_policy===undefined?undefined:validateCoefficientRuntime(extra.x_natlang_native_coefficient_policy);
  const uploaded = new Set<string>();
  const nativeUploaded = new Set<string>();
  // What the server declares (`/v1/neuralese/info`), read once; a server that does not declare streaming gets whole requests.
  let info: Promise<Json | null> | undefined;
  const serverInfo = (refresh = false) => { if (refresh) info = undefined; return info ??= fetchModel(http.endpoint.replace(/\/$/, '') + '/v1/neuralese/info', { headers: http.headers })
    .then(async response => response.ok ? await response.json() as Json : null).catch(() => null);
  };
  const whole = httpChatTransport({ ...http, stream: false });
  const streamed = httpChatTransport({ ...http, stream: true });
  // Exact native actor coefficients remain native model parameters, never GGUF LoRA transport.
  const adapterParts = async (bound: readonly { id: string; scale: number }[] = [], active: readonly AdapterBinding[] = activeAdapters()) => {
    if(coefficientPolicy)requireCoefficientAttestation(coefficientPolicy,await serverInfo());
    const adapters = coefficientBindings(coefficientPolicy,active,bound);
    return adapters;
  };
  const blockIds = (messages: unknown[], adapters: readonly { id: string }[]) =>
    [...requestBlockIds(messages), ...adapters.map(item => item.id)];
  const bindNativeStateContext = async (messages: unknown[], states: readonly { id: string }[], capabilities: Json | null) => {
    if (!states.length) return messages;
    const consumer = capabilities ? nativeStateConsumerFromInfo(capabilities) : undefined;
    if (!consumer || !store?.getNativeState)
      throw Object.assign(new Error('neuralese-native-state-unavailable: withNativeStates requires a declared native reader and an archived state store'),
        { code: 'neuralese-native-state-unavailable' });
    const parts: RuntimeContentPart[] = [];
    const seen = new Set<string>();
    for (const { id } of states) {
      if (seen.has(id)) continue;
      seen.add(id);
      const artifact = await store.getNativeState(id);
      if (!artifact)
        throw Object.assign(new Error(`neuralese-unknown-native-state: withNativeStates cannot resolve ${id} from its archive`),
          { code: 'neuralese-unknown-native-state' });
      const identity = { architecture: artifact.metadata.architecture,
        checkpoint_sha256: artifact.metadata.checkpoint_sha256,
        tokenizer_fingerprint: artifact.metadata.tokenizer_fingerprint,
        embedding_weight_sha256: artifact.metadata.embedding_weight_sha256 };
      const readerIdentity = { architecture: consumer.architecture,
        checkpoint_sha256: consumer.checkpoint_sha256,
        tokenizer_fingerprint: consumer.tokenizer_fingerprint,
        embedding_weight_sha256: consumer.embedding_weight_sha256 };
      if (!nativeModelIdentityEqual(identity, readerIdentity))
        throw Object.assign(new Error(`neuralese-native-state-identity: ${id} was written by another actor; write or semantically adapt it for the current exact reader`),
          { code: 'neuralese-native-state-identity' });
      parts.push({ type: 'native_state', id });
    }
    if (!Array.isArray(messages))
      throw Object.assign(new Error('neuralese-native-state-prompt: model call has no message list for stored state context'),
        { code: 'neuralese-native-state-prompt' });
    return [{ role: 'system', content: parts }, ...messages];
  };
  const requireExactNativeImports = (capture: unknown, ids: readonly string[]) => {
    const imports = (capture as { native_state_imports?: unknown } | undefined)?.native_state_imports;
    if (!Array.isArray(imports))
      throw Object.assign(new Error('neuralese-native-state-import: server returned no actual native-state import receipt'),
        { code: 'neuralese-native-state-import' });
    for (const id of new Set(ids)) {
      const receipt = imports.find(value => value && typeof value === 'object' && (value as Json).id === id) as Json | undefined;
      if (!receipt || receipt.mode !== 'exact-native-state' || receipt.observed_native_state_id !== id)
        throw Object.assign(new Error(`neuralese-native-state-import: server did not consume ${id} as exact native state`),
          { code: 'neuralese-native-state-import' });
    }
  };
  const transport: ChatTransport = async (body, signal, meta) => {
    const scopedNativeStates = activeNativeStates();
    const capabilities = await serverInfo(scopedNativeStates.length > 0);
    if (scopedNativeStates.length) body = { ...body,
      messages: await bindNativeStateContext(body.messages as unknown[], scopedNativeStates, capabilities) };
    const recorder=activeRecorder();
    const nativeRecording=(!!recorder||scopedNativeStates.length>0)&&(capabilities?.native_state as {enabled?:boolean}|undefined)?.enabled===true;
    if(nativeRecording)body={...body,x_natlang_native_recording:true};
    if(Array.isArray(capabilities?.capabilities)&&capabilities.capabilities.includes('native.protocol-recovery'))body={...body,x_natlang_native_protocol_recovery:true};
    if (body.guidance && capabilities && !serves(capabilities, 'guidance.check'))
      throw Object.assign(new Error('neuralese-guidance-unavailable: this model server does not implement the requested guidance; use a server with guidance support'), { code: 'neuralese-guidance-unavailable' });
    if (body.neuralese_template && capabilities) {
      const template = body.neuralese_template as Record<string, unknown>;
      const required = ['template', ...(template.value_type ? ['template.value-type'] : []),
        ...(template.argument_path ? ['template.argument-path'] : [])];
      for (const capability of required) if (!serves(capabilities, capability))
        throw Object.assign(new Error(`neuralese-${capability.replaceAll('.', '-')}-unavailable: use a server that implements this typed readout`),
          { code: `neuralese-${capability.replaceAll('.', '-')}-unavailable` });
    }
    const consumptionModes=coefficientStateModes(coefficientPolicy,requestNativeStateIds(materializeNativeStateSentinels(body.messages) as unknown[]),nativeStateModes);
    if (consumptionModes) body = { ...body, x_natlang_native_state_modes: consumptionModes };
    if(coefficientPolicy){body={...body};delete body.x_natlang_native_coefficient_policy;}
    const adapters = await adapterParts((body.x_natlang_adapters ?? []) as { id: string; scale: number }[]);
    const inner = (await serverInfo())?.stream === true ? streamed : whole;
    // A streamed reply is assembled here, where its body is post-processed; its deltas go to the turn's observer, and a
    // request re-sent after restoring blocks first voids what the failed attempt streamed.
    let emitted = false;
    const onDelta = meta?.onDelta && ((delta: ModelTurnDelta) => { emitted = true; meta.onDelta!(delta); });
    const preparedMessages = materializeNativeStateSentinels(body.messages) as unknown[];
    const nativeIds = requestNativeStateIds(preparedMessages);
    const reply = await withRestoredBlocks(remote, store, blockIds(preparedMessages, adapters), async () =>
      await withRestoredNativeStates(remote, store, nativeIds, async () => {
        if (onDelta && emitted) { onDelta({ type: 'reset' }); emitted = false; }
        const requestBody = adapters.length ? { ...body, messages: preparedMessages, x_natlang_adapters: adapters } :
          { ...body, messages: preparedMessages };
        const answer = await inner(requestBody, signal);
        return isStream(answer) ? await assembleChatCompletion(answer, onDelta) : answer;
      }, nativeUploaded), uploaded);
    if (scopedNativeStates.length)
      requireExactNativeImports(reply.x_natlang_rl_capture, scopedNativeStates.map(state => state.id));
    const archived = await archiveNeuraleseReplyArtifacts(reply, remote, store);
    for (const id of archived.blocks) uploaded.add(id);
    const archiveIds = archived.native;
    for (const id of archiveIds) nativeUploaded.add(id);
    if(recorder){
      const message=(reply.choices as Json[]|undefined)?.[0]?.message as Json|undefined;
      const capture=reply.x_natlang_rl_capture as Record<string,unknown>|undefined;
      if(nativeRecording&&(!capture||!['natlang.native-rl-sampling/1','natlang.native-rl-sampling/2','natlang.native-rl-sampling/3','natlang.native-rl-sampling/4','natlang.native-structural-capture/1','natlang.native-structural-capture/2','natlang.native-teacher-forced-capture/1'].includes(String(capture.schema))||
          !Array.isArray(capture.prompt)||!Array.isArray(capture.samples)))
        throw Object.assign(new Error('neuralese-native-grad-invalid: native learning needs the actual server sampling receipt'),{code:'neuralese-native-grad-invalid'});
      if(nativeRecording&&archiveIds.size&&!store)
        throw Object.assign(new Error('neuralese-native-archive-unavailable: configure a durable native store before recording state communication'),{code:'neuralese-native-archive-unavailable'});
      const native=nativeRecording?{schema:'natlang.native-recorded-turn/1' as const,kind:'generation' as const,id:`native-turn:${globalThis.crypto.randomUUID()}`,
        sampling:structuredClone(capture!),retry:meta?.retry??0,...(meta?.invocation_id?{invocation_id:meta.invocation_id}:{})}:undefined;
      if(native)reply.x_natlang_native_recorded_turn=native.id;
      if(message)recorder.record({messages:structuredClone((nativeRecording?preparedMessages:body.messages) as unknown[]),
        tools:body.tools?structuredClone(body.tools as unknown[]):undefined,reply:structuredClone(message),
        blocks:structuredClone([...(((reply.neuralese as Json|undefined)?.blocks??[]) as Json[]),
          ...(((reply.native_state as Json|undefined)?.blocks??[]) as Json[])]),
        ...(adapters.length?{adapters:[...adapters]}:{}),
        ...(consumptionModes?{native_state_modes:{...consumptionModes}}:{}),...(native?{native}:{})});
    }
    return normalizeReply(reply);
  };
  const request: Json = { ...extra,
    ...(neuraleseTemperature === undefined ? {} : { neuralese_temperature: neuraleseTemperature }),
    ...(neuraleseMaxLength === undefined ? {} : { neuralese_max_length: neuraleseMaxLength }) };
  const driver = chatCompletionModelTurn(transport, { request, onExchange, onTurn });
  // Decision readout: one prompt pass, every option scored from its cache (serve/grad.py `decide`).
  type Item = Parameters<DecisionScorer>[0];
  type Settled = PromiseSettledResult<DecisionScores>;
  // A coalesced decision is scored in the batch's flush, whose asynchronous context is another caller's (or none): the
  // recorder and the dynamically scoped adapters are taken where the decision was asked for (`scope`), so concurrent
  // objectives each record their own turn and adapters do not leak between batched callers.
  type Scope = { recorder: TurnRecorder | undefined; adapters: readonly AdapterBinding[]; nativeStates: readonly { id: string }[] };
  const scopes = new WeakMap<object, Scope>();
  const here = (): Scope => ({ recorder: activeRecorder(), adapters: activeAdapters(), nativeStates: activeNativeStates() });
  const scopeOf = (item: object): Scope => scopes.get(item) ?? here();
  const prepare = async (messages: unknown[], replies: readonly string[], bound: readonly { id: string; scale: number }[] | undefined,
      scope: Scope = here()):Promise<Item> => {
    const adapters = await adapterParts(bound ?? [], scope.adapters);
    const info = await serverInfo(scope.nativeStates.length > 0);
    const native=info?.native_state as {enabled?:boolean}|undefined;
    const preparedMessages=materializeNativeStateSentinels(await bindNativeStateContext(messages,scope.nativeStates,info)) as unknown[];
    const modes=coefficientStateModes(coefficientPolicy,requestNativeStateIds(preparedMessages),nativeStateModes);
    if(!native?.enabled)scope.recorder?.record({messages:structuredClone(messages),reply:{role:'assistant',content:null},blocks:[],
      decision:{options:[...replies]},...(adapters.length?{adapters:[...adapters]}:{}),...(modes?{native_state_modes:modes}:{})});
    return {messages:preparedMessages,options:[...replies],...(adapters.length?{adapters:[...adapters]}:{}),
      ...(modes?{native_state_modes:modes}:{}),...(native?.enabled&&(scope.recorder||scope.nativeStates.length)?{native_recording:true}:{})};
  };
  const finish = async (item:Item,scope:Scope,scores:DecisionScores):Promise<DecisionScores> => {
    if(!item.native_recording)return scores;
    const capture=scores.native_decision_capture;
    if(capture?.schema!=='natlang.native-decision-capture/1'||!Array.isArray(capture.prompt)||
        !Array.isArray(capture.continuations)||capture.continuations.length!==item.options.length)
      throw Object.assign(new Error('neuralese-native-grad-invalid: native decision learning needs its actual rendered continuation receipt'),{code:'neuralese-native-grad-invalid'});
    if(scope.nativeStates.length)requireExactNativeImports(capture,scope.nativeStates.map(state=>state.id));
    if(!scope.recorder)return scores;
    const ids=new Set<string>(requestNativeStateIds(item.messages));
    for(const segment of capture.prompt as {kind?:string;id?:string;source_state_id?:string}[])if(segment.kind==='native_state')for(const id of [segment.id,segment.source_state_id])if(id)ids.add(id);
    for(const receipt of (capture.native_state_imports??[]) as {id?:string;observed_native_state_id?:string}[])
      for(const id of [receipt.id,receipt.observed_native_state_id])if(id)ids.add(id);
    if(ids.size&&!store)throw Object.assign(new Error('neuralese-native-archive-unavailable: configure a durable native store before recording decision inputs'),{code:'neuralese-native-archive-unavailable'});
    for(const id of ids){await archiveNativeArtifactFrom(remote,store!,id);nativeUploaded.add(id);}
    const native={schema:'natlang.native-recorded-turn/1' as const,kind:'decision' as const,
      id:`native-turn:${globalThis.crypto.randomUUID()}`,decision:structuredClone(capture),retry:0};
    scope.recorder.record({messages:structuredClone(item.messages),reply:{role:'assistant',content:null},blocks:[],
      decision:{options:[...item.options]},...(item.adapters?.length?{adapters:[...item.adapters]}:{}),
      ...(item.native_state_modes?{native_state_modes:{...item.native_state_modes}}:{}),native});
    return {...scores,x_natlang_native_recorded_turn:native.id};
  };
  const restore = async (item:Item,run:()=>Promise<DecisionScores>):Promise<DecisionScores> =>
    withRestoredBlocks(remote,store,blockIds(item.messages,item.adapters??[]),()=>
      withRestoredNativeStates(remote,store,requestNativeStateIds(item.messages),run,nativeUploaded),uploaded);
  const issue = async (item:Item,signal?:AbortSignal):Promise<DecisionScores> => restore(item,async()=>{
    const response=await fetchModel(http.endpoint.replace(/\/$/,'')+'/v1/neuralese/decide',{method:'POST',signal,
      headers:{'content-type':'application/json',...http.headers},body:JSON.stringify(item)});
    if(response.status===404||response.status===501)throw new Error('decision-unsupported: the server does not serve /v1/neuralese/decide');
    if(!response.ok)throw new Error(`neuralese decide HTTP ${response.status}: ${(await response.text()).slice(0,2000)}`);
    return await response.json() as DecisionScores;
  });
  const decide:DecisionScorer=async(request,signal)=>{
    const scope=scopeOf(request),item=await prepare(request.messages,request.options,request.adapters,scope);
    return finish(item,scope,await issue(item,signal));
  };
  const perItem=(items:Item[],signal?:AbortSignal)=>Promise.allSettled(items.map(item=>issue(item,signal)));
  const batched=explicitBatchScoreMany({url:http.endpoint.replace(/\/$/,'')+'/v1/natlang/score',headers:http.headers},perItem);
  const scoreMany=async(items:Item[],signal?:AbortSignal):Promise<Settled[]>=>{
    const prepared:({index:number;item:Item;scope:Scope;key:string}|undefined)[]=[],results:Settled[]=new Array(items.length);
    await Promise.all(items.map(async(original,index)=>{
      try{
        const scope=scopeOf(original),item=await prepare(original.messages,original.options,original.adapters,scope);
        await restore(item,async()=>({log_probs:[]}));
        prepared[index]={index,item,scope,key:JSON.stringify([item.messages,item.adapters])};
      }catch(error){results[index]={status:'rejected',reason:error};}
    }));
    const order=prepared.filter(entry=>entry!==undefined).sort((a,b)=>a.key<b.key?-1:a.key>b.key?1:a.index-b.index);
    const scored=await (serves((await serverInfo())??{},'score')?batched:perItem)(order.map(entry=>entry.item),signal);
    await Promise.all(order.map(async(entry,position)=>{
      const result=scored[position]!;
      if(result.status==='rejected'){results[entry.index]=result;return;}
      try{results[entry.index]={status:'fulfilled',value:await finish(entry.item,entry.scope,result.value)};}
      catch(error){results[entry.index]={status:'rejected',reason:error};}
    }));
    return results;
  };
  const coalesced = coalescingScorer(Object.assign((...args: Parameters<DecisionScorer>) => decide(...args), { scoreMany }));
  const scoped: DecisionScorer = Object.assign((request: Item, signal?: AbortSignal) => {
    scopes.set(request, here());
    return coalesced(request, signal);
  }, { scoreMany: coalesced.scoreMany });
  return Object.assign(driver, { neuralese: true as const, blocks: remote, decide: scoped });
}
