/**
 * Transport to a native-state-capable model server (spec/NEURALESE_PORT.md, "Wire protocol"; S4 §3–4).
 *
 * The reference server is `python -m natlang_neuralese.serve` (training/neuralese). Requests are ordinary chat
 * completions whose message content and tool-call arguments carry native positional state references. The server
 * returns exact native-state artifacts and import receipts; projected value blocks are not a transport format.
 *
 * Replies stream when the server declares `stream: true` in `/v1/neuralese/info` (plans/STREAMING.md §1.2): text and
 * each written native state arrives as the turn's deltas (`ModelTurnOptions.onDelta`), and the final `x_natlang_message` is the
 * reply. A server that does not declare it (the llama.cpp fork until it serves SSE) gets whole requests.
 *
 * The driver keeps native state and named tensor custody in step across the runtime archive and model server.
 *
 * The runtime's store is also the archive that makes native artifacts outlive the server: `withRestoredNativeArtifacts` moves
 * exact native states and tensor artifacts runtime → server, and when the server reports an unknown native artifact it
 * re-uploads it from the archive and
 * retries once. A client names itself with `owner` (the `x-natlang-owner` header), so its pins and collections never
 * touch other sessions' artifacts.
 */
import { assembleChatCompletion, chatCompletionModelTurn, fetchModel, httpChatTransport, isStream, type ChatCompletionOptions, type ChatTransport,
  type HttpChatOptions } from './chat-completion.js';
import { coalescingScorer, explicitBatchScoreMany } from './scoring.js';
import type { DecisionScorer, DecisionScores, ModelTurn, ModelTurnDelta, ModelTurnOptions, ModelTurnRequest } from '../contracts.js';
import { activeNativeAdapters, activeNativeStates, activeRecorder, activeSoftStates, type NativeAdapterBinding, type TurnRecorder } from '../native/model-recording.js';
import { serves } from './neuralese-info.js';
import {validateCoefficientRuntime,requireCoefficientAttestation,coefficientBindings,coefficientStateModes} from './native-coefficient-policy.js';
import { isContentParts, nativeStateSentinel, nativeStateSentinelIds,
  isNativeStateId, runtimeTextToParts, type RuntimeContentPart } from '../native/neuralese.js';
import { nativeStateConsumerFromInfo } from '../native/neuralese.js';
import { decodeNativeState, encodeNativeState, nativeStateContentId, nativeStateBlockMeta,
  decodeNativeTensorArtifact, encodeNativeTensorArtifact, nativeTensorArtifactContentId, nativeTensorArtifactDependencies, nativeStateArtifactDependencies, archiveNativeArtifactFrom, archiveNativeTensorFrom,
  nativeTensorArtifactBlockMeta, nativeModelIdentityEqual, type NeuraleseBlockMeta,
  type NativeStateArtifact, type NativeTensorArtifact, type NeuraleseStore } from '../native/neuralese-store.js';

type Json = Record<string, unknown>;

/** Header naming the owner of requests (a session or runtime ID): its pins and collections are its own (serve/store.py). */
export const OWNER_HEADER = 'x-natlang-owner';

/** Whether `error` says the native server lost a named state or tensor artifact. */
export function isUnknownNativeArtifactError(error: unknown): boolean {
  const found = error as { providerCode?: unknown; code?: unknown; message?: unknown } | null;
  return found?.providerCode === 'neuralese-unknown-native-state' || found?.code === 'neuralese-unknown-native-state' ||
    found?.providerCode === 'neuralese-unknown-native-artifact' || found?.code === 'neuralese-unknown-native-artifact' ||
    (typeof found?.message === 'string' && /neuralese-unknown-native-(?:state|artifact)/.test(found.message));
}

/**
 * Upload to the server every native artifact of `ids` it lacks, from `archive` (the runtime's content-addressed store) or the
 * registered constants. Returns the IDs that neither has; the caller reports them in its own terms.
 */
export async function restoreNativeArtifacts(remote: NeuraleseStore, archive: NeuraleseStore | undefined,
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
    throw Object.assign(new TypeError(`native artifact restore requires an nzs1 state or nnt1 tensor ID, got ${JSON.stringify(id)}`),
      { code: 'neuralese-native-artifact-id' });
  };
  for (const id of new Set(ids)) await restore(id);
  return missing;
}

/**
 * Run a server request that reads `ids`, with those native artifacts on the server first. `known` caches IDs already uploaded
 * (they are not checked again); when the server still answers `neuralese-unknown-native-*` (it restarted or collected
 * them since), every block is checked and restored from `archive`, and the request runs once more.
 */
export async function withRestoredNativeArtifacts<T>(remote: NeuraleseStore, archive: NeuraleseStore | undefined,
    ids: readonly string[], run: () => Promise<T>, known?: Set<string>): Promise<T> {
  const restore = async (all: boolean) => {
    if (all) for (const id of ids) known?.delete(id);
    const pending = known ? ids.filter(id => !known.has(id)) : ids;
    const missing = await restoreNativeArtifacts(remote, archive, pending);
    if (missing.length) throw new Error(`neuralese-unknown-native-artifact: ${missing.join(', ')} ${missing.length === 1 ? 'is' : 'are'} ` +
      `neither on the server nor in the runtime's store`);
    for (const id of pending) known?.add(id);
  };
  await restore(false);
  try {
    return await run();
  } catch (error) {
    if (!isUnknownNativeArtifactError(error)) throw error;
    await restore(true);
    return await run();
  }
}

/** Native artifact endpoints as a store. The server has no projected block API. */
export class HttpNeuraleseStore implements NeuraleseStore {
  private readonly base: string;
  constructor(endpoint: string, private readonly headers: Record<string, string> = {}) {
    this.base = endpoint.replace(/\/$/, '') + '/v1/neuralese';
  }
  private async request(path: string, init: RequestInit = {}): Promise<Response> {
    return fetchModel(this.base + path, { ...init, headers: { ...this.headers, ...(init.headers as Record<string, string> ?? {}) } });
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
    return undefined;
  }
  async has(id: string): Promise<boolean> { return (await this.meta(id)) !== undefined; }
  async pin(id: string): Promise<void> {
    const family = id.startsWith('nzs1_') ? 'native-states' : id.startsWith('nnt1_') ? 'native-tensor-artifacts' : undefined;
    if (!family) throw new Error(`neuralese-native-archive-api: only native states and tensor artifacts can be pinned (${id})`);
    const response = await this.request(`/${family}/${id}/pin`, { method: 'POST' });
    if (!response.ok) throw new Error(`neuralese-unknown-native-artifact: ${id}`);
  }
  async unpin(id: string): Promise<void> {
    const family = id.startsWith('nzs1_') ? 'native-states' : id.startsWith('nnt1_') ? 'native-tensor-artifacts' : undefined;
    if (!family) throw new Error(`neuralese-native-archive-api: only native states and tensor artifacts can be unpinned (${id})`);
    await this.request(`/${family}/${id}/unpin`, { method: 'POST' });
  }
  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    const response = await this.request('/collect', { method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ referenced: [...referenced] }) });
    return ((await response.json()) as { removed: string[] }).removed;
  }
}

export type NeuraleseServerOptions = Omit<HttpChatOptions, 'stream'> & Pick<ChatCompletionOptions, 'onExchange' | 'onTurn'> & {
  /** The runtime's store: native state and tensor artifacts are uploaded and restored after a server restart. */
  store?: NeuraleseStore;
  /** This session's or runtime's ID on the server (`x-natlang-owner`): native artifact pins and collections are scoped to it. */
  owner?: string;
  /** Explicit raw-state exchange between actors of the exact same native base. */
  nativeStateModes?: Record<string, 'same-base-adapter-state'>;
  /** Extra request fields. */
  request?: Json;
};

function malformedNativePartList(value: unknown): boolean {
  return Array.isArray(value) && value.length > 0 && value.every(part => part && typeof part === 'object' &&
    ['text', 'native_state'].includes(part.type)) &&
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

/** One native-state import at its request location, retaining its declared writer occurrence when present. */
export type NativeStateRequestBinding = { id: string; source_occurrence?: string; path: string };

/**
 * Native IDs alone are enough to restore artifacts, but not to bind execution
 * graph edges: repeated same-ID imports can have distinct writer occurrences.
 * Keep every occurrence and a stable request path for graph consumers.
 */
export function requestNativeStateBindings(messages: readonly unknown[]): NativeStateRequestBinding[] {
  const bindings: NativeStateRequestBinding[] = [];
  const add = (id: unknown, source: unknown, path: string) => {
    if (!isNativeStateId(id)) return;
    if (source !== undefined && (typeof source !== 'string' || !source))
      throw Object.assign(new Error(`neuralese-native-grad-invalid: malformed native source occurrence at ${path}`),
        { code: 'neuralese-native-grad-invalid' });
    bindings.push({ id, ...(typeof source === 'string' ? { source_occurrence: source } : {}), path });
  };
  const visit = (value: unknown, key: string | undefined, path: string) => {
    if (key === 'arguments' && typeof value === 'string') {
      try { visit(JSON.parse(value), undefined, path); } catch (error) {
        if (error instanceof SyntaxError) return; // The transport reports malformed tool arguments separately.
        throw error;
      }
      return;
    }
    if (malformedNativePartList(value)) return;
    if (Array.isArray(value) && value.length > 0 && value.every(part => part && typeof part === 'object' &&
        (part.type === 'text' && typeof part.text === 'string' ||
         part.type === 'native_state' && isNativeStateId(part.id)))) {
      value.forEach((part, index) => {
        if (part.type === 'native_state') add(part.id, part.source_occurrence, `${path}[${index}]`);
      });
      return;
    }
    if (typeof value === 'string') {
      let occurrence = 0;
      for (const part of runtimeTextToParts(value)) if (part.type === 'native_state')
        add(part.id, part.source_occurrence, `${path}.sentinel[${occurrence++}]`);
      return;
    }
    if (Array.isArray(value)) {
      value.forEach((item, index) => visit(item, undefined, `${path}[${index}]`));
    } else if (value && typeof value === 'object') {
      const part = value as { type?: unknown; id?: unknown; source_occurrence?: unknown };
      if (part.type === 'native_state' && isNativeStateId(part.id)) {
        add(part.id, part.source_occurrence, path);
        return;
      }
      for (const [name, item] of Object.entries(value)) visit(item, name, `${path}.${name}`);
    }
  };
  visit(messages, undefined, 'messages');
  return bindings;
}

function materializeNativeStateSentinels(value: unknown, key?: string): unknown {
  if (Array.isArray(value)) {
    if (value.some(part => part && typeof part === 'object' && (part as Json).type === 'neuralese'))
      throw new TypeError('native chat content does not accept flat vector blocks');
    if (value.some(part => part && typeof part === 'object' && (part as Json).type === 'native_state' &&
        !isNativeStateId((part as Json).id)))
      throw new TypeError('native chat content requires an archived nzs1 state ID');
  }
  if (key === 'arguments' && typeof value === 'string') {
    try { return JSON.stringify(materializeNativeStateSentinels(JSON.parse(value))); }
    catch { return value; } // preserve malformed input for the server's existing argument error path
  }
  if (malformedNativePartList(value)) return value;
  if (typeof value === 'string') {
    return nativeStateSentinelIds(value).length ? runtimeTextToParts(value) : value;
  }
  if (Array.isArray(value) && value.length && value.every(part => part && typeof part === 'object' &&
      (part.type === 'text' && typeof part.text === 'string' ||
       part.type === 'native_state' && isNativeStateId(part.id))))
    return value.flatMap(part => {
      return part.type === 'text' ? runtimeTextToParts(part.text) : [part];
    });
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
      const absent = await restoreNativeArtifacts(remote, archive, nativeStateArtifactDependencies(local));
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
    Promise<{ native: Set<string> }> {
  const projectedBlocks = (reply.neuralese as { blocks?: unknown } | undefined)?.blocks;
  if (Array.isArray(projectedBlocks) && projectedBlocks.length)
    throw new Error('neuralese-projected-api-retired: server returned flat neuralese blocks');
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
  return { native };
}

/** A model-turn driver for a Neuralese server. It advertises `neuralese: true`. */
export function neuraleseServerModelTurn(options: NeuraleseServerOptions):
    ((request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => Promise<ModelTurn>) & { neuralese: true; artifacts: HttpNeuraleseStore; decide: DecisionScorer; neuraleseRuntime: () => Promise<import('../native/neuralese.js').NeuraleseRuntimeOptions> } {
  const { store, request: extra, onExchange, onTurn, owner, nativeStateModes,
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
  const adapterParts = async (bound: readonly { id: string; scale: number }[] = [], active: readonly NativeAdapterBinding[] = activeNativeAdapters()) => {
    if(coefficientPolicy)requireCoefficientAttestation(coefficientPolicy,await serverInfo());
    const adapters = coefficientBindings(coefficientPolicy,active,bound);
    return adapters;
  };
  const adapterArtifactIds = (adapters: readonly { id: string }[]) => adapters.map(item => item.id);
  const bindNativeStateContext = async (messages: unknown[], states: readonly { id: string; source_occurrence?: string; stop_gradient?: true }[], capabilities: Json | null) => {
    if (!states.length) return messages;
    const consumer = capabilities ? nativeStateConsumerFromInfo(capabilities) : undefined;
    if (!consumer || !store?.getNativeState)
      throw Object.assign(new Error('neuralese-native-state-unavailable: withNativeStates requires a declared native reader and an archived state store'),
        { code: 'neuralese-native-state-unavailable' });
    const parts: RuntimeContentPart[] = [];
    const alreadyPresent = new Set(requestNativeStateIds(messages));
    for (const binding of states) {
      const { id } = binding;
      // A caller may already have put this exact state in the request. Keep
      // those explicit occurrences and do not implicitly reinsert that ID.
      // Repeated scope entries remain repeated prepared imports.
      if (alreadyPresent.has(id)) continue;
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
      const sourceBaseIdentity = artifact.metadata.adapter_binding?.base_native_identity;
      if (!nativeModelIdentityEqual(identity, readerIdentity) &&
          (!sourceBaseIdentity || !nativeModelIdentityEqual(sourceBaseIdentity, readerIdentity)))
        throw Object.assign(new Error(`neuralese-native-state-identity: ${id} was written by another actor; write or semantically adapt it for the current exact reader`),
          { code: 'neuralese-native-state-identity' });
      const sourceOccurrence = binding.source_occurrence ?? activeRecorder()?.nativeStateWriter?.(id);
      parts.push({ type: 'native_state', id, ...(sourceOccurrence ? { source_occurrence: sourceOccurrence } : {}),
        ...(binding.stop_gradient ? { stop_gradient: true } : {}) });
    }
    if (!parts.length) return messages;
    if (!Array.isArray(messages))
      throw Object.assign(new Error('neuralese-native-state-prompt: model call has no message list for stored state context'),
        { code: 'neuralese-native-state-prompt' });
    return [{ role: 'system', content: parts }, ...messages];
  };
  const nativeStateModesFor = async (messages: readonly unknown[]) => {
    const ids = requestNativeStateIds(messages);
    const actualIds = new Set(ids);
    const modes: Record<string, 'same-base-adapter-state'> = Object.fromEntries(
      Object.entries(nativeStateModes ?? {}).filter(([id]) => actualIds.has(id)));
    if (store?.getNativeState) for (const id of ids) {
      const artifact = await store.getNativeState(id);
      if (artifact?.metadata.adapter_binding) modes[id] = 'same-base-adapter-state';
    }
    // A tuned call's own soft states were written by the exact base and are read by its adapted stack.
    if (activeNativeAdapters().length) for (const id of activeSoftStates()) if (actualIds.has(id)) modes[id] = 'same-base-adapter-state';
    return coefficientStateModes(coefficientPolicy, ids, Object.keys(modes).length ? modes : undefined);
  };
  const requireExactNativeImports = (capture: unknown, ids: readonly string[],
      modes?: Readonly<Record<string, 'same-base-adapter-state'>>) => {
    const imports = (capture as { native_state_imports?: unknown } | undefined)?.native_state_imports;
    if (!Array.isArray(imports))
      throw Object.assign(new Error(`neuralese-native-state-import: server returned no actual native-state import receipt (capture=${
        (capture as { schema?: unknown } | undefined)?.schema ?? 'missing'})`),
        { code: 'neuralese-native-state-import' });
    const prompt = (capture as { prompt?: unknown } | undefined)?.prompt;
    if (!Array.isArray(prompt))
      throw Object.assign(new Error('neuralese-native-state-import: server returned no exact native prompt occurrence list'),
        { code: 'neuralese-native-state-import' });
    for (const id of new Set(ids)) {
      const expectedCount = prompt.filter(value => value && typeof value === 'object' &&
        (value as Json).kind === 'native_state' && (value as Json).id === id).length;
      const receipts = imports.filter(value => value && typeof value === 'object' && (value as Json).id === id) as Json[];
      const expectedMode = modes?.[id] ?? 'exact-native-state';
      if (!expectedCount || receipts.length !== expectedCount || receipts.some(receipt =>
          receipt.mode !== expectedMode || receipt.observed_native_state_id !== id))
        throw Object.assign(new Error(`neuralese-native-state-import: server did not consume ${id} with declared mode ${expectedMode} (imports=${
          JSON.stringify(imports).slice(0, 1200)})`),
          { code: 'neuralese-native-state-import' });
    }
  };
  const transport: ChatTransport = async (body, signal, meta) => {
    const scopedNativeStates = activeNativeStates();
    let implicitNativeReadOccurrences: Json[] = [];
    const capabilities = await serverInfo(scopedNativeStates.length > 0);
    const preexistingStateIds = requestNativeStateIds(materializeNativeStateSentinels(body.messages as unknown[]) as unknown[]);
    const implicitStateBindings = scopedNativeStates.filter(state => !preexistingStateIds.includes(state.id));
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
    const consumptionModes=await nativeStateModesFor(materializeNativeStateSentinels(body.messages) as unknown[]);
    if (consumptionModes) body = { ...body, x_natlang_native_state_modes: consumptionModes };
    if(coefficientPolicy){body={...body};delete body.x_natlang_native_coefficient_policy;}
    const adapters = await adapterParts((body.x_natlang_adapters ?? []) as { id: string; scale: number }[]);
    const inner = (await serverInfo())?.stream === true ? streamed : whole;
    // A streamed reply is assembled here, where its body is post-processed; its deltas go to the turn's observer, and a
    // request re-sent after restoring native artifacts first voids what the failed attempt streamed.
    let emitted = false;
    const onDelta = meta?.onDelta && ((delta: ModelTurnDelta) => { emitted = true; meta.onDelta!(delta); });
    const preparedMessages = materializeNativeStateSentinels(body.messages) as unknown[];
    if (implicitStateBindings.length) {
      const first = preparedMessages[0] as { content?: unknown } | undefined;
      const parts = Array.isArray(first?.content) ? first.content as Array<{ type?: string; id?: string; source_occurrence?: string; stop_gradient?: true }> : [];
      const usedPartIndices = new Set<number>();
      const occurrences = implicitStateBindings.flatMap(binding => {
        const index = parts.findIndex((part, partIndex) => partIndex >= 0 && !usedPartIndices.has(partIndex) &&
          part.type === 'native_state' && part.id === binding.id);
        if (index < 0) return [];
        usedPartIndices.add(index);
        const part = parts[index]!;
        return [{ id: binding.id, positions: [[0, index]],
          ...(part.source_occurrence ? { source_occurrence: part.source_occurrence } : {}),
          ...(part.stop_gradient ? { stop_gradient: true } : {}) }];
      });
      if (occurrences.length !== implicitStateBindings.length)
        throw Object.assign(new Error('neuralese-native-state-import: scoped state was not present at its prepared prompt position'),
          { code: 'neuralese-native-state-import' });
      // This is local transport provenance for NativeToolAgent's graph recorder.
      // It is derived from the exact request sent and does not come from a shim.
      implicitNativeReadOccurrences = occurrences;
    }
    const nativeIds = requestNativeStateIds(preparedMessages);
    const reply = await withRestoredNativeArtifacts(remote, store, adapterArtifactIds(adapters), async () =>
      await withRestoredNativeStates(remote, store, nativeIds, async () => {
        if (onDelta && emitted) { onDelta({ type: 'reset' }); emitted = false; }
        const requestBody = adapters.length ? { ...body, messages: preparedMessages, x_natlang_adapters: adapters } :
          { ...body, messages: preparedMessages };
        const answer = await inner(requestBody, signal);
        return isStream(answer) ? await assembleChatCompletion(answer, onDelta) : answer;
      }, nativeUploaded), uploaded);
    if (implicitNativeReadOccurrences.length)
      (reply as Json).__natlang_native_read_occurrences = implicitNativeReadOccurrences;
    const rawCapture = reply.x_natlang_rl_capture as Record<string, unknown> | undefined;
    const nativeStateImports = (reply.native_state as { imports?: unknown } | undefined)?.imports;
    if (scopedNativeStates.length && !Array.isArray(nativeStateImports))
      throw Object.assign(new Error('neuralese-native-state-import: response lacks the authoritative native_state.imports receipt list'),
        { code: 'neuralese-native-state-import' });
    if (Array.isArray(nativeStateImports) && Array.isArray(rawCapture?.native_state_imports) &&
        JSON.stringify(nativeStateImports) !== JSON.stringify(rawCapture.native_state_imports))
      throw Object.assign(new Error('neuralese-native-state-import: capture and response native-state receipts disagree'),
        { code: 'neuralese-native-state-import' });
    // The engine carries actual import receipts alongside native_state
    // artifacts; the sampling capture carries prompts, samples and writes.
    // Combine those two server-owned pieces before validating or recording a
    // turn so exact external imports remain available to learning and replay.
    const capture = rawCapture && Array.isArray(nativeStateImports) ?
      { ...rawCapture, native_state_imports: structuredClone(nativeStateImports) } : rawCapture;
    if (scopedNativeStates.length)
      requireExactNativeImports(capture, scopedNativeStates.map(state => state.id), consumptionModes);
    const implicitById = new Map<string, Json[]>();
    for (const expected of implicitNativeReadOccurrences) {
      const id = expected.id as string;
      implicitById.set(id, [...(implicitById.get(id) ?? []), expected]);
    }
    for (const [id, expected] of implicitById) {
      const promptOccurrences = ((capture?.prompt ?? []) as Json[]).filter(segment => segment.kind === 'native_state' && segment.id === id);
      const importReceipts = ((capture?.native_state_imports ?? []) as Json[]).filter(receipt => receipt.id === id);
      // A scoped list is an occurrence list, not a set of IDs. Repeated IDs
      // deliberately create repeated prompt parts and receipts, each sharing
      // the same source node when they refer to the same produced state.
      const sourcesMatch = (actual: Json[]) => actual.length === expected.length &&
        actual.every((item, index) => item.source_occurrence === expected[index]!.source_occurrence);
      if (!sourcesMatch(promptOccurrences) || !sourcesMatch(importReceipts))
        throw Object.assign(new Error(`neuralese-native-state-import: ${expected.length} prepared occurrence(s) for ${id} differ from the server prompt and import receipts`),
          { code: 'neuralese-native-state-import' });
    }
    const archived = await archiveNeuraleseReplyArtifacts(reply, remote, store);
    const archiveIds = archived.native;
    for (const id of archiveIds) nativeUploaded.add(id);
    if(recorder){
      const message=(reply.choices as Json[]|undefined)?.[0]?.message as Json|undefined;
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
        blocks:structuredClone((((reply.native_state as Json|undefined)?.blocks??[]) as Json[])),
        ...(adapters.length?{adapters:[...adapters]}:{}),
        ...(consumptionModes?{native_state_modes:{...consumptionModes}}:{}),...(native?{native}:{})});
    }
    return normalizeReply(reply);
  };
  const request: Json = { ...extra };
  const driver = chatCompletionModelTurn(transport, { request, onExchange, onTurn });
  // Decision readout: one prompt pass, every option scored from its cache (serve/grad.py `decide`).
  type Item = Parameters<DecisionScorer>[0];
  type Settled = PromiseSettledResult<DecisionScores>;
  // A coalesced decision is scored in the batch's flush, whose asynchronous context is another caller's (or none): the
  // recorder and the dynamically scoped adapters are taken where the decision was asked for (`scope`), so concurrent
  // objectives each record their own turn and adapters do not leak between batched callers.
  type Scope = { recorder: TurnRecorder | undefined; adapters: readonly NativeAdapterBinding[]; nativeStates: readonly { id: string }[] };
  const scopes = new WeakMap<object, Scope>();
  const here = (): Scope => ({ recorder: activeRecorder(), adapters: activeNativeAdapters(), nativeStates: activeNativeStates() });
  const scopeOf = (item: object): Scope => scopes.get(item) ?? here();
  const prepare = async (messages: unknown[], replies: readonly string[], bound: readonly { id: string; scale: number }[] | undefined,
      scope: Scope = here()):Promise<Item> => {
    const adapters = await adapterParts(bound ?? [], scope.adapters);
    const info = await serverInfo(scope.nativeStates.length > 0);
    const native=info?.native_state as {enabled?:boolean}|undefined;
    const preparedMessages=materializeNativeStateSentinels(await bindNativeStateContext(messages,scope.nativeStates,info)) as unknown[];
    const modes=await nativeStateModesFor(preparedMessages);
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
    if(scope.nativeStates.length)requireExactNativeImports(capture,scope.nativeStates.map(state=>state.id),item.native_state_modes);
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
    withRestoredNativeArtifacts(remote,store,adapterArtifactIds(item.adapters??[]),()=>
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
  return Object.assign(driver, { neuralese: true as const, artifacts: remote, decide: scoped,
    neuraleseRuntime: async () => (await import('./native-state-client.js')).nativeStateRuntimeOptions(
      { endpoint: http.endpoint, headers: http.headers, store: remote }, { write: 'available' }) });
}
