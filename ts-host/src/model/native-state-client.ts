/** Exact native state service plumbing shared by typed writes, body init and views. */
import { fetchModel } from '../model/chat-completion.js';
import { HttpNeuraleseStore, requestNativeStateIds, restoreBlocks } from '../model/neuralese-server.js';
import { archiveNativeTensorFrom, nativeStateBlockMeta, type NativeStateArtifact, type NeuraleseBlockMeta,
  type NeuralesePort, type NeuraleseStore, type NativeAdapterBinding } from '../native/neuralese-store.js';
import { hexDigest } from '../native/hash.js';
import { neuraleseServerInfo, serves, type NeuraleseServerInfo } from '../model/neuralese-info.js';

export type NativeAdapterStateConsumption = { schema: 'natlang.native-adapter-state-consumption/1';
  mode: 'same-base-adapter-state'; source_state_id: string; source_native_identity: Record<string, string>;
  target_native_identity: Record<string, string>; base_native_identity: Record<string, string>;
  source_adapter_binding: NativeAdapterBinding | null; target_adapter_binding: NativeAdapterBinding | null; target_position_start: number; tokens: number };

export type NativeSourceCapture = { policy: { checkpoint_sha256: string; tokenizer_fingerprint: string;
  embedding_weight_sha256: string; architecture: string };
  segments: ({ kind: 'tokens'; token_ids: number[]; producing: boolean[] } |
    { kind: 'native_state'; id: string; native_identity: Record<string, string>; adapter_state_consumption?: NativeAdapterStateConsumption })[];
  payload_start: number; payload_stop: number; imports?: { id: string; mode: string; observed_native_state_id?: string;
    adapter_state_consumption?: NativeAdapterStateConsumption;
    semantic_adaptation?: { adapter_sha256: string; source_native_identity: Record<string, string>; target_native_identity: Record<string, string> } }[] };
export type NativeTextResult = { id: string; native_state: Record<string, unknown>; source_capture: NativeSourceCapture;
  source_text_sha256: string; init?: Record<string, unknown>; gate?: null };
export type NativeClientOptions = { endpoint: string; headers?: Record<string, string>; store: NeuraleseStore;
  info?: NeuraleseServerInfo };

const failure = (code: string, detail: string): never => { throw Object.assign(new Error(`${code}: ${detail}`), { code }); };

export async function archiveNativeState(options: NativeClientOptions, id: string): Promise<NativeStateArtifact> {
  if (!options.store.putNativeState || !options.store.getNativeState)
    return failure('neuralese-native-archive-unavailable', 'configure a store that preserves native multi-tensor state');
  const found = await options.store.getNativeState(id);
  const remote = new HttpNeuraleseStore(options.endpoint, options.headers);
  const artifact = found ?? await remote.getNativeState(id);
  if (!artifact) return failure('neuralese-unknown-native-state', `native source ${id} is unavailable; restore its exact artifact`);
  for (const binding of artifact.metadata.adapter_binding?.adapters ?? []) await archiveNativeTensorFrom(remote, options.store, binding.id);
  if (found) return found;
  const { id: _, ...input } = artifact;
  const saved = await options.store.putNativeState(input);
  if (saved.id !== id) return failure('neuralese-block-integrity', `native source ${id} changed during archival`);
  return saved;
}

/** Restore every native part before a contextual write, including nested tool arguments. */
export async function restoreNativeInputs(options: NativeClientOptions, value: unknown): Promise<void> {
  const remote = new HttpNeuraleseStore(options.endpoint, options.headers);
  for (const id of requestNativeStateIds([value])) {
    if (await remote.hasNativeState(id)) continue;
    const artifact = await options.store.getNativeState?.(id);
    if (!artifact) return failure('neuralese-unknown-native-state', `restore ${id} to the configured native archive before initialization`);
    const { id: _, ...input } = artifact;
    if ((await remote.putNativeState(input)).id !== id)
      return failure('neuralese-block-integrity', `native input ${id} changed during restoration`);
  }
}

function validateSource(result: NativeTextResult, artifact: NativeStateArtifact, text: string, info: NeuraleseServerInfo): void {
  if (result.source_text_sha256 !== hexDigest(text) || artifact.metadata.text !== text)
    return failure('neuralese-native-source-integrity', 'native writer must preserve the exact supplied source text');
  const source = result.source_capture;
  if (!source || !Array.isArray(source.segments) || !source.policy)
    return failure('neuralese-native-source-integrity', 'native writer must return its actual source token, role and import capture');
  for (const name of ['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'] as const)
    if (source.policy[name] !== artifact.metadata[name])
      return failure('neuralese-native-source-integrity', `source ${name} differs from the captured artifact`);
  const own: number[] = [], roles: boolean[] = [];
  for (const segment of source.segments) {
    if (segment.kind === 'tokens') {
      if (!Array.isArray(segment.token_ids) || !Array.isArray(segment.producing) ||
          segment.token_ids.length !== segment.producing.length ||
          segment.token_ids.some(token => !Number.isSafeInteger(token) || token < 0) ||
          segment.producing.some(role => typeof role !== 'boolean'))
        return failure('neuralese-native-source-integrity', 'source token IDs and producing flags disagree');
      for (const token of segment.token_ids) own.push(token);
      for (const role of segment.producing) roles.push(role);
    } else if (segment.kind !== 'native_state' || !/^nzs1_[a-z2-7]{52}$/.test(segment.id))
      return failure('neuralese-native-source-integrity', 'source capture contains an invalid native import');
  }
  const start = source.payload_start, stop = source.payload_stop;
  const markers = (info.native_state as { role_marker_ids?: number[] }).role_marker_ids;
  if (!Array.isArray(markers) || markers.length !== 4 || own[start-1] !== markers[0] ||
      own[stop] !== markers[1] || roles[start-1] !== false || roles[stop] !== false)
    return failure('neuralese-native-source-integrity', 'native producer must include actual ordinary-role beginning and end markers around its payload');
  const ids = artifact.tensors.token_ids!, view = new DataView(ids.data.buffer, ids.data.byteOffset, ids.data.byteLength);
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(stop) || start < 0 || stop < start || stop > own.length ||
      stop-start !== ids.shape[0] || roles.slice(start, stop).some(role => !role) ||
      own.slice(start, stop).some((token, index) => BigInt(token) !== view.getBigInt64(index*8, true)))
    return failure('neuralese-native-source-integrity', 'captured payload tokens do not match the actual producer source span');
}

export async function nativeTextOperation(options: NativeClientOptions, operation: 'write-text' | 'init-body',
    body: Record<string, unknown> & { text: string }): Promise<{ result: NativeTextResult; meta: NeuraleseBlockMeta }> {
  const info = options.info ?? await neuraleseServerInfo(options.endpoint, { headers: options.headers });
  const capability = operation === 'write-text' ? 'native.text-write' : 'native.init-body';
  if (!(info.native_state as { enabled?: boolean } | undefined)?.enabled || !serves(info, capability))
    return failure(`neuralese-${capability.replaceAll('.', '-')}-unavailable`, 'use a native server implementing this exact producer operation');
  if (!options.store.putNativeState || !options.store.getNativeState)
    return failure('neuralese-native-archive-unavailable', 'configure a native-capable archive before writing typed state');
  await restoreNativeInputs(options, body);
  if (Array.isArray(body.adapters)) {
    const ids = body.adapters.map(entry => typeof entry === 'string' ? entry : (entry as { id?: string }).id);
    if (ids.some(id => typeof id !== 'string')) return failure('neuralese-native-adapter-invalid', 'native adapters require exact coefficient artifact IDs');
    const missing = await restoreBlocks(new HttpNeuraleseStore(options.endpoint, options.headers), options.store, ids as string[]);
    if (missing.length) return failure('neuralese-unknown-native-artifact', `restore native adapter coefficients and bases ${missing.join(', ')}`);
  }
  const response = await fetchModel(`${options.endpoint.replace(/\/$/, '')}/v1/neuralese/native-states/${operation}`, {
    method: 'POST', headers: { 'content-type': 'application/json', ...options.headers }, body: JSON.stringify(body) });
  if (!response.ok) return failure('neuralese-native-write', `${operation} failed (${response.status}): ${(await response.text()).slice(0, 1000)}`);
  const result = await response.json() as NativeTextResult;
  if (!/^nzs1_[a-z2-7]{52}$/.test(result.id)) return failure('neuralese-block-integrity', 'native writer returned no native content address');
  const artifact = await archiveNativeState(options, result.id);
  validateSource(result, artifact, body.text, info);
  for (const segment of result.source_capture.segments) if (segment.kind === 'native_state') {
    const imported = await archiveNativeState(options, segment.id);
    for (const key of ['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'] as const)
      if (segment.native_identity?.[key] !== imported.metadata[key] || segment.native_identity[key] !== result.source_capture.policy[key])
        return failure('neuralese-native-source-integrity', `source import ${segment.id} has an invalid ${key} binding`);
  }
  for (const imported of result.source_capture.imports ?? []) {
    const original = await archiveNativeState(options, imported.id);
    if (imported.mode === 'best-effort-text-retokenization') {
      if (imported.observed_native_state_id || imported.semantic_adaptation)
        return failure('neuralese-native-source-integrity', 'text fallback cannot attest an imported native KV artifact');
      continue;
    }
    if (!['exact-native-state', 'learned-semantic-state-deltas'].includes(imported.mode) || !imported.observed_native_state_id)
      return failure('neuralese-native-source-integrity', 'native import must retain its exact observed state and actual consumption mode');
    const observed = await archiveNativeState(options, imported.observed_native_state_id);
    if (observed.metadata.schema !== 'neuralese.native-state/1' ||
        !result.source_capture.segments.some(segment => segment.kind === 'native_state' && segment.id === imported.observed_native_state_id))
      return failure('neuralese-native-source-integrity', 'native import observation is missing from the actual source stream');
    for (const key of ['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'] as const)
      if (observed.metadata[key] !== result.source_capture.policy[key])
        return failure('neuralese-native-source-integrity', 'native import observation differs from the source policy');
    if (imported.mode === 'exact-native-state') {
      if (imported.id !== imported.observed_native_state_id || imported.semantic_adaptation)
        return failure('neuralese-native-source-integrity', 'exact native import must preserve its original artifact');
    } else {
      const adaptation = imported.semantic_adaptation;
      const adapter = (info.native_state as { semantic_adapter_sha256?: string }).semantic_adapter_sha256;
      if (!adapter || adaptation?.adapter_sha256 !== adapter || original.metadata.schema === 'neuralese.semantic-state/1' && original.metadata.adapter_sha256 !== adapter)
        return failure('neuralese-native-source-integrity', 'learned native import must bind the configured adapter and original message');
      for (const key of ['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture'] as const)
        if (adaptation.source_native_identity[key] !== original.metadata[key] ||
            adaptation.target_native_identity[key] !== observed.metadata[key])
          return failure('neuralese-native-source-integrity', 'learned native import source/target identities differ from their artifacts');
    }
  }
  return { result, meta: { ...nativeStateBlockMeta(artifact), ...(typeof body.type === 'string' ? { type: body.type } : {}),
    producer: { ...(body.producer as Record<string, unknown> | undefined), native_source_capture: result.source_capture,
      source_text_sha256: result.source_text_sha256 } } };
}

/** The existing write-port interface, backed by actual native producer execution. */
export async function nativeStateWritePort(options: NativeClientOptions): Promise<NeuralesePort> {
  const info = options.info ?? await neuraleseServerInfo(options.endpoint, { headers: options.headers });
  const native = info.native_state as { enabled?: boolean; architecture?: string; semantic_adapter_sha256?: string } | undefined;
  if (!native?.enabled || !native.architecture || !serves(info, 'native.text-write'))
    return failure('neuralese-native-text-write-unavailable', 'configure a native server with forced text producer support');
  const dialect = native.semantic_adapter_sha256 ? `semantic-state/${native.semantic_adapter_sha256}` : `native-state/${native.architecture}`;
  return { dialect, async write(text, extra = {}) {
    return (await nativeTextOperation({ ...options, info }, 'write-text', { text, ...extra })).meta;
  } };
}
