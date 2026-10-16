/** Native graph-learning HTTP client.
 *
 * Native learning operates on captured invocation graphs and named tensor
 * artifacts; it cannot consume the old flattened `nz1` GradSession payload.
 */
import { validateNativeAdapterBinding, type NativeAdapterBinding, nativeModelIdentityEqual, type NativeModelIdentity, type NativeTensorArtifact, type NeuraleseStore } from '../native/neuralese-store.js';
import { archiveNativeTensorArtifact, restoreNativeTensorArtifacts } from './native-tensor-client.js';

export type NativeGraphRow = Record<string, unknown> & { schema: 'natlang.neuralese-native-graph/1'; split: string };
export type NativeGradientTarget = { kind: 'model-parameters'; names: string[] } |
  { kind: 'native-state'; id: string; names?: string[] } | { kind: 'adapter'; id: string; names?: string[] };
export type NativePolicyObjectiveBinding = { kind: 'native-policy'; path: string; sha256: string };
export type NativeOptimizerSpec = { name: 'sgd' | 'adam'; learning_rate?: number; weight_decay?: number;
  beta1?: number; beta2?: number; eps?: number; momentum?: number };

async function post<T>(endpoint: string, path: string, body: unknown,
    headers: Record<string, string> = {}): Promise<T> {
  const response = await fetch(endpoint.replace(/\/$/, '') + path, { method: 'POST',
    headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
  if (!response.ok) {
    const error = await response.json().catch(() => ({})) as { error?: string; message?: string };
    throw Object.assign(new Error(`${error.error ?? 'neuralese-native-learning'}: ${error.message ?? response.statusText}`),
      { code: error.error ?? 'neuralese-native-learning', status: response.status });
  }
  return await response.json() as T;
}

/** Request gradients from the exact graph/actor and archive the named result. */
export async function nativeGraphGradient(endpoint: string, store: NeuraleseStore, request: {
  graph: NativeGraphRow; target: NativeGradientTarget; nativeIdentity: NativeModelIdentity;
  adapterGradientPrecision?: 'storage' | 'fp32';
  nativeStateModes?: Record<string, 'same-base-adapter-state'>;
  objective?: 'graph-ce' | NativePolicyObjectiveBinding; adapters?: (string | { id: string; scale?: number })[]; headers?: Record<string, string>;
}): Promise<{ loss: number; stats: unknown; gradient: NativeTensorArtifact; native_identity: NativeModelIdentity; objective: string }> {
  if (request.graph.split !== 'train') throw new RangeError('native graph gradients accept train rows only');
  if (request.adapters?.length) {
    const ids = request.adapters.map(entry => typeof entry === 'string' ? entry : entry.id);
    const missing = await restoreNativeTensorArtifacts(endpoint, store, ids, request.headers);
    if (missing.length) throw new Error(`neuralese-unknown-native-artifact: restore adapter coefficients and bases for ${missing.join(', ')}`);
  }
  const reply = await post<{ loss: number; stats: unknown; gradient: { id: string };
    native_identity: NativeModelIdentity; objective: string }>(
    endpoint, '/v1/neuralese/grad', { schema: 'natlang.native-gradient-request/1', graph: request.graph,
      target: request.target, native_identity: request.nativeIdentity, adapters: request.adapters,
      adapter_gradient_precision: request.adapterGradientPrecision, native_state_modes: request.nativeStateModes, objective: request.objective ?? 'graph-ce' }, request.headers);
  if (!nativeModelIdentityEqual(reply.native_identity, request.nativeIdentity))
    throw new Error('neuralese-native-identity: gradient reply is bound to another checkpoint');
  const gradient = await archiveNativeTensorArtifact(endpoint, store, reply.gradient.id, request.headers);
  if (!nativeModelIdentityEqual(gradient.metadata.native_identity, request.nativeIdentity))
    throw new Error('neuralese-native-gradient-binding: archived gradient belongs to another checkpoint');
  if (gradient.metadata.kind !== 'parameter-gradient' && gradient.metadata.kind !== 'state-gradient' && gradient.metadata.kind !== 'adapter-gradient')
    throw new Error('neuralese-native-gradient-kind: server returned a non-gradient tensor artifact');
  if (request.target.kind === 'adapter' && (gradient.metadata.kind !== 'adapter-gradient' ||
      gradient.metadata.source_artifact_id !== request.target.id))
    throw new Error('neuralese-native-gradient-binding: coefficient gradient does not bind its requested adapter');
  return { ...reply, gradient };
}

/** Publish an optimizer update as a new immutable effective checkpoint. */
export async function nativeOptimizerStep(endpoint: string, store: NeuraleseStore, request: {
  gradient: NativeTensorArtifact; nativeIdentity: NativeModelIdentity; optimizer: NativeOptimizerSpec;
  optimizerStateId?: string; headers?: Record<string, string>;
}): Promise<{ checkpoint: { path: string; sha256: string; schema: string; native_identity: NativeModelIdentity };
  parameters: NativeTensorArtifact; optimizer_state_id: string | null; native_identity: NativeModelIdentity; step: number }> {
  if (request.gradient.metadata.kind !== 'parameter-gradient')
    throw new RangeError('native optimizer needs a named model-parameter gradient');
  if (!nativeModelIdentityEqual(request.gradient.metadata.native_identity, request.nativeIdentity))
    throw new Error('neuralese-native-identity: gradient and live checkpoint identities differ');
  if (!store.putNativeTensorArtifact) throw new Error('neuralese-native-archive-unavailable: configure a named tensor store');
  const uploaded = await store.putNativeTensorArtifact({ metadata: request.gradient.metadata, tensors: request.gradient.tensors });
  if (uploaded.id !== request.gradient.id) throw new Error('neuralese-block-integrity: gradient artifact changed before update');
  const gradientMissing = await restoreNativeTensorArtifacts(endpoint, store, [request.gradient.id], request.headers);
  if (gradientMissing.length) throw new Error(`neuralese-unknown-native-artifact: gradient ${gradientMissing[0]} is not available to the server`);
  if (request.optimizerStateId) {
    const missing = await restoreNativeTensorArtifacts(endpoint, store, [request.optimizerStateId], request.headers);
    if (missing.length) throw new Error(`neuralese-unknown-native-artifact: optimizer state ${missing[0]} is not archived`);
  }
  const reply = await post<{ checkpoint: { path: string; sha256: string; schema: string;
    native_identity: NativeModelIdentity }; parameters: { id: string }; optimizer_state_id: string | null;
    native_identity: NativeModelIdentity; step: number }>(
    endpoint, '/v1/neuralese/optim', { schema: 'natlang.native-optimizer-request/1',
      native_identity: request.nativeIdentity, gradient_id: request.gradient.id,
      optimizer_state_id: request.optimizerStateId, optimizer: request.optimizer }, request.headers);
  const parameters = await archiveNativeTensorArtifact(endpoint, store, reply.parameters.id, request.headers);
  if (reply.optimizer_state_id) {
    const state = await archiveNativeTensorArtifact(endpoint, store, reply.optimizer_state_id, request.headers);
    if (state.metadata.kind !== 'optimizer-state' || state.metadata.source_artifact_id !== parameters.id ||
        !nativeModelIdentityEqual(state.metadata.native_identity, request.nativeIdentity))
      throw new Error('neuralese-native-update-binding: optimizer state does not belong to the published update');
  }
  if (reply.checkpoint.sha256 !== reply.native_identity.checkpoint_sha256 ||
      !nativeModelIdentityEqual(reply.checkpoint.native_identity, reply.native_identity) ||
      reply.native_identity.architecture !== request.nativeIdentity.architecture ||
      reply.native_identity.tokenizer_fingerprint !== request.nativeIdentity.tokenizer_fingerprint)
    throw new Error('neuralese-native-update-binding: checkpoint identity differs from the published model');
  if (parameters.metadata.kind !== 'parameter-update' || parameters.metadata.source_artifact_id !== request.gradient.id ||
      !nativeModelIdentityEqual(parameters.metadata.native_identity, request.nativeIdentity))
    throw new Error('neuralese-native-update-binding: published parameters do not match their parent checkpoint');
  return { ...reply, parameters };
}


/** Create one immutable exact-base native adapter and archive its frozen bases. */
export async function nativeAdapterCreate(endpoint: string, store: NeuraleseStore, request: {
  nativeIdentity: NativeModelIdentity; parameterNames: string[]; kind?: 'xs' | 'tiny';
  rank?: number; u?: number; seed?: number; headers?: Record<string, string>;
}): Promise<NativeTensorArtifact> {
  const reply = await post<{ id: string }>(endpoint, '/v1/neuralese/adapters', {
    kind: request.kind ?? 'xs', rank: request.rank ?? 8, u: request.u ?? 0, seed: request.seed ?? 0,
    parameter_names: request.parameterNames, native_identity: request.nativeIdentity }, request.headers);
  const adapter = await archiveNativeTensorArtifact(endpoint, store, reply.id, request.headers);
  if (adapter.metadata.kind !== 'adapter-coefficients' ||
      !nativeModelIdentityEqual(adapter.metadata.native_identity, request.nativeIdentity))
    throw new Error('neuralese-native-adapter-binding: created coefficients do not match the exact requested native base');
  return adapter;
}

/** Update adapter coefficients without changing the live model or frozen bases. */
export async function nativeAdapterOptimizerStep(endpoint: string, store: NeuraleseStore, request: {
  gradient: NativeTensorArtifact; nativeIdentity: NativeModelIdentity; optimizer: NativeOptimizerSpec;
  optimizerStateId?: string; headers?: Record<string, string>;
}): Promise<{ adapter: NativeTensorArtifact; optimizer_state_id: string; step: number;
  native_identity: NativeModelIdentity; effective_native_identity: NativeModelIdentity; adapter_binding: NativeAdapterBinding }> {
  if (request.gradient.metadata.kind !== 'adapter-gradient' ||
      !nativeModelIdentityEqual(request.gradient.metadata.native_identity, request.nativeIdentity))
    throw new RangeError('native coefficient update requires an exact-base adapter gradient');
  if (!store.putNativeTensorArtifact) throw new Error('neuralese-native-archive-unavailable: configure a named tensor archive');
  const { id: _, ...input } = request.gradient;
  if ((await store.putNativeTensorArtifact(input)).id !== request.gradient.id)
    throw new Error('neuralese-native-gradient-binding: gradient changed before update');
  const missing = await restoreNativeTensorArtifacts(endpoint, store,
    [request.gradient.id, ...(request.optimizerStateId ? [request.optimizerStateId] : [])], request.headers);
  if (missing.length) throw new Error(`neuralese-unknown-native-artifact: restore exact adapter/optimizer artifacts ${missing.join(', ')}`);
  const reply = await post<{ adapter: { id: string }; optimizer_state_id: string; step: number;
    native_identity: NativeModelIdentity; effective_native_identity: NativeModelIdentity; adapter_binding: NativeAdapterBinding }>(endpoint, '/v1/neuralese/optim', {
    schema: 'natlang.native-optimizer-request/1', native_identity: request.nativeIdentity,
    gradient_id: request.gradient.id, optimizer_state_id: request.optimizerStateId, optimizer: request.optimizer }, request.headers);
  const adapter = await archiveNativeTensorArtifact(endpoint, store, reply.adapter.id, request.headers);
  const state = await archiveNativeTensorArtifact(endpoint, store, reply.optimizer_state_id, request.headers);
  if (adapter.metadata.kind !== 'adapter-coefficients' || state.metadata.kind !== 'optimizer-state' ||
      !nativeModelIdentityEqual(reply.native_identity, request.nativeIdentity) ||
      !nativeModelIdentityEqual(adapter.metadata.native_identity, request.nativeIdentity) ||
      !nativeModelIdentityEqual(state.metadata.native_identity, request.nativeIdentity) ||
      state.metadata.source_artifact_id !== adapter.id || state.metadata.provenance.gradient_id !== request.gradient.id ||
      state.metadata.provenance.parent_adapter_id !== request.gradient.metadata.source_artifact_id)
    throw new Error('neuralese-native-update-binding: adapter continuation differs from its exact gradient and optimizer');
  validateNativeAdapterBinding(reply.adapter_binding, reply.effective_native_identity);
  const previous = request.gradient.metadata.provenance.adapter_binding as NativeAdapterBinding | undefined;
  const expectedAdapters = previous ? previous.adapters.map(entry => ({ ...entry,
    id: entry.id === request.gradient.metadata.source_artifact_id ? adapter.id : entry.id })) :
    [{ id: adapter.id, scale_hex: '0x1.0000000000000p+0' }];
  if (!nativeModelIdentityEqual(reply.adapter_binding.base_native_identity, request.nativeIdentity) ||
      reply.adapter_binding.adapters.length !== expectedAdapters.length ||
      expectedAdapters.some((entry, index) => entry.id !== reply.adapter_binding.adapters[index]?.id ||
        entry.scale_hex !== reply.adapter_binding.adapters[index]?.scale_hex))
    throw new Error('neuralese-native-update-binding: adapter update changed the actor order or scales');
  return { ...reply, adapter };
}
