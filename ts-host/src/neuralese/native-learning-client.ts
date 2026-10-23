/** Native graph-learning HTTP client.
 *
 * Native learning operates on captured invocation graphs and named tensor
 * artifacts; it cannot consume the old flattened `nz1` GradSession payload.
 */
import { nativeCanonicalJson, validateNativeInvocationActorReceipt, type NativeInvocationActorReceipt, validateNativeAdapterBinding, type NativeAdapterBinding, nativeModelIdentityEqual, type NativeModelIdentity, type NativeTensorArtifact, type NativeStateArtifact, type NeuraleseStore } from '../native/neuralese-store.js';
import { providerErrorBody } from '../model/chat-completion.js';
import { hexDigest } from '../native/hash.js';
import { HttpNeuraleseStore, restoreNativeStates, restoreNativeArtifacts } from '../model/neuralese-server.js';
import { archiveNativeState } from '../model/native-state-client.js';
import { archiveNativeTensorArtifact, restoreNativeTensorArtifacts } from './native-tensor-client.js';
import type { RecordedTurn } from '../native/model-recording.js';

export type NativeGraphRow = Record<string, unknown> & { schema: 'natlang.neuralese-native-graph/1'; split: string };
export type NativeGradientTarget = { kind: 'model-parameters'; names: string[] } |
  { kind: 'native-state'; id: string; names?: string[] } | { kind: 'adapter'; id: string; names?: string[] };
export type NativeRecordedGradientTarget = Exclude<NativeGradientTarget, { kind: 'model-parameters' }> & { root: string };
export type NativePolicyObjectiveBinding = { kind: 'native-policy'; path: string; sha256: string };
export type NativeOptimizerSpec = { name: 'sgd' | 'adam'; learning_rate?: number; weight_decay?: number;
  beta1?: number; beta2?: number; eps?: number; momentum?: number };

/** Differentiate public learning objectives against the exact native turns and occurrence graph they recorded. */
export async function nativeRecordedGraphGradient(endpoint: string, store: NeuraleseStore, request: {
  turns: readonly RecordedTurn[]; events: readonly Record<string, unknown>[]; terms: readonly Record<string, unknown>[];
  target: NativeGradientTarget; nativeIdentity: NativeModelIdentity;
  nativeStateModes?: Record<string, 'same-base-adapter-state'>; headers?: Record<string, string>;
}): Promise<{ loss: number; stats: unknown; gradient: NativeTensorArtifact; native_identity: NativeModelIdentity; objective: string }> {
  if (!request.turns.length || !request.events.length || !request.terms.length)
    throw new RangeError('native recorded learning needs actual turns, graph events and objective terms');
  if (!request.turns.some(turn => turn.native)) throw new RangeError('native recorded learning needs native turn receipts');
  if (request.target.kind === 'adapter') {
    const adapter = await store.getNativeTensorArtifact?.(request.target.id);
    if (!adapter || adapter.metadata.kind !== 'adapter-coefficients' ||
        !nativeModelIdentityEqual(adapter.metadata.native_identity, request.nativeIdentity))
      throw new Error('neuralese-native-adapter-binding: recorded target must be an archived exact-base nnt1 coefficient artifact');
    const names = request.target.names ?? Object.keys(adapter.tensors).sort();
    if (!names.length || names.some(name => !Object.hasOwn(adapter.tensors, name)))
      throw new RangeError('native adapter target names must select exact archived coefficient tensors');
  }
  const reply = await post<{ loss: number; stats: unknown; gradient: { id: string }; native_identity: NativeModelIdentity; objective: string }>(
    endpoint, '/v1/neuralese/grad', { native_recorded_objective: {
      schema: 'natlang.native-recorded-objective-request/1', turns: request.turns, events: request.events,
      terms: request.terms, target: request.target, native_state_modes: request.nativeStateModes } }, request.headers);
  if (!nativeModelIdentityEqual(reply.native_identity, request.nativeIdentity))
    throw new Error('neuralese-native-identity: recorded gradient reply is bound to another checkpoint');
  const gradient = await archiveNativeTensorArtifact(endpoint, store, reply.gradient.id, request.headers);
  const expectedKind = request.target.kind === 'adapter' ? 'adapter-gradient' :
    request.target.kind === 'native-state' ? 'state-gradient' : 'parameter-gradient';
  if (gradient.metadata.kind !== expectedKind || !nativeModelIdentityEqual(gradient.metadata.native_identity, request.nativeIdentity) ||
      (request.target.kind !== 'model-parameters' && gradient.metadata.source_artifact_id !== request.target.id))
    throw new Error('neuralese-native-gradient-binding: recorded gradient differs from its exact target and actor');
  return { ...reply, gradient };
}

/** Differentiate one authenticated recorded objective against several exact nnt1/nzs1 roots in one graph pass. */
export async function nativeRecordedGraphGradients(endpoint: string, store: NeuraleseStore, request: {
  turns: readonly RecordedTurn[]; events: readonly Record<string, unknown>[]; terms: readonly Record<string, unknown>[];
  targets: readonly NativeRecordedGradientTarget[]; nativeIdentity: NativeModelIdentity;
  nativeStateModes?: Record<string, 'same-base-adapter-state'>; headers?: Record<string, string>;
}): Promise<{ loss: number; stats: unknown; gradients: Record<string, NativeTensorArtifact>;
  native_identity: NativeModelIdentity; objective: string }> {
  if (!request.turns.length || !request.events.length || !request.terms.length || !request.targets.length)
    throw new RangeError('native recorded learning needs actual turns, graph events, objective terms and target roots');
  if (!request.turns.some(turn => turn.native)) throw new RangeError('native recorded learning needs native turn receipts');
  if (new Set(request.targets.map(target => target.root)).size !== request.targets.length)
    throw new RangeError('native recorded target roots must be unique');
  const remote = new HttpNeuraleseStore(endpoint, request.headers);
  const adapterIds = request.targets.filter((target): target is Extract<NativeRecordedGradientTarget, { kind: 'adapter' }> =>
    target.kind === 'adapter').map(target => target.id);
  if (adapterIds.length) {
    const missing = await restoreNativeTensorArtifacts(endpoint, store, adapterIds, request.headers);
    if (missing.length) throw new Error(`neuralese-unknown-native-artifact: restore exact adapter targets ${missing.join(', ')}`);
  }
  const stateIds = request.targets.filter((target): target is Extract<NativeRecordedGradientTarget, { kind: 'native-state' }> =>
    target.kind === 'native-state').map(target => target.id);
  if (stateIds.length) await restoreNativeStates(remote, store, stateIds);
  for (const target of request.targets) if (target.kind === 'adapter') {
    const adapter = await store.getNativeTensorArtifact?.(target.id);
    if (!adapter || adapter.metadata.kind !== 'adapter-coefficients' ||
        !nativeModelIdentityEqual(adapter.metadata.native_identity, request.nativeIdentity))
      throw new Error('neuralese-native-adapter-binding: recorded target must be an archived exact-base nnt1 coefficient artifact');
    const names = target.names ?? Object.keys(adapter.tensors).sort();
    if (!names.length || names.some(name => !Object.hasOwn(adapter.tensors, name)))
      throw new RangeError('native adapter target names must select exact archived coefficient tensors');
  }
  const reply = await post<{ loss: number; stats: unknown; gradients: Record<string, { id: string }>;
    native_identity: NativeModelIdentity; objective: string }>(endpoint, '/v1/neuralese/grad', {
    native_recorded_objective: { schema: 'natlang.native-recorded-objective-request/1', turns: request.turns,
      events: request.events, terms: request.terms, targets: request.targets,
      native_state_modes: request.nativeStateModes } }, request.headers);
  if (!nativeModelIdentityEqual(reply.native_identity, request.nativeIdentity))
    throw new Error('neuralese-native-identity: recorded gradient reply is bound to another checkpoint');
  const gradients: Record<string, NativeTensorArtifact> = {};
  for (const target of request.targets) {
    const id = reply.gradients?.[target.root]?.id;
    if (typeof id !== 'string') throw new Error(`neuralese-native-gradient-binding: missing gradient for target root ${target.root}`);
    const gradient = await archiveNativeTensorArtifact(endpoint, store, id, request.headers);
    const kind = target.kind === 'adapter' ? 'adapter-gradient' : 'state-gradient';
    if (gradient.metadata.kind !== kind || !nativeModelIdentityEqual(gradient.metadata.native_identity, request.nativeIdentity) ||
        gradient.metadata.source_artifact_id !== target.id)
      throw new Error(`neuralese-native-gradient-binding: recorded gradient differs from target root ${target.root}`);
    gradients[target.root] = gradient;
  }
  return { ...reply, gradients };
}

/** Execute a finite differentiable native learning program and archive each outer root gradient. */
export async function nativeDerivativeProgramGradient(endpoint: string, store: NeuraleseStore, program: {
  schema: 'natlang.native-derivative-program/1'; inputs: Record<string, { kind: string; id: string; names?: string[] }>;
  nodes: readonly Record<string, unknown>[]; outer: Record<string, unknown>; wrt: readonly string[];
  arithmetic: { schema: 'natlang.native-reference-arithmetic/1'; query_chunk_tokens: number };
}, headers?: Record<string, string>): Promise<{ loss: number; gradients: Record<string, NativeTensorArtifact>; execution_id: string }> {
  const remote = new HttpNeuraleseStore(endpoint, headers);
  const adapterIds = Object.values(program.inputs).filter(input => input.kind === 'adapter').map(input => input.id);
  const missing = await restoreNativeTensorArtifacts(endpoint, store, adapterIds, headers);
  if (missing.length) throw new Error(`neuralese-unknown-native-artifact: restore native derivative adapters ${missing.join(', ')}`);
  const histories = program.nodes.map(node => node.kind === 'optimizer' ? node.optimizer_state : null)
    .filter((id): id is string => typeof id === 'string' && id.startsWith('nnt1_'));
  const missingHistory = await restoreNativeTensorArtifacts(endpoint, store, histories, headers);
  if (missingHistory.length) throw new Error(`neuralese-unknown-native-artifact: restore native derivative optimizer history ${missingHistory.join(', ')}`);
  const stateIds = Object.values(program.inputs).filter(input => input.kind === 'native-state').map(input => input.id);
  if (stateIds.length) await restoreNativeStates(remote, store, stateIds);
  const reply = await post<{ loss: number; execution_id: string; gradients: Record<string, { id: string }> }>(
    endpoint, '/v1/neuralese/grad', { native_derivative_program: program }, headers);
  const gradients: Record<string, NativeTensorArtifact> = {};
  for (const root of program.wrt) {
    const input = program.inputs[root], id = reply.gradients?.[root]?.id;
    if (!input || typeof id !== 'string') throw new Error(`neuralese-native-derivative: missing gradient for root ${root}`);
    const gradient = await archiveNativeTensorArtifact(endpoint, store, id, headers);
    const kind = input.kind === 'adapter' ? 'adapter-gradient' : input.kind === 'native-state' ? 'state-gradient' : 'parameter-gradient';
    if (gradient.metadata.kind !== kind || (input.kind !== 'model-parameters' && gradient.metadata.source_artifact_id !== input.id))
      throw new Error(`neuralese-native-derivative: root ${root} gradient changed its exact target binding`);
    gradients[root] = gradient;
  }
  return { loss: reply.loss, gradients, execution_id: reply.execution_id };
}

async function post<T>(endpoint: string, path: string, body: unknown,
    headers: Record<string, string> = {}): Promise<T> {
  const response = await fetch(endpoint.replace(/\/$/, '') + path, { method: 'POST',
    headers: { 'content-type': 'application/json', ...headers }, body: JSON.stringify(body) });
  if (!response.ok) {
    const error = providerErrorBody(await response.json().catch(() => ({})));
    const code = typeof error?.code === 'string' ? error.code :
      typeof error?.error === 'string' ? error.error : 'neuralese-native-learning';
    const message = typeof error?.message === 'string' ? error.message : response.statusText;
    throw Object.assign(new Error(message.startsWith(`${code}:`) ? message : `${code}: ${message}`),
      { code, status: response.status });
  }
  return await response.json() as T;
}

/** Request gradients from the exact graph/actor and archive the named result. */
export async function nativeGraphGradient(endpoint: string, store: NeuraleseStore, request: {
  graph: NativeGraphRow; target: NativeGradientTarget; nativeIdentity: NativeModelIdentity;
  adapterGradientPrecision?: 'storage' | 'fp32';
  invocationAdapters?: Record<string, (string | { id: string; scale?: number })[]>;
  nativeStateModes?: Record<string, 'same-base-adapter-state'>;
  objective?: 'graph-ce' | NativePolicyObjectiveBinding; adapters?: (string | { id: string; scale?: number })[]; headers?: Record<string, string>;
}): Promise<{ loss: number; stats: unknown; gradient: NativeTensorArtifact; native_identity: NativeModelIdentity; objective: string }> {
  if (request.graph.split !== 'train') throw new RangeError('native graph gradients accept train rows only');
  const bindings = [...(request.adapters ?? []), ...Object.values(request.invocationAdapters ?? {}).flat()];
  if (bindings.length) {
    const ids = bindings.map(entry => typeof entry === 'string' ? entry : entry.id);
    const missing = await restoreNativeTensorArtifacts(endpoint, store, ids, request.headers);
    if (missing.length) throw new Error(`neuralese-unknown-native-artifact: restore adapter coefficients and bases for ${missing.join(', ')}`);
  }
  const external = request.graph.external_states as Record<string, {id:string}> | undefined;
  if (external) {
    const ids=Object.values(external).map(binding => binding.id);
    await restoreNativeStates(new HttpNeuraleseStore(endpoint,request.headers),store,ids);
    for (const id of ids) await archiveNativeState({endpoint,headers:request.headers,store},id);
  }
  const reply = await post<{ loss: number; stats: unknown; gradient: { id: string };
    native_identity: NativeModelIdentity; objective: string }>(
    endpoint, '/v1/neuralese/grad', { schema: 'natlang.native-gradient-request/1', graph: request.graph,
      target: request.target, native_identity: request.nativeIdentity, adapters: request.adapters, invocation_adapters: request.invocationAdapters,
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
  nativeIdentity: NativeModelIdentity; parameterNames: string[]; kind?: 'xs';
  rank?: number; seed?: number; headers?: Record<string, string>;
}): Promise<NativeTensorArtifact> {
  const reply = await post<{ id: string }>(endpoint, '/v1/neuralese/adapters', {
    kind: request.kind ?? 'xs', rank: request.rank ?? 8, seed: request.seed ?? 0,
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
  native_identity: NativeModelIdentity; effective_native_identity?: NativeModelIdentity; adapter_binding?: NativeAdapterBinding;
  invocation_actors?: NativeInvocationActorReceipt; actor_graph_sha256?: string; invocation_source_names?: Record<string,string> }> {
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
    native_identity: NativeModelIdentity; effective_native_identity?: NativeModelIdentity; adapter_binding?: NativeAdapterBinding;
  invocation_actors?: NativeInvocationActorReceipt; actor_graph_sha256?: string; invocation_source_names?: Record<string,string> }>(endpoint, '/v1/neuralese/optim', {
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
  const priorActors = request.gradient.metadata.provenance.invocation_actors as NativeInvocationActorReceipt | undefined;
  if (priorActors) {
    validateNativeInvocationActorReceipt(priorActors,request.nativeIdentity);
    if (!reply.invocation_actors || reply.actor_graph_sha256 !== hexDigest(nativeCanonicalJson(reply.invocation_actors)))
      throw new Error('neuralese-native-update-binding: invocation actor continuation needs its exact receipt hash');
    validateNativeInvocationActorReceipt(reply.invocation_actors,request.nativeIdentity);
    if (nativeCanonicalJson(reply.invocation_source_names ?? null) !== nativeCanonicalJson(request.gradient.metadata.provenance.invocation_source_names ?? null) ||
        nativeCanonicalJson(state.metadata.provenance.invocation_actors) !== nativeCanonicalJson(reply.invocation_actors) ||
        state.metadata.provenance.actor_graph_sha256 !== reply.actor_graph_sha256)
      throw new Error('neuralese-native-update-binding: optimizer archive changed its exact invocation continuation');
    if (Object.keys(priorActors.invocations).sort().join('\0') !== Object.keys(reply.invocation_actors.invocations).sort().join('\0'))
      throw new Error('neuralese-native-update-binding: invocation inventory changed during coefficient update');
    for (const [name,before] of Object.entries(priorActors.invocations)) {
      const after=reply.invocation_actors.invocations[name]!;
      const expected=before.adapter_binding ? {...before.adapter_binding,adapters:before.adapter_binding.adapters.map(entry =>
        ({...entry,id:entry.id === request.gradient.metadata.source_artifact_id ? adapter.id : entry.id}))} : null;
      if (nativeCanonicalJson(after.adapter_binding) !== nativeCanonicalJson(expected))
        throw new Error('neuralese-native-update-binding: invocation actor order, scale or unrelated coefficients changed');
    }
    if (reply.adapter_binding || reply.effective_native_identity)
      throw new Error('neuralese-native-update-binding: heterogeneous graph cannot be relabeled as one actor');
    return {...reply,adapter};
  }
  if (reply.invocation_actors || !reply.adapter_binding || !reply.effective_native_identity)
    throw new Error('neuralese-native-update-binding: coefficient update changed its actor contract');
  const currentBinding=reply.adapter_binding;
  validateNativeAdapterBinding(currentBinding, reply.effective_native_identity);
  const previous = request.gradient.metadata.provenance.adapter_binding as NativeAdapterBinding | undefined;
  const expectedAdapters = previous ? previous.adapters.map(entry => ({ ...entry,
    id: entry.id === request.gradient.metadata.source_artifact_id ? adapter.id : entry.id })) :
    [{ id: adapter.id, scale_hex: '0x1.0000000000000p+0' }];
  if (!nativeModelIdentityEqual(currentBinding.base_native_identity, request.nativeIdentity) ||
      currentBinding.adapters.length !== expectedAdapters.length ||
      expectedAdapters.some((entry, index) => entry.id !== currentBinding.adapters[index]?.id ||
        entry.scale_hex !== currentBinding.adapters[index]?.scale_hex))
    throw new Error('neuralese-native-update-binding: adapter update changed the actor order or scales');
  return { ...reply, adapter };
}


/** Optimize exact KV/hybrid coordinates, preserving their producer and full update ancestry. */
export async function nativeStateOptimizerStep(endpoint:string,store:NeuraleseStore,request:{
  gradient:NativeTensorArtifact;nativeIdentity:NativeModelIdentity;optimizer:NativeOptimizerSpec;
  optimizerStateId?:string;headers?:Record<string,string>;
}):Promise<{native_state_id:string;native_state:NativeStateArtifact;optimizer_state_id:string;update_artifact_id:string;step:number;native_identity:NativeModelIdentity}>{
  if(request.gradient.metadata.kind!=='state-gradient'||!nativeModelIdentityEqual(request.gradient.metadata.native_identity,request.nativeIdentity))
    throw new RangeError('native state update requires an exact-base named state gradient');
  if(!store.putNativeTensorArtifact)throw new Error('neuralese-native-archive-unavailable: configure a named tensor archive');
  const {id:_,...input}=request.gradient;
  if((await store.putNativeTensorArtifact(input)).id!==request.gradient.id)throw new Error('neuralese-native-gradient-binding: state gradient changed before upload');
  const remote=new HttpNeuraleseStore(endpoint,request.headers);
  const missing=await restoreNativeArtifacts(remote,store,[request.gradient.id,...(request.optimizerStateId?[request.optimizerStateId]:[])]);
  if(missing.length)throw new Error(`neuralese-unknown-native-artifact: restore exact state update dependencies ${missing.join(', ')}`);
  const reply=await post<{native_state_id:string;optimizer_state_id:string;update_artifact_id:string;step:number;native_identity:NativeModelIdentity}>(
    endpoint,'/v1/neuralese/optim',{schema:'natlang.native-optimizer-request/1',native_identity:request.nativeIdentity,
      gradient_id:request.gradient.id,optimizer_state_id:request.optimizerStateId,optimizer:request.optimizer},request.headers);
  if(!nativeModelIdentityEqual(reply.native_identity,request.nativeIdentity))throw new Error('neuralese-native-update-binding: state update changed its exact native base');
  const artifact=await archiveNativeState({endpoint,store,headers:request.headers},reply.native_state_id);
  const derivation=artifact.metadata.derivation;
  if(!derivation||derivation.kind!=='optimizer'||derivation.parent_state_id!==request.gradient.metadata.source_artifact_id||derivation.gradient_id!==request.gradient.id||
    derivation.step!==reply.step||derivation.update_artifact_id!==reply.update_artifact_id||derivation.optimizer_state_parent_id!==(request.optimizerStateId??null))
    throw new Error('neuralese-native-update-binding: state update differs from its exact gradient, parent or optimizer history');
  const history=await archiveNativeTensorArtifact(endpoint,store,reply.optimizer_state_id,request.headers);
  if(history.metadata.kind!=='optimizer-state'||history.metadata.source_artifact_id!==artifact.id||history.metadata.provenance.step!==reply.step)
    throw new Error('neuralese-native-update-binding: state optimizer history differs from the actual updated artifact');
  return {...reply,native_state:artifact};
}
