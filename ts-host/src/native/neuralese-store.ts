/**
 * Neuralese tensor store (S0 §3.4, S4 §2.4).
 *
 * Every soft value is an immutable, content-addressed block of vectors. The runtime's value model holds only
 * references (`{ $neuralese: { type, id } }`); bytes move between this store, `.nz` files and model servers.
 */
import { digest } from './hash.js';

export type NeuraleseDtype = 'f32' | 'f16' | 'bf16';

/** Metadata of one stored block. `id` is derived from dialect, shape, dtype and payload bytes. */
export type NeuraleseBlockMeta = {
  id: string;
  dialect: string;
  /** Canonical type string of the value, when the writer knows it. The ID does not depend on it. */
  type?: string;
  length: number;
  width: number;
  dtype: NeuraleseDtype;
  /** Invocation, file, combinator call or gradient step that produced the block. */
  producer?: Record<string, unknown>;
  /** Present for exact multi-tensor state; this metadata is not a flat-vector payload. */
  native_state?: NativeStateMetadata & { embedding_dtype: NativeStateDtype };
  /** Named gradients/parameters live in their own tensor domain, never a flat value payload. */
  native_tensor_artifact?: NativeTensorArtifactMetadata;
  /** Writing hit the runtime's hard maximum instead of stopping. */
  truncated?: boolean;
};

export type NeuraleseBlock = { meta: NeuraleseBlockMeta; data: Uint8Array };

export type NeuraleseBlockInput = Omit<NeuraleseBlockMeta, 'id'> & { data: Uint8Array };

export type NativeStateDtype = 'F16' | 'BF16' | 'F32' | 'F64' | 'I64';
export type NativeStateTensor = { dtype: NativeStateDtype; shape: number[]; data: Uint8Array };
export type NativeAdapterBinding = { schema: 'natlang.native-adapter/1'; base_native_identity: NativeModelIdentity;
  adapters: { id: string; scale_hex: string }[] };
export type NativeAdapterStateConsumption = { schema: 'natlang.native-adapter-state-consumption/1';
  mode: 'same-base-adapter-state'; source_state_id: string; source_native_identity: Record<string, string>;
  target_native_identity: Record<string, string>; base_native_identity: Record<string, string>;
  source_adapter_binding: NativeAdapterBinding | null; target_adapter_binding: NativeAdapterBinding | null; target_position_start: number; tokens: number };


export type NativeStateMetadata = { text: string; architecture: string;
  tokenizer_fingerprint: string; checkpoint_sha256: string; embedding_weight_sha256: string; layer_kinds: string[];
  adapter_binding?: NativeAdapterBinding } &
  ({ schema: 'neuralese.native-state/1' } | { schema: 'neuralese.semantic-state/1';
    source_model: string; adapter_sha256: string; offsets: number[][]; posterior: 'mean';
    offset_policy: 'exact-tokenizer-offsets' | 'actual-token-decoder-alignment' | 'actual-token-decoder-text-alignment' });
export type NativeStateArtifact = { id: string; metadata: NativeStateMetadata; tensors: Record<string, NativeStateTensor> };
export type NativeTensorDtype = 'F16' | 'BF16' | 'F32' | 'F64';
export type NativeTensorValue = { dtype: NativeTensorDtype; shape: number[]; data: Uint8Array };
export type NativeModelIdentity = { checkpoint_sha256: string; tokenizer_fingerprint: string;
  embedding_weight_sha256: string; architecture: string };

/** Compare exact model identity fields independently of JSON property order. */
export function nativeModelIdentityEqual(left: NativeModelIdentity, right: NativeModelIdentity): boolean {
  const keys = ['checkpoint_sha256','tokenizer_fingerprint','embedding_weight_sha256','architecture'] as const;
  return Object.keys(left).length === keys.length && Object.keys(right).length === keys.length &&
    keys.every(key => left[key] === right[key]);
}

export type NativeTensorArtifactMetadata = { schema: 'natlang.neuralese-native-tensor-artifact/1';
  kind: 'parameter-set' | 'parameter-gradient' | 'parameter-update' | 'state-gradient' | 'optimizer-state' | 'adapter-basis' | 'adapter-coefficients' | 'adapter-gradient';
  native_identity: NativeModelIdentity; source_artifact_id: string | null; provenance: Record<string, unknown>;
  tensors: { name: string; dtype: NativeTensorDtype; shape: number[] }[] };
export type NativeTensorArtifact = { id: string; metadata: NativeTensorArtifactMetadata;
  tensors: Record<string, NativeTensorValue>; canonicalMetadata?: string };

/** Existing typed-value metadata derived from a verified native artifact. */
export function nativeStateBlockMeta(artifact: Pick<NativeStateArtifact, 'id' | 'metadata'> &
    { tensors: Record<string, Pick<NativeStateTensor, 'dtype' | 'shape'>> }): NeuraleseBlockMeta {
  const embeddings = artifact.tensors.embeddings!;
  const dialect = artifact.metadata.schema === 'neuralese.semantic-state/1'
    ? `semantic-state/${artifact.metadata.adapter_sha256}` : `native-state/${artifact.metadata.architecture}`;
  return { id: artifact.id, dialect,
    length: artifact.tensors.token_ids!.shape[0]!, width: embeddings.shape[1]!,
    dtype: embeddings.dtype === 'BF16' ? 'bf16' : embeddings.dtype === 'F16' ? 'f16' : 'f32',
    native_state: { ...structuredClone(artifact.metadata), embedding_dtype: embeddings.dtype },
    producer: { checkpoint_sha256: artifact.metadata.checkpoint_sha256,
      tokenizer_fingerprint: artifact.metadata.tokenizer_fingerprint,
      embedding_weight_sha256: artifact.metadata.embedding_weight_sha256 } };
}

export function nativeTensorArtifactBlockMeta(artifact: Pick<NativeTensorArtifact, 'id' | 'metadata'>): NeuraleseBlockMeta {
  return { id: artifact.id, dialect: `native-tensor/${artifact.metadata.kind}`, length: artifact.metadata.tensors.length,
    width: 1, dtype: 'f32', native_tensor_artifact: structuredClone(artifact.metadata),
    producer: { ...artifact.metadata.native_identity, objective: artifact.metadata.provenance.objective } };
}

/** Required executable dependencies, distinct from historical provenance references. */
export function nativeTensorArtifactDependencies(artifact: Pick<NativeTensorArtifact, 'metadata'>): string[] {
  const meta = artifact.metadata;
  const dependencies = (meta.kind === 'adapter-coefficients' || meta.kind === 'adapter-gradient' ||
      meta.kind === 'optimizer-state' && meta.provenance.scope === 'natlang.native-adapter/1') &&
      meta.source_artifact_id ? [meta.source_artifact_id] : [];
  const binding = meta.kind === 'adapter-gradient' ? meta.provenance.adapter_binding as NativeAdapterBinding | undefined : undefined;
  if (binding) {
    validateNativeAdapterBinding(binding, meta.provenance.effective_native_identity as NativeModelIdentity);
    dependencies.push(...binding.adapters.map(entry => entry.id));
  }
  const contracts = meta.provenance.external_adapter_state_contracts as Record<string, {
    source_adapter_binding?: NativeAdapterBinding; target_adapter_binding?: NativeAdapterBinding;
    source_native_identity: NativeModelIdentity; target_native_identity: NativeModelIdentity }> | undefined;
  for (const contract of Object.values(contracts ?? {})) for (const side of ['source', 'target'] as const) {
    const actor = contract[`${side}_adapter_binding`];
    if (actor) {
      validateNativeAdapterBinding(actor, contract[`${side}_native_identity`]);
      dependencies.push(...actor.adapters.map(entry => entry.id));
    }
  }
  return [...new Set(dependencies)];
}

/** Archive one immutable named native artifact and its executable dependencies. */
export async function archiveNativeTensorFrom(remote: NeuraleseStore, archive: NeuraleseStore,
    id: string): Promise<NativeTensorArtifact> {
  if (!archive.getNativeTensorArtifact || !archive.putNativeTensorArtifact)
    throw new Error('neuralese-native-archive-unavailable: configure a named tensor archive');
  const found = await archive.getNativeTensorArtifact(id);
  const artifact = found ?? await remote.getNativeTensorArtifact?.(id);
  if (!artifact) throw new Error(`neuralese-unknown-native-artifact: restore native tensor ${id}`);
  for (const dependency of nativeTensorArtifactDependencies(artifact)) await archiveNativeTensorFrom(remote, archive, dependency);
  if (found) return found;
  const { id: _, ...input } = artifact;
  const saved = await archive.putNativeTensorArtifact(input);
  if (saved.id !== id) throw new Error('neuralese-block-integrity: native artifact changed during archival');
  return saved;
}

/** Exact executable dependency closure for retention and portable values. */
export async function nativeArtifactDependencyClosure(store: Pick<NeuraleseStore, 'getNativeState' | 'getNativeTensorArtifact'>,
    roots: Iterable<string>): Promise<Set<string>> {
  const keep = new Set(roots), pending = [...keep];
  for (let index = 0; index < pending.length; index++) {
    const id = pending[index]!;
    const artifact = id.startsWith('nnt1_') ? await store.getNativeTensorArtifact?.(id) : undefined;
    const state = id.startsWith('nzs1_') ? await store.getNativeState?.(id) : undefined;
    const dependencies = artifact ? nativeTensorArtifactDependencies(artifact) :
      state?.metadata.adapter_binding?.adapters.map(entry => entry.id) ?? [];
    for (const dependency of dependencies) if (!keep.has(dependency)) { keep.add(dependency); pending.push(dependency); }
  }
  return keep;
}

/** Client interface to a tensor store: in memory, on disk, or a model server's block endpoints. */
export interface NeuraleseStore {
  put(block: NeuraleseBlockInput): Promise<NeuraleseBlockMeta>;
  get(id: string): Promise<NeuraleseBlock | undefined>;
  meta(id: string): Promise<NeuraleseBlockMeta | undefined>;
  has(id: string): Promise<boolean>;
  /**
   * Metadata of a block this store holds, or was told about with `note`, without I/O (a process-local store), for
   * synchronous checks such as a value's dialect when it is coerced to a type (native/values.ts) or a block's length
   * in a context estimate. Undefined when the store would have to ask.
   */
  peek?(id: string): NeuraleseBlockMeta | undefined;
  /**
   * Remember the metadata of a block held elsewhere (a model server's `GET /v1/neuralese/blocks/{id}/meta`), so that
   * `peek` answers for it. It does not make the block present: `has`, `get` and `meta` still answer for held blocks only.
   */
  note?(meta: NeuraleseBlockMeta): void;
  /** Keep a block through garbage collection (for example while a file or a running call refers to it). */
  pin(id: string): Promise<void>;
  unpin(id: string): Promise<void>;
  /** Drop every unpinned block that `referenced` does not name. Returns the IDs removed. */
  collect(referenced: ReadonlySet<string>): Promise<string[]>;
  /** Exact positional-KV/hybrid-state artifacts share this store's ownership lifecycle when supported. */
  putNativeState?(artifact: Omit<NativeStateArtifact, 'id'>): Promise<NativeStateArtifact>;
  getNativeState?(id: string): Promise<NativeStateArtifact | undefined>;
  hasNativeState?(id: string): Promise<boolean>;
  putNativeTensorArtifact?(artifact: Omit<NativeTensorArtifact, 'id'>): Promise<NativeTensorArtifact>;
  getNativeTensorArtifact?(id: string): Promise<NativeTensorArtifact | undefined>;
  hasNativeTensorArtifact?(id: string): Promise<boolean>;
}

const BYTES: Record<NeuraleseDtype, number> = { f32: 4, f16: 2, bf16: 2 };
const BASE32 = 'abcdefghijklmnopqrstuvwxyz234567';

function base32(bytes: Uint8Array): string {
  let bits = 0, value = 0, out = '';
  for (const byte of bytes) {
    value = (value << 8) | byte; bits += 8;
    while (bits >= 5) { out += BASE32[(value >>> (bits - 5)) & 31]; bits -= 5; }
  }
  if (bits > 0) out += BASE32[(value << (5 - bits)) & 31];
  return out;
}

/** Content ID: `nz1_` + base32 SHA-256 over dialect, shape, dtype and payload bytes. */
/** Blocks that are pure constants of a dialect (the zero-length identity of `combine`), known to every runtime. */
const constants = new Map<string, NeuraleseBlock>();

/** The zero-length block of a dialect and width; registered so that upload paths can supply it. */
export function emptyBlock(dialect: string, width: number): NeuraleseBlock {
  const id = neuraleseContentId({ dialect, length: 0, width, dtype: 'f32', data: new Uint8Array(0) });
  let block = constants.get(id);
  if (!block) constants.set(id, block = { meta: { id, dialect, type: 'Neuralese<unknown>', length: 0, width, dtype: 'f32',
    producer: { kind: 'constant', name: 'empty' } }, data: new Uint8Array(0) });
  return block;
}

/** A registered constant block, when `id` names one. */
export function constantBlock(id: string): NeuraleseBlock | undefined {
  return constants.get(id);
}

export function neuraleseContentId(block: Pick<NeuraleseBlockInput, 'dialect' | 'length' | 'width' | 'dtype' | 'data'>): string {
  const header = new TextEncoder().encode(`natlang.neuralese-block/1\0${block.dialect}\0${block.length}x${block.width}\0${block.dtype}\0`);
  const joined = new Uint8Array(header.length + block.data.length);
  joined.set(header); joined.set(block.data, header.length);
  return `nz1_${base32(digest(joined))}`;
}

const NATIVE_DTYPE_BYTES: Record<NativeStateDtype, number> = { F16: 2, BF16: 2, F32: 4, F64: 8, I64: 8 };
const NATIVE_FLOAT_DTYPE_BYTES: Record<NativeTensorDtype, number> = { F16: 2, BF16: 2, F32: 4, F64: 8 };
const SHA256 = /^[0-9a-f]{64}$/;
function validNativeAdapterScaleHex(value: unknown): boolean {
  if (typeof value !== 'string') return false;
  if (/^-?0x0\.0p\+0$/.test(value)) return true;
  const match = /^-?0x([01])\.([0-9a-f]{13})p([+-]\d+)$/.exec(value);
  if (!match) return false;
  const exponent = Number(match[3]);
  if (match[3] !== `${exponent >= 0 ? '+' : ''}${exponent}` || exponent < -1022 || exponent > 1023) return false;
  return match[1] === '1' || exponent === -1022 && Number.parseInt(match[2]!, 16) !== 0;
}

export function validateNativeAdapterBinding(binding: NativeAdapterBinding, metadata: NativeModelIdentity): void {
    const base = binding?.base_native_identity;
    if (!metadata || !binding || binding.schema !== 'natlang.native-adapter/1' || !base ||
        Object.keys(base).sort().join(',') !== 'architecture,checkpoint_sha256,embedding_weight_sha256,tokenizer_fingerprint' ||
        !SHA256.test(base.checkpoint_sha256) || !nativeModelIdentityEqual(base, {
          checkpoint_sha256: base.checkpoint_sha256, tokenizer_fingerprint: metadata.tokenizer_fingerprint,
          embedding_weight_sha256: metadata.embedding_weight_sha256, architecture: metadata.architecture }) ||
        !Array.isArray(binding.adapters) || !binding.adapters.length ||
        binding.adapters.some(entry => !entry || Object.keys(entry).sort().join(',') !== 'id,scale_hex' ||
          !/^nnt1_[a-z2-7]{52}$/.test(entry.id) || !validNativeAdapterScaleHex(entry.scale_hex)) ||
        Object.keys(binding).sort().join(',') !== 'adapters,base_native_identity,schema')
      throw new RangeError('invalid native adapter state binding');
    const hash = [...digest(new TextEncoder().encode(nativeCanonicalJson(binding)))].map(value => value.toString(16).padStart(2, '0')).join('');
    if (hash !== metadata.checkpoint_sha256) throw new RangeError('native state differs from its exact adapted model identity');
}

/** Exact actor receipt verifier; importing another actor is always caller-declared. */
export function validateNativeAdapterStateConsumption(receipt: NativeAdapterStateConsumption,
    source: NativeStateArtifact, target: NativeStateMetadata): void {
  const identity = (value: NativeModelIdentity): NativeModelIdentity => ({ checkpoint_sha256: value.checkpoint_sha256,
    tokenizer_fingerprint: value.tokenizer_fingerprint, embedding_weight_sha256: value.embedding_weight_sha256, architecture: value.architecture });
  const equal = (a: unknown, b: unknown): boolean => nativeCanonicalJson(a) === nativeCanonicalJson(b);
  const sourceIdentity = identity(source.metadata), targetIdentity = identity(target);
  const sourceBinding = source.metadata.adapter_binding ?? null, targetBinding = target.adapter_binding ?? null;
  if (!receipt || Object.keys(receipt).sort().join(',') !== 'base_native_identity,mode,schema,source_adapter_binding,source_native_identity,source_state_id,target_adapter_binding,target_native_identity,target_position_start,tokens' ||
      receipt.schema !== 'natlang.native-adapter-state-consumption/1' || receipt.mode !== 'same-base-adapter-state' ||
      source.metadata.schema !== 'neuralese.native-state/1' || receipt.source_state_id !== source.id ||
      !equal(receipt.source_native_identity, sourceIdentity) || !equal(receipt.target_native_identity, targetIdentity) ||
      !equal(receipt.source_adapter_binding, sourceBinding) || !equal(receipt.target_adapter_binding, targetBinding) ||
      !Number.isSafeInteger(receipt.target_position_start) || receipt.target_position_start < 0 ||
      receipt.tokens !== source.tensors.token_ids?.shape[0]) throw new RangeError('invalid exact-base adapter-state consumption receipt');
  if (sourceBinding) validateNativeAdapterBinding(sourceBinding, sourceIdentity);
  if (targetBinding) validateNativeAdapterBinding(targetBinding, targetIdentity);
  const sourceBase = sourceBinding?.base_native_identity ?? sourceIdentity;
  const targetBase = targetBinding?.base_native_identity ?? targetIdentity;
  if (!equal(sourceBase, targetBase) || !equal(receipt.base_native_identity, sourceBase))
    throw new RangeError('adapter-state consumption requires the exact same immutable base');
}

function validateNativeState(metadata: NativeStateMetadata, tensors: Record<string, NativeStateTensor>): void {
  if (!metadata || !['neuralese.native-state/1', 'neuralese.semantic-state/1'].includes(metadata.schema) || typeof metadata.text !== 'string' ||
      typeof metadata.architecture !== 'string' || !metadata.architecture ||
      !SHA256.test(metadata.tokenizer_fingerprint) || !SHA256.test(metadata.checkpoint_sha256) ||
      !SHA256.test(metadata.embedding_weight_sha256) || !Array.isArray(metadata.layer_kinds) ||
      (metadata.schema === 'neuralese.native-state/1' && !metadata.layer_kinds.length))
    throw new RangeError('invalid native-state metadata');
  if (metadata.adapter_binding !== undefined) validateNativeAdapterBinding(metadata.adapter_binding, metadata);
  const expected = new Set(['token_ids', 'positions', 'embeddings']);
  const count = (name: string, rank: number): number[] => {
    const tensor = tensors[name];
    if (!tensor || tensor.shape.length !== rank || tensor.shape.some((size, index) => !Number.isSafeInteger(size) || (index === 0 ? size < 0 : size <= 0)) ||
        !Number.isSafeInteger(tensor.shape.reduce((n, size) => n * size, 1)) ||
        tensor.data.length !== tensor.shape.reduce((n, size) => n * size, 1) * NATIVE_DTYPE_BYTES[tensor.dtype])
      throw new RangeError(`invalid native-state tensor ${name}`);
    return tensor.shape;
  };
  const ids = tensors.token_ids, positions = tensors.positions;
  const tokenShape = count('token_ids', 1), positionShape = count('positions', 1), embeddingShape = count('embeddings', 2);
  if (ids!.dtype !== 'I64' || positions!.dtype !== 'I64' || tokenShape[0] !== positionShape[0] ||
      embeddingShape[0] !== tokenShape[0])
    throw new RangeError('native-state token axes or integer tensor types differ');
  const i64 = (tensor: NativeStateTensor) => new DataView(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
  const iv = i64(ids!), pv = i64(positions!);
  for (let i = 0; i < tokenShape[0]!; i++) {
    const token = iv.getBigInt64(i * 8, true), position = pv.getBigInt64(i * 8, true);
    if (token < 0n || position < 0n || (i && position !== pv.getBigInt64((i - 1) * 8, true) + 1n))
      throw new RangeError('native-state token IDs or positions are invalid');
  }
  const finite = (name: string, tensor: NativeStateTensor) => {
    if (tensor.dtype === 'I64') throw new RangeError(`native-state ${name} must be floating point`);
    const view = new DataView(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
    for (let at = 0; at < tensor.data.length; at += NATIVE_DTYPE_BYTES[tensor.dtype]) {
      let value: number;
      if (tensor.dtype === 'F16') {
        const h = view.getUint16(at, true), sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
        value = exponent === 31 ? Number.NaN : sign * (exponent === 0 ? mantissa * 2 ** -24 : (1 + mantissa / 1024) * 2 ** (exponent - 15));
      } else if (tensor.dtype === 'BF16') {
        const bits = view.getUint16(at, true) << 16;
        const word = new ArrayBuffer(4), wordView = new DataView(word);
        wordView.setUint32(0, bits, true);
        value = wordView.getFloat32(0, true);
      } else value = tensor.dtype === 'F32' ? view.getFloat32(at, true) : view.getFloat64(at, true);
      if (!Number.isFinite(value)) throw new RangeError(`native-state ${name} contains non-finite values`);
    }
  };
  finite('embeddings', tensors.embeddings!);
  if (metadata.schema === 'neuralese.semantic-state/1') {
    if (metadata.layer_kinds.length || typeof metadata.source_model !== 'string' || !metadata.source_model ||
        !SHA256.test(metadata.adapter_sha256) || metadata.posterior !== 'mean' ||
        !['exact-tokenizer-offsets', 'actual-token-decoder-alignment', 'actual-token-decoder-text-alignment'].includes(metadata.offset_policy) ||
        !Array.isArray(metadata.offsets) || metadata.offsets.length !== tokenShape[0] ||
        metadata.offsets.some(pair => !Array.isArray(pair) || pair.length !== 2 ||
          pair.some(value => !Number.isSafeInteger(value)) || pair[0]! < 0 || pair[1]! < pair[0]! || pair[1]! > [...metadata.text].length))
      throw new RangeError('invalid semantic message metadata or source token offsets');
    const codeShape = count('code', 2);
    if (codeShape[0] !== tokenShape[0]) throw new RangeError('semantic message code token count differs');
    expected.add('code'); finite('code', tensors.code!);
  }
  for (let index = 0; index < metadata.layer_kinds.length; index++) {
    const kind = metadata.layer_kinds[index], prefix = `layers.${index}.`;
    if (kind === 'attention') {
      const k = count(`${prefix}raw_k`, 3), v = count(`${prefix}v`, 3), kt = tensors[`${prefix}raw_k`]!, vt = tensors[`${prefix}v`]!;
      if (k[0] !== tokenShape[0] || v.some((size, axis) => size !== k[axis]) || kt.dtype !== vt.dtype)
        throw new RangeError(`invalid native-state attention layer ${index}`);
      expected.add(`${prefix}raw_k`); expected.add(`${prefix}v`); finite(`${prefix}raw_k`, kt); finite(`${prefix}v`, vt);
    } else if (kind === 'finite_convolution') {
      const name = `${prefix}gated_inputs`, shape = count(name, 2), tensor = tensors[name]!;
      if (shape[0] !== tokenShape[0]) throw new RangeError(`invalid native-state convolution layer ${index}`);
      expected.add(name); finite(name, tensor);
    } else throw new RangeError(`unsupported native-state layer kind ${String(kind)}`);
  }
  if (Object.keys(tensors).length !== expected.size || Object.keys(tensors).some(name => !expected.has(name)))
    throw new RangeError('native-state tensor inventory differs from its layer layout');
}
function u32(value: number): Uint8Array { const b = new Uint8Array(4); new DataView(b.buffer).setUint32(0, value, true); return b; }
function u64(value: number): Uint8Array { const b = new Uint8Array(8); new DataView(b.buffer).setBigUint64(0, BigInt(value), true); return b; }
function joinBytes(parts: readonly Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, part) => n + part.length, 0)); let at = 0;
  for (const part of parts) { out.set(part, at); at += part.length; } return out;
}
export function nativeCanonicalJson(value: unknown): string {
  const sort = (item: unknown): unknown => Array.isArray(item) ? item.map(sort) :
    item && typeof item === 'object' ? Object.fromEntries(Object.keys(item).sort().map(key =>
      [key, sort((item as Record<string, unknown>)[key])])) : item;
  return JSON.stringify(sort(value));
}

function validateNativeTensorArtifact(metadata: NativeTensorArtifactMetadata,
    tensors: Record<string, NativeTensorValue>): void {
  if (!metadata || metadata.schema !== 'natlang.neuralese-native-tensor-artifact/1' ||
      !['parameter-set', 'parameter-gradient', 'parameter-update', 'state-gradient', 'optimizer-state', 'adapter-basis', 'adapter-coefficients', 'adapter-gradient'].includes(metadata.kind) ||
      !metadata.native_identity || !SHA256.test(metadata.native_identity.checkpoint_sha256) ||
      !SHA256.test(metadata.native_identity.tokenizer_fingerprint) ||
      !SHA256.test(metadata.native_identity.embedding_weight_sha256) ||
      typeof metadata.native_identity.architecture !== 'string' || !metadata.native_identity.architecture ||
      (metadata.source_artifact_id !== null && !/^(?:nzs1_|nnt1_)[a-z2-7]{52}$/.test(metadata.source_artifact_id)) ||
      !metadata.provenance || typeof metadata.provenance !== 'object' || Array.isArray(metadata.provenance) ||
      !tensors || typeof tensors !== 'object' || !Object.keys(tensors).length)
    throw new RangeError('invalid native tensor artifact metadata');
  if (metadata.kind === 'adapter-gradient' && metadata.provenance.adapter_binding !== undefined) {
    const binding = metadata.provenance.adapter_binding as NativeAdapterBinding;
    validateNativeAdapterBinding(binding, metadata.provenance.effective_native_identity as NativeModelIdentity);
    if (!nativeModelIdentityEqual(binding.base_native_identity, metadata.native_identity))
      throw new RangeError('native adapter gradient changed its exact base identity');
  }
  const inventory = Object.keys(tensors).sort().map(name => {
    const tensor = tensors[name]!;
    if (!name || !tensor || !(tensor.dtype in NATIVE_FLOAT_DTYPE_BYTES) || !Array.isArray(tensor.shape) ||
        !tensor.shape.length || tensor.shape.some(size => !Number.isSafeInteger(size) || size <= 0) ||
        !Number.isSafeInteger(tensor.shape.reduce((n, size) => n * size, 1)) ||
        tensor.data.length !== tensor.shape.reduce((n, size) => n * size, 1) * NATIVE_FLOAT_DTYPE_BYTES[tensor.dtype])
      throw new RangeError(`invalid native artifact tensor ${name}`);
    const view = new DataView(tensor.data.buffer, tensor.data.byteOffset, tensor.data.byteLength);
    for (let at = 0; at < tensor.data.length; at += NATIVE_FLOAT_DTYPE_BYTES[tensor.dtype]) {
      let value: number;
      if (tensor.dtype === 'F16') {
        const h = view.getUint16(at, true), sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 31, mantissa = h & 1023;
        value = exponent === 31 ? Number.NaN : sign * (exponent === 0 ? mantissa * 2 ** -24 : (1 + mantissa / 1024) * 2 ** (exponent - 15));
      } else if (tensor.dtype === 'BF16') {
        const word = new ArrayBuffer(4), wordView = new DataView(word);
        wordView.setUint32(0, view.getUint16(at, true) << 16, true); value = wordView.getFloat32(0, true);
      } else value = tensor.dtype === 'F32' ? view.getFloat32(at, true) : view.getFloat64(at, true);
      if (!Number.isFinite(value)) throw new RangeError(`native artifact tensor ${name} contains non-finite values`);
    }
    return { name, dtype: tensor.dtype, shape: [...tensor.shape] };
  });
  if (nativeCanonicalJson(metadata.tensors) !== nativeCanonicalJson(inventory))
    throw new RangeError('native tensor metadata differs from the exact named tensor inventory');
}

/** ID algorithm shared byte-for-byte with model.native_artifact.native_tensor_artifact_id. */
export function nativeTensorArtifactContentId(metadata: NativeTensorArtifactMetadata,
    tensors: Record<string, NativeTensorValue>, canonicalMetadata?: string): string {
  validateNativeTensorArtifact(metadata, tensors);
  const exact = canonicalMetadata ?? nativeCanonicalJson(metadata);
  if (nativeCanonicalJson(JSON.parse(exact)) !== nativeCanonicalJson(metadata))
    throw new RangeError('native artifact preserved metadata bytes differ from its actual fields');
  const metaBytes = new TextEncoder().encode(exact);
  const parts: Uint8Array[] = [new TextEncoder().encode('natlang.neuralese-native-tensor-artifact/1\0'),
    u64(metaBytes.length), metaBytes];
  for (const name of Object.keys(tensors).sort()) {
    const tensor = tensors[name]!, nameBytes = new TextEncoder().encode(name);
    const descriptor = new TextEncoder().encode(nativeCanonicalJson({ dtype: tensor.dtype, shape: tensor.shape }));
    parts.push(u32(nameBytes.length), nameBytes, u32(descriptor.length), descriptor,
      u64(tensor.data.length), tensor.data);
  }
  return `nnt1_${base32(digest(joinBytes(parts)))}`;
}

/** Versioned native gradient/parameter safetensors body matching Python exactly. */
export function encodeNativeTensorArtifact(artifact: Omit<NativeTensorArtifact, 'id'>): Uint8Array {
  const id = nativeTensorArtifactContentId(artifact.metadata, artifact.tensors, artifact.canonicalMetadata);
  const entries: Record<string, unknown> = {}, payloads: Uint8Array[] = [];
  let offset = 0;
  for (const name of Object.keys(artifact.tensors).sort()) {
    const tensor = artifact.tensors[name]!;
    entries[name] = { dtype: tensor.dtype, shape: tensor.shape, data_offsets: [offset, offset + tensor.data.length] };
    payloads.push(tensor.data); offset += tensor.data.length;
  }
  const exact = artifact.canonicalMetadata ?? nativeCanonicalJson(artifact.metadata);
  entries.__metadata__ = { 'natlang.native_tensor_artifact': `{"id":${JSON.stringify(id)},${exact.slice(1)}` };
  const raw = new TextEncoder().encode(nativeCanonicalJson(entries));
  const header = new Uint8Array(Math.ceil(raw.length / 8) * 8).fill(0x20); header.set(raw);
  return joinBytes([u64(header.length), header, ...payloads]);
}

export function decodeNativeTensorArtifact(bytes: Uint8Array): NativeTensorArtifact {
  if (bytes.length < 8) throw new RangeError('not a native tensor artifact safetensors file');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength), headerSize = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(headerSize) || 8 + headerSize > bytes.length) throw new RangeError('native tensor artifact header exceeds body');
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerSize))) as Record<string, unknown>;
  const metadataText = (header.__metadata__ as Record<string, unknown> | undefined)?.['natlang.native_tensor_artifact'];
  if (typeof metadataText !== 'string') throw new RangeError('missing canonical native tensor artifact metadata');
  const stored = JSON.parse(metadataText) as NativeTensorArtifactMetadata & { id?: string };
  const { id, ...metadata } = stored;
  const payload = bytes.subarray(8 + headerSize), tensors: Record<string, NativeTensorValue> = {}, spans: [number, number][] = [];
  for (const [name, raw] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const entry = raw as { dtype?: NativeTensorDtype; shape?: number[]; data_offsets?: [number, number] };
    if (!entry || !entry.dtype || !(entry.dtype in NATIVE_FLOAT_DTYPE_BYTES) || !Array.isArray(entry.shape) ||
        !Array.isArray(entry.data_offsets) || entry.data_offsets.length !== 2)
      throw new RangeError(`invalid native artifact descriptor ${name}`);
    const [start, end] = entry.data_offsets, count = entry.shape.reduce((n, size) => n * size, 1);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end - start !== count * NATIVE_FLOAT_DTYPE_BYTES[entry.dtype] || end > payload.length)
      throw new RangeError(`invalid native artifact byte range ${name}`);
    spans.push([start, end]); tensors[name] = { dtype: entry.dtype, shape: [...entry.shape], data: payload.slice(start, end) };
  }
  spans.sort((a, b) => a[0] - b[0]);
  if (!spans.length || spans[0]![0] !== 0 || spans.at(-1)![1] !== payload.length ||
      spans.some((span, index) => index > 0 && spans[index - 1]![1] !== span[0]))
    throw new RangeError('native tensor artifact payload has gaps or overlaps');
  const prefix = `{"id":${JSON.stringify(id)},`;
  const canonicalMetadata = metadataText.startsWith(prefix) ? '{' + metadataText.slice(prefix.length) : undefined;
  const actual = nativeTensorArtifactContentId(metadata, tensors, canonicalMetadata);
  if (actual !== id) throw new RangeError(`native tensor artifact ID mismatch: ${actual}`);
  return { id: actual, metadata, tensors, canonicalMetadata };
}

/** Versioned ID shared byte-for-byte with `serve.store.native_state_id` (Python). */
export function nativeStateContentId(metadata: NativeStateMetadata, tensors: Record<string, NativeStateTensor>): string {
  validateNativeState(metadata, tensors);
  const metaBytes = new TextEncoder().encode(nativeCanonicalJson(metadata));
  const parts: Uint8Array[] = [new TextEncoder().encode('natlang.native-state/1\0'), u64(metaBytes.length), metaBytes];
  for (const name of Object.keys(tensors).sort()) {
    const tensor = tensors[name]!;
    const elements = tensor.shape.reduce((product, size) => product * size, 1);
    if (!tensor.shape.length || tensor.shape.some((size, index) => !Number.isSafeInteger(size) || (index === 0 ? size < 0 : size <= 0)) ||
        tensor.data.length !== elements * NATIVE_DTYPE_BYTES[tensor.dtype]) throw new RangeError(`invalid native tensor ${name}`);
    const nameBytes = new TextEncoder().encode(name);
    const descriptor = new TextEncoder().encode(nativeCanonicalJson({ dtype: tensor.dtype, shape: tensor.shape }));
    parts.push(u32(nameBytes.length), nameBytes, u32(descriptor.length), descriptor, u64(tensor.data.length), tensor.data);
  }
  return `nzs1_${base32(digest(joinBytes(parts)))}`;
}

/** Multi-tensor safetensors body matching Python `encode_native_state`. */
export function encodeNativeState(artifact: Omit<NativeStateArtifact, 'id'>): Uint8Array {
  const id = nativeStateContentId(artifact.metadata, artifact.tensors);
  const entries: Record<string, unknown> = {};
  const payloads: Uint8Array[] = [];
  let offset = 0;
  for (const name of Object.keys(artifact.tensors).sort()) {
    const tensor = artifact.tensors[name]!;
    entries[name] = { dtype: tensor.dtype, shape: tensor.shape, data_offsets: [offset, offset + tensor.data.length] };
    payloads.push(tensor.data); offset += tensor.data.length;
  }
  entries.__metadata__ = { 'natlang.native_state': nativeCanonicalJson({ id, ...artifact.metadata }) };
  let header = new TextEncoder().encode(nativeCanonicalJson(entries));
  const aligned = new Uint8Array(Math.ceil(header.length / 8) * 8).fill(0x20); aligned.set(header); header = aligned;
  const body = joinBytes([u64(header.length), header, ...payloads]);
  return body;
}

/** Decode and verify an immutable native state safetensors artifact. */
export function decodeNativeState(bytes: Uint8Array): NativeStateArtifact {
  if (bytes.length < 8) throw new RangeError('not a native state safetensors file');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const headerSize = Number(view.getBigUint64(0, true));
  if (!Number.isSafeInteger(headerSize) || 8 + headerSize > bytes.length) throw new RangeError('native state header exceeds the body');
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + headerSize))) as Record<string, unknown>;
  const metaField = (header.__metadata__ as Record<string, unknown> | undefined)?.natlang_native_state;
  // Safetensors metadata keys cannot contain dots in some consumers, so the
  // wire uses the dotted key as the canonical spelling and this fallback only
  // makes malformed third-party rewrites fail with a clear error.
  const metadataText = (header.__metadata__ as Record<string, unknown> | undefined)?.['natlang.native_state'];
  if (typeof metadataText !== 'string' || metaField !== undefined) throw new RangeError('missing canonical native-state metadata');
  const stored = JSON.parse(metadataText) as NativeStateMetadata & { id?: string };
  const { id, ...metadata } = stored;
  if (!['neuralese.native-state/1', 'neuralese.semantic-state/1'].includes(metadata.schema) || typeof metadata.text !== 'string' ||
      typeof metadata.architecture !== 'string' || !Array.isArray(metadata.layer_kinds))
    throw new RangeError('unsupported native state metadata');
  const payload = bytes.subarray(8 + headerSize);
  const tensors: Record<string, NativeStateTensor> = {};
  const spans: Array<[number, number]> = [];
  for (const [name, raw] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const entry = raw as { dtype?: NativeStateDtype; shape?: number[]; data_offsets?: [number, number] };
    if (!entry || !entry.dtype || !(entry.dtype in NATIVE_DTYPE_BYTES) || !Array.isArray(entry.shape) ||
        !entry.shape.length || entry.shape.some((size, index) => !Number.isSafeInteger(size) || (index === 0 ? size < 0 : size <= 0)) ||
        !Array.isArray(entry.data_offsets) || entry.data_offsets.length !== 2)
      throw new RangeError(`invalid native tensor descriptor ${name}`);
    const [start, end] = entry.data_offsets;
    const count = entry.shape.reduce((n, size) => n * size, 1);
    if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start < 0 || end - start !== count * NATIVE_DTYPE_BYTES[entry.dtype] || end > payload.length)
      throw new RangeError(`invalid native tensor byte range ${name}`);
    spans.push([start, end]);
    tensors[name] = { dtype: entry.dtype, shape: [...entry.shape], data: payload.slice(start, end) };
  }
  spans.sort((a, b) => a[0] - b[0]);
  if (!spans.length || spans[0]![0] !== 0 || spans.at(-1)![1] !== payload.length ||
      spans.some((span, index) => index > 0 && spans[index - 1]![1] !== span[0]))
    throw new RangeError('native state tensor payload has gaps or overlaps');
  const actual = nativeStateContentId(metadata, tensors);
  if (actual !== id) throw new RangeError(`native state content ID mismatch: ${actual}`);
  return { id: actual, metadata, tensors };
}

export function checkBlockShape(block: NeuraleseBlockInput): void {
  if (!Number.isSafeInteger(block.length) || block.length < 0 || !Number.isSafeInteger(block.width) || block.width <= 0)
    throw new RangeError('a Neuralese block needs a nonnegative integer length and a positive integer width');
  if (block.data.length !== block.length * block.width * BYTES[block.dtype])
    throw new RangeError(`a ${block.length}x${block.width} ${block.dtype} block has ${block.length * block.width * BYTES[block.dtype]} bytes, ` +
      `got ${block.data.length}`);
}

/** Process-local store; the default for runtimes without a model-side store, and for tests. */
export class MemoryNeuraleseStore implements NeuraleseStore {
  private readonly blocks = new Map<string, NeuraleseBlock>();
  private readonly nativeStates = new Map<string, NativeStateArtifact>();
  private readonly nativeTensorArtifacts = new Map<string, NativeTensorArtifact>();
  /** Metadata of blocks held elsewhere (`note`). */
  private readonly noted = new Map<string, NeuraleseBlockMeta>();
  private readonly pins = new Map<string, number>();
  async put(block: NeuraleseBlockInput): Promise<NeuraleseBlockMeta> {
    checkBlockShape(block);
    const id = neuraleseContentId(block);
    const found = this.blocks.get(id);
    if (found) {
      // Same bytes: keep the first producer; learn a type the first writer did not know.
      if (!found.meta.type && block.type) found.meta = { ...found.meta, type: block.type };
      return { ...found.meta };
    }
    const { data, ...rest } = block;
    const meta: NeuraleseBlockMeta = { id, ...rest };
    this.blocks.set(id, { meta, data: data.slice() });
    return { ...meta };
  }
  async get(id: string): Promise<NeuraleseBlock | undefined> {
    const found = this.blocks.get(id);
    return found && { meta: { ...found.meta }, data: found.data.slice() };
  }
  async meta(id: string): Promise<NeuraleseBlockMeta | undefined> {
    const found = this.peek(id);
    return found && { ...found };
  }
  async has(id: string): Promise<boolean> { return this.blocks.has(id) || this.nativeStates.has(id) || this.nativeTensorArtifacts.has(id); }
  async putNativeState(artifact: Omit<NativeStateArtifact, 'id'>): Promise<NativeStateArtifact> {
    const id = nativeStateContentId(artifact.metadata, artifact.tensors);
    const stored = { id, metadata: structuredClone(artifact.metadata), tensors: Object.fromEntries(
      Object.entries(artifact.tensors).map(([name, tensor]) => [name, { ...tensor, shape: [...tensor.shape], data: tensor.data.slice() }])) };
    const found = this.nativeStates.get(id);
    if (found) return structuredClone(found);
    this.nativeStates.set(id, stored);
    this.noted.delete(id);
    return structuredClone(stored);
  }
  async getNativeState(id: string): Promise<NativeStateArtifact | undefined> {
    const found = this.nativeStates.get(id);
    return found && structuredClone(found);
  }
  async hasNativeState(id: string): Promise<boolean> { return this.nativeStates.has(id); }
  async putNativeTensorArtifact(artifact: Omit<NativeTensorArtifact, 'id'>): Promise<NativeTensorArtifact> {
    const wire = encodeNativeTensorArtifact(artifact), stored = decodeNativeTensorArtifact(wire);
    const found = this.nativeTensorArtifacts.get(stored.id);
    if (found) return structuredClone(found);
    this.nativeTensorArtifacts.set(stored.id, stored); this.noted.delete(stored.id);
    return structuredClone(stored);
  }
  async getNativeTensorArtifact(id: string): Promise<NativeTensorArtifact | undefined> {
    const found = this.nativeTensorArtifacts.get(id); return found && structuredClone(found);
  }
  async hasNativeTensorArtifact(id: string): Promise<boolean> { return this.nativeTensorArtifacts.has(id); }
  peek(id: string): NeuraleseBlockMeta | undefined {
    const native = this.nativeStates.get(id);
    const tensor = this.nativeTensorArtifacts.get(id);
    const found = this.blocks.get(id)?.meta ?? (native && nativeStateBlockMeta(native)) ??
      (tensor && nativeTensorArtifactBlockMeta(tensor)) ?? this.noted.get(id);
    return found && { ...found };
  }
  note(meta: NeuraleseBlockMeta): void {
    if (!this.blocks.has(meta.id)) this.noted.set(meta.id, { ...meta });
  }
  async pin(id: string): Promise<void> {
    if (!this.blocks.has(id) && !this.nativeStates.has(id) && !this.nativeTensorArtifacts.has(id)) throw new Error(`neuralese-unknown-block: ${id}`);
    this.pins.set(id, (this.pins.get(id) ?? 0) + 1);
  }
  async unpin(id: string): Promise<void> {
    const count = (this.pins.get(id) ?? 0) - 1;
    if (count > 0) this.pins.set(id, count); else this.pins.delete(id);
  }
  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    referenced = await nativeArtifactDependencyClosure(this, [...referenced, ...this.pins.keys()]);
    const removed: string[] = [];
    for (const id of [...this.blocks.keys(), ...this.nativeStates.keys(), ...this.nativeTensorArtifacts.keys()])
      if (!referenced.has(id) && !this.pins.has(id)) {
        this.blocks.delete(id); this.nativeStates.delete(id); this.nativeTensorArtifacts.delete(id); removed.push(id);
      }
    for (const id of [...this.noted.keys()]) if (!referenced.has(id)) this.noted.delete(id);
    return removed;
  }
  get size(): number { return this.blocks.size + this.nativeStates.size + this.nativeTensorArtifacts.size; }
}

/** Write port: turns what a model wrote inside a literal into a stored block. */
export interface NeuralesePort {
  readonly dialect: string;
  write(text: string, options?: { type?: string; producer?: Record<string, unknown>; truncated?: boolean }): Promise<NeuraleseBlockMeta>;
}

/** Text to its token embeddings, one vector per token, each `width` long. */
export type TokenEmbedder = (text: string) => Float32Array[] | Promise<Float32Array[]>;

/**
 * The untrained stand-in port (S4 §4.3): until a trained writer exists, a written block is the token embeddings of
 * the text the model wrote inside the literal. It exercises protocol, store and runtime paths, not writing quality.
 */
export class StandInNeuralesePort implements NeuralesePort {
  constructor(private readonly store: NeuraleseStore, private readonly embed: TokenEmbedder,
    readonly width: number, readonly dialect = 'nd:standin@0') {}
  async write(text: string, options: { type?: string; producer?: Record<string, unknown>; truncated?: boolean } = {}):
      Promise<NeuraleseBlockMeta> {
    const vectors = await this.embed(text);
    const data = new Float32Array(vectors.length * this.width);
    vectors.forEach((vector, index) => {
      if (vector.length !== this.width) throw new RangeError(`embedding width ${vector.length}, expected ${this.width}`);
      data.set(vector, index * this.width);
    });
    return this.store.put({ dialect: this.dialect, type: options.type, length: vectors.length, width: this.width, dtype: 'f32',
      data: new Uint8Array(data.buffer), producer: { kind: 'stand-in', ...(options.producer ?? {}) },
      ...(options.truncated ? { truncated: true } : {}) });
  }
}

/**
 * A deterministic stand-in embedder for runtimes without a model's embedding table, and for tests: one vector per
 * whitespace-separated token, derived from the token's hash. It carries no meaning.
 * Text transports can retain whitespace fragments so distinct literal bodies have distinct payloads.
 */
export function hashingEmbedder(width: number, options: { preserveWhitespace?: boolean } = {}): TokenEmbedder {
  return text => (options.preserveWhitespace ? text.match(/\s+|\S+/g) ?? [] : text.split(/\s+/).filter(Boolean)).map(token => {
    const vector = new Float32Array(width);
    let seed = digest(token);
    for (let index = 0; index < width; index++) {
      if (index && index % seed.length === 0) seed = digest(seed);
      vector[index] = (seed[index % seed.length]! - 127.5) / 127.5;
    }
    return vector;
  });
}
