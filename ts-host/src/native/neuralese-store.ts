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
  /** Writing hit the runtime's hard maximum instead of stopping. */
  truncated?: boolean;
};

export type NeuraleseBlock = { meta: NeuraleseBlockMeta; data: Uint8Array };

export type NeuraleseBlockInput = Omit<NeuraleseBlockMeta, 'id'> & { data: Uint8Array };

export type NativeStateDtype = 'F16' | 'BF16' | 'F32' | 'F64' | 'I64';
export type NativeStateTensor = { dtype: NativeStateDtype; shape: number[]; data: Uint8Array };
export type NativeStateMetadata = { schema: 'neuralese.native-state/1'; text: string; architecture: string;
  tokenizer_fingerprint: string; checkpoint_sha256: string; embedding_weight_sha256: string; layer_kinds: string[] };
export type NativeStateArtifact = { id: string; metadata: NativeStateMetadata; tensors: Record<string, NativeStateTensor> };

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
const SHA256 = /^[0-9a-f]{64}$/;
function validateNativeState(metadata: NativeStateMetadata, tensors: Record<string, NativeStateTensor>): void {
  if (!metadata || metadata.schema !== 'neuralese.native-state/1' || typeof metadata.text !== 'string' ||
      typeof metadata.architecture !== 'string' || !metadata.architecture ||
      !SHA256.test(metadata.tokenizer_fingerprint) || !SHA256.test(metadata.checkpoint_sha256) ||
      !SHA256.test(metadata.embedding_weight_sha256) || !Array.isArray(metadata.layer_kinds) || !metadata.layer_kinds.length)
    throw new RangeError('invalid native-state metadata');
  const expected = new Set(['token_ids', 'positions', 'embeddings']);
  const count = (name: string, rank: number): number[] => {
    const tensor = tensors[name];
    if (!tensor || tensor.shape.length !== rank || tensor.shape.some(size => !Number.isSafeInteger(size) || size <= 0) ||
        !Number.isSafeInteger(tensor.shape.reduce((n, size) => n * size, 1)) ||
        tensor.data.length !== tensor.shape.reduce((n, size) => n * size, 1) * NATIVE_DTYPE_BYTES[tensor.dtype])
      throw new RangeError(`invalid native-state tensor ${name}`);
    return tensor.shape;
  };
  const ids = tensors.token_ids, positions = tensors.positions;
  const tokenShape = count('token_ids', 1), positionShape = count('positions', 1), embeddingShape = count('embeddings', 2);
  if (ids!.dtype !== 'I64' || positions!.dtype !== 'I64' || tokenShape[0] !== positionShape[0] ||
      embeddingShape[0] !== tokenShape[0] || tokenShape[0]! <= 0)
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
function nativeCanonicalJson(value: unknown): string {
  const sort = (item: unknown): unknown => Array.isArray(item) ? item.map(sort) :
    item && typeof item === 'object' ? Object.fromEntries(Object.keys(item).sort().map(key =>
      [key, sort((item as Record<string, unknown>)[key])])) : item;
  return JSON.stringify(sort(value));
}

/** Versioned ID shared byte-for-byte with `serve.store.native_state_id` (Python). */
export function nativeStateContentId(metadata: NativeStateMetadata, tensors: Record<string, NativeStateTensor>): string {
  validateNativeState(metadata, tensors);
  const metaBytes = new TextEncoder().encode(nativeCanonicalJson(metadata));
  const parts: Uint8Array[] = [new TextEncoder().encode('natlang.native-state/1\0'), u64(metaBytes.length), metaBytes];
  for (const name of Object.keys(tensors).sort()) {
    const tensor = tensors[name]!;
    const elements = tensor.shape.reduce((product, size) => product * size, 1);
    if (!tensor.shape.length || tensor.shape.some(size => !Number.isSafeInteger(size) || size <= 0) ||
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
  if (metadata.schema !== 'neuralese.native-state/1' || typeof metadata.text !== 'string' ||
      typeof metadata.architecture !== 'string' || !Array.isArray(metadata.layer_kinds))
    throw new RangeError('unsupported native state metadata');
  const payload = bytes.subarray(8 + headerSize);
  const tensors: Record<string, NativeStateTensor> = {};
  const spans: Array<[number, number]> = [];
  for (const [name, raw] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const entry = raw as { dtype?: NativeStateDtype; shape?: number[]; data_offsets?: [number, number] };
    if (!entry || !entry.dtype || !(entry.dtype in NATIVE_DTYPE_BYTES) || !Array.isArray(entry.shape) ||
        !entry.shape.length || entry.shape.some(size => !Number.isSafeInteger(size) || size <= 0) ||
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
    const found = this.blocks.get(id);
    return found && { ...found.meta };
  }
  async has(id: string): Promise<boolean> { return this.blocks.has(id) || this.nativeStates.has(id); }
  async putNativeState(artifact: Omit<NativeStateArtifact, 'id'>): Promise<NativeStateArtifact> {
    const id = nativeStateContentId(artifact.metadata, artifact.tensors);
    const stored = { id, metadata: structuredClone(artifact.metadata), tensors: Object.fromEntries(
      Object.entries(artifact.tensors).map(([name, tensor]) => [name, { ...tensor, shape: [...tensor.shape], data: tensor.data.slice() }])) };
    const found = this.nativeStates.get(id);
    if (found) return structuredClone(found);
    this.nativeStates.set(id, stored);
    return structuredClone(stored);
  }
  async getNativeState(id: string): Promise<NativeStateArtifact | undefined> {
    const found = this.nativeStates.get(id);
    return found && structuredClone(found);
  }
  async hasNativeState(id: string): Promise<boolean> { return this.nativeStates.has(id); }
  peek(id: string): NeuraleseBlockMeta | undefined {
    const found = this.blocks.get(id)?.meta ?? this.noted.get(id);
    return found && { ...found };
  }
  note(meta: NeuraleseBlockMeta): void {
    if (!this.blocks.has(meta.id)) this.noted.set(meta.id, { ...meta });
  }
  async pin(id: string): Promise<void> {
    if (!this.blocks.has(id) && !this.nativeStates.has(id)) throw new Error(`neuralese-unknown-block: ${id}`);
    this.pins.set(id, (this.pins.get(id) ?? 0) + 1);
  }
  async unpin(id: string): Promise<void> {
    const count = (this.pins.get(id) ?? 0) - 1;
    if (count > 0) this.pins.set(id, count); else this.pins.delete(id);
  }
  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    const removed: string[] = [];
    for (const id of [...this.blocks.keys(), ...this.nativeStates.keys()])
      if (!referenced.has(id) && !this.pins.has(id)) {
        this.blocks.delete(id); this.nativeStates.delete(id); removed.push(id);
      }
    for (const id of [...this.noted.keys()]) if (!referenced.has(id)) this.noted.delete(id);
    return removed;
  }
  get size(): number { return this.blocks.size + this.nativeStates.size; }
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
