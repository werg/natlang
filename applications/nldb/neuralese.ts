/**
 * Native state and named tensor archive for NLDB, plus semantic indexing over native-state token embeddings.
 * Database rows retain content-addressed `nzs1` IDs; tensor artifacts use the same folder archive and are available
 * through the complete NeuraleseStore contract, but are not treated as semantic vectors.
 */
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';
import { FileNeuraleseStore, type NeuraleseBlockMeta, type NeuraleseStore,
  type NativeStateArtifact, type NativeTensorArtifact } from '@natlang/node';

type NativeFolder = FileNeuraleseStore;

/** The NLDB-owned `blocks/` folder is a persistent native artifact archive. */
export class FolderBlockStore implements NeuraleseStore {
  private readonly archive: Promise<NativeFolder>;
  private readonly metadata = new Map<string, NeuraleseBlockMeta>();

  constructor(readonly root: string) {
    this.archive = (async () => {
      await mkdir(root, { recursive: true });
      const store = await FileNeuraleseStore.open(join(root, 'blocks'));
      for (const id of store.ids()) {
        const meta = await store.meta(id);
        if (meta) this.metadata.set(id, meta);
      }
      return store;
    })();
  }

  static async open(root: string): Promise<FolderBlockStore> {
    const store = new FolderBlockStore(root);
    await store.archive;
    return store;
  }

  private async opened(): Promise<NativeFolder> { return this.archive; }
  async meta(id: string): Promise<NeuraleseBlockMeta | undefined> {
    const meta = await (await this.opened()).meta(id);
    if (meta) this.metadata.set(id, meta); else this.metadata.delete(id);
    return meta;
  }
  async has(id: string): Promise<boolean> {
    const present = await (await this.opened()).has(id);
    if (!present) this.metadata.delete(id);
    return present;
  }
  peek(id: string): NeuraleseBlockMeta | undefined {
    const meta = this.metadata.get(id);
    return meta && structuredClone(meta);
  }
  note(meta: NeuraleseBlockMeta): void {
    if (!this.metadata.has(meta.id)) this.metadata.set(meta.id, structuredClone(meta));
    void this.archive.then(store => store.note(meta)).catch(() => undefined);
  }
  async pin(id: string): Promise<void> { await (await this.opened()).pin(id); }
  async unpin(id: string): Promise<void> { await (await this.opened()).unpin(id); }
  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    const removed = await (await this.opened()).collect(referenced);
    for (const id of removed) this.metadata.delete(id);
    return removed;
  }
  async putNativeState(input: Omit<NativeStateArtifact, 'id'>): Promise<NativeStateArtifact> {
    const artifact = await (await this.opened()).putNativeState(input);
    this.metadata.set(artifact.id, (await this.meta(artifact.id))!);
    return artifact;
  }
  async getNativeState(id: string): Promise<NativeStateArtifact | undefined> {
    const artifact = await (await this.opened()).getNativeState(id);
    if (!artifact) this.metadata.delete(id);
    return artifact;
  }
  async hasNativeState(id: string): Promise<boolean> { return (await this.opened()).hasNativeState(id); }
  async putNativeTensorArtifact(input: Omit<NativeTensorArtifact, 'id'>): Promise<NativeTensorArtifact> {
    const artifact = await (await this.opened()).putNativeTensorArtifact(input);
    this.metadata.set(artifact.id, (await this.meta(artifact.id))!);
    return artifact;
  }
  async getNativeTensorArtifact(id: string): Promise<NativeTensorArtifact | undefined> {
    const artifact = await (await this.opened()).getNativeTensorArtifact(id);
    if (!artifact) this.metadata.delete(id);
    return artifact;
  }
  async hasNativeTensorArtifact(id: string): Promise<boolean> {
    return (await this.opened()).hasNativeTensorArtifact(id);
  }
}

/** Exact identity of the embedding coordinate space used by the semantic index. */
export function nativeStateDialect(state: NativeStateArtifact): string {
  const { architecture, checkpoint_sha256, tokenizer_fingerprint, embedding_weight_sha256 } = state.metadata;
  return `native-state/${architecture}/${checkpoint_sha256}/${tokenizer_fingerprint}/${embedding_weight_sha256}`;
}

/** A native state's token embeddings as f32 numbers (bf16 and f16 widened). */
export function rows(state: NativeStateArtifact): Float32Array {
  const embeddings = state.tensors.embeddings;
  if (!embeddings || embeddings.shape.length !== 2 || !['F32', 'F16', 'BF16'].includes(embeddings.dtype))
    throw new TypeError('native state has no supported rank-two embedding tensor');
  const [length, width] = embeddings.shape;
  if (!Number.isSafeInteger(length) || !Number.isSafeInteger(width) || length! < 1 || width! < 1)
    throw new TypeError('native state embedding tensor has invalid dimensions');
  const count = length! * width!, view = new DataView(embeddings.data.buffer, embeddings.data.byteOffset, embeddings.data.byteLength);
  if (embeddings.data.byteLength !== count * (embeddings.dtype === 'F32' ? 4 : 2))
    throw new TypeError('native state embedding tensor byte length differs from its shape');
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    if (embeddings.dtype === 'F32') out[i] = view.getFloat32(i * 4, true);
    else if (embeddings.dtype === 'BF16') {
      const scratch = new DataView(new ArrayBuffer(4));
      scratch.setUint32(0, view.getUint16(i * 2, true) << 16, true);
      out[i] = scratch.getFloat32(0, true);
    } else {
      const h = view.getUint16(i * 2, true), sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
      out[i] = sign * (exponent === 0 ? fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : Infinity) : (1 + fraction / 1024) * 2 ** (exponent - 15));
    }
  }
  return out;
}

/** Pool the native state's learned token embeddings to a normalized semantic index vector. */
export function pooled(state: NativeStateArtifact): Float32Array {
  const values = rows(state), length = state.tensors.token_ids?.shape[0], width = state.tensors.embeddings?.shape[1];
  if (!length || !width || values.length !== length * width) throw new TypeError('native state token and embedding lengths differ');
  const out = new Float32Array(width);
  for (let r = 0; r < length; r++) for (let c = 0; c < width; c++) out[c]! += values[r * width + c]! / length;
  return normalized(out);
}
function normalized(vector: Float32Array): Float32Array {
  const norm = Math.hypot(...vector) || 1;
  return vector.map(value => value / norm);
}
const dot = (a: Float32Array, b: Float32Array) => { let sum = 0; for (let i = 0; i < a.length; i++) sum += a[i]! * b[i]!; return sum; };

export type IndexEntry = { id: string, vector: Float32Array };
export type Hit = { id: string, score: number };

/** An inverted-file index over pooled block vectors of one dialect and width. */
export class IvfIndex {
  centroids: Float32Array[] = [];
  readonly lists = new Map<number, IndexEntry[]>();
  constructor(readonly dialect: string, readonly width: number) {}

  /** Pick `k` centroids by k-means over these vectors, seeded deterministically by farthest points. */
  train(vectors: Float32Array[], k: number, iterations = 10): this {
    this.centroids = vectors.length ? [vectors[0]!.slice()] : [];
    while (this.centroids.length < Math.min(k, vectors.length)) {
      // The next seed is the vector least similar to every seed so far.
      const closeness = vectors.map(vector => Math.max(...this.centroids.map(seed => dot(vector, seed))));
      this.centroids.push(vectors[closeness.indexOf(Math.min(...closeness))]!.slice());
    }
    for (let round = 0; round < iterations; round++) {
      const sums = this.centroids.map(() => new Float32Array(this.width)), counts = this.centroids.map(() => 0);
      for (const vector of vectors) { const c = this.nearest(vector, 1)[0]!; counts[c]!++; sums[c]!.forEach((_, i) => { sums[c]![i]! += vector[i]!; }); }
      this.centroids = this.centroids.map((centroid, c) => counts[c] ? normalized(sums[c]!) : centroid);
    }
    return this;
  }

  private nearest(vector: Float32Array, count: number): number[] {
    return this.centroids.map((centroid, c) => [c, dot(vector, centroid)] as const).sort((a, b) => b[1] - a[1]).slice(0, count).map(([c]) => c);
  }

  private check(meta: Pick<NeuraleseBlockMeta, 'dialect' | 'width'>) {
    if (meta.dialect !== this.dialect || meta.width !== this.width) {
      throw new RangeError(`this index serves dialect ${this.dialect} (width ${this.width}); a ${meta.dialect} block (width ${meta.width}) must be re-encoded for the exact checkpoint/tokenizer embedding identity first`);
    }
  }

  add(state: NativeStateArtifact): void {
    const meta = { dialect: nativeStateDialect(state), width: state.tensors.embeddings?.shape[1] ?? 0 };
    this.check(meta);
    if (!this.centroids.length) throw new Error('train the index before adding blocks');
    const vector = pooled(state), list = this.nearest(vector, 1)[0]!;
    const entries = this.lists.get(list) ?? [];
    if (!entries.some(entry => entry.id === state.id)) entries.push({ id: state.id, vector });
    this.lists.set(list, entries);
  }

  /** The `count` most similar blocks, probing the `probes` closest lists. */
  search(query: NativeStateArtifact, count = 5, probes = 2): Hit[] {
    this.check({ dialect: nativeStateDialect(query), width: query.tensors.embeddings?.shape[1] ?? 0 });
    const vector = pooled(query);
    return this.nearest(vector, probes).flatMap(list => this.lists.get(list) ?? [])
      .map(entry => ({ id: entry.id, score: dot(vector, entry.vector) })).sort((a, b) => b.score - a.score).slice(0, count);
  }

  toJSON() {
    const encode = (vector: Float32Array) => Buffer.from(vector.buffer, vector.byteOffset, vector.byteLength).toString('base64');
    return { format: 'natlang.ivf/1', dialect: this.dialect, width: this.width, centroids: this.centroids.map(encode),
      lists: [...this.lists].map(([list, entries]) => [list, entries.map(entry => [entry.id, encode(entry.vector)])]) };
  }

  static fromJSON(value: ReturnType<IvfIndex['toJSON']>): IvfIndex {
    const decode = (text: string) => { const bytes = Buffer.from(text, 'base64'); return new Float32Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)); };
    const index = new IvfIndex(value.dialect, value.width);
    index.centroids = value.centroids.map(decode);
    for (const [list, entries] of value.lists as [number, [string, string][]][]) index.lists.set(list, entries.map(([id, vector]) => ({ id, vector: decode(vector) })));
    return index;
  }
}
