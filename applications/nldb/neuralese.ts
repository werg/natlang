/**
 * Neuralese columns for the natural-language database: block storage and similarity indexing (see NEURALESE.md).
 *
 * A neuralese value is an immutable block of vectors in one dialect (the model channel that wrote it), named by its
 * content ID. A table column of type `neuralese` holds IDs; the blocks live in blocks/<dialect>/<id>.safetensors.
 * An index serves one dialect and width: each block is pooled to one normalized vector (the mean of its rows) and
 * filed under its nearest centroid (IVF); a search probes the closest lists and ranks by cosine similarity.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { neuraleseContentId, type NeuraleseBlock, type NeuraleseBlockInput, type NeuraleseBlockMeta, type NeuraleseStore } from '@natlang/node';

const SAFE_DTYPE = { f32: 'F32', f16: 'F16', bf16: 'BF16' } as const;
const dialectDirectory = (dialect: string) => encodeURIComponent(dialect);

/** One block as a safetensors file: the natlang metadata in the header, the payload as tensor `payload`. */
export function encodeBlock(block: NeuraleseBlock): Uint8Array {
  const header = JSON.stringify({ __metadata__: { 'natlang.block': JSON.stringify(block.meta) },
    payload: { dtype: SAFE_DTYPE[block.meta.dtype], shape: [block.meta.length, block.meta.width], data_offsets: [0, block.data.length] } });
  const padded = new TextEncoder().encode(header.padEnd(Math.ceil(header.length / 8) * 8, ' '));
  const out = new Uint8Array(8 + padded.length + block.data.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(padded.length), true);
  out.set(padded, 8);
  out.set(block.data, 8 + padded.length);
  return out;
}

export function decodeBlock(bytes: Uint8Array): NeuraleseBlock {
  const size = Number(new DataView(bytes.buffer, bytes.byteOffset).getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(bytes.subarray(8, 8 + size))) as { __metadata__: Record<string, string>, payload: { data_offsets: [number, number] } };
  const meta = JSON.parse(header.__metadata__['natlang.block']!) as NeuraleseBlockMeta;
  const [start, end] = header.payload.data_offsets;
  return { meta, data: bytes.slice(8 + size + start, 8 + size + end) };
}

/** Blocks stored as files under a database folder; content-addressed, so storing the same block twice is a no-op. */
export class FolderBlockStore implements NeuraleseStore {
  private readonly pins = new Map<string, number>();
  constructor(readonly root: string) {}
  private path(meta: Pick<NeuraleseBlockMeta, 'id' | 'dialect'>) { return join(this.root, 'blocks', dialectDirectory(meta.dialect), `${meta.id}.safetensors`); }
  private find(id: string): string | undefined {
    const base = join(this.root, 'blocks');
    if (!existsSync(base)) return undefined;
    for (const dialect of readdirSync(base)) { const path = join(base, dialect, `${id}.safetensors`); if (existsSync(path)) return path; }
    return undefined;
  }
  async put(block: NeuraleseBlockInput): Promise<NeuraleseBlockMeta> {
    const meta: NeuraleseBlockMeta = { id: neuraleseContentId(block), dialect: block.dialect, length: block.length, width: block.width,
      dtype: block.dtype, ...(block.type ? { type: block.type } : {}), ...(block.producer ? { producer: block.producer } : {}) };
    const path = this.path(meta);
    if (!existsSync(path)) {
      mkdirSync(dirname(path), { recursive: true });
      writeFileSync(`${path}.tmp`, encodeBlock({ meta, data: block.data }));
      renameSync(`${path}.tmp`, path);
    }
    return meta;
  }
  async get(id: string): Promise<NeuraleseBlock | undefined> { const path = this.find(id); return path ? decodeBlock(new Uint8Array(readFileSync(path))) : undefined; }
  async meta(id: string): Promise<NeuraleseBlockMeta | undefined> { return (await this.get(id))?.meta; }
  async has(id: string): Promise<boolean> { return this.find(id) !== undefined; }
  async pin(id: string): Promise<void> { this.pins.set(id, (this.pins.get(id) ?? 0) + 1); }
  async unpin(id: string): Promise<void> { const n = (this.pins.get(id) ?? 0) - 1; if (n > 0) this.pins.set(id, n); else this.pins.delete(id); }
  /** Blocks no row refers to and nobody pinned are removed; rows are the references. */
  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    const removed: string[] = [];
    const base = join(this.root, 'blocks');
    if (!existsSync(base)) return removed;
    const { rmSync } = await import('node:fs');
    for (const dialect of readdirSync(base)) for (const file of readdirSync(join(base, dialect))) {
      const id = file.replace(/\.safetensors$/, '');
      if (!referenced.has(id) && !this.pins.has(id)) { rmSync(join(base, dialect, file)); removed.push(id); }
    }
    return removed;
  }
}

/** A block's rows as f32 numbers (bf16 and f16 widened). */
export function rows(block: NeuraleseBlock): Float32Array {
  const { dtype, length, width } = block.meta, view = new DataView(block.data.buffer, block.data.byteOffset, block.data.byteLength);
  const out = new Float32Array(length * width);
  for (let i = 0; i < out.length; i++) {
    if (dtype === 'f32') out[i] = view.getFloat32(i * 4, true);
    else if (dtype === 'bf16') { const scratch = new DataView(new ArrayBuffer(4)); scratch.setUint32(0, view.getUint16(i * 2, true) << 16); out[i] = scratch.getFloat32(0); }
    else {
      const h = view.getUint16(i * 2, true), sign = h & 0x8000 ? -1 : 1, exponent = (h >> 10) & 0x1f, fraction = h & 0x3ff;
      out[i] = sign * (exponent === 0 ? fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : Infinity) : (1 + fraction / 1024) * 2 ** (exponent - 15));
    }
  }
  return out;
}

/** The block's index vector: the mean of its rows, scaled to unit length. */
export function pooled(block: NeuraleseBlock): Float32Array {
  const values = rows(block), { length, width } = block.meta, out = new Float32Array(width);
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
      throw new RangeError(`this index serves dialect ${this.dialect} (width ${this.width}); a ${meta.dialect} block (width ${meta.width}) must be re-encoded in it first`);
    }
  }

  add(block: NeuraleseBlock): void {
    this.check(block.meta);
    if (!this.centroids.length) throw new Error('train the index before adding blocks');
    const vector = pooled(block), list = this.nearest(vector, 1)[0]!;
    const entries = this.lists.get(list) ?? [];
    if (!entries.some(entry => entry.id === block.meta.id)) entries.push({ id: block.meta.id, vector });
    this.lists.set(list, entries);
  }

  /** The `count` most similar blocks, probing the `probes` closest lists. */
  search(query: NeuraleseBlock, count = 5, probes = 2): Hit[] {
    this.check(query.meta);
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
