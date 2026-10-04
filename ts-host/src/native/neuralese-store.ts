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

/** Client interface to a tensor store: in memory, on disk, or a model server's block endpoints. */
export interface NeuraleseStore {
  put(block: NeuraleseBlockInput): Promise<NeuraleseBlockMeta>;
  get(id: string): Promise<NeuraleseBlock | undefined>;
  meta(id: string): Promise<NeuraleseBlockMeta | undefined>;
  has(id: string): Promise<boolean>;
  /** Keep a block through garbage collection (for example while a file or a running call refers to it). */
  pin(id: string): Promise<void>;
  unpin(id: string): Promise<void>;
  /** Drop every unpinned block that `referenced` does not name. Returns the IDs removed. */
  collect(referenced: ReadonlySet<string>): Promise<string[]>;
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
  async has(id: string): Promise<boolean> { return this.blocks.has(id); }
  async pin(id: string): Promise<void> {
    if (!this.blocks.has(id)) throw new Error(`neuralese-unknown-block: ${id}`);
    this.pins.set(id, (this.pins.get(id) ?? 0) + 1);
  }
  async unpin(id: string): Promise<void> {
    const count = (this.pins.get(id) ?? 0) - 1;
    if (count > 0) this.pins.set(id, count); else this.pins.delete(id);
  }
  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    const removed: string[] = [];
    for (const id of [...this.blocks.keys()])
      if (!referenced.has(id) && !this.pins.has(id)) { this.blocks.delete(id); removed.push(id); }
    return removed;
  }
  get size(): number { return this.blocks.size; }
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
 */
export function hashingEmbedder(width: number): TokenEmbedder {
  return text => text.split(/\s+/).filter(Boolean).map(token => {
    const vector = new Float32Array(width);
    let seed = digest(token);
    for (let index = 0; index < width; index++) {
      if (index && index % seed.length === 0) seed = digest(seed);
      vector[index] = (seed[index % seed.length]! - 127.5) / 127.5;
    }
    return vector;
  });
}
