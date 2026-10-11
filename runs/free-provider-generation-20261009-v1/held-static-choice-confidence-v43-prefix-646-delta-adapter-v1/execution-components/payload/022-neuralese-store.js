/**
 * Neuralese tensor store (S0 §3.4, S4 §2.4).
 *
 * Every soft value is an immutable, content-addressed block of vectors. The runtime's value model holds only
 * references (`{ $neuralese: { type, id } }`); bytes move between this store, `.nz` files and model servers.
 */
import { digest } from './hash.js';
const BYTES = { f32: 4, f16: 2, bf16: 2 };
const BASE32 = 'abcdefghijklmnopqrstuvwxyz234567';
function base32(bytes) {
    let bits = 0, value = 0, out = '';
    for (const byte of bytes) {
        value = (value << 8) | byte;
        bits += 8;
        while (bits >= 5) {
            out += BASE32[(value >>> (bits - 5)) & 31];
            bits -= 5;
        }
    }
    if (bits > 0)
        out += BASE32[(value << (5 - bits)) & 31];
    return out;
}
/** Content ID: `nz1_` + base32 SHA-256 over dialect, shape, dtype and payload bytes. */
/** Blocks that are pure constants of a dialect (the zero-length identity of `combine`), known to every runtime. */
const constants = new Map();
/** The zero-length block of a dialect and width; registered so that upload paths can supply it. */
export function emptyBlock(dialect, width) {
    const id = neuraleseContentId({ dialect, length: 0, width, dtype: 'f32', data: new Uint8Array(0) });
    let block = constants.get(id);
    if (!block)
        constants.set(id, block = { meta: { id, dialect, type: 'Neuralese<unknown>', length: 0, width, dtype: 'f32',
                producer: { kind: 'constant', name: 'empty' } }, data: new Uint8Array(0) });
    return block;
}
/** A registered constant block, when `id` names one. */
export function constantBlock(id) {
    return constants.get(id);
}
export function neuraleseContentId(block) {
    const header = new TextEncoder().encode(`natlang.neuralese-block/1\0${block.dialect}\0${block.length}x${block.width}\0${block.dtype}\0`);
    const joined = new Uint8Array(header.length + block.data.length);
    joined.set(header);
    joined.set(block.data, header.length);
    return `nz1_${base32(digest(joined))}`;
}
export function checkBlockShape(block) {
    if (!Number.isSafeInteger(block.length) || block.length < 0 || !Number.isSafeInteger(block.width) || block.width <= 0)
        throw new RangeError('a Neuralese block needs a nonnegative integer length and a positive integer width');
    if (block.data.length !== block.length * block.width * BYTES[block.dtype])
        throw new RangeError(`a ${block.length}x${block.width} ${block.dtype} block has ${block.length * block.width * BYTES[block.dtype]} bytes, ` +
            `got ${block.data.length}`);
}
/** Process-local store; the default for runtimes without a model-side store, and for tests. */
export class MemoryNeuraleseStore {
    blocks = new Map();
    /** Metadata of blocks held elsewhere (`note`). */
    noted = new Map();
    pins = new Map();
    async put(block) {
        checkBlockShape(block);
        const id = neuraleseContentId(block);
        const found = this.blocks.get(id);
        if (found) {
            // Same bytes: keep the first producer; learn a type the first writer did not know.
            if (!found.meta.type && block.type)
                found.meta = { ...found.meta, type: block.type };
            return { ...found.meta };
        }
        const { data, ...rest } = block;
        const meta = { id, ...rest };
        this.blocks.set(id, { meta, data: data.slice() });
        return { ...meta };
    }
    async get(id) {
        const found = this.blocks.get(id);
        return found && { meta: { ...found.meta }, data: found.data.slice() };
    }
    async meta(id) {
        const found = this.blocks.get(id);
        return found && { ...found.meta };
    }
    async has(id) { return this.blocks.has(id); }
    peek(id) {
        const found = this.blocks.get(id)?.meta ?? this.noted.get(id);
        return found && { ...found };
    }
    note(meta) {
        if (!this.blocks.has(meta.id))
            this.noted.set(meta.id, { ...meta });
    }
    async pin(id) {
        if (!this.blocks.has(id))
            throw new Error(`neuralese-unknown-block: ${id}`);
        this.pins.set(id, (this.pins.get(id) ?? 0) + 1);
    }
    async unpin(id) {
        const count = (this.pins.get(id) ?? 0) - 1;
        if (count > 0)
            this.pins.set(id, count);
        else
            this.pins.delete(id);
    }
    async collect(referenced) {
        const removed = [];
        for (const id of [...this.blocks.keys()])
            if (!referenced.has(id) && !this.pins.has(id)) {
                this.blocks.delete(id);
                removed.push(id);
            }
        for (const id of [...this.noted.keys()])
            if (!referenced.has(id))
                this.noted.delete(id);
        return removed;
    }
    get size() { return this.blocks.size; }
}
/**
 * The untrained stand-in port (S4 §4.3): until a trained writer exists, a written block is the token embeddings of
 * the text the model wrote inside the literal. It exercises protocol, store and runtime paths, not writing quality.
 */
export class StandInNeuralesePort {
    store;
    embed;
    width;
    dialect;
    constructor(store, embed, width, dialect = 'nd:standin@0') {
        this.store = store;
        this.embed = embed;
        this.width = width;
        this.dialect = dialect;
    }
    async write(text, options = {}) {
        const vectors = await this.embed(text);
        const data = new Float32Array(vectors.length * this.width);
        vectors.forEach((vector, index) => {
            if (vector.length !== this.width)
                throw new RangeError(`embedding width ${vector.length}, expected ${this.width}`);
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
export function hashingEmbedder(width, options = {}) {
    return text => (options.preserveWhitespace ? text.match(/\s+|\S+/g) ?? [] : text.split(/\s+/).filter(Boolean)).map(token => {
        const vector = new Float32Array(width);
        let seed = digest(token);
        for (let index = 0; index < width; index++) {
            if (index && index % seed.length === 0)
                seed = digest(seed);
            vector[index] = (seed[index % seed.length] - 127.5) / 127.5;
        }
        return vector;
    });
}
