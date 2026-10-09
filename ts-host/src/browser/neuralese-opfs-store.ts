/**
 * The browser runtime's Neuralese block archive (S4 §6, §11.3): content-addressed blocks in the origin-private file
 * system (OPFS), one file per block named by its ID, holding the block as a safetensors body (`encodeBlockBody`, the
 * block endpoints' codec). It is the archive behind block restore (`withRestoredBlocks`): the in-page WebAssembly
 * service keeps its blocks in memory and loses them on a page reload or an engine restart; the runtime uploads them
 * again from here.
 *
 * Metadata of every block is read from the files' headers when the store opens and kept in memory, so `peek`, `meta`
 * and `has` need no I/O. Payloads are read on demand, checked against their ID (a block whose bytes do not hash to its
 * name is removed and reported as `neuralese-block-integrity`), and kept resident within a byte budget, least recently
 * used first out. Blocks move between the store and `.nz` files by ID (`importNz`, `exportNz`).
 *
 * The directory is per origin: tabs of one origin share it. Writes are safe to share (equal IDs are equal bytes; a
 * file is replaced only when its write closes); a block another tab wrote is found on a lookup miss. Pins are this
 * store object's own, and `collect` deletes files for the whole origin: collect only with every runtime's references.
 */
import { checkBlockShape, MemoryNeuraleseStore, neuraleseContentId, type NeuraleseBlock, type NeuraleseBlockInput, type NeuraleseBlockMeta,
  type NeuraleseDtype, type NeuraleseStore } from '../native/neuralese-store.js';
import { decodeBlockBody, encodeBlockBody } from '../model/neuralese-server.js';
import { decodeNz, encodeNz, NZ_FORMAT, NzFileError, type NzBlockHeader, type NzHeader } from '../native/nz-file.js';

type Writable = { write(data: Uint8Array): Promise<void>; close(): Promise<void>; abort(): Promise<void> };
type OpfsFileHandle = { getFile(): Promise<Blob>; createWritable(): Promise<Writable> };
export type OpfsDirectoryHandle = {
  getDirectoryHandle(name: string, options?: { create?: boolean }): Promise<OpfsDirectoryHandle>;
  getFileHandle(name: string, options?: { create?: boolean }): Promise<OpfsFileHandle>;
  removeEntry(name: string): Promise<void>;
  keys(): AsyncIterable<string>;
};
/** `navigator.storage`, or a stand-in with its `getDirectory`. */
export type OpfsStorage = { getDirectory(): Promise<OpfsDirectoryHandle> };

export type OpfsNeuraleseStoreOptions = {
  /** Defaults to `navigator.storage`. */
  storage?: OpfsStorage;
  /** Directory under the OPFS root (default `natlang-neuralese-blocks`). */
  directory?: string;
  /** Payload bytes kept in memory after a put or get (default 64 MiB); older payloads are read from OPFS again. */
  residentBytes?: number;
};

const DIRECTORY = 'natlang-neuralese-blocks';
const ID = /^nz1_[a-z2-7]{52}$/;
const DTYPE_TO_FILE: Record<NeuraleseDtype, NzBlockHeader['dtype']> = { f32: 'F32', f16: 'F16', bf16: 'BF16' };

/** Whether this browser offers OPFS (`navigator.storage.getDirectory`). */
export function opfsAvailable(storage: unknown = globalThis.navigator?.storage): boolean {
  return typeof (storage as OpfsStorage | undefined)?.getDirectory === 'function';
}

async function bytesOf(blob: Blob, start?: number, end?: number): Promise<Uint8Array> {
  return new Uint8Array(await (start === undefined ? blob : blob.slice(start, end)).arrayBuffer());
}

/** A block file's metadata from its safetensors header alone, without reading the payload. */
async function readMeta(file: Blob): Promise<NeuraleseBlockMeta> {
  if (file.size < 8) throw new Error('not a safetensors body');
  const size = Number(new DataView((await bytesOf(file, 0, 8)).buffer).getBigUint64(0, true));
  if (8 + size > file.size) throw new Error('header length exceeds the file');
  const header = JSON.parse(new TextDecoder().decode(await bytesOf(file, 8, 8 + size))) as Record<string, any>;
  const meta = JSON.parse(String(header.__metadata__?.['natlang.block'] ?? 'null')) as NeuraleseBlockMeta | null;
  if (!meta?.id) throw new Error('no natlang.block metadata');
  return meta;
}

export class OpfsNeuraleseStore implements NeuraleseStore {
  private readonly index = new Map<string, NeuraleseBlockMeta>();
  /** Resident payloads, least recently used first (Map order). */
  private readonly resident = new Map<string, Uint8Array>();
  private residentSize = 0;
  private readonly pins = new Map<string, number>();
  /** One write at a time per block ID. */
  private readonly writes = new Map<string, Promise<unknown>>();

  private constructor(private readonly directory: OpfsDirectoryHandle, readonly residentBytes: number,
    /** Files at open that were not readable blocks (left in place, not indexed). */
    readonly unreadable: readonly string[]) {}

  /** Open (creating) the archive and index the metadata of every block it holds. */
  static async open(options: OpfsNeuraleseStoreOptions = {}): Promise<OpfsNeuraleseStore> {
    const storage = options.storage ?? (globalThis.navigator?.storage as unknown as OpfsStorage | undefined);
    if (!opfsAvailable(storage)) throw new Error('OpfsNeuraleseStore: this browser offers no origin-private file system');
    const directory = await (await storage!.getDirectory()).getDirectoryHandle(options.directory ?? DIRECTORY, { create: true });
    const entries: [string, NeuraleseBlockMeta][] = [], unreadable: string[] = [];
    for await (const name of directory.keys()) {
      if (!ID.test(name)) continue; // swap files of unfinished writes and anything else
      try {
        const meta = await readMeta(await (await directory.getFileHandle(name)).getFile());
        if (meta.id !== name) throw new Error(`names block ${meta.id}`);
        entries.push([name, meta]);
      } catch { unreadable.push(name); }
    }
    const store = new OpfsNeuraleseStore(directory, options.residentBytes ?? 64 * 2 ** 20, unreadable);
    for (const [id, meta] of entries) store.index.set(id, meta);
    return store;
  }

  private remember(id: string, data: Uint8Array): void {
    const old = this.resident.get(id);
    if (old) { this.resident.delete(id); this.residentSize -= old.length; }
    if (data.length > this.residentBytes) return;
    this.resident.set(id, data);
    this.residentSize += data.length;
    for (const [oldest, bytes] of this.resident) {
      if (this.residentSize <= this.residentBytes) break;
      this.resident.delete(oldest); this.residentSize -= bytes.length;
    }
  }

  private forget(id: string): void {
    this.index.delete(id);
    const old = this.resident.get(id);
    if (old) { this.resident.delete(id); this.residentSize -= old.length; }
  }

  private serialized<T>(id: string, run: () => Promise<T>): Promise<T> {
    const next = (this.writes.get(id) ?? Promise.resolve()).catch(() => undefined).then(run);
    this.writes.set(id, next);
    void next.finally(() => { if (this.writes.get(id) === next) this.writes.delete(id); }).catch(() => undefined);
    return next;
  }

  private async writeFile(meta: NeuraleseBlockMeta, data: Uint8Array): Promise<void> {
    const writable = await (await this.directory.getFileHandle(meta.id, { create: true })).createWritable();
    try {
      await writable.write(encodeBlockBody(meta, data));
      await writable.close();
    } catch (error) {
      await writable.abort().catch(() => undefined);
      throw error;
    }
  }

  /** Metadata of a block on file that is not indexed (another tab wrote it), indexed from now on. */
  private async discover(id: string): Promise<NeuraleseBlockMeta | undefined> {
    if (!ID.test(id)) return undefined;
    let file: Blob;
    try { file = await (await this.directory.getFileHandle(id)).getFile(); } catch { return undefined; }
    try {
      const meta = await readMeta(file);
      if (meta.id !== id) return undefined;
      this.index.set(id, meta);
      return meta;
    } catch { return undefined; }
  }

  async put(block: NeuraleseBlockInput): Promise<NeuraleseBlockMeta> {
    checkBlockShape(block);
    const id = neuraleseContentId(block);
    return this.serialized(id, async () => {
      const found = this.index.get(id) ?? await this.discover(id);
      if (found) {
        // Same bytes: keep the first producer; learn a type the first writer did not know (and keep it on file).
        if (!found.type && block.type) {
          const meta = { ...found, type: block.type };
          const data = this.resident.get(id) ?? (await this.read(id)).data;
          await this.writeFile(meta, data);
          this.index.set(id, meta);
        }
        return { ...this.index.get(id)! };
      }
      const { data, ...rest } = block;
      const meta: NeuraleseBlockMeta = { id, ...rest };
      const copy = data.slice();
      await this.writeFile(meta, copy);
      this.index.set(id, meta);
      this.remember(id, copy);
      return { ...meta };
    });
  }

  /** Read and verify a block's file. A block whose bytes do not hash to its ID is removed. */
  private async read(id: string): Promise<NeuraleseBlock> {
    const file = await (await this.directory.getFileHandle(id)).getFile();
    let block: NeuraleseBlock | undefined, actual: string | undefined;
    try {
      block = decodeBlockBody(await bytesOf(file));
      actual = neuraleseContentId({ ...block.meta, data: block.data });
    } catch { /* unreadable: reported below */ }
    if (!block || actual !== id || block.meta.id !== id) {
      this.forget(id);
      await this.directory.removeEntry(id).catch(() => undefined);
      throw new NzFileError('neuralese-block-integrity', `block ${id} in the browser store ${actual ? `hashes to ${actual}` :
        'is unreadable'}; it was removed`);
    }
    return block;
  }

  async get(id: string): Promise<NeuraleseBlock | undefined> {
    const meta = this.index.get(id) ?? await this.discover(id);
    if (!meta) return undefined;
    const cached = this.resident.get(id);
    if (cached) { this.remember(id, cached); return { meta: { ...meta }, data: cached.slice() }; }
    let block: NeuraleseBlock;
    try { block = await this.read(id); }
    catch (error) {
      if ((error as { name?: string }).name === 'NotFoundError') { this.forget(id); return undefined; } // deleted by another tab
      throw error;
    }
    this.remember(id, block.data);
    return { meta: { ...meta }, data: block.data.slice() };
  }

  async meta(id: string): Promise<NeuraleseBlockMeta | undefined> {
    const meta = this.index.get(id) ?? await this.discover(id);
    return meta && { ...meta };
  }

  async has(id: string): Promise<boolean> { return (await this.meta(id)) !== undefined; }

  peek(id: string): NeuraleseBlockMeta | undefined {
    const meta = this.index.get(id);
    return meta && { ...meta };
  }

  async pin(id: string): Promise<void> {
    if (!(await this.has(id))) throw new Error(`neuralese-unknown-block: ${id}`);
    this.pins.set(id, (this.pins.get(id) ?? 0) + 1);
  }

  async unpin(id: string): Promise<void> {
    const count = (this.pins.get(id) ?? 0) - 1;
    if (count > 0) this.pins.set(id, count); else this.pins.delete(id);
  }

  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    const removed: string[] = [];
    for (const id of [...this.index.keys()]) {
      if (referenced.has(id) || this.pins.has(id)) continue;
      await this.serialized(id, async () => {
        await this.directory.removeEntry(id).catch(() => undefined);
        this.forget(id);
      });
      removed.push(id);
    }
    return removed;
  }

  /** IDs of the blocks this store holds (as indexed). */
  ids(): string[] { return [...this.index.keys()]; }
  get size(): number { return this.index.size; }
  /** Payload bytes currently held in memory. */
  get residentInUse(): number { return this.residentSize; }

  /** Put every block of a `.nz` file (checked against its ID by the decoder). Returns the block IDs, in file order. */
  async importNz(bytes: Uint8Array): Promise<string[]> {
    const { blocks } = decodeNz(bytes);
    for (const block of blocks.values()) {
      const { id: _, ...rest } = block.meta;
      await this.put({ ...rest, data: block.data });
    }
    return [...blocks.keys()];
  }

  /**
   * A `.nz` file holding the blocks `ids` (and no exports unless given; `saveNz` writes typed exports from any store).
   * `dialect` defaults to the first block's.
   */
  async exportNz(ids: Iterable<string>, options: { dialect?: string; exports?: NzHeader['exports'];
    provenance?: Record<string, unknown> } = {}): Promise<Uint8Array> {
    const headers: Record<string, NzBlockHeader> = {}, data = new Map<string, Uint8Array>();
    for (const id of new Set(ids)) {
      const block = await this.get(id);
      if (!block) throw new NzFileError('neuralese-unknown-block', `block ${id} is not in the browser store`);
      const { meta } = block;
      headers[id] = { dialect: meta.dialect, length: meta.length, width: meta.width, dtype: DTYPE_TO_FILE[meta.dtype],
        ...(meta.producer ? { producer: meta.producer } : {}), ...(meta.truncated ? { truncated: true } : {}) };
      data.set(id, block.data);
    }
    const dialect = options.dialect ?? Object.values(headers)[0]?.dialect;
    if (!dialect) throw new NzFileError('neuralese-file-format', 'an empty .nz export needs a dialect');
    return encodeNz({ format: NZ_FORMAT, dialect, exports: options.exports ?? {}, blocks: headers,
      ...(options.provenance ? { provenance: options.provenance } : {}) }, data);
  }
}

/**
 * The browser runtime's block archive: an `OpfsNeuraleseStore` when this browser offers OPFS (blocks outlive page
 * reloads and engine restarts), else a `MemoryNeuraleseStore` (blocks live as long as the page).
 */
export async function openBrowserNeuraleseStore(options: Omit<OpfsNeuraleseStoreOptions, 'storage'> & { storage?: OpfsStorage | null } = {}):
    Promise<NeuraleseStore> {
  const storage = options.storage === undefined ? globalThis.navigator?.storage as unknown as OpfsStorage | undefined : options.storage;
  if (!storage || !opfsAvailable(storage)) return new MemoryNeuraleseStore();
  try { return await OpfsNeuraleseStore.open({ ...options, storage }); }
  catch { return new MemoryNeuraleseStore(); } // OPFS refused (a private window, blocked site data)
}
