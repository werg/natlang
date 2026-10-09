/**
 * Node: a persistent Neuralese block archive in a directory (the Node twin of the browser's `OpfsNeuraleseStore`,
 * browser/neuralese-opfs-store.ts). Blocks are content-addressed: one file per block, named by its ID, holding the block
 * as a safetensors body (`encodeBlockBody`, the block endpoints' codec). It is the archive behind block restore
 * (`withRestoredBlocks`): a model server keeps its blocks in memory and loses them when it restarts; the runtime uploads
 * them again from here, also after the host process itself restarted.
 *
 * Metadata of every block is read from the files' headers when the store opens and kept in memory, so `peek`, `meta`
 * and `has` need no I/O. Payloads are read on demand, checked against their ID (a block whose bytes do not hash to its
 * name is removed and reported as `neuralese-block-integrity`), and kept resident within a byte budget, least recently
 * used first out. `note` remembers metadata of blocks held elsewhere (a server's), for `peek` only.
 *
 * Several processes may share a directory: writes go to a temporary file renamed into place (equal IDs are equal bytes),
 * and a block another process wrote is found on a lookup miss. Pins are this store object's own, and `collect` deletes
 * files for every process: collect only with every user's references.
 *
 * Node-only (node:fs): exported from the Node entry point, never part of the browser graph.
 */
import { mkdir, open, readFile, readdir, rename, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { checkBlockShape, neuraleseContentId, type NeuraleseBlock, type NeuraleseBlockInput, type NeuraleseBlockMeta,
  type NeuraleseStore } from '../native/neuralese-store.js';
import { decodeBlockBody, encodeBlockBody } from '../model/neuralese-server.js';
import { NzFileError } from '../native/nz-file.js';

export type FileNeuraleseStoreOptions = {
  /** Payload bytes kept in memory after a put or get (default 64 MiB); older payloads are read from disk again. */
  residentBytes?: number;
};

const ID = /^nz1_[a-z2-7]{52}$/;

/** A block file's metadata from its safetensors header alone, without reading the payload. */
async function readMeta(path: string): Promise<NeuraleseBlockMeta> {
  const file = await open(path, 'r');
  try {
    const { size: fileSize } = await file.stat();
    if (fileSize < 8) throw new Error('not a safetensors body');
    const prefix = Buffer.alloc(8);
    await file.read(prefix, 0, 8, 0);
    const size = Number(prefix.readBigUInt64LE(0));
    if (8 + size > fileSize) throw new Error('header length exceeds the file');
    const header = Buffer.alloc(size);
    await file.read(header, 0, size, 8);
    const parsed = JSON.parse(header.toString('utf8')) as Record<string, any>;
    const meta = JSON.parse(String(parsed.__metadata__?.['natlang.block'] ?? 'null')) as NeuraleseBlockMeta | null;
    if (!meta?.id) throw new Error('no natlang.block metadata');
    return meta;
  } finally { await file.close(); }
}

const missing = (error: unknown) => (error as { code?: string } | null)?.code === 'ENOENT';

export class FileNeuraleseStore implements NeuraleseStore {
  private readonly index = new Map<string, NeuraleseBlockMeta>();
  /** Metadata of blocks held elsewhere (`note`). */
  private readonly noted = new Map<string, NeuraleseBlockMeta>();
  /** Resident payloads, least recently used first (Map order). */
  private readonly resident = new Map<string, Uint8Array>();
  private residentSize = 0;
  private readonly pins = new Map<string, number>();
  /** One write at a time per block ID. */
  private readonly writes = new Map<string, Promise<unknown>>();

  private constructor(readonly directory: string, readonly residentBytes: number,
    /** Files at open that were not readable blocks (left in place, not indexed). */
    readonly unreadable: readonly string[]) {}

  /** Open (creating) the archive at `directory` and index the metadata of every block it holds. */
  static async open(directory: string, options: FileNeuraleseStoreOptions = {}): Promise<FileNeuraleseStore> {
    await mkdir(directory, { recursive: true });
    const entries: [string, NeuraleseBlockMeta][] = [], unreadable: string[] = [];
    for (const name of await readdir(directory)) {
      if (!ID.test(name)) continue; // temporary files of unfinished writes and anything else
      try {
        const meta = await readMeta(join(directory, name));
        if (meta.id !== name) throw new Error(`names block ${meta.id}`);
        entries.push([name, meta]);
      } catch { unreadable.push(name); }
    }
    const store = new FileNeuraleseStore(directory, options.residentBytes ?? 64 * 2 ** 20, unreadable);
    for (const [id, meta] of entries) store.index.set(id, meta);
    return store;
  }

  private path(id: string): string { return join(this.directory, id); }

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

  /** Write a block's file whole: a temporary file renamed into place, so a reader never sees a partial block. */
  private async writeBlock(meta: NeuraleseBlockMeta, data: Uint8Array): Promise<void> {
    const temporary = `${this.path(meta.id)}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
    try {
      await writeFile(temporary, encodeBlockBody(meta, data));
      await rename(temporary, this.path(meta.id));
    } catch (error) {
      await rm(temporary, { force: true }).catch(() => undefined);
      throw error;
    }
  }

  /** Metadata of a block on disk that is not indexed (another process wrote it), indexed from now on. */
  private async discover(id: string): Promise<NeuraleseBlockMeta | undefined> {
    if (!ID.test(id)) return undefined;
    try {
      const meta = await readMeta(this.path(id));
      if (meta.id !== id) return undefined;
      this.index.set(id, meta);
      this.noted.delete(id);
      return meta;
    } catch { return undefined; }
  }

  async put(block: NeuraleseBlockInput): Promise<NeuraleseBlockMeta> {
    checkBlockShape(block);
    const id = neuraleseContentId(block);
    return this.serialized(id, async () => {
      const found = this.index.get(id) ?? await this.discover(id);
      if (found) {
        // Same bytes: keep the first producer; learn a type the first writer did not know (and keep it on disk).
        if (!found.type && block.type) {
          const meta = { ...found, type: block.type };
          const data = this.resident.get(id) ?? (await this.read(id)).data;
          await this.writeBlock(meta, data);
          this.index.set(id, meta);
        }
        return { ...this.index.get(id)! };
      }
      const { data, ...rest } = block;
      const meta: NeuraleseBlockMeta = { id, ...rest };
      const copy = data.slice();
      await this.writeBlock(meta, copy);
      this.index.set(id, meta);
      this.noted.delete(id);
      this.remember(id, copy);
      return { ...meta };
    });
  }

  /** Read and verify a block's file. A block whose bytes do not hash to its ID is removed. */
  private async read(id: string): Promise<NeuraleseBlock> {
    const bytes = await readFile(this.path(id));
    let block: NeuraleseBlock | undefined, actual: string | undefined;
    try {
      block = decodeBlockBody(new Uint8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength));
      actual = neuraleseContentId({ ...block.meta, data: block.data });
    } catch { /* unreadable: reported below */ }
    if (!block || actual !== id || block.meta.id !== id) {
      this.forget(id);
      await rm(this.path(id), { force: true }).catch(() => undefined);
      throw new NzFileError('neuralese-block-integrity', `block ${id} in the store at ${this.directory} ${actual ? `hashes to ${actual}` :
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
      if (missing(error)) { this.forget(id); return undefined; } // deleted by another process
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
    const meta = this.index.get(id) ?? this.noted.get(id);
    return meta && { ...meta };
  }

  note(meta: NeuraleseBlockMeta): void {
    if (!this.index.has(meta.id)) this.noted.set(meta.id, { ...meta });
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
        await rm(this.path(id), { force: true });
        this.forget(id);
      });
      removed.push(id);
    }
    for (const id of [...this.noted.keys()]) if (!referenced.has(id)) this.noted.delete(id);
    return removed;
  }

  /** IDs of the blocks this store holds (as indexed). */
  ids(): string[] { return [...this.index.keys()]; }
  get size(): number { return this.index.size; }
  /** Payload bytes currently held in memory. */
  get residentInUse(): number { return this.residentSize; }
}
