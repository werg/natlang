import {verifyNativeDerivativeGradientOrigin} from '../native/native-derivative-origin.js';
import { verifyNativeStateDerivation } from '../native/native-state-derivation.js';
/**
 * Node: a persistent native artifact archive in a directory (the Node twin of the browser's `OpfsNeuraleseStore`,
 * browser/neuralese-opfs-store.ts). Positional/hybrid states and named tensor artifacts are content-addressed and
 * stored as their typed safetensors wire format. It is the archive behind native artifact restore: the runtime can
 * upload exact artifacts again after a model server or host process restarts.
 *
 * Metadata of every artifact is read from the files' headers when the store opens and kept in memory, so `peek`, `meta`
 * and `has` need no I/O. Payloads are read on demand, checked against their ID (an artifact whose bytes do not hash to
 * its name is removed and reported as `neuralese-block-integrity`), and kept resident within a byte budget, least recently
 * used first out. `note` remembers metadata of blocks held elsewhere (a server's), for `peek` only.
 *
 * Several processes may share a directory: writes go to a temporary file renamed into place (equal IDs are equal bytes),
 * and an artifact another process wrote is found on a lookup miss. Pins are this store object's own, and `collect` deletes
 * files for every process: collect only with every user's references.
 *
 * Node-only (node:fs): exported from the Node entry point, never part of the browser graph.
 */
import { mkdir, open, readFile, readdir, rename, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { nativeArtifactDependencyClosure, decodeNativeState, encodeNativeState, nativeStateBlockMeta,
  decodeNativeTensorArtifact, encodeNativeTensorArtifact, nativeTensorArtifactBlockMeta,
  type NativeStateArtifact, type NativeTensorArtifact,
  type NeuraleseBlockMeta, type NeuraleseStore } from '../native/neuralese-store.js';
import { NzFileError } from '../native/nz-file.js';

export type FileNeuraleseStoreOptions = {
  /** Payload bytes kept in memory after a put or get (default 64 MiB); older payloads are read from disk again. */
  residentBytes?: number;
};

const ID = /^(?:nzs1|nnt1)_[a-z2-7]{52}$/;

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
    const native = JSON.parse(String(parsed.__metadata__?.['natlang.native_state'] ?? 'null'));
    if (native?.id) return nativeStateBlockMeta({ id: native.id, metadata: native, tensors: parsed as never });
    const tensor = JSON.parse(String(parsed.__metadata__?.['natlang.native_tensor_artifact'] ?? 'null'));
    if (tensor?.id) return nativeTensorArtifactBlockMeta({ id: tensor.id, metadata: tensor });
    throw new Error('no native artifact metadata');
  } finally { await file.close(); }
}

const missing = (error: unknown) => (error as { code?: string } | null)?.code === 'ENOENT';

export class FileNeuraleseStore implements NeuraleseStore {
  private readonly index = new Map<string, NeuraleseBlockMeta>();
  /** Metadata of native artifacts held elsewhere (`note`). */
  private readonly noted = new Map<string, NeuraleseBlockMeta>();
  /** Resident payloads, least recently used first (Map order). */
  private readonly resident = new Map<string, Uint8Array>();
  private residentSize = 0;
  private readonly pins = new Map<string, number>();
  /** One write at a time per block ID. */
  private readonly writes = new Map<string, Promise<unknown>>();

  private constructor(readonly directory: string, readonly residentBytes: number,
    /** Files at open that were not readable native artifacts (left in place, not indexed). */
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

  private async writeWire(id: string, data: Uint8Array): Promise<void> {
    const temporary = `${this.path(id)}.${process.pid}.${Math.random().toString(36).slice(2)}.tmp`;
    try {
      const file = await open(temporary, 'wx');
      try { await file.writeFile(data); await file.sync(); } finally { await file.close(); }
      await rename(temporary, this.path(id));
      const directory = await open(this.directory, 'r');
      try { await directory.sync(); } finally { await directory.close(); }
    } catch (error) {
      await rm(temporary, { force: true }).catch(() => undefined);
      throw error;
    }
  }

  /** Metadata of a native artifact on disk that is not indexed (another process wrote it), indexed from now on. */
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

  async putNativeState(input: Omit<NativeStateArtifact, 'id'>): Promise<NativeStateArtifact> {
    if(input.metadata.derivation)await verifyNativeStateDerivation(this,input);
    const wire = encodeNativeState(input);
    const artifact = decodeNativeState(wire);
    return this.serialized(artifact.id, async () => {
      // Validate an existing file before accepting its immutable identity.
      if (await this.has(artifact.id)) {
        const existing = await this.getNativeState(artifact.id);
        if (existing) return existing;
      }
      await this.writeWire(artifact.id, wire);
      this.index.set(artifact.id, nativeStateBlockMeta(artifact));
      this.noted.delete(artifact.id);
      this.remember(artifact.id, wire);
      return structuredClone(artifact);
    });
  }

  async getNativeState(id: string): Promise<NativeStateArtifact | undefined> {
    if (!/^nzs1_[a-z2-7]{52}$/.test(id) || !(await this.has(id))) return undefined;
    let wire = this.resident.get(id);
    if (!wire) try { wire = await readFile(this.path(id)); }
    catch (error) { if (missing(error)) { this.forget(id); return undefined; } throw error; }
    try {
      const artifact = decodeNativeState(wire);
      if (artifact.id !== id) throw new Error('content address differs');
      this.remember(id, wire);
      return artifact;
    } catch (error) {
      this.forget(id);
      await rm(this.path(id), { force: true });
      throw new NzFileError('neuralese-block-integrity', `native state ${id}: ${String(error)}`);
    }
  }

  async hasNativeState(id: string): Promise<boolean> {
    return /^nzs1_[a-z2-7]{52}$/.test(id) && await this.has(id);
  }

  async putNativeTensorArtifact(input: Omit<NativeTensorArtifact, 'id'>): Promise<NativeTensorArtifact> {
    await verifyNativeDerivativeGradientOrigin(this,input);
    const wire = encodeNativeTensorArtifact(input), artifact = decodeNativeTensorArtifact(wire);
    return this.serialized(artifact.id, async () => {
      if (await this.has(artifact.id)) {
        const existing = await this.getNativeTensorArtifact(artifact.id);
        if (existing) return existing;
      }
      await this.writeWire(artifact.id, wire);
      this.index.set(artifact.id, nativeTensorArtifactBlockMeta(artifact));
      this.noted.delete(artifact.id); this.remember(artifact.id, wire);
      return structuredClone(artifact);
    });
  }

  async getNativeTensorArtifact(id: string): Promise<NativeTensorArtifact | undefined> {
    if (!/^nnt1_[a-z2-7]{52}$/.test(id) || !(await this.has(id))) return undefined;
    let wire = this.resident.get(id);
    if (!wire) try { wire = await readFile(this.path(id)); }
    catch (error) { if (missing(error)) { this.forget(id); return undefined; } throw error; }
    try {
      const artifact = decodeNativeTensorArtifact(wire);
      if (artifact.id !== id) throw new Error('content address differs');
      this.remember(id, wire); return artifact;
    } catch (error) {
      this.forget(id); await rm(this.path(id), { force: true });
      throw new NzFileError('neuralese-block-integrity', `native tensor artifact ${id}: ${String(error)}`);
    }
  }

  async hasNativeTensorArtifact(id: string): Promise<boolean> {
    return /^nnt1_[a-z2-7]{52}$/.test(id) && await this.has(id);
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
    if (!(await this.has(id))) throw new Error(`neuralese-unknown-native-artifact: ${id}`);
    this.pins.set(id, (this.pins.get(id) ?? 0) + 1);
  }

  async unpin(id: string): Promise<void> {
    const count = (this.pins.get(id) ?? 0) - 1;
    if (count > 0) this.pins.set(id, count); else this.pins.delete(id);
  }

  async collect(referenced: ReadonlySet<string>): Promise<string[]> {
    referenced = await nativeArtifactDependencyClosure(this, [...referenced, ...this.pins.keys()]);
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

  /** IDs of the native artifacts this store holds (as indexed). */
  ids(): string[] { return [...this.index.keys()]; }
  get size(): number { return this.index.size; }
  /** Payload bytes currently held in memory. */
  get residentInUse(): number { return this.residentSize; }
}
