/**
 * Model files in the browser: downloaded once into the origin-private file system (OPFS) and opened from there on
 * later loads. A cached file is a disk-backed `File`, so the Neuralese worker mounts it (WORKERFS) and reads the
 * slices it needs without holding the whole model in memory. Files are keyed by their SHA-256 when the manifest gives
 * one (a new release is a new file), else by URL. `createWritable` writes to a swap file that replaces the target only
 * when closed, so an interrupted download leaves no truncated file behind. Without OPFS the file is fetched each time.
 */
import type { NeuraleseStore } from '../native/neuralese-store.js';
import { openBrowserNeuraleseStore, type OpfsStorage } from './neuralese-opfs-store.js';
import { chooseNeuraleseBuild, startBrowserNeuralese, type NeuraleseWasmOptions, type StartedNeuralese } from './neuralese-wasm.js';

export type ModelFileRef = { url: string; bytes?: number; sha256?: string };

/**
 * A Neuralese model for the browser: the GGUF model, its port heads, the dialect they speak, and weight adapters it
 * ships as GGUF LoRAs by adapter block ID (`python -m natlang_neuralese.export.adapters`).
 */
export type NeuraleseModelManifest = { id: string; label: string; dialect: string; model: ModelFileRef; heads: ModelFileRef;
  contextTokens?: number; adapters?: readonly { id: string; lora: ModelFileRef }[] };

type FileHandle = { getFile(): Promise<File>; createWritable(): Promise<WritableStream<Uint8Array> & { close(): Promise<void>; abort(): Promise<void> }> };
type DirectoryHandle = { getDirectoryHandle(name: string, options?: { create?: boolean }): Promise<DirectoryHandle>;
  getFileHandle(name: string, options?: { create?: boolean }): Promise<FileHandle>; removeEntry(name: string): Promise<void> };
export type ModelFileStorage = { getDirectory(): Promise<DirectoryHandle> };

const DIRECTORY = 'natlang-model-files';

async function cacheKey(file: ModelFileRef): Promise<string> {
  if (file.sha256) return `sha256-${file.sha256}`;
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(file.url));
  return `url-${[...new Uint8Array(digest)].map(b => b.toString(16).padStart(2, '0')).join('')}`;
}

/**
 * The file at `file.url`, from the OPFS cache or downloaded into it. `onProgress(received, total)` reports a download.
 * A cached file whose size differs from the manifest's `bytes` is downloaded again.
 */
export async function cachedModelFile(file: ModelFileRef, options: { storage?: ModelFileStorage | null; fetcher?: typeof fetch;
  onProgress?: (received: number, total: number | null) => void } = {}): Promise<Blob> {
  const fetcher = options.fetcher ?? globalThis.fetch;
  const storage = options.storage === undefined ? (globalThis.navigator?.storage as unknown as ModelFileStorage | undefined) : options.storage;
  const download = async (): Promise<Response> => {
    const response = await fetcher(file.url);
    if (!response.ok || !response.body) throw new Error(`model file ${file.url}: HTTP ${response.status}`);
    return response;
  };
  if (!storage?.getDirectory) return (await download()).blob();
  const directory = await (await storage.getDirectory()).getDirectoryHandle(DIRECTORY, { create: true });
  const name = await cacheKey(file);
  try {
    const cached = await (await directory.getFileHandle(name)).getFile();
    if (file.bytes === undefined ? cached.size > 0 : cached.size === file.bytes) return cached;
  } catch { /* not cached yet */ }
  const response = await download();
  const total = file.bytes ?? (Number(response.headers.get('content-length')) || null);
  const handle = await directory.getFileHandle(name, { create: true });
  const writable = await handle.createWritable();
  let received = 0;
  const counting = new TransformStream<Uint8Array, Uint8Array>({ transform(chunk, controller) {
    received += chunk.byteLength;
    options.onProgress?.(received, total);
    controller.enqueue(chunk);
  } });
  try {
    await response.body!.pipeThrough(counting).pipeTo(writable);
  } catch (error) {
    await writable.abort().catch(() => undefined);
    throw error;
  }
  const stored = await handle.getFile();
  if (file.bytes !== undefined && stored.size !== file.bytes) {
    await directory.removeEntry(name).catch(() => undefined);
    throw new Error(`model file ${file.url}: got ${stored.size} bytes, the manifest says ${file.bytes}`);
  }
  return stored;
}

/**
 * Start a Neuralese model from its manifest in a Web Worker: the build this browser runs best
 * (`chooseNeuraleseBuild`), model and heads from the OPFS cache. `moduleBase` is the directory (ending in `/`) that
 * holds the `neuralese-wasm*.mjs` builds. `store` is the block archive; by default the OPFS block store in `storage`
 * (an in-memory store without OPFS), so blocks survive a page reload or an engine restart and are restored from it.
 */
export async function startNeuraleseModel(manifest: NeuraleseModelManifest, options: NeuraleseWasmOptions & { worker: Worker;
  moduleBase: string | URL; endpoint?: string; storage?: ModelFileStorage | null; fetcher?: typeof fetch; store?: NeuraleseStore;
  onProgress?: (file: 'model' | 'heads', received: number, total: number | null) => void; gpu?: boolean }):
    Promise<StartedNeuralese & { build: string; reason: string;
      /** The manifest's adapter LoRAs (from the OPFS cache), for `neuraleseServerModelTurn({ adapterLoras })`. */
      adapterLoras: (id: string) => Promise<Uint8Array | null> }> {
  const [model, heads] = await Promise.all((['model', 'heads'] as const).map(which => cachedModelFile(manifest[which], {
    storage: options.storage, fetcher: options.fetcher, onProgress: (received, total) => options.onProgress?.(which, received, total) })));
  const chosen = await chooseNeuraleseBuild(globalThis as never, { gpu: options.gpu });
  const { worker, moduleBase, endpoint, storage: _, fetcher: __, onProgress: ___, gpu: ____, store: _____, ...rest } = options;
  const store = options.store ?? await openBrowserNeuraleseStore({ storage: options.storage as unknown as OpfsStorage | null | undefined });
  const started = await startBrowserNeuralese({ ...rest, worker, model: model!, heads: heads!, endpoint, store,
    threads: options.threads ?? chosen.threads, gpuLayers: options.gpuLayers ?? chosen.gpuLayers,
    nCtx: options.nCtx ?? manifest.contextTokens, dialect: options.dialect ?? manifest.dialect,
    moduleUrl: new URL(`${chosen.build}.mjs`, moduleBase).href });
  const adapterLoras = async (id: string) => {
    const shipped = manifest.adapters?.find(adapter => adapter.id === id);
    if (!shipped) return null;
    const file = await cachedModelFile(shipped.lora, { storage: options.storage, fetcher: options.fetcher });
    return new Uint8Array(await file.arrayBuffer());
  };
  return { ...started, build: chosen.build, reason: chosen.reason, adapterLoras };
}
