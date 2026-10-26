/**
 * Model files in the browser: downloaded once into the origin-private file system (OPFS) and opened from there on
 * later loads (a tuned release's GGUF files, tuned-release.ts). A cached file is a disk-backed `File`, read in slices
 * without holding the whole model in memory. Files are keyed by their SHA-256 when the manifest gives
 * one (a new release is a new file), else by URL. `createWritable` writes to a swap file that replaces the target only
 * when closed, so an interrupted download leaves no truncated file behind. Without OPFS the file is fetched each time.
 */
export type ModelFileRef = { url: string; bytes?: number; sha256?: string };

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
 * A cached file whose size differs from the manifest's `bytes`, or that fails `verify`, is downloaded again; a download
 * that fails `verify` is removed and refused.
 */
export async function cachedModelFile(file: ModelFileRef, options: { storage?: ModelFileStorage | null; fetcher?: typeof fetch;
  onProgress?: (received: number, total: number | null) => void;
  /** Content check of the whole file (e.g. its SHA-256 for small files). */
  verify?: (file: Blob) => Promise<boolean> } = {}): Promise<Blob> {
  const verified = async (blob: Blob) => !options.verify || await options.verify(blob);
  const fetcher = options.fetcher ?? globalThis.fetch;
  const storage = options.storage === undefined ? (globalThis.navigator?.storage as unknown as ModelFileStorage | undefined) : options.storage;
  const download = async (): Promise<Response> => {
    const response = await fetcher(file.url);
    if (!response.ok || !response.body) throw new Error(`model file ${file.url}: HTTP ${response.status}`);
    return response;
  };
  if (!storage?.getDirectory) {
    const blob = await (await download()).blob();
    if (!await verified(blob)) throw new Error(`model file ${file.url}: content does not match its manifest`);
    return blob;
  }
  const directory = await (await storage.getDirectory()).getDirectoryHandle(DIRECTORY, { create: true });
  const name = await cacheKey(file);
  try {
    const cached = await (await directory.getFileHandle(name)).getFile();
    if ((file.bytes === undefined ? cached.size > 0 : cached.size === file.bytes) && await verified(cached)) return cached;
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
  if (!await verified(stored)) {
    await directory.removeEntry(name).catch(() => undefined);
    throw new Error(`model file ${file.url}: content does not match its manifest`);
  }
  return stored;
}

/** Remove these files from the OPFS cache (e.g. adapters of a release that was replaced). Missing files are ignored. */
export async function removeCachedModelFiles(files: readonly ModelFileRef[], options: { storage?: ModelFileStorage | null } = {}): Promise<void> {
  const storage = options.storage === undefined ? (globalThis.navigator?.storage as unknown as ModelFileStorage | undefined) : options.storage;
  if (!storage?.getDirectory) return;
  const directory = await (await storage.getDirectory()).getDirectoryHandle(DIRECTORY, { create: true });
  for (const file of files) await directory.removeEntry(await cacheKey(file)).catch(() => undefined);
}
