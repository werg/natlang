/**
 * A tuned release in the page (plans/HIERARCHICAL_LORA_TUNING.md §7): the manifest and its per-node GGUF exports are
 * fetched from a release directory URL and verified by content hash. The adapters go to `loadBrowserLocalModel`
 * (`loraAdapters`), the binding to `createNatlangRuntime({ weights })`; each call then runs its own lexical stack.
 * Adapter files go through the model-file OPFS cache keyed by their SHA-256, verified on every read, so a reload or
 * another release sharing a residual reads them locally.
 */
import { parseManifest, type WeightManifest } from '../adaptation/weights/manifest.js';
import { WeightBinding, type WeightBindingOptions } from '../adaptation/weights/binding.js';
import { hexDigest } from '../native/hash.js';
import { cachedModelFile, type ModelFileStorage } from './model-files.js';

export type BrowserRelease = { readonly manifest: WeightManifest; readonly adapters: readonly { readonly id: string; readonly blob: Blob }[];
  readonly weights: readonly WeightBinding[] };

/**
 * Fetch `natlang.adapters.json` and every GGUF export under `releaseUrl` (a release directory, e.g. served from
 * `.natlang/tune/releases/<digest>/`). `baseGgufSha256` is the sha256 of the base GGUF the page loads: a release
 * exported for another base file runs untuned as a whole. Residuals without a GGUF export are unavailable and their
 * dependents fall back coherently.
 */
export async function loadBrowserRelease(releaseUrl: string | URL, options: Omit<WeightBindingOptions, 'available' | 'incompatible'> & {
  baseGgufSha256?: string; fetch?: typeof fetch;
  /** The OPFS adapter cache (default `navigator.storage` when available); `null` fetches every time. */
  storage?: ModelFileStorage | null } = {}): Promise<BrowserRelease> {
  const { baseGgufSha256, fetch: fetchFile = globalThis.fetch, storage, ...binding } = options;
  const base = new URL(String(releaseUrl).endsWith('/') ? String(releaseUrl) : `${releaseUrl}/`, globalThis.location?.href);
  const get = async (path: string) => {
    const response = await fetchFile(new URL(path, base));
    if (!response.ok) throw new Error(`tuned release file unavailable: ${new URL(path, base)} (${response.status})`);
    return response;
  };
  const manifest = parseManifest(await (await get('natlang.adapters.json')).text());
  // Program residuals and published family versions alike (a family's GGUF serves its gated level).
  const exported = [...manifest.artifacts, ...manifest.families].filter(item => item.exports?.gguf);
  const wrongBase = baseGgufSha256 ? exported.find(item => item.exports!.gguf!.baseGgufSha256 !== baseGgufSha256) : undefined;
  if (wrongBase) return { manifest, adapters: [], weights: [new WeightBinding(manifest, { ...binding,
    incompatible: `GGUF exports bind base ${wrongBase.exports!.gguf!.baseGgufSha256.slice(0, 12)}, the page serves ${baseGgufSha256!.slice(0, 12)}` })] };
  const adapters = await Promise.all(exported.map(async item => {
    const gguf = item.exports!.gguf!;
    const url = new URL(gguf.file, base).href;
    const blob = await cachedModelFile({ url, sha256: gguf.sha256 }, { storage, fetcher: (input => get(String(input))) as typeof fetch,
      verify: async file => hexDigest(new Uint8Array(await file.arrayBuffer())) === gguf.sha256 })
      .catch(error => { throw new Error(`tuned release file ${gguf.file}: ${(error as Error).message}`); });
    return { id: item.artifact, blob };
  }));
  return { manifest, adapters, weights: [new WeightBinding(manifest, { ...binding, available: new Set(adapters.map(adapter => adapter.id)) })] };
}
