/**
 * A tuned release in the page (plans/HIERARCHICAL_LORA_TUNING.md §7): the manifest and its per-node GGUF exports are
 * fetched from a release directory URL and verified by content hash. The adapters go to `loadBrowserLocalModel`
 * (`loraAdapters`), the binding to `createNatlangRuntime({ weights })`; each call then runs its own lexical stack.
 */
import { parseManifest, type WeightManifest } from '../adaptation/weights/manifest.js';
import { WeightBinding, type WeightBindingOptions } from '../adaptation/weights/binding.js';
import { hexDigest } from '../native/hash.js';

export type BrowserRelease = { readonly manifest: WeightManifest; readonly adapters: readonly { readonly id: string; readonly blob: Blob }[];
  readonly weights: readonly WeightBinding[] };

/**
 * Fetch `natlang.adapters.json` and every GGUF export under `releaseUrl` (a release directory, e.g. served from
 * `.natlang/tune/releases/<digest>/`). `baseGgufSha256` is the sha256 of the base GGUF the page loads: a release
 * exported for another base file runs untuned as a whole. Residuals without a GGUF export are unavailable and their
 * dependents fall back coherently.
 */
export async function loadBrowserRelease(releaseUrl: string | URL, options: Omit<WeightBindingOptions, 'available' | 'incompatible'> & {
  baseGgufSha256?: string; fetch?: typeof fetch } = {}): Promise<BrowserRelease> {
  const { baseGgufSha256, fetch: fetchFile = globalThis.fetch, ...binding } = options;
  const base = new URL(String(releaseUrl).endsWith('/') ? String(releaseUrl) : `${releaseUrl}/`, globalThis.location?.href);
  const get = async (path: string) => {
    const response = await fetchFile(new URL(path, base));
    if (!response.ok) throw new Error(`tuned release file unavailable: ${new URL(path, base)} (${response.status})`);
    return response;
  };
  const manifest = parseManifest(await (await get('natlang.adapters.json')).text());
  const exported = manifest.artifacts.filter(item => item.exports?.gguf);
  const wrongBase = baseGgufSha256 ? exported.find(item => item.exports!.gguf!.baseGgufSha256 !== baseGgufSha256) : undefined;
  if (wrongBase) return { manifest, adapters: [], weights: [new WeightBinding(manifest, { ...binding,
    incompatible: `GGUF exports bind base ${wrongBase.exports!.gguf!.baseGgufSha256.slice(0, 12)}, the page serves ${baseGgufSha256!.slice(0, 12)}` })] };
  const adapters = await Promise.all(exported.map(async item => {
    const gguf = item.exports!.gguf!;
    const bytes = new Uint8Array(await (await get(gguf.file)).arrayBuffer());
    if (hexDigest(bytes) !== gguf.sha256) throw new Error(`tuned release file ${gguf.file} does not match its sha256`);
    return { id: item.artifact, blob: new Blob([bytes]) };
  }));
  return { manifest, adapters, weights: [new WeightBinding(manifest, { ...binding, available: new Set(adapters.map(adapter => adapter.id)) })] };
}
