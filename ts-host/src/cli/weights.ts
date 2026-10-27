/**
 * Automatic tuned-weight discovery for `natlang run`/`call` (plans/HIERARCHICAL_LORA_TUNING.md §1.1, §6): the
 * program's active release is found beside it and served on the selected backend without flags or manifests.
 * A release the backend cannot serve runs the base model, marked degraded with the reason; never as tuned execution.
 */
import { createReadStream, existsSync, readdirSync, readFileSync } from 'node:fs';
import { HttpNeuraleseStore, restoreNativeArtifacts } from '../model/neuralese-server.js';
import { decodeNativeState, decodeNativeTensorArtifact } from '../native/neuralese-store.js';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { WeightBinding } from '../adaptation/weights/binding.js';
import { currentRelease } from '../adaptation/weights/release.js';
import { managedModelSha256, supportsNativeStates } from '../model/local-server.js';
import { discoverLlamaRuntime } from '../model/llama-runtime.js';
import type { ResolvedModelChoice } from '../model/config.js';

export type TunedServing = {
  /** For `createNatlangRuntime({ weights })`. */
  weights?: readonly WeightBinding[];
  /** For the managed llama.cpp session: the GGUF adapters to load. */
  loraAdapters?: () => readonly { id: string; path: string }[];
  /** For the managed llama.cpp fork: soft states as native state spans to load (`--native-state`). */
  nativeStates?: () => readonly { id: string; path: string }[];
  /** Whole-stack LoRA names a vLLM server loaded, for its session's request routing. */
  vllmStacks?: readonly string[];
  /** One author-facing line when the program is tuned but not served as tuned, else undefined. */
  notice?: string;
};

async function fileSha256(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

export async function tunedServing(root: string, program: ProgramDescriptor | undefined, choice: ResolvedModelChoice,
  environment: NodeJS.ProcessEnv = process.env): Promise<TunedServing> {
  const release = currentRelease(root);
  if (!release) return {};
  const { manifest, directory } = release;
  let incompatible: string | undefined;
  const loaded: { id: string; path: string }[] = [];
  if (choice.kind === 'external' && choice.neuraleseServer) {
    // The native reference server applies the exact native artifacts and imports soft states.
    const store = new HttpNeuraleseStore(choice.endpoint, { ...choice.headers,
      ...(environment[choice.apiKeyEnv] ? { authorization: 'Bearer ' + environment[choice.apiKeyEnv] } : {}) });
    const uploaded = await uploadRelease(store, directory);
    const binding = new WeightBinding(manifest, { ...(program ? { program } : {}), available: uploaded, nativeState: true });
    return { weights: [binding], ...(binding.status !== 'ready' ? { notice: `natlang: tuned weights ${binding.status}: ${binding.reason}` } : {}) };
  }
  if (choice.kind === 'external' && choice.vllm) {
    const loaded = await loadVllmStacks(choice.endpoint, directory, { ...choice.headers,
      ...(environment[choice.apiKeyEnv] ? { authorization: 'Bearer ' + environment[choice.apiKeyEnv] } : {}) });
    const binding = new WeightBinding(manifest, { ...(program ? { program } : {}), available: loaded.available, nativeState: false });
    return { weights: [binding], vllmStacks: loaded.names,
      ...(binding.status !== 'ready' ? { notice: `natlang: tuned weights ${binding.status}: ${binding.reason}` } : {}) };
  }
  let base: string | undefined;
  if (choice.kind !== 'managed-local') incompatible = `the ${choice.kind} backend does not serve tuned weights; use the managed local model or the native server`;
  else {
    base = await managedModelSha256(environment);
    const exported = manifest.artifacts.filter(item => item.exports?.gguf);
    const families = manifest.families.map(family => manifest.artifacts.find(item => item.artifact === family.artifact)).filter(Boolean);
    if (!exported.length && !families.length) incompatible = 'this release has no llama.cpp export; rerun natlang tune for the managed backend';
    else if (exported.some(item => item.exports!.gguf!.baseGgufSha256 !== base))
      incompatible = 'the tuned adapters were exported for another base model file; rerun natlang tune';
    else for (const item of exported) {
      const path = join(directory, item.exports!.gguf!.file);
      if (await fileSha256(path).catch(() => '') !== item.exports!.gguf!.sha256)
        throw new Error(`tuned adapter ${item.exports!.gguf!.file} is missing or differs from its release receipt; rerun natlang tune`);
      loaded.push({ id: item.artifact, path });
    }
  }
  // Soft states: exact native state spans on the llama.cpp fork (`--native-state`); the upstream llama-server has no
  // native-state import, so there soft-state bindings run degraded.
  const states: { id: string; path: string }[] = [];
  const executable = choice.kind === 'managed-local' ? discoverLlamaRuntime(environment).selected?.path : undefined;
  const nativeState = !incompatible && manifest.softStates.length > 0 && !!executable && supportsNativeStates(executable)
    && manifest.softStates.every(item => item.exports?.gguf?.baseGgufSha256 === base);
  if (nativeState) for (const item of manifest.softStates) {
    const path = join(directory, item.exports!.gguf!.file);
    if (await fileSha256(path).catch(() => '') !== item.exports!.gguf!.sha256)
      throw new Error(`tuned soft state ${item.exports!.gguf!.file} is missing or differs from its release receipt; rerun natlang tune`);
    states.push({ id: item.state, path });
  }
  const binding = new WeightBinding(manifest, { ...(program ? { program } : {}), available: new Set([...loaded, ...states].map(item => item.id)),
    nativeState, ...(incompatible ? { incompatible } : {}) });
  return { weights: [binding], ...(loaded.length ? { loraAdapters: () => loaded } : {}), ...(states.length ? { nativeStates: () => states } : {}),
    ...(binding.status !== 'ready' ? { notice: `natlang: tuned weights ${binding.status}: ${binding.reason}` } : {}) };
}

/**
 * Upload a release's native artifacts (`artifacts/*.nnt`: coefficients, bases) and soft states (`states/*`) to a
 * native server store. Each file must hold exactly the content its name claims. Returns the IDs now served.
 */
export async function uploadRelease(store: HttpNeuraleseStore, directory: string): Promise<Set<string>> {
  const files = new Map<string, string>();
  for (const [folder, extension] of [['artifacts', '.nnt'], ['states', '.nzs']] as const) {
    const path = join(directory, folder);
    if (existsSync(path)) for (const file of readdirSync(path).sort()) {
      if (!file.endsWith(extension)) throw new Error(`unexpected native release file ${folder}/${file}`);
      files.set(file.slice(0, -extension.length), join(path, file));
    }
  }
  const read = (id: string, state: boolean) => {
    const path = files.get(id);
    if (!path) return undefined;
    const bytes = new Uint8Array(readFileSync(path));
    const artifact = state ? decodeNativeState(bytes) : decodeNativeTensorArtifact(bytes);
    if (artifact.id !== id) throw new Error(`release file ${path} does not hold its named content (${artifact.id})`);
    return artifact;
  };
  // Use the same dependency traversal as ordinary native execution: parents,
  // gradients and numeric histories arrive before their derived state. Existing
  // exact objects are reused rather than repeatedly uploading whole histories.
  const missing = await restoreNativeArtifacts(store, {
    getNativeState: async id => read(id, true) as ReturnType<typeof decodeNativeState> | undefined,
    getNativeTensorArtifact: async id => read(id, false) as ReturnType<typeof decodeNativeTensorArtifact> | undefined,
  }, files.keys());
  if (missing.length) throw new Error(`release is missing native dependencies: ${missing.slice(0, 3).join(', ')}`);
  return new Set(files.keys());
}

/**
 * Load a release's whole-stack exports (`vllm/stack-exports.json`, written by tune/stack_export.py) into a vLLM server
 * through its runtime LoRA API. The export paths must be visible to the server at the same absolute path. A stack that
 * fails to load withdraws all its artifacts from `available`, so its bindings fall back coherently instead of failing.
 */
export async function loadVllmStacks(endpoint: string, directory: string, headers: Record<string, string> = {}):
  Promise<{ names: string[]; available: Set<string> }> {
  const index = join(directory, 'vllm', 'stack-exports.json');
  if (!existsSync(index)) return { names: [], available: new Set() };
  const stacks = JSON.parse(readFileSync(index, 'utf8')) as { identity: string; directory: string; artifacts?: string[] }[];
  const root = endpoint.replace(/\/v1\/?$/, '').replace(/\/$/, '');
  const names: string[] = [], available = new Set<string>(), withdrawn = new Set<string>();
  for (const stack of stacks) {
    const response = await fetch(root + '/v1/load_lora_adapter', { method: 'POST', headers: { 'content-type': 'application/json', ...headers },
      body: JSON.stringify({ lora_name: stack.identity, lora_path: join(directory, 'vllm', stack.directory) }) }).catch(() => undefined);
    // An adapter that is already loaded under the same name is the same content (names hash the stack entries).
    const ok = !!response && (response.ok || /already/.test(await response.text().catch(() => '')));
    if (ok) { names.push(stack.identity); for (const id of stack.artifacts ?? []) available.add(id); }
    else for (const id of stack.artifacts ?? []) withdrawn.add(id);
  }
  for (const id of withdrawn) available.delete(id);
  return { names, available };
}
