/**
 * Automatic tuned-weight discovery for `natlang run`/`call` (plans/HIERARCHICAL_LORA_TUNING.md §1.1, §6): the
 * program's active release is found beside it and served on the selected backend without flags or manifests.
 * A release the backend cannot serve runs the base model, marked degraded with the reason; never as tuned execution.
 */
import { createReadStream, existsSync, readdirSync, readFileSync } from 'node:fs';
import { HttpNeuraleseStore } from '../model/neuralese-server.js';
import { decodeNativeState, decodeNativeTensorArtifact } from '../native/neuralese-store.js';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import type { ProgramDescriptor } from '../adaptation/types.js';
import { WeightBinding } from '../adaptation/weights/binding.js';
import { currentRelease } from '../adaptation/weights/release.js';
import { managedModelSha256 } from '../model/local-server.js';
import type { ResolvedModelChoice } from '../model/config.js';

export type TunedServing = {
  /** For `createNatlangRuntime({ weights })`. */
  weights?: readonly WeightBinding[];
  /** For the managed llama.cpp session: the GGUF adapters to load. */
  loraAdapters?: () => readonly { id: string; path: string }[];
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
  if (choice.kind !== 'managed-local') incompatible = `the ${choice.kind} backend does not serve tuned weights; use the managed local model or the native server`;
  else {
    const base = await managedModelSha256(environment);
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
  // The managed upstream llama-server has no native-state import: soft-state bindings run degraded there.
  const binding = new WeightBinding(manifest, { ...(program ? { program } : {}), available: new Set(loaded.map(item => item.id)), nativeState: false,
    ...(incompatible ? { incompatible } : {}) });
  return { weights: [binding], ...(loaded.length ? { loraAdapters: () => loaded } : {}),
    ...(binding.status !== 'ready' ? { notice: `natlang: tuned weights ${binding.status}: ${binding.reason}` } : {}) };
}

/**
 * Upload a release's native artifacts (`artifacts/*.nnt`: coefficients, bases) and soft states (`states/*`) to a
 * native server store. Each file must hold exactly the content its name claims. Returns the IDs now served.
 */
export async function uploadRelease(store: HttpNeuraleseStore, directory: string): Promise<Set<string>> {
  const served = new Set<string>();
  const put = async (bytes: Uint8Array, file: string) => file.endsWith('.nzs')
    ? (await store.putNativeState(decodeNativeState(bytes))).id : (await store.putNativeTensorArtifact(decodeNativeTensorArtifact(bytes))).id;
  for (const folder of ['artifacts', 'states']) {
    const path = join(directory, folder);
    if (!existsSync(path)) continue;
    // Derivation chains (tuned states, their updates and histories) need parents first; retry until no progress.
    let pending = readdirSync(path).sort(), failures: string[] = [];
    while (pending.length) {
      const next: string[] = []; failures = [];
      for (const file of pending) {
        try {
          const id = await put(new Uint8Array(readFileSync(join(path, file))), file);
          if (id !== file.replace(/\.[^.]+$/, '')) throw new Error(`release file ${folder}/${file} does not hold its named content (${id})`);
          served.add(id);
        } catch (error) {
          if (/does not hold its named content/.test(String(error))) throw error;
          next.push(file); failures.push(`${file}: ${(error as Error).message}`);
        }
      }
      if (next.length === pending.length) throw new Error(`release upload made no progress: ${failures.slice(0, 3).join('; ')}`);
      pending = next;
    }
  }
  return served;
}
