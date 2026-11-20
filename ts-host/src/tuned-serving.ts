/**
 * Automatic tuned-weight discovery for `natlang run`/`call` (plans/HIERARCHICAL_LORA_TUNING.md §1.1, §6): the
 * program's active release is found beside it and served on the selected backend without flags or manifests.
 * A release the backend cannot serve runs the base model, marked degraded with the reason; never as tuned execution.
 */
import { createReadStream, existsSync, readFileSync } from 'node:fs';
import { HttpNeuraleseStore, restoreNativeArtifacts } from './model/neuralese-server.js';
import { decodeNativeState, decodeNativeTensorArtifact } from './native/neuralese-store.js';
import { createHash } from 'node:crypto';
import { join } from 'node:path';
import type { ProgramDescriptor } from './adaptation/types.js';
import { WeightBinding } from './adaptation/weights/binding.js';
import { softStateFile } from './adaptation/weights/manifest.js';
import { currentRelease } from './adaptation/weights/release.js';
import { managedModelSha256, profileEnvironment, supportsNativeStates } from './model/local-server.js';
import { discoverLlamaRuntime } from './model/llama-runtime.js';
import type { ResolvedModelChoice } from './model/config.js';
import { nativeReleaseObjectClosure, verifyNativeInferenceSnapshotReceipt, type NativeInferenceMaterializationReceipt } from './native/release-artifact-closure.js';
import { materializeExistingInferenceRelease } from './adaptation/weights/inference-release.js';

export type TunedServing = {
  /** For `createNatlangRuntime({ weights })`. */
  weights?: readonly WeightBinding[];
  /** For the managed llama.cpp session: the GGUF adapters to load. */
  loraAdapters?: () => readonly { id: string; path: string }[];
  /** For the managed llama.cpp fork: soft states as native state spans to load (`--native-state`). */
  nativeStates?: () => readonly { id: string; path: string }[];
  /** One author-facing line when the program is tuned but not served as tuned, else undefined. */
  notice?: string;
};

async function fileSha256(path: string): Promise<string> {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
}

/**
 * Whether the selected managed llama.cpp profile serves this release as tuned: every adapter has a GGUF export bound to
 * exactly the profile's base GGUF, and soft states (their store files) need the fork's `--native-state`.
 */
type Exported = { exports?: { gguf?: { baseGgufSha256?: string } } };
export async function llamaServesRelease(manifest: { artifacts: readonly Exported[]; families?: readonly Exported[];
  softStates: readonly unknown[] }, choice: ResolvedModelChoice,
  environment: NodeJS.ProcessEnv = process.env): Promise<boolean> {
  // A fit of soft states alone (every rank 0, CSPD's frozen backbone) has no adapter to export: its states are the release.
  if (choice.kind !== 'managed-local' || (!manifest.artifacts.length && !manifest.softStates.length)) return false;
  environment = profileEnvironment(choice, environment);
  let base: string;
  try { base = await managedModelSha256(environment, choice.release); } catch { return false; }
  const bound = (item: { exports?: { gguf?: { baseGgufSha256?: string } } }) => item.exports?.gguf?.baseGgufSha256 === base;
  // Families without members in this program are not exported; every exported one must bind the same base.
  if (!manifest.artifacts.every(bound)
    || !(manifest.families ?? []).filter(item => item.exports?.gguf).every(bound)) return false;
  const executable = discoverLlamaRuntime(environment).selected?.path;
  return !manifest.softStates.length || (!!executable && supportsNativeStates(executable));
}

export async function tunedServing(root: string, program: ProgramDescriptor | undefined, choice: ResolvedModelChoice,
  environment: NodeJS.ProcessEnv = process.env): Promise<TunedServing> {
  environment = profileEnvironment(choice, environment);
  const release = currentRelease(root);
  if (!release) return {};
  const compact = materializeExistingInferenceRelease(root, release.manifest, release.directory);
  const { manifest, directory } = compact;
  let incompatible: string | undefined;
  const loaded: { id: string; path: string }[] = [];
  if (choice.kind === 'external' && choice.neuraleseServer) {
    // The native reference server applies the exact native artifacts and imports soft states.
    const store = new HttpNeuraleseStore(choice.endpoint, { ...choice.headers,
      ...(environment[choice.apiKeyEnv] ? { authorization: 'Bearer ' + environment[choice.apiKeyEnv] } : {}) });
    const uploaded = await uploadRelease(store, directory, [...manifest.artifacts.map(item => item.artifact),
      ...manifest.families.map(item => item.artifact), ...manifest.softStates.map(item => item.state)], manifest.receipts.inferenceStateMaterializations);
    const binding = new WeightBinding(manifest, { ...(program ? { program } : {}), available: uploaded, nativeState: true });
    return { weights: [binding], ...(binding.status !== 'ready' ? { notice: `natlang: tuned weights ${binding.status}: ${binding.reason}` } : {}) };
  }
  let base: string | undefined;
  if (choice.kind !== 'managed-local') incompatible = `the ${choice.kind} backend does not serve tuned weights; use the managed local model or the native server`;
  else {
    base = await managedModelSha256(environment, choice.release);
    const exported = manifest.artifacts.filter(item => item.exports?.gguf);
    const families = manifest.families.map(family => manifest.artifacts.find(item => item.artifact === family.artifact)).filter(Boolean);
    if (!exported.length && !families.length && (manifest.artifacts.length || !manifest.softStates.length))
      incompatible = 'this release has no llama.cpp export; rerun natlang tune for the managed backend';
    else if (exported.some(item => item.exports!.gguf!.baseGgufSha256 !== base))
      incompatible = 'the tuned adapters were exported for another base model file; rerun natlang tune';
    else for (const item of exported) {
      const path = join(directory, item.exports!.gguf!.file);
      if (await fileSha256(path).catch(() => '') !== item.exports!.gguf!.sha256)
        throw new Error(`tuned adapter ${item.exports!.gguf!.file} is missing or differs from its release receipt; rerun natlang tune`);
      loaded.push({ id: item.artifact, path });
    }
  }
  // Soft states: their store files (`states/<id>.nzs`) as exact native state spans on the llama.cpp fork
  // (`--native-state`, which verifies each file's content ID and identity at load); the upstream llama-server has no
  // native-state import, so there soft-state bindings run degraded.
  const states: { id: string; path: string }[] = [];
  const executable = choice.kind === 'managed-local' ? discoverLlamaRuntime(environment).selected?.path : undefined;
  const nativeState = !incompatible && manifest.softStates.length > 0 && !!executable && supportsNativeStates(executable);
  if (nativeState) for (const item of manifest.softStates) {
    const path = join(directory, softStateFile(item));
    if (!existsSync(path)) throw new Error(`tuned soft state ${softStateFile(item)} is missing from its release; rerun natlang tune`);
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
export async function uploadRelease(store: HttpNeuraleseStore, directory: string, roots: Iterable<string>, receipts?: unknown): Promise<Set<string>> {
  const files = nativeReleaseObjectClosure([directory], roots);
  const read = (id: string, state: boolean) => {
    const path = files.get(id);
    if (!path) return undefined;
    const bytes = new Uint8Array(readFileSync(path));
    const artifact = state ? decodeNativeState(bytes) : decodeNativeTensorArtifact(bytes);
    if (artifact.id !== id) throw new Error(`release file ${path} does not hold its named content (${artifact.id})`);
    return artifact;
  };
  if (receipts !== undefined) {
    if (!Array.isArray(receipts)) throw new Error('release inference-state materialization receipts must be an array');
    for (const value of receipts) {
      if (!value || typeof value !== 'object' || typeof (value as NativeInferenceMaterializationReceipt).inferenceState !== 'string')
        throw new Error('release inference-state materialization receipt is malformed');
      const receipt = value as NativeInferenceMaterializationReceipt;
      const state = read(receipt.inferenceState, true);
      if (!state) throw new Error(`release materialization snapshot ${receipt.inferenceState} is absent`);
      verifyNativeInferenceSnapshotReceipt(state as ReturnType<typeof decodeNativeState>, receipt);
    }
  }
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

