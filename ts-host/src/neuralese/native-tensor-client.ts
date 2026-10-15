/** Archive and restore identity-bound native gradient/parameter artifacts. */
import { HttpNeuraleseStore } from '../model/neuralese-server.js';
import type { NativeTensorArtifact, NeuraleseStore } from '../native/neuralese-store.js';

function fail(code: string, message: string): never { throw Object.assign(new Error(`${code}: ${message}`), { code }); }

export async function archiveNativeTensorArtifact(endpoint: string, store: NeuraleseStore,
    id: string, headers?: Record<string, string>): Promise<NativeTensorArtifact> {
  if (!store.putNativeTensorArtifact || !store.getNativeTensorArtifact)
    return fail('neuralese-native-archive-unavailable', 'configure a store that preserves named native tensors');
  const found = await store.getNativeTensorArtifact(id);
  if (found) return found;
  const remote = new HttpNeuraleseStore(endpoint, headers), artifact = await remote.getNativeTensorArtifact(id);
  if (!artifact) return fail('neuralese-unknown-native-artifact', `native tensor artifact ${id} is unavailable`);
  const { id: _, ...input } = artifact, saved = await store.putNativeTensorArtifact(input);
  if (saved.id !== id) return fail('neuralese-block-integrity', `native tensor artifact ${id} changed during archival`);
  return saved;
}

export async function restoreNativeTensorArtifacts(endpoint: string, store: NeuraleseStore,
    ids: Iterable<string>, headers?: Record<string, string>): Promise<string[]> {
  if (!store.getNativeTensorArtifact) return fail('neuralese-native-archive-unavailable', 'configure a named tensor archive');
  const remote = new HttpNeuraleseStore(endpoint, headers), missing: string[] = [];
  for (const id of new Set(ids)) {
    if (await remote.hasNativeTensorArtifact(id)) continue;
    const artifact = await store.getNativeTensorArtifact(id);
    if (!artifact) { missing.push(id); continue; }
    const { id: _, ...input } = artifact, restored = await remote.putNativeTensorArtifact(input);
    if (restored.id !== id) return fail('neuralese-block-integrity', `native tensor artifact ${id} changed during restoration`);
  }
  return missing;
}
