/** Archive and restore identity-bound native gradient/parameter artifacts. */
import { HttpNeuraleseStore, restoreBlocks } from '../model/neuralese-server.js';
import { archiveNativeTensorFrom, type NativeTensorArtifact, type NeuraleseStore } from '../native/neuralese-store.js';

function fail(code: string, message: string): never { throw Object.assign(new Error(`${code}: ${message}`), { code }); }

export async function archiveNativeTensorArtifact(endpoint: string, store: NeuraleseStore,
    id: string, headers?: Record<string, string>): Promise<NativeTensorArtifact> {
  return archiveNativeTensorFrom(new HttpNeuraleseStore(endpoint, headers), store, id);
}

export async function restoreNativeTensorArtifacts(endpoint: string, store: NeuraleseStore,
    ids: Iterable<string>, headers?: Record<string, string>): Promise<string[]> {
  if (!store.getNativeTensorArtifact) return fail('neuralese-native-archive-unavailable', 'configure a named tensor archive');
  return restoreBlocks(new HttpNeuraleseStore(endpoint, headers), store, ids);
}
