import type { SearchState } from './types.js';
import type { RunStore } from './run-store.js';
import { fingerprint } from '../adaptation/identity.js';
export const ENGINE_VERSION = 'natlang.authored-gepa/v1';
export function saveCheckpoint(store: RunStore, state: SearchState): void {
  store.write('checkpoint.json', { ...state, checkpointDigest: fingerprint(state, 'natlang.checkpoint/v1') });
}
export function readCheckpoint(store: RunStore, suiteHash: string, programHash: string, optionsHash: string): SearchState {
  const { checkpointDigest, ...state } = store.read<SearchState & { checkpointDigest: string }>('checkpoint.json');
  if (state.schema !== 'natlang.adaptation-run/v1' || state.engine !== ENGINE_VERSION) throw new Error('checkpoint belongs to a different authored engine; rebuild the run');
  if (checkpointDigest !== fingerprint(state, 'natlang.checkpoint/v1')) throw new Error('optimization checkpoint integrity mismatch; restore the last intact checkpoint');
  if (state.suiteHash !== suiteHash || state.programHash !== programHash || state.optionsHash !== optionsHash) throw new Error('optimization resume fingerprint mismatch');
  return state;
}
