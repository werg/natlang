/**
 * The machine's call store as the Node platform and the CLI configure it. The environment (`NATLANG_CALL_STORE`,
 * `XDG_DATA_HOME`, `NATLANG_SPECIALIZATION`) and the home directory are read here and handed to the store as options;
 * the stores themselves (calls/) read neither.
 */
import { mkdtempSync, rmSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { CallStore } from './calls/store.js';

type Environment = Record<string, string | undefined>;

/** Where this machine's store lives: `$NATLANG_CALL_STORE`, else `~/.local/share/natlang/calls`; `off` disables it. */
export function machineStoreRoot(env: Environment = process.env, home: string = homedir()): string | null {
  const configured = env.NATLANG_CALL_STORE?.trim();
  if (configured && ['off', '0', 'false', 'none'].includes(configured.toLowerCase())) return null;
  return resolve(configured || join(env.XDG_DATA_HOME || join(home, '.local', 'share'), 'natlang', 'calls'));
}

/** The store at `root` with the environment's `NATLANG_SPECIALIZATION` as its mode override (read live, so it can change). */
export function openMachineStore(root: string, env: Environment = process.env): CallStore {
  return CallStore.open(root, { mode: () => env.NATLANG_SPECIALIZATION });
}

// Hermetic episodes: a collector episode (training, held evaluation, replay, teacher generation) runs with a fresh call
// store of its own, created here; runtime/episode-call-store.ts holds which episode store the caller runs in.

/**
 * `NATLANG_EPISODE_CALL_STORE=machine` (debugging only): collector episodes record to the machine's store as before,
 * with specialization still off.
 */
export function episodeCallStoreShared(env: Environment = process.env): boolean {
  return env.NATLANG_EPISODE_CALL_STORE?.trim().toLowerCase() === 'machine';
}
/** A fresh, empty episode store in a temporary directory, removed by `dispose`. */
export function createEpisodeCallStore(): { root: string; store: CallStore; dispose(): void } {
  const root = mkdtempSync(join(tmpdir(), 'natlang-episode-calls-'));
  const store = CallStore.open(root);
  store.writeSettings({ specialization: 'off', minFreeBytes: 0 });
  return { root, store, dispose() { try { store.close(); } finally { rmSync(root, { recursive: true, force: true }); } } };
}
