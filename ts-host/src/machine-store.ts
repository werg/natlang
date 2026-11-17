/**
 * The machine's call store as the Node platform and the CLI configure it. The environment (`NATLANG_CALL_STORE`,
 * `XDG_DATA_HOME`, `NATLANG_SPECIALIZATION`) and the home directory are read here and handed to the store as options;
 * the stores themselves (calls/) read neither.
 */
import { AsyncLocalStorage } from 'node:async_hooks';
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

/**
 * Hermetic episodes. A collector episode (training, held evaluation, replay, teacher generation) runs with a fresh call
 * store of its own: what earlier episodes or runs stored (compiled cases, tier outputs, iterateOn statistics,
 * refinement verdicts) can never change what this one does, while its own records stay readable within it (the
 * item-result binding reads its host captures). Every runtime first used inside `runInEpisodeCallStore`, worker threads
 * started with `episodeCallStoreRoot()` in their data included, records to it instead of the machine's store. Its
 * settings switch specialization off, so nothing compiled within the episode serves its later calls either.
 */
const episodeStore = new AsyncLocalStorage<string>();
/** The episode store's root when the caller runs inside one. */
export function episodeCallStoreRoot(): string | undefined { return episodeStore.getStore(); }
/** Run `body` with the store at `root` as the default store of every runtime it creates. */
export function runInEpisodeCallStore<T>(root: string, body: () => T): T { return episodeStore.run(root, body); }
/** A worker thread joins its parent's episode store for the rest of its life. */
export function enterEpisodeCallStore(root: string | undefined): void { if (root) episodeStore.enterWith(root); }
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
