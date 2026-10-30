/**
 * The machine's call store as the Node platform and the CLI configure it. The environment (`NATLANG_CALL_STORE`,
 * `XDG_DATA_HOME`, `NATLANG_SPECIALIZATION`) and the home directory are read here and handed to the store as options;
 * the stores themselves (calls/) read neither.
 */
import { homedir } from 'node:os';
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
