/**
 * The machine's call record store under Node (§3.4): `calls.sqlite` in WAL mode, so concurrent tasks and processes can
 * write, and a `blobs/` directory of SHA-256-named JSON files for values, event streams and compilation files. The rows
 * and their logic are in store-core.ts, shared with the browser store. Node only.
 */
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readFileSync, readlinkSync, renameSync, rmSync, statSync, statfsSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import type { DatabaseSync as Database } from 'node:sqlite';
import { CallStore, type StoreMedium } from './store-core.js';
import type { CallStoreSettings } from './types.js';

export { CallStore } from './store-core.js';
export type { AuditJob, CallFilter, CallSummary, FindingInput, FindingRow, HotDefinition, SavingsRow, SqlDatabase, StoreMedium } from './store-core.js';

/** This process's boot and PID namespace (`boot:namespace`), so a PID is only checked where it means something. */
const PROCESS_SCOPE = (() => {
  let boot = '', namespace = '';
  try { boot = readFileSync('/proc/sys/kernel/random/boot_id', 'utf8').trim(); } catch { /* not Linux */ }
  try { namespace = readlinkSync('/proc/self/ns/pid').replace(/\D/g, ''); } catch { /* not Linux */ }
  return `${boot}:${namespace}`;
})();

const require = createRequire(import.meta.url);
let DatabaseSync: typeof Database | undefined;
/** node:sqlite warns that it is experimental on first load; that warning is not the application's business. */
function sqlite(): typeof Database {
  if (DatabaseSync) return DatabaseSync;
  const emit = process.emitWarning;
  process.emitWarning = ((warning: string | Error, ...rest: unknown[]) => {
    if (String((warning as Error)?.message ?? warning).includes('SQLite')) return;
    return (emit as (...args: unknown[]) => void).call(process, warning, ...rest);
  }) as typeof process.emitWarning;
  try { DatabaseSync = (require('node:sqlite') as { DatabaseSync: typeof Database }).DatabaseSync; }
  finally { process.emitWarning = emit; }
  return DatabaseSync;
}

/** Where this machine's store lives: `$NATLANG_CALL_STORE`, else `~/.local/share/natlang/calls`; `off` disables it. */
export function machineStoreRoot(env: Record<string, string | undefined> = process.env): string | null {
  const configured = env.NATLANG_CALL_STORE?.trim();
  if (configured && ['off', '0', 'false', 'none'].includes(configured.toLowerCase())) return null;
  return resolve(configured || join(env.XDG_DATA_HOME || join(homedir(), '.local', 'share'), 'natlang', 'calls'));
}

/** Blobs as files under `blobs/`, settings in `config.json`, processes by PID within this boot and PID namespace. */
export class FileMedium implements StoreMedium {
  readonly process = { pid: process.pid, scope: PROCESS_SCOPE };
  constructor(readonly root: string) { mkdirSync(join(root, 'blobs'), { recursive: true }); }
  private blobPath(hash: string): string { return join(this.root, 'blobs', hash.slice(0, 2), hash.slice(2)); }
  putBlob(hash: string, text: string): void {
    const path = this.blobPath(hash);
    if (existsSync(path)) return;
    mkdirSync(dirname(path), { recursive: true });
    const temporary = `${path}.${process.pid}.${Math.random().toString(36).slice(2)}`;
    writeFileSync(temporary, text);
    renameSync(temporary, path);
  }
  blob(hash: string): string | undefined {
    try { return readFileSync(this.blobPath(hash), 'utf8'); } catch { return undefined; }
  }
  deleteBlob(hash: string): void { rmSync(this.blobPath(hash), { force: true }); }
  readConfig(): Partial<CallStoreSettings> { return JSON.parse(readFileSync(join(this.root, 'config.json'), 'utf8')); }
  writeConfig(config: Partial<CallStoreSettings>): void { writeFileSync(join(this.root, 'config.json'), JSON.stringify(config, null, 2) + '\n'); }
  freeBytes(): number { const stats = statfsSync(this.root); return Number(stats.bavail) * Number(stats.bsize); }
  databaseBytes(): number {
    let bytes = 0;
    for (const name of ['calls.sqlite', 'calls.sqlite-wal']) try { bytes += statSync(join(this.root, name)).size; } catch { /* absent */ }
    return bytes;
  }
  modeOverride(): string | undefined { return process.env.NATLANG_SPECIALIZATION; }
  /**
   * A row's process is known gone when the row is from an earlier boot, or from this PID namespace and no process has
   * its PID. A row from another namespace (a container) is not: its PID means nothing here.
   */
  gone(pid: number | null, scope: string | null): boolean {
    const [boot, namespace] = PROCESS_SCOPE.split(':');
    const [rowBoot, rowNamespace] = (scope ?? '').split(':');
    if (rowBoot && boot && rowBoot !== boot) return true;
    if (pid === null || (rowNamespace ? rowNamespace !== namespace : !!scope)) return false;
    try { process.kill(pid, 0); return false; } catch (error) { return (error as NodeJS.ErrnoException).code === 'ESRCH'; }
  }
}

const opened = new Map<string, CallStore>();
/** The store at `root` (one instance per root per process). */
export function openCallStore(root: string): CallStore {
  const absolute = resolve(root);
  let store = opened.get(absolute);
  if (store) return store;
  const medium = new FileMedium(absolute);
  const db = new (sqlite())(join(absolute, 'calls.sqlite'));
  db.exec('PRAGMA busy_timeout = 10000; PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;');
  store = new CallStore(absolute, db, medium);
  store.onClose = () => opened.delete(absolute);
  opened.set(absolute, store);
  return store;
}
CallStore.opener = openCallStore;
