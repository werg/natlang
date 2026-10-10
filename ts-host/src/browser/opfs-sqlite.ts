/**
 * SQLite databases in the origin private file system, through sqlite-wasm's OPFS SAH-pool VFS (no cross-origin
 * isolation headers needed). Sync access handles exist only in dedicated workers, so this opens a database there and
 * nowhere else; each pool's handles are exclusive, so one worker of the origin holds a pool at a time. The call store's
 * worker (call-store-worker.ts) keeps its database here; applications open their own pools beside it.
 */
import sqlite3InitModule from '@sqlite.org/sqlite-wasm';
import type { WasmDb } from '../calls/store-wasm.js';

export type OpfsSqliteOptions = {
  /** The VFS pool's name (default `natlang-NAME`): its files are a directory of the origin's private file system. */
  pool?: string;
  /** Files the pool holds before it grows (default 6: a database and its journal, with room). */
  initialCapacity?: number;
  /** Where `sqlite3.wasm` is served (default: beside the module that loads sqlite-wasm). */
  wasmUrl?: string;
};

/**
 * Open (creating it when missing) the database `NAME.sqlite` in its pool. The SAH-pool VFS has no shared memory, so
 * the journal is a rollback journal, never WAL: the database is this worker's alone.
 */
export async function openOpfsSqlite(name: string, options: OpfsSqliteOptions = {}): Promise<WasmDb> {
  // The package's types omit Emscripten's module argument; its `locateFile` names where the WebAssembly is served.
  const init = sqlite3InitModule as (module?: { locateFile(file: string, prefix: string): string }) => ReturnType<typeof sqlite3InitModule>;
  const wasmUrl = options.wasmUrl;
  const sqlite3 = await init(wasmUrl ? { locateFile: (file, prefix) => file === 'sqlite3.wasm' ? wasmUrl : prefix + file } : undefined);
  const pool = await sqlite3.installOpfsSAHPoolVfs({ name: options.pool ?? `natlang-${name}`, initialCapacity: options.initialCapacity ?? 6 });
  const db = new pool.OpfsSAHPoolDb(`/${name}.sqlite`) as unknown as WasmDb;
  db.exec({ sql: 'PRAGMA journal_mode = TRUNCATE; PRAGMA synchronous = NORMAL;' });
  return db;
}
