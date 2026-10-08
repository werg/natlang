/**
 * The call store over SQLite compiled to WebAssembly (@sqlite.org/sqlite-wasm): its `oo1.DB` as the store's
 * `SqlDatabase`, and a medium that keeps blobs and settings in the same database. In the browser the database lives in
 * OPFS (browser/call-store-worker.ts); tests use the in-memory database of the package's Node build.
 */
import { byteLength, CallStore, type SqlDatabase, type SqlValue, type StoreMedium } from './store-core.js';
import type { CallStoreSettings } from './types.js';

/** The part of sqlite-wasm's `oo1.DB` used here. */
export type WasmDb = {
  exec(options: { sql: string; bind?: SqlValue[]; rowMode?: 'object'; returnValue?: 'resultRows' }): unknown;
  close(): void;
};

/** `oo1.DB` behind the store's synchronous statement interface. */
export function wasmDatabase(db: WasmDb): SqlDatabase {
  const bind = (params: SqlValue[]) => params.map(value => value === undefined ? null : value);
  return {
    exec: sql => { db.exec({ sql }); },
    prepare: sql => ({
      run: (...params) => { db.exec({ sql, bind: bind(params) }); },
      get: (...params) => (db.exec({ sql, bind: bind(params), rowMode: 'object', returnValue: 'resultRows' }) as unknown[])[0],
      all: (...params) => db.exec({ sql, bind: bind(params), rowMode: 'object', returnValue: 'resultRows' }) as unknown[],
    }),
    close: () => db.close(),
  };
}

const MEDIUM_SCHEMA = `
CREATE TABLE IF NOT EXISTS blob_texts (hash TEXT PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS store_config (key TEXT PRIMARY KEY, value TEXT NOT NULL);`;

/**
 * Blobs and settings as rows. One session holds the database at a time (OPFS access handles are exclusive), so a row
 * left running by any other session is from a session that ended.
 */
export class DatabaseMedium implements StoreMedium {
  readonly process: { pid: number | null; scope: string };
  /** Free bytes as last estimated (navigator.storage.estimate is asynchronous; the worker refreshes this). */
  free = Infinity;
  constructor(private readonly db: SqlDatabase, scope = `session:${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`,
    private readonly mode?: () => string | undefined) {
    db.exec(MEDIUM_SCHEMA);
    this.process = { pid: null, scope };
  }
  putBlob(hash: string, text: string): void { this.db.prepare('INSERT OR IGNORE INTO blob_texts (hash, text) VALUES (?, ?)').run(hash, text); }
  blob(hash: string): string | undefined {
    return (this.db.prepare('SELECT text FROM blob_texts WHERE hash = ?').get(hash) as { text: string } | undefined)?.text;
  }
  deleteBlob(hash: string): void { this.db.prepare('DELETE FROM blob_texts WHERE hash = ?').run(hash); }
  readConfig(): Partial<CallStoreSettings> {
    const row = this.db.prepare("SELECT value FROM store_config WHERE key = 'settings'").get() as { value: string } | undefined;
    return row ? JSON.parse(row.value) : {};
  }
  writeConfig(config: Partial<CallStoreSettings>): void {
    this.db.prepare(`INSERT INTO store_config (key, value) VALUES ('settings', ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value`)
      .run(JSON.stringify(config));
  }
  freeBytes(): number { return this.free; }
  databaseBytes(): number {
    const pages = this.db.prepare('PRAGMA page_count').get() as { page_count: number };
    const size = this.db.prepare('PRAGMA page_size').get() as { page_size: number };
    return Number(pages.page_count) * Number(size.page_size);
  }
  modeOverride(): string | undefined { return this.mode?.(); }
  gone(_pid: number | null, scope: string | null): boolean { return scope !== this.process.scope; }
}

/** A call store in a sqlite-wasm database. */
export function wasmCallStore(root: string, db: WasmDb, options: { scope?: string; mode?: () => string | undefined } = {}):
  { store: CallStore; medium: DatabaseMedium } {
  const sql = wasmDatabase(db);
  const medium = new DatabaseMedium(sql, options.scope, options.mode);
  return { store: new CallStore(root, sql, medium), medium };
}

export { byteLength };
