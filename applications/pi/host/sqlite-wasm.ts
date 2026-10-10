/**
 * pi-durable's SQLite storage over SQLite compiled to WebAssembly (@sqlite.org/sqlite-wasm): its `oo1.DB` as the
 * storage's `SqliteDatabase`, beside pi-durable's own Node and Durable Object adapters (storage/sqlite/node.ts,
 * cloudflare.ts). In the browser the database lives in the origin private file system (ts-host's `openOpfsSqlite`, in a
 * dedicated worker); anywhere, an in-memory `oo1.DB` works too. The database is typed structurally, so this needs no
 * dependency on sqlite-wasm itself.
 */
import type { SqliteDatabase, SqliteExecutor, SqliteValue } from '../vendor/durable/src/storage/sqlite/database.ts';
import { SqliteStorage } from '../vendor/durable/src/storage/sqlite/storage.ts';

/** The part of sqlite-wasm's `oo1.DB` used here (ts-host's `WasmDb`). */
export type WasmSqliteDb = {
  exec(options: { sql: string; bind?: SqliteValue[]; rowMode?: 'object'; returnValue?: 'resultRows' }): unknown;
  close(): void;
};

/** Runs operations one at a time in call order, so a transaction excludes every other operation. */
class SerialQueue {
  #tail: Promise<unknown> = Promise.resolve();
  run<T>(operation: () => T | Promise<T>): Promise<T> {
    const result = this.#tail.then(operation);
    this.#tail = result.catch(() => {});
    return result;
  }
}

/** Executes statements on the database; sqlite-wasm prepares each from its text. */
abstract class WasmExecutor implements SqliteExecutor {
  protected readonly db: WasmSqliteDb;
  constructor(db: WasmSqliteDb) { this.db = db; }
  exec(sql: string): Promise<void> { return this.operation(() => { this.db.exec({ sql }); }); }
  run(sql: string, ...params: SqliteValue[]): Promise<void> { return this.operation(() => { this.db.exec({ sql, bind: params }); }); }
  get<T extends object>(sql: string, ...params: SqliteValue[]): Promise<T | undefined> {
    return this.operation(() => this.#rows<T>(sql, params)[0]);
  }
  all<T extends object>(sql: string, ...params: SqliteValue[]): Promise<T[]> { return this.operation(() => this.#rows<T>(sql, params)); }
  #rows<T>(sql: string, params: SqliteValue[]): T[] {
    return this.db.exec({ sql, bind: params, rowMode: 'object', returnValue: 'resultRows' }) as T[];
  }
  protected abstract operation<T>(body: () => T): Promise<T>;
}

/** A transaction's handle: valid only while its callback runs, and never queued behind the transaction itself. */
class WasmTransaction extends WasmExecutor {
  readonly #scope: { active: boolean };
  constructor(db: WasmSqliteDb, scope: { active: boolean }) { super(db); this.#scope = scope; }
  protected async operation<T>(body: () => T): Promise<T> {
    if (!this.#scope.active) throw new Error('SQLite transaction handle is no longer active');
    return body();
  }
}

/** `SqliteDatabase` over a sqlite-wasm database. */
export class WasmSqliteDatabase extends WasmExecutor implements SqliteDatabase {
  readonly #queue = new SerialQueue();
  #closed = false;

  transaction<T>(callback: (transaction: SqliteExecutor) => Promise<T>): Promise<T> {
    return this.#queue.run(async () => {
      this.db.exec({ sql: 'BEGIN IMMEDIATE' });
      const scope = { active: true };
      try {
        const result = await callback(new WasmTransaction(this.db, scope));
        scope.active = false;
        this.db.exec({ sql: 'COMMIT' });
        return result;
      } catch (error) {
        scope.active = false;
        try { this.db.exec({ sql: 'ROLLBACK' }); }
        catch (rollbackError) { throw new AggregateError([error, rollbackError], 'SQLite transaction failed and rollback failed'); }
        throw error;
      }
    });
  }

  close(): Promise<void> {
    return this.#queue.run(() => {
      if (this.#closed) return;
      this.#closed = true;
      this.db.close();
    });
  }

  protected operation<T>(body: () => T): Promise<T> { return this.#queue.run(body); }
}

/** Durable storage in a sqlite-wasm database (migrated to pi-durable's schema on open). */
export async function openWasmSqliteStorage(db: WasmSqliteDb): Promise<SqliteStorage> {
  return SqliteStorage.open(new WasmSqliteDatabase(db));
}
