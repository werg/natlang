/**
 * The call record store (§3.4), independent of where it lives: SQLite rows for calls, annotations, compilations, their
 * cases and the links between cases and calls (§7.2), and content-addressed blobs for values, event streams and
 * compilation files. A `StoreMedium` keeps the blobs and settings and knows the free space and the processes; Node
 * keeps them in files beside `calls.sqlite` (store.ts), the browser in the same OPFS database (browser/call-store-worker.ts).
 */
import { hexDigest } from '../native/hash.js';
import type { IterationStatisticsStore, SiteStatistics } from '../runtime/iterate.js';
import { applyDecision, crispPolicy, ruleOf, RECENT, type EvidenceSummary, type PromotionDecision } from './evidence.js';
import { DEFAULT_SETTINGS, type CallRecord, type CallStoreSettings, type CaseRole, type CaseStats, type CaseTier,
  type CompilationRow, type DeclineReason, type DeclineRow, type ValueRef, type Verdict } from './types.js';

export type SqlValue = string | number | bigint | null | Uint8Array;
/** The part of a synchronous SQLite binding the store uses (node:sqlite's DatabaseSync has it). */
export interface SqlDatabase {
  exec(sql: string): void;
  prepare(sql: string): { run(...params: SqlValue[]): unknown; get(...params: SqlValue[]): unknown; all(...params: SqlValue[]): unknown[] };
  close(): void;
}

/** Where a store keeps what is not rows, and what it knows of its surroundings. */
export interface StoreMedium {
  /** Store a blob outside the database; true when this call created it (a file a rolled-back transaction must remove). */
  putBlob(hash: string, text: string): boolean;
  blob(hash: string): string | undefined;
  deleteBlob(hash: string): void;
  readConfig(): Partial<CallStoreSettings>;
  writeConfig(config: Partial<CallStoreSettings>): void;
  /** Free bytes where the store lives; Infinity when unknown. */
  freeBytes(): number;
  /** Bytes the database itself takes on the medium (its files, WAL included). */
  databaseBytes(): number;
  /**
   * The allocation unit of a blob kept outside the database (a file takes its bytes rounded up to it), or 0 when blobs
   * live in the database and its pages already count them.
   */
  blobBlockBytes(): number;
  /**
   * Room the database's log takes beside its pages (Node: the write-ahead log and its index, at least the size they are
   * truncated to after a checkpoint, more while a reader holds a longer log); 0 when there is none. The bound leaves it.
   */
  logBytes(): number;
  /** A specialization mode set from outside the store (Node: NATLANG_SPECIALIZATION, supplied by the machine-store module). */
  modeOverride(): string | undefined;
  /** This process, for the rows of the calls it runs. */
  readonly process: { pid: number | null; scope: string };
  /** Whether the process that wrote a running row is known to be gone. */
  gone(pid: number | null, scope: string | null): boolean;
}

export const byteLength = (text: string): number => new TextEncoder().encode(text).length;

export type CallSummary = { call_id: string; parent_call_id: string | null; definition_id: string; definition_name: string;
  definition_key: string; definition_source: string | null; executor: string; case_hash: string | null; outcome: string;
  started_at: string; wall_ms: number; tokens_in: number; tokens_out: number; approach_hash: string | null;
  program_root: string | null; audit_of: string | null; parent_action_index?: number | null };
export type CallFilter = { definition?: string; key?: string; source?: string; outcome?: string; executor?: string;
  caseHash?: string; since?: string; program?: string; limit?: number; after?: string; audits?: boolean };
export type HotDefinition = { definition_key: string; definition_id: string; definition_name: string; definition_source: string | null;
  program_root: string | null; calls: number; agent_calls: number; crisp_calls: number; tokens: number; wall_ms: number; last_at: string };
export type FindingInput = { definitionKey: string; definitionId: string; definitionName: string; definitionSource: string | null;
  kind: string; summary: string; detail?: unknown };
export type FindingRow = { id: number; fingerprint: string; definition_key: string; definition_id: string; definition_name: string;
  definition_source: string | null; kind: string; summary: string; detail: unknown; seen: number; status: string; created_at: string; updated_at: string };
export type SavingsRow = { definition_key: string; definition_name: string; definition_source: string | null; agent_calls: number; served: number;
  agent_tokens_per_call: number; agent_ms_per_call: number; crisp_ms_per_call: number; saved_tokens: number; saved_ms: number;
  spent_tokens: number; spent_ms: number };
export type AuditJob = { id: number; kind: 'audit' | 'shadow'; case_hash: string; call_id: string; enqueued_at: string };

const SCHEMA = `
CREATE TABLE IF NOT EXISTS blobs (hash TEXT PRIMARY KEY, bytes INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS calls (
  call_id TEXT PRIMARY KEY, parent_call_id TEXT, parent_action_index INTEGER, task_id TEXT, program_id TEXT, build_hash TEXT,
  program_root TEXT, definition_id TEXT NOT NULL, definition_name TEXT NOT NULL, definition_source TEXT, definition_key TEXT NOT NULL,
  interface_hash TEXT, site TEXT, executor TEXT NOT NULL, model_id TEXT, case_hash TEXT, outcome TEXT, started_at TEXT, ended_at TEXT,
  wall_ms INTEGER, tokens_in INTEGER, tokens_out INTEGER, model_requests INTEGER, evals INTEGER, approach_hash TEXT, audit_of TEXT,
  record_hash TEXT NOT NULL, events_hash TEXT, pinned INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS calls_key ON calls(definition_key, started_at);
CREATE INDEX IF NOT EXISTS calls_definition ON calls(definition_id, started_at);
CREATE INDEX IF NOT EXISTS calls_started ON calls(started_at);
CREATE INDEX IF NOT EXISTS calls_case ON calls(case_hash);
CREATE INDEX IF NOT EXISTS calls_parent ON calls(parent_call_id);
CREATE TABLE IF NOT EXISTS call_blobs (call_id TEXT NOT NULL, hash TEXT NOT NULL, kind TEXT NOT NULL, PRIMARY KEY (call_id, hash, kind));
CREATE INDEX IF NOT EXISTS call_blobs_hash ON call_blobs(hash);
CREATE TABLE IF NOT EXISTS annotations (id INTEGER PRIMARY KEY AUTOINCREMENT, call_id TEXT NOT NULL, kind TEXT NOT NULL,
  value TEXT NOT NULL, source TEXT, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS annotations_call ON annotations(call_id);
CREATE TABLE IF NOT EXISTS compilations (id TEXT PRIMARY KEY, definition_key TEXT NOT NULL, definition_id TEXT NOT NULL,
  definition_name TEXT NOT NULL, definition_source TEXT, interface_hash TEXT NOT NULL, program_root TEXT, folder_hash TEXT NOT NULL,
  parent_id TEXT, created_at TEXT NOT NULL, status TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS compilations_key ON compilations(definition_key, status);
CREATE TABLE IF NOT EXISTS compilation_files (compilation_id TEXT NOT NULL, path TEXT NOT NULL, hash TEXT NOT NULL, PRIMARY KEY (compilation_id, path));
CREATE INDEX IF NOT EXISTS compilation_files_hash ON compilation_files(hash);
CREATE TABLE IF NOT EXISTS cases (hash TEXT NOT NULL, compilation_id TEXT NOT NULL, position INTEGER NOT NULL, tier TEXT NOT NULL,
  served INTEGER NOT NULL DEFAULT 0, handed_off INTEGER NOT NULL DEFAULT 0, compared INTEGER NOT NULL DEFAULT 0,
  worse INTEGER NOT NULL DEFAULT 0, better INTEGER NOT NULL DEFAULT 0, audited INTEGER NOT NULL DEFAULT 0,
  audit_worse INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, promoted_at TEXT, demoted_at TEXT, note TEXT,
  PRIMARY KEY (hash, compilation_id));
CREATE TABLE IF NOT EXISTS case_calls (case_hash TEXT NOT NULL, call_id TEXT NOT NULL, role TEXT NOT NULL, verdict TEXT,
  created_at TEXT NOT NULL, PRIMARY KEY (case_hash, call_id, role));
CREATE INDEX IF NOT EXISTS case_calls_call ON case_calls(call_id);
CREATE TABLE IF NOT EXISTS declines (definition_key TEXT PRIMARY KEY, definition_id TEXT NOT NULL, reason TEXT NOT NULL, why TEXT NOT NULL,
  calls_at_decline INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, case_hash TEXT NOT NULL, call_id TEXT NOT NULL,
  status TEXT NOT NULL, enqueued_at TEXT NOT NULL, done_at TEXT, verdict TEXT, detail TEXT, UNIQUE (kind, case_hash, call_id));
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS findings (id INTEGER PRIMARY KEY AUTOINCREMENT, fingerprint TEXT NOT NULL UNIQUE, definition_key TEXT NOT NULL,
  definition_id TEXT NOT NULL, definition_name TEXT NOT NULL, definition_source TEXT, kind TEXT NOT NULL, summary TEXT NOT NULL, detail TEXT NOT NULL,
  seen INTEGER NOT NULL DEFAULT 1, status TEXT NOT NULL DEFAULT 'open', created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS findings_definition ON findings(definition_key, status);
CREATE TABLE IF NOT EXISTS spend (id INTEGER PRIMARY KEY AUTOINCREMENT, definition_key TEXT NOT NULL, activity TEXT NOT NULL,
  tokens INTEGER NOT NULL, wall_ms INTEGER NOT NULL, created_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS spend_definition ON spend(definition_key);
CREATE TABLE IF NOT EXISTS iteration_statistics (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS refinement_verdicts (key TEXT PRIMARY KEY, probability REAL NOT NULL, judge TEXT NOT NULL, created_at TEXT NOT NULL);
`;

/** The call store's `VerdictCache` (native/refinement.ts), with a count and a clear for tools. */
export type RefinementVerdictStore = import('../native/refinement.js').VerdictCache & { count(): number; clear(): void };

/**
 * Byte counters of the blobs table, kept by triggers so every writer (any process, any code version) maintains them in
 * the transaction that inserts or deletes a blob: `blob_bytes` the stored text, `blob_disk_bytes` what blob files take
 * on the medium (`blob_block_bytes` the allocation unit; 0 when blobs live in the database).
 */
const BLOB_TRIGGERS = `
CREATE TRIGGER IF NOT EXISTS blobs_count_insert AFTER INSERT ON blobs BEGIN
  UPDATE meta SET value = CAST(value AS INTEGER) + NEW.bytes WHERE key = 'blob_bytes';
  UPDATE meta SET value = CAST(value AS INTEGER) + (SELECT CASE WHEN block = 0 THEN 0 ELSE (NEW.bytes + block - 1) / block * block END
    FROM (SELECT CAST(value AS INTEGER) AS block FROM meta WHERE key = 'blob_block_bytes')) WHERE key = 'blob_disk_bytes';
END;
CREATE TRIGGER IF NOT EXISTS blobs_count_delete AFTER DELETE ON blobs BEGIN
  UPDATE meta SET value = CAST(value AS INTEGER) - OLD.bytes WHERE key = 'blob_bytes';
  UPDATE meta SET value = CAST(value AS INTEGER) - (SELECT CASE WHEN block = 0 THEN 0 ELSE (OLD.bytes + block - 1) / block * block END
    FROM (SELECT CAST(value AS INTEGER) AS block FROM meta WHERE key = 'blob_block_bytes')) WHERE key = 'blob_disk_bytes';
END;`;

const now = () => new Date().toISOString();
/** Calls per eviction transaction: the longest a writer in another process waits for an eviction. */
const EVICTION_CHUNK = 64;
/** Database pages one incremental-vacuum transaction returns (16 MiB at 4 KiB pages). */
const VACUUM_STEP_PAGES = 4096;
const refHashes = (ref: ValueRef | null | undefined): string[] => ref?.complete ? [ref.hash] : [];

/**
 * A call record store. Under Node, open the machine's with `CallStore.open(root)` (store.ts; one instance per root per
 * process); in the browser the call-store worker builds one over OPFS.
 */
export class CallStore {
  /** Opens a store at a path; set by the Node store module (store.ts). */
  static opener?: (root: string, options?: { mode?: () => string | undefined }) => CallStore;
  static open(root: string, options?: { mode?: () => string | undefined }): CallStore {
    if (!CallStore.opener) throw new Error('CallStore.open needs the Node store module (calls/store.js)');
    return CallStore.opener(root, options);
  }
  /** Called when the store closes (the Node module forgets its instance). */
  onClose?: () => void;

  private settingsCache?: { at: number; value: CallStoreSettings };

  readonly db: SqlDatabase;
  constructor(readonly root: string, db: SqlDatabase, readonly medium: StoreMedium) {
    // Prepared statements are reused: eviction runs a few statements per row over millions of rows, and a statement
    // prepared per use holds native memory until the garbage collector finalizes it.
    const statements = new Map<string, ReturnType<SqlDatabase['prepare']>>();
    this.db = { exec: sql => db.exec(sql), close: () => { statements.clear(); db.close(); },
      prepare: sql => {
        let statement = statements.get(sql);
        if (!statement) {
          if (statements.size >= 512) statements.clear();
          statements.set(sql, statement = db.prepare(sql));
        }
        return statement;
      } };
    this.db.exec(SCHEMA);
    const columns = (this.db.prepare('PRAGMA table_info(calls)').all() as { name: string }[]).map(column => column.name);
    if (!columns.includes('pid')) this.db.exec('ALTER TABLE calls ADD COLUMN pid INTEGER');
    if (!columns.includes('process_scope')) this.db.exec('ALTER TABLE calls ADD COLUMN process_scope TEXT');
    if (!columns.includes('run_label')) this.db.exec('ALTER TABLE calls ADD COLUMN run_label TEXT');
    this.db.exec('CREATE INDEX IF NOT EXISTS calls_run_label ON calls(run_label)');
    const caseColumns = (this.db.prepare('PRAGMA table_info(cases)').all() as { name: string }[]).map(column => column.name);
    if (!caseColumns.includes('live_compared')) this.db.exec('ALTER TABLE cases ADD COLUMN live_compared INTEGER NOT NULL DEFAULT 0');
    if (!caseColumns.includes('live_worse')) this.db.exec('ALTER TABLE cases ADD COLUMN live_worse INTEGER NOT NULL DEFAULT 0');
    this.countBlobs();
    this.markInterrupted();
  }

  /** Install the blob byte counters (BLOB_TRIGGERS) once per database, starting them from the blobs already stored. */
  private countBlobs(): void {
    const installed = () => !!this.db.prepare("SELECT 1 AS ok FROM meta WHERE key = 'blob_disk_bytes'").get();
    if (installed()) return;
    this.db.exec('BEGIN IMMEDIATE');
    try {
      if (!installed()) {
        const block = Math.max(0, Math.floor(this.medium.blobBlockBytes()));
        // The block size is an integer literal: a bound JS number may bind as a REAL, and the rounding needs integer division.
        const sums = this.db.prepare(`SELECT COALESCE(SUM(bytes), 0) AS bytes,
          COALESCE(SUM(${block === 0 ? '0' : `(bytes + ${block - 1}) / ${block} * ${block}`}), 0) AS disk FROM blobs`).get() as
          { bytes: number; disk: number };
        const put = this.db.prepare('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)');
        put.run('blob_block_bytes', String(block));
        put.run('blob_bytes', String(sums.bytes));
        put.run('blob_disk_bytes', String(sums.disk));
        this.db.exec(BLOB_TRIGGERS);
      }
      this.db.exec('COMMIT');
    } catch (error) { this.db.exec('ROLLBACK'); throw error; }
  }

  /** Calls left `running` by a process that is gone (as the medium knows it) become `interrupted`. */
  markInterrupted(): number {
    const rows = this.db.prepare("SELECT call_id, pid, process_scope FROM calls WHERE outcome = 'running'").all() as
      { call_id: string; pid: number | null; process_scope: string | null }[];
    let marked = 0;
    for (const row of rows) {
      if (!this.medium.gone(row.pid, row.process_scope)) continue;
      this.db.prepare("UPDATE calls SET outcome = 'interrupted' WHERE call_id = ? AND outcome = 'running'").run(row.call_id);
      marked++;
    }
    return marked;
  }

  close(): void { this.onClose?.(); this.db.close(); }

  // --- Settings ---------------------------------------------------------------------------------------------------

  /** Machine settings: defaults, then the store's config, then the medium's mode override (Node: NATLANG_SPECIALIZATION). Re-read every 5 s. */
  settings(): CallStoreSettings {
    const cached = this.settingsCache;
    if (cached && Date.now() - cached.at < 5000) return cached.value;
    let file: Partial<CallStoreSettings> = {};
    try { file = this.medium.readConfig(); } catch { /* defaults */ }
    const value = { ...DEFAULT_SETTINGS, ...file };
    const mode = this.medium.modeOverride()?.trim();
    if (mode === 'off' || mode === 'shadow' || mode === 'on') value.specialization = mode;
    this.settingsCache = { at: Date.now(), value };
    return value;
  }
  writeSettings(changes: Partial<CallStoreSettings>): CallStoreSettings {
    let file: Partial<CallStoreSettings> = {};
    try { file = this.medium.readConfig(); } catch { /* new */ }
    this.medium.writeConfig({ ...file, ...changes });
    this.settingsCache = undefined;
    return this.settings();
  }

  // --- Blobs ------------------------------------------------------------------------------------------------------

  /** Store `text` under its SHA-256; returns the hash. Existing blobs are not rewritten. */
  putBlob(text: string, hash = hexDigest(text)): string {
    if (this.medium.putBlob(hash, text)) this.created?.push(hash);
    this.db.prepare('INSERT OR IGNORE INTO blobs (hash, bytes) VALUES (?, ?)').run(hash, byteLength(text));
    return hash;
  }
  blob(hash: string): string | undefined {
    try { return this.medium.blob(hash); } catch { return undefined; }
  }
  /** The value a complete reference names, or `undefined` when it is incomplete or evicted. */
  value(ref: ValueRef | null | undefined): unknown {
    if (!ref?.complete) return undefined;
    const text = this.blob(ref.hash);
    return text === undefined ? undefined : JSON.parse(text);
  }

  // --- Calls ------------------------------------------------------------------------------------------------------

  /** Store one call record. `blobs` holds the text of every complete value the record references (by hash). */
  record(record: CallRecord, blobs: ReadonlyMap<string, string>, events?: string): void {
    // Publish blobs, the record and its links together: another writer's eviction
    // must never collect a value between insertion and linking.
    this.inWrite(() => {
      const kinds = new Map<string, string>();
      for (const [hash, text] of blobs) { this.putBlob(text, hash); kinds.set(hash, 'value'); }
      let eventsHash: string | null = null;
      if (events !== undefined) {
        eventsHash = this.putBlob(events);
        record = { ...record, events: { complete: true, hash: eventsHash, bytes: byteLength(events) } };
      }
      const recordHash = this.putBlob(JSON.stringify(record));
      const definition = record.definition;
      this.db.prepare(`INSERT OR REPLACE INTO calls (call_id, parent_call_id, parent_action_index, task_id, program_id, build_hash,
        program_root, definition_id, definition_name, definition_source, definition_key, interface_hash, site, executor, model_id, case_hash,
        outcome, started_at, ended_at, wall_ms, tokens_in, tokens_out, model_requests, evals, approach_hash, audit_of, record_hash, events_hash, run_label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
        record.call_id, record.parent_call_id, record.parent_action_index, record.task_id, record.program_id, record.build_hash,
        record.program_root, definition.id, definition.name, definition.source, definition.key, definition.interface, definition.site,
        record.executor.kind, record.executor.model_id, record.executor.case_hash ?? null, record.outcome, record.started_at, record.ended_at,
        record.cost.wall_ms, record.cost.tokens_in, record.cost.tokens_out, record.cost.model_requests, record.cost.evals,
        record.approach.hash, record.audit_of ?? null, recordHash, eventsHash, record.run_label ?? null);
      const link = this.db.prepare('INSERT OR IGNORE INTO call_blobs (call_id, hash, kind) VALUES (?, ?, ?)');
      link.run(record.call_id, recordHash, 'record');
      if (eventsHash) link.run(record.call_id, eventsHash, 'events');
      this.db.prepare("DELETE FROM annotations WHERE call_id = ? AND kind = 'call-checkpoint'").run(record.call_id);
      this.db.prepare("DELETE FROM call_blobs WHERE call_id = ? AND kind = 'capture'").run(record.call_id);
      const referenced = [definition.instructions, ...Object.values(record.inputs), ...Object.values(record.captures), record.output,
        ...record.capture_writes.map(write => write.after), ...record.effects.flatMap(effect => [effect.args, effect.result]),
        ...(record.folder?.changes.map(change => change.after) ?? []), ...(record.code_revisions?.map(revision => revision.after) ?? [])].flatMap(refHashes);
      for (const hash of new Set(referenced)) if (blobs.has(hash) || kinds.has(hash)) link.run(record.call_id, hash, 'value');
    });
    this.keepBound();
  }

  /**
   * After a write: when the store is over its bound, a bounded eviction pass (`AUTO_EVICTION_WORK`). Checked on every
   * write from the shared counters, so the bound holds whichever process writes and however few calls each records.
   * An eviction that fails (another writer holds the lock too long) is left to the next write.
   */
  private keepBound(): void {
    try {
      const bound = this.bound();
      if (this.storeBytes() > bound) this.evict(bound, CallStore.AUTO_EVICTION_WORK);
    } catch { /* the next write tries again */ }
  }
  /** Work (blob candidates and call rows) of one eviction pass after a write. */
  static AUTO_EVICTION_WORK = 1024;

  /** Save an unfinished call's actual inputs and effects without a fabricated completion. */
  checkpoint(callId: string, state: Record<string,unknown>, blobs: ReadonlyMap<string,string>, events?: string): void {
    if(state.call_id !== callId) throw new Error('call checkpoint identity differs');
    this.inWrite(() => {
      const row=this.db.prepare('SELECT outcome FROM calls WHERE call_id = ?').get(callId) as {outcome:string}|undefined;
      if(!row || row.outcome !== 'running') return;
      const link=this.db.prepare("INSERT OR IGNORE INTO call_blobs(call_id,hash,kind) VALUES (?,?,'capture')");
      for(const [hash,text] of blobs){this.putBlob(text,hash);link.run(callId,hash);}
      this.db.prepare("DELETE FROM annotations WHERE call_id = ? AND kind = 'call-checkpoint'").run(callId);
      this.annotate(callId,'call-checkpoint',state,'natlang-call-capture',false);
      if(events !== undefined) this.progressInWrite(callId,events);
    });
  }

  /** A row for a call that has started, so running calls show in listings; `record` replaces it when the call ends. */
  begin(row: { callId: string; parentCallId: string | null; parentActionIndex: number | null; taskId: string; programId: string | null;
    programRoot: string | null; definition: { id: string; name: string; source: string | null; key: string; interface: string; site: string };
    modelId: string | null; startedAt: string; auditOf: string | null; runLabel?: string }): void {
    this.db.prepare(`INSERT OR IGNORE INTO calls (call_id, parent_call_id, parent_action_index, task_id, program_id, program_root, definition_id,
      definition_name, definition_source, definition_key, interface_hash, site, executor, model_id, outcome, started_at, audit_of, record_hash, pid, process_scope,
      run_label)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'agent', ?, 'running', ?, ?, '', ?, ?, ?)`).run(row.callId, row.parentCallId, row.parentActionIndex,
      row.taskId, row.programId, row.programRoot, row.definition.id, row.definition.name, row.definition.source, row.definition.key,
      row.definition.interface, row.definition.site, row.modelId, row.startedAt, row.auditOf, this.medium.process.pid, this.medium.process.scope,
      row.runLabel ?? null);
  }

  /** Remove the row of a call that started and will not be recorded, with the event streams written while it ran. */
  discard(callId: string): void {
    this.inWrite(() => {
      const removed = this.db.prepare("DELETE FROM calls WHERE call_id = ? AND outcome = 'running' RETURNING call_id").get(callId);
      if (!removed) return;
      const hashes = this.db.prepare('SELECT hash FROM call_blobs WHERE call_id = ?').all(callId) as { hash: string }[];
      this.db.prepare('DELETE FROM call_blobs WHERE call_id = ?').run(callId);
      this.collectHashes(hashes.map(row => row.hash));
    });
  }

  /**
   * The trace events so far of a running call (written every so often while it runs). The stream it replaces is
   * collected at once, and the new blob is linked in the transaction that stores it, so eviction never sees it unlinked.
   */
  progress(callId: string, events: string): void {
    this.inWrite(() => this.progressInWrite(callId,events));
    this.keepBound();
  }
  private progressInWrite(callId: string,events: string): void {
      if(!this.db.prepare("SELECT 1 AS present FROM calls WHERE call_id = ? AND outcome IN ('running','interrupted')").get(callId)) return;
      const hash = this.putBlob(events);
      const old = this.db.prepare("SELECT hash FROM call_blobs WHERE call_id = ? AND kind = 'events' AND hash != ?").all(callId, hash) as { hash: string }[];
      this.db.prepare("UPDATE calls SET events_hash = ? WHERE call_id = ? AND outcome IN ('running', 'interrupted')").run(hash, callId);
      this.db.prepare("DELETE FROM call_blobs WHERE call_id = ? AND kind = 'events' AND hash != ?").run(callId, hash);
      this.db.prepare("INSERT OR IGNORE INTO call_blobs (call_id, hash, kind) VALUES (?, ?, 'events')").run(callId, hash);
      this.collectHashes(old.map(row => row.hash));
  }
  /** Blob files the current write transaction created, removed again if it rolls back. */
  private created: string[] | null = null;
  /**
   * Run `body` under the write lock (taken before any read, so SQLite never upgrades a reader). When it fails, the blob
   * files it created and no committed row names are removed: they would otherwise stay on disk, counted by nothing.
   */
  private inWrite<T>(body: () => T): T {
    this.db.exec('BEGIN IMMEDIATE');
    const created: string[] = [];
    this.created = created;
    try {
      const value = body();
      this.db.exec('COMMIT');
      return value;
    } catch (error) {
      try { this.db.exec('ROLLBACK'); } catch { /* already rolled back */ }
      for (const hash of created) try {
        if (!this.db.prepare('SELECT 1 AS ok FROM blobs WHERE hash = ?').get(hash)) this.medium.deleteBlob(hash);
      } catch { /* left for a later sweep */ }
      throw error;
    } finally { this.created = null; }
  }

  call(callId: string): CallRecord | undefined {
    const row = this.db.prepare('SELECT record_hash FROM calls WHERE call_id = ?').get(callId) as { record_hash: string } | undefined;
    const text = row && this.blob(row.record_hash);
    return text ? JSON.parse(text) as CallRecord : undefined;
  }
  /** The call's trace events (`natlang.trace`), when they have not been evicted. */
  events(callId: string): Record<string, unknown>[] | undefined {
    const row = this.db.prepare('SELECT events_hash FROM calls WHERE call_id = ?').get(callId) as { events_hash: string | null } | undefined;
    const text = row?.events_hash ? this.blob(row.events_hash) : undefined;
    return text === undefined ? undefined : text.split('\n').filter(Boolean).map(line => JSON.parse(line));
  }
  /** Calls matching `filter`, newest first. `after` continues from a call ID of an earlier page. */
  calls(filter: CallFilter = {}): CallSummary[] {
    const where: string[] = [], params: (string | number)[] = [];
    const add = (sql: string, value: string | number) => { where.push(sql); params.push(value); };
    if (filter.definition) { where.push('(definition_id = ? OR definition_name = ? OR definition_source = ?)'); params.push(filter.definition, filter.definition, filter.definition); }
    if (filter.key) add('definition_key = ?', filter.key);
    if (filter.source) add('definition_source = ?', filter.source);
    if (filter.outcome) add('outcome = ?', filter.outcome);
    if (filter.executor) add('executor = ?', filter.executor);
    if (filter.caseHash) add('case_hash = ?', filter.caseHash);
    if (filter.since) add('started_at >= ?', filter.since);
    if (filter.program) add('(program_root = ? OR program_id = ?)', filter.program), params.push(filter.program);
    if (!filter.audits) where.push('audit_of IS NULL');
    if (filter.after) {
      const at = this.db.prepare('SELECT started_at FROM calls WHERE call_id = ?').get(filter.after) as { started_at: string } | undefined;
      if (at) { where.push('(started_at < ? OR (started_at = ? AND call_id < ?))'); params.push(at.started_at, at.started_at, filter.after); }
    }
    const sql = `SELECT call_id, parent_call_id, definition_id, definition_name, definition_key, definition_source, executor, case_hash,
      outcome, started_at, wall_ms, tokens_in, tokens_out, approach_hash, program_root, audit_of FROM calls
      ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY started_at DESC, call_id DESC LIMIT ?`;
    return this.db.prepare(sql).all(...params, filter.limit ?? 50) as CallSummary[];
  }
  /** Definition revisions by volume or cost since `since`. */
  hot(options: { since?: string; program?: string; by?: 'calls' | 'tokens' | 'wall_ms'; limit?: number } = {}): HotDefinition[] {
    const order = options.by === 'tokens' ? 'tokens' : options.by === 'wall_ms' ? 'wall_ms' : 'calls';
    const where = ['audit_of IS NULL', "site != 'internal'"], params: string[] = [];
    if (options.since) { where.push('started_at >= ?'); params.push(options.since); }
    if (options.program) { where.push('(program_root = ? OR program_id = ?)'); params.push(options.program, options.program); }
    return this.db.prepare(`SELECT definition_key, definition_id, definition_name, MAX(definition_source) AS definition_source,
      MAX(program_root) AS program_root, COUNT(*) AS calls, SUM(executor = 'agent') AS agent_calls,
      SUM(executor = 'crisp') AS crisp_calls, SUM(COALESCE(tokens_in, 0) + COALESCE(tokens_out, 0)) AS tokens,
      SUM(COALESCE(wall_ms, 0)) AS wall_ms, MAX(started_at) AS last_at FROM calls WHERE ${where.join(' AND ')} AND outcome != 'running'
      GROUP BY definition_key ORDER BY ${order} DESC LIMIT ?`).all(...params, options.limit ?? 20) as HotDefinition[];
  }
  /** Successful agent calls of a definition revision, oldest first: the specializer's evidence. */
  agentCalls(definitionKey: string, limit = 2000): CallSummary[] {
    return this.db.prepare(`SELECT call_id, parent_call_id, definition_id, definition_name, definition_key, definition_source, executor,
      case_hash, outcome, started_at, wall_ms, tokens_in, tokens_out, approach_hash, program_root, audit_of FROM calls
      WHERE definition_key = ? AND executor != 'crisp' AND audit_of IS NULL ORDER BY started_at ASC LIMIT ?`).all(definitionKey, limit) as CallSummary[];
  }
  /** Calls started by a call, oldest first. */
  children(callId: string): CallSummary[] {
    return this.db.prepare(`SELECT call_id, parent_call_id, definition_id, definition_name, definition_key, definition_source, executor,
      case_hash, outcome, started_at, wall_ms, tokens_in, tokens_out, approach_hash, program_root, audit_of, parent_action_index FROM calls
      WHERE parent_call_id = ? ORDER BY started_at ASC`).all(callId) as CallSummary[];
  }
  /**
   * Calls a typed ID may mean, newest first: the exact ID, else every ID starting with `prefix`, or whose short form
   * (the ID without its task name and ordinal, `task-3-`) does.
   */
  matchCalls(prefix: string, limit = 20): Pick<CallSummary, 'call_id' | 'parent_call_id' | 'definition_name' | 'outcome' | 'started_at'>[] {
    const columns = 'call_id, parent_call_id, definition_name, outcome, started_at';
    const exact = this.db.prepare(`SELECT ${columns} FROM calls WHERE call_id = ?`).all(prefix) as ReturnType<CallStore['matchCalls']>;
    if (exact.length || !prefix) return exact;
    const like = prefix.replace(/[\\%_]/g, character => `\\${character}`);
    const rows = this.db.prepare(`SELECT ${columns} FROM calls WHERE call_id LIKE ? ESCAPE '\\' OR call_id LIKE ? ESCAPE '\\'
      ORDER BY started_at DESC, call_id DESC`).all(`${like}%`, `%-${like}%`) as ReturnType<CallStore['matchCalls']>;
    return rows.filter(row => row.call_id.startsWith(prefix) || row.call_id.replace(/^[^/]*?-\d+-/, '').startsWith(prefix)).slice(0, limit);
  }
  countCalls(definitionKey: string): number {
    return Number((this.db.prepare("SELECT COUNT(*) AS n FROM calls WHERE definition_key = ? AND audit_of IS NULL AND executor != 'crisp'")
      .get(definitionKey) as { n: number }).n);
  }
  /** Attach evidence to a call; it is pinned unless `pin` is false (routine annotations such as evaluation scores). */
  annotate(callId: string, kind: string, value: unknown, source?: string, pin = true): void {
    this.db.prepare('INSERT INTO annotations (call_id, kind, value, source, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(callId, kind, JSON.stringify(value), source ?? null, now());
    if (pin) this.pin(callId);
  }
  /** Tier events (calls/tiers.ts) in order, with the cost of the call each is attached to. */
  tierRows(): { call_id: string; value: unknown; definition_name: string | null; executor: string | null; model_id: string | null;
    tokens_in: number; tokens_out: number; wall_ms: number }[] {
    return (this.db.prepare(`SELECT a.call_id, a.value, c.definition_name, c.executor, c.model_id, COALESCE(c.tokens_in, 0) AS tokens_in,
      COALESCE(c.tokens_out, 0) AS tokens_out, COALESCE(c.wall_ms, 0) AS wall_ms FROM annotations a LEFT JOIN calls c ON c.call_id = a.call_id
      WHERE a.kind = 'tier' ORDER BY a.id`).all() as (Record<string, unknown> & { value: string })[])
      .map(row => ({ ...row, value: JSON.parse(row.value) })) as never;
  }
  /** Annotations of one kind across calls, newest first, with the call they are attached to (for example `escalation`). */
  annotationsOf(kind: string, options: { limit?: number } = {}): { call_id: string; value: unknown; definition_id: string | null; created_at: string }[] {
    return (this.db.prepare(`SELECT a.call_id, a.value, c.definition_id, a.created_at FROM annotations a LEFT JOIN calls c ON c.call_id = a.call_id
      WHERE a.kind = ? ORDER BY a.id DESC LIMIT ?`).all(kind, options.limit ?? 500) as
      { call_id: string; value: string; definition_id: string | null; created_at: string }[]).map(row => ({ ...row, value: JSON.parse(row.value) }));
  }
  annotations(callId: string): { kind: string; value: unknown; source: string | null; created_at: string }[] {
    return (this.db.prepare('SELECT kind, value, source, created_at FROM annotations WHERE call_id = ? ORDER BY id').all(callId) as
      { kind: string; value: string; source: string | null; created_at: string }[]).map(row => ({ ...row, value: JSON.parse(row.value) }));
  }
  /** Keep a call (and its values) when the store evicts. */
  pin(callId: string, pinned = true): void { this.db.prepare('UPDATE calls SET pinned = ? WHERE call_id = ?').run(pinned ? 1 : 0, callId); }

  // --- Bounds -----------------------------------------------------------------------------------------------------

  private counter(key: string): number {
    const row = this.db.prepare('SELECT CAST(value AS INTEGER) AS n FROM meta WHERE key = ?').get(key) as { n: number } | undefined;
    return Number(row?.n ?? 0);
  }
  /** Bytes of the stored blob texts. */
  totalBytes(): number { return this.counter('blob_bytes'); }
  /** Bytes of database pages not in use (reused by later writes; returned to the medium by `compact` and incremental vacuum). */
  private freeDatabaseBytes(): number {
    const pages = this.db.prepare('PRAGMA freelist_count').get() as { freelist_count: number };
    const size = this.db.prepare('PRAGMA page_size').get() as { page_size: number };
    return Number(pages.freelist_count) * Number(size.page_size);
  }
  /**
   * What the store occupies and eviction can reduce: blob files as allocated (blobs in the database count as its
   * pages) plus the database pages in use. Eviction cannot shrink the write-ahead log (it writes to it), so `bound`
   * leaves its room (`StoreMedium.logBytes`) out of `maxStoreBytes` instead.
   */
  storeBytes(): number {
    const pages = this.db.prepare('PRAGMA page_count').get() as { page_count: number };
    const size = this.db.prepare('PRAGMA page_size').get() as { page_size: number };
    return this.counter('blob_disk_bytes') + Number(pages.page_count) * Number(size.page_size) - this.freeDatabaseBytes();
  }

  /**
   * The store's bound now (on `storeBytes`): its size limit less the room of the database's log, so the files on disk
   * stay within `maxStoreBytes`; lowered further so the filesystem keeps `minFreeBytes` free.
   */
  bound(): number {
    const settings = this.settings();
    let free = Infinity;
    try { free = this.medium.freeBytes(); } catch { /* unknown */ }
    const total = this.storeBytes();
    let log = 0;
    try { log = this.medium.logBytes(); } catch { /* unknown */ }
    return Math.max(0, Math.min(settings.maxStoreBytes - log, free < settings.minFreeBytes ? total - (settings.minFreeBytes - free) : Infinity));
  }

  /**
   * Bring the store (`storeBytes`) under its bound, to 90% of it: first unlinked blobs, then the event streams of
   * unpinned calls, oldest first, then whole unpinned calls. A call is pinned when it is marked so, annotated, or linked
   * to a case. Returns the bytes freed.
   *
   * Work happens in short write transactions (one page of blob candidates or `EVICTION_CHUNK` calls), so writers in
   * other processes wait at most one of them. A finite work limit bounds the inspected blob candidates and processed call
   * rows; the orphan and event sweeps keep cursors in `meta`, so repeated passes resume instead of rescanning what they
   * passed. Automatic writes use that mode so a large store cannot stall the executing model; explicit eviction defaults
   * to all the work needed for the bound. Freed database pages go back to the medium when the database has incremental
   * auto-vacuum (`compact` converts an older one).
   */
  evict(maxBytes = this.bound(), workLimit = Infinity): number {
    if (workLimit !== Infinity && (!Number.isSafeInteger(workLimit) || workLimit < 2))
      throw new RangeError('eviction work limit must be an integer of at least 2');
    const before = this.storeBytes();
    if (before <= maxBytes) return 0;
    const target = Math.floor(maxBytes * 0.9);
    const unpinned = `pinned = 0 AND call_id NOT IN (SELECT call_id FROM case_calls)`;
    const over = () => this.storeBytes() > target;
    let remaining = workLimit;
    const cursor = (key: string) => (this.db.prepare('SELECT value FROM meta WHERE key = ?').get(key) as { value: string } | undefined)?.value ?? '';
    const setCursor = (key: string, value: string) => this.db.prepare('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)').run(key, value);

    // Unlinked blobs (left by a writer that failed between storing and linking, or by older versions), a page of
    // candidate keys at a time: LIMIT on matching orphans alone would scan every referenced blob under the lock.
    // A bounded pass looks at one page; an explicit one sweeps the whole table once.
    const orphanPage = (after: string, size: number) => {
      const rows = this.db.prepare('SELECT hash FROM blobs WHERE hash > ? ORDER BY hash LIMIT ?').all(after, size) as { hash: string }[];
      this.collectHashes(rows.map(row => row.hash));
      return rows.map(row => row.hash);
    };
    if (remaining !== Infinity) {
      const size = Math.min(256, Math.floor(remaining / 2));
      remaining -= this.inWrite(() => {
        const start = cursor('orphan_sweep_cursor');
        let hashes = orphanPage(start, size);
        if (!hashes.length && start) hashes = orphanPage('', size);
        setCursor('orphan_sweep_cursor', hashes.at(-1) ?? '');
        return hashes.length;
      });
    } else {
      for (let after = ''; over();) {
        const hashes = this.inWrite(() => orphanPage(after, 256));
        if (hashes.length < 256) break;
        after = hashes.at(-1)!;
      }
    }

    // Event streams of unpinned calls, oldest first, from where the last pass stopped (a stripped call is not looked
    // at again); at the newest call the cursor starts over.
    let eventsCursor = cursor('events_sweep_cursor');
    while (over() && remaining > 0) {
      const count = this.inWrite(() => {
        const [at = '', id = ''] = eventsCursor ? JSON.parse(eventsCursor) as [string, string] : [];
        const rows = this.db.prepare(`SELECT call_id, started_at FROM calls WHERE events_hash IS NOT NULL AND ${unpinned}
          AND started_at >= ? AND (started_at > ? OR call_id > ?) ORDER BY started_at ASC, call_id ASC LIMIT ?`)
          .all(at, at, id, Math.min(EVICTION_CHUNK, remaining)) as { call_id: string; started_at: string }[];
        for (const row of rows) {
          if (!over()) break;
          const hashes = this.db.prepare("SELECT hash FROM call_blobs WHERE call_id = ? AND kind = 'events'").all(row.call_id) as { hash: string }[];
          this.db.prepare('UPDATE calls SET events_hash = NULL WHERE call_id = ?').run(row.call_id);
          this.db.prepare("DELETE FROM call_blobs WHERE call_id = ? AND kind = 'events'").run(row.call_id);
          this.collectHashes(hashes.map(item => item.hash));
          eventsCursor = JSON.stringify([row.started_at, row.call_id]);
        }
        if (!rows.length) eventsCursor = '';
        setCursor('events_sweep_cursor', eventsCursor);
        return rows.length;
      });
      if (!count) break;
      remaining -= count;
    }

    // Whole unpinned calls, oldest first.
    while (over() && remaining > 0) {
      const count = this.inWrite(() => {
        const rows = this.db.prepare(`SELECT call_id FROM calls WHERE ${unpinned} ORDER BY started_at ASC LIMIT ?`)
          .all(Math.min(EVICTION_CHUNK, remaining)) as { call_id: string }[];
        for (const row of rows) {
          if (!over()) break;
          const hashes = this.db.prepare('SELECT hash FROM call_blobs WHERE call_id = ?').all(row.call_id) as { hash: string }[];
          this.db.prepare('DELETE FROM calls WHERE call_id = ?').run(row.call_id);
          this.db.prepare('DELETE FROM call_blobs WHERE call_id = ?').run(row.call_id);
          this.collectHashes(hashes.map(item => item.hash));
        }
        return rows.length;
      });
      if (!count) break;
      remaining -= count;
    }
    this.vacuumFreePages();
    return Math.max(0, before - this.storeBytes());
  }

  /** With incremental auto-vacuum, give free database pages back to the medium, a bounded step per transaction. */
  private vacuumFreePages(): void {
    const mode = this.db.prepare('PRAGMA auto_vacuum').get() as { auto_vacuum: number };
    if (Number(mode.auto_vacuum) !== 2) return;
    for (let step = 0; step < 4096 && this.freeDatabaseBytes() > 0; step++) this.db.exec(`PRAGMA incremental_vacuum(${VACUUM_STEP_PAGES})`);
  }

  /**
   * Return the database's free pages to the medium. A database made before incremental auto-vacuum is converted first
   * (one VACUUM, which rewrites the database under the write lock: run it after `evict` has made the database small).
   * Returns the bytes the database files shrank by.
   */
  compact(): number {
    const before = this.medium.databaseBytes();
    const mode = this.db.prepare('PRAGMA auto_vacuum').get() as { auto_vacuum: number };
    if (Number(mode.auto_vacuum) !== 2) this.db.exec('PRAGMA auto_vacuum = INCREMENTAL; VACUUM;');
    this.vacuumFreePages();
    try { this.db.exec('PRAGMA wal_checkpoint(TRUNCATE)'); } catch { /* a reader holds the WAL; a later checkpoint truncates */ }
    return Math.max(0, before - this.medium.databaseBytes());
  }
  /** Delete affected blobs only when no call or compilation still references them. */
  private collectHashes(hashes: readonly string[]): number {
    const orphan = this.db.prepare(`SELECT hash, bytes FROM blobs WHERE hash = ?
      AND NOT EXISTS (SELECT 1 FROM call_blobs WHERE call_blobs.hash = blobs.hash)
      AND NOT EXISTS (SELECT 1 FROM compilation_files WHERE compilation_files.hash = blobs.hash)`);
    const remove = this.db.prepare('DELETE FROM blobs WHERE hash = ?');
    let freed = 0;
    for (const hash of new Set(hashes)) {
      const row = orphan.get(hash) as { hash: string; bytes: number } | undefined;
      if (!row) continue;
      this.medium.deleteBlob(hash);
      remove.run(hash);
      freed += row.bytes;
    }
    return freed;
  }

  // --- Compilations -----------------------------------------------------------------------------------------------

  /** Bumped whenever compilations or case tiers change, so runtimes know to reload. */
  version(): number {
    const row = this.db.prepare("SELECT value FROM meta WHERE key = 'compilations_version'").get() as { value: string } | undefined;
    return row ? Number(row.value) : 0;
  }
  private bump(): void {
    this.db.prepare(`INSERT INTO meta (key, value) VALUES ('compilations_version', '1')
      ON CONFLICT(key) DO UPDATE SET value = CAST(value AS INTEGER) + 1`).run();
  }

  /** The compilation in use for a definition revision, with its files and cases. */
  currentCompilation(definitionKey: string): (CompilationRow & { cases: CaseStats[] }) | undefined {
    const row = this.db.prepare("SELECT * FROM compilations WHERE definition_key = ? AND status = 'current' ORDER BY created_at DESC LIMIT 1")
      .get(definitionKey) as Omit<CompilationRow, 'files'> | undefined;
    return row ? this.withFiles(row) : undefined;
  }
  compilation(id: string): (CompilationRow & { cases: CaseStats[] }) | undefined {
    const row = this.db.prepare('SELECT * FROM compilations WHERE id = ?').get(id) as Omit<CompilationRow, 'files'> | undefined;
    return row ? this.withFiles(row) : undefined;
  }
  private withFiles(row: Omit<CompilationRow, 'files'>): CompilationRow & { cases: CaseStats[] } {
    const files = Object.fromEntries((this.db.prepare('SELECT path, hash FROM compilation_files WHERE compilation_id = ?').all(row.id) as
      { path: string; hash: string }[]).map(file => [file.path, this.blob(file.hash) ?? '']));
    return { ...row, files, cases: this.cases(row.id) };
  }
  cases(compilationId: string): CaseStats[] {
    return this.db.prepare('SELECT * FROM cases WHERE compilation_id = ? ORDER BY position').all(compilationId) as CaseStats[];
  }
  caseStats(hash: string): CaseStats | undefined {
    return this.db.prepare(`SELECT cases.* FROM cases JOIN compilations ON compilations.id = cases.compilation_id
      WHERE cases.hash = ? ORDER BY compilations.status = 'current' DESC, compilations.created_at DESC LIMIT 1`).get(hash) as CaseStats | undefined;
  }
  /** Compilations, newest first. */
  compilations(filter: { definition?: string; program?: string; status?: string; limit?: number } = {}): Omit<CompilationRow, 'files'>[] {
    const where: string[] = [], params: string[] = [];
    if (filter.definition) { where.push('(definition_key = ? OR definition_id = ? OR definition_name = ? OR definition_source = ?)'); params.push(filter.definition, filter.definition, filter.definition, filter.definition); }
    if (filter.program) { where.push('program_root = ?'); params.push(filter.program); }
    if (filter.status) { where.push('status = ?'); params.push(filter.status); }
    return this.db.prepare(`SELECT * FROM compilations ${where.length ? `WHERE ${where.join(' AND ')}` : ''}
      ORDER BY created_at DESC LIMIT ?`).all(...params, filter.limit ?? 100) as Omit<CompilationRow, 'files'>[];
  }

  /**
   * Store a compilation folder as the current one for its definition revision. Cases whose hash the previous
   * compilation already had keep their tier and counts; new cases start in shadow. Returns the compilation ID.
   */
  saveCompilation(input: { definitionKey: string; definitionId: string; definitionName: string; definitionSource: string | null;
    interfaceHash: string; programRoot: string | null; files: Record<string, string>; caseHashes: string[];
    links?: { caseHash: string; callId: string; role: CaseRole; verdict?: Verdict | null }[] }): string {
    const folderHash = hexDigest(JSON.stringify(Object.entries(input.files).sort(([a], [b]) => a.localeCompare(b))));
    const id = `${input.definitionKey.slice(0, 12)}-${folderHash.slice(0, 12)}`;
    const previous = this.currentCompilation(input.definitionKey);
    const created = now();
    this.db.exec('BEGIN IMMEDIATE');
    try {
      if (previous && previous.id !== id)
        this.db.prepare("UPDATE compilations SET status = 'superseded' WHERE definition_key = ? AND status = 'current'").run(input.definitionKey);
      this.db.prepare(`INSERT OR REPLACE INTO compilations (id, definition_key, definition_id, definition_name, definition_source,
        interface_hash, program_root, folder_hash, parent_id, created_at, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'current')`).run(
        id, input.definitionKey, input.definitionId, input.definitionName, input.definitionSource, input.interfaceHash, input.programRoot,
        folderHash, previous && previous.id !== id ? previous.id : previous?.parent_id ?? null, created);
      this.db.prepare('DELETE FROM compilation_files WHERE compilation_id = ?').run(id);
      for (const [path, text] of Object.entries(input.files))
        this.db.prepare('INSERT INTO compilation_files (compilation_id, path, hash) VALUES (?, ?, ?)').run(id, path, this.putBlob(text));
      const kept = new Map((previous?.cases ?? []).map(item => [item.hash, item]));
      this.db.prepare('DELETE FROM cases WHERE compilation_id = ?').run(id);
      input.caseHashes.forEach((hash, position) => {
        const old = kept.get(hash);
        this.db.prepare(`INSERT INTO cases (hash, compilation_id, position, tier, served, handed_off, compared, worse, better, audited,
          audit_worse, live_compared, live_worse, created_at, promoted_at, demoted_at, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
          hash, id, position, old?.tier ?? 'shadow', old?.served ?? 0, old?.handed_off ?? 0, old?.compared ?? 0, old?.worse ?? 0,
          old?.better ?? 0, old?.audited ?? 0, old?.audit_worse ?? 0, old?.live_compared ?? 0, old?.live_worse ?? 0, old?.created_at ?? created,
          old?.promoted_at ?? null, old?.demoted_at ?? null, old?.note ?? null);
      });
      for (const link of input.links ?? []) this.linkCase(link.caseHash, link.callId, link.role, link.verdict ?? null);
      this.db.prepare('DELETE FROM declines WHERE definition_key = ?').run(input.definitionKey);
      this.bump();
      this.db.exec('COMMIT');
    } catch (error) { this.db.exec('ROLLBACK'); throw error; }
    for (const hash of input.caseHashes) this.reviewTier(hash);
    return id;
  }
  setCompilationStatus(id: string, status: 'current' | 'superseded' | 'disabled'): void {
    this.db.prepare('UPDATE compilations SET status = ? WHERE id = ?').run(status, id); this.bump();
  }

  linkCase(caseHash: string, callId: string, role: CaseRole, verdict: Verdict | null = null): void {
    this.db.prepare(`INSERT INTO case_calls (case_hash, call_id, role, verdict, created_at) VALUES (?, ?, ?, ?, ?)
      ON CONFLICT(case_hash, call_id, role) DO UPDATE SET verdict = excluded.verdict`).run(caseHash, callId, role, verdict, now());
  }
  caseCalls(caseHash: string, role?: CaseRole, limit = 100): { call_id: string; role: CaseRole; verdict: Verdict | null; created_at: string }[] {
    return this.db.prepare(`SELECT call_id, role, verdict, created_at FROM case_calls WHERE case_hash = ? ${role ? 'AND role = ?' : ''}
      ORDER BY created_at DESC LIMIT ?`).all(...(role ? [caseHash, role, limit] : [caseHash, limit])) as
      { call_id: string; role: CaseRole; verdict: Verdict | null; created_at: string }[];
  }
  /** The cases a call is linked to (served by, learned from, audited by, ...). */
  callCases(callId: string): { case_hash: string; role: CaseRole; verdict: Verdict | null }[] {
    return this.db.prepare('SELECT case_hash, role, verdict FROM case_calls WHERE call_id = ?').all(callId) as
      { case_hash: string; role: CaseRole; verdict: Verdict | null }[];
  }

  /** Count a served call or a hand-off for a case, and review its tier. */
  caseServed(caseHash: string, callId: string, handedOff: boolean): void {
    this.db.prepare(`UPDATE cases SET ${handedOff ? 'handed_off = handed_off + 1' : 'served = served + 1'}
      WHERE hash = ? AND compilation_id IN (SELECT id FROM compilations WHERE status = 'current')`).run(caseHash);
    this.linkCase(caseHash, callId, handedOff ? 'handed-off' : 'served');
    if (handedOff) this.reviewTier(caseHash);
  }
  /** Record a comparison of a case against the agent (held-out replay, shadow, audit) and review its tier. */
  caseVerdict(caseHash: string, callId: string, role: 'held-out' | 'shadow' | 'audit', verdict: Verdict): void {
    const worse = verdict === 'worse' || verdict === 'diverged' ? 1 : 0, better = verdict === 'better' ? 1 : 0;
    const audit = role === 'audit' ? 1 : 0, live = role === 'held-out' ? 0 : 1;
    this.db.prepare(`UPDATE cases SET compared = compared + 1, worse = worse + ?, better = better + ?, audited = audited + ?,
      audit_worse = audit_worse + ?, live_compared = live_compared + ?, live_worse = live_worse + ?
      WHERE hash = ? AND compilation_id IN (SELECT id FROM compilations WHERE status = 'current')`)
      .run(worse, better, audit, audit * worse, live, live * worse, caseHash);
    this.linkCase(caseHash, callId, role, verdict);
    this.reviewTier(caseHash);
  }
  /** A case's evidence in the shape every promotion policy judges (calls/evidence.ts), with its latest comparisons for recency. */
  caseSummary(caseHash: string): EvidenceSummary | undefined {
    const stats = this.caseStats(caseHash);
    if (!stats) return;
    const recent = this.db.prepare(`SELECT verdict FROM case_calls WHERE case_hash = ? AND verdict IS NOT NULL AND role IN ('held-out', 'shadow', 'audit')
      ORDER BY created_at DESC LIMIT ?`).all(caseHash, RECENT) as { verdict: Verdict }[];
    return { subject: 'case', id: caseHash, state: stats.tier, rule: ruleOf(this.settings()),
      evidence: { served: stats.served, handed_off: stats.handed_off, guard_misses: 0, infrastructure: 0, compared: stats.compared, worse: stats.worse,
        better: stats.better, live_compared: stats.live_compared ?? 0, live_worse: stats.live_worse ?? 0, audited: stats.audited,
        audit_worse: stats.audit_worse, recent_compared: recent.length,
        recent_worse: recent.filter(row => row.verdict === 'worse' || row.verdict === 'diverged').length } };
  }
  /**
   * Promotion and demotion by evidence (§6.2), by the shared rule `crispPolicy` (calls/evidence.ts), applied as evidence
   * arrives. Under `promotionPolicy: 'nl'` the natural-language policy decides instead, from the specializer loop
   * (`reviewPromotions`, runtime/promotion.ts), so nothing happens here.
   */
  reviewTier(caseHash: string): CaseTier | undefined {
    const summary = this.caseSummary(caseHash);
    if (!summary) return;
    if (this.settings().promotionPolicy === 'nl') return summary.state as CaseTier;
    return this.decideTier(summary, crispPolicy(summary));
  }
  /** Apply a policy's decision about a case; `by` names the policy in the case's note. */
  decideTier(summary: EvidenceSummary, verdict: PromotionDecision, by?: string): CaseTier {
    const tier = applyDecision(summary.state, verdict.decision) as CaseTier;
    if (tier !== summary.state)
      this.setTier(summary.id, tier, `${tier === 'demoted' ? 'demoted' : 'promoted'} by ${by ? `${by} policy: ${verdict.reason}` : 'evidence'}`.slice(0, 300));
    return tier;
  }
  setTier(caseHash: string, tier: CaseTier, note?: string): void {
    const column = tier === 'active' ? ', promoted_at = ?' : tier === 'demoted' || tier === 'disabled' ? ', demoted_at = ?' : '';
    this.db.prepare(`UPDATE cases SET tier = ?, note = ?${column} WHERE hash = ? AND compilation_id IN
      (SELECT id FROM compilations WHERE status = 'current')`).run(...[tier, note ?? null, ...(column ? [now()] : []), caseHash]);
    this.bump();
  }

  // --- Declines ---------------------------------------------------------------------------------------------------

  decline(input: { definitionKey: string; definitionId: string; reason: DeclineReason; why: string; calls: number }): void {
    this.db.prepare(`INSERT OR REPLACE INTO declines (definition_key, definition_id, reason, why, calls_at_decline, created_at)
      VALUES (?, ?, ?, ?, ?, ?)`).run(input.definitionKey, input.definitionId, input.reason, input.why, input.calls, now());
  }
  declineFor(definitionKey: string): DeclineRow | undefined {
    return this.db.prepare('SELECT * FROM declines WHERE definition_key = ?').get(definitionKey) as DeclineRow | undefined;
  }
  declines(limit = 100): DeclineRow[] {
    return this.db.prepare('SELECT * FROM declines ORDER BY created_at DESC LIMIT ?').all(limit) as DeclineRow[];
  }

  // --- Findings: what compiling learned about the program or its executor (§5.3) ----------------------------------

  /** Record a finding; the same finding again (definition, kind, summary) updates it and counts it. */
  finding(input: FindingInput): void {
    const fingerprint = hexDigest(JSON.stringify([input.definitionKey, input.kind, input.summary])).slice(0, 32);
    const at = now();
    this.db.prepare(`INSERT INTO findings (fingerprint, definition_key, definition_id, definition_name, definition_source, kind, summary, detail,
      created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) ON CONFLICT(fingerprint) DO UPDATE SET detail = excluded.detail,
      seen = seen + 1, updated_at = excluded.updated_at`).run(fingerprint, input.definitionKey, input.definitionId, input.definitionName,
      input.definitionSource, input.kind, input.summary, JSON.stringify(input.detail ?? {}), at, at);
  }
  /** Findings, newest first: open ones unless `all`. */
  findings(filter: { definition?: string; all?: boolean; limit?: number } = {}): FindingRow[] {
    const where: string[] = [], params: (string | number)[] = [];
    if (!filter.all) where.push("status = 'open'");
    if (filter.definition) { where.push('(definition_key = ? OR definition_id = ? OR definition_name = ? OR definition_source = ?)'); params.push(...Array(4).fill(filter.definition)); }
    return (this.db.prepare(`SELECT * FROM findings ${where.length ? `WHERE ${where.join(' AND ')}` : ''} ORDER BY updated_at DESC LIMIT ?`)
      .all(...params, filter.limit ?? 100) as (Omit<FindingRow, 'detail'> & { detail: string })[]).map(row => ({ ...row, detail: JSON.parse(row.detail) }));
  }
  /** Mark a finding acknowledged (it stays, and comes back open only if it is found again with a new summary). */
  acknowledgeFinding(id: number): boolean {
    const row = this.db.prepare("UPDATE findings SET status = 'acknowledged' WHERE id = ? RETURNING id").get(id);
    return !!row;
  }

  // --- Spend and savings (what specializing costs, what serving saves) --------------------------------------------

  /** Tokens and time the specializer spent on a definition revision (writing, verifying, shadow replays, audits). */
  spent(definitionKey: string, activity: string, tokens: number, wallMs: number): void {
    if (tokens <= 0 && wallMs <= 0) return;
    this.db.prepare('INSERT INTO spend (definition_key, activity, tokens, wall_ms, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(definitionKey, activity, Math.round(tokens), Math.round(wallMs), now());
  }
  /**
   * Per definition revision with a compilation or spend: calls served by cases, what the agent costs per call (the
   * average of its successful calls), the tokens and time saved by serving, and what specializing has cost.
   */
  savings(filter: { definition?: string } = {}): SavingsRow[] {
    const keys = this.db.prepare(`SELECT DISTINCT definition_key FROM compilations UNION SELECT DISTINCT definition_key FROM spend`).all() as { definition_key: string }[];
    const rows: SavingsRow[] = [];
    for (const { definition_key: key } of keys) {
      const agent = this.db.prepare(`SELECT definition_name, definition_source, COUNT(*) AS n, AVG(COALESCE(tokens_in, 0) + COALESCE(tokens_out, 0)) AS tokens,
        AVG(COALESCE(wall_ms, 0)) AS wall FROM calls WHERE definition_key = ? AND executor = 'agent' AND outcome = 'done' AND audit_of IS NULL`).get(key) as
        { definition_name: string | null; definition_source: string | null; n: number; tokens: number | null; wall: number | null };
      const crisp = this.db.prepare(`SELECT COUNT(*) AS n, AVG(COALESCE(wall_ms, 0)) AS wall FROM calls WHERE definition_key = ? AND executor = 'crisp'`)
        .get(key) as { n: number; wall: number | null };
      const spend = this.db.prepare('SELECT COALESCE(SUM(tokens), 0) AS tokens, COALESCE(SUM(wall_ms), 0) AS wall FROM spend WHERE definition_key = ?')
        .get(key) as { tokens: number; wall: number };
      const name = agent.definition_name ?? (this.db.prepare('SELECT definition_name FROM compilations WHERE definition_key = ? LIMIT 1').get(key) as { definition_name: string } | undefined)?.definition_name ?? key;
      if (filter.definition && ![key, name, agent.definition_source].includes(filter.definition)) continue;
      const tokensPerCall = Number(agent.tokens ?? 0), wallPerCall = Number(agent.wall ?? 0);
      rows.push({ definition_key: key, definition_name: name, definition_source: agent.definition_source, agent_calls: Number(agent.n),
        served: Number(crisp.n), agent_tokens_per_call: Math.round(tokensPerCall), agent_ms_per_call: Math.round(wallPerCall),
        crisp_ms_per_call: Math.round(Number(crisp.wall ?? 0)), saved_tokens: Math.round(Number(crisp.n) * tokensPerCall),
        saved_ms: Math.round(Number(crisp.n) * Math.max(0, wallPerCall - Number(crisp.wall ?? 0))), spent_tokens: Number(spend.tokens), spent_ms: Number(spend.wall) });
    }
    return rows.sort((a, b) => (b.saved_tokens - b.spent_tokens) - (a.saved_tokens - a.spent_tokens));
  }

  // --- Offline jobs (shadow replays and audits) -------------------------------------------------------------------

  enqueue(kind: 'audit' | 'shadow', caseHash: string, callId: string): void {
    this.db.prepare(`INSERT OR IGNORE INTO jobs (kind, case_hash, call_id, status, enqueued_at) VALUES (?, ?, ?, 'pending', ?)`)
      .run(kind, caseHash, callId, now());
    this.pin(callId);
  }
  pendingJobs(limit = 50): AuditJob[] {
    return this.db.prepare("SELECT id, kind, case_hash, call_id, enqueued_at FROM jobs WHERE status = 'pending' ORDER BY id LIMIT ?")
      .all(limit) as AuditJob[];
  }
  finishJob(id: number, status: 'done' | 'skipped' | 'failed', verdict?: Verdict | null, detail?: string): void {
    this.db.prepare('UPDATE jobs SET status = ?, verdict = ?, detail = ?, done_at = ? WHERE id = ?')
      .run(status, verdict ?? null, detail ?? null, now(), id);
  }
  jobs(caseHash?: string, limit = 100): (AuditJob & { status: string; verdict: string | null; detail: string | null })[] {
    return this.db.prepare(`SELECT * FROM jobs ${caseHash ? 'WHERE case_hash = ?' : ''} ORDER BY id DESC LIMIT ?`)
      .all(...(caseHash ? [caseHash, limit] : [limit])) as (AuditJob & { status: string; verdict: string | null; detail: string | null })[];
  }

  /** iterateOn site statistics kept in the store (runtime/iterate.ts), shared by every runtime on the machine. */
  iterationStatistics(): IterationStatisticsStore {
    return this.statistics ??= {
      read: key => { const row = this.db.prepare('SELECT value FROM iteration_statistics WHERE key = ?').get(key) as { value: string } | undefined;
        return row ? JSON.parse(row.value) as SiteStatistics : undefined; },
      write: (key, value) => { this.db.prepare(`INSERT INTO iteration_statistics (key, value, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at`).run(key, JSON.stringify(value), now()); },
      reset: (prefix = '') => { this.db.prepare("DELETE FROM iteration_statistics WHERE key LIKE ? ESCAPE '\\'")
        .run(`${prefix.replace(/[\\%_]/g, character => `\\${character}`)}%`); },
      export: () => Object.fromEntries((this.db.prepare('SELECT key, value FROM iteration_statistics').all() as { key: string; value: string }[])
        .map(row => [row.key, JSON.parse(row.value) as SiteStatistics])),
    };
  }
  private statistics?: IterationStatisticsStore;

  /**
   * Refinement verdicts (`Is<T, P>`, native/refinement.ts) kept with the machine's records: P(true) by content key
   * (value hash, normalized predicate, judge id), so a value is not judged twice in any process. The table is created on
   * open like every other, so existing stores gain it without a version bump.
   */
  refinementVerdicts(): RefinementVerdictStore {
    return this.verdicts ??= {
      get: key => { const row = this.db.prepare('SELECT probability, judge FROM refinement_verdicts WHERE key = ?').get(key) as
        { probability: number; judge: string } | undefined; return row ? { probability: row.probability, judge: row.judge } : undefined; },
      set: (key, verdict) => { this.db.prepare(`INSERT INTO refinement_verdicts (key, probability, judge, created_at) VALUES (?, ?, ?, ?)
        ON CONFLICT(key) DO UPDATE SET probability = excluded.probability, judge = excluded.judge, created_at = excluded.created_at`)
        .run(key, verdict.probability, verdict.judge, now()); },
      count: () => (this.db.prepare('SELECT COUNT(*) AS n FROM refinement_verdicts').get() as { n: number }).n,
      clear: () => { this.db.prepare('DELETE FROM refinement_verdicts').run(); },
    };
  }
  private verdicts?: RefinementVerdictStore;

  /** Bytes the store's files take on the medium now: blob files as allocated and the database files, free pages included. */
  diskBytes(): number {
    let bytes = this.counter('blob_disk_bytes');
    try { bytes += this.medium.databaseBytes(); } catch { /* unknown */ }
    return bytes;
  }
}
