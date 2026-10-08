/**
 * The machine's call record store (§3.4): SQLite rows in WAL mode, so concurrent tasks and processes can write, and a
 * `blobs/` directory of SHA-256-named JSON files for values, event streams and compilation files. Compilations, their
 * cases and the links between cases and calls live in the same database (§7.2). Node only.
 */
import { createRequire } from 'node:module';
import { existsSync, mkdirSync, readFileSync, renameSync, rmSync, statSync, statfsSync, writeFileSync } from 'node:fs';
import { homedir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import type { DatabaseSync as Database } from 'node:sqlite';
import { hexDigest } from '../native/hash.js';
import type { IterationStatisticsStore, SiteStatistics } from '../runtime/iterate.js';
import { DEFAULT_SETTINGS, type CallRecord, type CallStoreSettings, type CaseRole, type CaseStats, type CaseTier,
  type CompilationRow, type DeclineReason, type DeclineRow, type ValueRef, type Verdict } from './types.js';

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

export type CallSummary = { call_id: string; parent_call_id: string | null; definition_id: string; definition_name: string;
  definition_key: string; definition_source: string | null; executor: string; case_hash: string | null; outcome: string;
  started_at: string; wall_ms: number; tokens_in: number; tokens_out: number; approach_hash: string | null;
  program_root: string | null; audit_of: string | null };
export type CallFilter = { definition?: string; key?: string; source?: string; outcome?: string; executor?: string;
  caseHash?: string; since?: string; program?: string; limit?: number; after?: string; audits?: boolean };
export type HotDefinition = { definition_key: string; definition_id: string; definition_name: string; definition_source: string | null;
  program_root: string | null; calls: number; agent_calls: number; crisp_calls: number; tokens: number; wall_ms: number; last_at: string };
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
CREATE TABLE IF NOT EXISTS iteration_statistics (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
`;

const now = () => new Date().toISOString();
const refHashes = (ref: ValueRef | null | undefined): string[] => ref?.complete ? [ref.hash] : [];

/** The machine's call record store. Open it with `CallStore.open(root)`; one instance per root per process. */
export class CallStore {
  private static readonly opened = new Map<string, CallStore>();
  static open(root: string): CallStore {
    const absolute = resolve(root);
    let store = CallStore.opened.get(absolute);
    if (!store) CallStore.opened.set(absolute, store = new CallStore(absolute));
    return store;
  }

  readonly db: Database;
  private writes = 0;
  private settingsCache?: { at: number; value: CallStoreSettings };

  private constructor(readonly root: string) {
    mkdirSync(join(root, 'blobs'), { recursive: true });
    const Sqlite = sqlite();
    this.db = new Sqlite(join(root, 'calls.sqlite'));
    this.db.exec('PRAGMA busy_timeout = 10000; PRAGMA journal_mode = WAL; PRAGMA synchronous = NORMAL;');
    this.db.exec(SCHEMA);
  }

  close(): void { CallStore.opened.delete(this.root); this.db.close(); }

  // --- Settings ---------------------------------------------------------------------------------------------------

  /** Machine settings: defaults, then `config.json` in the store, then `NATLANG_SPECIALIZATION`. Re-read every 5 s. */
  settings(): CallStoreSettings {
    const cached = this.settingsCache;
    if (cached && Date.now() - cached.at < 5000) return cached.value;
    let file: Partial<CallStoreSettings> = {};
    try { file = JSON.parse(readFileSync(join(this.root, 'config.json'), 'utf8')); } catch { /* defaults */ }
    const value = { ...DEFAULT_SETTINGS, ...file };
    const mode = process.env.NATLANG_SPECIALIZATION?.trim();
    if (mode === 'off' || mode === 'shadow' || mode === 'on') value.specialization = mode;
    this.settingsCache = { at: Date.now(), value };
    return value;
  }
  writeSettings(changes: Partial<CallStoreSettings>): CallStoreSettings {
    let file: Partial<CallStoreSettings> = {};
    try { file = JSON.parse(readFileSync(join(this.root, 'config.json'), 'utf8')); } catch { /* new */ }
    writeFileSync(join(this.root, 'config.json'), JSON.stringify({ ...file, ...changes }, null, 2) + '\n');
    this.settingsCache = undefined;
    return this.settings();
  }

  // --- Blobs ------------------------------------------------------------------------------------------------------

  private blobPath(hash: string): string { return join(this.root, 'blobs', hash.slice(0, 2), hash.slice(2)); }
  /** Store `text` under its SHA-256; returns the hash. Existing blobs are not rewritten. */
  putBlob(text: string, hash = hexDigest(text)): string {
    const path = this.blobPath(hash);
    if (!existsSync(path)) {
      mkdirSync(dirname(path), { recursive: true });
      const temporary = `${path}.${process.pid}.${Math.random().toString(36).slice(2)}`;
      writeFileSync(temporary, text);
      renameSync(temporary, path);
    }
    this.db.prepare('INSERT OR IGNORE INTO blobs (hash, bytes) VALUES (?, ?)').run(hash, Buffer.byteLength(text));
    return hash;
  }
  blob(hash: string): string | undefined {
    try { return readFileSync(this.blobPath(hash), 'utf8'); } catch { return undefined; }
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
    const kinds = new Map<string, string>();
    for (const [hash, text] of blobs) { this.putBlob(text, hash); kinds.set(hash, 'value'); }
    let eventsHash: string | null = null;
    if (events !== undefined) {
      eventsHash = this.putBlob(events);
      record = { ...record, events: { complete: true, hash: eventsHash, bytes: Buffer.byteLength(events) } };
    }
    const recordHash = this.putBlob(JSON.stringify(record));
    const definition = record.definition;
    this.db.prepare(`INSERT OR REPLACE INTO calls (call_id, parent_call_id, parent_action_index, task_id, program_id, build_hash,
      program_root, definition_id, definition_name, definition_source, definition_key, interface_hash, site, executor, model_id, case_hash,
      outcome, started_at, ended_at, wall_ms, tokens_in, tokens_out, model_requests, evals, approach_hash, audit_of, record_hash, events_hash)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
      record.call_id, record.parent_call_id, record.parent_action_index, record.task_id, record.program_id, record.build_hash,
      record.program_root, definition.id, definition.name, definition.source, definition.key, definition.interface, definition.site,
      record.executor.kind, record.executor.model_id, record.executor.case_hash ?? null, record.outcome, record.started_at, record.ended_at,
      record.cost.wall_ms, record.cost.tokens_in, record.cost.tokens_out, record.cost.model_requests, record.cost.evals,
      record.approach.hash, record.audit_of ?? null, recordHash, eventsHash);
    const link = this.db.prepare('INSERT OR IGNORE INTO call_blobs (call_id, hash, kind) VALUES (?, ?, ?)');
    link.run(record.call_id, recordHash, 'record');
    if (eventsHash) link.run(record.call_id, eventsHash, 'events');
    const referenced = [...Object.values(record.inputs), ...Object.values(record.captures), record.output,
      ...record.capture_writes.map(write => write.after), ...record.effects.flatMap(effect => [effect.args, effect.result]),
      ...(record.folder?.changes.map(change => change.after) ?? [])].flatMap(refHashes);
    for (const hash of new Set(referenced)) if (blobs.has(hash) || kinds.has(hash)) link.run(record.call_id, hash, 'value');
    if (++this.writes % 200 === 0) this.evict();
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
      SUM(COALESCE(wall_ms, 0)) AS wall_ms, MAX(started_at) AS last_at FROM calls WHERE ${where.join(' AND ')}
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
      case_hash, outcome, started_at, wall_ms, tokens_in, tokens_out, approach_hash, program_root, audit_of FROM calls
      WHERE parent_call_id = ? ORDER BY started_at ASC`).all(callId) as CallSummary[];
  }
  countCalls(definitionKey: string): number {
    return Number((this.db.prepare("SELECT COUNT(*) AS n FROM calls WHERE definition_key = ? AND audit_of IS NULL AND executor != 'crisp'")
      .get(definitionKey) as { n: number }).n);
  }
  annotate(callId: string, kind: string, value: unknown, source?: string): void {
    this.db.prepare('INSERT INTO annotations (call_id, kind, value, source, created_at) VALUES (?, ?, ?, ?, ?)')
      .run(callId, kind, JSON.stringify(value), source ?? null, now());
    this.pin(callId);
  }
  annotations(callId: string): { kind: string; value: unknown; source: string | null; created_at: string }[] {
    return (this.db.prepare('SELECT kind, value, source, created_at FROM annotations WHERE call_id = ? ORDER BY id').all(callId) as
      { kind: string; value: string; source: string | null; created_at: string }[]).map(row => ({ ...row, value: JSON.parse(row.value) }));
  }
  /** Keep a call (and its values) when the store evicts. */
  pin(callId: string, pinned = true): void { this.db.prepare('UPDATE calls SET pinned = ? WHERE call_id = ?').run(pinned ? 1 : 0, callId); }

  // --- Bounds -----------------------------------------------------------------------------------------------------

  totalBytes(): number { return Number((this.db.prepare('SELECT COALESCE(SUM(bytes), 0) AS n FROM blobs').get() as { n: number }).n); }

  /** The store's bound now: its size limit, lowered so the filesystem keeps `minFreeBytes` free. */
  bound(): number {
    const settings = this.settings();
    let free = Infinity;
    try { const stats = statfsSync(this.root); free = Number(stats.bavail) * Number(stats.bsize); } catch { /* unknown */ }
    const total = this.totalBytes();
    return Math.max(0, Math.min(settings.maxStoreBytes, free < settings.minFreeBytes ? total - (settings.minFreeBytes - free) : Infinity));
  }

  /**
   * Bring the store under its bound: first the event streams of unpinned calls, oldest first; then whole unpinned calls.
   * A call is pinned when it is marked so, annotated, or linked to a case. Returns the bytes freed.
   */
  evict(maxBytes = this.bound()): number {
    const before = this.totalBytes();
    if (before <= maxBytes) return 0;
    const target = Math.floor(maxBytes * 0.9);
    const unpinned = `pinned = 0 AND call_id NOT IN (SELECT call_id FROM case_calls)`;
    let total = before;
    const batch = (sql: string) => this.db.prepare(sql).all() as { call_id: string }[];
    for (const row of batch(`SELECT call_id FROM calls WHERE events_hash IS NOT NULL AND ${unpinned} ORDER BY started_at ASC LIMIT 5000`)) {
      if (total <= target) break;
      this.db.prepare('UPDATE calls SET events_hash = NULL WHERE call_id = ?').run(row.call_id);
      this.db.prepare("DELETE FROM call_blobs WHERE call_id = ? AND kind = 'events'").run(row.call_id);
      total = before - this.collect();
    }
    for (const row of batch(`SELECT call_id FROM calls WHERE ${unpinned} ORDER BY started_at ASC LIMIT 20000`)) {
      if (total <= target) break;
      this.db.prepare('DELETE FROM calls WHERE call_id = ?').run(row.call_id);
      this.db.prepare('DELETE FROM call_blobs WHERE call_id = ?').run(row.call_id);
      total = this.totalBytes() - this.collect();
    }
    return before - this.totalBytes();
  }
  /** Delete blobs nothing references; returns their bytes. */
  private collect(): number {
    const orphans = this.db.prepare(`SELECT hash, bytes FROM blobs WHERE hash NOT IN (SELECT hash FROM call_blobs)
      AND hash NOT IN (SELECT hash FROM compilation_files) LIMIT 10000`).all() as { hash: string; bytes: number }[];
    let freed = 0;
    for (const { hash, bytes } of orphans) {
      rmSync(this.blobPath(hash), { force: true });
      this.db.prepare('DELETE FROM blobs WHERE hash = ?').run(hash);
      freed += bytes;
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
          audit_worse, created_at, promoted_at, demoted_at, note) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`).run(
          hash, id, position, old?.tier ?? 'shadow', old?.served ?? 0, old?.handed_off ?? 0, old?.compared ?? 0, old?.worse ?? 0,
          old?.better ?? 0, old?.audited ?? 0, old?.audit_worse ?? 0, old?.created_at ?? created, old?.promoted_at ?? null,
          old?.demoted_at ?? null, old?.note ?? null);
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
    const audit = role === 'audit' ? 1 : 0;
    this.db.prepare(`UPDATE cases SET compared = compared + 1, worse = worse + ?, better = better + ?, audited = audited + ?,
      audit_worse = audit_worse + ? WHERE hash = ? AND compilation_id IN (SELECT id FROM compilations WHERE status = 'current')`)
      .run(worse, better, audit, audit * worse, caseHash);
    this.linkCase(caseHash, callId, role, verdict);
    this.reviewTier(caseHash);
  }
  /**
   * Promotion and demotion by evidence (§6.2): a shadow case becomes active after `promotionComparisons` comparisons
   * with at most `acceptanceBound` of them worse; an active case is demoted when worse audits or hand-offs exceed it.
   */
  reviewTier(caseHash: string): CaseTier | undefined {
    const stats = this.caseStats(caseHash);
    if (!stats) return;
    const { acceptanceBound: bound, promotionComparisons: needed } = this.settings();
    let tier = stats.tier;
    if (tier === 'shadow' && stats.compared >= needed && stats.worse <= bound * stats.compared) tier = 'active';
    else if (tier === 'active' && ((stats.audited >= 5 && stats.audit_worse > bound * stats.audited) ||
      (stats.served + stats.handed_off >= 10 && stats.handed_off > bound * (stats.served + stats.handed_off)) ||
      (stats.compared >= needed && stats.worse > bound * stats.compared))) tier = 'demoted';
    if (tier !== stats.tier) this.setTier(caseHash, tier, tier === 'demoted' ? 'demoted by evidence' : 'promoted by evidence');
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

  /** Bytes on disk of the database and blobs (for `natlang traces status`). */
  diskBytes(): number {
    let bytes = this.totalBytes();
    for (const name of ['calls.sqlite', 'calls.sqlite-wal']) try { bytes += statSync(join(this.root, name)).size; } catch { /* absent */ }
    return bytes;
  }
}
