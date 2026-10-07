/**
 * A database you talk to in natural language: "keep track of our customers", "Ana from Lisbon signed up today",
 * "move 50 from Ana's account to Ben's", "which cities have more than two customers?". Two engines, one interface:
 *
 * - FolderDatabase (pure): the database is a folder in a documented format (FORMAT.md) and natural-language
 *   functions do the engine's work: schema design, transactions with constraint checks and index maintenance, query
 *   planning and execution. A transaction is one directory-reducer call, so its file changes exist only if it
 *   finishes; a failed check fails the call and nothing is kept.
 * - SqliteDatabase (optimized): a request compiles to SQL that SQLite runs exactly, as one transaction; conditions on
 *   meaning are judged once per stored value by a decision call and cached in the database.
 *
 * The host keeps only what must be exact: one writer at a time, a durable commit (the redo log is written before the
 * files, and an interrupted commit is finished on open), and read-only questions.
 */
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { DatabaseSync } from 'node:sqlite';
import { openFolder, saveFolder } from '@natlang/node';
import classify from './pure/classify.nl';
import clarify from './pure/clarify.nl';
import define from './pure/define.nl';
import transact from './pure/transact.nl';
import query from './pure/query.nl';
import translate from './sqlite/translate.nl';
import judge from './sqlite/judge.nl';
import type { Answer, Catalog, Kind, Report, SqlPlan } from './types.js';

export type Outcome =
  | { kind: 'question', answer: Answer }
  | { kind: 'change' | 'schema', report: Report }
  | { kind: 'unclear', clarification: string }
  | { kind: Kind, error: string };
/** Runs natural-language work in a task (e.g. `fn => runtime.run(fn)`). */
export type Run = <T>(fn: () => Promise<T>) => Promise<T>;
export interface Database { ask(request: string): Promise<Outcome>; close(): void }

const message = (error: unknown) => String((error as Error)?.message ?? error);
const FORMAT = ['../../FORMAT.md', '../../nldb/FORMAT.md'].map(path => fileURLToPath(new URL(path, import.meta.url))).find(path => existsSync(path))!;

/** One at a time: each task starts when the previous one has settled. */
function serial() {
  let tail: Promise<unknown> = Promise.resolve();
  return <T>(task: () => Promise<T>): Promise<T> => { const next = tail.then(task, task); tail = next.catch(() => {}); return next; };
}

function writeAtomically(path: string, content: string | Uint8Array) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(`${path}.tmp`, content);
  renameSync(`${path}.tmp`, path);
}

type LogRecord = { seq: number, at: string, request: string, kind: Kind, report: Report,
  files: { path: string, after: string | null }[] };

export class FolderDatabase implements Database {
  private readonly exclusive = serial();

  constructor(readonly path: string, private readonly run: Run) {
    if (!existsSync(join(path, 'catalog.json'))) {
      mkdirSync(join(path, 'tables'), { recursive: true });
      writeAtomically(join(path, 'catalog.json'), `${JSON.stringify({ version: 0, tables: {} }, null, 2)}\n`);
      writeAtomically(join(path, 'FORMAT.md'), readFileSync(FORMAT));
    }
    this.recover();
  }

  catalog(): Catalog { return JSON.parse(readFileSync(join(this.path, 'catalog.json'), 'utf8')) as Catalog; }

  async ask(request: string): Promise<Outcome> {
    const catalog = this.catalog();
    const kind: Kind = await this.run(() => classify(request, catalog));
    if (kind === 'unclear') return { kind, clarification: await this.run(() => clarify(request, catalog)) };
    if (kind === 'question') {
      // A direct reducer call: whatever the query writes is discarded.
      const folder = openFolder(this.path, 'overlay').root();
      try { return { kind, answer: await this.run(() => query(folder, request, catalog)) }; }
      catch (error) { return { kind, error: message(error) }; }
    }
    return this.exclusive(async () => {
      const current = this.catalog();
      const folder = openFolder(this.path, 'overlay').root();
      try {
        const report = await this.run(() => folder.apply(kind === 'schema' ? define : transact, request, current)) as Report;
        this.commit({ request, kind, report }, await folder.diff());
        return { kind, report };
      } catch (error) { return { kind, error: message(error) }; }
    });
  }

  /** The redo log record first (every changed file's new content), then the files themselves. */
  private commit(entry: Pick<LogRecord, 'request' | 'kind' | 'report'>, changes: Awaited<ReturnType<ReturnType<typeof openFolder>['diff']>>) {
    const kept = changes.changes.filter(change => !change.path.startsWith('log/') && change.path !== 'FORMAT.md');
    const decoder = new TextDecoder();
    const seq = this.lastSeq() + 1;
    const record: LogRecord = { seq, at: new Date().toISOString(), ...entry,
      files: kept.map(change => ({ path: change.path, after: change.after ? decoder.decode(change.after) : null })) };
    writeAtomically(join(this.path, 'log', `${String(seq).padStart(8, '0')}.json`), JSON.stringify(record));
    saveFolder(this.path, { changes: kept, moves: [] });
  }

  private lastSeq(): number {
    const names = existsSync(join(this.path, 'log')) ? readdirSync(join(this.path, 'log')).filter(name => /^\d+\.json$/.test(name)) : [];
    return names.length ? Math.max(...names.map(name => Number(name.slice(0, -5)))) : 0;
  }

  /** Finish a commit that was interrupted between its log record and its files. */
  private recover() {
    const seq = this.lastSeq();
    if (!seq) return;
    const record = JSON.parse(readFileSync(join(this.path, 'log', `${String(seq).padStart(8, '0')}.json`), 'utf8')) as LogRecord;
    for (const file of record.files) {
      const target = join(this.path, file.path);
      if (file.after === null) rmSync(target, { force: true });
      else if (!existsSync(target) || readFileSync(target, 'utf8') !== file.after) writeAtomically(target, file.after);
    }
  }

  close() {}
}

export class SqliteDatabase implements Database {
  private readonly db: DatabaseSync;
  private readonly exclusive = serial();
  private readonly judgments = new Map<string, number>();

  constructor(path: string, private readonly run: Run, private readonly today = () => new Date().toISOString().slice(0, 10)) {
    this.db = new DatabaseSync(path);
    this.db.exec('CREATE TABLE IF NOT EXISTS _nl_semantic (criterion TEXT NOT NULL, value TEXT NOT NULL, meets INTEGER NOT NULL, PRIMARY KEY (criterion, value))');
    for (const row of this.db.prepare('SELECT criterion, value, meets FROM _nl_semantic').all() as { criterion: string, value: string, meets: number }[]) {
      this.judgments.set(`${row.criterion}\u0000${row.value}`, row.meets);
    }
    // Conditions on meaning: judged ahead of the statement (judgeAll), looked up here.
    this.db.function('meets', { deterministic: true }, (value, criterion) =>
      value === null ? null : this.judgments.get(`${String(criterion)}\u0000${String(value)}`) ?? null);
  }

  schema(): string {
    return (this.db.prepare("SELECT sql FROM sqlite_master WHERE sql IS NOT NULL AND name NOT LIKE '\\_nl%' ESCAPE '\\' ORDER BY rowid").all() as { sql: string }[])
      .map(row => `${row.sql};`).join('\n');
  }

  async ask(request: string): Promise<Outcome> {
    return this.exclusive(async () => {
      let plan: SqlPlan;
      try { plan = await this.run(() => translate(request, this.schema(), this.today())); }
      catch (error) { return { kind: 'unclear' as Kind, error: message(error) }; }
      if (plan.kind === 'unclear') return { kind: 'unclear', clarification: plan.clarification };
      if (plan.kind === 'question' && (plan.statements.length !== 1 || !/^\s*(select|with)\b/i.test(plan.statements[0]!))) {
        return { kind: plan.kind, error: `a question must compile to one SELECT, not ${JSON.stringify(plan.statements)}` };
      }
      try { await this.judgeAll(plan.semantic); }
      catch (error) { return { kind: plan.kind, error: `judging ${JSON.stringify(plan.semantic)}: ${message(error)}` }; }
      this.db.exec('BEGIN');
      try {
        if (plan.kind === 'question') {
          const statement = this.db.prepare(plan.statements[0]!);
          const rows = statement.all() as Record<string, unknown>[];
          const columns = (statement as unknown as { columns?: () => { name: string }[] }).columns?.().map(column => column.name) ?? Object.keys(rows[0] ?? {});
          this.db.exec('ROLLBACK');
          return { kind: 'question', answer: { columns, rows: rows.map(row => columns.map(column => row[column] as string | number | null)),
            explanation: plan.explanation, assumptions: plan.assumptions } };
        }
        const changes: Report['changes'] = {};
        for (const sql of plan.statements) {
          const result = this.db.prepare(sql).run();
          const target = /^\s*(insert\s+(?:or\s+\w+\s+)?into|update(?:\s+or\s+\w+)?|delete\s+from)\s+["`[]?(\w+)/i.exec(sql);
          if (target) {
            const counts = changes[target[2]!] ??= { inserted: 0, updated: 0, deleted: 0 };
            const verb = target[1]!.toLowerCase();
            counts[verb.startsWith('insert') ? 'inserted' : verb.startsWith('update') ? 'updated' : 'deleted'] += Number(result.changes);
          }
        }
        this.db.exec('COMMIT');
        return { kind: plan.kind, report: { statements: plan.statements, changes, assumptions: plan.assumptions, summary: plan.explanation } };
      } catch (error) {
        this.db.exec('ROLLBACK');
        return { kind: plan.kind, error: message(error) };
      }
    });
  }

  /** Judge every stored value a semantic condition applies to that has no judgment yet; remember the verdicts. */
  private async judgeAll(semantic: SqlPlan['semantic']) {
    const quote = (name: string) => `"${name.replace(/"/g, '""')}"`;
    for (const { criterion, table, column } of semantic) {
      const values = (this.db.prepare(`SELECT DISTINCT ${quote(column)} AS value FROM ${quote(table)} WHERE ${quote(column)} IS NOT NULL`).all() as { value: unknown }[])
        .map(row => String(row.value)).filter(value => !this.judgments.has(`${criterion}\u0000${value}`));
      const verdicts = await Promise.all(values.map(value => this.run(() => judge(value, criterion))));
      const insert = this.db.prepare('INSERT OR REPLACE INTO _nl_semantic (criterion, value, meets) VALUES (?, ?, ?)');
      values.forEach((value, i) => {
        insert.run(criterion, value, verdicts[i] ? 1 : 0);
        this.judgments.set(`${criterion}\u0000${value}`, verdicts[i] ? 1 : 0);
      });
    }
  }

  close() { this.db.close(); }
}
