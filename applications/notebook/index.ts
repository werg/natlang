/**
 * A notebook of explicit cells: `sqlite` queries over an in-memory, read-only database and
 * `javascript` cells that receive only their declared dependencies as `deps`. Editing a cell bumps
 * its revision and invalidates its descendants. `runNotebook` walks the goal's dependency closure with
 * `iterateOn` (natlang picks among ready cells), then explains the bounded result samples.
 */
import { createHash } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import vm from 'node:vm';
import { iterateOn, type FolderHandle } from '@natlang/node';
import chooseGoal from './choose_goal.nl';
import chooseCell from './choose_cell.nl';
import explain from './explain.nl';
import type { Cell, CellResult, NotebookRun } from './types.js';

export type * from './types.js';
export type CellSource = { id: string, engine: Cell['engine'], needs: string[], source: string, description?: string };
export type Row = Record<string, unknown>;
export type NotebookConfig = { cells: CellSource[], tables?: Record<string, Row[]> };

const idPattern = /^[A-Za-z_][A-Za-z0-9_]*$/;
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
const CELL_TIMEOUT_MS = 2000;

export class NotebookWorkspace {
  readonly outputs = new Map<string, CellResult & { value: unknown }>();
  private readonly cells = new Map<string, CellSource & { revision: number }>();
  private readonly events: Record<string, unknown>[] = [];
  private readonly db = new DatabaseSync(':memory:');
  private readonly tableNames = new Set<string>();
  private revision = 0;
  private ready = false;

  constructor(cells: CellSource[], tables: Record<string, Row[]> = {}) {
    for (const cell of cells) this.putCell(cell, false);
    for (const [name, rows] of Object.entries(tables)) this.loadTable(name, rows);
    this.db.exec('PRAGMA query_only=ON');
    this.ready = true;
  }

  putCell(cell: CellSource, replacing = true): { id: string, revision: number, replaced: boolean } {
    if (!cell || !idPattern.test(cell.id) || !['sqlite', 'javascript'].includes(cell.engine) || !Array.isArray(cell.needs) ||
        cell.needs.some(need => !idPattern.test(need)) || typeof cell.source !== 'string') throw new Error('invalid notebook cell');
    const existing = this.cells.get(cell.id);
    if (existing && !replacing) throw new Error(`duplicate notebook cell: ${cell.id}`);
    const revision = existing ? ++this.revision : 0;
    this.cells.set(cell.id, { ...structuredClone(cell), revision });
    if (existing) this.events.push({ operation: 'notebook.edit', id: cell.id, revision, invalidated: this.invalidate(cell.id) });
    return { id: cell.id, revision, replaced: Boolean(existing) };
  }

  private invalidate(id: string): string[] {
    const affected = new Set([id]);
    for (let changed = true; changed;) {
      changed = false;
      for (const row of this.cells.values()) if (!affected.has(row.id) && row.needs.some(need => affected.has(need))) {
        affected.add(row.id); changed = true;
      }
    }
    for (const name of affected) this.outputs.delete(name);
    return [...affected].sort();
  }

  loadTable(name: string, rows: Row[]): void {
    if (!idPattern.test(name) || !Array.isArray(rows) || !rows.length) throw new Error('invalid table');
    const columns = Object.keys(rows[0]!);
    if (!columns.length || columns.some(column => !idPattern.test(column)) ||
        rows.some(row => Object.keys(row).sort().join('|') !== [...columns].sort().join('|')))
      throw new Error('inconsistent table columns');
    const type = (value: unknown) => value === null ? 'TEXT' : typeof value === 'number' ? 'REAL' : typeof value === 'boolean' ? 'INTEGER' : 'TEXT';
    if (this.ready) this.db.exec('PRAGMA query_only=OFF');
    try {
      if (this.tableNames.has(name)) this.db.exec(`DROP TABLE "${name}"`);
      this.db.exec(`CREATE TABLE "${name}" (${columns.map(column =>
        `"${column}" ${type(rows.find(row => row[column] !== null)?.[column] ?? null)}`).join(', ')})`);
      const insert = this.db.prepare(`INSERT INTO "${name}" (${columns.map(column => `"${column}"`).join(', ')}) VALUES (${columns.map(() => '?').join(', ')})`);
      for (const row of rows) insert.run(...columns.map(column => row[column] as never));
      this.tableNames.add(name);
      if (this.ready) this.outputs.clear();
    } finally { if (this.ready) this.db.exec('PRAGMA query_only=ON'); }
    this.events.push({ operation: 'notebook.table', name, rows: rows.length, columns });
  }

  catalog(): Cell[] {
    return [...this.cells.values()].map(cell => ({ id: cell.id, needs: [...cell.needs], description: String(cell.description ?? ''),
      engine: cell.engine, revision: cell.revision }));
  }

  query(source: string): Row[] {
    if (typeof source !== 'string' || !/^\s*(SELECT|WITH)\b/i.test(source) || /;\s*\S/.test(source))
      throw new Error('SQL cell must be one read-only query');
    return this.db.prepare(source).all().map(row => Object.fromEntries(Object.entries(row)));
  }

  /** Run one cell against the current outputs of its dependencies. */
  async execute(id: string): Promise<CellResult> {
    const cell = this.cells.get(id);
    if (!cell) throw new Error(`unknown cell: ${id}`);
    const revision = cell.revision, deps: Record<string, unknown> = {};
    for (const need of cell.needs) {
      const parent = this.cells.get(need), result = this.outputs.get(need);
      if (!parent || !result || result.status !== 'ok' || result.revision !== parent.revision)
        return this.failed(cell, `dependency is missing or stale: ${need}`);
      deps[need] = structuredClone(result.value);
    }
    try {
      const raw = cell.engine === 'sqlite' ? this.query(cell.source) :
        await vm.runInNewContext(`(async function (deps) {\n${cell.source}\n})(deps)`, { deps }, { timeout: CELL_TIMEOUT_MS });
      const value: unknown = JSON.parse(JSON.stringify(raw));
      if (cell.revision !== revision) return this.failed(cell, 'cell source changed during execution', 'stale');
      const output_sha256 = hash(value);
      const summary: CellResult = { id, status: 'ok', revision, output_sha256, sample: JSON.stringify(value).slice(0, 1000), detail: '' };
      this.outputs.set(id, { ...summary, value });
      this.events.push({ operation: 'notebook.cell', id, engine: cell.engine, revision, status: 'ok', output_sha256 });
      return summary;
    } catch (error) { return this.failed(cell, error instanceof Error ? error.message : String(error)); }
  }

  private failed(cell: CellSource & { revision: number }, detail: string, status: 'failed' | 'stale' = 'failed'): CellResult {
    this.events.push({ operation: 'notebook.cell', id: cell.id, engine: cell.engine, revision: cell.revision, status, detail });
    return { id: cell.id, status, revision: cell.revision, output_sha256: '', sample: '', detail };
  }

  edit(id: string, source: string): { revision: number, invalidated: string[] } {
    const cell = this.cells.get(id);
    if (!cell || typeof source !== 'string') throw new Error('invalid cell edit');
    cell.source = source; cell.revision = ++this.revision;
    const invalidated = this.invalidate(id);
    this.events.push({ operation: 'notebook.edit', id, revision: cell.revision, invalidated });
    return { revision: cell.revision, invalidated };
  }

  importConfig(config: NotebookConfig): { tables: string[], cells: { id: string, revision: number, replaced: boolean }[] } {
    if (!config || !Array.isArray(config.cells) || (config.tables !== undefined &&
        (!config.tables || typeof config.tables !== 'object' || Array.isArray(config.tables))))
      throw new Error('notebook config needs cells and optional tables');
    const tables = Object.entries(config.tables ?? {}).map(([name, rows]) => { this.loadTable(name, rows); return name; });
    return { tables, cells: config.cells.map(cell => this.putCell(cell)) };
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
  close(): void { this.db.close(); }
}

/** Cells of the goal's dependency closure that have not run and whose dependencies have. */
function readyCells(run: NotebookRun): Cell[] {
  const byId = new Map(run.cells.map(cell => [cell.id, cell]));
  const needed = new Set<string>();
  const pending = [run.goal];
  for (const id of pending) if (!needed.has(id)) { needed.add(id); pending.push(...byId.get(id)?.needs ?? []); }
  const done = new Set(run.order);
  return run.cells.filter(cell => needed.has(cell.id) && !done.has(cell.id) && cell.needs.every(parent => done.has(parent)));
}

/** Run one ready cell chosen by natlang. */
async function advance(notebook: NotebookWorkspace, run: NotebookRun, files?: FolderHandle): Promise<NotebookRun> {
  const ready = readyCells(run);
  const done = new Set(run.order);
  if (!ready.length) return { ...run, status: 'blocked', blocked: run.cells.filter(cell => !done.has(cell.id)).map(cell => cell.id).sort(),
    detail: 'No required cell is ready; check missing dependencies or a cycle.' };
  const chosen = await chooseCell(ready, run.goal, files);
  const cell = ready.find(row => row.id === chosen);
  if (!cell) return { ...run, status: 'invalid', detail: `cell is not ready: ${chosen}` };
  const result = await notebook.execute(cell.id);
  const results = [...run.results, result];
  if (result.revision !== cell.revision) return { ...run, results, status: 'stale', detail: `${cell.id}: source revision changed during this run` };
  if (result.status !== 'ok') return { ...run, results, status: result.status, detail: `${cell.id}: ${result.detail}` };
  return { ...run, order: [...run.order, cell.id], results, status: cell.id === run.goal ? 'done' : 'running', detail: '' };
}

/** Execute the goal's dependency graph, one natlang-chosen ready cell per step, and explain the result. */
export async function runNotebook(notebook: NotebookWorkspace, goal: string, question: string, files?: FolderHandle): Promise<NotebookRun> {
  const cells = notebook.catalog();
  const initial: NotebookRun = { goal, cells, order: [], results: [], blocked: [], answer: '', detail: '',
    status: cells.some(cell => cell.id === goal) ? 'running' : 'invalid' };
  if (initial.status === 'invalid') return { ...initial, detail: `unknown cell: ${goal}` };
  const finished = await iterateOn((run: NotebookRun) => advance(notebook, run, files), initial)
    .withLimit({ maxSteps: cells.length + 1 }).until(run => run.status !== 'running');
  return { ...finished, answer: await explain(question, finished, files) };
}

/** Answer a free-text request: pick the goal cell, then run and explain it. */
export async function answerRequest(notebook: NotebookWorkspace, request: string, files?: FolderHandle): Promise<NotebookRun> {
  return runNotebook(notebook, await chooseGoal(request, notebook.catalog()), request, files);
}

export const STARTER_NOTEBOOK: NotebookConfig = {
  tables: { orders: [{ region: 'north', amount: 12 }, { region: 'south', amount: 8 }, { region: 'north', amount: 5 }] },
  cells: [
    { id: 'regional_totals', engine: 'sqlite', needs: [], description: 'total order amount by region',
      source: 'SELECT region, SUM(amount) AS total FROM orders GROUP BY region ORDER BY region' },
    { id: 'largest_region', engine: 'javascript', needs: ['regional_totals'], description: 'identify the region with the largest total',
      source: 'return deps.regional_totals.toSorted((a, b) => b.total - a.total)[0];' },
  ],
};
