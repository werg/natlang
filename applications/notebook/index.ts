/**
 * A notebook of explicit cells: `sqlite` queries over an in-memory, read-only database and
 * `javascript` cells that receive only their declared dependencies as `deps`. Editing a cell bumps
 * its revision and invalidates its descendants. `runNotebook` walks the goal's dependency closure with
 * `iterateOn` (a structural measure ends the loop; the order among ready cells is a pluggable with a crisp default),
 * then explains the bounded result samples from host-computed evidence.
 */
import { createHash } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import vm from 'node:vm';
import { builtin, iterateOn, pluggable, untrusted, type FolderHandle, type PluggableSetting, type Untrusted } from '@natlang/node';
import chooseGoal from './choose_goal.nl';
import nextCell from './nextCell.nl';
import answerFromCells from './answerFromCells.nl';
import type { Cell, CellEvidence, CellResult, NotebookRun } from './types.js';

export type * from './types.js';
export type CellSource = { id: string, engine: Cell['engine'], needs: string[], source: string, description?: string };
export type Row = Record<string, unknown>;
export type NotebookConfig = { cells: CellSource[], tables?: Record<string, Row[]> };

const idPattern = /^[A-Za-z_][A-Za-z0-9_]*$/;
const hash = (value: unknown) => createHash('sha256').update(JSON.stringify(value)).digest('hex');
/** Default sandbox bound for one JavaScript cell, milliseconds; settable per workspace (`cellTimeoutMs`). */
export const CELL_TIMEOUT_MS = 2000;
/** The length of the sample kept with a cell result. */
const SAMPLE_CHARS = 1000;

export class NotebookWorkspace {
  readonly outputs = new Map<string, CellResult & { value: unknown }>();
  private readonly cells = new Map<string, CellSource & { revision: number }>();
  private readonly events: Record<string, unknown>[] = [];
  private readonly db = new DatabaseSync(':memory:');
  private readonly tableNames = new Set<string>();
  private revision = 0;
  private ready = false;

  private readonly cellTimeoutMs: number;

  constructor(cells: CellSource[], tables: Record<string, Row[]> = {}, { cellTimeoutMs = CELL_TIMEOUT_MS } = {}) {
    this.cellTimeoutMs = cellTimeoutMs;
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
        await vm.runInNewContext(`(async function (deps) {\n${cell.source}\n})(deps)`, { deps }, { timeout: this.cellTimeoutMs });
      const value: unknown = JSON.parse(JSON.stringify(raw));
      if (cell.revision !== revision) return this.failed(cell, 'cell source changed during execution', 'stale');
      const output_sha256 = hash(value);
      const summary: CellResult = { id, status: 'ok', revision, output_sha256, sample: JSON.stringify(value).slice(0, SAMPLE_CHARS), detail: '' };
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

/** The cells the goal needs: the goal and, transitively, everything it depends on. Ids that name no cell are included. */
function requiredIds(run: NotebookRun): Set<string> {
  const byId = new Map(run.cells.map(cell => [cell.id, cell]));
  const needed = new Set<string>();
  const pending = [run.goal];
  for (const id of pending) if (!needed.has(id)) { needed.add(id); pending.push(...byId.get(id)?.needs ?? []); }
  return needed;
}

/** Cells of the goal's dependency closure that have not run and whose dependencies have. */
export function readyCells(run: NotebookRun): Cell[] {
  const needed = requiredIds(run), done = new Set(run.order);
  return run.cells.filter(cell => needed.has(cell.id) && !done.has(cell.id) && cell.needs.every(parent => done.has(parent)));
}

/** The loop's measure: required cells not yet run. Each ok step runs one of them, so it falls by one; it is 0 once the run has stopped. */
export function remainingCells(run: NotebookRun): number {
  if (run.status !== 'running') return 0;
  const done = new Set(run.order);
  return [...requiredIds(run)].filter(id => !done.has(id)).length;
}

/** Ids of the cells that no other cell needs: the candidates for the goal. */
export function finalCells(cells: Cell[]): string[] {
  const needed = new Set(cells.flatMap(cell => cell.needs));
  return cells.filter(cell => !needed.has(cell.id)).map(cell => cell.id);
}

/** Which cell of `ready` runs first, with the pluggable's crisp default: the lowest id. */
export const lowestId = (ready: Cell[]): string => ready.map(cell => cell.id).reduce((low, id) => id < low ? id : low);

export type NotebookOptions = {
  /** How the next ready cell is chosen: `crisp` (lowest id, the default), `nl` (the model) or `shadow` (both run; the crisp choice is served and agreement is traced). */
  nextCellMode?: PluggableSetting;
};

/** An answer that is an id of `ids`, or the reason it is not. */
const idProblem = (answer: unknown, ids: string[], what: string): string | null =>
  typeof answer === 'string' && ids.includes(answer) ? null : `${JSON.stringify(answer)} is not one of the ${what}: ${ids.join(', ')}`;

/** Ask `choose` for an id of `ids`; on a wrong answer ask once more with the problem. Returns the answer and any problem left. */
async function chooseId(choose: (problem?: string) => Promise<string>, ids: string[], what: string): Promise<{ id: string, problem: string | null }> {
  const first = await choose();
  const problem = idProblem(first, ids, what);
  if (!problem) return { id: first, problem };
  const second = await choose(problem);
  return { id: second, problem: idProblem(second, ids, what) };
}

/** Run one ready cell. The choice among ready cells does not change the answer, so it is a pluggable with a crisp default. */
async function advance(notebook: NotebookWorkspace, run: NotebookRun, options: NotebookOptions): Promise<NotebookRun> {
  const ready = readyCells(run);
  const done = new Set(run.order);
  if (!ready.length) return { ...run, status: 'blocked', blocked: run.cells.filter(cell => !done.has(cell.id)).map(cell => cell.id).sort(),
    detail: 'No required cell is ready; check missing dependencies or a cycle.' };
  const ids = ready.map(cell => cell.id);
  const pick = pluggable({
    crisp: async (): Promise<string> => lowestId(ready),
    nl: async (): Promise<string> => (await chooseId(problem => (nextCell as (...args: unknown[]) => Promise<string>)(ready, run.goal, ...(problem === undefined ? [] : [problem])), ids, 'ready cells')).id,
  }, options.nextCellMode, { default: 'crisp', name: 'notebook.nextCell', serve: 'crisp' });
  const chosen = await pick();
  const cell = ready.find(row => row.id === chosen);
  if (!cell) return { ...run, status: 'invalid', detail: `${idProblem(chosen, ids, 'ready cells')}` };
  const result = await notebook.execute(cell.id);
  const results = [...run.results, result];
  if (result.revision !== cell.revision) return { ...run, results, status: 'stale', detail: `${cell.id}: source revision changed during this run` };
  if (result.status !== 'ok') return { ...run, results, status: result.status, detail: `${cell.id}: ${result.detail}` };
  return { ...run, order: [...run.order, cell.id], results, status: cell.id === run.goal ? 'done' : 'running', detail: '' };
}

const containsNull = (value: unknown): boolean => value === null || (typeof value === 'object' && Object.values(value as object).some(containsNull));
const isEmpty = (value: unknown): boolean => value === '' || Array.isArray(value) && value.length === 0 ||
  typeof value === 'object' && value !== null && !Array.isArray(value) && Object.keys(value).length === 0;

/** The facts the explanation reads about each cell that ran: computed from the stored results, not by the model. */
export function evidenceOf(notebook: NotebookWorkspace, run: NotebookRun): CellEvidence[] {
  return run.results.map(result => {
    const stored = notebook.outputs.get(result.id);
    const known = stored !== undefined && stored.status === 'ok' && stored.revision === result.revision;
    return { id: result.id, revision: result.revision, status: result.status, sample: untrusted(result.sample, `cell ${result.id}`),
      has_null: known && containsNull(stored.value), empty: known && isEmpty(stored.value),
      truncated: known && JSON.stringify(stored.value).length > SAMPLE_CHARS, detail: result.detail };
  });
}

/** What keeps the answer from being complete, in words the explanation states: failed, stale, blocked or invalid parts of the run. */
export function limitsOf(run: NotebookRun): string[] {
  const limits = run.results.filter(result => result.status !== 'ok')
    .map(result => `cell ${result.id} ${result.status === 'stale' ? 'became stale' : 'failed'}${result.detail ? ` (${result.detail})` : ''}`);
  if (run.status === 'blocked') limits.push(`cells ${run.blocked.join(', ')} did not run (${run.detail})`);
  else if (run.status === 'invalid') limits.push(run.detail);
  else if (run.status !== 'done' && !limits.length) limits.push(`the run stopped as ${run.status}${run.detail ? ` (${run.detail})` : ''}`);
  return limits;
}

/** The citation rule: every `id@revision` in the answer names a cell of the evidence at its revision, and an answer built on an ok cell cites one. */
export function checkCitations(answer: string, evidence: CellEvidence[]): string | null {
  const cited = [...answer.matchAll(/\b([A-Za-z_][A-Za-z0-9_]*)@(\d+)\b/g)];
  const wrong = cited.filter(([, id, revision]) => !evidence.some(row => row.id === id && row.revision === Number(revision)));
  if (wrong.length) return `${wrong.map(match => match[0]).join(', ')} ${wrong.length === 1 ? 'does' : 'do'} not name a cell and revision that ran; the cells that ran are ${evidence.map(row => `${row.id}@${row.revision}`).join(', ')}`;
  if (!cited.length && evidence.some(row => row.status === 'ok')) return `the answer cites no cell; cite the cells it uses as id@revision, from ${evidence.map(row => `${row.id}@${row.revision}`).join(', ')}`;
  return null;
}

/** Explain the run: the host computes the evidence and the limits, natlang answers, the host checks the citations (one retry with the problem). */
async function explain(notebook: NotebookWorkspace, question: string, run: NotebookRun, files?: FolderHandle): Promise<{ answer: string, problem: string | null }> {
  const evidence = evidenceOf(notebook, run), limits = limitsOf(run);
  const note = files ? await builtin('readNote')(question, files) as Untrusted<string> : untrusted('', 'note');
  const ask = (problem?: string) => (answerFromCells as (...args: unknown[]) => Promise<string>)(question, evidence, note, limits, ...(problem === undefined ? [] : [problem]));
  let answer = await ask();
  let problem = checkCitations(answer, evidence);
  if (problem) { answer = await ask(problem); problem = checkCitations(answer, evidence); }
  return { answer, problem };
}

/** Execute the goal's dependency graph one ready cell per step, until the required cells have run or a step stops the run, and explain the result. */
export async function runNotebook(notebook: NotebookWorkspace, goal: string, question: string, files?: FolderHandle, options: NotebookOptions = {}): Promise<NotebookRun> {
  const cells = notebook.catalog();
  const initial: NotebookRun = { goal, cells, order: [], results: [], blocked: [], answer: '', detail: '',
    status: cells.some(cell => cell.id === goal) ? 'running' : 'invalid' };
  if (initial.status === 'invalid') return { ...initial, detail: `unknown cell: ${goal}` };
  const finished = await iterateOn((run: NotebookRun) => advance(notebook, run, options), initial)
    .withMeasure(remainingCells).until(run => run.status !== 'running');
  const { answer, problem } = await explain(notebook, question, finished, files);
  return { ...finished, answer, detail: [finished.detail, problem && `citation check: ${problem}`].filter(Boolean).join('; ') };
}

/** Answer a free-text request: pick the goal cell, then run and explain it. */
export async function answerRequest(notebook: NotebookWorkspace, request: string, files?: FolderHandle, options: NotebookOptions = {}): Promise<NotebookRun> {
  const cells = notebook.catalog(), finals = finalCells(cells);
  const goal = await chooseId(problem => (chooseGoal as (...args: unknown[]) => Promise<string>)(request, cells, finals, ...(problem === undefined ? [] : [problem])),
    cells.map(cell => cell.id), 'cells');
  return runNotebook(notebook, goal.id, request, files, options);
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
