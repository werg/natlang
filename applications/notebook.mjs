/** Notebook cell store with explicit SQL/TypeScript engines and source revisions. */
import { createHash } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
import { TypeScriptEnvironment } from '../ts-host/dist/index.js';

const idPattern = /^[A-Za-z_][A-Za-z0-9_]*$/;
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');

export class NotebookWorkspace {
  constructor(cells, tables = {}) {
    this.cells = new Map();
    this.outputs = new Map();
    this.events = [];
    this.db = new DatabaseSync(':memory:');
    this.eval = new TypeScriptEnvironment({ mode: 'fresh', host: { notebook: {
      query: sql => this.query(sql),
    } } });
    this.revision = 0;
    for (const cell of cells) {
      if (!idPattern.test(cell.id) || this.cells.has(cell.id) ||
          !['sqlite', 'typescript-host'].includes(cell.engine) ||
          !Array.isArray(cell.needs) || typeof cell.source !== 'string')
        throw new Error('invalid notebook cell');
      this.cells.set(cell.id, { ...structuredClone(cell), revision: 0 });
    }
    for (const [name, rows] of Object.entries(tables)) this.loadTable(name, rows);
    this.db.exec('PRAGMA query_only=ON');
  }

  loadTable(name, rows) {
    if (!idPattern.test(name) || !Array.isArray(rows) || !rows.length) throw new Error('invalid table');
    const columns = Object.keys(rows[0]);
    if (!columns.length || columns.some(column => !idPattern.test(column)) ||
        rows.some(row => Object.keys(row).sort().join('|') !== [...columns].sort().join('|')))
      throw new Error('inconsistent table columns');
    const type = value => value === null ? 'TEXT' : typeof value === 'number' ? 'REAL' :
      typeof value === 'boolean' ? 'INTEGER' : 'TEXT';
    this.db.exec(`CREATE TABLE "${name}" (${columns.map(column =>
      `"${column}" ${type(rows.find(row => row[column] !== null)?.[column] ?? null)}`).join(', ')})`);
    const stmt = this.db.prepare(`INSERT INTO "${name}" (${columns.map(column => `"${column}"`).join(', ')}) VALUES (${columns.map(() => '?').join(', ')})`);
    for (const row of rows) stmt.run(...columns.map(column => row[column]));
    this.events.push({ operation: 'notebook.table', name, rows: rows.length, columns });
  }

  describe(goal) {
    if (!this.cells.has(goal)) throw new Error(`unknown cell: ${goal}`);
    return [...this.cells.values()].map(cell => ({ id: cell.id, needs: [...cell.needs],
      description: String(cell.description ?? ''), engine: cell.engine,
      revision: cell.revision }));
  }

  query(source) {
    if (typeof source !== 'string' || !/^\s*(SELECT|WITH)\b/i.test(source) || /;\s*\S/.test(source))
      throw new Error('SQL cell must be one read-only query');
    return this.db.prepare(source).all().map(row => Object.fromEntries(Object.entries(row)));
  }

  async execute(id) {
    const cell = this.cells.get(id);
    if (!cell) throw new Error(`unknown cell: ${id}`);
    const revision = cell.revision;
    const dependencies = {};
    for (const need of cell.needs) {
      const parent = this.cells.get(need), result = this.outputs.get(need);
      if (!parent || !result || result.status !== 'ok' || result.revision !== parent.revision)
        return this.failed(cell, `dependency is missing or stale: ${need}`);
      dependencies[need] = structuredClone(result.value);
    }
    try {
      let value;
      if (cell.engine === 'sqlite') {
        value = this.query(cell.source);
      } else {
        value = (await this.eval.executeAsync({ code: cell.source, body: true,
          path: `cell/${id}`, effectful: false,
          scope: { args: { deps: dependencies }, let: {} } })).result;
      }
      value = JSON.parse(JSON.stringify(value));
      if (cell.revision !== revision) return this.failed(cell, 'cell source changed during execution', 'stale');
      const output_sha256 = hash(value);
      const result = { id, status: 'ok', revision, output_sha256,
        sample: JSON.stringify(value).slice(0, 1000), detail: '', value };
      this.outputs.set(id, result);
      this.events.push({ operation: 'notebook.cell', id, engine: cell.engine,
        revision, status: 'ok', output_sha256 });
      const { value: _nativeValue, ...summary } = result;
      return summary;
    } catch (error) { return this.failed(cell, error instanceof Error ? error.message : String(error)); }
  }

  failed(cell, detail, status = 'failed') {
    this.events.push({ operation: 'notebook.cell', id: cell.id, engine: cell.engine,
      revision: cell.revision, status, detail });
    return { id: cell.id, status, revision: cell.revision, output_sha256: '', sample: '', detail };
  }

  edit(id, source) {
    const cell = this.cells.get(id);
    if (!cell || typeof source !== 'string') throw new Error('invalid cell edit');
    cell.source = source; cell.revision = ++this.revision;
    const affected = new Set([id]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const row of this.cells.values()) if (!affected.has(row.id) &&
          row.needs.some(need => affected.has(need))) { affected.add(row.id); changed = true; }
    }
    for (const name of affected) this.outputs.delete(name);
    this.events.push({ operation: 'notebook.edit', id, revision: cell.revision,
      invalidated: [...affected].sort() });
    return { revision: cell.revision, invalidated: [...affected].sort() };
  }

  drainEvents() { return this.events.splice(0); }
  close() { this.eval.close(); this.db.close(); }
}
