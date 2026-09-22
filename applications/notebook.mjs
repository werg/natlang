/** Notebook cell store with explicit SQL/TypeScript engines and source revisions. */
import { createHash } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';

const idPattern = /^[A-Za-z_][A-Za-z0-9_]*$/;
const hash = value => createHash('sha256').update(JSON.stringify(value)).digest('hex');

export class NotebookWorkspace {
  constructor(cells, tables = {}, { environment } = {}) {
    this.cells = new Map();
    this.outputs = new Map();
    this.events = [];
    this.db = new DatabaseSync(':memory:');
    this.eval = environment ?? null;
    this.revision = 0;
    this.ready = false; this.tableNames = new Set();
    for (const cell of cells) this.putCell(cell, false);
    for (const [name, rows] of Object.entries(tables)) this.loadTable(name, rows);
    this.db.exec('PRAGMA query_only=ON');
    this.ready = true;
  }

  putCell(cell, replacing = true) {
    if (!cell || !idPattern.test(cell.id) || !['sqlite', 'typescript-host'].includes(cell.engine) ||
        !Array.isArray(cell.needs) || cell.needs.some(need => !idPattern.test(need)) ||
        typeof cell.source !== 'string') throw new Error('invalid notebook cell');
    const existing = this.cells.get(cell.id);
    if (existing && !replacing) throw new Error(`duplicate notebook cell: ${cell.id}`);
    const revision = existing ? ++this.revision : 0;
    this.cells.set(cell.id, { ...structuredClone(cell), revision });
    if (existing) {
      const affected = this.invalidate(cell.id);
      this.events.push({ operation: 'notebook.edit', id: cell.id, revision, invalidated: affected });
    }
    return { id: cell.id, revision, replaced: Boolean(existing) };
  }

  invalidate(id) {
    const affected = new Set([id]); let changed = true;
    while (changed) {
      changed = false;
      for (const row of this.cells.values()) if (!affected.has(row.id) &&
          row.needs.some(need => affected.has(need))) { affected.add(row.id); changed = true; }
    }
    for (const name of affected) this.outputs.delete(name);
    return [...affected].sort();
  }

  loadTable(name, rows) {
    if (!idPattern.test(name) || !Array.isArray(rows) || !rows.length) throw new Error('invalid table');
    const columns = Object.keys(rows[0]);
    if (!columns.length || columns.some(column => !idPattern.test(column)) ||
        rows.some(row => Object.keys(row).sort().join('|') !== [...columns].sort().join('|')))
      throw new Error('inconsistent table columns');
    const type = value => value === null ? 'TEXT' : typeof value === 'number' ? 'REAL' :
      typeof value === 'boolean' ? 'INTEGER' : 'TEXT';
    if (this.ready) this.db.exec('PRAGMA query_only=OFF');
    try {
      if (this.tableNames.has(name)) this.db.exec(`DROP TABLE "${name}"`);
      this.db.exec(`CREATE TABLE "${name}" (${columns.map(column =>
        `"${column}" ${type(rows.find(row => row[column] !== null)?.[column] ?? null)}`).join(', ')})`);
      const stmt = this.db.prepare(`INSERT INTO "${name}" (${columns.map(column => `"${column}"`).join(', ')}) VALUES (${columns.map(() => '?').join(', ')})`);
      for (const row of rows) stmt.run(...columns.map(column => row[column]));
      this.tableNames.add(name);
      if (this.ready) this.outputs.clear();
    } finally { if (this.ready) this.db.exec('PRAGMA query_only=ON'); }
    this.events.push({ operation: 'notebook.table', name, rows: rows.length, columns });
  }

  describe(goal) {
    if (!this.cells.has(goal)) throw new Error(`unknown cell: ${goal}`);
    return [...this.cells.values()].map(cell => ({ id: cell.id, needs: [...cell.needs],
      description: String(cell.description ?? ''), engine: cell.engine,
      revision: cell.revision }));
  }

  catalog() {
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
        if (!this.eval) throw new Error('typescript-host cell needs an evaluator supplied by the host');
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
    const affected = this.invalidate(id);
    this.events.push({ operation: 'notebook.edit', id, revision: cell.revision,
      invalidated: affected });
    return { revision: cell.revision, invalidated: affected };
  }

  importConfig(config) {
    if (!config || !Array.isArray(config.cells) || (config.tables !== undefined &&
        (!config.tables || typeof config.tables !== 'object' || Array.isArray(config.tables))))
      throw new Error('notebook config needs cells and optional tables');
    const tables = Object.entries(config.tables ?? {}).map(([name, rows]) => { this.loadTable(name, rows); return name; });
    const cells = config.cells.map(cell => this.putCell(cell));
    return { tables, cells };
  }

  drainEvents() { return this.events.splice(0); }
  close() { this.eval?.close(); this.db.close(); }
}
