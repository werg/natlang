import assert from 'node:assert/strict';
import { test } from 'node:test';
import { existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createNatlangRuntime, MemoryNeuraleseStore } from '../dist/index.js';
import { FolderDatabase, SqliteDatabase } from '../../applications/dist/nldb/index.js';
import { judgeStep } from '../../applications/dist/nldb/main.js';
import { FolderBlockStore, IvfIndex, decodeBlock, encodeBlock, pooled } from '../../applications/dist/nldb/neuralese.js';
import { scriptedModel } from './support/natlang.mjs';

const CATALOG = { version: 1, tables: { customers: { description: 'A customer of the shop.', key: ['_id'], indexes: ['city'],
  pages: 1, rows: 0, nextId: 1, columns: [
    { name: 'name', type: 'text', description: 'Full name.', nullable: false, unique: true, references: null, check: null },
    { name: 'city', type: 'text', description: 'Where they live.', nullable: false, unique: false, references: null, check: null }] } } };

/**
 * The pure engine's stages, scripted: each reply is the eval code a model might write for that stage's instructions.
 * Decisions (classify, meets) are scored from what the request shows.
 */
function folderModel() {
  const seen = [];
  const model = scriptedModel(opening => {
    const field = name => JSON.parse(new RegExp(`${name}: string = ("(?:[^"\\\\]|\\\\.)*")`).exec(opening)?.[1] ?? '""');
    const stage = [['You are the server of the database', 'database'], ['Change the schema of the database in folder', 'define'],
      ['design the changes it asks for', 'design'], ['Read request, a kind, against catalog', 'parse'], ['into an equivalent plan', 'plan'],
      ['Run statement on the database in folder', 'execute'], ["Read table's pages in folder", 'scan'], ['Find the\nmatching pairs by binary search', 'lookup'],
      ['Group rows by the values', 'aggregate'], ['make a row holding each output', 'project'], ["Apply changes to table's pages", 'write'],
      ['For each indexed column of table', 'reindex'], ["Check table's changed rows", 'check'], ['Ask the one short question', 'clarify']]
      .find(([marker]) => opening.includes(marker))?.[1];
    seen.push(stage);
    if (stage === 'database') return `
      const catalog = await folder.file('catalog.json').readJson();
      const d = await decide(classify, request, catalog);
      if (d.value === 'unclear' || d.confidence < 0.5) return { kind: 'unclear', clarification: await clarify(request, catalog) };
      if (d.value === 'schema') return { kind: 'schema', report: await folder.apply(define, request, catalog) };
      try {
        const statement = await plan(await parse(request, d.value, catalog, today), catalog);
        if (statement.kind === 'question') {
          const run = await execute(folder, statement);
          return { kind: 'question', answer: { columns: statement.columns, rows: run.rows, explanation: statement.explanation, assumptions: statement.assumptions } };
        }
        const run = await folder.apply(execute, statement);
        return { kind: 'change', report: { statements: statement.sql, changes: run.changes, assumptions: statement.assumptions, summary: 'Done.' } };
      } catch (error) { return { kind: d.value, error: String(error.message ?? error) }; }`;
    if (stage === 'clarify') return 'return "How old is old: signed up before which date?";';
    if (stage === 'design') return `return ${JSON.stringify({ changes: [{ op: 'create table', table: 'customers', description: CATALOG.tables.customers.description,
      columns: CATALOG.tables.customers.columns, key: ['_id'], indexes: ['city'] }], ddl: ['CREATE TABLE customers (name TEXT NOT NULL UNIQUE, city TEXT NOT NULL)'], assumptions: [] })};`;
    if (stage === 'define') return `
      const planned = await design(request, catalog);
      const next = JSON.parse(JSON.stringify(catalog));
      for (const change of planned.changes) {
        next.tables[change.table] = { description: change.description, columns: change.columns, key: change.key, indexes: change.indexes, pages: 1, rows: 0, nextId: 1 };
        await folder.file('tables/' + change.table + '/0.jsonl').writeText('');
        for (const column of change.indexes) await folder.file('indexes/' + change.table + '/' + column + '.json').writeText('[]');
      }
      next.version += 1;
      await folder.file('catalog.json').writeJson(next);
      return { statements: planned.ddl, changes: {}, assumptions: planned.assumptions, summary: 'Created customers.' };`;
    if (stage === 'parse') {
      const request = field('request');
      if (request.endsWith('?')) return `return ${JSON.stringify({ kind: 'question', sql: ["SELECT count(*) FROM customers c WHERE c.city = 'Lisbon'"],
        plan: { steps: [{ op: 'scan', table: 'customers', alias: 'c', where: "c.city = 'Lisbon'" }, { op: 'aggregate', input: 0, groupBy: [], aggregates: [{ name: 'count', expression: 'count(*)' }] }] },
        columns: ['count'], explanation: 'Counted the Lisbon rows.', assumptions: [] })};`;
      const [, name, city] = /^(\w+) from (\w+)/.exec(request) ?? [];
      return `return ${JSON.stringify({ kind: 'change', sql: [`INSERT INTO customers (name, city) VALUES ('${name}', '${city}')`],
        changes: [{ op: 'insert', table: 'customers', rows: [{ name, city }], from: null }], assumptions: [] })};`;
    }
    if (stage === 'plan') return `
      if (statement.kind !== 'question') return statement;
      const first = statement.plan.steps[0];
      const value = /= '(\\w+)'/.exec(first.where)[1];
      const lookup = { op: 'lookup', table: first.table, alias: first.alias, column: 'city', values: [value], range: null, where: null };
      return { ...statement, plan: { steps: [lookup, ...statement.plan.steps.slice(1)] } };`;
    if (stage === 'execute') return `
      if (statement.kind === 'question') {
        const results = [];
        for (const step of statement.plan.steps) {
          if (step.op === 'lookup') results.push(await lookup(folder, step.table, step.alias, step.column, step.values, step.range, step.where));
          else if (step.op === 'aggregate') results.push(await aggregate(results[step.input], step.groupBy, step.aggregates));
          else throw new Error('unexpected step ' + step.op);
        }
        await folder.file('scratch.txt').writeText('an executor may write scratch files; a question keeps none');
        return { rows: results.at(-1).map(row => statement.columns.map(column => row[column])), changes: {} };
      }
      const counts = {};
      for (const change of statement.changes) {
        const stored = await folder.apply(write, change.table, change.rows.map(row => ({ before: null, after: row })));
        await folder.apply(reindex, change.table, stored);
        const problems = await check(folder, change.table, stored);
        if (problems.length) throw new Error(problems.join('; '));
        counts[change.table] = { inserted: stored.length, updated: 0, deleted: 0 };
      }
      return { rows: [], changes: counts };`;
    if (stage === 'lookup') return `
      const index = await folder.file('indexes/' + table + '/' + column + '.json').readJson();
      const ids = new Set(index.filter(([value]) => values.includes(value)).map(([, id]) => id));
      const rows = (await folder.file('tables/' + table + '/0.jsonl').readText()).split('\\n').filter(Boolean).map(line => JSON.parse(line));
      return rows.filter(row => ids.has(row._id)).map(row => Object.fromEntries(Object.entries(row).map(([k, v]) => [alias + '.' + k, v])));`;
    if (stage === 'aggregate') return 'return [{ count: rows.length }];';
    if (stage === 'write') return `
      const catalog = await folder.file('catalog.json').readJson();
      const entry = catalog.tables[table];
      const page = folder.file('tables/' + table + '/0.jsonl');
      const lines = (await page.readText()).split('\\n').filter(Boolean);
      const stored = changes.map(change => ({ before: null, after: { _id: entry.nextId++, ...change.after } }));
      await page.writeText([...lines, ...stored.map(change => JSON.stringify(change.after))].join('\\n') + '\\n');
      entry.rows += stored.length;
      await folder.file('catalog.json').writeJson(catalog);
      return stored;`;
    if (stage === 'reindex') return `
      const file = folder.file('indexes/' + table + '/city.json');
      const index = await file.readJson();
      for (const change of changes) index.push([change.after.city, change.after._id]);
      index.sort((a, b) => a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : a[1] - b[1]);
      await file.writeJson(index);
      return changes.length;`;
    if (stage === 'check') return `
      const rows = (await folder.file('tables/' + table + '/0.jsonl').readText()).split('\\n').filter(Boolean).map(line => JSON.parse(line));
      return changes.filter(change => rows.filter(row => row.name === change.after.name).length > 1)
        .map(change => 'customers row ' + change.after._id + ' breaks unique name: ' + change.after.name + ' exists');`;
    return null;
  });
  const decide = async ({ messages, options }) => {
    const request = /request: string = "([^"]*)"/.exec(JSON.stringify(messages).replace(/\\"/g, '"'))?.[1] ?? '';
    const winner = /^keep track/i.test(request) ? 'schema' : /\?$/.test(request) ? 'question' : /old/.test(request) ? 'unclear' : 'change';
    return { log_probs: options.map(option => Math.log(option === JSON.stringify(winner) ? 0.9 : 0.1 / (options.length - 1))) };
  };
  return { driver: Object.assign(model.driver, { decide }), seen };
}

test('the folder database: a natural-language server parses, plans and executes; writes pages, indexes and checks; commits atomically', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-nldb-'));
  const path = join(root, 'shop.nldb');
  const model = folderModel();
  const runtime = createNatlangRuntime({ model: { driver: model.driver, maxTurns: 6 } });
  const db = new FolderDatabase(path, fn => runtime.run(fn));
  try {
    assert.ok(existsSync(join(path, 'FORMAT.md')));
    const schema = await db.ask('Keep track of customers: name and city.');
    assert.equal(schema.kind, 'schema', JSON.stringify(schema) + JSON.stringify(model.seen));
    assert.equal(db.catalog().version, 1);
    assert.deepEqual(model.seen.splice(0), ['database', 'define', 'design']);
    for (const request of ['Ana from Lisbon joined.', 'Ben from Porto joined.', 'Carla from Lisbon joined.']) {
      const outcome = await db.ask(request);
      assert.equal(outcome.report?.changes.customers.inserted, 1, JSON.stringify(outcome));
    }
    assert.deepEqual(model.seen.splice(0, 7), ['database', 'parse', 'plan', 'execute', 'write', 'reindex', 'check']);
    assert.deepEqual(JSON.parse(readFileSync(join(path, 'indexes/customers/city.json'), 'utf8')), [['Lisbon', 1], ['Lisbon', 3], ['Porto', 2]]);
    const before = readFileSync(join(path, 'tables/customers/0.jsonl'), 'utf8');
    const failed = await db.ask('Ana from Faro joined.');
    assert.equal(failed.kind, 'change');
    assert.match(failed.error, /breaks unique name/);
    assert.equal(readFileSync(join(path, 'tables/customers/0.jsonl'), 'utf8'), before, 'the failed transaction left nothing');
    assert.equal(db.catalog().tables.customers.rows, 3);
    assert.deepEqual(readdirSync(join(path, 'log')).sort(), ['00000001.json', '00000002.json', '00000003.json', '00000004.json']);
    model.seen.splice(0);
    const answer = await db.ask('How many customers live in Lisbon?');
    assert.deepEqual(answer.answer?.rows, [[2]], JSON.stringify(answer));
    assert.deepEqual(model.seen.splice(0), ['database', 'parse', 'plan', 'execute', 'lookup', 'aggregate'], 'the optimizer chose the index');
    assert.equal(existsSync(join(path, 'scratch.txt')), false, 'a question changes no file');
    assert.equal(judgeStep({ request: '', kind: 'question', answer: [[2]] }, answer), null);
    const unclear = await db.ask('Remove the old customers.');
    assert.deepEqual(unclear, { kind: 'unclear', clarification: 'How old is old: signed up before which date?' });

    // A commit interrupted after its log record is finished when the database opens again.
    writeFileSync(join(path, 'tables/customers/0.jsonl'), 'torn');
    new FolderDatabase(path, fn => runtime.run(fn));
    assert.equal(readFileSync(join(path, 'tables/customers/0.jsonl'), 'utf8'), before);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('the SQLite database: requests compile to one transaction; a failing statement rolls all back; questions cannot write; meaning is judged once per value', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-nldb-'));
  const judged = [];
  const plan = (kind, statements, extra = {}) => `return ${JSON.stringify({ kind, statements, semantic: [], clarification: '', explanation: `${kind}.`, assumptions: [], ...extra })};`;
  const model = scriptedModel(opening => {
    if (opening.includes('Decide whether value, a value stored in a database, meets criterion')) {
      const value = /value: string = "([^"]*)"/.exec(opening)[1];
      judged.push(value);
      return `return ${/late/.test(value)};`;
    }
    const request = /request: string = "([^"]*)"/.exec(opening)?.[1] ?? '';
    if (/^Keep/.test(request)) return plan('schema', ['CREATE TABLE orders (id INTEGER PRIMARY KEY, note TEXT NOT NULL, total REAL NOT NULL CHECK (total >= 0))']);
    if (/^Record/.test(request)) return plan('change', ["INSERT INTO orders (note, total) VALUES ('arrived late', 20)", "INSERT INTO orders (note, total) VALUES ('fine', 5)", "INSERT INTO orders (note, total) VALUES ('very late again', 7)"]);
    if (/^Refund/.test(request)) return plan('change', ['UPDATE orders SET total = total - 6 WHERE id = 2', 'UPDATE orders SET total = total - 6 WHERE id = 1']);
    if (/sneaky/.test(request)) return plan('question', ['DELETE FROM orders']);
    if (/late/.test(request)) return plan('question', ["SELECT id, total FROM orders WHERE meets(note, 'mentions a late delivery') ORDER BY id"],
      { semantic: [{ criterion: 'mentions a late delivery', table: 'orders', column: 'note' }] });
    return null;
  });
  const runtime = createNatlangRuntime({ model: model.driver });
  const db = new SqliteDatabase(join(root, 'shop.sqlite'), fn => runtime.run(fn), () => '2026-10-07');
  try {
    assert.equal((await db.ask('Keep orders: a note and a total that is never negative.')).kind, 'schema');
    assert.deepEqual((await db.ask('Record three orders.')).report.changes, { orders: { inserted: 3, updated: 0, deleted: 0 } });
    const refund = await db.ask('Refund 6 on orders 1 and 2.');
    assert.match(refund.error, /CHECK constraint failed/);
    const late = await db.ask('Which orders came late?');
    assert.deepEqual(late.answer, { columns: ['id', 'total'], rows: [[1, 20], [3, 7]], explanation: 'question.', assumptions: [] }, 'the refund rolled back as a whole');
    assert.deepEqual(judged.sort(), ['arrived late', 'fine', 'very late again']);
    await db.ask('Which orders came late, again?');
    assert.equal(judged.length, 3, 'judgments are cached in the database');
    const sneaky = await db.ask('A sneaky question?');
    assert.match(sneaky.error, /one SELECT/);
    assert.equal((await db.ask('Which orders came late?')).answer.rows.length, 2);
  } finally { db.close(); rmSync(root, { recursive: true, force: true }); }
});

test('neuralese blocks: content-addressed safetensors files, per-dialect IVF search, refusal across dialects', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-nldb-'));
  const block = (dialect, rows) => {
    const data = new Uint8Array(new Float32Array(rows.flat()).buffer);
    return { dialect, length: rows.length, width: rows[0].length, dtype: 'f32', data };
  };
  try {
    const store = new FolderBlockStore(root);
    const memory = new MemoryNeuraleseStore();
    const input = block('nd:test@1', [[1, 0, 0, 0], [0.8, 0.2, 0, 0]]);
    const meta = await store.put(input);
    assert.equal(meta.id, (await memory.put(input)).id, 'the same content ID as the runtime store');
    assert.equal((await store.put(input)).id, meta.id);
    const stored = await store.get(meta.id);
    assert.deepEqual([...stored.data], [...input.data]);
    assert.deepEqual(decodeBlock(encodeBlock(stored)).meta, stored.meta);
    // Three clusters of blocks; a query near each finds its cluster first.
    const centers = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]];
    const blocks = [];
    for (const [c, center] of centers.entries()) for (let i = 0; i < 5; i++) {
      const rows = [center.map((v, j) => v + 0.05 * Math.sin(i + j + c)), center.map((v, j) => v + 0.05 * Math.cos(i * j + c))];
      blocks.push({ ...(await store.put(block('nd:test@1', rows))), cluster: c });
    }
    const loaded = await Promise.all(blocks.map(b => store.get(b.id)));
    const index = new IvfIndex('nd:test@1', 4).train(loaded.map(pooled), 3);
    for (const b of loaded) index.add(b);
    for (const [c, center] of centers.entries()) {
      const hits = index.search({ meta: { id: 'q', dialect: 'nd:test@1', length: 1, width: 4, dtype: 'f32' }, data: new Uint8Array(new Float32Array(center).buffer) }, 5, 1);
      assert.deepEqual(hits.map(hit => blocks.find(b => b.id === hit.id).cluster), [c, c, c, c, c]);
    }
    const copy = IvfIndex.fromJSON(JSON.parse(JSON.stringify(index)));
    assert.deepEqual(copy.search(loaded[0], 1), index.search(loaded[0], 1));
    assert.throws(() => index.search({ meta: { id: 'q', dialect: 'nd:other@1', length: 1, width: 4, dtype: 'f32' }, data: new Uint8Array(16) }),
      /serves dialect nd:test@1/);
    const removed = await store.collect(new Set(blocks.slice(0, 5).map(b => b.id)));
    assert.equal(removed.length, 11, 'blocks no row refers to are collected');
  } finally { rmSync(root, { recursive: true, force: true }); }
});
