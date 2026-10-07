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

/** The pure engine's interpreter, scripted: each reply is eval code using the folder the reducer received. */
function folderModel() {
  return scriptedModel(opening => {
    const request = /request: string = "([^"]*)"|question: string = "([^"]*)"/.exec(opening)?.slice(1).find(Boolean) ?? '';
    if (opening.includes('Decide what request asks of the database')) {
      return `return ${JSON.stringify(/^keep track/i.test(request) ? 'schema' : /\?$/.test(request) ? 'question' : /old/.test(request) ? 'unclear' : 'change')};`;
    }
    if (opening.includes('Ask the one short question')) return 'return "How old is old: signed up before which date?";';
    if (opening.includes('Change the schema of the database in folder')) {
      return `await folder.file('catalog.json').writeJson(${JSON.stringify(CATALOG)});
await folder.file('tables/customers/0.jsonl').writeText('');
await folder.file('indexes/customers/city.json').writeText('[]');
return { statements: ['CREATE TABLE customers (name TEXT NOT NULL UNIQUE, city TEXT NOT NULL)'], changes: {}, assumptions: [], summary: 'Created customers.' };`;
    }
    if (opening.includes('Carry out request as one transaction')) {
      const [, name, city] = /^(\w+) from (\w+)/.exec(request) ?? [];
      return `const current = await folder.file('catalog.json').readJson();
const page = await folder.file('tables/customers/0.jsonl').readText();
const rows = page.split('\\n').filter(Boolean).map(line => JSON.parse(line));
const row = { _id: current.tables.customers.nextId, name: ${JSON.stringify(name)}, city: ${JSON.stringify(city)} };
await folder.file('tables/customers/0.jsonl').writeText([...rows, row].map(r => JSON.stringify(r)).join('\\n') + '\\n');
current.tables.customers.nextId += 1; current.tables.customers.rows += 1;
await folder.file('catalog.json').writeJson(current);
if (rows.some(r => r.name === row.name)) throw new Error('unique violation: customers.name ' + row.name + ' exists');
return { statements: ['INSERT INTO customers (name, city) VALUES (' + row.name + ', ' + row.city + ')'], changes: { customers: { inserted: 1, updated: 0, deleted: 0 } }, assumptions: [], summary: 'Added ' + row.name + '.' };`;
    }
    if (opening.includes('Answer question from the database in folder')) {
      return `const rows = (await folder.file('tables/customers/0.jsonl').readText()).split('\\n').filter(Boolean).map(line => JSON.parse(line));
await folder.file('scratch.txt').writeText('a query may write scratch files; they are discarded');
return { columns: ['count'], rows: [[rows.filter(r => r.city === 'Lisbon').length]], explanation: 'Counted the Lisbon rows.', assumptions: [] };`;
    }
    return null;
  });
}

test('the folder database: schema, committed transactions with a redo log, a failed transaction leaves nothing, read-only questions', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-nldb-'));
  const path = join(root, 'shop.nldb');
  const runtime = createNatlangRuntime({ model: folderModel().driver });
  const db = new FolderDatabase(path, fn => runtime.run(fn));
  try {
    assert.ok(existsSync(join(path, 'FORMAT.md')));
    const schema = await db.ask('Keep track of customers: name and city.');
    assert.equal(schema.kind, 'schema', JSON.stringify(schema));
    assert.equal(db.catalog().version, 1);
    for (const request of ['Ana from Lisbon joined.', 'Ben from Porto joined.', 'Carla from Lisbon joined.']) {
      const outcome = await db.ask(request);
      assert.equal(outcome.report?.changes.customers.inserted, 1, JSON.stringify(outcome));
    }
    const before = readFileSync(join(path, 'tables/customers/0.jsonl'), 'utf8');
    const failed = await db.ask('Ana from Faro joined.');
    assert.equal(failed.kind, 'change');
    assert.match(failed.error, /unique violation/);
    assert.equal(readFileSync(join(path, 'tables/customers/0.jsonl'), 'utf8'), before, 'the failed transaction left nothing');
    assert.equal(db.catalog().tables.customers.rows, 3);
    assert.deepEqual(readdirSync(join(path, 'log')).sort(), ['00000001.json', '00000002.json', '00000003.json', '00000004.json']);
    const answer = await db.ask('How many customers live in Lisbon?');
    assert.deepEqual(answer.answer.rows, [[2]]);
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
