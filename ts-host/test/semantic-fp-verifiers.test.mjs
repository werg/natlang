// Trusted verifiers of authored semantic-FP tasks (training/semantic_fp_tasks/verifiers, plans/semantic-fp/05 O1):
// the trusted-verifier binding admits that root and carries graded scores and the run's files; each verifier accepts
// every outcome that meets its oracle, rejects misreadings, and gives no credit for leaving things as they were.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { test } from 'node:test';
import { fileURLToPath, pathToFileURL } from 'node:url';

const repo = fileURLToPath(new URL('../../', import.meta.url));
const sandbox = mkdtempSync(join(tmpdir(), 'sfp-verifiers-'));
process.env.NATLANG_REPO_ROOT = sandbox;
const { checkTrustedExactVerifier } = await import('../dist/teacher/trusted-exact-verifier.js');
const load = async name => import(pathToFileURL(resolve(repo, 'training/semantic_fp_tasks/verifiers', name)).href);

function bind(path, source) {
  mkdirSync(resolve(sandbox, path, '..'), { recursive: true });
  writeFileSync(resolve(sandbox, path), source);
  return { path, sha256: createHash('sha256').update(source).digest('hex'), oracle: {} };
}
const input = { value: 1, oracle: {}, inputs: {}, fixture: { 'a.md': 'x' } };

test('the binding admits semantic_fp_tasks, carries the score and passes the files the run left', async () => {
  const echo = bind('training/semantic_fp_tasks/verifiers/echo.mjs',
    'export function verifyCandidate(i) { return { ok: false, issues: [JSON.stringify(i.files ?? null)], score: 0.25 }; }\n');
  const verdict = await checkTrustedExactVerifier(echo, { ...input, files: { 'b.md': 'y' } });
  assert.equal(verdict.accepted, false);
  assert.equal(verdict.score, 0.25);
  assert.match(verdict.verdict, /b\.md/);
  const outside = bind('training/other/echo.mjs', 'export function verifyCandidate() { return { ok: true, issues: [] }; }\n');
  await assert.rejects(checkTrustedExactVerifier(outside, input), /invalid trusted exact verifier binding/);
  for (const [name, body] of [['range', '{ ok: false, issues: [], score: 2 }'], ['partial-pass', '{ ok: true, issues: [], score: 0.5 }']]) {
    const bad = bind(`training/semantic_fp_tasks/verifiers/${name}.mjs`, `export function verifyCandidate() { return ${body}; }\n`);
    await assert.rejects(checkTrustedExactVerifier(bad, input), /invalid trusted exact verifier result/);
  }
});

test('labels: any order and case, every item once, misreadings partially credited', async () => {
  const { verifyCandidate } = await load('labels.mjs');
  const oracle = { items: { a: ['keep'], b: ['drop'], c: ['keep', 'drop'] }, labels: ['keep', 'drop'], shape: 'by_label' };
  assert.equal(verifyCandidate({ value: { keep: ['A', 'c'], drop: ['b'] }, oracle }).ok, true);
  assert.equal(verifyCandidate({ value: { keep: ['a'], drop: ['b', 'c'] }, oracle }).ok, true);
  const wrong = verifyCandidate({ value: { keep: ['a', 'b', 'c'], drop: [] }, oracle });
  assert.equal(wrong.ok, false);
  assert.ok(Math.abs(wrong.score - 2 / 3) < 1e-9);
  assert.equal(verifyCandidate({ value: { keep: ['a', 'a', 'c'], drop: ['b'] }, oracle }).ok, false);
  const rows = { ...oracle, shape: 'rows', id_key: 'id', label_key: 'route' };
  assert.equal(verifyCandidate({ value: [{ id: 'a', route: 'keep' }, { id: 'b', route: 'drop' }, { id: 'c', route: 'keep' }], oracle: rows }).ok, true);
});

test('order: every order keeping the precedences passes; excluded items must be absent', async () => {
  const { verifyCandidate } = await load('order.mjs');
  const oracle = { items: ['strip', 'plaster', 'paint', 'tile', 'island'], before: [['strip', 'plaster'], ['plaster', 'paint']],
    excluded: ['island'] };
  for (const value of [['strip', 'plaster', 'paint', 'tile'], ['tile', 'strip', 'plaster', 'paint'], ['strip', 'tile', 'plaster', 'paint']])
    assert.equal(verifyCandidate({ value, oracle }).ok, true, value.join());
  assert.equal(verifyCandidate({ value: ['plaster', 'strip', 'paint', 'tile'], oracle }).ok, false);
  assert.equal(verifyCandidate({ value: ['strip', 'plaster', 'paint', 'tile', 'island'], oracle }).ok, false);
  assert.equal(verifyCandidate({ value: [{ job: 'strip' }, { job: 'plaster' }, { job: 'paint' }, { job: 'tile' }], oracle: { ...oracle, key: 'job' } }).ok, true);
});

test('assignment: constraints decide, not one expected assignment', async () => {
  const { verifyCandidate } = await load('assignment.mjs');
  const oracle = { items: ['ana', 'ben', 'cy'], slots: ['sat', 'sun'], allowed: { ana: ['sat', 'sun'], ben: ['sun'], cy: ['sat', 'sun', 'none'] },
    together: [['ana', 'ben']], apart: [], capacity: { sun: 2 }, shape: 'by_item' };
  assert.equal(verifyCandidate({ value: { ana: 'sun', ben: 'sun', cy: 'sat' }, oracle }).ok, true);
  assert.equal(verifyCandidate({ value: { ana: 'sun', ben: 'sun', cy: 'none' }, oracle }).ok, true);
  assert.equal(verifyCandidate({ value: { ana: 'sun', ben: 'sun', cy: 'sun' }, oracle }).ok, false);   // capacity
  assert.equal(verifyCandidate({ value: { ana: 'sat', ben: 'sun', cy: 'sat' }, oracle }).ok, false);   // together
  assert.equal(verifyCandidate({ value: { sun: ['Ana', 'Ben'] }, oracle: { ...oracle, shape: 'by_slot' } }).ok, true);
});

test('ledger: executes the operations; any list reaching the goal passes; doing nothing earns nothing', async () => {
  const { verifyCandidate } = await load('ledger.mjs');
  const initial = { B1: { day: 'Tue', chairs: 20 }, B2: { day: 'Wed', chairs: 40 }, B3: { day: 'Thu', chairs: 15 } };
  const goal = { B1: { day: 'Thu', chairs: 20 }, B2: { day: 'Wed', chairs: 50 }, B3: null };
  const oracle = { initial, goal, fields: ['day', 'chairs'] };
  const ops = [{ op: 'update', id: 'B1', set: { day: 'Thu' } }, { op: 'update', id: 'B2', set: { chairs: 50 } }, { op: 'remove', id: 'B3' }];
  assert.equal(verifyCandidate({ value: ops, oracle }).ok, true);
  assert.equal(verifyCandidate({ value: [...ops].reverse().concat([{ op: 'update', id: 'B2', set: { day: 'wed', chairs: '50' } }]), oracle }).ok, true);
  const nothing = verifyCandidate({ value: [], oracle });
  assert.equal(nothing.ok, false);
  assert.equal(nothing.score, 0);
  const cue = verifyCandidate({ value: [ops[0], { op: 'remove', id: 'B2' }, ops[2]], oracle });
  assert.equal(cue.ok, false);
  assert.ok(Math.abs(cue.score - 2 / 3) < 1e-9);
  assert.equal(verifyCandidate({ value: [...ops, { op: 'update', id: 'B1', set: { room: 'Oak' } }], oracle }).ok, false);
  assert.equal(verifyCandidate({ value: [...ops, { op: 'add', id: 'B9', record: { day: 'Fri', chairs: 1 } }], oracle }).ok, false);
});

test('placement: each file at a place its meaning allows, text kept, nothing stray; staying earns nothing', async () => {
  const { verifyCandidate } = await load('placement.mjs');
  const fixture = { 'inbox/a.md': 'Kettle unopened\n', 'inbox/b.md': 'Burnt toaster\n', 'inbox/c.md': 'Dented tin\n' };
  const oracle = { files: { 'inbox/a.md': ['restock/'], 'inbox/b.md': ['scrap/', 'removed'], 'inbox/c.md': ['inbox/', 'restock/'] },
    written: { 'summary.md': { includes: ['a.md'], lines: 1 } }, value: { moved: 2 } };
  const good = { 'restock/a.md': 'Kettle unopened', 'scrap/b.md': 'Burnt toaster\n', 'inbox/c.md': 'Dented tin\n', 'summary.md': 'a.md b.md\n' };
  assert.equal(verifyCandidate({ value: { moved: 2 }, oracle, fixture, files: good }).ok, true);
  const removed = { ...good }; delete removed['scrap/b.md'];
  assert.equal(verifyCandidate({ value: { moved: 2 }, oracle, fixture, files: removed }).ok, true);
  assert.equal(verifyCandidate({ value: { moved: 3 }, oracle, fixture, files: good }).ok, false);
  assert.equal(verifyCandidate({ value: { moved: 2 }, oracle, fixture, files: { ...good, 'restock/a.md': 'changed' } }).ok, false);
  assert.equal(verifyCandidate({ value: { moved: 2 }, oracle, fixture, files: { ...good, 'stray.md': 'x' } }).ok, false);
  const untouched = verifyCandidate({ value: null, oracle, fixture, files: fixture });
  assert.equal(untouched.ok, false);
  assert.equal(untouched.score, 0);
  assert.equal(verifyCandidate({ value: { moved: 2 }, oracle, fixture }).ok, false);
});

test('records: field by field, several acceptable readings, numbers within tolerance, no record where none is due', async () => {
  const { verifyCandidate } = await load('records.mjs');
  const oracle = { items: { r1: { amount: [40], payer: ['Ana'] }, r2: { amount: [12.5, 12], payer: ['Ben', 'Ben Ode'] }, r3: null },
    shape: 'by_item', tolerance: 0.01 };
  assert.equal(verifyCandidate({ value: { r1: { amount: '40.00', payer: 'ana' }, r2: { amount: 12, payer: 'Ben Ode' } }, oracle }).ok, true);
  assert.equal(verifyCandidate({ value: { r1: { amount: 40, payer: 'Ana' }, r2: { amount: 12.5, payer: 'Ben' }, r3: { amount: 1 } }, oracle }).ok, false);
  const wrong = verifyCandidate({ value: { r1: { amount: 30, payer: 'Ana' }, r2: { amount: 12.5, payer: 'Ben' } }, oracle });
  assert.equal(wrong.ok, false);
  assert.ok(Math.abs(wrong.score - 4 / 5) < 1e-9);
  const rows = { ...oracle, shape: 'rows', id_key: 'id' };
  assert.equal(verifyCandidate({ value: [{ id: 'r1', amount: 40, payer: 'Ana' }, { id: 'r2', amount: 12.5, payer: 'Ben' }], oracle: rows }).ok, true);
  assert.equal(verifyCandidate({ value: [{ id: 'r1', amount: 40, payer: 'Ana' }, { id: 'r1', amount: 40, payer: 'Ana' }], oracle: rows }).ok, false);
});

test('selection: required in, forbidden out, optional either way, bounds hold', async () => {
  const { verifyCandidate } = await load('selection.mjs');
  const oracle = { required: ['a', 'b'], forbidden: ['c', 'd'], optional: ['e'], max: 3 };
  assert.equal(verifyCandidate({ value: ['B', 'a'], oracle }).ok, true);
  assert.equal(verifyCandidate({ value: ['a', 'b', 'e'], oracle }).ok, true);
  assert.equal(verifyCandidate({ value: ['a', 'b', 'c'], oracle }).ok, false);
  assert.equal(verifyCandidate({ value: ['a'], oracle }).ok, false);
  assert.equal(verifyCandidate({ value: [], oracle }).score, 0.5);
  assert.equal(verifyCandidate({ value: ['b', 'a'], oracle: { ...oracle, ordered: true } }).ok, false);
  assert.equal(verifyCandidate({ value: [{ id: 'a' }, { id: 'b' }], oracle: { ...oracle, key: 'id' } }).ok, true);
});

test('clusters: only who is together counts, names are free, loose pairs either way', async () => {
  const { verifyCandidate } = await load('clusters.mjs');
  const oracle = { clusters: [['t1', 't3'], ['t2'], ['t4', 't5']], loose: [['t2', 't4']], shape: 'lists' };
  assert.equal(verifyCandidate({ value: [['T3', 't1'], ['t5', 't4'], ['t2']], oracle }).ok, true);
  assert.equal(verifyCandidate({ value: [['t1', 't3'], ['t2', 't4', 't5']], oracle }).ok, false);   // t2-t5 not loose
  assert.equal(verifyCandidate({ value: { x: 'a', t1: 'a' }, oracle: { ...oracle, shape: 'by_item' } }).ok, false);
  assert.equal(verifyCandidate({ value: { t1: 'a', t3: 'a', t2: 'b', t4: 'c', t5: 'c' }, oracle: { ...oracle, shape: 'by_item' } }).ok, true);
  assert.equal(verifyCandidate({ value: [['t1', 't2', 't3', 't4', 't5']], oracle }).ok, false);
  assert.equal(verifyCandidate({ value: { one: ['t1', 't3'], two: ['t2'], three: ['t4', 't5'] }, oracle: { ...oracle, shape: 'by_cluster' } }).ok, true);
});

test('numeric: the value or any accepted reading, record name by name, tolerance', async () => {
  const { verifyCandidate } = await load('numeric.mjs');
  assert.equal(verifyCandidate({ value: '1,240.00', oracle: { value: 1240 } }).ok, true);
  assert.equal(verifyCandidate({ value: 1250, oracle: { value: 1240, accept: [1250] } }).ok, true);
  assert.equal(verifyCandidate({ value: 0, oracle: { value: 1240 } }).ok, false);
  const oracle = { value: { owed: 30, refunded: 12.5 } };
  assert.equal(verifyCandidate({ value: { owed: 30, refunded: 12.5 }, oracle }).ok, true);
  const half = verifyCandidate({ value: { owed: 30, refunded: 10 }, oracle });
  assert.equal(half.ok, false);
  assert.equal(half.score, 0.5);
  assert.equal(verifyCandidate({ value: { owed: 30, refunded: 12.5, extra: 1 }, oracle }).ok, false);
});
