import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType } from '../dist/index.js';
import { MISSING, Reject, buildPending, coerce, dump, problems, unboundParts, createLiveIdentity, scopedLiveIdentity } from '../dist/native/values.js';
import { renderValue } from '../dist/native/agent.js';

test('execution-local live display identities preserve aliases and reset for fresh replay', () => {
  class Inventory { stock = 3; }
  const a = new Inventory(), b = new Inventory();
  const first = createLiveIdentity(), second = createLiveIdentity();
  assert.equal(first(a), 1); assert.equal(first(b), 2); assert.equal(first(a), 1);
  assert.equal(second(b), 1);
  assert.match(renderValue({ a, again: a, b }, { liveIdentity: first }), /a: \[Inventory #1/);
  assert.match(renderValue({ a, again: a, b }, { liveIdentity: first }), /again: \[Inventory #1/);
  assert.match(renderValue({ a, again: a, b }, { liveIdentity: first }), /b: \[Inventory #2/);
  const scope = {};
  assert.equal(scopedLiveIdentity(scope), scopedLiveIdentity(scope));
  assert.notEqual(scopedLiveIdentity(scope), scopedLiveIdentity({}));
});

test('native values distinguish missing, null and empty records', () => {
  const type = parseType('{ name: string, note?: string }');
  const draft = coerce({}, type, new TypeEnv(), 'return');
  assert.deepEqual(draft, {});
  assert.deepEqual(problems(draft, type, new TypeEnv(), 'return').holes.map(d => d.path), ['return/name']);
  assert.deepEqual(coerce({ name: 'Ada', note: null }, type, new TypeEnv()), { name: 'Ada' });
  assert.throws(() => coerce({ name: null }, type, new TypeEnv()), Reject);
  assert.notEqual(MISSING, null);
});

test('a record union is complete when a later structural arm is complete', () => {
  const env = new TypeEnv();
  const type = parseType('{ chains: ({ cause: string, invoiceId: string, creditId: string, refundIds: string[] } | { cause: string, creditId: string, refundIds: string[] })[] }');
  const valid = { chains: [
    { cause: 'invoice', invoiceId: 'I1', creditId: 'C1', refundIds: [] },
    { cause: 'credit', creditId: 'C2', refundIds: ['R1'] },
  ] };
  assert.deepEqual(problems(coerce(valid, type, env), type, env, 'value').holes, []);
  const missing = { chains: [{ cause: 'credit', creditId: 'C2' }] };
  assert.deepEqual(problems(coerce(missing, type, env), type, env, 'value').holes.map(d => d.path),
    ['value/chains/0/refundIds']);
});

test('lambda construction keeps typed parameters, instructions, and a record codebase', () => {
  const lam = buildPending({ $lambda: { type: '(item: number) => number', instructions: 'Double the item.' } });
  assert.equal(lam.nodeKind, 'lambda');
  assert.equal(lam.body, 'Double the item.\n');
  assert.deepEqual(unboundParts(lam, new TypeEnv(), '').map(d => d.path), ['/args/item']);
  assert.equal(dump(lam).$lambda.instructions, 'Double the item.\n');
  for (const removed of ['$map', '$fold', '$iterate']) assert.throws(() => buildPending({ [removed]: { type: 'number' } }), Reject);
  assert.throws(() => buildPending({ $lambda: { type: '() => number', code: 'return 1;' } }), Reject);
});

test('function-typed slots hold live functions and live values dump by identity', () => {
  const fn = () => 1;
  assert.equal(coerce(fn, parseType('(x: number) => number'), new TypeEnv()), fn);
  assert.throws(() => coerce({ $lambda: {} }, parseType('() => number'), new TypeEnv()), Reject);
  const when = new Date(0);
  assert.equal(coerce(when, parseType('Date'), new TypeEnv()), when);
  assert.match(JSON.stringify(dump({ when })), /"\$live":\{"type":"Date","id":\d+\}/);
});

test('integers beyond 2^53 are JS numbers in typed slots, as their inferred type says', async () => {
  const { inferValueType } = await import('../dist/native/runtime.js');
  // A row id parsed from JSON: JS holds the nearest double, and the typed boundary keeps exactly that double.
  const rows = JSON.parse('[[1390763244713922561, "Joe", 3.5]]');
  const type = parseType(inferValueType(rows));
  assert.deepEqual(coerce(rows, type, new TypeEnv()), rows);
  assert.equal(coerce(rows[0][0], parseType('number'), new TypeEnv()), rows[0][0]);
  assert.deepEqual(coerce([569731104070115300], parseType('number[]'), new TypeEnv()), [569731104070115300]);
  for (const bad of [Infinity, NaN]) assert.throws(() => coerce(bad, parseType('number'), new TypeEnv()), Reject);
});

// writing-failures 2026-11-13: a record under 128 characters with more than two fields was replaced by its keys
// ("fields: …; values not shown") whenever its parent handed it a budget equal to its own length, so every vcs digest
// lost one or more of its file records, including the only file of a one-file change.
test('a record that fits its budget is shown whole, however small the budget', () => {
  const row = { a: 'xxxxxxxxxxxx', b: 'yyyyyyyyyyyy', c: 'zzzzzzzzzz' };
  const rows = { rows: [row, { ...row }], n: 1 };
  const literal = '{ a: "xxxxxxxxxxxx", b: "yyyyyyyyyyyy", c: "zzzzzzzzzz" }';
  assert.equal(renderValue(rows, { holder: 'r', budget: 4096 }), `{ rows: [${literal}, ${literal}], n: 1 }`);
  assert.equal(renderValue([row, row], { holder: 'r', budget: 4096 }), `[${literal}, ${literal}]`);
  assert.equal(renderValue(row, { holder: 'r', budget: literal.length }), literal);
  const file = (path, change, status) => ({ path, change, unit: 'packages/shared', status });
  const digest = { units: [{ unit: 'packages/shared', files: [file('packages/shared/src/rpc.ts', 'edited', 'conflict'),
    file('packages/shared/README.md', 'edited', 'composed'), file('packages/shared/src/ids.ts', 'added', 'adopt')],
  intents: [{ text: 'Rename the rpc envelope fields', tier: 'stated' }] }], fileCount: 3, truncated: false };
  for (const budget of [4096, 20000]) {
    const shown = renderValue(digest, { holder: 'digest', budget });
    assert.doesNotMatch(shown, /<</);
    for (const path of ['rpc.ts', 'README.md', 'ids.ts']) assert.match(shown, new RegExp(`path: "packages/shared/(src/)?${path.replace('.', '\\.')}"`));
  }
  const single = { units: [{ unit: 'workers/pubsub', files: [{ path: 'workers/pubsub/index.ts', change: 'edited', unit: 'workers/pubsub',
    status: 'adopt' }] }], fileCount: 1, truncated: false };
  assert.match(renderValue(single, { holder: 'digest', budget: 3584 }), /files: \[\{ path: "workers\/pubsub\/index\.ts", change: "edited", unit: "workers\/pubsub", status: "adopt" \}\]/);
});

test('a record that does not fit a small budget is still shown by its keys', () => {
  const row = { first: 'a'.repeat(160), second: 'b'.repeat(160), third: 'c'.repeat(160) };
  assert.equal(renderValue(row, { holder: 'row', budget: 100 }),
    '{ <<fields: first, second, third; values not shown; only this display is shortened, row holds the complete value>> }');
});
