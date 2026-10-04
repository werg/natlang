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
