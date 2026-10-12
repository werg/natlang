import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { deepMerge } = await import(pathToFileURL(`${process.cwd()}/src/merge.js`).href);
const freeze = (v) => { if (v && typeof v === 'object') { Object.values(v).forEach(freeze); Object.freeze(v); } return v; };
test('inputs untouched (frozen)', () => {
  const a = freeze({ a: { b: [1, { c: 2 }] }, keep: 1 });
  const b = freeze({ a: { d: 3 }, list: [{ e: 4 }] });
  const out = deepMerge(a, b);
  assert.deepEqual(out, { a: { b: [1, { c: 2 }], d: 3 }, keep: 1, list: [{ e: 4 }] });
  assert.notEqual(out, a);
});
test('no shared structure', () => {
  const a = { a: { b: [1, { c: 2 }] } };
  const b = { list: [{ e: 4 }], n: { m: 1 } };
  const out = deepMerge(a, b);
  out.a.b[1].c = 99; out.a.b.push(5); out.list[0].e = 99; out.n.m = 99;
  assert.deepEqual(a, { a: { b: [1, { c: 2 }] } });
  assert.deepEqual(b, { list: [{ e: 4 }], n: { m: 1 } });
});
test('replacement rules', () => {
  assert.deepEqual(deepMerge({ x: { y: 1 } }, { x: [1] }), { x: [1] });
  assert.deepEqual(deepMerge({ x: [1, 2] }, { x: { y: 1 } }), { x: { y: 1 } });
  assert.deepEqual(deepMerge({ x: { y: 1 } }, { x: null }), { x: null });
  assert.deepEqual(deepMerge({ x: 1 }, {}), { x: 1 });
});
