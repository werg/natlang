import { test } from 'node:test';
import assert from 'node:assert/strict';
import { deepMerge } from '../src/merge.js';
import { settings } from '../src/index.js';

test('nested merge', () => {
  assert.deepEqual(deepMerge({ a: { x: 1, y: 2 } }, { a: { y: 3 }, b: 4 }), { a: { x: 1, y: 3 }, b: 4 });
});
test('arrays are replaced', () => {
  assert.deepEqual(deepMerge({ list: [1, 2, 3] }, { list: [9] }), { list: [9] });
});
test('defaults are not modified', () => {
  settings({ theme: { color: 'red' } });
  assert.equal(settings({}).theme.color, 'blue');
});
