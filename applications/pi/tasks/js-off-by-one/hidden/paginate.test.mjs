import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { paginate } = await import(pathToFileURL(`${process.cwd()}/src/paginate.js`).href);
const items = Array.from({ length: 25 }, (_, i) => i + 1);
test('hasNext', () => { assert.equal(paginate(items, 2).hasNext, true); assert.equal(paginate(items, 3).hasNext, false); });
test('size', () => assert.deepEqual(paginate(items, 2, 4).items, [5, 6, 7, 8]));
test('rejects page 0', () => assert.throws(() => paginate(items, 0)));
test('empty', () => assert.deepEqual(paginate([], 1), { page: 1, pages: 1, items: [], hasNext: false }));
