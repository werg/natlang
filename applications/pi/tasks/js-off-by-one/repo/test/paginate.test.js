import { test } from 'node:test';
import assert from 'node:assert/strict';
import { paginate } from '../src/paginate.js';

const items = Array.from({ length: 25 }, (_, i) => i + 1);
test('first page', () => assert.deepEqual(paginate(items, 1).items, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
test('last page is partial', () => assert.deepEqual(paginate(items, 3).items, [21, 22, 23, 24, 25]));
test('page count', () => assert.equal(paginate(items, 1).pages, 3));
