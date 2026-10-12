import { test } from 'node:test';
import assert from 'node:assert/strict';
import { slugify, uniqueSlug } from '../src/slug.js';

test('ascii', () => assert.equal(slugify('Hello, World!'), 'hello-world'));
test('accents', () => assert.equal(slugify('Crème Brûlée'), 'creme-brulee'));
test('unique', () => {
  const taken = new Set();
  assert.deepEqual([uniqueSlug('A b', taken), uniqueSlug('a-b', taken)], ['a-b', 'a-b-2']);
});
