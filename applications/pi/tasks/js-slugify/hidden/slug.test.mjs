import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { slugify } = await import(pathToFileURL(`${process.cwd()}/src/slug.js`).href);
test('hidden', () => {
  assert.equal(slugify('Ångström Über 2024'), 'angstrom-uber-2024');
  assert.equal(slugify('naïve café'), 'naive-cafe');
  assert.equal(slugify('ÉCOLE Señor Çava'), 'ecole-senor-cava');
  assert.equal(slugify('  --Hello,   World!-- '), 'hello-world');
  assert.equal(slugify(''), '');
  assert.equal(slugify('日本語'), '');
  assert.equal(slugify('été'), 'ete');
});
