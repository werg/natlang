import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { countMatches, highlight } = await import(pathToFileURL(`${process.cwd()}/src/highlight.js`).href);
test('hidden', () => {
  assert.equal(highlight('cost $5 (approx) [x]', '(approx)'), 'cost $5 <mark>(approx)</mark> [x]');
  assert.equal(highlight('cost $5 [x]', '$5'), 'cost <mark>$5</mark> [x]');
  assert.equal(highlight('a[x]b', '[X]'), 'a<mark>[x]</mark>b');
  assert.equal(highlight('path\\to', '\\'), 'path<mark>\\</mark>to');
  assert.equal(countMatches('why? why? ^_^ a|b', '?'), 2);
  assert.equal(countMatches('^_^ a|b ab', 'a|b'), 1);
  assert.equal(countMatches('*** {2}', '{2}'), 1);
  assert.equal(highlight('nothing', ''), 'nothing');
});
