import test from 'node:test';
import assert from 'node:assert/strict';
import { SHOWN_CHARS, shownCharsFor } from '../dist/native/cutoff.js';
import { PageStore } from '../dist/native/pages.js';

// Owner 2026-10-11: no fixed 2,000-character page. The page follows the call's context budget.
test('the shown size of a tool output follows the context budget, never below the floor', () => {
  assert.equal(shownCharsFor(null), SHOWN_CHARS);
  assert.equal(shownCharsFor(undefined), SHOWN_CHARS);
  assert.equal(shownCharsFor(1024), SHOWN_CHARS);
  assert.equal(shownCharsFor(16384), 7168);
  assert.equal(shownCharsFor(65536), 28672);
  assert.ok(shownCharsFor(131072) > shownCharsFor(65536));
});

test('a text that fits the budget is shown whole; a longer one is paged at the budget', () => {
  const store = new PageStore();
  store.budget = shownCharsFor(65536);
  const evidence = 'x'.repeat(20000);
  assert.equal(store.show(evidence, 'item.evidence'), evidence);
  const long = Array.from({ length: 2000 }, (_, i) => `line ${i} ${'y'.repeat(40)}`).join('\n');
  const shown = store.show(long, 'item.evidence');
  assert.match(shown, /<<cut off: \d+ of \d+ characters not shown; item\.evidence holds all of it; read_page\("amber", 2\) shows the next part>>/);
  assert.ok(shown.length > 20000 && shown.length < store.budget + 300);
  const page2 = store.read('amber', 2);
  assert.ok(page2.length > 20000);
  const small = new PageStore();
  assert.equal(small.budget, SHOWN_CHARS);
});
