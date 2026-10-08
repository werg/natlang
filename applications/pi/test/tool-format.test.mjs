/**
 * The tool task's crisp formats equal pi-durable's own: output bounding, content bounding, the truncation diagnostic,
 * the final-result assembly, and fromSlot. Run: node --test applications/pi/test/tool-format.test.mjs
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { boundOutput as piBoundOutput } from '../vendor/durable/src/harness/output.ts';
import { boundContent as piBoundContent, fromSlot as piFromSlot, truncated as piTruncated } from '../vendor/durable/src/harness/tool.ts';
import { assemble, bound, boundContent, boundOutput, truncated } from '../tool/run/format.ts';
import fromSlot from '../tool/fromSlot.ts';

let seed = 7;
const random = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
const pieces = ['a', 'bc', '\n', '\n\n', 'é', '€', '😀', 'line of text', '  ', '\t', 'x'.repeat(40)];
const text = () => Array.from({ length: Math.floor(random() * 40) }, () => pieces[Math.floor(random() * pieces.length)]).join('');
const limits = () => ({ maxBytes: Math.floor(random() * 60), maxLines: Math.floor(random() * 8), retain: random() < 0.5 ? 'head' : 'tail' });

test('boundOutput equals pi on random text and limits', () => {
  for (let i = 0; i < 5000; i++) {
    const t = text(), l = limits();
    assert.deepEqual(boundOutput(t, l), piBoundOutput(t, l), JSON.stringify({ t, l }));
  }
});

test('boundContent and truncated equal pi', () => {
  for (let i = 0; i < 2000; i++) {
    const content = Array.from({ length: Math.floor(random() * 4) }, () => random() < 0.8 ? { type: 'text', text: text() } :
      { type: 'image', data: 'AAAA', mimeType: 'image/png' });
    const l = limits();
    assert.deepEqual(boundContent(content, l), piBoundContent(content, l));
  }
  for (const retain of ['head', 'tail']) assert.deepEqual(truncated({ droppedBytes: 5, droppedLines: 2 }, retain), piTruncated({ droppedBytes: 5, droppedLines: 2 }, retain));
  assert.deepEqual(truncated({ droppedBytes: 5, droppedLines: 2 }, null), piTruncated({ droppedBytes: 5, droppedLines: 2 }));
});

test('fromSlot equals pi', () => {
  const slots = [undefined, {}, { output: '' }, { output: 'partial\n', droppedBytes: 10, droppedLines: 3, details: { a: 1 },
    diagnostics: [{ severity: 'info', message: 'spill at /tmp/x' }] }, { output: 'x', droppedBytes: 0 }];
  for (const slot of slots) assert.deepEqual(fromSlot(slot ?? null, 'aborted', 'Tool bash was aborted'), piFromSlot(slot, 'aborted', 'Tool bash was aborted'));
});

test('assemble then bound gives pi final result without hooks', () => {
  const limit = { maxBytes: 20, maxLines: 3, retain: 'head' };
  const execution = { retained: { text: 'one\ntwo\nthree\n', droppedBytes: 9, droppedLines: 2 }, diagnostics: [{ severity: 'info', message: 'running' }], details: { n: 1 } };
  const own = assemble({ diagnostics: [{ severity: 'warn', message: 'own' }] }, execution);
  assert.deepEqual(own.result.content, [{ type: 'text', text: 'one\ntwo\nthree\n' }]);
  assert.deepEqual(own.result.details, { n: 1 });
  const final = bound(own.result, limit, own.retained);
  assert.deepEqual(final.diagnostics.map(item => item.message), ['running', 'own', 'Output truncated to its beginning: 2 lines, 9 bytes dropped']);
  const explicit = assemble({ content: [{ type: 'text', text: 'a\n'.repeat(10) }], details: { m: 2 } }, execution);
  assert.equal(explicit.retained, null);
  const bounded = bound(explicit.result, limit, explicit.retained);
  assert.deepEqual(bounded.content, [{ type: 'text', text: 'a\na\na\n' }]);
  assert.deepEqual(bounded.details, { m: 2 });
  assert.equal(bounded.diagnostics.at(-1).message, 'Output truncated to its beginning: 7 lines, 14 bytes dropped');
  // A hook that replaced the retained content: no retained-truncation diagnostic.
  const replaced = bound({ ...own.result, content: [{ type: 'text', text: 'short' }] }, limit, own.retained);
  assert.deepEqual(replaced.diagnostics.map(item => item.message), ['running', 'own']);
});
