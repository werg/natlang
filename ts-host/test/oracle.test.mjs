import test from 'node:test';
import assert from 'node:assert/strict';
import { agreement, checkFiles, checkOracle, spanF1 } from '../dist/teacher/oracle.js';

test('normalized oracle accepts equivalent casing, spacing, numbers and dates', async () => {
  assert.equal((await checkOracle('  YES   Please ', 'yes please', 'normalized')).accepted, true);
  assert.equal((await checkOracle('1,250', 1250, 'normalized')).accepted, true);
  assert.equal((await checkOracle('09/27/2026', '2026-09-27', 'normalized')).accepted, true);
  assert.equal((await checkOracle('different', 'answer', { level: 'normalized', alternates: [' Different '] })).accepted, true);
  assert.equal((await checkOracle('no', 'yes', 'normalized')).accepted, false);
});

test('span oracle scores overlap and enforces its threshold', async () => {
  assert.equal(spanF1('pay within 30 days', 'within 30 days'), 6 / 7);
  const verdict = await checkOracle('pay within 30 days', 'within 30 days', { level: 'span', threshold: 0.8 });
  assert.equal(verdict.accepted, true);
  assert.equal(verdict.score, 6 / 7);
  assert.equal((await checkOracle('no overlap', 'within 30 days', 'span')).accepted, false);
  await assert.rejects(checkOracle('x', 'x', { level: 'span', threshold: 2 }), /threshold/);
});

test('judged cases need an external verdict and record it', async () => {
  assert.deepEqual(await checkOracle('summary', 'gold', 'judged'),
    { accepted: false, level: 'judged', verdict: 'no judge supplied' });
  let seen;
  const verdict = await checkOracle('summary', 'gold', { level: 'judged', rubric: 'Cover the decision.' },
    async input => { seen = input; return { accepted: true, verdict: 'Decision is covered.' }; });
  assert.deepEqual(seen, { actual: 'summary', expected: 'gold', rubric: 'Cover the decision.' });
  assert.deepEqual(verdict, { accepted: true, level: 'judged', verdict: 'Decision is covered.' });
});

test('agreement scores many-part answers and accepts at its threshold', async () => {
  assert.equal(agreement({ a: 12, b: 4 }, { a: 11, b: 5 }), 1 - 2 / 32);
  assert.equal(agreement([1, 2, 3, 4], [1, 2, 3, 5]), 0.75);
  assert.equal(agreement(95, 100), 0.95);
  assert.equal((await checkOracle({ a: 12, b: 4 }, { a: 11, b: 5 }, { level: 'agreement', threshold: 0.9 })).accepted, true);
  assert.equal((await checkOracle({ a: 16 }, { a: 11, b: 5 }, { level: 'agreement', threshold: 0.9 })).accepted, false);
});

test('a files oracle checks what should change and what did, item by item', () => {
  const input = { 'inbox/1.md': 'one', 'inbox/2.md': 'two', 'keep.md': 'k' };
  const expected = { 'a/1.md': 'one', 'b/2.md': 'two', 'keep.md': 'k' };
  // One file misplaced: of the four changed paths, two pass (inbox/1.md and a/1.md moved as expected).
  const moved = checkFiles({ 'a/1.md': 'one', 'a/2.md': 'two', 'keep.md': 'k' }, expected, input, { threshold: 0.5 });
  assert.deepEqual([moved.passed, moved.items, moved.accepted], [3, 5, true]);
  assert.equal(checkFiles({ ...expected, 'keep.md': 'changed' }, expected, input).accepted, false);
  const draft = '---\nid: x\n---\nThe cat sat on the mat and was happy.';
  const target = '---\nid: x\n---\nThe cat, happy, sat on the mat.';
  const rewrite = { compare: 'rewrite' };
  assert.equal(checkFiles({ 'd.md': '---\nid: x\n---\nOn the mat the cat sat, happy.' }, { 'd.md': target }, { 'd.md': draft }, rewrite).accepted, true);
  assert.equal(checkFiles({ 'd.md': draft }, { 'd.md': target }, { 'd.md': draft }, rewrite).accepted, false, 'unchanged');
  assert.equal(checkFiles({ 'd.md': '---\nid: y\n---\nOn the mat the cat sat.' }, { 'd.md': target }, { 'd.md': draft }, rewrite).accepted,
    false, 'front matter changed');
  const csv = { compare: 'csv', span: 0.5, threshold: 0.6 };
  const gold = { 'c.csv': 'id,clause\n"a","shall not compete within two years"\n"b",""\n"c","no solicitation of employees"\n' };
  const got = { 'c.csv': 'id,clause\na,The party shall not compete within two years\nb,\nc,\n' };
  assert.deepEqual(checkFiles(got, gold, {}, csv), { accepted: true, score: 2 / 3, passed: 2, items: 3, failed: ['c.csv:c'] });
  const counts = checkFiles({ 'INDEX.md': 'a: 12\nb: 4\n' }, { 'INDEX.md': 'a: 11\nb: 5\n' }, {}, { compare: 'counts' });
  assert.equal(counts.accepted, true);
});
