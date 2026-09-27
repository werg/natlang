import test from 'node:test';
import assert from 'node:assert/strict';
import { checkOracle, spanF1 } from '../dist/teacher/oracle.js';

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
