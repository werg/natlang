import test from 'node:test';
import assert from 'node:assert/strict';
import { agreement, checkFiles, checkFilesWithJudge, checkFileReturn, csvRows, checkOracle, spanF1 } from '../dist/teacher/oracle.js';

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
    { accepted: false, level: 'judged', verdict: 'no judge supplied', needs_review: true });
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
  assert.deepEqual([moved.passed, moved.items, moved.accepted], [3, 5, false]);
  const moveItems = checkFiles({ 'a/1.md': 'one', 'a/2.md': 'two', 'keep.md': 'k' }, expected, input, { compare: 'moves', threshold: 0.5 });
  assert.deepEqual([moveItems.passed, moveItems.items, moveItems.accepted], [2, 3, true]);
  assert.equal(checkFiles({ ...expected, 'keep.md': 'changed' }, expected, input).accepted, false);
  const draft = '---\nid: x\n---\nThe cat sat on the mat and was happy.';
  const target = '---\nid: x\n---\nThe cat, happy, sat on the mat.';
  const rewrite = { compare: 'rewrite' };
  assert.equal(checkFiles({ 'd.md': '---\nid: x\n---\nOn the mat the cat sat, happy.' }, { 'd.md': target }, { 'd.md': draft }, rewrite).accepted, false, 'overlap is not proof of a valid rewrite');
  assert.equal(checkFiles({ 'd.md': draft }, { 'd.md': target }, { 'd.md': draft }, rewrite).accepted, false, 'unchanged');
  assert.equal(checkFiles({ 'd.md': '---\nid: y\n---\nOn the mat the cat sat.' }, { 'd.md': target }, { 'd.md': draft }, rewrite).accepted,
    false, 'front matter changed');
  const csv = { compare: 'csv', span: 0.5, threshold: 0.6 };
  const gold = { 'c.csv': 'id,clause\n"a","shall not compete within two years"\n"b",""\n"c","no solicitation of employees"\n' };
  const got = { 'c.csv': 'id,clause\na,The party shall not compete within two years\nb,\nc,\n' };
  const checked = checkFiles(got, gold, {}, csv);
  assert.equal(checked.accepted, false, 'missing half the positive rows is not rescued by empty rows');
  assert.equal(checked.positive_recall, 0.5);
  assert.equal(checked.score, 2 / 3);
  const counts = checkFiles({ 'INDEX.md': 'a: 12\nb: 4\n' }, { 'INDEX.md': 'a: 11\nb: 5\n' }, {}, { compare: 'counts' });
  assert.equal(counts.accepted, true);
});


test('CSV contracts reject malformed quoting, row widths, duplicate ids and hidden source changes', () => {
  const expected = { 'r.csv': 'id,clause\na,shall not compete\nb,\n' };
  for (const report of ['id,clause\na,"unterminated', 'id,clause\na,x,y\nb,\n',
    'id,clause\na,shall not compete\na,shall not compete\nb,\n', 'id,clause\na,"x"junk\nb,\n'])
    assert.equal(checkFiles({ 'r.csv': report }, expected, {}, { compare: 'csv' }).accepted, false);
  assert.deepEqual(csvRows('id,clause\r\na,"first\nsecond ""quoted"""\r\n'), [['id', 'clause'], ['a', 'first\nsecond "quoted"']]);
  assert.equal(checkFiles({ ...expected, 'source.md': 'corrupted' }, { ...expected, 'source.md': 'source' },
    { 'source.md': 'source' }, { compare: 'csv', threshold: 0.1 }).accepted, false);
});

test('extraction requires positive recall, precision, literal source quotes and consistent returned count', () => {
  const input = { 'a.md': 'A shall not compete with B.', 'b.md': 'No clause here.' };
  const expected = { ...input, 'r.csv': 'id,clause\na,A shall not compete with B.\nb,\n' };
  const spec = { compare: 'csv', threshold: 0.8, quote_sources: { a: 'a.md', b: 'b.md' },
    return_count: 'csv_nonempty', report: 'r.csv' };
  assert.equal(checkFiles({ ...input, 'r.csv': 'id,clause\na,\nb,\n' }, expected, input, spec).accepted, false);
  assert.equal(checkFiles({ ...input, 'r.csv': 'id,clause\na,A shall not compete with B.\nb,A shall not compete with B.\n' },
    expected, input, spec).accepted, false);
  assert.equal(checkFiles(expected, expected, input, spec).accepted, true);
  assert.equal(checkFileReturn(1, expected, input, spec), true);
  assert.equal(checkFileReturn(2, expected, input, spec), false);
  assert.equal(checkFileReturn(1, { 'r.csv': 'id,clause\na,x\na,x\n' }, input, spec), false);
});

test('rewrites accept reference alternatives or a recorded external rubric verdict, never overlap alone', async () => {
  const input = { 'd.md': '---\nid: x\n---\nThe cat is happy.' };
  const gold = { 'd.md': '---\nid: x\n---\nThe cat feels happy.' };
  const changed = { 'd.md': '---\nid: x\n---\nThe cat is not happy.' };
  const spec = { compare: 'rewrite', rubric: 'Keep meaning and improve clarity.', return_count: 'changed' };
  assert.equal(checkFiles(changed, gold, input, spec).accepted, false);
  assert.equal((await checkFilesWithJudge(changed, gold, input, spec)).accepted, false);
  const verdict = await checkFilesWithJudge(changed, gold, input, spec, async request => {
    assert.equal(request.actual.original, 'The cat is happy.');
    return { accepted: false, verdict: 'The candidate reverses the assertion.' };
  });
  assert.equal(verdict.accepted, false);
  assert.match(verdict.judgments['d.md'].verdict, /reverses/);
  assert.equal((await checkFilesWithJudge(input, gold, input, spec,
    async () => ({ accepted: true, verdict: 'Original already satisfies the instruction.' }))).accepted, true);
  assert.equal(checkFileReturn(0, input, input, spec), true);
  assert.equal(checkFileReturn(1, input, input, spec), false);
});

test('count reports reject ignored garbage, negatives and duplicate labels; invalid agreement fails closed', async () => {
  for (const text of ['a: 1\nignored', 'a: -1', 'a: 1\na: 1'])
    assert.equal(checkFiles({ 'INDEX.md': text }, { 'INDEX.md': 'a: 1' }, {}, { compare: 'counts' }).accepted, false);
  assert.equal(agreement(NaN, NaN), 0);
  assert.equal(agreement({ a: Infinity }, { a: Infinity }), 0);
  await assert.rejects(checkOracle(1, 1, { level: 'agreement', threshold: -0.1 }), /threshold/);
  assert.throws(() => checkFiles({}, {}, {}, { threshold: NaN }), /threshold/);
});

test('QA answer normalization ignores articles but never accepts an embellished yes/no', async () => {
  const spec = { level: 'span', threshold: 0.8, normalization: 'qa' };
  assert.equal((await checkOracle('the United Kingdom', 'United Kingdom', spec)).accepted, true);
  assert.equal((await checkOracle('yes but no', 'yes', spec)).accepted, false);
  assert.equal((await checkOracle('Friday', 'May 26, 2000', spec)).accepted, false);
});


test('borderline short answers may use an independent source-aware rubric, not unchecked substring acceptance', async () => {
  const spec = { level: 'span', threshold: 0.8, normalization: 'qa', rubric: 'Check the count and its unit.',
    context: { question: 'How many surgeries?', supporting_text: 'The person underwent 14 surgeries.' } };
  assert.equal((await checkOracle('14 surgeries', '14', spec)).accepted, false);
  const accepted = await checkOracle('14 surgeries', '14', spec, async request => {
    assert.equal(request.expected.answer, '14');
    assert.equal(request.expected.context.question, 'How many surgeries?');
    return { accepted: true, verdict: 'The correct count is accompanied by the correct unit.' };
  });
  assert.equal(accepted.accepted, true);
  assert.match(accepted.verdict, /correct unit/);
  assert.equal((await checkOracle('May 25, 2000', 'May 26, 2000', spec)).accepted, false);
});

test('an overlapping alternate contract quote needs an independent semantic verdict', async () => {
  const input = { 'a.md': 'The party shall not compete with B. The party shall not solicit employees of B.' };
  const expected = { ...input, 'r.csv': 'id,clause\na,The party shall not compete with B.\n' };
  const actual = { ...input, 'r.csv': 'id,clause\na,The party shall not solicit employees of B.\n' };
  const spec = { compare: 'csv', quote_sources: { a: 'a.md' }, rubric: 'Only non-compete; exclude non-solicitation.' };
  const first = checkFiles(actual, expected, input, spec);
  assert.equal(first.accepted, false);
  assert.ok(first.pending.includes('unverified_quote:a'));
  const graded = await checkFilesWithJudge(actual, expected, input, spec,
    async () => ({ accepted: false, verdict: 'This is non-solicitation, not non-compete.' }));
  assert.equal(graded.accepted, false);
  assert.equal(graded.judgments['r.csv:a'].accepted, false);
});


test('insufficient judging evidence remains pending and cannot become an acceptance', async () => {
  const input = { 'd.md': 'The cat is happy.' }, expected = { 'd.md': 'The cat feels happy.' };
  const pending = await checkFilesWithJudge({ 'd.md': 'The happy cat.' }, expected, input,
    { compare: 'rewrite', rubric: 'Preserve meaning.' }, async () => ({ accepted: true, needs_review: true, verdict: 'Cannot establish meaning.' }));
  assert.equal(pending.accepted, false);
  assert.deepEqual(pending.pending, ['unverified_rewrite:d.md']);
  assert.equal((await checkOracle('x', 'y', 'judged',
    async () => ({ accepted: true, needs_review: true, verdict: 'Insufficient evidence.' }))).accepted, false);
});
