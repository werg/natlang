import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { abReplay, canonical, selectRecords } from '../scripts/ab-replay.mjs';

function folder(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-ab-'));
  for (const [path, content] of Object.entries(files)) { mkdirSync(join(root, path, '..'), { recursive: true }); writeFileSync(join(root, path), content); }
  return root;
}
const stage = body => `---\nargs:\n  word: string\nreturns: string\n---\n${body}\n`;
const record = (word, output, source = 'app/pick.nl') => ({ call_id: `c-${word}`, definition: { name: 'pick', source }, inputs: { word }, output, outcome: 'done' });

test('selectRecords keeps calls of the function, one per distinct input', () => {
  const rows = selectRecords([record('a', 'A'), record('a', 'A'), record('b', 'B'), record('c', 'C', 'other/pick.nl'), { ...record('d', 'D'), audit_of: 'x' }], 'app/pick.nl');
  assert.deepEqual(rows.map(row => row.inputs.word), ['a', 'b']);
  assert.equal(canonical({ b: 1, a: [2, { d: 3, c: undefined }] }), '{"a":[2,{"d":3}],"b":1}');
});

test('old against new prompt bytes on recorded inputs, scored by validity and the recorded value', async () => {
  const oldRoot = folder({ 'natlang.json': '{}', 'app/pick.nl': stage('OLD-PROMPT. Return word in upper case.') });
  const newRoot = folder({ 'natlang.json': '{}', 'app/pick.nl': stage('NEW-PROMPT. Return word in upper case.') });
  const seen = { old: 0, new: 0 };
  let turns = 0;
  const driver = ({ messages }) => {
    if (++turns > 100) throw new Error('the scripted model was asked too often');
    const text = JSON.stringify(messages);
    const word = /word: string = \\"([a-z])/.exec(text)?.[1] ?? '?';
    if (text.includes('OLD-PROMPT')) { seen.old++; return { calls: [['return_result', { status: 'success', value: word }]] }; }
    seen.new++;
    return { calls: [['return_result', { status: 'success', value: word.toUpperCase() }]] };
  };
  const summary = await abReplay({ records: [record('a', 'A'), record('b', 'B')], functionPath: 'app/pick.nl', oldRoot, newRoot, driver, samples: 6, concurrency: 3, label: 'pick' });
  assert.equal(summary.distinct_inputs, 2);
  assert.equal(summary.variants.old.n, 6);
  assert.equal(summary.variants.new.n, 6);
  assert.equal(summary.variants.old.ok, 6);
  assert.equal(summary.variants.new.ok, 6);
  assert.equal(summary.variants.new.exact, 6, JSON.stringify(summary.variants.new.rows.slice(0, 2)));
  assert.equal(summary.variants.old.exact, 0);
  assert.ok(seen.old >= 6 && seen.new >= 6);
  assert.match(summary.row, /^pick: old 6\/6 valid, 0\/6 equal to recorded/);
});

test('a call that ends without a valid value counts against its variant', async () => {
  const root = folder({ 'natlang.json': '{}', 'app/pick.nl': stage('Return word.') });
  let turns = 0;
  const driver = () => {
    if (++turns > 40) throw new Error('the scripted model was asked too often');
    return { calls: [['return_result', { status: 'failure', reason: 'cannot' }]] };
  };
  const summary = await abReplay({ records: [record('a', 'a')], functionPath: 'app/pick.nl', oldRoot: root, newRoot: root, driver, samples: 2, concurrency: 2 });
  assert.equal(summary.variants.old.ok, 0);
  assert.equal(summary.variants.new.exact, 0);
  assert.equal(Object.values(summary.variants.old.errors).reduce((a, b) => a + b, 0), 2, JSON.stringify(summary.variants.old.errors));
});
