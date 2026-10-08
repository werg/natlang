import test from 'node:test';
import assert from 'node:assert/strict';
const { checkFiles, checkOracle, qaStringMapCanonical } = await import(
  process.env.NATLANG_ORACLES_MODULE ?? '../dist/evaluation/oracles.js');

test('QA string-map normalization is per answer value and keeps exact keys', async () => {
  const expected = { 'item-01': 'the Middleweight division', 'item-02': 'U.S. Army' };
  const actual = { 'item-01': 'Middleweight division', 'item-02': 'us army' };
  assert.equal((await checkOracle(actual, expected, { level: 'normalized', normalization: 'qa-string-map' })).accepted, true);
  assert.equal(qaStringMapCanonical({ 'item-01': 'the Middleweight division' }),
    qaStringMapCanonical({ 'item-01': 'Middleweight, Division!' }));
  assert.equal((await checkOracle({ 'item-1': 'Middleweight division' }, { 'item-01': 'the Middleweight division' },
    { level: 'normalized', normalization: 'qa-string-map' })).accepted, false);
  assert.equal((await checkOracle({ 'item-01': 42 }, { 'item-01': '42' },
    { level: 'normalized', normalization: 'qa-string-map' })).accepted, false);
  assert.equal(qaStringMapCanonical('{"item-01":"A","item-01":"B"}'), null);
});

test('QA answer-map file comparison normalizes answers while preserving file and map contracts', () => {
  const input = { 'task.json': '{"items":["item-01"]}\n', 'item-01.json': '{"question":"Q"}\n', 'answers.json': '{}\n' };
  const expected = { ...input, 'answers.json': '{"item-01":"the Middleweight division"}\n' };
  const equivalent = { ...input, 'answers.json': '{ "item-01": "Middleweight division" }\n' };
  const passed = checkFiles(equivalent, expected, input, { compare: 'qa-string-map' });
  assert.equal(passed.accepted, true);
  assert.equal(passed.quality_version, 3);
  assert.equal(passed.comparison_version, 'squad-token-map/1');
  assert.equal(checkFiles({ ...equivalent, 'task.json': 'changed' }, expected, input,
    { compare: 'qa-string-map' }).accepted, false);
  assert.equal(checkFiles({ ...equivalent, 'answers.json': '{"extra":"Middleweight division"}' }, expected, input,
    { compare: 'qa-string-map' }).accepted, false);
  assert.equal(checkFiles({ ...equivalent, 'answers.json': '{"item-01":"wrong"}' }, expected, input,
    { compare: 'qa-string-map' }).accepted, false);
});
