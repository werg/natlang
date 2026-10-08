import test from 'node:test';
import assert from 'node:assert/strict';

const { checkFiles, checkOracle, checkFileReturn, jsonStringRecordWithNumericKeysCanonical } = await import(
  process.env.NATLANG_ORACLES_MODULE ?? '../dist/evaluation/oracles.js');

test('explicit numeric answer-map keys compare by exact decimal value only', async () => {
  const spec = { level: 'normalized', normalization: 'json-string-record', numeric_keys: ['item-02', 'item-04'] };
  const actual = { 'item-02': '175', 'item-03': 'A-175', 'item-04': '3.000' };
  const expected = { 'item-02': '175.0', 'item-03': 'A-175', 'item-04': '3' };
  const verdict = await checkOracle(actual, expected, spec);
  assert.equal(verdict.accepted, true);
  assert.equal(verdict.comparison_version, 'json-string-record-decimal-fields/1');
  assert.equal((await checkOracle({ ...actual, 'item-03': 'A-175.0' }, expected, spec)).accepted, false);
  assert.equal((await checkOracle({ ...actual, 'item-02': '175.0000000000000000001' }, expected, spec)).accepted, false);
});

test('plain returned JSON records match their serialized form without loosening strings or key sets', async () => {
  const spec = { level: 'normalized', normalization: 'json-string-record' };
  const verdict = await checkOracle({ a: '175.0' }, '{"a":"175.0"}', spec);
  assert.equal(verdict.accepted, true);
  assert.equal(verdict.comparison_version, 'json-string-record-object/1');
  assert.equal((await checkOracle({ a: '175' }, '{"a":"175.0"}', spec)).accepted, false);
  assert.equal((await checkOracle({ a: '175.0', b: 'extra' }, '{"a":"175.0"}', spec)).accepted, false);
});

test('numeric whitelist must be unique, present on both maps, and decimal-valued', async () => {
  const base = { level: 'normalized', normalization: 'json-string-record', numeric_keys: ['answer'] };
  assert.equal((await checkOracle({ answer: '3' }, { answer: '3.0' }, base)).accepted, true);
  assert.equal((await checkOracle({}, {}, base)).accepted, false);
  assert.equal((await checkOracle({ answer: '3' }, {}, base)).accepted, false);
  assert.equal((await checkOracle({ answer: 'three' }, { answer: '3.0' }, base)).accepted, false);
  assert.equal(jsonStringRecordWithNumericKeysCanonical('{"answer":"1","answer":"1.0"}', ['answer']), null);
  assert.equal(jsonStringRecordWithNumericKeysCanonical('{"answer":"NaN"}', ['answer']), null);
  assert.equal(jsonStringRecordWithNumericKeysCanonical('{"answer":"1"}', ['answer', 'answer']), null);
});

test('file map and returned object use the same decimal-key contract without loosening source files', () => {
  const input = { 'task.json': '{"kind":"table"}\n', 'answers.json': '{}\n' };
  const expected = { ...input, 'answers.json': '{"item-02":"175.0","item-03":"A-175"}\n' };
  const actual = { ...input, 'answers.json': '{ "item-03": "A-175", "item-02": "175" }\n' };
  const spec = { compare: 'json-string-record', threshold: 1, numeric_keys: ['item-02'] };
  const result = checkFiles(actual, expected, input, spec);
  assert.equal(result.accepted, true);
  assert.equal(result.comparison_version, 'json-string-record-decimal-fields/1');
  assert.equal(checkFileReturn({ 'item-02': '175', 'item-03': 'A-175' }, actual, input, spec), true);
  assert.equal(checkFileReturn({ 'item-02': '175', 'item-03': 'a-175' }, actual, input, spec), false);
  assert.equal(checkFiles({ ...actual, 'task.json': 'changed' }, expected, input, spec).accepted, false);
  assert.equal(checkFiles(actual, expected, input, { compare: 'json-string-record', threshold: 1 }).accepted, false);
});
