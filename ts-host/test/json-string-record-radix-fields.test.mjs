import test from 'node:test';
import assert from 'node:assert/strict';

const { checkFiles, checkOracle, checkFileReturn, jsonStringRecordWithRadixFieldsCanonical } = await import(
  process.env.NATLANG_ORACLES_MODULE ?? '../dist/evaluation/oracles.js');

test('explicit radix fields compare exact unsigned integer values in their declared base', async () => {
  const fields = [{ key: 'answer', base: 25 }];
  const actual = { answer: '02743f', label: 'Keep Case' };
  const expected = { answer: '2743f', label: 'Keep Case' };
  const verdict = await checkOracle(actual, expected, { level: 'normalized', normalization: 'json-string-record', radix_fields: fields });
  assert.equal(verdict.accepted, true);
  assert.equal(verdict.comparison_version, 'json-string-record-radix-fields/1');
  assert.equal((await checkOracle({ ...actual, label: 'keep case' }, expected,
    { level: 'normalized', normalization: 'json-string-record', radix_fields: fields })).accepted, false);
});

test('radix comparison rejects invalid digits, wrong bases, uppercase, signs, missing and duplicate metadata', async () => {
  assert.equal(jsonStringRecordWithRadixFieldsCanonical({ answer: '02743f' }, [{ key: 'answer', base: 25 }]),
    jsonStringRecordWithRadixFieldsCanonical({ answer: '2743f' }, [{ key: 'answer', base: 25 }]));
  assert.equal(jsonStringRecordWithRadixFieldsCanonical({ answer: 'z' }, [{ key: 'answer', base: 25 }]), null);
  assert.equal(jsonStringRecordWithRadixFieldsCanonical({ answer: 'A' }, [{ key: 'answer', base: 25 }]), null);
  assert.equal(jsonStringRecordWithRadixFieldsCanonical({ answer: '-1' }, [{ key: 'answer', base: 25 }]), null);
  assert.equal(jsonStringRecordWithRadixFieldsCanonical({}, [{ key: 'answer', base: 25 }]), null);
  assert.equal(jsonStringRecordWithRadixFieldsCanonical({ answer: '10' }, [{ key: 'answer', base: 1 }]), null);
  assert.equal(jsonStringRecordWithRadixFieldsCanonical({ answer: '10' }, [
    { key: 'answer', base: 10 }, { key: 'answer', base: 16 },
  ]), null);
  const conflicting = { level: 'normalized', normalization: 'json-string-record', numeric_keys: ['answer'],
    radix_fields: [{ key: 'answer', base: 10 }] };
  assert.equal((await checkOracle({ answer: '10' }, { answer: '10' }, conflicting)).accepted, false);
  assert.equal((await checkOracle({ answer: '10' }, { answer: '10' },
    { level: 'normalized', normalization: 'json-string-record', radix_fields: {} })).accepted, false);
});

test('saved answer files and returned records share only their declared radix equivalence', () => {
  const input = { 'task.json': '{}\n', 'answers.json': '{}\n' };
  const expected = { ...input, 'answers.json': '{"answer":"2743f","label":"Exact"}\n' };
  const actual = { ...input, 'answers.json': '{ "label": "Exact", "answer": "02743f" }\n' };
  const spec = { compare: 'json-string-record', threshold: 1, radix_fields: [{ key: 'answer', base: 25 }] };
  const verdict = checkFiles(actual, expected, input, spec);
  assert.equal(verdict.accepted, true);
  assert.equal(verdict.comparison_version, 'json-string-record-radix-fields/1');
  assert.equal(checkFileReturn({ answer: '02743f', label: 'Exact' }, actual, input, spec), true);
  assert.equal(checkFileReturn({ answer: '02743f', label: 'exact' }, actual, input, spec), false);
  assert.equal(checkFiles({ ...actual, 'task.json': '{"mutated":true}\n' }, expected, input, spec).accepted, false);
});
