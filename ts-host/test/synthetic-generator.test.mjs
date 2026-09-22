import test from 'node:test';
import assert from 'node:assert/strict';
import { generateSyntheticRecord, NATIVE_SYNTHETIC_FAMILIES, syntheticRecords, validateSyntheticRecord } from '../dist/teacher/synthetic-generator.js';

test('native synthetic programs are deterministic and independently indexed', () => {
  for (const family of NATIVE_SYNTHETIC_FAMILIES) {
    const first = generateSyntheticRecord(902, 17, family);
    assert.deepEqual(first, generateSyntheticRecord(902, 17, family));
    assert.notDeepEqual(first, generateSyntheticRecord(902, 18, family));
    assert.doesNotThrow(() => validateSyntheticRecord(first));
  }
});

test('native synthetic batch uses global-index round robin and emits valid IR', () => {
  const rows = syntheticRecords(31, 2, 5);
  assert.deepEqual(rows.map(row => row.family === 'algo_algorithm_pipeline' ? 'pipeline' :
    row.family === 'algo_staged_ranking' ? 'ranking' : 'array'), ['pipeline', 'array', 'ranking', 'pipeline', 'array']);
  assert.equal(rows[1].semantics.operation, 'algorithm');
  assert.equal(rows[4].semantics.operation, 'algorithm');
  rows.forEach(row => assert.doesNotThrow(() => validateSyntheticRecord(row)));
});

test('invalid ranges and unsupported families fail early', () => {
  assert.throws(() => syntheticRecords(0, -1, 1), /nonnegative/);
  assert.throws(() => generateSyntheticRecord(0, 0, 'judge'), /unsupported native family/);
});
