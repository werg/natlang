import assert from 'node:assert/strict';
import {test} from 'node:test';
import {checkOracle} from '../dist/teacher/oracle.js';

const accepted = async (actual, expected, oracle) => (await checkOracle(actual, expected, oracle)).accepted;

test('result-set compares query rows as a multiset with typed numbers', async () => {
  const oracle = {level: 'normalized', normalization: 'result-set'};
  assert.equal(await accepted([[3, 'x'], [1, 'y']], [[1, 'y'], [3, 'x']], oracle), true);
  assert.equal(await accepted([['3', 'x'], [1.0, 'y']], [[1, 'y'], [3, 'x']], oracle), true);
  assert.equal(await accepted(5, [[5]], oracle), true);
  assert.equal(await accepted([[3, 'x']], [[1, 'y'], [3, 'x']], oracle), false);
  assert.equal(await accepted([[3, 'x'], [3, 'x']], [[3, 'x']], oracle), false);
  assert.equal(await accepted([['x', 3]], [[3, 'x']], oracle), false);
  assert.equal(await accepted([[2], [1]], [[1], [2]], {...oracle, ordered: true}), false);
});

test('numeric-tolerance accepts close numbers and declared percent equivalence only', async () => {
  const oracle = {level: 'normalized', normalization: 'numeric-tolerance', tolerance: 1e-3};
  assert.equal(await accepted(3.1416, 3.14159, oracle), true);
  assert.equal(await accepted('$1,200', 1200, oracle), true);
  assert.equal(await accepted(3.2, 3.14159, oracle), false);
  assert.equal(await accepted(12.5, 0.125, oracle), false);
  assert.equal(await accepted('12.5%', 0.125, {...oracle, percent_equivalent: true}), true);
  assert.equal(await accepted('many', 3, oracle), false);
});

test('answer-list compares short answers without order', async () => {
  const oracle = {level: 'normalized', normalization: 'answer-list'};
  assert.equal(await accepted(['b', 'A.'], ['a', 'b'], oracle), true);
  assert.equal(await accepted('2004', ['2004'], oracle), true);
  assert.equal(await accepted(['a'], ['a', 'b'], oracle), false);
  assert.equal(await accepted(['a', 'a'], ['a', 'b'], oracle), false);
});

test('qa-alias-map needs every key to match one of its aliases', async () => {
  const oracle = {level: 'normalized', normalization: 'qa-alias-map'};
  const gold = {q1: ['Saint Bernadette Soubirous'], q2: ['1858', 'in 1858']};
  assert.equal(await accepted({q1: 'saint bernadette soubirous', q2: 'in 1858'}, gold, oracle), true);
  assert.equal(await accepted({q1: 'Saint Bernadette Soubirous', q2: '1859'}, gold, oracle), false);
  assert.equal(await accepted({q1: 'Saint Bernadette Soubirous'}, gold, oracle), false);
  assert.equal(await accepted({q1: 'Saint Bernadette Soubirous', q2: '1858', q3: 'x'}, gold, oracle), false);
});
