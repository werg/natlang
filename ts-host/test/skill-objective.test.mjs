import test from 'node:test';
import assert from 'node:assert/strict';
import { exactObjectiveBounds, scoreSkillObjective } from '../src/skills/objective.ts';

test('knapsack quality is recomputed from selected IDs and infeasibility is a separate gate', () => {
  const instance = { capacity: 5, items: [{ id: 'a', weight: 3, value: 7 }, { id: 'b', weight: 2, value: 4 }, { id: 'c', weight: 4, value: 8 }] };
  const bounds = exactObjectiveBounds('knapsack', instance);
  assert.deepEqual(bounds, { kind: 'objective-bound', worst: 0, best: 11 });
  assert.equal(scoreSkillObjective('knapsack', JSON.stringify(instance), { selectedIds: ['a'] , objective: 999, feasible: false }, bounds).quality, 7 / 11);
  const bad = scoreSkillObjective('knapsack', instance, { selectedIds: ['a', 'c'] }, bounds);
  assert.equal(bad.quality, 0); assert.equal(bad.gates.feasible, false);
  assert.equal(scoreSkillObjective('knapsack', instance, { selectedIds: ['a'] }, { ...bounds, best: 999 }).gates.reference_bounds_match, false);
});

test('bin packing verifies exact coverage/capacity and scores a feasible nonoptimal packing', () => {
  const instance = { capacity: 10, items: [{ id: 'a', size: 6 }, { id: 'b', size: 4 }, { id: 'c', size: 5 }, { id: 'd', size: 5 }] };
  const bounds = exactObjectiveBounds('bin-packing', instance);
  assert.deepEqual(bounds, { kind: 'objective-bound', worst: 4, best: 2 });
  const score = scoreSkillObjective('bin-packing', instance, { bins: [{ itemIds: ['a'] }, { itemIds: ['b'] }, { itemIds: ['c', 'd'] }] }, bounds);
  assert.equal(score.quality, 0.5); assert.equal(score.gates.feasible, true);
  assert.equal(scoreSkillObjective('bin-packing', instance, { bins: [{ itemIds: ['a', 'b', 'c'] }] }, bounds).gates.feasible, false);
});

test('weighted tardiness checks a full permutation and recomputes completion costs', () => {
  const instance = { jobs: [{ id: 'a', processing: 3, due: 3, weight: 3 }, { id: 'b', processing: 2, due: 2, weight: 1 }, { id: 'c', processing: 1, due: 6, weight: 2 }] };
  const bounds = exactObjectiveBounds('weighted-tardiness', instance);
  assert.equal(scoreSkillObjective('weighted-tardiness', instance, { order: ['b', 'a', 'c'], objective: 999 }, bounds).gates.feasible, true);
  assert.equal(scoreSkillObjective('weighted-tardiness', instance, { order: ['b', 'b', 'c'] }, bounds).gates.feasible, false);
  assert.equal(scoreSkillObjective('weighted-tardiness', instance, 'not-json', bounds).gates.feasible, false);
});
