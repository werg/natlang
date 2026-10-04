import test from 'node:test';
import assert from 'node:assert/strict';
import { exactExtendedObjectiveBounds, scoreExtendedObjective, EXTENDED_OBJECTIVE_KINDS } from '../src/skills/extended-objective.ts';

const enumerate = (n, visit) => {
  for (let mask = 0; mask < 2 ** n; mask++) visit(mask);
};

test('weighted set cover bounds match independent subset enumeration and rejects uncovered solutions', () => {
  const x = { universe: ['a', 'b', 'c'], sets: [
    { id: 'ab', cost: 3, elements: ['a', 'b'] }, { id: 'c', cost: 4, elements: ['c'] },
    { id: 'ac', cost: 2, elements: ['a', 'c'] }, { id: 'bc', cost: 1, elements: ['b', 'c'] },
  ] };
  const feasible = [];
  enumerate(x.sets.length, mask => {
    const selected = x.sets.filter((_, i) => mask & (1 << i));
    if (x.universe.every(id => selected.some(set => set.elements.includes(id))))
      feasible.push(selected.reduce((sum, set) => sum + set.cost, 0));
  });
  const bounds = exactExtendedObjectiveBounds('weighted-set-cover', x);
  assert.deepEqual([bounds.best, bounds.worst], [Math.min(...feasible), Math.max(...feasible)]);
  assert.equal(scoreExtendedObjective('weighted-set-cover', x, { selectedIds: ['ac', 'bc'] }, bounds).quality, 1);
  assert.equal(scoreExtendedObjective('weighted-set-cover', x, { selectedIds: ['ab'] }, bounds).gates.feasible, false);
});

test('budgeted max coverage scores feasible partial coverage against independently enumerated subsets', () => {
  const x = { budget: 4, universe: [{ id: 'a', weight: 5 }, { id: 'b', weight: 3 }, { id: 'c', weight: 2 }], items: [
    { id: 'x', cost: 2, elements: ['a', 'b'] }, { id: 'y', cost: 2, elements: ['b', 'c'] },
    { id: 'z', cost: 4, elements: ['a', 'c'] },
  ] };
  let best = 0;
  enumerate(x.items.length, mask => {
    const selected = x.items.filter((_, i) => mask & (1 << i));
    if (selected.reduce((sum, item) => sum + item.cost, 0) <= x.budget) {
      const covered = new Set(selected.flatMap(item => item.elements));
      best = Math.max(best, x.universe.filter(row => covered.has(row.id)).reduce((sum, row) => sum + row.weight, 0));
    }
  });
  const bounds = exactExtendedObjectiveBounds('budgeted-max-coverage', x);
  assert.deepEqual([bounds.best, bounds.worst], [best, 0]);
  const partial = scoreExtendedObjective('budgeted-max-coverage', x, { selectedIds: ['x'] }, bounds);
  assert.equal(partial.gates.feasible, true); assert.equal(partial.objective, 8); assert.equal(partial.quality, 0.8);
  assert.equal(scoreExtendedObjective('budgeted-max-coverage', x, { selectedIds: ['x', 'z'] }, bounds).gates.feasible, false);
});

test('facility location exact bounds enumerate all client assignments and verify recomputed opening cost', () => {
  const x = { clients: ['a', 'b'], facilities: [
    { id: 'f1', openCost: 1, serviceCosts: { a: 1, b: 5 } },
    { id: 'f2', openCost: 2, serviceCosts: { a: 4, b: 1 } },
  ] };
  const costs = [];
  for (const fa of x.facilities) for (const fb of x.facilities) {
    const used = new Set([fa, fb]);
    costs.push([...used].reduce((s, f) => s + f.openCost, 0) + fa.serviceCosts.a + fb.serviceCosts.b);
  }
  const bounds = exactExtendedObjectiveBounds('facility-location', x);
  assert.deepEqual([bounds.best, bounds.worst], [Math.min(...costs), Math.max(...costs)]);
  assert.equal(scoreExtendedObjective('facility-location', x, { assignments: { a: 'f1', b: 'f2' } }, bounds).objective, 5);
  assert.equal(scoreExtendedObjective('facility-location', x, { assignments: { a: 'f1' } }, bounds).gates.feasible, false);
});

test('bottleneck assignment bounds match all task permutations and reject duplicate assignments', () => {
  const x = { agents: ['a', 'b', 'c'], tasks: ['x', 'y', 'z'], costs: [[1, 6, 4], [4, 2, 5], [7, 3, 1]] };
  const permutations = [];
  const walk = (row, used, max) => {
    if (row === x.agents.length) { permutations.push(max); return; }
    for (let task = 0; task < x.tasks.length; task++) if (!used.has(task)) {
      used.add(task); walk(row + 1, used, Math.max(max, x.costs[row][task])); used.delete(task);
    }
  };
  walk(0, new Set(), 0);
  const bounds = exactExtendedObjectiveBounds('bottleneck-assignment', x);
  assert.deepEqual([bounds.best, bounds.worst], [Math.min(...permutations), Math.max(...permutations)]);
  assert.equal(scoreExtendedObjective('bottleneck-assignment', x, { assignments: { a: 'x', b: 'y', c: 'z' } }, bounds).objective, 2);
  assert.equal(scoreExtendedObjective('bottleneck-assignment', x, { assignments: { a: 'x', b: 'x', c: 'z' } }, bounds).gates.feasible, false);
});

test('matrix-chain bounds match independent parenthesization enumeration and score a valid nonoptimal tree', () => {
  const x = { dimensions: [10, 30, 5, 60] };
  const allCosts = (i, j) => {
    if (i === j) return [0];
    const costs = [];
    for (let k = i; k < j; k++) for (const left of allCosts(i, k)) for (const right of allCosts(k + 1, j))
      costs.push(left + right + x.dimensions[i] * x.dimensions[k + 1] * x.dimensions[j + 1]);
    return costs;
  };
  const costs = allCosts(0, 2), bounds = exactExtendedObjectiveBounds('matrix-chain', x);
  assert.deepEqual([bounds.best, bounds.worst], [Math.min(...costs), Math.max(...costs)]);
  const slower = { tree: { left: { matrix: 'A1' }, right: { left: { matrix: 'A2' }, right: { matrix: 'A3' } } } };
  assert.equal(scoreExtendedObjective('matrix-chain', x, slower, bounds).objective, 27000);
  assert.equal(scoreExtendedObjective('matrix-chain', x, slower, bounds).quality, 0);
  assert.equal(scoreExtendedObjective('matrix-chain', x, { tree: { matrix: 'A2' } }, bounds).gates.feasible, false);
});

test('extended objective classes reject malformed, out-of-range and oversized instances', () => {
  assert.deepEqual([...EXTENDED_OBJECTIVE_KINDS], ['weighted-set-cover', 'budgeted-max-coverage', 'facility-location', 'bottleneck-assignment', 'matrix-chain']);
  assert.throws(() => exactExtendedObjectiveBounds('weighted-set-cover', { universe: ['x'], sets: [{ id: 's', cost: 1, elements: [] }] }), /not coverable/);
  assert.throws(() => exactExtendedObjectiveBounds('budgeted-max-coverage', { budget: 2, universe: [{ id: 'x', weight: 1 }], items: Array.from({ length: 19 }, (_, i) => ({ id: `i${i}`, cost: 1, elements: ['x'] })) }), /oversized/);
  const x = { dimensions: [2, 3, 4] }, b = exactExtendedObjectiveBounds('matrix-chain', x);
  assert.equal(scoreExtendedObjective('matrix-chain', x, { tree: { left: { matrix: 'A1' }, right: { matrix: 'A2' } } }, { ...b, best: b.best + 1 }).gates.reference_bounds_match, false);
  assert.equal(scoreExtendedObjective('bottleneck-assignment', { agents: ['a'], tasks: ['t'], costs: [[1]] }, { assignments: { a: 't' } }, { kind: 'objective-bound', best: 1, worst: 0 }).gates.feasible, false);
});
