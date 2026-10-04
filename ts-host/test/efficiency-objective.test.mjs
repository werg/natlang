import test from 'node:test';
import assert from 'node:assert/strict';
import { exactEfficiencyObjectiveBounds, scoreEfficiencyObjective, EFFICIENCY_OBJECTIVE_KINDS } from '../src/skills/efficiency-objective.ts';

function bruteDnfBounds(instance) {
  const n = instance.variables.length, all = [];
  for (let code = 0; code < 3 ** n; code++) {
    let rest = code, pattern = '';
    for (let i = 0; i < n; i++) { pattern = ['0', '1', '-'][rest % 3] + pattern; rest = Math.floor(rest / 3); }
    const rows = [];
    let valid = true;
    for (let row = 0; row < 2 ** n; row++) {
      const bits = row.toString(2).padStart(n, '0');
      if ([...pattern].every((symbol, i) => symbol === '-' || symbol === bits[i])) {
        if (!instance.truthTable[row]) valid = false;
        rows.push(row);
      }
    }
    if (valid && rows.length) all.push({ pattern, rows, cost: instance.termCost + [...pattern].filter(x => x !== '-').length * instance.literalCost });
  }
  const on = new Set(instance.truthTable.flatMap((value, row) => value ? [row] : []));
  const feasible = [];
  for (let mask = 0; mask < 2 ** all.length; mask++) {
    const cover = new Set(), terms = [];
    all.forEach((term, i) => { if (mask & (1 << i)) { terms.push(term); term.rows.forEach(row => cover.add(row)); } });
    if (cover.size === on.size && [...on].every(row => cover.has(row))) feasible.push({ cost: terms.reduce((sum, term) => sum + term.cost, 0), terms });
  }
  return { feasible, best: Math.min(...feasible.map(row => row.cost)), worst: Math.max(...feasible.map(row => row.cost)) };
}

test('Boolean DNF min and max bounds equal independent enumeration of all canonical terms', () => {
  const instance = { variables: ['a', 'b'], truthTable: [true, true, true, false], literalCost: 1, termCost: 1 };
  const brute = bruteDnfBounds(instance), bounds = exactEfficiencyObjectiveBounds('boolean-dnf', instance);
  assert.deepEqual([bounds.best, bounds.worst], [brute.best, brute.worst]);
  assert.equal(brute.best, 4);
  assert.equal(scoreEfficiencyObjective('boolean-dnf', instance, { terms: ['0-', '-0'] }, bounds).quality, 1);
  const partialQuality = scoreEfficiencyObjective('boolean-dnf', instance, { terms: ['00', '01', '10'] }, bounds);
  assert.equal(partialQuality.gates.correct, true);
  assert.equal(partialQuality.objective, 9);
  assert.ok(partialQuality.quality > 0 && partialQuality.quality < 1);
});

test('DNF correctness is a hard gate and duplicate or malformed terms are rejected', () => {
  const instance = { variables: ['p', 'q'], truthTable: [true, true, true, false], literalCost: 2, termCost: 1 };
  const bounds = exactEfficiencyObjectiveBounds('boolean-dnf', instance);
  assert.equal(scoreEfficiencyObjective('boolean-dnf', instance, { terms: ['0-'] }, bounds).gates.truth_table_equivalence, false);
  assert.equal(scoreEfficiencyObjective('boolean-dnf', instance, { terms: ['0-', '0-'] }, bounds).gates.valid_implicants, false);
  assert.equal(scoreEfficiencyObjective('boolean-dnf', instance, { terms: ['0?'] }, bounds).gates.valid_implicants, false);
  assert.equal(scoreEfficiencyObjective('boolean-dnf', instance, { terms: ['0-', '-0'] }, { ...bounds, best: bounds.best + 1 }).gates.reference_bounds_match, false);
});

test('constant functions, string inputs, and bounded truth tables are handled exactly', () => {
  assert.deepEqual([...EFFICIENCY_OBJECTIVE_KINDS], ['boolean-dnf']);
  const allFalse = { variables: ['x'], truthTable: [false, false], literalCost: 2, termCost: 1 };
  const falseBounds = exactEfficiencyObjectiveBounds('boolean-dnf', JSON.stringify(allFalse));
  assert.deepEqual([falseBounds.best, falseBounds.worst], [0, 0]);
  assert.equal(scoreEfficiencyObjective('boolean-dnf', allFalse, { terms: [] }, falseBounds).quality, 1);
  const allTrue = { variables: ['x'], truthTable: [true, true], literalCost: 2, termCost: 1 };
  const trueBounds = exactEfficiencyObjectiveBounds('boolean-dnf', allTrue);
  assert.equal(scoreEfficiencyObjective('boolean-dnf', allTrue, { terms: ['-'] }, trueBounds).quality, 1);
  assert.throws(() => exactEfficiencyObjectiveBounds('boolean-dnf', { variables: ['a', 'b', 'c', 'd', 'e'], truthTable: Array(32).fill(true), literalCost: 1, termCost: 1 }), /oversized/);
});
