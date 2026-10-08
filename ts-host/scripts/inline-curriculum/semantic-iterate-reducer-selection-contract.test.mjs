import test from 'node:test';
import assert from 'node:assert/strict';
import { selectionContract } from './semantic-iterate-reducer-selection-contract.mjs';

test('singleton contract selects one deterministic tie winner and excludes other ties', () => {
  const contract = selectionContract({ count: 1, direction: 'desc', measureName: 'score' });
  const rows = [{ id: 'ID-B', metric: 9 }, { id: 'ID-A', metric: 9 }];
  const sorted = [...rows].sort((a, b) => b.metric - a.metric || a.id.localeCompare(b.id));
  assert.deepEqual(contract.select(sorted), [{ id: 'ID-A', metric: 9 }]);
  assert.match(contract.selectionFormat, /Exactly one/);
  assert.match(contract.instruction, /Do not add further tied items/);
  assert.deepEqual(contract.select([]), []);
  assert.equal(contract.format([]), 'none');
  assert.equal(contract.formatMeasure([]), 'none');
});

test('multi-item contract preserves rank and gives one measure per selected ID without summing', () => {
  const contract = selectionContract({ count: 2, direction: 'asc', measureName: 'cost' });
  const rows = [{ id: 'ID-C', metric: 12 }, { id: 'ID-A', metric: 8 }, { id: 'ID-B', metric: 8 }];
  const sorted = [...rows].sort((a, b) => a.metric - b.metric || a.id.localeCompare(b.id));
  const selected = contract.select(sorted);
  assert.deepEqual(selected.map(row => row.id), ['ID-A', 'ID-B']);
  assert.equal(contract.format(selected), 'ID-A; ID-B');
  assert.equal(contract.formatMeasure(selected), '8; 8');
  assert.match(contract.selectionFormat, /Up to 2/);
  assert.match(contract.measureFormat, /do not sum them/);
  assert.deepEqual(contract.select([]), []);
  assert.equal(contract.format([]), 'none');
});

test('selection cardinality is explicit and validated', () => {
  assert.throws(() => selectionContract({ count: 0, direction: 'desc', measureName: 'score' }), /positive integer/);
  assert.throws(() => selectionContract({ count: 2, direction: 'sideways', measureName: 'score' }), /direction/);
});
