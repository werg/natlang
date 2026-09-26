import assert from 'node:assert/strict';
import { test } from 'node:test';
import { migrateRow } from '../scripts/inline-curriculum/migrate-status.mjs';

const oldRow = () => ({ id: 'old', provenance: { model: 'm' },
  trajectory: [{ tools_offered: [{ type: 'function', function: { name: 'return_result', parameters: { properties: { value: { type: 'number' } } } } },
    { type: 'function', function: { name: 'failed', parameters: {} } }],
    model_response: { calls: [['failed', { message: 'impossible' }]], raw_calls: [{ function: { name: 'failed', arguments: '{"message": "impossible"}' } }] },
    assistant: { calls: [{ tool: 'failed', source_tool: 'failed', arguments: { message: 'impossible' } }] } }],
  outcome: { action_ledger: [{ name: 'failed', arguments: { message: 'impossible' } }, { name: 'return_result', arguments: { value: 3 } }] } });

test('status migration rewrites calls and the action ledger alike, and marks the row', () => {
  const row = migrateRow(oldRow());
  assert.deepEqual(row.trajectory[0].assistant.calls[0].arguments, { status: 'failed', reason: 'impossible' });
  assert.deepEqual(row.outcome.action_ledger, [{ name: 'return_result', arguments: { status: 'failed', reason: 'impossible' } },
    { name: 'return_result', arguments: { status: 'success', value: 3 } }]);
  assert.equal(row.provenance.finish_surface_migration, 'return_result-status/2');
});

test('status migration is idempotent and leaves current rows, raw text included, untouched', () => {
  const once = migrateRow(oldRow());
  assert.equal(migrateRow(once), once);
  const current = { id: 'new', provenance: { model: 'm' }, trajectory: [{ model_response: { calls: [['return_result', { status: 'success', value: { a: 1 } }]],
    raw_calls: [{ function: { name: 'return_result', arguments: '{"status":"success","value":{"a": 1}}' } }] } }],
    outcome: { action_ledger: [{ name: 'return_result', arguments: { status: 'success', value: { a: 1 } } }] } };
  assert.equal(migrateRow(current), current, 'no rewrite, no mark, raw arguments not re-serialized');
});
