import assert from 'node:assert/strict';
import { test } from 'node:test';
import { migrateRow } from '../scripts/inline-curriculum/migrate-history-reasoning.mjs';
import { OPENING_THOUGHT } from '../dist/native/agent.js';

const call = id => ({ id, type: 'function', function: { name: 'eval', arguments: '{}' } });
const opening = [{ role: 'system', content: 's' }, { role: 'user', content: 'u' },
  { role: 'assistant', content: '', tool_calls: [call('scope_0')] }, { role: 'tool', tool_call_id: 'scope_0', content: 'x' }];
const row = { provenance: {}, trajectory: [
  { context: opening, model_response: { raw_calls: [call('a1')] }, assistant: { reasoning: 'First step.' } },
  { context: [...opening, { role: 'assistant', content: '', tool_calls: [call('a1')] }, { role: 'tool', tool_call_id: 'a1', content: 'y' }],
    model_response: { raw_calls: [], text: 'Done.' }, assistant: { reasoning: 'Say it.' } },
  { context: [...opening, { role: 'assistant', content: '', tool_calls: [call('a1')] }, { role: 'tool', tool_call_id: 'a1', content: 'y' },
    { role: 'assistant', content: 'Done.' }, { role: 'user', content: 'feedback' }], model_response: { raw_calls: [] }, assistant: {} },
] };

test('history turns get the reasoning of the decision that made them, and the opening its thought', () => {
  const migrated = migrateRow(structuredClone(row));
  const assistants = migrated.trajectory[2].context.filter(m => m.role === 'assistant');
  assert.deepEqual(assistants.map(m => m.reasoning_content), [OPENING_THOUGHT, 'First step.', 'Say it.']);
  assert.equal(migrated.provenance.history_reasoning_migration, 'reasoning/1');
  assert.equal(migrateRow(migrated), migrated);
});
