import assert from 'node:assert/strict';
import { test } from 'node:test';
import { migrateRow } from '../scripts/inline-curriculum/migrate-opening-builtins.mjs';
import { BUILT_INS_LINE } from '../dist/native/agent.js';

test('each call opening gains the built-ins line once, and nothing else changes', () => {
  const opening = { role: 'user', content: 'You are inside this call: f(): number\n\nInstructions:\nDo it.' };
  const row = { provenance: {}, trajectory: [{ context: [{ role: 'system', content: 's' }, opening, { role: 'user', content: 'feedback' }] }] };
  const migrated = migrateRow(structuredClone(row));
  assert.equal(migrated.trajectory[0].context[1].content, `${opening.content}\n\n${BUILT_INS_LINE}`);
  assert.equal(migrated.trajectory[0].context[2].content, 'feedback');
  assert.equal(migrated.provenance.opening_builtins_migration, 'builtins-line/1');
  assert.equal(migrateRow(migrated), migrated);
});
