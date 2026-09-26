import assert from 'node:assert/strict';
import { test } from 'node:test';
import { migrateLine } from '../scripts/inline-curriculum/migrate-code-tools.mjs';
import { READ_CODE_DESCRIPTION } from '../dist/native/agent.js';

const row = {
  id: 'r1', provenance: { model: 'm' },
  trajectory: [{ tools_offered: [{ type: 'function', function: { name: 'read_function',
    description: 'Read the source of an imported function by its listed name, or the declaration of an external service or an importable package ("pkg" lists its exports, "pkg.name" shows one).' } }],
  assistant: { reasoning: 'I should read_function helper first.', calls: [{ tool: 'read_function', arguments: { name: 'helper' } }] } }],
  outcome: { action_ledger: [{ name: 'read_function', arguments: { name: 'helper' } }, { name: 'diff_functions' }] },
};

test('the code tools are renamed throughout a row, with their current descriptions, and the row is marked', () => {
  const migrated = JSON.parse(migrateLine(JSON.stringify(row)));
  const turn = migrated.trajectory[0];
  assert.equal(turn.tools_offered[0].function.name, 'read_code');
  assert.equal(turn.tools_offered[0].function.description, READ_CODE_DESCRIPTION);
  assert.equal(turn.assistant.calls[0].tool, 'read_code');
  assert.equal(turn.assistant.reasoning, 'I should read_code helper first.');
  assert.deepEqual(migrated.outcome.action_ledger.map(e => e.name), ['read_code', 'diff_code']);
  assert.equal(migrated.provenance.code_tools_migration, 'read_code/1');
});

test('migration is idempotent, and a row without the tools is left byte for byte', () => {
  const once = migrateLine(JSON.stringify(row));
  assert.equal(migrateLine(once), once);
  const plain = JSON.stringify({ id: 'r2', provenance: {}, note: 'reading functions' });
  assert.equal(migrateLine(plain), plain);
});

test('a rewritten program IR carries its digest along, so the row still matches its migrated program', async () => {
  const { recordDigest } = await import('../dist/teacher/collector.js');
  const { migrateIrLine } = await import('../scripts/inline-curriculum/migrate-code-tools.mjs');
  const program = { id: 'p', reference: { root: [['read_function', { name: 'helper' }]] } };
  const line = JSON.stringify({ ...row, task: { program_ir: program }, provenance: { program_ir_sha256: recordDigest(program) } });
  const migrated = JSON.parse(migrateLine(line));
  assert.equal(migrated.provenance.program_ir_sha256, recordDigest(JSON.parse(migrateIrLine(JSON.stringify(program)))));
  const stale = JSON.stringify({ ...row, task: { program_ir: program }, provenance: { program_ir_sha256: 'other' } });
  assert.equal(JSON.parse(migrateLine(stale)).provenance.program_ir_sha256, 'other');
});
