import assert from 'node:assert/strict';
import { test } from 'node:test';
import { migrateRow } from '../scripts/inline-curriculum/migrate-brief-prompt.mjs';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';

test('a context with the prompt from before "Be brief" gets the current prompt, once', () => {
  const before = TOOLS_PROMPT.slice(0, TOOLS_PROMPT.lastIndexOf('\n\nBe brief.')) + '\n';
  const row = { provenance: {}, trajectory: [{ context: [{ role: 'system', content: before }, { role: 'user', content: 'x' }] }] };
  const once = migrateRow(row);
  assert.equal(once.trajectory[0].context[0].content, TOOLS_PROMPT);
  assert.equal(once.trajectory[0].context[1].content, 'x');
  assert.equal(once.provenance.brief_prompt_migration, 'be-brief/1');
  assert.equal(migrateRow(once), once);
  const other = { provenance: {}, trajectory: [{ context: [{ role: 'system', content: 'another prompt' }] }] };
  assert.equal(migrateRow(other), other, 'a prompt that is not the earlier one is left alone');
});
