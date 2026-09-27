import assert from 'node:assert/strict';
import { test } from 'node:test';
import { currentPrompt, migrateRow } from '../scripts/inline-curriculum/migrate-system-prompt.mjs';
import { DIRECTORY_REDUCER_PROMPT, FUNCTION_TOOLS_PROMPT, TOOLS_PROMPT } from '../dist/native/prompt.js';

const history = { TOOLS_PROMPT: ['old base long', 'old base'], APPROACH_PROMPT: ['\nguide'],
  FUNCTION_TOOLS_PROMPT: ['\nold functions'], DIRECTORY_REDUCER_PROMPT: ['\nold folder'] };

test('an earlier prompt made of known parts becomes the current one, keeping its parts but not the approach guide', () => {
  assert.equal(currentPrompt('old base', history), TOOLS_PROMPT);
  assert.equal(currentPrompt('old base long\nguide\nold functions', history), TOOLS_PROMPT + FUNCTION_TOOLS_PROMPT);
  assert.equal(currentPrompt('old base\nold functions\nold folder', history), TOOLS_PROMPT + FUNCTION_TOOLS_PROMPT + DIRECTORY_REDUCER_PROMPT);
  assert.equal(currentPrompt('old base and something else', history), undefined);
});

test('rows are rewritten once, and a row with an unknown prompt is left out', () => {
  const row = { provenance: {}, trajectory: [{ context: [{ role: 'system', content: 'old base' }] }] };
  const once = migrateRow(row, history);
  assert.equal(once.trajectory[0].context[0].content, TOOLS_PROMPT);
  assert.equal(once.provenance.system_prompt_migration, 'current-prompt/1');
  assert.equal(migrateRow(once, { ...history, TOOLS_PROMPT: [TOOLS_PROMPT, ...history.TOOLS_PROMPT] }), once);
  const custom = { provenance: {}, trajectory: [...row.trajectory, { context: [{ role: 'system', content: 'custom' }] }] };
  assert.equal(migrateRow(custom, history), null);
});
