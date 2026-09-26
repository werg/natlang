import assert from 'node:assert/strict';
import { test } from 'node:test';
import { migrateLine } from '../scripts/inline-curriculum/migrate-return-description.mjs';
import { RETURN_RESULT_DESCRIPTION, RETURN_RESULT_DESCRIPTION_BEFORE } from '../dist/native/agent.js';

test('the earlier return_result description becomes the current one, once', () => {
  const line = JSON.stringify({ provenance: {}, tools: [{ function: { name: 'return_result', description: RETURN_RESULT_DESCRIPTION_BEFORE } }] });
  const once = migrateLine(line);
  assert.equal(JSON.parse(once).tools[0].function.description, RETURN_RESULT_DESCRIPTION);
  assert.equal(JSON.parse(once).provenance.return_description_migration, 'negative-answers/1');
  assert.equal(migrateLine(once), once);
});
