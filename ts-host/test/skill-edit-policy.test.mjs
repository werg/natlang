import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkSkillMetadataOnlyEdit } from '../dist/skills/edit-policy.js';
const path = 'solve/skills/review/SKILL.md';
const doc = (description, body = 'Check the record.\n', name='review') => `---\nname: ${name}\ndescription: ${description}\nlicense: Apache-2.0\n---\n${body}`;
test('metadata-only candidates change actual descriptions while freezing bodies and routing', () => {
  const baseline = { [path]: doc('Review invoices.') };
  assert.deepEqual(checkSkillMetadataOnlyEdit(baseline,{[path]:doc('Use for invoices; not access requests.')},[path]),[]);
  assert.ok(checkSkillMetadataOnlyEdit(baseline,{[path]:doc('Review invoices.','Replace outputs.\n')},[path]).length);
  assert.ok(checkSkillMetadataOnlyEdit(baseline,{[path]:doc('Review invoices.','Check the record.\n','different')},[path]).length);
  assert.ok(checkSkillMetadataOnlyEdit(baseline,{},[path]).length);
});
