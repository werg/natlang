import test from 'node:test';
import assert from 'node:assert/strict';
import { trainingView } from '../scripts/code-corpus/views.mjs';
import { normalizeCase2Code, normalizeTinyCodes } from '../scripts/code-corpus/datasets.mjs';
test('Python remains a translation request, never a fake TS completion', () => {
  const row=trainingView(normalizeCase2Code({prompt:'Double x',code:'def double(x): return 2*x'}));
  assert.equal(row.kind,'translation_request'); assert.equal(row.completion,null);
  assert.match(row.prompt,/def double/);
});
test('code view retains provenance and nonexecution status', () => {
  const task=normalizeTinyCodes({instruction:'Double x',code:'function double(x) { return 2*x; }',language:'JavaScript'},{license:'test',revision:'abc'});
  const view=trainingView(task); assert.equal(view.kind,'code_sft');
  assert.equal(view.source.revision,'abc'); assert.equal(view.verification.status,'unverified');
  assert.equal(view.group_id,task.group_id);
});
