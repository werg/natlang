import test from 'node:test';
import assert from 'node:assert/strict';
import { applyReviewedWorkflowTypedReturns } from './workflow-typed-return-reviewed.mjs';

const record = (questions, expected) => ({
  version: 'natlang.program/2', id: 'base', kind: 'lambda_source', source: 'workflowevals:customer-service', split: 'train',
  source_ids: ['src'], source_groups: ['grp'], source_revisions: ['rev'], license: 'Apache-2.0',
  generation: { generator: 'adapter-v4' },
  curriculum: { family: 'workflow_customer-service', variant: 'directory-v1', slice: 'folder_failure', split_group: 'grp' },
  semantics: { root: 'review_jobs.nl', files: { 'review_jobs.nl': '---\nargs: {}\nreturns: "Record<string, unknown>"\nkind: directory-reducer\n---\nPreserve all files. This batch covers only the included questions.\n' },
    folder_files: Object.fromEntries(questions.map(([key, question]) => [`jobs/${key}.json`, JSON.stringify({ question, state: {} })])),
    expected, expected_files: {} },
  external_source: { source_id: 'source', snapshot_sha256: 'snapshot' },
});

test('typed return migration derives boolean, choice and score types from visible question schemas', () => {
  const base = record([
    ['bool', { type: 'noul', criteria: { true: 'yes', false: 'no' } }],
    ['choice', { type: 'choice', criteria: { alpha: 'a', beta: 'b' } }],
    ['score', { type: 'score', criteria: { low: 'low', high: 'high' } }],
  ], { bool: false, choice: 'beta', score: '1' });
  const original = structuredClone(base);
  const next = applyReviewedWorkflowTypedReturns(base);
  const src = next.semantics.files['review_jobs.nl'];
  const declared = JSON.parse(/^returns: (.+)$/m.exec(src)[1]);
  assert.match(declared, /"bool": boolean/);
  assert.match(declared, /"choice": "alpha" \| "beta"/);
  assert.match(declared, /"score": "0" \| "1"/);
  assert.deepEqual(next.semantics.expected, base.semantics.expected);
  assert.deepEqual(next.semantics.folder_files, base.semantics.folder_files);
  assert.deepEqual(next.source_ids, base.source_ids);
  assert.deepEqual(next.source_groups, base.source_groups);
  assert.match(next.id, /:workflow-typed-returns-v1$/);
  assert.match(src, /Do not stringify booleans/);
  assert.match(src, /one option, not a requirement/);
  assert.deepEqual(base, original, 'the base IR remains immutable');
  assert.strictEqual(applyReviewedWorkflowTypedReturns(next), next, 'the reviewed variant is idempotent');
});

test('unrecognized schemas and unexpected/gold-inconsistent fields fail closed', () => {
  const unsupported = record([['a', { type: 'other', criteria: { yes: 'y' } }]], { a: 'yes' });
  assert.throws(() => applyReviewedWorkflowTypedReturns(unsupported), /unsupported_question_type/);
  const extra = record([['a', { type: 'noul', criteria: { true: 'y', false: 'n' } }]], { a: true, b: false });
  assert.throws(() => applyReviewedWorkflowTypedReturns(extra), /expected_extra_keys/);
  const missing = record([['a', { type: 'noul', criteria: { true: 'y', false: 'n' } }]], { a: true });
  missing.semantics.files['review_jobs.nl'] = missing.semantics.files['review_jobs.nl'].replace('Preserve all files.', '');
  assert.throws(() => applyReviewedWorkflowTypedReturns(missing), /preserve_anchor_missing/);
});
