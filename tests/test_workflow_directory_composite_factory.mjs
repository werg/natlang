import test from 'node:test';
import assert from 'node:assert/strict';
import { buildQuestionBatchPayload, componentFileRecord, standaloneComponentProblem } from '../scripts/workflow_directory_composite_factory.mjs';

function primitive(id, answer) {
  return { id, source_ids: [`src:${id}`], semantics: { root: 'q.nl', files: { 'q.nl': '---\nargs: {}\nreturns: boolean\n---\nIs this true?\n' }, inputs: { state: { issue: 'x' } }, expected: answer } };
}

test('nested directory or state-less component is excluded before composition', () => {
  const nested = { id: 'directory', semantics: { root: 'index.nl', files: { 'index.nl': '---\nargs: {}\nreturns: boolean\nkind: directory-reducer\n---\nRead jobs/*.json\n' }, inputs: {}, folder_files: { 'jobs/a.json': '{}' }, expected_files: { 'jobs/a.json': '{}' }, expected: {} } };
  assert.equal(standaloneComponentProblem(nested), 'missing_explicit_state');
  assert.throws(() => componentFileRecord(nested), /missing_explicit_state/);
});

test('primitive question/state/gold are copied exactly to their visible job file', () => {
  const rows = [primitive('a', true), primitive('b', false)];
  const payload = buildQuestionBatchPayload(rows, ['a', 'b']);
  assert.deepEqual(JSON.parse(payload.folder_files['jobs/a.json']), { source_question_id: 'src:a', question: rows[0].semantics.files['q.nl'], state: rows[0].semantics.inputs.state });
  assert.deepEqual(JSON.parse(payload.folder_files['jobs/b.json']), { source_question_id: 'src:b', question: rows[1].semantics.files['q.nl'], state: rows[1].semantics.inputs.state });
  assert.deepEqual(payload.expected, { a: true, b: false });
  assert.deepEqual(payload.folder_files, payload.expected_files);
});

test('directory-reducer marker is rejected even if a malformed input state exists', () => {
  const malformed = { ...primitive('nested', true), semantics: { ...primitive('nested', true).semantics, files: { 'q.nl': '---\nkind: directory-reducer\n---\nRead files\n' } } };
  assert.equal(standaloneComponentProblem(malformed), 'directory_reducer_component');
});
