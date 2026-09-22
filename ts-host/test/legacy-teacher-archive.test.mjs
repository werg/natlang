import assert from 'node:assert/strict';
import { test } from 'node:test';
import { archiveLegacyRow } from '../scripts/archive-legacy-teacher.mjs';

test('legacy teacher archive retains choices and reasoning without rendered transcripts', () => {
  const row = { version: 'natlang.teacher_trajectory/1', id: 'old-1',
    task: { program_ir: { id: 'p1' }, source_program_ids: ['p1'] },
    provenance: { model: 'teacher', teacher_system_prompt: 'old rendered prompt' }, outcome: { accepted: true },
    trajectory: [{ function: 'main', context: [{ role: 'system', content: 'template' }],
      tools_offered: [{ type: 'function', function: { name: 'write' } }],
      assistant: { content: '', reasoning: 'Use the checked source.', calls: [{ tool: 'write', arguments: { value: 1 } }] },
      executions: [{ status: 'ok' }], raw_response_sha256: 'raw' }] };
  const archived = archiveLegacyRow(row, { file: 'old.jsonl', line: 1, file_sha256: 'f', row_sha256: 'abcdef1234567890' });
  assert.equal(archived.version, 'natlang.teacher_decision_archive/1');
  assert.equal(archived.decisions[0].assistant.reasoning, 'Use the checked source.');
  assert.deepEqual(archived.decisions[0].offered_tool_names, ['write']);
  assert.equal(archived.decisions[0].context, undefined);
  assert.equal(archived.provenance.teacher_system_prompt, undefined);
  assert.equal(archived.task.program_ir, undefined);
  assert.equal(archived.task.program_ir_id, 'p1');
  assert.equal(archived.regeneration.policy, 'native-program-ir');
  assert.equal(archived.training_admission.approved, false);
});
