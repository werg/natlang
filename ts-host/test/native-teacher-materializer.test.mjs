import assert from 'node:assert/strict';
import { test } from 'node:test';
import { materializeNativeRows } from '../dist/teacher/native-materializer.js';

const system = { role: 'system', content: 'Use the native scope tools.' };
const opening = { role: 'user', content: '1 [ ] Compute the result. Current inputs: {"n": 3}' };
const firstCall = { tool: 'write', source_tool: 'write', arguments: { path: 'return', type: 'number', value: 6 }, call_id: null };
const secondCall = { tool: 'mark_done', source_tool: 'mark_done', arguments: { start: 1, end: 1 }, call_id: null };
const schema = [{ type: 'function', function: { name: 'write', description: 'Write a value.',
  parameters: { type: 'object', properties: { path: { type: 'string' } } } } }];
const nativeRow = (id, accepted = true) => ({ version: 'natlang.teacher_trajectory.native/1', id,
  task: { kind: 'whole_program', program_ir: { id: 'program-1', version: 'natlang.program/2' }, source_program_ids: ['program-1'] },
  provenance: { model: 'fixture-teacher', tool_schema: 'scope-eval-v1', context_tokens: 16384 },
  outcome: { status: 'done', accepted, value: 6, action_ledger: [
    { seq: 12, name: 'write', arguments: firstCall.arguments, outcome: 'ok', result_text: 'stored value' },
    { seq: 17, name: 'mark_done', arguments: secondCall.arguments, outcome: 'ok', result_text: 'marked line done' },
  ] },
  trajectory: [
    { phase: 'action', context: [system, opening], tools_offered: schema,
      assistant: { content: '', reasoning: 'The result is twice n.', calls: [firstCall] }, raw_response_sha256: 'raw-1' },
    { phase: 'action', context: [system, opening,
      { role: 'assistant', content: '', tool_calls: [{ id: 'call-1', type: 'function',
        function: { name: 'write', arguments: JSON.stringify(firstCall.arguments) } }] },
      { role: 'tool', tool_call_id: 'call-1', content: 'stored value' }],
      tools_offered: schema, assistant: { content: '', reasoning: 'The computation is complete.', calls: [secondCall] },
      raw_response_sha256: 'raw-2' },
  ], capture_limits: [] });

test('accepted native rows become linked template neutral decisions with their exact contexts', () => {
  const accepted = nativeRow('teacher-1'), rejected = nativeRow('teacher-rejected', false);
  const result = materializeNativeRows([accepted, rejected]);
  assert.equal(result.acceptedRows, 1);
  assert.equal(result.rejectedRows, 1);
  assert.equal(result.turns.length, 2);
  const [actionTurn, nextTurn] = result.turns;
  const action = actionTurn.decision, next = nextTurn.decision;
  assert.equal(action.assistant.reasoning, 'The result is twice n.');
  assert.deepEqual(action.tool_schemas, [{ name: 'write', description: 'Write a value.',
    parameters: { type: 'object', properties: { path: { type: 'string' } } } }]);
  assert.deepEqual(action.assistant.calls[0].outcome, { event_index: 0, trace_seq: 12,
    name: 'write', arguments: firstCall.arguments, status: 'ok', result: 'stored value', diagnostics: [] });
  assert.equal(next.assistant.calls[0].outcome.trace_seq, 17);
  // The conversation continues: the next decision sees the earlier call and its result.
  assert.deepEqual(next.context.map(message => message.role), ['system', 'user', 'assistant', 'tool']);
  assert.deepEqual(next.durable_opening.map(message => message.role), ['system', 'user']);
  assert.equal(actionTurn.source_ref.trajectory_id, 'teacher-1');
  assert.deepEqual(actionTurn.provenance, accepted.provenance);
  assert.deepEqual(actionTurn.messages, [system, opening]);
  assert.deepEqual(actionTurn.tools, schema);
  assert.equal(actionTurn.skill, 'write');
  assert.equal(actionTurn.teacher_reasoning, 'The result is twice n.');
  assert.deepEqual(actionTurn.target, { role: 'assistant', content: '', tool_calls: [{
    id: 'teacher_0_0', type: 'function', function: {
      name: 'write', arguments: JSON.stringify(firstCall.arguments),
    },
  }] });
});

test('rows collected with conversation rollover are rejected', () => {
  const row = nativeRow('rolled-over');
  row.trajectory.splice(1, 0, { phase: 'checkpoint', context: [system, opening], tools_offered: [],
    assistant: { content: 'A working note.', reasoning: null, calls: [] }, raw_response_sha256: 'raw-note' });
  const result = materializeNativeRows([row]);
  assert.equal(result.acceptedRows, 0); assert.equal(result.rejectedRows, 1); assert.equal(result.turns.length, 0);
});

test('accepted rows with an unlinked or reordered action outcome are rejected', () => {
  const row = nativeRow('bad-link');
  row.outcome.action_ledger[0].arguments = { path: 'return', value: 99 };
  assert.throws(() => materializeNativeRows([row]), /action outcomes have no teacher decision link/);
});

test('failed and unexecuted proposals remain in IR but are excluded from SFT admission', () => {
  const row = nativeRow('negative-decisions');
  row.outcome.action_ledger[0].outcome = 'rejected';
  row.trajectory[0].assistant.calls.push({ tool: 'read_page', source_tool: 'read_page',
    arguments: { id: 'amber', page: 2 }, call_id: null });
  const result = materializeNativeRows([row]);
  assert.equal(result.turns[0].training_admission.approved, false);
  assert.equal(result.turns[0].decision.assistant.calls[0].outcome.status, 'rejected');
  assert.equal(result.turns[0].decision.assistant.calls[1].outcome.status, 'not_executed');
  assert.equal(result.turns[1].training_admission.approved, true);
});

test('coerced proposals are retained but denied positive SFT admission', () => {
  const row = nativeRow('coerced-decision');
  row.outcome.action_ledger[0].diagnostics = ['coerced-redundant-self-alias'];
  const result = materializeNativeRows([row]);
  assert.equal(result.turns[0].training_admission.approved, false);
});

test('unsupported row versions and missing admission decisions fail closed', () => {
  const row = nativeRow('bad-version');
  row.version = 'natlang.teacher_trajectory/1';
  assert.throws(() => materializeNativeRows([row]), /unsupported native teacher row version/);
  const missing = nativeRow('no-accepted-flag');
  delete missing.outcome.accepted;
  assert.throws(() => materializeNativeRows([missing]), /outcome.accepted must be boolean/);
});

test('oracle-accepted student decisions keep student provenance and can be rehearsed', () => {
  const row = nativeRow('student-success');
  row.provenance.collection_role = 'student';
  const result = materializeNativeRows([row]);
  assert.equal(result.turns[0].source, 'student-native');
  assert.deepEqual(result.turns[0].gold_sources, ['checked-student-trajectory', 'exact-runtime-oracle']);
  assert.equal(result.turns[0].training_admission.approved, true);
});
