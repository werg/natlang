import test from 'node:test';
import assert from 'node:assert/strict';
import { qwenTeacherForcedDriver } from '../dist/teacher/native-forced-replay.js';

const binding = { sourcePath: '/immutable/qwen/corrections.jsonl', sourceFileSha256: 'a'.repeat(64),
  sourceView: 'privileged-hint', actorSourceSha256: 'b'.repeat(64), executionIdentitySha256: 'c'.repeat(64),
  model: 'nvidia/Qwen3.6-35B-A3B-NVFP4', revision: '1355db6a052410cfd62085d94b58866fd0f2c3c5' };
function row() { return { version: 'natlang.teacher_trajectory.native/1', id: 'qwen-row', task: { program_ir: { id: 'exact' } },
  trajectory: [{ phase: 'action', invocation_id: 'source-run/call-1', context: [{ role: 'user', content: 'privileged hint here' }],
    tools_offered: [], assistant: { content: 'plan <|neuralese|>native body<|/neuralese|>', calls: [] } }] }; }

test('Qwen forced replay binds exact turn IDs and keeps the privileged context out of student messages', async () => {
  const received = [];
  const { driver, runId } = qwenTeacherForcedDriver(row(), binding, async request => { received.push(request); return { text: '' }; });
  assert.equal(runId, 'source-run');
  const request = { invocation_id: 'source-run/call-1', messages: [{ role: 'user', content: 'original task only' }], tools: [] };
  await driver(request);
  assert.deepEqual(received[0].messages, request.messages);
  assert.deepEqual(received[0].native_teacher_forced_turn.assistant.content,
    [{ type: 'text', text: 'plan ' }, { type: 'teacher_native_text', text: 'native body' }]);
  assert.equal(received[0].native_teacher_forced_turn.source_receipt.source_view, 'privileged-hint');
  assert.equal(received[0].native_teacher_forced_turn.source_receipt.invocation_id, request.invocation_id);
  await assert.rejects(() => driver({ ...request, invocation_id: 'other-run/call-1' }), /exact Qwen task\/run identity/);
  await assert.rejects(() => driver(request), /no exact Qwen source invocation\/turn match/);
});
