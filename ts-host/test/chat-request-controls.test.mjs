import test from 'node:test';
import assert from 'node:assert/strict';
import { chatRequestControls } from '../dist/teacher/chat-request-controls.js';

test('chat request controls carry validated adapter references for S7 rollouts', () => {
  const controls = chatRequestControls({ max_tokens: 256, x_natlang_adapters: [{ id: 'nz1_abc', scale: 1 }] });
  assert.deepEqual(controls.x_natlang_adapters, [{ id: 'nz1_abc', scale: 1 }]);
  assert.throws(() => chatRequestControls({ x_natlang_adapters: [{ id: 'nz1_abc' }] }), /adapter references/);
  assert.throws(() => chatRequestControls({ x_natlang_adapters: 'nz1_abc' }), /adapter references/);
  assert.throws(() => chatRequestControls({ x_natlang_adapters: [{ id: 'a', scale: 1, extra: 2 }] }), /adapter references/);
  assert.throws(() => chatRequestControls({ temperature: 1 }), /unsupported chat request control/);
});


test('RL collection explicitly captures behavior and disables guided sampling', () => {
  assert.deepEqual(chatRequestControls({x_natlang_rl_capture:true, guidance:false}),
    {x_natlang_rl_capture:true, guidance:false});
  assert.throws(() => chatRequestControls({x_natlang_rl_capture:'yes'}), /must be boolean/);
  assert.throws(() => chatRequestControls({guidance:true}), /only disable/);
});

test('JOLT+ teacher attempts carry the world privileged context alongside RL capture', () => {
  const context = { schema: 'natlang.privileged-request/1', custody: { split: 'train' },
    facts: [{ id: 'expected_result:0', kind: 'expected_result', text: 'The verified result is 2.' }] };
  assert.deepEqual(chatRequestControls({ x_natlang_rl_capture: true, x_natlang_privileged_context: context })
    .x_natlang_privileged_context, context);
  assert.throws(() => chatRequestControls({ x_natlang_privileged_context: context }), /privileged_context/);
  assert.throws(() => chatRequestControls({ x_natlang_rl_capture: true,
    x_natlang_privileged_context: { ...context, facts: [] } }), /privileged_context/);
  assert.throws(() => chatRequestControls({ x_natlang_rl_capture: true,
    x_natlang_privileged_block: { insert_at: 1, token_ids: [5] } }), /unsupported chat request control/);
});
