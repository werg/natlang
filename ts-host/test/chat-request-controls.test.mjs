import test from 'node:test';
import assert from 'node:assert/strict';
import { chatRequestControls, wireControls } from '../dist/teacher/chat-request-controls.js';

test('chat request controls carry validated adapter references for S7 rollouts', () => {
  const controls = chatRequestControls({ max_tokens: 256, x_natlang_adapters: [{ id: 'nz1_abc', scale: 1 }] });
  assert.deepEqual(controls.x_natlang_adapters, [{ id: 'nz1_abc', scale: 1 }]);
  assert.throws(() => chatRequestControls({ x_natlang_adapters: [{ id: 'nz1_abc' }] }), /adapter references/);
  assert.throws(() => chatRequestControls({ x_natlang_adapters: 'nz1_abc' }), /adapter references/);
  assert.throws(() => chatRequestControls({ x_natlang_adapters: [{ id: 'a', scale: 1, extra: 2 }] }), /adapter references/);
  assert.throws(() => chatRequestControls({ temperature: 1 }), /unsupported chat request control/);
});


test('a hosted teacher may omit the seed field', () => {
  assert.deepEqual(chatRequestControls({ x_natlang_omit_seed: true }), { x_natlang_omit_seed: true });
  assert.throws(() => chatRequestControls({ x_natlang_omit_seed: false }), /only be true/);
});

test('host call fields for a hosted teacher are an object', () => {
  const fields = { extra_content: { google: { thought_signature: 'x' } } };
  assert.deepEqual(chatRequestControls({ x_natlang_host_call_fields: fields }), { x_natlang_host_call_fields: fields });
  assert.throws(() => chatRequestControls({ x_natlang_host_call_fields: 'x' }), /object of tool-call fields/);
});

test('RL collection explicitly captures behavior and disables guided sampling', () => {
  assert.deepEqual(chatRequestControls({x_natlang_rl_capture:true, guidance:false}),
    {x_natlang_rl_capture:true, guidance:false});
  assert.throws(() => chatRequestControls({x_natlang_rl_capture:'yes'}), /must be boolean/);
  assert.throws(() => chatRequestControls({guidance:true}), /only disable/);
});

test('training guidance is a validated run-level control that never goes on the wire', () => {
  const run = { schema: 'natlang.guidance-run/1', role: 'teacher', custody: { split: 'train' },
    facts: [{ id: 'expected_result:0', kind: 'expected_result', text: 'The verified result is 2.', invocation: null, match: [] }] };
  const controls = chatRequestControls({ x_natlang_rl_capture: true, max_tokens: 64, x_natlang_guidance: run });
  assert.deepEqual(controls.x_natlang_guidance, run);
  assert.deepEqual(wireControls(controls), { x_natlang_rl_capture: true, max_tokens: 64 });
  assert.deepEqual(chatRequestControls({ x_natlang_rl_capture: true,
    x_natlang_guidance: { ...run, role: 'student', selection: 'shadow', budgetTokens: 256 } }).x_natlang_guidance.selection, 'shadow');
  assert.throws(() => chatRequestControls({ x_natlang_guidance: run }), /x_natlang_rl_capture: true/);
  for (const bad of [{ ...run, facts: [] }, { ...run, role: 'judge' }, { ...run, schema: 'natlang.privileged-request/1' },
    { ...run, selection: 'natlang' }, { ...run, writer: 'text' }, { ...run, budgetTokens: 0 }, { ...run, custody: null }, { ...run, extra: 1 },
    { ...run, facts: [{ ...run.facts[0], kind: 'hint' }] }, { ...run, facts: [run.facts[0], run.facts[0]] },
    { ...run, facts: [{ ...run.facts[0], match: 'x' }] }])
    assert.throws(() => chatRequestControls({ x_natlang_rl_capture: true, x_natlang_guidance: bad }), /guidance-run-invalid/);
  // The former transport-completed request is gone: guidance is written in-run.
  assert.throws(() => chatRequestControls({ x_natlang_rl_capture: true,
    x_natlang_privileged_context: { schema: 'natlang.privileged-request/1' } }), /unsupported chat request control/);
});
