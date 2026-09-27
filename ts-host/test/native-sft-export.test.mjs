import assert from 'node:assert/strict';
import { test } from 'node:test';
import { renderSftTurn } from '../scripts/export-native-sft.mjs';

const render = async messages => messages.map(message => {
  const calls = (message.tool_calls ?? []).map(call => `${call.function.name}:${call.function.arguments}`).join('|');
  const reasoning = message.reasoning_content ? `<think>${message.reasoning_content}</think>` : '';
  return `<${message.role}>${reasoning}${message.content ?? ''}${calls}` +
    (message.role === 'assistant' ? '<END>' : '');
}).join('');

test('native SFT export renders approved reasoning and tool choices through the selected template', async () => {
  const row = { id: 'turn-1', program_id: 'p', source_groups: ['p'], family: 'teacher_program', skill: 'eval',
    teacher_trajectory_id: 't', teacher_trajectory_digest: 'd', training_admission: { approved: true },
    messages: [{ role: 'user', content: 'Compute.' }], tools: [], teacher_reasoning: 'Use exact arithmetic.',
    target: { role: 'assistant', content: '', tool_calls: [{ id: 'old', type: 'function',
      function: { name: 'eval', arguments: '{"code":"2+2"}' } }] } };
  const rendered = await renderSftTurn(row, render, '<END>');
  assert.equal(rendered.prompt, '<user>Compute.');
  assert.match(rendered.completion, /<think>Use exact arithmetic.<\/think>eval:/);
  assert.ok(rendered.completion.endsWith('<END>'));
});

test('native SFT export skips decisions denied training admission', async () => {
  assert.equal(await renderSftTurn({ training_admission: { approved: false } }, render, '<END>'), null);
});

test('reasoning in the history is trimmed as the target is, so a later turn extends an earlier one', async () => {
  const base = { program_id: 'p', source_groups: ['p'], family: 'teacher_program', skill: 'eval',
    teacher_trajectory_id: 't', teacher_trajectory_digest: 'd', training_admission: { approved: true }, tools: [] };
  const call = { id: 'c', type: 'function', function: { name: 'eval', arguments: '{"code":"1"}' } };
  const first = await renderSftTurn({ ...base, id: 'turn-1', messages: [{ role: 'user', content: 'Go.' }],
    teacher_reasoning: 'Start.\n', target: { role: 'assistant', content: '', tool_calls: [call] } }, render, '<END>');
  const second = await renderSftTurn({ ...base, id: 'turn-2', messages: [{ role: 'user', content: 'Go.' },
    { role: 'assistant', content: '', reasoning_content: 'Start.\n', tool_calls: [{ ...call, id: 'teacher_0' }] },
    { role: 'tool', tool_call_id: 'teacher_0', content: '1' }], teacher_reasoning: 'Done.',
    target: { role: 'assistant', content: 'ok' } }, render, '<END>');
  assert.ok(second.prompt.startsWith(first.prompt + first.completion));
});
