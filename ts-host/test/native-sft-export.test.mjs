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
