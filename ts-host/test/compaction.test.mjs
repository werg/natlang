import assert from 'node:assert/strict';
import { test } from 'node:test';
import { ELIDED_CODE, ELIDED_OUTPUT, NativeToolAgent, compactMessages } from '../dist/native/agent.js';
import { session as open } from './support/natlang.mjs';

test('compaction elides the oldest tool outputs first and keeps the opening and the latest exchange', () => {
  const big = 'x'.repeat(500);
  const messages = [{ role: 'system', content: 'system' }, { role: 'user', content: 'task' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', content: big + '1' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', content: 'short' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', content: big + '2' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', content: big + '3' }];
  const size = () => JSON.stringify(messages).length;
  const target = size() - 300;
  assert.equal(compactMessages(messages, 2, 2, () => size() > target), 1);
  assert.equal(messages[3].content, ELIDED_OUTPUT, 'the oldest large output goes first');
  assert.equal(messages[7].content, big + '2', 'later outputs stay while the target is met');
  assert.equal(compactMessages(messages, 2, 2, () => true), 1, 'short outputs and the latest exchange are never elided');
  const code = 'const total = items.reduce((sum, item) => sum + item.value, 0);\n'.repeat(5);
  const calls = [{ role: 'system', content: 's' }, { role: 'user', content: 't' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'a', type: 'function', function: { name: 'eval', arguments: JSON.stringify({ code }) } }] },
    { role: 'tool', content: ELIDED_OUTPUT }, { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', content: 'ok' }];
  assert.equal(compactMessages(calls, 2, 2, () => true), 1, 'with outputs already elided, old eval code goes next');
  assert.equal(JSON.parse(calls[2].tool_calls[0].function.arguments).code, ELIDED_CODE);
  assert.equal(messages[5].content, 'short'); assert.equal(messages[9].content, big + '3');
  assert.deepEqual(messages.map(message => message.role), ['system', 'user', 'assistant', 'tool', 'assistant', 'tool',
    'assistant', 'tool', 'assistant', 'tool'], 'messages keep their order and number');
});

test('a long call compacts old outputs deterministically instead of rolling over, and later requests share the prefix', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  const requests = [];
  let turn = 0;
  // Each eval prints about 2,000 characters; the server reports 1 token per 4 characters of request.
  const driver = request => {
    requests.push(structuredClone(request.messages));
    const prompt = Math.round((JSON.stringify(request.messages).length + JSON.stringify(request.tools).length) / 4);
    turn++;
    return turn < 12 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${turn}` }]], prompt_tokens: prompt } :
      { calls: [['return_result', { status: 'success', value: turn }]], prompt_tokens: prompt };
  };
  await new NativeToolAgent(driver, { contextTokens: 4096, maxTurns: 20 }).run(session);
  assert.equal(session.lam.return, 12);
  const last = requests.at(-1);
  assert.ok(last.some(message => message.content === ELIDED_OUTPUT), 'old outputs were elided');
  assert.ok(last.length > 20, 'the conversation was never cut: every earlier message is still there');
  assert.equal(last.filter(message => message.role === 'user').length, 1, 'no checkpoint request was sent');
  assert.ok(Math.round(JSON.stringify(last).length / 4) < 4096, 'the prompt stays within the budget');
  // Once elided, an output stays elided, so consecutive requests agree on everything they share.
  const previous = requests.at(-2);
  assert.deepEqual(last.slice(0, previous.length - 6), previous.slice(0, previous.length - 6));
});

test('without a budget nothing is compacted', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  let turn = 0, last;
  const driver = request => { last = request.messages; turn++;
    return turn < 8 ? { calls: [['eval', { code: `console.log('${'y'.repeat(2000)}'); ${turn}` }]], prompt_tokens: 100000 } :
      { calls: [['return_result', { status: 'success', value: turn }]] }; };
  await new NativeToolAgent(driver, { contextTokens: null, maxTurns: 20 }).run(session);
  assert.equal(last.some(message => message.content === ELIDED_OUTPUT), false);
});
