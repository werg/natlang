import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeToolAgent, compactMessages, elidedCode, elidedOutput } from '../dist/native/agent.js';
import { session as open } from './support/natlang.mjs';

const elided = message => typeof message.content === 'string' && message.content.startsWith('[Output elided');

test('compaction elides the oldest tool outputs first and keeps the opening and the latest exchange', () => {
  const big = 'x'.repeat(500);
  const messages = [{ role: 'system', content: 'system' }, { role: 'user', content: 'task' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', tool_call_id: 'a', content: big + '1' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', tool_call_id: 'b', content: 'short' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', tool_call_id: 'c', content: big + '2' },
    { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', tool_call_id: 'd', content: big + '3' }];
  const size = () => JSON.stringify(messages).length;
  const target = size() - 300;
  const entries = { a: 0, b: 1, c: 2, d: 3 };
  assert.equal(compactMessages(messages, 2, 2, () => size() > target, id => entries[id]), 1);
  assert.equal(messages[3].content, elidedOutput(0), 'the oldest large output goes first, and its stub names its transcript entry');
  assert.match(messages[3].content, /transcript\[0\]\.output/);
  assert.equal(messages[7].content, big + '2', 'later outputs stay while the target is met');
  assert.equal(compactMessages(messages, 2, 2, () => true, id => entries[id]), 1, 'short outputs and the latest exchange are never elided');
  assert.equal(messages[5].content, 'short'); assert.equal(messages[9].content, big + '3');
  assert.equal(compactMessages(messages, 2, 2, () => true, id => entries[id]), 0, 'a stub is never elided again');
  assert.deepEqual(messages.map(message => message.role), ['system', 'user', 'assistant', 'tool', 'assistant', 'tool',
    'assistant', 'tool', 'assistant', 'tool'], 'messages keep their order and number');
  const code = 'const total = items.reduce((sum, item) => sum + item.value, 0);\n'.repeat(5);
  const calls = [{ role: 'system', content: 's' }, { role: 'user', content: 't' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'e', type: 'function', function: { name: 'eval', arguments: JSON.stringify({ code }) } }] },
    { role: 'tool', tool_call_id: 'e', content: elidedOutput(4) }, { role: 'assistant', content: '', tool_calls: [] }, { role: 'tool', content: 'ok' }];
  assert.equal(compactMessages(calls, 2, 2, () => true, id => id === 'e' ? 4 : undefined), 1, 'with outputs already elided, old eval code goes next');
  assert.equal(JSON.parse(calls[2].tool_calls[0].function.arguments).code, elidedCode(4));
});

test('a long call compacts old outputs instead of rolling over, and the model can read them back from transcript', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  const requests = [];
  let turn = 0, recovered;
  // Each eval prints about 2,000 characters; the server reports 1 token per 4 characters of request.
  const driver = request => {
    requests.push(structuredClone(request.messages));
    const prompt = Math.round((JSON.stringify(request.messages).length + JSON.stringify(request.tools).length) / 4);
    turn++;
    if (turn < 12) return { calls: [['eval', { code: `console.log('y'.repeat(2000) + ' mark${turn}'); ${turn}` }]], prompt_tokens: prompt };
    if (turn === 12) return { calls: [['eval', { code: 'transcript.filter(entry => / mark1\\b/.test(entry.output)).map(entry => entry.turn)' }]], prompt_tokens: prompt };
    recovered = requests.at(-1).at(-1).content;
    return { calls: [['return_result', { status: 'success', value: turn }]], prompt_tokens: prompt };
  };
  await new NativeToolAgent(driver, { contextTokens: 4096, maxTurns: 20 }).run(session);
  assert.equal(session.lam.return, 13);
  const last = requests.at(-1);
  const stub = last.find(elided);
  assert.ok(stub, 'old outputs were elided');
  const index = Number(stub.content.match(/transcript\[(\d+)\]/)[1]);
  assert.match(session.transcript[index].output, /y{2000} mark/, 'the stub names the entry that holds the output');
  assert.match(recovered, /\[\s*1\s*\]/, 'an eval found the elided first output in transcript');
  assert.ok(last.length > 20, 'the conversation was never cut: every earlier message is still there');
  assert.equal(last.filter(message => message.role === 'user').length, 1, 'no checkpoint request was sent');
  assert.ok(Math.round(JSON.stringify(last).length / 4) < 4096, 'the prompt stays within the budget');
  const previous = requests.at(-2);
  assert.deepEqual(last.slice(0, previous.length - 6), previous.slice(0, previous.length - 6),
    'once elided, an output stays elided, so consecutive requests share their prefix');
});

test('without a budget nothing is compacted, and a parameter named transcript keeps its meaning', async () => {
  const { session } = open({ type: '(transcript: string) => string', instructions: 'Echo.', args: { transcript: 'mine' } });
  let turn = 0, last;
  const driver = request => { last = request.messages; turn++;
    return turn < 8 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${turn}` }]], prompt_tokens: 100000 } :
      { calls: [['eval', { code: 'return transcript' }]] }; };
  await new NativeToolAgent(driver, { contextTokens: null, maxTurns: 20 }).run(session);
  assert.equal(last.some(elided), false);
  assert.equal(session.lam.return, 'mine');
});
