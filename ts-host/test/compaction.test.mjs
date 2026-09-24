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

const toolNames = request => request.tools.map(tool => tool.function.name);
const promptOf = request => Math.round((JSON.stringify(request.messages).length + JSON.stringify(request.tools).length) / 4);

test('near the budget the model is asked to compact: only compact_history is offered, and its note is pinned', async () => {
  const { session } = open({ type: '() => number', instructions: 'Count up.' });
  const requests = [];
  let turn = 0, notes = 0;
  const driver = request => {
    requests.push(structuredClone(request));
    turn++;
    if (toolNames(request).length === 1 && toolNames(request)[0] === 'compact_history')
      return { calls: [['compact_history', { note: `Counting; reached ${turn}; stop at 16. Note ${++notes}.` }]], prompt_tokens: promptOf(request) };
    return turn < 16 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${turn}` }]], prompt_tokens: promptOf(request) } :
      { calls: [['return_result', { status: 'success', value: turn }]], prompt_tokens: promptOf(request) };
  };
  await new NativeToolAgent(driver, { contextTokens: 4096, maxTurns: 30 }).run(session);
  assert.equal(session.completed, true);
  const forced = requests.filter(request => toolNames(request).join() === 'compact_history');
  assert.ok(forced.length >= 2, 'the model was asked to compact, more than once in a long call');
  assert.ok(forced.every(request => request.tool_choice === 'required'), 'the compaction turn requires the tool call');
  assert.ok(forced.every(request => /near its context limit/.test(request.messages.at(-1).content)), 'the compaction turn says why');

  assert.ok(requests.filter(request => toolNames(request).length > 1).every(request => request.tool_choice === undefined));
  const kinds = requests.map(request => toolNames(request).join() === 'compact_history');
  assert.equal(kinds.some((forcedTurn, index) => forcedTurn && kinds[index + 1]), false, 'compaction never repeats back to back');
  const last = requests.at(-1);
  const pinned = last.messages.filter(message => message.role === 'user' && /^Your note from compacting/.test(message.content));
  assert.equal(pinned.length, 1, 'only the latest note is kept');
  assert.match(pinned[0].content, new RegExp(`Note ${notes}\\.`));
  assert.match(pinned[0].content, /Continue from where this note leaves off\. The full history of this call is in transcript/);
  assert.ok(last.messages.some(elided), 'older outputs moved to transcript');
  assert.ok(requests.every(request => promptOf(request) < 4096), 'no request exceeded the budget');
  assert.ok(session.transcript.some(entry => entry.tool === 'compact_history'), 'the compaction is part of the transcript');
});

test('the model may compact on its own, a note over the limit is rejected, and a text reply to the compaction turn is not a result', async () => {
  const { session } = open({ type: '() => string', instructions: 'Say hello.' });
  let turn = 0, sawRejection = false;
  const driver = request => {
    turn++;
    if (turn === 1) return { calls: [['eval', { code: "console.log('z'.repeat(3000)); 1" }]] };
    if (turn === 2) return { calls: [['compact_history', { note: 'x'.repeat(601) }]] };
    if (turn === 3) { sawRejection = /1 to 600 characters/.test(request.messages.at(-1).content);
      return { calls: [['compact_history', { note: 'Printed a long line; next say hello.' }]] }; }
    return { calls: [['return_result', { status: 'success', value: 'hello' }]] };
  };
  await new NativeToolAgent(driver, { maxTurns: 10 }).run(session);
  assert.equal(sawRejection, true);
  assert.equal(session.lam.return, 'hello');
  const forcedText = open({ type: '() => string', instructions: 'Say hello.' });
  let calls = 0;
  const texting = request => {
    calls++;
    if (toolNames(request).join() === 'compact_history') return { text: 'I would rather not.', prompt_tokens: promptOf(request) };
    return calls < 10 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${calls}` }]], prompt_tokens: promptOf(request) } :
      { calls: [['return_result', { status: 'success', value: 'hello' }]], prompt_tokens: promptOf(request) };
  };
  await new NativeToolAgent(texting, { contextTokens: 4096, maxTurns: 30 }).run(forcedText.session);
  assert.equal(forcedText.lam.return, 'hello', 'the text reply to the compaction turn did not become the result');
});
