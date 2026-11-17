import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeToolAgent } from '../dist/native/agent.js';
import { chatCompletionModelTurn } from '../dist/model/chat-completion.js';
import { AUTOMATIC_NOTE } from '../dist/native/prompt.js';
import { interpreter, lambda } from './support/natlang.mjs';

// writing-failures 2026-11-13: LFM3 thinks 800-1,500 tokens before acting. A reply cut at the turn limit used to be closed,
// answered with "your reply was cut off" and started over, so six 512-token turns were six fresh attempts. A driver that
// continues replies now resumes the cut reply as the same turn.

const cut = (reasoning, text = '', tokens = 100, prompt = 1000) => ({ calls: [], text, reasoning, truncated: true,
  completion_tokens: tokens, prompt_tokens: prompt });
const answer = (reasoning, value = 1) => ({ calls: [['return_result', { status: 'success', value }]], reasoning, text: '',
  completion_tokens: 20, prompt_tokens: 1000 });

async function run(replies, options = {}, { continues = true, type = '() => number' } = {}) {
  const requests = [];
  const driver = Object.assign(request => { requests.push(structuredClone(request)); return replies(requests.length, request); },
    continues ? { continuesReplies: true } : {});
  const agent = new NativeToolAgent(driver, { contextTokens: 16384, ...options });
  const runtime = interpreter({ agent: session => agent.run(session) });
  const result = await runtime.run(lambda({ type, instructions: 'Return one.' }));
  return { result, requests, events: runtime.trace.events };
}

test('a reply cut off before any tool call is continued as the same turn, with the conversation unchanged', async () => {
  const { result, requests, events } = await run(index => index === 1 ? cut('Let me think. The answer') :
    index === 2 ? cut('Let me think. The answer is one, because') : answer('Let me think. The answer is one, because it says so.'),
  { maxTurns: 2, turnTokens: 100 });
  assert.equal(result.value, 1);
  assert.equal(requests.length, 3);
  assert.equal(requests[0].continue, undefined);
  assert.deepEqual(requests[1].continue, { text: '', reasoning: 'Let me think. The answer' });
  assert.deepEqual(requests[2].continue, { text: '', reasoning: 'Let me think. The answer is one, because' });
  // The continued requests send exactly the conversation the cut one did: no cut-off note, no new turn.
  assert.deepEqual(requests[1].messages, requests[0].messages);
  assert.deepEqual(requests[2].messages, requests[0].messages);
  // Continuations are not turns: two turns would have made the second request the last turn, offering only return_result.
  assert.equal(requests[2].tool_choice, undefined);
  assert.deepEqual(events.filter(event => event.kind === 'model_request' && event.phase === 'start').map(event => event.turn), [1, 1, 1]);
  assert.deepEqual(events.filter(event => event.kind === 'continuation').map(event => event.message_tokens), [100, 200]);
  const nodes = events.filter(event => event.kind === 'model_turn');
  assert.equal(nodes.length, 3);
  assert.equal(new Set(nodes.map(node => node.node)).size, 3, 'each request keeps its own graph identity');
  assert.deepEqual(nodes.map(node => node.seed), requests.map(request => request.seed));
});

test('the call token budget bounds continuations', async () => {
  const { result, requests } = await run(() => cut('still thinking'), { maxTurns: 6, maxTokens: 350, turnTokens: 100 });
  assert.equal(result.outcome.kind, 'quiesced');
  assert.match(result.outcome.detail, /budget exhausted/);
  assert.equal(requests.length, 4);
  assert.deepEqual(requests.map(request => request.max_tokens), [100, 100, 100, 50]);
});

test('a reply that no longer fits the context is closed with the cut-off note instead of continued', async () => {
  const { requests } = await run(index => index === 1 ? cut('thinking', '', 512, 3500) : answer('done thinking'),
    { maxTurns: 4, turnTokens: 512, contextTokens: 4096 });
  assert.equal(requests.length, 2);
  assert.equal(requests[1].continue, undefined);
  assert.match(requests[1].messages.at(-1).content, /^Your reply was cut off at the 512-token limit before any tool call\. Take the next step with one tool call\./);
});

test('a driver that cannot continue replies gets the cut-off note as before', async () => {
  const { requests } = await run(index => index === 1 ? cut('thinking') : answer('done'), { maxTurns: 4, turnTokens: 100 }, { continues: false });
  assert.equal(requests.length, 2);
  assert.equal(requests[1].continue, undefined);
  assert.match(requests[1].messages.at(-1).content, /Your reply was cut off at the 100-token limit before any tool call/);
});

test('the chat adapter sends the cut reply as the final assistant message and takes the whole message back', async () => {
  const sent = [], deltas = [];
  const whole = { choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'The answer.',
    reasoning_content: 'Let me think. The answer is one.' } }], usage: { prompt_tokens: 40, completion_tokens: 9 } };
  const driver = chatCompletionModelTurn(async body => { sent.push(structuredClone(body)); return whole; }, { continuesReplies: true });
  const request = { messages: [{ role: 'user', content: 'Answer.' }], tools: [], seed: null, max_tokens: 64,
    continue: { text: 'The', reasoning: 'Let me think. The answer' } };
  const turn = await driver(request, undefined, { onDelta: delta => deltas.push(delta) });
  assert.deepEqual(sent[0].messages.at(-1), { role: 'assistant', content: 'The', reasoning: 'Let me think. The answer',
    reasoning_content: 'Let me think. The answer' });
  assert.equal(sent[0].stream, false);
  assert.equal(turn.text, 'The answer.');
  assert.equal(turn.reasoning, 'Let me think. The answer is one.');
  assert.deepEqual(deltas, [{ type: 'reasoning', text: ' is one.' }, { type: 'text', text: ' answer.' }]);
  // A server that answers with only what it generated breaks the contract, and says so.
  const partial = chatCompletionModelTurn(async () => ({ choices: [{ finish_reason: 'stop', message: { role: 'assistant',
    content: ' answer.', reasoning_content: ' is one.' } }] }), { continuesReplies: true });
  await assert.rejects(partial(request), /continuation-mismatch/);
  const plain = chatCompletionModelTurn(async () => whole);
  await assert.rejects(plain(request), /continuation-unsupported/);
});

test('no context-limit note before anything was moved to transcript', async () => {
  // An opening larger than the estimate allows: the first turn has nothing to shorten, so it gets no note.
  const long = 'Return one. ' + 'Consider every detail of this request carefully. '.repeat(200);
  const requests = [];
  const agent = new NativeToolAgent(request => { requests.push(structuredClone(request)); return answer('ok'); },
    { contextTokens: 2048, maxTurns: 4 });
  const runtime = interpreter({ agent: session => agent.run(session) });
  const result = await runtime.run(lambda({ type: '() => number', instructions: long }));
  const events = runtime.trace.events;
  assert.equal(result.value, 1);
  assert.ok(!JSON.stringify(requests[0].messages).includes(AUTOMATIC_NOTE.slice(0, 40)));
  assert.equal(events.filter(event => event.kind === 'compaction').length, 0);
});

test('a reply without a tool call is not an answer: the model is told to use return_result', async () => {
  const { result, requests } = await run(index => index === 1 ? { text: 'Brief guidance.', completion_tokens: 3 } :
    { calls: [['return_result', { status: 'success', value: 'Brief guidance.' }]], completion_tokens: 3 }, { maxTurns: 4 },
  { type: '() => string' });
  assert.equal(result.value, 'Brief guidance.');
  assert.equal(requests.length, 2);
  assert.match(requests[1].messages.at(-1).content, /^There is no result yet\. Answer with return_result: status "success" and a string\.$/);
});

test('edit_code and diff_code are offered only to a call that can change code', async () => {
  const { requests } = await run(() => answer('ok'), { maxTurns: 2 });
  const names = requests[0].tools.map(tool => tool.function.name);
  assert.ok(names.includes('read_code'));
  assert.ok(!names.includes('edit_code') && !names.includes('diff_code'));
});

test('a truncated tool call preserves reasoning but cannot prefill lost structured arguments', async () => {
  const raw = [{ id: 'call_partial', type: 'function', function: { name: 'eval', arguments: '{"code":' } }];
  const driver = chatCompletionModelTurn(async () => ({ choices: [{ finish_reason: 'length', message: {
    role: 'assistant', content: '', reasoning_content: 'I should compute it.', tool_calls: raw,
  } }], usage: { prompt_tokens: 1000, completion_tokens: 100 } }), { continuesReplies: true });
  const partial = await driver({ messages: [], tools: [], seed: null, max_tokens: 100 });
  assert.equal(partial.truncated, true);
  assert.equal(partial.reasoning, 'I should compute it.');
  assert.deepEqual(partial.raw_calls, raw);
  const { requests, result } = await run(index => index === 1 ? partial : answer('Done.'), { maxTurns: 4, turnTokens: 100 });
  assert.equal(result.value, 1);
  assert.equal(requests[1].continue, undefined);
});
