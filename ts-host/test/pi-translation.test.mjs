import test from 'node:test';
import assert from 'node:assert/strict';
import { toPiContext, piToolChoiceOptions, piDeltaTranslator, piAssistantContent, piAssistantToTurn } from '../dist/model/pi-translation.js';

// Completed assistant messages and stream events in the two Pi shapes natlang meets: upstream @earendil-works/pi-ai
// 1.4.0 (the Node CLI) and Vibestudio's @panticonic/pi-ai 1.2.0-vibestudio.3. The fork adds fields (cost.totalFromProvider,
// authType, responseModel) and one stream event, `prompt_progress`; the translation reads neither.
const usage = { input: 11, output: 5, cacheRead: 0, cacheWrite: 0, totalTokens: 16,
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };
const content = [{ type: 'thinking', thinking: 'consider' }, { type: 'text', text: 'ok ' }, { type: 'text', text: 'done' },
  { type: 'toolCall', id: 'c1', name: 'eval', arguments: { code: '1+1' }, thoughtSignature: 'sig' },
  { type: 'toolCall', id: 'c2', name: 'return_result', arguments: {} }];
const upstream = { role: 'assistant', content, api: 'openai-completions', provider: 'openai', model: 'm', usage, stopReason: 'toolUse', timestamp: 1 };
const fork = { ...upstream, responseModel: 'm-2026', usage: { ...usage, cost: { ...usage.cost, totalFromProvider: true } },
  diagnostics: [] };

for (const [shape, reply] of [['upstream pi-ai 1.4.0', upstream], ['vibestudio fork 1.2.0', fork]]) {
  test(`a completed ${shape} assistant message becomes a model turn`, () => {
    const turn = piAssistantToTurn(reply, 'p');
    assert.deepEqual(turn.calls, [['eval', { code: '1+1' }], ['return_result', {}]]);
    assert.deepEqual(turn.raw_calls, [
      { id: 'c1', type: 'function', function: { name: 'eval', arguments: '{"code":"1+1"}' }, thought_signature: 'sig' },
      { id: 'c2', type: 'function', function: { name: 'return_result', arguments: '{}' } }]);
    assert.equal(turn.text, 'ok done'); assert.equal(turn.reasoning, 'consider');
    assert.equal(turn.prompt_tokens, 11); assert.equal(turn.completion_tokens, 5);
    assert.equal(turn.truncated, undefined);
    assert.deepEqual(turn.raw_response, { stop_reason: 'toolUse', usage: reply.usage });
    assert.equal(piAssistantContent(reply).calls.length, 2);
  });
}

test('truncated and failed Pi replies', () => {
  assert.equal(piAssistantToTurn({ ...upstream, stopReason: 'length' }, 'p').truncated, true);
  assert.throws(() => piAssistantToTurn({ ...upstream, stopReason: 'error', errorMessage: 'quota' }, 'p'), /quota/);
  assert.throws(() => piAssistantToTurn({ ...upstream, stopReason: 'aborted' }, 'p'), /provider p failed/);
});

test('stream events become deltas in call order, and events natlang does not read are ignored', () => {
  const next = piDeltaTranslator();
  const partial = { content: [{ type: 'thinking', thinking: '' }, { type: 'toolCall', id: 'c1', name: 'eval', arguments: {} }] };
  const events = [
    { type: 'start', partial },
    { type: 'thinking_start', contentIndex: 0, partial },
    { type: 'thinking_delta', contentIndex: 0, delta: 'hm', partial },
    { type: 'thinking_start', contentIndex: 2, partial },
    { type: 'text_delta', contentIndex: 1, delta: '', partial },
    { type: 'text_delta', contentIndex: 1, delta: 'hi', partial },
    // the fork's extra event
    { type: 'prompt_progress', total: 10, processed: 5, cache: 0, partial },
    { type: 'toolcall_start', contentIndex: 1, partial },
    { type: 'toolcall_delta', contentIndex: 1, delta: '{"code"', partial },
    { type: 'toolcall_delta', contentIndex: 9, delta: 'stray', partial },
    { type: 'done', reason: 'toolUse', message: upstream },
  ];
  assert.deepEqual(events.map(next).filter(Boolean), [
    { type: 'reasoning', text: 'hm' }, { type: 'reasoning', text: '\n' }, { type: 'text', text: 'hi' },
    { type: 'tool_call', index: 0, id: 'c1', name: 'eval' }, { type: 'tool_call', index: 0, arguments: '{"code"' }]);
});

test('tool_choice required maps to each API\'s word for it', () => {
  const required = { tool_choice: 'required' };
  assert.deepEqual(piToolChoiceOptions('anthropic-messages', required), { toolChoice: 'any' });
  assert.deepEqual(piToolChoiceOptions('google-vertex', required), { toolChoice: 'any' });
  assert.deepEqual(piToolChoiceOptions('openai-responses', required), { toolChoice: 'required' });
  assert.deepEqual(piToolChoiceOptions('openai-codex-responses', required), { toolChoice: 'required' });
  assert.deepEqual(piToolChoiceOptions('some-other-api', required), {});
  assert.deepEqual(piToolChoiceOptions('openai-responses', { tool_choice: 'auto' }), {});
  assert.deepEqual(piToolChoiceOptions('openai-responses', {}), {});
});

test('a request becomes a Pi context, with host-generated calls as user notes on Google APIs', () => {
  const request = { seed: null, max_tokens: null,
    tools: [{ type: 'function', function: { name: 'eval', description: 'run', parameters: { type: 'object', properties: {} } } }],
    messages: [{ role: 'system', content: 'S' }, { role: 'user', content: 'U' },
      { role: 'assistant', content: 'a', reasoning_content: 'r',
        tool_calls: [{ id: 't1', function: { name: 'eval', arguments: '{"code":"x"}' }, thought_signature: 'ts' }] },
      { role: 'tool', tool_call_id: 't1', content: 'result' },
      { role: 'assistant', content: '', natlang_host_generated: true,
        tool_calls: [{ id: 't2', function: { name: 'read_code', arguments: '' } }] },
      { role: 'tool', tool_call_id: 't2', content: 'code' }] };
  const model = { api: 'openai-completions', provider: 'openai', id: 'm' };
  const context = toPiContext(request, model);
  assert.deepEqual(context.tools, [{ name: 'eval', description: 'run', parameters: { type: 'object', properties: {} } }]);
  assert.deepEqual(context.messages.map(message => message.role), ['system', 'user', 'assistant', 'toolResult', 'assistant', 'toolResult']);
  const assistant = context.messages[2];
  assert.deepEqual(assistant.content, [{ type: 'thinking', thinking: 'r' }, { type: 'text', text: 'a' },
    { type: 'toolCall', id: 't1', name: 'eval', arguments: { code: 'x' }, thoughtSignature: 'ts' }]);
  assert.equal(assistant.stopReason, 'toolUse'); assert.equal(assistant.api, 'openai-completions');
  assert.equal(context.messages[3].toolName, 'eval');
  const google = toPiContext(request, { ...model, api: 'google-generative-ai' });
  assert.deepEqual(google.messages.map(message => message.role), ['system', 'user', 'assistant', 'toolResult', 'user', 'user']);
  assert.match(google.messages[4].content, /Host-provided operation read_code arguments: \{\}/);
  assert.equal(google.messages[5].content, 'Host-provided result for read_code: code');
});
