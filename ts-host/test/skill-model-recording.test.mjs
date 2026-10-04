import test from 'node:test';
import assert from 'node:assert/strict';
import { recordingModelDriver } from '../scripts/skills/record-model-turn.mjs';
import { chatCompletionModelTurn } from '../dist/model/chat-completion.js';

const request = () => ({ messages: [{ role: 'user', content: 'Return seven' }],
  tools: [{ type: 'function', function: { name: 'eval', parameters: {
    type: 'object', properties: { code: { type: 'string' } }, required: ['code'] } } }],
  seed: 1, max_tokens: 100 });

test('record effective synthesized calls rather than inferring them from wire tools', async () => {
  const records = [], input = request();
  const drive = recordingModelDriver({ record: row => records.push(row),
    createDriver: onExchange => chatCompletionModelTurn(async () => ({ choices: [{
      finish_reason: 'stop', message: { content: '{"code":"return 7"}' } }] }), { onExchange }) });
  const turn = await drive(input);
  assert.deepEqual(records[0].turn.calls, [['eval', { code: 'return 7' }]]);
  assert.deepEqual(records[0].turn, turn);
  assert.equal(records[0].wireExchanges[0].wireResponse.choices[0].message.tool_calls, undefined);
  input.messages.push({ role: 'assistant', content: 'later' }); turn.calls[0][1].code = 'return 99';
  assert.equal(records[0].request.messages.length, 1);
  assert.equal(records[0].turn.calls[0][1].code, 'return 7');
});

test('retry attempts remain wire evidence, with one normalized aliased runtime turn', async () => {
  const records = [], wires = []; let attempts = 0;
  const drive = recordingModelDriver({ record: row => records.push(row), recordWire: row => wires.push(row),
    createDriver: onExchange => chatCompletionModelTurn(async () => ({
      usage: { prompt_tokens: 10, completion_tokens: 5 }, choices: [{ finish_reason: 'tool_calls',
        message: { content: '', tool_calls: [{ id: 'a', type: 'function', function: {
          name: 'execute', arguments: ++attempts === 1 ? '{' : '{"code":"return 7"}' } }] } }] }),
    { onExchange, toolAliases: { eval: 'execute' } }) });
  await drive(request());
  assert.equal(records.length, 1); assert.equal(wires.length, 2);
  assert.equal(records[0].wireExchanges.length, 2);
  assert.deepEqual(records[0].turn.calls, [['eval', { code: 'return 7' }]]);
  assert.equal(records[0].turn.prompt_tokens, 20);
  assert.equal(records[0].turn.completion_tokens, 10);
});

test('failed turns preserve wire diagnostics without inventing an effective turn', async () => {
  const records = [], wires = [];
  const drive = recordingModelDriver({ record: row => records.push(row), recordWire: row => wires.push(row),
    createDriver: onExchange => async req => {
      await onExchange({ request: req, wireResponse: { failure: 'malformed' } });
      throw Error('bad response');
    } });
  await assert.rejects(drive(request()), /bad response/);
  assert.equal(records.length, 0); assert.equal(wires.length, 1);
});
