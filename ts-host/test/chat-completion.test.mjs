import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { assembleChatCompletion, chatCompletionModelTurn, httpChatTransport } from '../dist/model/chat-completion.js';

const request = (tools = [{ type: 'function', function: { name: 'eval', parameters: { type: 'object',
  properties: { code: { type: 'string', 'x-natlang': 'private' } } } } }]) =>
  ({ messages: [{ role: 'user', content: 'Write seven.' }], tools, seed: 3, max_tokens: null });

/** A local chat-completions server; `reply(body)` returns chunks to stream, or a whole body. */
async function server(reply) {
  const bodies = [];
  const instance = createServer((incoming, response) => {
    let text = '';
    incoming.setEncoding('utf8'); incoming.on('data', part => { text += part; });
    incoming.on('end', () => {
      const body = JSON.parse(text); bodies.push(body);
      const answer = reply(body, bodies.length);
      if (!Array.isArray(answer)) { response.writeHead(200, { 'content-type': 'application/json' }); response.end(JSON.stringify(answer)); return; }
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      // Chunks split across writes, as a real stream may deliver them.
      const wire = answer.map(chunk => `data: ${JSON.stringify(chunk)}\n\n`).join('') + 'data: [DONE]\n\n';
      response.write(wire.slice(0, 17)); setTimeout(() => response.end(wire.slice(17)), 5);
    });
  });
  await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
  return { endpoint: `http://127.0.0.1:${instance.address().port}`, bodies, close: () => new Promise(resolve => instance.close(resolve)) };
}

const delta = (value, finish = null, extra = {}) => ({ id: 'c1', model: 'm', created: 1, choices: [{ index: 0, delta: value, finish_reason: finish }], ...extra });

test('streamed deltas assemble into the response a non-streaming request returns', async () => {
  async function* chunks() {
    yield delta({ role: 'assistant', reasoning_content: 'Seven ' });
    yield delta({ reasoning_content: 'is easy.' });
    yield delta({ tool_calls: [{ index: 0, id: 'call_a', type: 'function', function: { name: 'eval', arguments: '{"co' } }] });
    yield delta({ tool_calls: [{ index: 0, function: { arguments: 'de":"return 7"}' } }] });
    yield delta({ tool_calls: [{ index: 1, id: 'call_b', type: 'function', function: { name: 'read_page', arguments: '{}' } }] });
    yield delta({}, 'tool_calls');
    yield { id: 'c1', choices: [], usage: { prompt_tokens: 12, completion_tokens: 5 }, timings: { cache_n: 4 } };
  }
  const body = await assembleChatCompletion(chunks());
  assert.deepEqual(body.choices[0], { index: 0, finish_reason: 'tool_calls', message: { role: 'assistant', content: null,
    reasoning_content: 'Seven is easy.', tool_calls: [
      { id: 'call_a', type: 'function', function: { name: 'eval', arguments: '{"code":"return 7"}' } },
      { id: 'call_b', type: 'function', function: { name: 'read_page', arguments: '{}' } }] } });
  assert.deepEqual(body.usage, { prompt_tokens: 12, completion_tokens: 5 });
  assert.equal(body.id, 'c1');
});

test('the HTTP transport streams, strips private schema keys, and decodes tool calls', async () => {
  const model = await server(() => [
    delta({ role: 'assistant', content: '' }),
    delta({ tool_calls: [{ index: 0, id: 'x', type: 'function', function: { name: 'eval', arguments: '{"code":' } }] }),
    delta({ tool_calls: [{ index: 0, function: { arguments: '"return 7"}' } }] }, 'tool_calls'),
    { id: 'c1', choices: [], usage: { prompt_tokens: 20, completion_tokens: 6 } }]);
  try {
    const stats = [];
    const turn = await chatCompletionModelTurn(httpChatTransport({ endpoint: model.endpoint, model: 'm' }),
      { onTurn: value => stats.push(value) })(request());
    assert.deepEqual(turn.calls, [['eval', { code: 'return 7' }]]);
    assert.equal(turn.prompt_tokens, 20); assert.equal(turn.completion_tokens, 6);
    const sent = model.bodies[0];
    assert.equal(sent.stream, true); assert.deepEqual(sent.stream_options, { include_usage: true });
    assert.equal(sent.seed, 3); assert.equal(sent.model, 'm');
    assert.equal(JSON.stringify(sent.tools).includes('x-natlang'), false);
    assert.equal(stats.length, 1); assert.equal(stats[0].retries, 0);
  } finally { await model.close(); }
});

test('a server that ignores stream, a truncated reply, and one malformed call are handled alike on every transport', async () => {
  const model = await server((body, count) => count === 1 ?
    { choices: [{ finish_reason: 'tool_calls', message: { tool_calls: [{ type: 'function', function: { name: 'eval', arguments: '{bad' } }] } }] } :
    count === 2 ? { choices: [{ finish_reason: 'tool_calls', message: { reasoning_content: 'Seven is the answer.', tool_calls: [{ type: 'function', function: { name: 'eval', arguments: '{"code":"7"}' } }] } }] } :
    { choices: [{ finish_reason: 'length', message: { content: 'partial', tool_calls: [{ type: 'function', function: { name: 'eval', arguments: '{"co' } }] } }] });
  try {
    const drive = chatCompletionModelTurn(httpChatTransport({ endpoint: model.endpoint, model: 'm' }));
    const retried = await drive(request());
    assert.deepEqual(retried.calls, [['eval', { code: '7' }]]);
    assert.equal(retried.reasoning, 'Seven is the answer.', 'the model turn carries the reasoning the server returned');
    assert.match(model.bodies[1].messages.at(-1).content, /last tool call was malformed/);
    const cut = await drive(request());
    assert.equal(cut.truncated, true); assert.deepEqual(cut.calls, []); assert.equal(cut.text, 'partial');
  } finally { await model.close(); }
});

test('tool aliases rename tools on the wire and back', async () => {
  const model = await server(() => ({ choices: [{ finish_reason: 'tool_calls', message: { tool_calls: [
    { type: 'function', function: { name: 'run_code', arguments: { code: '1' } } }] } }] }));
  try {
    const turn = await chatCompletionModelTurn(httpChatTransport({ endpoint: model.endpoint, model: 'm', stream: false }),
      { toolAliases: { eval: 'run_code' } })(request());
    assert.equal(model.bodies[0].tools[0].function.name, 'run_code'); assert.equal(model.bodies[0].stream, undefined);
    assert.deepEqual(turn.calls, [['eval', { code: '1' }]]);
  } finally { await model.close(); }
});
