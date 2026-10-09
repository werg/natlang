/**
 * Model output as a stream (plans/STREAMING.md §1.1–1.2): a turn's deltas come from the one place streamed chunks are
 * assembled, in order; the assembled turn equals the turn of the same reply sent whole; a Neuralese server's written
 * block arrives as one delta; the Neuralese driver streams only when the server declares it in `/v1/neuralese/info`.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { chatCompletionModelTurn, httpChatTransport } from '../dist/model/chat-completion.js';
import { openAICompatibleModelTurn } from '../dist/model/openai-compatible.js';
import { neuraleseServerModelTurn } from '../dist/model/neuralese-server.js';

const BLOCK = 'nz1_' + 'a'.repeat(52);
const tools = [{ type: 'function', function: { name: 'eval', parameters: { type: 'object',
  properties: { code: { type: 'string' } }, required: ['code'] } } }];
const request = () => ({ messages: [{ role: 'user', content: 'Write seven.' }], tools, seed: 3, max_tokens: null });
const chunk = (delta, finish = null, extra = {}) => ({ id: 'c1', object: 'chat.completion.chunk', created: 1, model: 'm',
  choices: [{ index: 0, delta, finish_reason: finish }], ...extra });

/** The reply as an OpenAI-style stream and as the whole body a non-streaming request gets. */
const plain = {
  chunks: [chunk({ role: 'assistant', reasoning_content: 'Seven ' }), chunk({ reasoning_content: 'is easy.' }),
    chunk({ content: 'Calling ' }), chunk({ content: 'eval.' }),
    chunk({ tool_calls: [{ index: 0, id: 'call_a', type: 'function', function: { name: 'run_code', arguments: '{"co' } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: 'de":"return 7"}' } }] }), chunk({}, 'tool_calls'),
    { id: 'c1', choices: [], usage: { prompt_tokens: 12, completion_tokens: 5 } }],
  whole: { id: 'c1', object: 'chat.completion', created: 1, model: 'm', choices: [{ index: 0, finish_reason: 'tool_calls',
    message: { role: 'assistant', content: 'Calling eval.', reasoning_content: 'Seven is easy.', tool_calls: [
      { id: 'call_a', type: 'function', function: { name: 'run_code', arguments: '{"code":"return 7"}' } }] } }],
    usage: { prompt_tokens: 12, completion_tokens: 5 } },
};

/** A Neuralese reply as the reference server streams it (serve/http.py `_stream`) and as its whole body. */
const blockMeta = { id: BLOCK, dialect: 'nd:test@1', length: 2, width: 4, dtype: 'f32', type: 'Neuralese<string>' };
const neuraleseMessage = { role: 'assistant', content: [{ type: 'text', text: 'Note: ' }, { type: 'neuralese', id: BLOCK },
  { type: 'text', text: ' done.' }] };
const neuraleseReply = {
  chunks: [chunk({ role: 'assistant' }), chunk({ content: 'Note: ' }),
    chunk({ content: [{ type: 'neuralese', id: BLOCK }] }, null, { neuralese: { block: blockMeta } }),
    chunk({ content: ' done.' }),
    chunk({}, 'stop', { x_natlang_message: neuraleseMessage, usage: { prompt_tokens: 9, completion_tokens: 4 },
      neuralese: { dialect: 'nd:test@1', blocks: [blockMeta] } })],
  whole: { id: 'c1', object: 'chat.completion', created: 1, model: 'm',
    choices: [{ index: 0, finish_reason: 'stop', message: neuraleseMessage }],
    usage: { prompt_tokens: 9, completion_tokens: 4 }, neuralese: { dialect: 'nd:test@1', blocks: [blockMeta] } },
};

/** A local server: chat completions answer `reply` streamed when the request asks for it; `info` is the Neuralese info. */
async function server(reply, info) {
  const bodies = [];
  const instance = createServer((incoming, response) => {
    let text = '';
    incoming.setEncoding('utf8'); incoming.on('data', part => { text += part; });
    incoming.on('end', () => {
      if (incoming.url === '/v1/neuralese/info' && info) {
        response.writeHead(200, { 'content-type': 'application/json' }); response.end(JSON.stringify(info)); return;
      }
      if (incoming.url !== '/v1/chat/completions') { response.writeHead(404); response.end(); return; }
      const body = JSON.parse(text); bodies.push(body);
      if (!body.stream) { response.writeHead(200, { 'content-type': 'application/json' }); response.end(JSON.stringify(reply.whole)); return; }
      response.writeHead(200, { 'content-type': 'text/event-stream' });
      // One write per chunk, split mid-line, as a real stream may deliver them.
      const wire = reply.chunks.map(item => `data: ${JSON.stringify(item)}\n\n`).join('') + 'data: [DONE]\n\n';
      const pieces = wire.match(/[\s\S]{1,23}/g);
      const send = () => pieces.length ? (response.write(pieces.shift()), setImmediate(send)) : response.end();
      send();
    });
  });
  await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
  return { endpoint: `http://127.0.0.1:${instance.address().port}`, bodies,
    close: () => new Promise(resolve => instance.close(resolve)) };
}

/** A turn without what differs by transport (the raw body). */
const content = ({ raw_response: _, ...turn }) => turn;

test('streamed deltas arrive in order, with runtime tool names, and the assembled turn equals the whole one', async () => {
  const model = await server(plain);
  try {
    const deltas = [];
    const drive = stream => chatCompletionModelTurn(httpChatTransport({ endpoint: model.endpoint, model: 'm', stream }),
      { toolAliases: { eval: 'run_code' } });
    const streamed = await drive(true)(request(), undefined, { onDelta: delta => deltas.push(delta) });
    assert.deepEqual(deltas, [
      { type: 'reasoning', text: 'Seven ' }, { type: 'reasoning', text: 'is easy.' },
      { type: 'text', text: 'Calling ' }, { type: 'text', text: 'eval.' },
      { type: 'tool_call', index: 0, id: 'call_a', name: 'eval', arguments: '{"co' },
      { type: 'tool_call', index: 0, arguments: 'de":"return 7"}' }]);
    const whole = [];
    const unstreamed = await drive(false)(request(), undefined, { onDelta: delta => whole.push(delta) });
    assert.deepEqual(whole, [], 'a reply received whole emits no deltas');
    assert.deepEqual(content(streamed), content(unstreamed));
    assert.deepEqual(streamed.calls, [['eval', { code: 'return 7' }]]);
    assert.equal(deltas.filter(item => item.type === 'text').map(item => item.text).join(''), streamed.text);
  } finally { await model.close(); }
});

test('the OpenAI-compatible driver passes deltas through its scheduler; a throwing observer does not affect the turn', async () => {
  const model = await server(plain);
  try {
    const drive = openAICompatibleModelTurn({ endpoint: model.endpoint, model: 'm', concurrency: 2 });
    const deltas = [];
    const turn = await drive(request(), undefined, { onDelta: delta => { deltas.push(delta); throw new Error('observer'); } });
    assert.equal(deltas.length, 6);
    assert.equal(turn.text, 'Calling eval.');
  } finally { await model.close(); }
});

test('a malformed call that is sent again first voids the earlier attempt\'s deltas', async () => {
  let count = 0;
  const bad = { chunks: [chunk({ role: 'assistant', content: 'x' }),
    chunk({ tool_calls: [{ index: 0, id: 'a', type: 'function', function: { name: 'eval', arguments: '{bad' } }] }, 'tool_calls')] };
  const model = await server({ get chunks() { return ++count === 1 ? bad.chunks : plain.chunks; } });
  try {
    const deltas = [];
    await chatCompletionModelTurn(httpChatTransport({ endpoint: model.endpoint, model: 'm' }))(request(), undefined,
      { onDelta: delta => deltas.push(delta.type) });
    assert.deepEqual(deltas.slice(0, 3), ['text', 'tool_call', 'reset']);
    assert.equal(deltas.filter(type => type === 'reset').length, 1);
  } finally { await model.close(); }
});

test('a Neuralese server that declares streaming streams: a written block is one delta, and the final message decides', async () => {
  const model = await server(neuraleseReply, { dialects: ['nd:test@1'], stream: true });
  const whole = await server(neuraleseReply, { dialects: ['nd:test@1'] });
  try {
    const deltas = [];
    const streamed = await neuraleseServerModelTurn({ endpoint: model.endpoint, model: 'm' })(request(), undefined,
      { onDelta: delta => deltas.push(delta) });
    assert.equal(model.bodies[0].stream, true);
    assert.deepEqual(deltas, [{ type: 'text', text: 'Note: ' },
      { type: 'neuralese', part: { type: 'neuralese', id: BLOCK }, block: blockMeta }, { type: 'text', text: ' done.' }]);
    const unstreamed = await neuraleseServerModelTurn({ endpoint: whole.endpoint, model: 'm' })(request());
    assert.equal(whole.bodies[0].stream, undefined);
    assert.deepEqual(content(streamed), content(unstreamed));
    assert.deepEqual(streamed.raw_response.neuralese, neuraleseReply.whole.neuralese);
  } finally { await model.close(); await whole.close(); }
});

test('a Neuralese server that does not declare streaming gets whole requests', async () => {
  for (const info of [{ dialects: ['nd:test@1'] }, { dialects: ['nd:test@1'], stream: false }]) {
    const model = await server(neuraleseReply, info);
    try {
      const deltas = [];
      const turn = await neuraleseServerModelTurn({ endpoint: model.endpoint, model: 'm' })(request(), undefined,
        { onDelta: delta => deltas.push(delta) });
      assert.equal(model.bodies[0].stream, undefined, JSON.stringify(info));
      assert.deepEqual(deltas, []);
      assert.ok(turn.text.startsWith('Note: '));
    } finally { await model.close(); }
  }
});
