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
import { nativeStateSentinel } from '../dist/native/neuralese.js';
import { MemoryNeuraleseStore, encodeNativeState, decodeNativeState, nativeStateContentId } from '../dist/native/neuralese-store.js';

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

test('native state parts survive streamed and whole chat-completion turns as runtime markers', async () => {
  const id = 'nzs1_' + 'b'.repeat(52);
  const message = { role: 'assistant', content: [{ type: 'text', text: 'Used ' }, { type: 'native_state', id }] };
  const capture = { schema: 'natlang.native-rl-sampling/1', prompt: [], samples: [], writes: [],
    sampling_law: { temperature: 1, seed: 9, guidance: false, probabilities: 'post-processing-normalized' } };
  const whole = { choices: [{ finish_reason: 'stop', message }], native_state: { blocks: [{ id }] },
    x_natlang_rl_capture: capture };
  const deltas = [];
  const direct = await chatCompletionModelTurn(async () => whole)(request(), undefined,
    { onDelta: delta => deltas.push(delta) });
  assert.equal(direct.text, `Used ${nativeStateSentinel(id)}`);
  assert.deepEqual(direct.raw_response.x_natlang_rl_capture, capture);
  const streamed = await chatCompletionModelTurn(async (_body, _signal, meta) => (async function* () {
    yield chunk({ content: 'Used ' });
    yield chunk({ content: [{ type: 'native_state', id }] }, null, { native_state: { block: { id } } });
    yield chunk({}, 'stop', { x_natlang_message: message, native_state: { blocks: [{ id }] }, x_natlang_rl_capture: capture });
  })())(request(), undefined, { onDelta: delta => deltas.push(delta) });
  assert.equal(streamed.text, `Used ${nativeStateSentinel(id)}`);
  assert.deepEqual(streamed.raw_response.x_natlang_rl_capture, capture);
  assert.deepEqual(deltas.find(delta => delta.type === 'native_state'),
    { type: 'native_state', part: { type: 'native_state', id }, block: { id } });
});

test('native server transport restores input artifacts and archives producer artifacts through shared endpoints', async () => {
  const nativeMetadata = text => ({ schema: 'neuralese.native-state/1', text, architecture: 'fixture',
    tokenizer_fingerprint: '1'.repeat(64), checkpoint_sha256: '2'.repeat(64), embedding_weight_sha256: '3'.repeat(64),
    layer_kinds: ['attention', 'finite_convolution'] });
  const tensors = () => ({
    token_ids: { dtype: 'I64', shape: [2], data: Uint8Array.from([11, 0, 0, 0, 0, 0, 0, 0, 12, 0, 0, 0, 0, 0, 0, 0]) },
    positions: { dtype: 'I64', shape: [2], data: Uint8Array.from([7, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0]) },
    embeddings: { dtype: 'BF16', shape: [2, 2], data: Uint8Array.from([205, 61, 77, 62, 154, 62, 205, 62]) },
    'layers.0.raw_k': { dtype: 'F16', shape: [2, 1, 2], data: Uint8Array.from([0, 60, 0, 60, 0, 60, 0, 60]) },
    'layers.0.v': { dtype: 'F16', shape: [2, 1, 2], data: new Uint8Array(8) },
    'layers.1.gated_inputs': { dtype: 'F32', shape: [2, 3], data: Uint8Array.from([0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63,
      0, 0, 128, 63, 0, 0, 128, 63, 0, 0, 128, 63]) },
  });
  const artifact = text => ({ metadata: nativeMetadata(text), tensors: tensors() });
  const local = new MemoryNeuraleseStore();
  const input = await local.putNativeState(artifact('input'));
  const output = artifact('output');
  const outputId = nativeStateContentId(output.metadata, output.tensors);
  const remoteStates = new Map([[outputId, encodeNativeState(output)]]);
  const uploads = [], bodies = [];
  const instance = createServer((incoming, response) => {
    const chunks = [];
    incoming.on('data', chunk => chunks.push(chunk));
    incoming.on('end', () => {
      const body = Buffer.concat(chunks);
      if (incoming.url === '/v1/neuralese/info') {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ dialects: ['native-state/fixture'], native_state: { enabled: true }, stream: false })); return;
      }
      if (incoming.url === '/v1/chat/completions') {
        bodies.push(JSON.parse(body.toString('utf8')));
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: [
          { type: 'text', text: 'Used ' }, { type: 'native_state', id: outputId }] } }],
          native_state: { blocks: [{ id: outputId, ...nativeMetadata('output') }] } })); return;
      }
      const match = /^\/v1\/neuralese\/native-states\/(nzs1_[a-z2-7]+)(\/meta)?$/.exec(incoming.url ?? '');
      if (match && incoming.method === 'PUT') {
        const decoded = decodeNativeState(new Uint8Array(body)); uploads.push(decoded.id); remoteStates.set(decoded.id, new Uint8Array(body));
        response.writeHead(201, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ id: decoded.id })); return;
      }
      if (match && match[2] === '/meta') {
        response.writeHead(remoteStates.has(match[1]) ? 200 : 404, { 'content-type': 'application/json' });
        response.end(JSON.stringify(remoteStates.has(match[1]) ? { id: match[1] } : {})); return;
      }
      if (match) {
        const bytes = remoteStates.get(match[1]);
        if (!bytes) { response.writeHead(404); response.end(); return; }
        response.writeHead(200, { 'content-type': 'application/octet-stream' }); response.end(bytes); return;
      }
      response.writeHead(404); response.end();
    });
  });
  await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
  try {
    const driver = neuraleseServerModelTurn({ endpoint: `http://127.0.0.1:${instance.address().port}`, model: 'm', store: local });
    const turn = await driver({ messages: [{ role: 'user', content: [{ type: 'native_state', id: input.id }] }],
      tools: [], seed: 9, max_tokens: null });
    assert.deepEqual(uploads, [input.id]);
    assert.deepEqual(bodies[0].messages[0].content, [{ type: 'native_state', id: input.id }]);
    assert.equal(turn.text, `Used ${nativeStateSentinel(outputId)}`);
    assert.ok(await local.hasNativeState(outputId));
    assert.equal((await local.getNativeState(outputId)).id, outputId);
    await driver({ messages: [{ role: 'assistant', tool_calls: [{ type: 'function', id: 'call_1',
      function: { name: 'consume', arguments: JSON.stringify({ value: { type: 'native_state', id: input.id } }) } }] }],
      tools: [], seed: 10, max_tokens: null });
    assert.deepEqual(JSON.parse(bodies[1].messages[0].tool_calls[0].function.arguments),
      { value: [{ type: 'native_state', id: input.id }] });
  } finally { await new Promise(resolve => instance.close(resolve)); }
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
