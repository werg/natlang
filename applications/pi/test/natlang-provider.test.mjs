/**
 * The natlang transport for the agent model (host/natlang-provider.ts) against a local fake model server: pi-ai's
 * context reaches the wire as natlang's model-turn request (system prompt, thinking, tool calls and results, tools), the
 * reply comes back as pi-ai's events and message, a Neuralese part reaches a Neuralese server unchanged, a text reader
 * refuses one loudly, and the declared reader is checked against the server at startup. A streamed reply's deltas
 * become pi-ai's events as they arrive, and the final turn decides the message (plans/STREAMING.md §1.4).
 *
 * Run: node --test applications/pi/test/natlang-provider.test.mjs (builds the app into .natlang/test-build-provider first).
 */
import assert from 'node:assert/strict';
import { test, before, after } from 'node:test';
import { createServer } from 'node:http';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { join } from 'node:path';

const root = fileURLToPath(new URL('..', import.meta.url));
const hostDist = fileURLToPath(new URL('../../../ts-host/dist/', import.meta.url));
const outDir = join(root, '.natlang', 'test-build-provider');
const BLOCK = 'nz1_' + 'a'.repeat(24);
let m, server, endpoint;
/** What the fake server was sent, by path, and what it answers next to a chat request. */
const seen = [];
let reply = () => ({ status: 200, body: { choices: [{ message: { content: 'ok' }, finish_reason: 'stop' }] } });
let info = { status: 200, body: { dialects: ['d1'], width: 8, dtype: 'f32', max_block_length: 32 } };

before(async () => {
  const natlang = await import(pathToFileURL(join(hostDist, 'index.js')).href);
  const runtime = pathToFileURL(join(hostDist, 'index.js'));
  const result = natlang.buildProject({ project: root, outDir, runtimeModule: { url: runtime.href, path: fileURLToPath(runtime),
    types: join(hostDist, 'index.d.ts'), specifiers: ['@natlang/node'] } });
  if (!result.ok) throw new Error(natlang.formatDiagnostics(result.diagnostics));
  const load = path => import(pathToFileURL(join(outDir, path)).href);
  m = { natlang, provider: await load('host/natlang-provider.js'), main: await load('main.js'), ai: await import('@earendil-works/pi-ai'),
    aiHost: await load('host/ai.js'), durableHost: await load('host/durable.js'), chord: await import('@earendil-works/chord/context') };
  server = createServer((request, response) => {
    let data = '';
    request.on('data', chunk => { data += chunk; });
    request.on('end', () => {
      const body = data ? JSON.parse(data) : undefined;
      seen.push({ method: request.method, path: request.url, body });
      const send = ({ status, body }) => { response.writeHead(status, { 'content-type': 'application/json' }); response.end(JSON.stringify(body)); };
      if (request.url === '/v1/neuralese/info') return send(info);
      if (request.url.startsWith('/v1/neuralese/blocks/') && request.url.endsWith('/meta')) {
        const id = request.url.split('/')[4];
        // A block whose ID starts nz1_lost is one the server does not have.
        if (id.startsWith('nz1_lost')) return send({ status: 404, body: { error: { message: 'neuralese-unknown-block' } } });
        return send({ status: 200, body: { id, dialect: 'd1', length: 4, width: 8, dtype: 'f32' } });
      }
      if (request.url === '/v1/chat/completions') {
        const answer = reply(body);
        if (!answer.chunks) return send(answer);
        // A streamed reply: its chunks as server-sent events, the ones from `hold` on only once `release` resolves.
        response.writeHead(200, { 'content-type': 'text/event-stream' });
        const write = chunks => { for (const item of chunks) response.write(`data: ${JSON.stringify(item)}\n\n`); };
        const hold = answer.hold ?? answer.chunks.length;
        write(answer.chunks.slice(0, hold));
        return void Promise.resolve(answer.release).then(() => {
          write(answer.chunks.slice(hold));
          response.end('data: [DONE]\n\n');
        });
      }
      send({ status: 404, body: { error: { message: 'not found' } } });
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  endpoint = `http://127.0.0.1:${server.address().port}`;
});

after(() => server?.close());

const chats = () => seen.filter(item => item.path === '/v1/chat/completions');
const agent = (...extra) => m.main.agentModels(['--agent-endpoint', endpoint, '--agent-model', 'fake', '--agent-transport', 'natlang', ...extra]);
const usage = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } };

test('a turn round-trips text, thinking and tool calls through the natlang transport', async () => {
  seen.length = 0;
  reply = () => ({ status: 200, body: { id: 'r1', choices: [{ finish_reason: 'tool_calls', message: { content: 'Reading it.',
    reasoning_content: 'I should read a.txt.', tool_calls: [{ id: 'call_b', type: 'function', function: { name: 'read', arguments: '{"path":"a.txt"}' } }] } }],
    usage: { prompt_tokens: 120, completion_tokens: 30 } } });
  const { models, ref } = await agent();
  const model = models.getModel(ref.provider, ref.modelId);
  assert.deepEqual(model.reader, { kind: 'text' });
  const tools = [{ name: 'read', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } }];
  const context = { systemPrompt: 'You are pi.', tools, messages: [
    { role: 'user', content: 'Fix it', timestamp: 1 },
    { role: 'assistant', content: [{ type: 'thinking', thinking: 'Look first.' }, { type: 'text', text: 'Looking.' },
      { type: 'toolCall', id: 'call_a', name: 'read', arguments: { path: 'b.txt' } }],
      api: model.api, provider: 'agent', model: 'fake', usage, stopReason: 'toolUse', timestamp: 2 },
    { role: 'toolResult', toolCallId: 'call_a', toolName: 'read', content: [{ type: 'text', text: 'line 1' }, { type: 'text', text: 'line 2' }],
      isError: false, timestamp: 3 },
  ] };
  const stream = models.streamSimple(model, context, { reasoning: 'high', maxTokens: 500 });
  const events = [];
  for await (const event of stream) events.push(event.type);
  const message = await stream.result();
  assert.deepEqual(events, ['start', 'thinking_start', 'thinking_delta', 'thinking_end', 'text_start', 'text_delta', 'text_end',
    'toolcall_start', 'toolcall_delta', 'toolcall_end', 'done']);
  assert.equal(message.stopReason, 'toolUse');
  assert.equal(message.responseId, 'r1');
  assert.deepEqual(message.content, [{ type: 'thinking', thinking: 'I should read a.txt.' }, { type: 'text', text: 'Reading it.' },
    { type: 'toolCall', id: 'call_b', name: 'read', arguments: { path: 'a.txt' } }]);
  assert.equal(message.usage.input, 120);
  assert.equal(message.usage.output, 30);
  assert.equal(message.usage.totalTokens, 150);

  const [request] = chats();
  assert.equal(request.body.model, 'fake');
  assert.equal(request.body.max_tokens, 500);
  assert.deepEqual(request.body.chat_template_kwargs, { enable_thinking: true });
  assert.deepEqual(request.body.tools, [{ type: 'function', function: tools[0] }]);
  assert.deepEqual(request.body.messages, [
    { role: 'system', content: 'You are pi.' },
    { role: 'user', content: 'Fix it' },
    { role: 'assistant', content: 'Looking.', reasoning: 'Look first.', reasoning_content: 'Look first.',
      tool_calls: [{ id: 'call_a', type: 'function', function: { name: 'read', arguments: '{"path":"b.txt"}' } }] },
    { role: 'tool', tool_call_id: 'call_a', content: 'line 1\nline 2' },
  ]);

  // Without thinking, the other driver: the template argument says so.
  reply = () => ({ status: 200, body: { choices: [{ finish_reason: 'length', message: { content: 'cut' } }] } });
  const cut = await models.completeSimple(model, { messages: [{ role: 'user', content: 'hi', timestamp: 1 }] });
  assert.equal(cut.stopReason, 'length');
  assert.deepEqual(chats().at(-1).body.chat_template_kwargs, { enable_thinking: false });
});

test('server errors keep their text, so pi-ai classifies overflow and retries', async () => {
  const { models, ref } = await agent();
  const model = models.getModel(ref.provider, ref.modelId);
  const context = { messages: [{ role: 'user', content: 'hi', timestamp: 1 }] };
  reply = () => ({ status: 400, body: { error: { message: 'the request exceeds the available context size, try increasing it' } } });
  const overflow = await models.completeSimple(model, context);
  assert.equal(overflow.stopReason, 'error');
  assert.ok(m.ai.isContextOverflow(overflow), overflow.errorMessage);
  reply = () => ({ status: 503, body: { error: { message: 'busy' } } });
  const busy = await models.completeSimple(model, context);
  assert.equal(busy.stopReason, 'error');
  assert.ok(m.ai.isRetryableAssistantError(busy), busy.errorMessage);
});

test('a Neuralese part reaches a Neuralese reader unchanged, and a written block comes back as a part', async () => {
  seen.length = 0;
  const { models, ref } = await agent('--agent-reader', 'd1');
  const model = models.getModel(ref.provider, ref.modelId);
  assert.deepEqual(model.reader, { kind: 'neuralese', dialect: 'd1', maxBlockLength: 32 });
  reply = () => ({ status: 200, body: { choices: [{ finish_reason: 'stop', message: { content: [{ type: 'text', text: 'Noted: ' },
    { type: 'neuralese', id: BLOCK }] } }] } });
  const content = [{ type: 'text', text: 'Read this: ' }, { type: 'neuralese', id: BLOCK, value_type: 'string' }];
  const message = await models.completeSimple(model, { messages: [{ role: 'user', content, timestamp: 1 }] });
  assert.equal(message.stopReason, 'stop', message.errorMessage);
  assert.deepEqual(chats()[0].body.messages, [{ role: 'user', content }]);
  assert.deepEqual(message.content, [{ type: 'text', text: 'Noted: ' }, { type: 'neuralese', id: BLOCK }]);
  // The reply's block is sent back as it came when the conversation goes on.
  await models.completeSimple(model, { messages: [{ role: 'user', content, timestamp: 1 }, message,
    { role: 'user', content: 'And?', timestamp: 3 }] });
  assert.deepEqual(chats().at(-1).body.messages[1], { role: 'assistant', content: [{ type: 'text', text: 'Noted: ' }, { type: 'neuralese', id: BLOCK }] });
});

test('a text reader refuses a Neuralese part loudly, before any request', async () => {
  seen.length = 0;
  const { models, ref } = await agent();
  const model = models.getModel(ref.provider, ref.modelId);
  const messages = [{ role: 'toolResult', toolCallId: 'c', toolName: 'read', content: [{ type: 'neuralese', id: BLOCK }], isError: false, timestamp: 1 }];
  const message = await models.completeSimple(model, { messages: [{ role: 'user', content: 'go', timestamp: 0 },
    { role: 'assistant', content: [{ type: 'toolCall', id: 'c', name: 'read', arguments: {} }], api: model.api, provider: 'agent', model: 'fake',
      usage, stopReason: 'toolUse', timestamp: 1 }, ...messages] });
  assert.equal(message.stopReason, 'error');
  assert.match(message.errorMessage, /^neuralese-unsupported-backend: /);
  assert.equal(chats().length, 0);
  // The harness's ai service applies the same rule to every transport.
  const runtime = { models, signal: undefined };
  const service = m.aiHost.aiService(runtime, m.chord.BACKGROUND_CONTEXT);
  const refused = await service.turn(ref, messages, { thinkingLevel: 'off', sessionId: 's' });
  assert.match(refused.errorMessage, /^neuralese-unsupported-backend: /);
});

/** A stored block of `length` vectors of width 8, and its ID. */
async function storedBlock(store, length, seed = 1) {
  const data = new Float32Array(length * 8).map((_, i) => seed + i);
  return (await store.put({ dialect: 'd1', length, width: 8, dtype: 'f32', data: new Uint8Array(data.buffer) })).id;
}
const blockMessage = ids => ({ role: 'user', content: [{ type: 'text', text: 'Read: ' }, ...ids.map(id => ({ type: 'neuralese', id }))], timestamp: 1 });

test('a token estimate counts each Neuralese block at its real length, from the runtime\'s store', async () => {
  const { models } = await agent('--agent-reader', 'd1');
  const store = new m.natlang.MemoryNeuraleseStore();
  const [short, long] = [await storedBlock(store, 3), await storedBlock(store, 17, 2)];
  const service = m.aiHost.aiService({ models, signal: undefined }, m.chord.BACKGROUND_CONTEXT, undefined, {}, store);
  const [none, one, both] = service.estimateTokens([blockMessage([]), blockMessage([short]), blockMessage([short, long])]);
  assert.equal(one - none, 3);
  assert.equal(both - none, 20);
  // The part stays a pure reference: the length is never written into it.
  assert.deepEqual(blockMessage([short]).content[1], { type: 'neuralese', id: short });
});

test('the crisp context estimate (pi-durable\'s, through durable.estimate) counts blocks as ai.estimateTokens does', async () => {
  const { models } = await agent('--agent-reader', 'd1');
  const store = new m.natlang.MemoryNeuraleseStore();
  const [short, long] = [await storedBlock(store, 3), await storedBlock(store, 17, 2)];
  const ai = m.aiHost.aiService({ models, signal: undefined }, m.chord.BACKGROUND_CONTEXT, undefined, {}, store);
  const runtime = { conversationId: 1, taskId: 1, now: () => 0 };
  const durable = m.durableHost.durableService(runtime, m.chord.BACKGROUND_CONTEXT, {}, {}, {}, ai.estimateTokens);
  const piDurable = m.durableHost.durableService(runtime, m.chord.BACKGROUND_CONTEXT, {}, {}, {});
  const view = messages => ({ head: null, entries: [], contributions: [], messages, sections: [], tools: [] });
  const text = blockMessage([]);
  // Crisp: the text estimate plus each block's real length, in the view and in extra alike.
  assert.equal(durable.estimate(view([blockMessage([short, long])]), []), durable.estimate(view([text]), []) + 20);
  assert.equal(durable.estimate(view([text]), [blockMessage([long])]), 2 * durable.estimate(view([text]), []) + 17);
  assert.equal(durable.estimate(view([blockMessage([short])]), []), ai.estimateTokens([blockMessage([short])])[0]);
  // An image part is pi-ai's estimate, unchanged.
  const image = { role: 'user', content: [{ type: 'text', text: 'see' }, { type: 'image', data: 'AAAA', mimeType: 'image/png' }], timestamp: 1 };
  assert.equal(durable.estimate(view([image]), []), piDurable.estimate(view([image]), []));
  assert.ok(durable.estimate(view([image]), []) > 1000);
  // A block the store does not know fails the estimate, naming it, instead of counting as an image.
  assert.throws(() => durable.estimate(view([blockMessage([BLOCK])]), []), /^Error: neuralese-unknown-block-length: /);
});

test('metadata the store lacks is fetched once from the agent model\'s server and kept in the store', async () => {
  const { models, ref } = await agent('--agent-reader', 'd1');
  const store = new m.natlang.MemoryNeuraleseStore();
  const service = m.aiHost.aiService({ models, signal: undefined }, m.chord.BACKGROUND_CONTEXT, undefined, {}, store);
  const messages = [blockMessage([BLOCK]), blockMessage([BLOCK])];
  assert.throws(() => service.estimateTokens(messages), new RegExp(`^Error: neuralese-unknown-block-length: .*${BLOCK}.*ai\\.blockMeta`));
  seen.length = 0;
  assert.deepEqual(await Promise.all([service.blockMeta(messages, ref), service.blockMeta(messages, ref)]), [[], []]);
  assert.deepEqual(await service.blockMeta(messages, ref), []);
  assert.deepEqual(seen.map(item => item.path), [`/v1/neuralese/blocks/${BLOCK}/meta`]);
  assert.equal(store.peek(BLOCK).length, 4);
  assert.equal(await store.has(BLOCK), false, 'noted metadata does not make the block present');
  const [withBlock, without] = service.estimateTokens([messages[0], blockMessage([])]);
  assert.equal(withBlock - without, 4);
});

test('a block whose length stays unknown is an error naming it, never a guess', async () => {
  const { models, ref } = await agent('--agent-reader', 'd1');
  const store = new m.natlang.MemoryNeuraleseStore();
  const lost = 'nz1_lost' + 'b'.repeat(20);
  const service = m.aiHost.aiService({ models, signal: undefined }, m.chord.BACKGROUND_CONTEXT, undefined, {}, store);
  assert.deepEqual(await service.blockMeta([blockMessage([lost])], ref), [lost]);
  assert.throws(() => service.estimateTokens([blockMessage([lost])]),
    new RegExp(`^Error: neuralese-unknown-block-length: .*${lost}.*GET /v1/neuralese/blocks/${lost}/meta.*it is lost`));
  // Without a store there is nowhere to read lengths from: the error says so.
  const bare = m.aiHost.aiService({ models, signal: undefined }, m.chord.BACKGROUND_CONTEXT);
  assert.throws(() => bare.estimateTokens([blockMessage([BLOCK])]), /^Error: neuralese-unknown-block-length: block nz1_a+ .*no process-local Neuralese store/);
});

test('the declared reader is checked against the server at startup', async () => {
  await assert.rejects(agent('--agent-reader', 'd2'), /^Error: neuralese-dialect-mismatch: .*"d2".*"d1"/);
  info = { status: 404, body: { error: { message: 'not found' } } };
  try {
    await assert.rejects(agent('--agent-reader', 'd1'), /neuralese-unsupported-backend: .*HTTP 404/);
    // A text reader needs no Neuralese server.
    assert.deepEqual((await agent()).models.getModel('agent', 'fake').reader, { kind: 'text' });
  } finally { info = { status: 200, body: { dialects: ['d1'], width: 8, dtype: 'f32', max_block_length: 32 } }; }
  await assert.rejects(m.main.agentModels(['--agent-endpoint', endpoint, '--agent-model', 'fake', '--agent-reader', 'd1']),
    /--agent-reader needs --agent-transport natlang/);
});

const chunk = (delta, finish = null, extra = {}) => ({ id: 'r2', object: 'chat.completion.chunk', created: 1, model: 'fake',
  choices: [{ index: 0, delta, finish_reason: finish }], ...extra });
const readTool = [{ name: 'read', description: 'Read a file', parameters: { type: 'object', properties: { path: { type: 'string' } }, required: ['path'] } }];
/** A stream's events, each with a copy of the partial message as it stood when the event was taken. */
async function collect(stream, onEvent = () => {}) {
  const events = [];
  for await (const event of stream) {
    events.push({ ...event, partial: event.partial && structuredClone(event.partial) });
    onEvent(event);
  }
  return { events, types: events.map(event => event.type), message: await stream.result() };
}
/** What the final message holds that the provider decides (its timestamp is the request's). */
const settled = message => ({ content: message.content, stopReason: message.stopReason, usage: message.usage,
  responseId: message.responseId, errorMessage: message.errorMessage });

test('a streamed reply reaches pi-ai as it arrives, and its message equals the reply sent whole', async () => {
  seen.length = 0;
  const { models, ref } = await agent();
  const model = models.getModel(ref.provider, ref.modelId);
  const context = { systemPrompt: 'You are pi.', tools: readTool, messages: [{ role: 'user', content: 'Fix it', timestamp: 1 }] };
  let release, timedOut = false;
  const released = new Promise(resolve => { release = resolve; });
  const timer = setTimeout(() => { timedOut = true; release(); }, 5000);
  reply = () => ({ hold: 6, release: released, chunks: [chunk({ role: 'assistant', reasoning_content: 'Let me ' }),
    chunk({ reasoning_content: 'look.' }), chunk({ content: 'Reading ' }), chunk({ content: 'it.' }),
    chunk({ tool_calls: [{ index: 0, id: 'call_s', type: 'function', function: { name: 'read', arguments: '{"pa' } }] }),
    chunk({ tool_calls: [{ index: 0, function: { arguments: 'th":"a.txt"}' } }] }),
    chunk({}, 'tool_calls'), { id: 'r2', choices: [], usage: { prompt_tokens: 40, completion_tokens: 9 } }] });
  // The server holds the end of the reply until the call's arguments have reached the consumer as deltas.
  let held;
  const streamed = await collect(models.streamSimple(model, context, { reasoning: 'high' }), event => {
    if (event.type === 'toolcall_delta' && event.delta.endsWith('}')) { held = structuredClone(event.partial.content); release(); }
  });
  clearTimeout(timer);
  assert.equal(timedOut, false, 'the deltas did not arrive before the reply ended');
  // While the reply is held, the partial already holds every part, the call's arguments parsed from its deltas.
  assert.deepEqual(held, [{ type: 'thinking', thinking: 'Let me look.' }, { type: 'text', text: 'Reading it.' },
    { type: 'toolCall', id: 'call_s', name: 'read', arguments: { path: 'a.txt' } }]);
  assert.equal(chats()[0].body.stream, true);
  assert.deepEqual(streamed.types, ['start', 'thinking_start', 'thinking_delta', 'thinking_delta', 'thinking_end',
    'text_start', 'text_delta', 'text_delta', 'text_end', 'toolcall_start', 'toolcall_delta', 'toolcall_delta', 'toolcall_end', 'done']);
  assert.deepEqual(streamed.events.filter(event => event.type.endsWith('_delta')).map(event => [event.contentIndex, event.delta]),
    [[0, 'Let me '], [0, 'look.'], [1, 'Reading '], [1, 'it.'], [2, '{"pa'], [2, 'th":"a.txt"}']]);
  assert.deepEqual(streamed.events.find(event => event.type === 'toolcall_end').toolCall,
    { type: 'toolCall', id: 'call_s', name: 'read', arguments: { path: 'a.txt' } });

  reply = () => ({ status: 200, body: { id: 'r2', choices: [{ finish_reason: 'tool_calls', message: { content: 'Reading it.',
    reasoning_content: 'Let me look.', tool_calls: [{ id: 'call_s', type: 'function', function: { name: 'read', arguments: '{"path":"a.txt"}' } }] } }],
    usage: { prompt_tokens: 40, completion_tokens: 9 } } });
  const whole = await collect(models.streamSimple(model, context, { reasoning: 'high' }));
  assert.deepEqual(settled(streamed.message), settled(whole.message));
  assert.deepEqual(streamed.message.content, [{ type: 'thinking', thinking: 'Let me look.' }, { type: 'text', text: 'Reading it.' },
    { type: 'toolCall', id: 'call_s', name: 'read', arguments: { path: 'a.txt' } }]);
  // A whole reply's parts follow each other in the same order: start, one delta, end.
  assert.deepEqual(whole.types, ['start', 'thinking_start', 'thinking_delta', 'thinking_end', 'text_start', 'text_delta', 'text_end',
    'toolcall_start', 'toolcall_delta', 'toolcall_end', 'done']);
});

test('a written Neuralese block streams as one block-reference event, and the final message decides', async () => {
  info = { status: 200, body: { ...info.body, stream: true } };
  try {
    seen.length = 0;
    const store = new m.natlang.MemoryNeuraleseStore();
    const { models, ref } = await m.main.agentModels(['--agent-endpoint', endpoint, '--agent-model', 'fake', '--agent-transport', 'natlang',
      '--agent-reader', 'd1'], undefined, store);
    const model = models.getModel(ref.provider, ref.modelId);
    const context = { messages: [{ role: 'user', content: 'Note it.', timestamp: 1 }] };
    const final = { role: 'assistant', content: [{ type: 'text', text: 'Noted: ' }, { type: 'neuralese', id: BLOCK }] };
    reply = () => ({ chunks: [chunk({ role: 'assistant', content: 'Noted: ' }),
      chunk({ content: [{ type: 'neuralese', id: BLOCK }] }, null, { neuralese: { block: { id: BLOCK, dialect: 'd1', length: 4 } } }),
      chunk({}, 'stop', { x_natlang_message: final })] });
    const streamed = await collect(models.streamSimple(model, context));
    assert.equal(chats()[0].body.stream, true);
    assert.deepEqual(streamed.types, ['start', 'text_start', 'text_delta', 'text_end', 'neuralese', 'done']);
    const block = streamed.events.find(event => event.type === 'neuralese');
    assert.equal(block.contentIndex, 1);
    assert.deepEqual(block.content, { type: 'neuralese', id: BLOCK });
    assert.deepEqual(block.partial.content[1], { type: 'neuralese', id: BLOCK });
    // The metadata the block streamed with is in the runtime's store, where estimates read its length.
    assert.equal(store.peek(BLOCK).length, 4);
    // The same reply sent whole gives the same message.
    info = { status: 200, body: { ...info.body, stream: false } };
    reply = () => ({ status: 200, body: { id: 'r2', choices: [{ finish_reason: 'stop', message: final }] } });
    const whole = await collect((await agent('--agent-reader', 'd1')).models.streamSimple(model, context));
    assert.deepEqual(whole.types, ['start', 'text_start', 'text_delta', 'text_end', 'neuralese', 'done']);
    assert.deepEqual(settled(streamed.message), settled(whole.message));
    assert.deepEqual(streamed.message.content, [{ type: 'text', text: 'Noted: ' }, { type: 'neuralese', id: BLOCK }]);

    // The final message wins where the stream said otherwise: an open part it extends is finished with the rest, a
    // part it contradicts ends as streamed, and the final parts follow whole.
    info = { status: 200, body: { ...info.body, stream: true } };
    const streamedModel = (await agent('--agent-reader', 'd1')).models;
    reply = () => ({ chunks: [chunk({ content: 'Fin' }), chunk({}, 'stop', { x_natlang_message: { role: 'assistant', content: 'Final.' } })] });
    const extended = await collect(streamedModel.streamSimple(model, context));
    assert.deepEqual(extended.events.filter(event => event.type === 'text_delta').map(event => event.delta), ['Fin', 'al.']);
    assert.deepEqual(extended.types, ['start', 'text_start', 'text_delta', 'text_delta', 'text_end', 'done']);
    assert.deepEqual(extended.message.content, [{ type: 'text', text: 'Final.' }]);
    reply = () => ({ chunks: [chunk({ content: 'Draft' }), chunk({ content: [{ type: 'neuralese', id: BLOCK }] }),
      chunk({}, 'stop', { x_natlang_message: { role: 'assistant', content: 'Answer.' } })] });
    const replaced = await collect(streamedModel.streamSimple(model, context));
    assert.deepEqual(replaced.types, ['start', 'text_start', 'text_delta', 'text_end', 'neuralese', 'text_start', 'text_delta', 'text_end', 'done']);
    assert.deepEqual(replaced.events.filter(event => event.type === 'text_start').map(event => event.contentIndex), [0, 0]);
    assert.deepEqual(replaced.events.at(-2).content, 'Answer.');
    assert.deepEqual(replaced.message.content, [{ type: 'text', text: 'Answer.' }]);
  } finally { info = { status: 200, body: { dialects: ['d1'], width: 8, dtype: 'f32', max_block_length: 32 } }; }
});

test('a re-sent request replaces the abandoned attempt in the partial; only the final reply is the message', async () => {
  seen.length = 0;
  const { models, ref } = await agent();
  const model = models.getModel(ref.provider, ref.modelId);
  let sent = 0;
  // The first reply's call has malformed arguments, so the driver sends the request again (its malformed-call retry).
  reply = () => ++sent === 1 ? { chunks: [chunk({ content: 'Calling.' }),
    chunk({ tool_calls: [{ index: 0, id: 'call_x', type: 'function', function: { name: 'read', arguments: '{bad' } }] }),
    chunk({}, 'tool_calls')] } : { chunks: [chunk({ content: 'Done.' }), chunk({}, 'stop')] };
  const { events, types, message } = await collect(models.streamSimple(model, { tools: readTool,
    messages: [{ role: 'user', content: 'Read a.txt', timestamp: 1 }] }));
  assert.equal(chats().length, 2);
  // The abandoned attempt's parts end as they stood; the re-sent request's parts start again at index 0.
  assert.deepEqual(types, ['start', 'text_start', 'text_delta', 'text_end', 'toolcall_start', 'toolcall_delta', 'toolcall_end',
    'text_start', 'text_delta', 'text_end', 'done']);
  const restart = types.indexOf('text_start', 2);
  assert.equal(events[restart].contentIndex, 0);
  assert.deepEqual(events[restart - 1].toolCall, { type: 'toolCall', id: 'call_x', name: 'read', arguments: {} });
  // The partial pi-durable publishes no longer holds the abandoned parts.
  assert.deepEqual(events[restart].partial.content.map(part => part.type), ['text']);
  assert.equal(message.stopReason, 'stop');
  assert.deepEqual(message.content, [{ type: 'text', text: 'Done.' }]);
});
