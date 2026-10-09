/**
 * The natlang transport for the agent model (host/natlang-provider.ts) against a local fake model server: pi-ai's
 * context reaches the wire as natlang's model-turn request (system prompt, thinking, tool calls and results, tools), the
 * reply comes back as pi-ai's events and message, a Neuralese part reaches a Neuralese server unchanged, a text reader
 * refuses one loudly, and the declared reader is checked against the server at startup.
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
    aiHost: await load('host/ai.js'), chord: await import('@earendil-works/chord/context') };
  server = createServer((request, response) => {
    let data = '';
    request.on('data', chunk => { data += chunk; });
    request.on('end', () => {
      const body = data ? JSON.parse(data) : undefined;
      seen.push({ method: request.method, path: request.url, body });
      const send = ({ status, body }) => { response.writeHead(status, { 'content-type': 'application/json' }); response.end(JSON.stringify(body)); };
      if (request.url === '/v1/neuralese/info') return send(info);
      if (request.url.startsWith('/v1/neuralese/blocks/') && request.url.endsWith('/meta'))
        return send({ status: 200, body: { id: request.url.split('/')[4], dialect: 'd1', length: 4, width: 8, dtype: 'f32' } });
      if (request.url === '/v1/chat/completions') return send(reply(body));
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
  assert.deepEqual(request.body.chat_template_kwargs, { enable_thinking: true, preserve_thinking: true });
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
  assert.deepEqual(chats().at(-1).body.chat_template_kwargs, { enable_thinking: false, preserve_thinking: true });
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
  // The harness's ai service applies the same rule to every transport, and counts blocks in its estimates.
  const runtime = { models, signal: undefined };
  const service = m.aiHost.aiService(runtime, m.chord.BACKGROUND_CONTEXT);
  const refused = await service.turn(ref, messages, { thinkingLevel: 'off', sessionId: 's' });
  assert.match(refused.errorMessage, /^neuralese-unsupported-backend: /);
  const [withBlock, without] = service.estimateTokens([messages[0], { ...messages[0], content: [] }]);
  assert.equal(withBlock - without, m.provider.NEURALESE_BLOCK_TOKENS);
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
