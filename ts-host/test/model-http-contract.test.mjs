// Mock-HTTP-server tests for model/openai-compatible.ts and the chat-completion transport (B8). Every test talks to a
// node:http server on an ephemeral 127.0.0.1 port that this file starts; no real model server is ever contacted.
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { chatCompletionModelTurn, httpChatTransport, limitedTransport, requestLimit } from '../dist/model/chat-completion.js';
import { openAICompatibleModelTurn, serverContextWindow } from '../dist/model/openai-compatible.js';

const tool = name => ({ type: 'function', function: { name, description: name, parameters: { type: 'object',
  properties: { code: { type: 'string', 'x-natlang-note': 'private' } }, required: ['code'] } } });
const request = (extra = {}) => ({ messages: [{ role: 'user', content: 'go' }], tools: [tool('eval'), tool('done')], seed: 1, max_tokens: null, ...extra });
const sse = (...chunks) => chunks.map(chunk => `data: ${typeof chunk === 'string' ? chunk : JSON.stringify(chunk)}\n\n`).join('');
const delta = (delta, finish = null) => ({ id: 'c1', model: 'm', created: 1, choices: [{ index: 0, delta, finish_reason: finish }] });
const usage = (prompt, completion, extra = {}) => ({ id: 'c1', choices: [], usage: { prompt_tokens: prompt, completion_tokens: completion, ...extra } });

/**
 * A mock server. `handler(context)` receives { req, res, body, count, path } for each request and answers it; the server
 * records every request (headers, parsed JSON body) and the highest number of requests open at once.
 */
async function mock(handler) {
  const seen = [];
  let open = 0, peak = 0, count = 0;
  const server = createServer((req, res) => {
    open++; peak = Math.max(peak, open);
    res.on('close', () => { open--; });
    const parts = [];
    req.on('data', part => parts.push(part));
    req.on('end', () => {
      const text = Buffer.concat(parts).toString('utf8');
      let body; try { body = text ? JSON.parse(text) : undefined; } catch { body = text; }
      const entry = { method: req.method, path: req.url, headers: req.headers, body };
      seen.push(entry);
      Promise.resolve(handler({ req, res, body, path: req.url, count: ++count, entry })).catch(error => { res.statusCode = 500; res.end(String(error)); });
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const root = `http://127.0.0.1:${server.address().port}`;
  return { root, endpoint: `${root}/v1`, seen, get peak() { return peak; },
    close: () => new Promise(resolve => { server.closeAllConnections?.(); server.close(resolve); }) };
}
const json = (res, value, status = 200, headers = {}) => { res.writeHead(status, { 'content-type': 'application/json', ...headers }); res.end(JSON.stringify(value)); };
const stream = (res, text, split) => {
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  if (!split) { res.end(text); return; }
  // Write in tiny slices so chunk boundaries fall inside lines and inside multi-byte characters.
  const bytes = Buffer.from(text, 'utf8');
  let at = 0;
  const next = () => { if (at >= bytes.length) { res.end(); return; } res.write(bytes.subarray(at, at += split)); setTimeout(next, 0); };
  next();
};
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
/** Poll until a condition holds (a bounded wait for an event the test cannot observe directly); fails with an error, not a guessed delay. */
const until = async (condition, limitMs = 10_000) => { const deadline = Date.now() + limitMs; while (!condition()) { if (Date.now() > deadline) throw new Error("condition not reached"); await wait(5); } };

// --- streaming assembly --------------------------------------------------------------------------------------------

test('SSE replies are assembled across arbitrary byte splits, CRLF line ends, comments and multi-byte text', async () => {
  const body = sse(delta({ role: 'assistant', content: 'café \u{1F600} ' }), delta({ content: 'ready' }, 'stop'), usage(11, 3), '[DONE]');
  const server = await mock(({ res }) => stream(res, `: keep-alive\r\n${body.replaceAll('\n\n', '\r\n\r\n')}`, 3));
  try {
    const turn = await openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm' })(request());
    assert.equal(turn.text, 'café \u{1F600} ready');
    assert.deepEqual(turn.calls, []);
    assert.equal(turn.prompt_tokens, 11); assert.equal(turn.completion_tokens, 3);
    assert.equal(server.seen[0].path, '/v1/chat/completions');
    assert.equal(server.seen[0].body.stream, true);
    assert.deepEqual(server.seen[0].body.stream_options, { include_usage: true });
  } finally { await server.close(); }
});

test('streamed tool calls arrive in fragments, in several indexes, with reasoning and cached-token accounting', async () => {
  const server = await mock(({ res }) => stream(res, sse(
    delta({ role: 'assistant', reasoning_content: 'think ' }),
    delta({ reasoning_content: 'hard', tool_calls: [{ index: 0, id: 'a', type: 'function', function: { name: 'ev', arguments: '' } }] }),
    delta({ tool_calls: [{ index: 0, function: { name: 'al', arguments: '{"code":' } }, { index: 1, id: 'b', function: { name: 'done', arguments: '{"code"' } }] }),
    delta({ tool_calls: [{ index: 1, function: { arguments: ':"x"}' } }, { index: 0, function: { arguments: '"1"}' } }] }, 'tool_calls'),
    usage(30, 9, { prompt_tokens_details: { cached_tokens: 20 } }), '[DONE]'), 7));
  try {
    const stats = [];
    const model = chatCompletionModelTurn(httpChatTransport({ endpoint: server.root, model: 'm' }), { onTurn: value => stats.push(value) });
    const turn = await model(request());
    assert.deepEqual(turn.calls, [['eval', { code: '1' }], ['done', { code: 'x' }]], 'name and argument fragments are concatenated per index');
    assert.equal(turn.reasoning, 'think hard');
    assert.equal(turn.raw_calls.length, 2);
    assert.deepEqual([stats[0].promptTokens, stats[0].completionTokens, stats[0].cachedTokens, stats[0].retries], [30, 9, 20, 0]);
  } finally { await server.close(); }
});

test('a server that answers one JSON body to a streaming request is accepted', async () => {
  const server = await mock(({ res }) => json(res, { id: 'c', choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'whole' } }], usage: { prompt_tokens: 2, completion_tokens: 1 } }));
  try {
    const turn = await openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm' })(request());
    assert.equal(turn.text, 'whole'); assert.equal(turn.prompt_tokens, 2);
  } finally { await server.close(); }
});

// --- tool calls written as text ------------------------------------------------------------------------------------

test('a reply that is only one tool\'s JSON arguments is that call; prose and ambiguous JSON stay text', async () => {
  const replies = ['```json\n{"code": "return 1"}\n```', 'I would write {"code": "return 1"} here.', '{"code": "x", "extra": 1}', '{}', '{"code": "both"}'];
  const server = await mock(({ res, count }) => json(res, { choices: [{ finish_reason: 'stop', message: { content: replies[count - 1] } }] }));
  try {
    // Offer one tool whose only parameter is `code`; two identical tools would make the JSON ambiguous.
    const model = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm' });
    const single = request({ tools: [tool('eval')] });
    assert.deepEqual((await model(single)).calls, [['eval', { code: 'return 1' }]]);
    assert.deepEqual((await model(single)).calls, [], 'prose around the JSON is a reply, not a call');
    assert.deepEqual((await model(single)).calls, [], 'a key the tool lacks is not that tool\'s arguments');
    assert.deepEqual((await model(single)).calls, [], 'a missing required parameter is not a call');
    assert.deepEqual((await openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm' })(request())).calls, [],
      'with two tools offered, JSON matching both is ambiguous');
  } finally { await server.close(); }
});

test('tool-call markup left in the text is retried once with the malformed-call notice, then returned empty', async () => {
  const server = await mock(({ res }) => json(res, { choices: [{ finish_reason: 'stop', message: { content: '<tool_call>{"name":"eval"}</tool_call>' } }] }));
  try {
    const exchanges = [];
    const turn = await openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm', onExchange: exchange => exchanges.push(exchange) })(request());
    assert.deepEqual(turn.calls, []);
    assert.equal(turn.text, '', 'markup never reads as the answer');
    assert.equal(server.seen.length, 2);
    assert.match(server.seen[1].body.messages.at(-1).content, /last tool call was malformed/);
    assert.equal(exchanges.length, 2);
  } finally { await server.close(); }
});

test('malformed arguments are retried once, then fail with the number of attempts', async () => {
  const server = await mock(({ res }) => json(res, { choices: [{ finish_reason: 'tool_calls', message: { tool_calls: [{ type: 'function', function: { name: 'eval', arguments: '{nope' } }] } }] }));
  try {
    await assert.rejects(openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm' })(request()), /malformed tool arguments after 2 attempts/);
    assert.equal(server.seen.length, 2);
  } finally { await server.close(); }
});

// --- request building ----------------------------------------------------------------------------------------------

test('requests carry the model, auth and custom headers, aliases, sampling fields and no private schema keys', async () => {
  const server = await mock(({ res }) => json(res, { choices: [{ finish_reason: 'tool_calls', message: { tool_calls: [{ type: 'function', function: { name: 'run_js', arguments: '{"code":"1"}' } }] } }] }));
  try {
    const model = openAICompatibleModelTurn({ endpoint: `${server.root}/`, model: 'served-model', apiKey: 'secret', headers: { 'x-trace': 't1' },
      toolAliases: { eval: 'run_js' }, request: { temperature: 0.2, max_tokens: 500, top_p: 0.9 } });
    const turn = await model(request({ temperature: 0.7, max_tokens: 100 }));
    assert.deepEqual(turn.calls, [['eval', { code: '1' }]], 'the alias is mapped back to the runtime name');
    const { headers, body } = server.seen[0];
    assert.equal(headers.authorization, 'Bearer secret');
    assert.equal(headers['x-trace'], 't1');
    assert.match(headers['content-type'], /application\/json/);
    assert.equal(body.model, 'served-model');
    assert.deepEqual([body.tools[0].function.name, body.tool_choice, body.top_p, body.temperature, body.seed], ['run_js', 'auto', 0.9, 0.7, 1]);
    assert.equal(body.max_tokens, 100, 'the smaller of the configured cap and the runtime allowance');
    assert.equal(JSON.stringify(body.tools).includes('x-natlang'), false);
    assert.equal(JSON.stringify(body.tools).includes('private'), false);
  } finally { await server.close(); }
});

// --- request limit -------------------------------------------------------------------------------------------------

test('concurrency bounds requests in flight, and a streamed reply holds its slot until it is read to the end', async () => {
  const server = await mock(async ({ res }) => { await wait(40); stream(res, sse(delta({ content: 'ok' }, 'stop'), usage(1, 1), '[DONE]')); });
  try {
    const model = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm', concurrency: 2 });
    const turns = await Promise.all(Array.from({ length: 6 }, () => model(request())));
    assert.equal(turns.length, 6);
    assert.ok(turns.every(turn => turn.text === 'ok'));
    assert.equal(server.seen.length, 6);
    assert.ok(server.peak <= 2, `peak ${server.peak} requests at once`);
    assert.ok(server.peak >= 2, 'the limit was reached, so the test exercised queueing');
  } finally { await server.close(); }
});

test('a request limit can be shared by several drivers, and an aborted waiter leaves the queue without taking a slot', async () => {
  const gate = [];
  const server = await mock(({ res }) => { gate.push(() => json(res, { choices: [{ finish_reason: 'stop', message: { content: 'ok' } }] })); });
  try {
    const limit = requestLimit(1);
    const a = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'a', concurrency: limit });
    const b = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'b', concurrency: limit });
    const first = a(request());
    await until(() => gate.length === 1);
    const controller = new AbortController();
    const waiting = b(request(), controller.signal);
    waiting.catch(() => {});
    await wait(30);
    assert.equal(gate.length, 1, 'the second driver is queued behind the shared limit');
    controller.abort(new Error('stop waiting'));
    await assert.rejects(waiting, /stop waiting|aborted/);
    gate.shift()();
    assert.equal((await first).text, 'ok');
    const third = b(request());
    await until(() => gate.length === 1);      // arrives only if the slot was freed
    gate.shift()();
    assert.equal((await third).text, 'ok');
  } finally { await server.close(); }
});

test('request limits reject sizes that are not positive integers, and a failed request frees its slot', async () => {
  for (const size of [0, -1, 1.5, NaN]) assert.throws(() => requestLimit(size), RangeError);
  const limit = requestLimit(1);
  const failing = limitedTransport(async () => { throw new Error('boom'); }, limit);
  await assert.rejects(failing({}), /boom/);
  const release = await limit.acquire();
  release(); release();                                       // releasing twice must not free a second slot
  const first = await limit.acquire();
  let second = false;
  const pending = limit.acquire().then(done => { second = true; return done; });
  await wait(10);
  assert.equal(second, false);
  first();
  (await pending)();
});

// --- errors --------------------------------------------------------------------------------------------------------

test('HTTP errors carry status, headers and provider retry metadata', async () => {
  const server = await mock(({ res, count }) => count === 1 ?
    json(res, { error: { code: 'rate_limit', message: 'slow down', provider_retryable: true, retry_after_ms: 1500 } }, 429, { 'retry-after': '2' }) :
    count === 2 ? (res.writeHead(502, { 'content-type': 'text/html' }), res.end('<html>bad gateway</html>')) :
    json(res, { error: 'plain' }, 401));
  try {
    const model = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm' });
    const limited = await model(request()).then(() => null, error => error);
    assert.match(limited.message, /^model HTTP 429: /);
    assert.deepEqual([limited.status, limited.providerCode, limited.providerRetryable, limited.retry_after_ms], [429, 'rate_limit', true, 1500]);
    assert.equal(limited.headers.get('retry-after'), '2');
    const gateway = await model(request()).then(() => null, error => error);
    assert.equal(gateway.status, 502);
    assert.match(gateway.message, /bad gateway/);
    assert.equal(gateway.providerCode, undefined);
    const denied = await model(request()).then(() => null, error => error);
    assert.equal(denied.status, 401);
  } finally { await server.close(); }
});

test('an error chunk inside a stream, and a stream that dies half way, reject the turn', async () => {
  const server = await mock(({ res, count }) => {
    if (count === 1) return stream(res, sse(delta({ content: 'par' }), { error: { message: 'out of memory', type: 'server_error' } }));
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write(sse(delta({ content: 'par' })));
    setTimeout(() => res.destroy(), 20);
  });
  try {
    const model = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm' });
    await assert.rejects(model(request()), /model stream error: .*out of memory/);
    await assert.rejects(model(request()));
  } finally { await server.close(); }
});

test('a refused connection names the URL, and an abort signal stops a pending request', async () => {
  const closed = await mock(() => {});
  const { root } = closed;
  await closed.close();
  await assert.rejects(openAICompatibleModelTurn({ endpoint: root, model: 'm' })(request()),
    error => error.message.startsWith(`model request to ${root}/v1/chat/completions failed:`));
  const hung = await mock(() => { /* never answers */ });
  try {
    const controller = new AbortController();
    const turn = openAICompatibleModelTurn({ endpoint: hung.root, model: 'm' })(request(), controller.signal);
    turn.catch(() => {});
    await wait(30);
    controller.abort();
    await assert.rejects(turn);
    await assert.rejects(openAICompatibleModelTurn({ endpoint: hung.root, model: 'm' })(request(), AbortSignal.abort()), /aborted/);
  } finally { await hung.close(); }
});

test('endpoint and model are required before any request is made', () => {
  assert.throws(() => openAICompatibleModelTurn({ endpoint: '', model: 'm' }), /endpoint and ID are required/);
  assert.throws(() => openAICompatibleModelTurn({ endpoint: 'http://127.0.0.1:1', model: '' }), /endpoint and ID are required/);
});

// --- context window and decisions ----------------------------------------------------------------------------------

test('the context window is read from vLLM, llama.cpp or a context_length field, and undefined when unknown', async () => {
  const pages = {
    vllm: { '/v1/models': { data: [{ id: 'other', max_model_len: 1 }, { id: 'm', max_model_len: 32768 }] } },
    llama: { '/v1/models': { data: [{ id: 'only', meta: { n_ctx_train: 4096 } }] }, '/props': { default_generation_settings: { n_ctx: 8192 } } },
    meta: { '/v1/models': { data: [{ id: 'only', meta: { n_ctx_train: 4096 } }] } },
    field: { '/v1/models': { data: [{ id: 'm', context_length: 16000 }] } },
    none: { '/v1/models': { data: [] } },
  };
  let current = pages.vllm;
  const server = await mock(({ res, path, entry }) => current[path] ? json(res, current[path]) : json(res, { error: 'nope' }, 404));
  try {
    const ask = () => serverContextWindow({ endpoint: server.endpoint, model: 'm', apiKey: 'k' });
    assert.equal(await ask(), 32768);
    assert.equal(server.seen[0].headers.authorization, 'Bearer k');
    current = pages.llama; assert.equal(await ask(), 8192);
    current = pages.meta; assert.equal(await ask(), 4096);
    current = pages.field; assert.equal(await ask(), 16000);
    current = pages.none; assert.equal(await ask(), undefined);
    const model = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm' });
    current = pages.vllm;
    assert.equal(await model.contextWindow(), 32768);
    current = pages.none;
    assert.equal(await model.contextWindow(), 32768, 'asked once per driver');
    assert.equal(await serverContextWindow({ endpoint: '', model: 'm' }), undefined);
  } finally { await server.close(); }
});

test('decide scores options from prompt_logprobs over a non-streaming request, and refuses servers without them', async () => {
  const server = await mock(({ res, body }) => {
    const reply = body.messages.at(-1).content;
    if (body.messages.length > 5) return json(res, { choices: [] });
    const rows = [null, { 11: { logprob: -1 } }, { 12: { logprob: -0.5 } },
      reply === 'yes' ? { 21: { logprob: -0.25 } } : { 22: { logprob: -2 } }];
    json(res, { choices: [], prompt_logprobs: rows });
  });
  try {
    const model = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm', request: { temperature: 1, max_tokens: 50, chat_template_kwargs: { think: false } } });
    const scored = await model.decide({ messages: [{ role: 'user', content: 'ok?' }], options: ['yes', 'no'] });
    assert.deepEqual(scored.log_probs, [-0.25, -2], 'tokens both options share are left out of the score');
    assert.deepEqual(scored.tokens, [1, 1]);
    const sent = server.seen[0].body;
    assert.equal(sent.stream, undefined, 'decisions are not streamed');
    assert.deepEqual([sent.prompt_logprobs, sent.max_tokens, sent.add_generation_prompt, sent.temperature], [0, 1, false, undefined]);
    assert.deepEqual(sent.chat_template_kwargs, { think: false }, 'extra request fields still apply');
    await assert.rejects(model.decide({ messages: Array(6).fill({ role: 'user', content: 'x' }), options: ['a'] }), /decision-unsupported/);
    await assert.rejects(model.decide({ messages: [], options: [] }), /at least one option/);
  } finally { await server.close(); }
});
