import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { createResolvedModelSession, openAICompatibleModelTurn, resolveModelChoice } from '../dist/index.js';
import { executorIdentityForChoice } from '../dist/model/config.js';

/** A chat-completions server that answers slowly, counts requests in flight, and scores replies when asked. */
async function slowServer({ scores = true } = {}) {
  const seen = { active: 0, most: 0, requests: 0 };
  const server = createServer(async (request, response) => {
    let text = '';
    for await (const chunk of request) text += chunk;
    const body = JSON.parse(text);
    seen.requests++; seen.active++; seen.most = Math.max(seen.most, seen.active);
    await new Promise(resolve => setTimeout(resolve, 25));
    seen.active--;
    const reply = body.prompt_logprobs === undefined
      ? { choices: [{ message: { role: 'assistant', content: 'ok' }, finish_reason: 'stop' }] }
      : { choices: [{ message: { role: 'assistant', content: '' } }],
        ...scores ? { prompt_logprobs: [null, { [body.messages.at(-1).content]: { logprob: body.messages.at(-1).content === '"a"' ? -0.1 : -2 } }] } : {} };
    response.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(reply));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return { seen, endpoint: `http://127.0.0.1:${server.address().port}`, close: () => new Promise(resolve => server.close(resolve)) };
}
const turn = { messages: [{ role: 'user', content: 'hi' }], tools: [], seed: null, max_tokens: 8 };
const decision = { messages: [{ role: 'user', content: 'pick' }], options: ['"a"', '"b"', '"c"'] };

test('one request limit covers turns and decision scoring together', async () => {
  const server = await slowServer();
  try {
    const driver = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'm', concurrency: 2 });
    const [scores, ...turns] = await Promise.all([driver.decide(decision), ...Array.from({ length: 5 }, () => driver(turn))]);
    assert.deepEqual(turns.map(result => result.text), ['ok', 'ok', 'ok', 'ok', 'ok']);
    assert.deepEqual(scores.log_probs, [-0.1, -2, -2]);
    assert.equal(server.seen.requests, 8);
    assert.equal(server.seen.most, 2);
  } finally { await server.close(); }
});

test('a model session scores decisions under its profile limit and stops asking a server that cannot', async () => {
  const scoring = await slowServer(), plain = await slowServer({ scores: false });
  try {
    const session = createResolvedModelSession(resolveModelChoice({ endpoint: scoring.endpoint, model: 'm', concurrency: 1 }), {});
    const [scores] = await Promise.all([session.decide(decision), session.turn(turn), session.turn(turn)]);
    assert.deepEqual(scores.log_probs, [-0.1, -2, -2]);
    assert.equal(scoring.seen.most, 1);
    await session.close();
    const unscored = createResolvedModelSession(resolveModelChoice({ endpoint: plain.endpoint, model: 'm' }), {});
    await assert.rejects(unscored.decide(decision), /^Error: decision-unsupported/);
    const asked = plain.seen.requests;
    await assert.rejects(unscored.decide(decision), /^Error: decision-unsupported/);
    assert.equal(plain.seen.requests, asked, 'not asked again');
    await unscored.close();
  } finally { await scoring.close(); await plain.close(); }
});

test('profile concurrency is validated and does not change the executor identity', () => {
  assert.throws(() => resolveModelChoice({ endpoint: 'http://127.0.0.1:1', model: 'm', concurrency: 0 }), /concurrency must be an integer >= 1/);
  const limited = executorIdentityForChoice(resolveModelChoice({ endpoint: 'http://127.0.0.1:1', model: 'm', concurrency: 3 }));
  assert.deepEqual(limited, executorIdentityForChoice(resolveModelChoice({ endpoint: 'http://127.0.0.1:1', model: 'm' })));
});
