/** `scoreMany` of the Neuralese server driver against a fake server (plans/BATCHED_EXECUTION.md, contract). */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { neuraleseServerModelTurn } from '../dist/model/neuralese-server.js';

async function fake(handler) {
  const posts = [];
  const instance = createServer((incoming, response) => {
    const chunks = [];
    incoming.on('data', chunk => chunks.push(chunk));
    incoming.on('end', () => {
      const body = chunks.length ? JSON.parse(Buffer.concat(chunks).toString()) : null;
      posts.push({ path: incoming.url, body });
      handler(incoming.url, body, response);
    });
  });
  await new Promise(resolve => instance.listen(0, '127.0.0.1', resolve));
  return { posts, endpoint: `http://127.0.0.1:${instance.address().port}`,
    close: () => new Promise(resolve => instance.close(resolve)) };
}
const json = (response, status, body) => { response.writeHead(status, { 'content-type': 'application/json' }); response.end(JSON.stringify(body)); };
const user = text => [{ role: 'user', content: text }];

test('decisions arriving together are one POST /v1/natlang/score; prompts are adjacent; an item error stays with its caller', async () => {
  const server = await fake((path, body, response) => {
    if (path !== '/v1/natlang/score') return json(response, 404, {});
    json(response, 200, { results: body.items.map(item => item.continuations[0] === 'boom' ? { error: 'too long' } :
      { log_probs: item.continuations.map((_, index) => -index), tokens: item.continuations.map(() => 1) }) });
  });
  try {
    const { decide } = neuraleseServerModelTurn({ endpoint: server.endpoint, model: 'm' });
    const results = await Promise.allSettled([
      decide({ messages: user('a'), options: ['x', 'y'] }), decide({ messages: user('b'), options: ['boom'] }),
      decide({ messages: user('a'), options: ['z'] })]);
    assert.equal(server.posts.length, 1);
    assert.deepEqual(server.posts[0].body.items.map(item => item.messages[0].content), ['a', 'a', 'b']);
    assert.deepEqual(results[0].value.log_probs, [0, -1]);
    assert.deepEqual(results[2].value.log_probs, [0]);
    assert.equal(results[1].status, 'rejected');
    assert.match(String(results[1].reason), /too long/);
  } finally { await server.close(); }
});

test('a server without the score endpoint is asked per item through /v1/neuralese/decide, and not asked again', async () => {
  const server = await fake((path, body, response) => path === '/v1/neuralese/decide'
    ? json(response, 200, { log_probs: body.options.map(() => -2), tokens: body.options.map(() => 1) }) : json(response, 404, {}));
  try {
    const { decide } = neuraleseServerModelTurn({ endpoint: server.endpoint, model: 'm' });
    const run = () => Promise.all([decide({ messages: user('a'), options: ['x'] }), decide({ messages: user('b'), options: ['y', 'z'] })]);
    const first = await run();
    assert.deepEqual(first[1].log_probs, [-2, -2]);
    await run();
    assert.equal(server.posts.filter(post => post.path === '/v1/natlang/score').length, 1);
    assert.equal(server.posts.filter(post => post.path === '/v1/neuralese/decide').length, 4);
  } finally { await server.close(); }
});
