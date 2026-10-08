// The default context budget follows the context window the model server reports.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer } from 'node:http';
import { openAICompatibleModelTurn } from '../dist/index.js';
import { defaultContextBudget } from '../dist/native/agent.js';

async function serve(routes) {
  const server = createServer((request, response) => {
    const body = routes[request.url];
    response.writeHead(body ? 200 : 404, { 'content-type': 'application/json' });
    response.end(body ? JSON.stringify(body) : '{}');
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return { endpoint: `http://127.0.0.1:${server.address().port}`, close: () => new Promise(resolve => server.close(resolve)) };
}

test('a vLLM server reports max_model_len; the budget leaves an eighth for the reply', async () => {
  const server = await serve({ '/v1/models': { data: [{ id: 'other', max_model_len: 8192 }, { id: 'm', max_model_len: 65536 }] } });
  try {
    const driver = openAICompatibleModelTurn({ endpoint: `${server.endpoint}/v1`, model: 'm' });
    assert.equal(await driver.contextWindow(), 65536);
    assert.equal(await defaultContextBudget(driver), 57344);
  } finally { await server.close(); }
});

test('a llama.cpp server reports n_ctx under /props', async () => {
  const server = await serve({ '/v1/models': { data: [{ id: 'gguf' }] }, '/props': { default_generation_settings: { n_ctx: 8192 } } });
  try {
    const driver = openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'gguf' });
    assert.equal(await driver.contextWindow(), 8192);
    assert.equal(await defaultContextBudget(driver), 8192 - 2048);
  } finally { await server.close(); }
});

test('without a reported window the budget is 16384', async () => {
  const server = await serve({});
  try {
    assert.equal(await defaultContextBudget(openAICompatibleModelTurn({ endpoint: server.endpoint, model: 'x' })), 16384);
    assert.equal(await defaultContextBudget(async () => ({})), 16384);
  } finally { await server.close(); }
});
