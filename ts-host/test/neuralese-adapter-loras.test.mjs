/**
 * Weight adapters on a server that applies them as LoRAs (the llama.cpp fork: `info.adapters === 'lora'`): the driver
 * uploads each bound adapter's LoRA once before the first request that binds it, from `adapterLoras`. A mock fork
 * records the uploads and refuses requests whose adapters have no LoRA, as the fork does.
 */
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { neuraleseServerModelTurn, referenceAdapterLoras } from '../dist/model/neuralese-server.js';

const ADAPTER = 'nz1_' + 'b'.repeat(52);

async function mockServer(kind) {
  const loaded = new Map(), seen = [];
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const body = Buffer.concat(chunks);
    seen.push(`${request.method} ${request.url}`);
    const send = (status, value, type = 'application/json') => {
      response.writeHead(status, { 'content-type': type });
      response.end(type === 'application/json' ? JSON.stringify(value) : value);
    };
    if (request.url === '/v1/neuralese/info') return send(200, { adapters: kind });
    if (request.url.startsWith('/v1/neuralese/blocks/')) return send(200, { id: ADAPTER });
    const lora = request.url.match(/^\/v1\/neuralese\/adapters\/(nz1_[a-z2-7]+)\/lora$/);
    if (lora && request.method === 'PUT') { loaded.set(lora[1], body); return send(201, { id: lora[1], loaded: true }); }
    if (lora && request.method === 'GET') return send(200, Buffer.from('GGUF-test'), 'application/octet-stream');
    if (request.url === '/v1/neuralese/decide') {
      const { adapters = [] } = JSON.parse(body.toString());
      const missing = adapters.find(a => kind === 'lora' && !loaded.has(a.id));
      if (missing) return send(409, { error: { code: 'neuralese-adapter-not-loaded', message: missing.id } });
      return send(200, { log_probs: [-1, -2], tokens: [1, 1] });
    }
    send(404, { error: { code: 'not-found' } });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  return { endpoint: `http://127.0.0.1:${server.address().port}`, loaded, seen, close: () => server.close() };
}

test('a LoRA server gets each bound adapter\'s LoRA once, then the request binds it', async () => {
  const fork = await mockServer('lora');
  const reference = await mockServer(['xs', 'tiny']);
  try {
    const driver = neuraleseServerModelTurn({ endpoint: fork.endpoint, model: 'm', adapterLoras: referenceAdapterLoras(reference.endpoint) });
    const question = { messages: [{ role: 'user', content: 'Which?' }], options: ['a', 'b'], adapters: [{ id: ADAPTER, scale: 1 }] };
    assert.deepEqual((await driver.decide(question)).log_probs, [-1, -2]);
    await driver.decide(question);
    assert.equal(fork.loaded.get(ADAPTER)?.toString(), 'GGUF-test');
    assert.equal(fork.seen.filter(line => line.startsWith('PUT /v1/neuralese/adapters/')).length, 1, 'uploaded once');
    assert.equal(reference.seen.filter(line => line.includes('/lora')).length, 1, 'exported once');
  } finally { fork.close(); reference.close(); }
});

test('a server that applies adapters itself gets no LoRA; a LoRA server without a source refuses', async () => {
  const reference = await mockServer(['xs', 'tiny']);
  const fork = await mockServer('lora');
  try {
    let asked = 0;
    const driver = neuraleseServerModelTurn({ endpoint: reference.endpoint, model: 'm', adapterLoras: async () => { asked++; return null; } });
    const question = { messages: [{ role: 'user', content: 'Which?' }], options: ['a', 'b'], adapters: [{ id: ADAPTER, scale: 1 }] };
    await driver.decide(question);
    assert.equal(asked, 0);
    const bare = neuraleseServerModelTurn({ endpoint: fork.endpoint, model: 'm' });
    await assert.rejects(() => bare.decide(question), /409/);
  } finally { fork.close(); reference.close(); }
});
