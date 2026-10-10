import assert from 'node:assert/strict';
import test from 'node:test';
import { serveLocally } from '../dist/browser/neuralese-wasm.js';
import { fetchModel } from '../dist/model/chat-completion.js';

const encode = text => new TextEncoder().encode(text);

test('serveLocally: a streamed reply is answered at its first event and carries each event as it arrives', async () => {
  let release;
  const gate = new Promise(resolve => { release = resolve; });
  const stop = serveLocally('http://neuralese.stream.test', async (method, path, body, onEvent) => {
    if (path === '/v1/chat/completions') {
      onEvent(encode('{"n":1}'));
      await gate;  // the response is already out while the request still runs
      onEvent(encode('{"n":2}'));
      onEvent(encode('[DONE]'));
      return { status: 200, body: new Uint8Array(), contentType: 'text/event-stream' };
    }
    return { status: 201, body: encode(JSON.stringify({ method, path, body: new TextDecoder().decode(body) })), contentType: 'application/json' };
  });
  try {
    const response = await fetchModel('http://neuralese.stream.test/v1/chat/completions', { method: 'POST', body: '{"stream":true}' });
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('content-type'), 'text/event-stream');
    const reader = response.body.getReader();
    const first = await reader.read();
    assert.equal(new TextDecoder().decode(first.value), 'data: {"n":1}\n\n');
    release();
    let rest = '';
    for (let chunk = await reader.read(); !chunk.done; chunk = await reader.read()) rest += new TextDecoder().decode(chunk.value);
    assert.equal(rest, 'data: {"n":2}\n\ndata: [DONE]\n\n');
    const plain = await fetchModel('http://neuralese.stream.test/v1/neuralese/encode', { method: 'POST', body: '{"text":"x"}' });
    assert.equal(plain.status, 201);
    assert.deepEqual(await plain.json(), { method: 'POST', path: '/v1/neuralese/encode', body: '{"text":"x"}' });
  } finally { stop(); }
});
