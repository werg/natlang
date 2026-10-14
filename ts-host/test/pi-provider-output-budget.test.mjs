import assert from 'node:assert/strict';
import { test } from 'node:test';
import { zstdDecompressSync } from 'node:zlib';
import { createPiModelBackend } from '../dist/model/pi-provider.js';

test('Codex Responses receives the collector action output budget', async () => {
  const claim = Buffer.from(JSON.stringify({ 'https://api.openai.com/auth':
    { chatgpt_account_id: 'local-payload-probe' } })).toString('base64url');
  let sent;
  const backend = createPiModelBackend('openai-codex', 'gpt-6-luna',
    { NATLANG_FAKE_CODEX_KEY: `a.${claim}.z` }, 'NATLANG_FAKE_CODEX_KEY', undefined,
    { transport: 'sse', maxRetries: 0,
      fetch: async (_url, init) => {
        const compressed = new Headers(init.headers).get('content-encoding') === 'zstd';
        sent = JSON.parse(Buffer.from(compressed ? zstdDecompressSync(init.body) : init.body).toString());
        return new Response('{"error":{"message":"intercepted"}}',
          { status: 400, headers: { 'content-type': 'application/json' } });
      } }, undefined, 'native', { max_output_tokens: 9000 });
  await assert.rejects(backend.turn({ messages: [{ role: 'user', content: 'payload probe' }],
    tools: [], seed: null, max_tokens: 4096 }), /intercepted/);
  assert.equal(sent.model, 'gpt-6-luna');
  assert.equal(sent.max_output_tokens, 4096);
});
