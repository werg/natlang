import assert from 'node:assert/strict';
import { test } from 'node:test';
import { zstdDecompressSync } from 'node:zlib';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createPiModelBackend } from '../dist/model/pi-provider.js';

test('Codex Responses receives the collector action output budget', async t => {
  const claim = Buffer.from(JSON.stringify({ 'https://api.openai.com/auth':
    { chatgpt_account_id: 'local-payload-probe' } })).toString('base64url');
  const directory = mkdtempSync(join(tmpdir(), 'natlang-codex-budget-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  // Codex is OAuth-only: an API-key option cannot replace its credential store.
  // Keep the request mock independent of the host's real account and tokens.
  mkdirSync(join(directory, 'credentials'));
  writeFileSync(join(directory, 'credentials', 'openai-codex.json'), JSON.stringify({
    type: 'oauth', access: `a.${claim}.z`, refresh: 'unused-fixture-refresh',
    expires: Date.now() + 3600000,
  }), { mode: 0o600 });
  let sent;
  const backend = createPiModelBackend('openai-codex', 'gpt-6-luna',
    { NATLANG_CONFIG_HOME: directory, NATLANG_FAKE_CODEX_KEY: `a.${claim}.z` }, 'NATLANG_FAKE_CODEX_KEY', undefined,
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
