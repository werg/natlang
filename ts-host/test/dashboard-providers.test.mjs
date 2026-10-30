import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startDashboard } from '../dist/dashboard/server.js';
import { recommendedProviderModel } from '../dist/model/provider-recommendations.js';

test('provider defaults are the newest Luna or Haiku in the catalog', () => {
  const ids = ['gpt-5.6-luna', 'gpt-6-luna', 'gpt-6-luna-pro', 'gpt-6-sol', 'claude-haiku-4-5', 'claude-haiku-5-5', 'claude-haiku-4-5-20251001'];
  assert.deepEqual(recommendedProviderModel('openai', ids), { luna: 'gpt-6-luna', haiku: 'claude-haiku-5-5', model: 'gpt-6-luna' });
  assert.equal(recommendedProviderModel('anthropic', ids).model, 'claude-haiku-5-5');
  assert.equal(recommendedProviderModel('openrouter', ['openai/gpt-6-luna', 'openai/gpt-6-luna:batch', '~openai/gpt-luna-latest']).model, 'openai/gpt-6-luna');
  assert.equal(recommendedProviderModel('github-copilot', ['claude-haiku-5.5', 'gpt-6.1-luna', 'gpt-6-luna']).model, 'gpt-6.1-luna');
  assert.deepEqual(recommendedProviderModel('groq', ['llama-4']), {});
});

test('dashboard connects an API-key provider through its own prompts and recommends its model', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-dashboard-providers-'));
  // No ambient provider keys: only what the page stores counts.
  const environment = { PATH: process.env.PATH, HOME: root, NATLANG_CONFIG_HOME: join(root, '.config'), NATLANG_STATE_HOME: join(root, '.state'), NATLANG_CALL_STORE: 'off' };
  const dashboard = await startDashboard({ root, environment, version: 'test', port: 0, log: { write() {} } });
  const url = new URL(dashboard.url), token = url.searchParams.get('token');
  const call = async (method, path, body) => {
    const response = await fetch(url.origin + path, { method, headers: { 'x-natlang-token': token, cookie: `natlang_dashboard=${token}`, 'content-type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) });
    const value = await response.json();
    if (!response.ok) throw new Error(`${method} ${path}: ${value.error}`);
    return value;
  };
  try {
    const providers = await call('GET', '/api/providers');
    const anthropic = providers.find(item => item.id === 'anthropic'), openai = providers.find(item => item.id === 'openai');
    assert.equal(anthropic.recommended.model, 'claude-haiku-5-5');
    assert.equal(anthropic.oauth.subscription, true);
    assert.equal(openai.recommended.model, 'gpt-6-luna');
    assert.equal(openai.connected, null);
    assert.equal(providers.find(item => item.id === 'openai-codex').recommended.model, 'gpt-6-luna');
    const listed = await call('GET', '/api/providers/anthropic/models');
    assert.equal(listed[0].id, 'claude-haiku-5-5'); assert.equal(listed[0].recommended, true);

    let login = await call('POST', '/api/providers/openai/login', { type: 'api_key' });
    for (let i = 0; i < 50 && login.state === 'running' && !login.prompt; i++) {
      await new Promise(resolveWait => setTimeout(resolveWait, 20)); login = await call('GET', `/api/logins/${login.id}`);
    }
    assert.equal(login.prompt?.type, 'secret');
    await assert.rejects(call('POST', `/api/logins/${login.id}/answer`, { prompt: login.prompt.id + 1, value: 'x' }), /no longer open/);
    login = await call('POST', `/api/logins/${login.id}/answer`, { prompt: login.prompt.id, value: ' sk-test-key ' });
    for (let i = 0; i < 50 && login.state === 'running'; i++) {
      await new Promise(resolveWait => setTimeout(resolveWait, 20)); login = await call('GET', `/api/logins/${login.id}`);
    }
    assert.equal(login.state, 'done', login.error ?? '');
    assert.doesNotMatch(JSON.stringify(login), /sk-test-key/);
    assert.equal(JSON.parse(readFileSync(join(root, '.config', 'credentials', 'openai.json'), 'utf8')).key, 'sk-test-key');
    const connected = (await call('GET', '/api/providers')).find(item => item.id === 'openai');
    assert.equal(connected.connected.type, 'api_key'); assert.equal(connected.stored, 'api_key');

    await assert.rejects(call('POST', '/api/providers/mistral/login', { type: 'oauth' }), /no subscription sign-in/);
    await call('DELETE', '/api/providers/openai/credential');
    assert.equal((await call('GET', '/api/providers')).find(item => item.id === 'openai').connected, null);
  } finally { await dashboard.close(); rmSync(root, { recursive: true, force: true }); }
});
