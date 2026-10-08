import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { buildFreeModelConfig } from '../scripts/opencode-cli-loopback-bootstrap.mjs';

test('effective Step 5 config pins both main and title work to the explicit free model', () => {
  const actual = buildFreeModelConfig('/test/action-server.mjs', '/test/action-calls.jsonl', '/test/mcp-handshake.jsonl');
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/opencode-step5-effective-config.json', import.meta.url), 'utf8'));
  assert.deepEqual(actual, fixture);
  assert.equal(actual.model, actual.small_model);
  assert.equal(actual.provider['zen-step5-free'].options.baseURL, 'https://opencode.ai/zen/v1');
  assert.equal(actual.provider['zen-step5-free'].options.apiKey, '{env:OPENCODE_API_KEY}');
  assert.equal(Object.hasOwn(actual.provider['zen-step5-free'].options, 'headers'), false);
  assert.deepEqual(actual.permission, { '*': 'deny', natlang_action_bridge_submit_action: 'allow' });
  assert.equal(actual.permission['*'], 'deny');
  assert.equal(actual.permission.natlang_action_bridge_submit_action, 'allow');
});

test('an alternate configured free model stays on its own provider model ID', () => {
  const actual = buildFreeModelConfig('/test/action-server.mjs', '/test/action-calls.jsonl',
    '/test/mcp-handshake.jsonl', 'ling-3.1-flash-free');
  assert.equal(actual.model, 'opencode/ling-3.1-flash-free');
  assert.equal(actual.small_model, 'opencode/ling-3.1-flash-free');
  assert.deepEqual(actual.permission, { '*': 'deny', natlang_action_bridge_submit_action: 'allow' });
  assert.equal(actual.provider, undefined);
});
