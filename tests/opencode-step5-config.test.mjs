import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { buildFreeModelConfig, buildToolSurfaceReceipt } from '../scripts/opencode-cli-loopback-bootstrap.mjs';

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

test('bootstrap receipt separates installed tools from the denied model-facing surface', () => {
  assert.deepEqual(buildToolSurfaceReceipt(['bash', 'read', 'natlang_action_bridge_submit_action']), {
    installed_tool_ids: ['bash', 'read', 'natlang_action_bridge_submit_action'],
    model_tool_surface: {
      native_tools: 'disabled and omitted from model-facing requests by wildcard deny',
      natlang_action_bridge_submit_action: 'allowed; the only enabled action tool'
    },
    permission_policy: 'official wildcard deny disables and hides native built-in tools; exact Natlang action MCP tool allowed; permission requests outside this tool surface are rejected; session history is not an execution barrier'
  });
});
