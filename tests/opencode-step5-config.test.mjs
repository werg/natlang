import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { buildFreeModelConfig, buildToolSurfaceReceipt } from '../scripts/opencode-cli-loopback-bootstrap.mjs';

test('effective Step 5 config pins both main and title work to the explicit free model', () => {
  const actual = buildFreeModelConfig('/test/action-server.mjs', '/test/action-calls.jsonl', '/test/mcp-handshake.jsonl');
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/opencode-step5-effective-config.json', import.meta.url), 'utf8'));
  assert.deepEqual(actual, fixture);
  assert.equal(actual.model, actual.small_model);
  assert.equal(actual.model, 'opencode/step-5-preview-free');
  assert.equal(actual.provider, undefined);
  assert.deepEqual(actual.permission, { '*': 'ask', natlang_action_bridge_submit_action: 'allow' });
  assert.equal(actual.permission['*'], 'ask');
  assert.equal(actual.permission.natlang_action_bridge_submit_action, 'allow');
});

test('an alternate configured free model stays on its own provider model ID', () => {
  const actual = buildFreeModelConfig('/test/action-server.mjs', '/test/action-calls.jsonl',
    '/test/mcp-handshake.jsonl', 'ling-3.1-flash-free');
  assert.equal(actual.model, 'opencode/ling-3.1-flash-free');
  assert.equal(actual.small_model, 'opencode/ling-3.1-flash-free');
  assert.deepEqual(actual.permission, { '*': 'ask', natlang_action_bridge_submit_action: 'allow' });
  assert.equal(actual.provider, undefined);
});

test('bootstrap receipt distinguishes visible but permission-gated tools from the Natlang action', () => {
  assert.deepEqual(buildToolSurfaceReceipt(['bash', 'read', 'natlang_action_bridge_submit_action']), {
    installed_tool_ids: ['bash', 'read', 'natlang_action_bridge_submit_action'],
    model_tool_surface: {
      native_tools: 'official schemas remain model-visible; all require permission and are auto-rejected by the adapter',
      natlang_action_bridge_submit_action: 'allowed; the only enabled action tool'
    },
    permission_policy: 'official wildcard ask keeps native schemas visible but gated; adapter rejects every native tool permission; exact Natlang action MCP tool allowed; session history is not an execution barrier'
  });
});
