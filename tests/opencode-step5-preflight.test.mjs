import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { verifyStep5ModelPair } from '../scripts/opencode-step5-preflight.mjs';

const alias = 'opencode/step-5-preview-free';
const bootstrapConfig = { model_alias: alias, main_model: alias, small_model: alias };
const plan = { model: { main_model: alias, small_model: alias, collector_model_id: alias },
  command_templates: { collector: `node cli.js input jobs output --model-id ${alias} --workers 1` } };

test('Step 5 preflight accepts an argv matching the immutable official bridge model', () => {
  assert.deepEqual(verifyStep5ModelPair({ plan, bootstrapConfig,
    collectorArgv: ['cli.js', 'input', 'jobs', 'output', '--model-id', alias, '--workers', '1'] }),
  { ok: true, model_alias: alias, collector_model_id: alias });
});

test('Step 5 preflight rejects a stale custom provider alias before collector launch', () => {
  assert.throws(() => verifyStep5ModelPair({ plan, bootstrapConfig,
    collectorArgv: ['cli.js', '--model-id', 'zen-step5-free/step-5-preview-free'] }),
  /actual collector argv --model-id does not match/);
});

test('Step 5 preflight rejects plan and bridge main/small model mismatches', () => {
  assert.throws(() => verifyStep5ModelPair({ plan: { ...plan,
    model: { ...plan.model, small_model: 'opencode/another-free-model' } }, bootstrapConfig,
  collectorArgv: ['cli.js', '--model-id', alias] }), /plan small_model does not match/);
  assert.throws(() => verifyStep5ModelPair({ plan, bootstrapConfig: { ...bootstrapConfig,
    small_model: 'opencode/another-free-model' }, collectorArgv: ['cli.js', '--model-id', alias] }),
  /immutable bridge main\/small model/);
});

test('Step 5 preflight pins the catalog-supported reasoning variant independently of model ID', () => {
  const lowPlan = { ...plan, model: { ...plan.model, variant: 'low' },
    command_templates: { ...plan.command_templates, bridge: `node bridge --model step-5-preview-free --variant low` } };
  const lowBootstrap = { ...bootstrapConfig, model_variant: 'low' };
  assert.deepEqual(verifyStep5ModelPair({ plan: lowPlan, bootstrapConfig: lowBootstrap,
    collectorArgv: ['cli.js', '--model-id', alias] }),
  { ok: true, model_alias: alias, collector_model_id: alias });
  assert.throws(() => verifyStep5ModelPair({ plan: lowPlan, bootstrapConfig,
    collectorArgv: ['cli.js', '--model-id', alias] }), /plan model variant does not match/);
  assert.throws(() => verifyStep5ModelPair({ plan: lowPlan,
    bootstrapConfig: { ...lowBootstrap, model_variant: 'high' }, collectorArgv: ['cli.js', '--model-id', alias] }),
  /plan model variant does not match/);
});

test('Step 5 preflight binds and verifies the effective Natlang-only agent tool surface', () => {
  const surfacePlan = { ...plan, provider: { tool_surface_mode: 'natlang-only' },
    model: { ...plan.model, variant: 'low' },
    command_templates: { ...plan.command_templates,
      bridge: 'node scripts/opencode-cli-loopback-bwrap-launch.mjs --model step-5-preview-free --variant low --tool-surface natlang-only' } };
  const surfaceConfig = { ...bootstrapConfig, model_variant: 'low', tool_surface_mode: 'natlang-only', cli_agent: 'build', mcp_status: 'connected',
    effective_agent_tool_surface: { agent: 'build', tools: { '*': false, natlang_action_bridge_submit_action: true } } };
  assert.equal(verifyStep5ModelPair({ plan: surfacePlan, bootstrapConfig: surfaceConfig,
    collectorArgv: ['cli.js', '--model-id', alias] }).tool_surface_mode, 'natlang-only');
  assert.throws(() => verifyStep5ModelPair({ plan: surfacePlan, bootstrapConfig: { ...bootstrapConfig, model_variant: 'low' },
    collectorArgv: ['cli.js', '--model-id', alias] }), /bootstrap tool surface/);
  assert.throws(() => verifyStep5ModelPair({ plan: surfacePlan,
    bootstrapConfig: { ...surfaceConfig, effective_agent_tool_surface: { agent: 'build',
      tools: { '*': false, natlang_action_bridge_submit_action: true, bash: true } } },
    collectorArgv: ['cli.js', '--model-id', alias] }), /do not match the Natlang-only/);
});

test('Step 5 preflight verifies exact bridge receipt bytes and official client pins', () => {
  const bridgeText = JSON.stringify({ model_alias: alias, main_model: alias, small_model: alias,
    official_cli: '/pinned/opencode', official_cli_sha256: 'cli-hash',
    official_sdk_module: '/pinned/sdk.mjs', official_sdk_module_sha256: 'sdk-hash' });
  const pinnedPlan = { ...plan, provider: { bootstrap_config_sha256: createHash('sha256').update(bridgeText).digest('hex'),
    official_cli: '/pinned/opencode', official_cli_sha256: 'cli-hash',
    official_sdk_module: '/pinned/sdk.mjs', official_sdk_module_sha256: 'sdk-hash' } };
  const pinnedConfig = JSON.parse(bridgeText);
  assert.equal(verifyStep5ModelPair({ plan: pinnedPlan, bootstrapConfig: pinnedConfig,
    bootstrapConfigText: bridgeText, collectorArgv: ['cli.js', '--model-id', alias] }).ok, true);
  assert.throws(() => verifyStep5ModelPair({ plan: pinnedPlan, bootstrapConfig: pinnedConfig,
    bootstrapConfigText: `${bridgeText}\n`, collectorArgv: ['cli.js', '--model-id', alias] }), /bytes do not match/);
  assert.throws(() => verifyStep5ModelPair({ plan: { ...pinnedPlan,
    provider: { ...pinnedPlan.provider, official_cli_sha256: 'wrong' } }, bootstrapConfig: pinnedConfig,
    bootstrapConfigText: bridgeText, collectorArgv: ['cli.js', '--model-id', alias] }), /official_cli_sha256/);
});
