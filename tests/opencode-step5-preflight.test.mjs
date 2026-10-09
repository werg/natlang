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

test('Step 5 preflight binds a post-readiness bridge port without weakening model or URL checks', () => {
  const dynamicServer = 'http://127.0.0.1:{adapter-port}/v1';
  const actualServer = 'http://127.0.0.1:33183/v1';
  const pendingPlan = { ...plan,
    provider: { bootstrap_config_sha256: null, bootstrap_config_binding: 'post-readiness',
      fresh_bootstrap_config_path: 'runs/fresh/bridge/bootstrap-config.json' },
    collector: { server: dynamicServer },
    model: { ...plan.model, variant: 'low' },
    command_templates: { bridge: 'node bridge --model step-5-preview-free --variant low',
      collector: `node cli.js input jobs output --model-id ${alias} --server ${dynamicServer}` } };
  const config = { ...bootstrapConfig, main_model: alias, small_model: alias, model_variant: 'low',
    adapter_url: actualServer, official_cli: '/pinned/opencode', official_cli_sha256: 'cli-hash',
    official_sdk_module: '/pinned/sdk.mjs', official_sdk_module_sha256: 'sdk-hash' };
  const configText = JSON.stringify(config);
  const argv = ['cli.js', 'input', 'jobs', 'output', '--model-id', alias, '--server', actualServer];
  const verified = verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv });
  assert.equal(verified.ok, true);
  assert.equal(verified.bootstrap_config_sha256_bound, createHash('sha256').update(configText).digest('hex'));

  assert.throws(() => verifyStep5ModelPair({ plan: { ...pendingPlan,
    provider: { ...pendingPlan.provider, bootstrap_config_binding: undefined } }, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv }), /explicit post-readiness binding/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/other/bridge/bootstrap-config.json', collectorArgv: argv }),
  /actual bootstrap config path does not match/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: ['cli.js', '--model-id', 'opencode/paid-model', '--server', actualServer] }),
  /actual collector argv --model-id/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: configText, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: ['cli.js', '--model-id', alias, '--server', 'http://127.0.0.1:36833/v1'] }),
  /actual collector argv --server/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan,
    bootstrapConfig: { ...config, adapter_url: 'http://example.com/v1' }, bootstrapConfigText: configText,
    bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv }), /valid loopback/);
  assert.throws(() => verifyStep5ModelPair({ plan: pendingPlan,
    bootstrapConfig: { ...config, small_model: 'opencode/paid-model' }, bootstrapConfigText: configText,
    bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv }), /immutable bridge main\/small model/);
  const exactNewlineConfig = `${configText}\n`;
  const newlineBound = verifyStep5ModelPair({ plan: pendingPlan, bootstrapConfig: config,
    bootstrapConfigText: exactNewlineConfig, bootstrapConfigPath: 'runs/fresh/bridge/bootstrap-config.json', collectorArgv: argv });
  assert.equal(newlineBound.bootstrap_config_sha256_bound,
    createHash('sha256').update(exactNewlineConfig).digest('hex'));
});
