import assert from 'node:assert/strict';
import test from 'node:test';
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
