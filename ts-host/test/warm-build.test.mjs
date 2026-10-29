import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { openingOf, pruneWarmStates, sharedPrefix } from '../dist/package/warm.js';
import { parsePackageManifest } from '../dist/package/manifest.js';
import { loadModelConfiguration } from '../dist/model/config.js';
import { progressReporter } from '../dist/cli/progress.js';

test('the shared prefix of two renderings ends just after a special token', () => {
  const render = content => `<|startoftext|><|im_start|>system\nRules<|im_end|>\n<|im_start|>user\n${content}<|im_end|>\n`;
  assert.equal(sharedPrefix(render('⁣A'), render('⁣B')), '<|startoftext|><|im_start|>system\nRules<|im_end|>\n<|im_start|>');
  assert.equal(sharedPrefix('abc', 'xyz'), '');
});

test('a request opening is its system prompt and tools, keyed by their exact content', async () => {
  const tools = [{ type: 'function', function: { name: 'eval' } }], long = 'R'.repeat(300);
  const first = openingOf({ messages: [{ role: 'system', content: long }, { role: 'user', content: '1' }], tools });
  const second = openingOf({ messages: [{ role: 'system', content: long }, { role: 'user', content: '2' }], tools });
  assert.equal(first.key, second.key);
  assert.notEqual(openingOf({ messages: [{ role: 'system', content: long }], tools: [] }).key, first.key, 'tools are part of the opening');
  assert.equal(openingOf({ messages: [{ role: 'user', content: long }] }), undefined);
  const directory = mkdtempSync(join(tmpdir(), 'natlang-warm-'));
  for (let index = 0; index < 4; index++) writeFileSync(join(directory, `${index}.slot`), 'state');
  pruneWarmStates(directory, 2);
  assert.equal(readdirSync(directory).filter(name => name.endsWith('.slot')).length, 2);
});

test('natlang.json declares a model and warm-up calls, and the declared model applies unless the command selects one', () => {
  const base = { schema: 'natlang.package/v2', name: 'p', version: '1.0.0', include: ['a.nl'] };
  const manifest = parsePackageManifest({ ...base, model: { stock: 'lfm3-1.7b-q4_k_m', contextTokens: 8192 },
    warmup: [{ function: 'a.nl', inputs: { x: 1 } }] });
  assert.deepEqual(manifest.warmup, [{ function: 'a.nl', inputs: { x: 1 } }]);
  assert.throws(() => parsePackageManifest({ ...base, warmup: [{ function: 'a.ts', inputs: {} }] }), /\.nl file/);
  assert.throws(() => parsePackageManifest({ ...base, model: { stock: 'x', provider: 'anthropic', model: 'y' } }), /local model/);
  const environment = { NATLANG_CONFIG_HOME: mkdtempSync(join(tmpdir(), 'natlang-config-')) };
  const declared = loadModelConfiguration(undefined, {}, environment, manifest.model);
  assert.equal(declared.choice.kind, 'managed-local');
  assert.equal(declared.choice.release.id, 'lfm3-1.7b-q4_k_m'); assert.equal(declared.choice.local.contextTokens, 8192);
  const overridden = loadModelConfiguration(undefined, { provider: 'anthropic', model: 'claude-opus-5-5' }, environment, manifest.model);
  assert.equal(overridden.choice.kind, 'pi-provider');
  assert.equal(loadModelConfiguration(undefined, {}, environment).choice.release.id, 'lfm3-1.7b-q8_0', 'the built-in default is stock LFM3 1.7B');
});

test('progress reports a failed call in one line and stays quiet without a terminal between heartbeats', () => {
  let written = '';
  const reporter = progressReporter({ write: text => { written += text; return true; }, isTTY: false }, 'auto');
  const call = { taskId: 't', callId: 'c1', parentCallId: null, definitionId: 'd', name: 'classify' };
  reporter.live({ type: 'call_start', ...call });
  reporter.live({ type: 'trace', ...call, event: { kind: 'model_request', phase: 'end', turn: 1, prompt_tokens: 10, completion_tokens: 2 } });
  reporter.live({ type: 'call_end', ...call, outcome: 'quiesced', detail: 'model_incomplete: repeated_failed_action (q is not defined)' });
  reporter.close();
  assert.equal(written, 'natlang: classify stopped: model_incomplete: repeated_failed_action (q is not defined)\n');
});
