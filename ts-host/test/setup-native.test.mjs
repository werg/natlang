// `natlang setup --native`: pin, actor view, GGUF sibling, escalation, and a refresh to a newer checkpoint.
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';

const bin = new URL('../bin/natlang.mjs', import.meta.url).pathname;

test('setup pins with an actor view, writes escalation to both profiles and keeps the GGUF sibling current', () => {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-setup-native-')), models = join(directory, 'models');
  const python = join(directory, 'python');
  // Stands in for the training interpreter: pin copy, actor view receipt, GGUF export manifest.
  writeFileSync(python, `#!/bin/sh
if [ "$1" = -c ]; then cp "$3" "$4"; sha=$(sha256sum "$3" | cut -c1-64); echo "{\\"sha256\\":\\"$sha\\",\\"bytes\\":$(stat -c %s "$3"),\\"source_sha256\\":\\"$sha\\"}"; exit 0; fi
if [ "$2" = natlang_neuralese.model.native_bundle ]; then mkdir -p "$7"; echo r > "$7/r.json"; echo "{\\"path\\":\\"$7/r.json\\",\\"sha256\\":\\"${'e'.repeat(64)}\\",\\"bytes\\":2}"; exit 0; fi
if [ "$2" = natlang_neuralese.export.native_gguf ]; then mkdir -p "$4"; echo gguf > "$4/model-$6.gguf"; echo t > "$4/template.jinja"
  mkdir -p "$4/hf-config"; echo "{\\"hf_config\\":\\"hf-config\\",\\"checkpoint_sha256\\":\\"$(sha256sum "$3" | cut -c1-64)\\",\\"files\\":[{\\"outtype\\":\\"$6\\",\\"file\\":\\"model-$6.gguf\\",\\"sha256\\":\\"${'f'.repeat(64)}\\"}],\\"serving\\":{\\"chat_template\\":\\"template.jinja\\",\\"chat_template_kwargs\\":{},\\"logit_bias\\":{}}}" > "$4/manifest.json"; exit 0; fi
exit 3
`);
  chmodSync(python, 0o755);
  const environment = { ...process.env, NATLANG_CONFIG_HOME: directory, NATLANG_NATIVE_MODELS: models, NATLANG_NATIVE_PYTHON: python };
  const run = (...argv) => spawnSync(process.execPath, [bin, 'setup', ...argv, '--json'], { encoding: 'utf8', env: environment });
  const config = () => JSON.parse(readFileSync(join(directory, 'config.json'), 'utf8'));
  try {
    mkdirSync(models);
    writeFileSync(join(directory, 'config.json'), JSON.stringify({ profiles: { teacher: { endpoint: 'http://127.0.0.1:9', model: 'q' } } }));
    const pointer = join(directory, 'current.json'), first = join(directory, 'step1.pt'), second = join(directory, 'step2.pt');
    writeFileSync(first, 'weights one'); writeFileSync(second, 'weights two');
    writeFileSync(pointer, JSON.stringify({ global_step: 1, local_full_state: { path: first } }));
    const made = run('--native', pointer, '--profile', 'lfm', '--gguf', 'q8_0', '--escalate', 'teacher', '--default');
    assert.equal(made.status, 0, made.stderr);
    const report = JSON.parse(made.stdout);
    assert.ok(report.actorView?.path.endsWith('r.json'));
    let profiles = config().profiles;
    assert.equal(config().defaultProfile, 'lfm-gguf');
    assert.equal(profiles.lfm.escalate, 'teacher');
    assert.equal(profiles['lfm-gguf'].escalate, 'teacher');
    assert.deepEqual(profiles.lfm.native.actorView, report.actorView);
    assert.match(profiles['lfm-gguf'].local.modelId, /^native-gguf-[0-9a-f]{12}-q8_0$/);
    assert.equal(run('--native', pointer, '--profile', 'lfm', '--escalate', 'nobody').status, 1, 'an unknown teacher profile is refused');
    // A newer checkpoint: the GGUF sibling is exported again in its format, escalation is kept, the old pin goes.
    writeFileSync(pointer, JSON.stringify({ global_step: 2, local_full_state: { path: second } }));
    const refreshed = run('--native', pointer, '--profile', 'lfm');
    assert.equal(refreshed.status, 0, refreshed.stderr);
    const after = JSON.parse(refreshed.stdout);
    profiles = config().profiles;
    assert.equal(profiles.lfm.native.step, 2);
    assert.equal(after.ggufProfile, 'lfm-gguf');
    assert.ok(existsSync(profiles['lfm-gguf'].local.modelPath), 'the GGUF profile points at the new export');
    assert.ok(profiles['lfm-gguf'].local.modelId.includes(profiles.lfm.native.sha256.slice(0, 12)));
    assert.equal(profiles['lfm-gguf'].escalate, 'teacher');
    assert.ok(after.removedPin && !existsSync(after.removedPin));
    assert.equal(run('--native', pointer, '--profile', 'lfm', '--escalate', 'none').status, 0);
    assert.equal(config().profiles.lfm.escalate, undefined);
    assert.equal(config().profiles['lfm-gguf'].escalate, undefined);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
