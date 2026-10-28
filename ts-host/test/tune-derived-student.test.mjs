// Zero-config tuning: without a tune block, the native profile natlang runs is the student and it teaches itself
// (privileged self-teaching, the default teacher since the owner's 2026-10-28 no-teacher-generation decision).
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { deriveStudent, loadTuneSettings } from '../dist/tune/settings.js';

const sha = 'ab'.repeat(32);
const identity = { architecture: 'lfm2', checkpoint_sha256: sha, embedding_weight_sha256: 'cd'.repeat(32), tokenizer_fingerprint: 'ef'.repeat(32) };

test('the default native profile (or the native pin behind a GGUF default) is the student and its own privileged teacher', () => {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-tune-derived-')), models = join(directory, 'models');
  const environment = { ...process.env, NATLANG_CONFIG_HOME: directory, NATLANG_NATIVE_MODELS: models, NATLANG_PROFILE: '' };
  delete environment.NATLANG_PROFILE;
  const write = config => writeFileSync(join(directory, 'config.json'), JSON.stringify(config));
  const native = { checkpoint: join(models, `${sha}.pt`), sha256: sha, python: '/venv/python', pythonPath: ['/repo/training', '/repo'], device: 'cuda', dtype: 'float32' };
  try {
    mkdirSync(models);
    write({ defaultProfile: 'lfm-latest-gguf', profiles: {
      'lfm-latest': { native, escalate: 'teacher' }, teacher: { endpoint: 'http://127.0.0.1:9', model: 'qwen' },
      'lfm-latest-gguf': { local: { modelPath: '/x.gguf', modelId: `native-gguf-${sha.slice(0, 12)}-q8_0` } } } });
    assert.equal(deriveStudent(environment).profile, 'lfm-latest');
    assert.equal(deriveStudent(environment).student, undefined, 'no identity before its server has reported one');
    assert.throws(() => loadTuneSettings(environment), /base identity of native profile lfm-latest is not known/);
    writeFileSync(join(models, 'pins.json'), JSON.stringify({ [sha]: { checkpoint: native.checkpoint, sha256: sha, identity } }));
    const { settings, student } = loadTuneSettings(environment);
    assert.equal(student.name, 'lfm-latest');
    assert.deepEqual(student.base, identity);
    assert.equal(student.checkpoint, native.checkpoint);
    assert.deepEqual(student.trainer, ['/venv/python', '-m', 'natlang_neuralese.tune.build']);
    assert.deepEqual(student.env, { PYTHONPATH: '/repo/training:/repo' });
    assert.deepEqual(student.engine, { device: 'cuda', dtype: 'float32' });
    assert.equal(student.moduleMap, 'lfm2');
    assert.equal(settings.backend, 'native');
    assert.equal(settings.servingProfile, 'lfm-latest');
    assert.equal(settings.teacher, 'self', 'a profile escalate is not a tune teacher: self-teaching is the default');
    assert.equal(student.baseGgufSha256, undefined, 'no llama.cpp base without an export manifest');
    // The GGUF sibling's export manifest gives the llama.cpp base: tuned adapters then also export for llama.cpp.
    const exportDir = join(models, `gguf-${sha.slice(0, 12)}`);
    mkdirSync(join(exportDir, 'hf-config'), { recursive: true });
    writeFileSync(join(exportDir, 'manifest.json'), JSON.stringify({ checkpoint_sha256: sha, hf_config: 'hf-config',
      files: [{ outtype: 'q8_0', file: 'model-q8_0.gguf', sha256: '12'.repeat(32) }] }));
    write({ defaultProfile: 'lfm-latest-gguf', profiles: {
      'lfm-latest': { native, escalate: 'teacher' }, teacher: { endpoint: 'http://127.0.0.1:9', model: 'qwen' },
      'lfm-latest-gguf': { local: { modelPath: join(exportDir, 'model-q8_0.gguf'), modelId: `native-gguf-${sha.slice(0, 12)}-q8_0` } } } });
    const withGguf = loadTuneSettings(environment).student;
    assert.equal(withGguf.baseGgufSha256, '12'.repeat(32));
    assert.equal(withGguf.hfConfigDir, join(exportDir, 'hf-config'));
    // A partial tune block still applies; an explicit teacher wins over escalate.
    write({ defaultProfile: 'lfm-latest', tune: { recipe: { epochs: 2 }, teacher: 'other' }, profiles: {
      'lfm-latest': { native, escalate: 'teacher' }, teacher: { endpoint: 'http://127.0.0.1:9' }, other: { endpoint: 'http://127.0.0.1:9' } } });
    const partial = loadTuneSettings(environment).settings;
    assert.equal(partial.teacher, 'other');
    assert.deepEqual(partial.recipe, { epochs: 2 });
    // No teacher anywhere: the student teaches itself; a named teacher profile must still exist.
    write({ defaultProfile: 'lfm-latest', profiles: { 'lfm-latest': { native } } });
    assert.equal(loadTuneSettings(environment).settings.teacher, 'self');
    write({ defaultProfile: 'lfm-latest', tune: { teacher: 'missing' }, profiles: { 'lfm-latest': { native } } });
    assert.throws(() => loadTuneSettings(environment), /teacher profile missing does not exist/);
    // A configured student is used as configured.
    write({ defaultProfile: 'lfm-latest', profiles: { 'lfm-latest': { native }, t: { endpoint: 'http://127.0.0.1:9' } },
      tune: { student: 's', teacher: 't', backend: 'native', students: { s: { base: identity, checkpoint: '/c.pt', trainer: ['py'], moduleMap: 'lfm2' } } } });
    const configured = loadTuneSettings(environment);
    assert.equal(configured.student.name, 's');
    assert.equal(configured.settings.servingProfile, undefined);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
