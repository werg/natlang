// A release tuned natively with GGUF exports runs on the llama.cpp profile whose base GGUF it was exported for.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { resolveModelChoice } from '../dist/model/config.js';
import { llamaServesRelease } from '../dist/tuned-serving.js';

test('only the profile GGUF every adapter was exported for serves the release on llama.cpp', async () => {
  const directory = mkdtempSync(join(tmpdir(), 'natlang-tuned-llama-'));
  try {
    const gguf = join(directory, 'model-q8_0.gguf'), other = join(directory, 'other.gguf');
    writeFileSync(gguf, 'GGUF base'); writeFileSync(other, 'GGUF other');
    const sha = createHash('sha256').update('GGUF base').digest('hex');
    const environment = { ...process.env, NATLANG_CACHE_HOME: directory, XDG_CACHE_HOME: directory };
    delete environment.NATLANG_MODEL_PATH;
    const exported = base => ({ exports: { gguf: { file: 'x.gguf', sha256: 'a'.repeat(64), baseGgufSha256: base } } });
    const manifest = { artifacts: [exported(sha), exported(sha)], families: [{}, exported(sha)], softStates: [] };
    const choice = resolveModelChoice({ local: { modelPath: gguf, modelId: 'native-gguf-570cbbae401b-q8_0' } });
    assert.equal(await llamaServesRelease(manifest, choice, environment), true);
    assert.equal(await llamaServesRelease(manifest, resolveModelChoice({ local: { modelPath: other } }), environment), false, 'another base GGUF');
    assert.equal(await llamaServesRelease({ ...manifest, artifacts: [exported(sha), {}] }, choice, environment), false, 'an adapter without export');
    assert.equal(await llamaServesRelease(manifest, resolveModelChoice({ endpoint: 'http://127.0.0.1:9' }), environment), false, 'not llama.cpp');
    // Soft states need the fork's native-state import.
    const states = { ...manifest, softStates: [exported(sha)] };
    assert.equal(await llamaServesRelease(states, choice, { ...environment, NATLANG_LLAMA_SERVER: join(directory, 'missing-server') }), false);
  } finally { rmSync(directory, { recursive: true, force: true }); }
});
