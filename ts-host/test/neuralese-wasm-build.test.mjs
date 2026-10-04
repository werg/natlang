import assert from 'node:assert/strict';
import test from 'node:test';
import { chooseNeuraleseBuild } from '../dist/browser/neuralese-wasm.js';

const gpu = features => ({ requestAdapter: async () => ({ features: new Set(features) }) });

test('chooseNeuraleseBuild: WebGPU only with shader-f16, else threads when isolated, else one thread', async () => {
  assert.equal((await chooseNeuraleseBuild({ navigator: { gpu: gpu(['shader-f16']), hardwareConcurrency: 16 }, crossOriginIsolated: true })).build,
    'neuralese-wasm-gpu');
  const noF16 = await chooseNeuraleseBuild({ navigator: { gpu: gpu([]), hardwareConcurrency: 16 }, crossOriginIsolated: true });
  assert.deepEqual([noF16.build, noF16.threads, noF16.gpuLayers], ['neuralese-wasm-mt', 8, 0]);
  assert.equal((await chooseNeuraleseBuild({ navigator: { gpu: gpu(['shader-f16']) }, crossOriginIsolated: false }, { gpu: false })).build,
    'neuralese-wasm');
  const failing = { requestAdapter: async () => { throw new Error('no'); } };
  assert.equal((await chooseNeuraleseBuild({ navigator: { gpu: failing, hardwareConcurrency: 4 }, crossOriginIsolated: true })).threads, 4);
});
