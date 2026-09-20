import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { test } from 'node:test';

async function browserApi() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

test('browser local inference drives the native tool loop without a server', async () => {
  const { BrowserLocalModel, BrowserNatlangHost } = await browserApi();
  const requests = [];
  const fake = {
    isSupportWebGPU: () => true,
    isModelLoaded: () => true,
    async loadModelFromHF() {}, async loadModelFromUrl() {}, async loadModel() {}, async exit() {},
    async createChatCompletion(request) {
      requests.push(request);
      return requests.length === 1 ? { choices: [{ finish_reason: 'tool_calls', message: {
        content: null, tool_calls: [{ id: 'local_1', type: 'function', function: {
          name: 'write', arguments: '{"path":"return","type":"Num","value":7}',
        } }],
      } }], usage: { completion_tokens: 9 } } :
        { choices: [{ finish_reason: 'stop', message: { content: 'finished' } }],
          usage: { completion_tokens: 2 } };
    },
  };
  const model = new BrowserLocalModel({ engine: fake });
  assert.equal(model.supportsWebGPU, true);
  const host = new BrowserNatlangHost({ model });
  try {
    const result = await host.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', instructions: 'Write seven.' } } },
    options: { seed: { mode: 'compatibility' } } });
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value, 7);
    assert.equal(requests.length, 2);
    assert.equal(requests[0].seed, 0);
    assert.equal(requests[0].max_tokens, undefined);
    assert.equal(requests[0].tool_choice, 'auto');
    assert.equal(requests[0].tools.find(tool => tool.function.name === 'write').function.parameters['x-natlang-alternatives'], undefined);
    assert.equal(requests[1].messages.at(-1).role, 'tool');
    assert.equal(requests[1].messages.at(-1).tool_call_id, 'call_1_0');
  } finally { host.close(); await model.close(); }
});

test('local model validates calls and exposes local loading options', async () => {
  const { BrowserLocalModel } = await browserApi();
  let loaded = false, params;
  const fake = { isSupportWebGPU: () => false, isModelLoaded: () => loaded,
    async loadModelFromHF(_model, received) { params = received; loaded = true; },
    async loadModelFromUrl() {}, async loadModel() {}, async exit() {},
    async createChatCompletion() { return { choices: [{ finish_reason: 'tool_calls', message: {
      tool_calls: [{ type: 'function', function: { name: 'write', arguments: 'not-json' } }],
    } }], usage: { completion_tokens: 1 } }; } };
  const model = new BrowserLocalModel({ engine: fake });
  try {
    await assert.rejects(() => model.turn({ messages: [], tools: [], temperature: 0, seed: 0,
      max_tokens: 10 }), /load a local GGUF model/);
    await model.loadFromHuggingFace({ repo: 'example/model', quant: 'Q4_K_M' },
      { contextTokens: 2048, gpuLayers: 0 });
    assert.equal(params.n_ctx, 2048);
    assert.equal(params.n_gpu_layers, 0);
    assert.equal(params.jinja, true);
    await assert.rejects(() => model.turn({ messages: [{ role: 'user', content: 'x' }], tools: [],
      temperature: 0, seed: null, max_tokens: 10 }), /invalid JSON arguments/);
  } finally { await model.close(); }
  assert.equal(statSync(new URL('../dist/browser/wllama.wasm', import.meta.url)).size > 1_000_000, true);
  assert.equal(statSync(new URL('../dist/browser/wllama-compat.wasm', import.meta.url)).size > 1_000_000, true);
  assert.equal(statSync(new URL('../dist/browser/wllama-compat.js', import.meta.url)).size > 10_000, true);
});

test('local model requests full GPU offload by default', async () => {
  const { BrowserLocalModel } = await browserApi();
  let params;
  const fake = { isSupportWebGPU: () => true, isModelLoaded: () => true,
    async loadModelFromHF(_model, received) { params = received; },
    async loadModelFromUrl() {}, async loadModel() {}, async exit() {},
    async createChatCompletion() { return { choices: [{ message: { content: '' } }] }; } };
  const model = new BrowserLocalModel({ engine: fake });
  try {
    await model.loadFromHuggingFace({ repo: 'example/model' });
    assert.equal(params.n_gpu_layers, 99999);
    await assert.rejects(() => model.loadFromHuggingFace({ repo: 'example/model' },
      { gpuLayers: -1 }), /gpuLayers must be a nonnegative integer/);
  } finally { await model.close(); }
});
