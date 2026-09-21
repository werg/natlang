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
      const write = request.tools.find(tool => tool.function.name.startsWith('write_alt_') &&
        (tool.function.parameters.properties.path.const === 'return' ||
          tool.function.parameters.properties.path.enum?.includes('return')) &&
        tool.function.parameters.properties.type?.const === 'Num');
      return requests.length === 1 ? { choices: [{ finish_reason: 'tool_calls', message: {
        content: null, tool_calls: [{ id: 'local_1', type: 'function', function: {
          name: write.function.name, arguments: '{"path":"return","type":"Num","value":7}',
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
    assert.ok(requests[0].tools.some(tool => tool.function.name.startsWith('write_alt_')));
    assert.equal(requests[0].tools.some(tool => tool.function.parameters['x-natlang-alternatives']), false);
    assert.equal(requests[0].cache_prompt, true);
    assert.equal(requests[1].messages.at(-1).role, 'tool');
    assert.equal(requests[1].messages.at(-1).tool_call_id, 'local_1');
    assert.deepEqual(requests[1].messages.at(-2).tool_calls[0].function.arguments,
      { path: 'return', type: 'Num', value: 7 });
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
      { contextTokens: 2048, gpuLayers: 0, chatTemplate: 'official template' });
    assert.equal(params.n_ctx, 2048);
    assert.equal(params.n_gpu_layers, 0);
    assert.equal(params.jinja, true);
    assert.equal(params.chat_template, 'official template');
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
  const model = new BrowserLocalModel({ engine: fake, gpuProbe: async () => ({
    apiAvailable: true, adapterAvailable: true, shaderF16: true,
    browser: 'chromium', usable: true, reason: 'WebGPU adapter supports shader-f16' }) });
  try {
    await model.loadFromHuggingFace({ repo: 'example/model' });
    assert.equal(params.n_gpu_layers, 99999);
    await assert.rejects(() => model.loadFromHuggingFace({ repo: 'example/model' },
      { gpuLayers: -1 }), /gpuLayers must be a nonnegative integer/);
  } finally { await model.close(); }
});

test('browser GPU selection checks adapter features and keeps CPU fallback', async () => {
  const { BrowserLocalModel, probeBrowserGpu } = await browserApi();
  const capable = { requestAdapter: async () => ({ features: new Set(['shader-f16']) }) };
  const unsupported = { requestAdapter: async () => ({ features: new Set() }) };
  assert.equal((await probeBrowserGpu({ gpu: capable, userAgent: 'Chrome/140' })).usable, true);
  assert.match((await probeBrowserGpu({ gpu: unsupported, userAgent: 'Chrome/140' })).reason,
    /lacks shader-f16/);
  assert.equal((await probeBrowserGpu({ gpu: capable, userAgent: 'Firefox/140' })).usable, false);
  assert.equal((await probeBrowserGpu({ gpu: capable, userAgent: 'Firefox/140',
    firefoxCompatibility: true })).usable, true);
  let params;
  const fake = { isSupportWebGPU: () => true, isModelLoaded: () => true,
    async loadModelFromHF(_model, received) { params = received; },
    async loadModelFromUrl() {}, async loadModel() {}, async exit() {} };
  const model = new BrowserLocalModel({ engine: fake, gpuProbe: () => probeBrowserGpu({
    gpu: unsupported, userAgent: 'Chrome/140' }) });
  try {
    await model.loadFromHuggingFace({ repo: 'example/model' });
    assert.equal(params.n_gpu_layers, 0);
    assert.match(model.diagnostics.gpuSelectionReason, /Automatic CPU fallback/);
    await assert.rejects(() => model.loadFromHuggingFace({ repo: 'example/model' },
      { gpuLayers: 3 }), /Cannot request GPU layers/);
  } finally { await model.close(); }
});

test('typed browser tools preserve destination and source constraints while compacting equal values', async () => {
  const { compileBrowserTools } = await browserApi();
  const tools = [{ type: 'function', function: { name: 'write', parameters: {
    type: 'object', 'x-natlang-alternatives': [
      { path: { const: 'return/a' }, type: { const: 'Num' }, value: { type: 'number' } },
      { path: { const: 'return/b' }, type: { const: 'Num' }, value: { type: 'number' } },
      { path: { const: 'return/c' }, type: { const: 'Num' }, source: { enum: ['args/a'] } },
    ], properties: { path: { type: 'string' } }, required: ['path'] } } }];
  const typed = compileBrowserTools(tools);
  assert.equal(typed.tools.length, 2);
  assert.deepEqual(typed.tools[0].function.parameters.properties.path.enum, ['return/a', 'return/b']);
  assert.deepEqual(typed.tools[1].function.parameters.properties.source.enum, ['args/a']);
  assert.equal(typed.names.get('write_alt_0'), 'write');
  assert.equal(typed.tools[0].function.parameters['x-natlang-alternatives'], undefined);
  const broad = compileBrowserTools(tools, 'broad');
  assert.equal(broad.tools.length, 1);
  assert.equal(broad.tools[0].function.name, 'write');
});

test('browser model records token use and retries one malformed local tool call', async () => {
  const { BrowserLocalModel } = await browserApi();
  let attempts = 0;
  const fake = { isSupportWebGPU: () => false, isModelLoaded: () => true,
    async loadModelFromHF() {}, async loadModelFromUrl() {}, async loadModel() {}, async exit() {},
    async createChatCompletion(request) {
      attempts++;
      assert.equal(request.cache_prompt, true);
      return { choices: [{ finish_reason: 'tool_calls', message: { tool_calls: [{ id: 'raw_1',
        type: 'function', function: { name: 'write', arguments: attempts === 1 ? '{bad' : '{"path":"return","value":7}' } }] } }],
      usage: { prompt_tokens: 20, completion_tokens: 4, prompt_tokens_details: { cached_tokens: attempts === 1 ? 0 : 10 } } };
    } };
  const model = new BrowserLocalModel({ engine: fake, schemaMode: 'broad' });
  try {
    const turn = await model.turn({ messages: [{ role: 'user', content: 'write seven' }],
      tools: [{ type: 'function', function: { name: 'write', parameters: { type: 'object' } } }],
      temperature: 0, seed: null, max_tokens: 50 });
    assert.equal(attempts, 2);
    assert.equal(turn.prompt_tokens, 40);
    assert.equal(turn.completion_tokens, 8);
    assert.equal(model.lastTurn.cachedTokens, 10);
    assert.equal(model.lastTurn.retries, 1);
    assert.equal(turn.raw_calls[0].id, 'raw_1');
  } finally { await model.close(); }
});

test('natlang model manifest and storage headroom are explicit', async () => {
  const { BROWSER_MODEL_CATALOG, checkModelStorage } = await browserApi();
  const spec = BROWSER_MODEL_CATALOG[0];
  assert.ok(spec.url.startsWith('/models/'));
  assert.ok(spec.url.endsWith(spec.file));
  assert.ok(spec.templateUrl.endsWith('.jinja'));
  assert.equal(spec.sha256.length, 64);
  const status = await checkModelStorage(spec, { async estimate() { return { usage: 100, quota: spec.bytes }; },
    async persisted() { return false; } });
  assert.equal(status.enough, false);
  assert.equal(status.persisted, false);
  assert.ok(status.recommendedFree > spec.bytes);
});
