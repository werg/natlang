import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { test } from 'node:test';

async function browserApi() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

test('browser local inference drives the native tool loop without a server', async () => {
  const api = await browserApi();
  const { BrowserLocalModel, createNatlangRuntime, compileVirtualProject } = api;
  const requests = [];
  const fake = {
    isSupportWebGPU: () => true,
    isModelLoaded: () => true,
    async loadModelFromHF() {}, async loadModelFromUrl() {}, async loadModel() {}, async exit() {},
    async createChatCompletion(request) {
      requests.push(request);
      const names = request.tools.map(tool => tool.function.name);
      assert.ok(names.includes('eval'));
      assert.ok(names.includes('mark_lines'));
      assert.equal(names.some(name => name.includes('_alt_')), false);
      return requests.length === 1 ? { choices: [{ finish_reason: 'tool_calls', message: {
        content: null, tool_calls: [{ id: 'local_1', type: 'function', function: {
          name: 'eval', arguments: '{"code":"7"}',
        } }],
      } }], usage: { completion_tokens: 9 } } : requests.length === 2 ?
        { choices: [{ finish_reason: 'tool_calls', message: { tool_calls: [{ id: 'local_2',
          type: 'function', function: { name: 'mark_lines', arguments: '{"start":1}' } }] } }],
          usage: { completion_tokens: 4 } } :
        { choices: [{ finish_reason: 'stop', message: { content: 'finished' } }],
          usage: { completion_tokens: 2 } };
    },
  };
  const model = new BrowserLocalModel({ engine: fake });
  assert.equal(model.supportsWebGPU, true);
  const runtime = createNatlangRuntime({ model: request => model.turn(request), seed: { mode: 'compatibility' } });
  const project = compileVirtualProject({ files: { 'main.ts':
    "import { nl } from '@natlang/browser';\nexport async function main(): Promise<number> { return await nl<number>`Write seven.`(); }\n" } }, api);
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  try {
    const value = await runtime.run(() => project.require('main.ts').main());
    assert.equal(value, 7);
    assert.equal(requests.length, 2);
    assert.equal(requests[0].seed, 0);
    assert.equal(requests[0].max_tokens, undefined);
    assert.equal(requests[0].tool_choice, 'auto');
    assert.deepEqual(requests[0].tools.map(tool => tool.function.name), ['eval', 'read_value', 'mark_lines',
      'report_blocker', 'report_error']);
    assert.equal(requests[0].cache_prompt, true);
    assert.equal(requests[1].messages.at(-1).role, 'tool');
    assert.equal(requests[1].messages.at(-1).tool_call_id, 'local_1');
    assert.deepEqual(requests[1].messages.at(-2).tool_calls[0].function.arguments, { code: '7' });
  } finally { runtime.close(); await model.close(); }
});

test('local model validates calls and exposes local loading options', async () => {
  const { BrowserLocalModel } = await browserApi();
  let loaded = false, params;
  const fake = { isSupportWebGPU: () => false, isModelLoaded: () => loaded,
    async loadModelFromHF(_model, received) { params = received; loaded = true; },
    async loadModelFromUrl() {}, async loadModel() {}, async exit() {},
    async createChatCompletion() { return { choices: [{ finish_reason: 'tool_calls', message: {
      tool_calls: [{ type: 'function', function: { name: 'eval', arguments: 'not-json' } }],
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

test('browser tools preserve scope-eval names and definitions', async () => {
  const { compileBrowserTools } = await browserApi();
  const tools = ['eval', 'read_value', 'mark_lines', 'report_blocker', 'report_error'].map(name => ({
    type: 'function', function: { name, description: `${name} description`, parameters: {
      type: 'object', properties: { code: { type: 'string', 'x-natlang': 'private-hint' } },
      required: ['code'], additionalProperties: false } } }));
  const compiled = compileBrowserTools(tools);
  assert.deepEqual(compiled.map(tool => tool.function.name), tools.map(tool => tool.function.name));
  assert.deepEqual(compiled[0].function.parameters.properties.code, { type: 'string' });
  assert.equal(compiled[0].function.description, 'eval description');
  assert.equal(compileBrowserTools([]).length, 0);
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
        type: 'function', function: { name: 'eval', arguments: attempts === 1 ? '{bad' : '{"code":"7"}' } }] } }],
      usage: { prompt_tokens: 20, completion_tokens: 4, prompt_tokens_details: { cached_tokens: attempts === 1 ? 0 : 10 } } };
    } };
  const model = new BrowserLocalModel({ engine: fake });
  try {
    const turn = await model.turn({ messages: [{ role: 'user', content: 'write seven' }],
      tools: [{ type: 'function', function: { name: 'eval', parameters: { type: 'object' } } }],
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
