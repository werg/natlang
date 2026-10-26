import assert from 'node:assert/strict';
import { statSync } from 'node:fs';
import { browserTest as test } from './support/natlang.mjs';

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
      assert.ok(names.includes('return_result'));
      assert.equal(names.some(name => name.includes('_alt_')), false);
      return requests.length === 1 ? { choices: [{ finish_reason: 'tool_calls', message: {
        content: null, tool_calls: [{ id: 'local_1', type: 'function', function: {
          name: 'eval', arguments: '{"code":"return 7"}',
        } }],
      } }], usage: { completion_tokens: 9 } } :
        { choices: [{ finish_reason: 'stop', message: { content: 'done' } }],
          usage: { completion_tokens: 2 } };
    },
  };
  const model = new BrowserLocalModel({ engine: fake });
  assert.equal(model.supportsWebGPU, true);
  const runtime = createNatlangRuntime({ model: request => model.turn(request), seed: { mode: 'derived',root:0 } });
  const project = compileVirtualProject({ files: { 'main.ts':
    "import { nl } from '@natlang/browser';\nexport async function main(): Promise<number> { return await nl<number>`Write seven.`(); }\n" } }, api);
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  try {
    const value = await runtime.run(() => project.require('main.ts').main());
    assert.equal(value, 7);
    assert.equal(requests.length, 2);
    assert.ok(Number.isSafeInteger(requests[0].seed));
    assert.equal(requests[0].max_tokens, 4096, 'a turn may use a quarter of the default 16,384-token window');
    assert.equal(requests[0].tool_choice, 'auto');
    assert.deepEqual(requests[0].tools.map(tool => tool.function.name), ['eval', 'read_page', 'read_code', 'compact_history', 'return_result']);
    assert.equal(requests[0].cache_prompt, true);
    assert.equal(requests[1].messages.at(-1).role, 'tool');
    assert.equal(requests[1].messages.at(-1).tool_call_id, 'local_1');
    assert.deepEqual(requests[1].messages.at(-2).tool_calls[0].function.arguments, { code: 'return 7' });
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
      temperature: 0, seed: null, max_tokens: 10 }), /malformed tool arguments/);
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
  const tools = ['eval', 'read_page', 'return_result', 'blocked', 'failed'].map(name => ({
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

test('a tuned release loads into the in-page model and each request applies only its own call\'s stack', async () => {
  const api = await browserApi();
  const host = await import('../dist/index.js');
  const { hexDigest } = await import('../dist/native/hash.js');
  const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
  const project = api.compileVirtualProject({ files: { 'outer.nl': named('Pass value to inner.'), 'outer/inner.nl': named('Return value in uppercase.'),
    'main.ts': "import outer from './outer.nl'; export default outer;" } }, api);
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const program = project.manifest.adaptation;
  const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
  const baseGguf = 'b'.repeat(64);
  const { tree, bindings } = host.compileScopeTree(program);
  const files = new Map(tree.map((node, index) => [`adapters/${index}.gguf`, new TextEncoder().encode(`gguf ${node.id}`)]));
  const manifest = host.sealManifest({ schema: 'natlang.adapters/v1', program: { id: program.id, buildHash: program.buildHash }, base, tree, bindings,
    artifacts: tree.map((node, index) => ({ node: node.id, artifact: 'nnt1:' + node.id, rank: 2, parameterization: 'native-subspace', requires: [],
      exports: { gguf: { file: `adapters/${index}.gguf`, sha256: hexDigest(files.get(`adapters/${index}.gguf`)), baseGgufSha256: baseGguf } } })),
    families: [], memberships: [], softStates: [], scales: {}, validatedReductions: [], fallback: 'base', receipts: {} });
  files.set('natlang.adapters.json', new TextEncoder().encode(JSON.stringify(manifest)));
  const fetch = async url => { const bytes = files.get(new URL(url).pathname.replace('/release/', ''));
    return bytes ? new Response(bytes) : new Response('missing', { status: 404 }); };

  const release = await api.loadBrowserRelease('https://example.test/release', { fetch, baseGgufSha256: baseGguf, executor: base });
  assert.equal(release.adapters.length, tree.length);
  let params; const requests = [];
  const fake = { isSupportWebGPU: () => false, isModelLoaded: () => !!params, capabilities: { loraBlobs: true },
    async loadModelFromHF() {}, async loadModelFromUrl(_url, received) { params = received; }, async loadModel() {}, async exit() {},
    async createChatCompletion(request) {
      requests.push(request);
      if (request.messages.at(-1).role === 'tool') return { choices: [{ finish_reason: 'stop', message: { content: 'done' } }] };
      const outer = JSON.stringify(request.messages).includes('Pass value');
      return { choices: [{ finish_reason: 'tool_calls', message: { content: null, tool_calls: [{ id: `c${requests.length}`, type: 'function',
        function: { name: 'eval', arguments: JSON.stringify({ code: outer ? 'return await inner(value)' : 'return value.toUpperCase()' }) } }] } }] };
    } };
  const { model } = await api.loadBrowserLocalModel({ kind: 'url', url: 'https://example.test/base.gguf' },
    { engine: { engine: fake, gpuProbe: async () => ({ usable: false, reason: 'test' }) }, gpuLayers: 0, loraAdapters: release.adapters });
  assert.equal(params.lora_init_without_apply, true, 'every adapter starts off');
  assert.deepEqual(params.lora_adapters.map(item => item.scale), tree.map(() => 0));
  const runtime = api.createNatlangRuntime({ model: request => model.turn(request), weights: release.weights, seed: { mode: 'derived', root: 0 } });
  try {
    assert.equal(await runtime.run(() => project.require('main.ts').default('hi')), 'HI');
    const on = request => request.lora.filter(item => item.scale !== 0).map(item => tree[item.id].id);
    const lambda = path => tree.find(node => node.level === 'lambda' && node.path === path).id;
    const outer = requests.find(request => JSON.stringify(request.messages).includes('Pass value'));
    const inner = requests.find(request => JSON.stringify(request.messages).includes('uppercase'));
    assert.ok(requests.every(request => request.lora.length === tree.length), 'every loaded adapter is listed on every request');
    assert.deepEqual(on(outer).sort(), ['tree:', lambda('outer.nl')].sort());
    assert.deepEqual(on(inner).sort(), ['folder:outer', 'tree:', lambda('outer/inner.nl')].sort(), 'the callee runs its own stack, not the caller\'s');
    // After the task finished, a direct model call outside any task runs the base: no adapter scale leaks from it.
    await model.turn({ messages: [{ role: 'user', content: 'direct' }], tools: [], temperature: 0, seed: null, max_tokens: 4 });
    assert.deepEqual(on(requests.at(-1)), []);
  } finally { runtime.close(); await model.close(); }

  // A wllama build that cannot load LoRA blobs refuses the adapters instead of loading them as model shards.
  const stock = { ...fake, capabilities: undefined, isModelLoaded: () => false };
  await assert.rejects(api.loadBrowserLocalModel({ kind: 'url', url: 'https://example.test/base.gguf' },
    { engine: { engine: stock, gpuProbe: async () => ({ usable: false, reason: 'test' }) }, gpuLayers: 0, loraAdapters: release.adapters }),
  /weights-unavailable: this wllama build cannot load LoRA adapters/);

  // A release exported for another base file runs untuned as a whole: no adapters, no per-request scales.
  const other = await api.loadBrowserRelease('https://example.test/release/', { fetch, baseGgufSha256: 'd'.repeat(64), executor: base });
  assert.equal(other.adapters.length, 0);
  assert.equal(other.weights[0].status, 'degraded');
});

test('tuned release adapters are cached in OPFS by content hash and re-fetched when a cached file is corrupt', async () => {
  const api = await browserApi();
  const host = await import('../dist/index.js');
  const { hexDigest } = await import('../dist/native/hash.js');
  const { removeCachedModelFiles } = await import('../dist/browser/model-files.js');
  const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
  const project = api.compileVirtualProject({ files: { 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" } }, api);
  const program = project.manifest.adaptation;
  const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
  const { tree, bindings } = host.compileScopeTree(program);
  const files = new Map(tree.map((node, index) => [`adapters/${index}.gguf`, new TextEncoder().encode(`gguf ${node.id}`)]));
  const familyBytes = new TextEncoder().encode('gguf family');
  files.set('exports/family.gguf', familyBytes);
  const gguf = file => ({ file, sha256: hexDigest(files.get(file)), baseGgufSha256: 'b'.repeat(64) });
  const manifest = host.sealManifest({ schema: 'natlang.adapters/v1', program: { id: program.id, buildHash: program.buildHash }, base, tree, bindings,
    artifacts: tree.map((node, index) => ({ node: node.id, artifact: 'nnt1:' + node.id, rank: 2, parameterization: 'native-subspace', requires: [],
      exports: { gguf: gguf(`adapters/${index}.gguf`) } })),
    families: [{ family: 'shout', version: 'v1', artifact: 'nnt1:family', amplitudeHex: '0x1.0000000000000p+0', exports: { gguf: gguf('exports/family.gguf') } }],
    memberships: [], softStates: [], scales: {}, validatedReductions: [], fallback: 'base', receipts: {} });
  files.set('natlang.adapters.json', new TextEncoder().encode(JSON.stringify(manifest)));
  const fetched = [];
  const fetch = async url => { const path = new URL(url).pathname.replace('/release/', ''); fetched.push(path);
    const bytes = files.get(path); return bytes ? new Response(bytes) : new Response('missing', { status: 404 }); };
  const stored = new Map();
  const directory = {
    async getDirectoryHandle() { return directory; },
    async getFileHandle(name, options = {}) {
      if (!stored.has(name) && !options.create) throw new DOMException('not found', 'NotFoundError');
      return { async getFile() { return new File([stored.get(name) ?? new Uint8Array()], name); },
        async createWritable() { const chunks = [];
          const stream = new WritableStream({ write(chunk) { chunks.push(chunk); }, close() { stored.set(name, Buffer.concat(chunks)); } });
          stream.abort = async () => undefined; return stream; } };
    },
    async removeEntry(name) { stored.delete(name); },
  };
  const storage = { async getDirectory() { return directory; } };
  const first = await api.loadBrowserRelease('https://example.test/release', { fetch, storage, executor: base });
  assert.equal(first.adapters.length, tree.length + 1, 'family exports load with the program residuals');
  assert.equal(stored.size, tree.length + 1);
  fetched.length = 0;
  const second = await api.loadBrowserRelease('https://example.test/release', { fetch, storage, executor: base });
  assert.deepEqual(fetched, ['natlang.adapters.json'], 'a reload reads every adapter from the cache');
  assert.deepEqual(new Uint8Array(await second.adapters.at(-1).blob.arrayBuffer()), familyBytes);
  // A cached file that no longer matches its hash is fetched again and replaced.
  const key = `sha256-${hexDigest(familyBytes)}`;
  stored.set(key, new TextEncoder().encode('corrupt'));
  fetched.length = 0;
  await api.loadBrowserRelease('https://example.test/release', { fetch, storage, executor: base });
  assert.deepEqual(fetched, ['natlang.adapters.json', 'exports/family.gguf']);
  assert.deepEqual(new Uint8Array(stored.get(key)), familyBytes);
  await removeCachedModelFiles([{ url: 'x', sha256: hexDigest(familyBytes) }], { storage });
  assert.equal(stored.has(key), false);
});

test('a tuned release\'s soft states load into the in-page model as native state spans and prompts carry them as parts', async () => {
  const api = await browserApi();
  const host = await import('../dist/index.js');
  const { hexDigest } = await import('../dist/native/hash.js');
  const { nativeStateSentinel } = await import('../dist/native/neuralese.js');
  const named = body => `---\nargs: { value: string }\nreturns: string\n---\n${body}\n`;
  const project = api.compileVirtualProject({ files: { 'echo.nl': named('Return value.'), 'main.ts': "import echo from './echo.nl'; export default echo;" } }, api);
  const program = project.manifest.adaptation;
  const base = { architecture: 'lfm2', checkpoint_sha256: 'c'.repeat(64), tokenizer_fingerprint: 't', embedding_weight_sha256: 'e'.repeat(64) };
  const baseGguf = 'b'.repeat(64), state = 'nzs1_' + 'a'.repeat(52);
  const { tree, bindings } = host.compileScopeTree(program);
  const files = new Map([[`exports/${state}.state.gguf`, new TextEncoder().encode('span')]]);
  const gguf = (file, bind = baseGguf) => ({ file, sha256: hexDigest(files.get(file)), baseGgufSha256: bind });
  const binding = bindings.find(item => item.component.endsWith('echo.nl::instructions'));
  const sealed = bind => host.sealManifest({ schema: 'natlang.adapters/v1', program: { id: program.id, buildHash: program.buildHash }, base, tree, bindings,
    artifacts: [], families: [], memberships: [], scales: {}, validatedReductions: [], fallback: 'base', receipts: {},
    softStates: [{ binding: binding.id, kind: 'instructions', target: null, state, textSha256: hexDigest('Return value.\n'), tokens: 3,
      exports: { gguf: gguf(`exports/${state}.state.gguf`, bind) } }] });
  let manifest = sealed(baseGguf);
  const fetch = async url => { const path = new URL(url).pathname.replace('/release/', '');
    const bytes = path === 'natlang.adapters.json' ? new TextEncoder().encode(JSON.stringify(manifest)) : files.get(path);
    return bytes ? new Response(bytes) : new Response('missing', { status: 404 }); };
  const release = await api.loadBrowserRelease('https://example.test/release', { fetch, baseGgufSha256: baseGguf, executor: base, storage: null });
  assert.deepEqual(release.nativeStates.map(item => item.id), [state]);
  assert.equal(await release.nativeStates[0].blob.text(), 'span');
  assert.equal(release.weights[0].resolve('', binding.component, null).soft.instructions.state, state, 'the binding is served with its state');

  let params; const requests = [];
  const fake = { isSupportWebGPU: () => false, isModelLoaded: () => !!params, capabilities: { loraBlobs: true, nativeStates: true },
    async loadModelFromHF() {}, async loadModelFromUrl(_url, received) { params = received; }, async loadModel() {}, async exit() {},
    async createChatCompletion(request) { requests.push(request); return { choices: [{ finish_reason: 'stop', message: { content: 'ok' } }] }; } };
  const engine = { engine: fake, gpuProbe: async () => ({ usable: false, reason: 'test' }) };
  const { model } = await api.loadBrowserLocalModel({ kind: 'url', url: 'https://example.test/base.gguf' },
    { engine, gpuLayers: 0, nativeStates: release.nativeStates });
  assert.deepEqual(params.native_states, [{ blob: release.nativeStates[0].blob }]);
  try {
    await model.turn({ messages: [{ role: 'system', content: `Rules: ${nativeStateSentinel(state)} end` }, { role: 'user', content: 'plain' }],
      tools: [], temperature: 0, seed: null, max_tokens: 4 });
    assert.deepEqual(requests[0].messages[0].content, [{ type: 'text', text: 'Rules: ' }, { type: 'native_state', id: state }, { type: 'text', text: ' end' }]);
    assert.equal(requests[0].messages[1].content, 'plain');
    // A state the page did not load is never sent as text.
    await assert.rejects(model.turn({ messages: [{ role: 'user', content: nativeStateSentinel('nzs1_' + 'b'.repeat(52)) }],
      tools: [], temperature: 0, seed: null, max_tokens: 4 }), /weights-unavailable: native state/);
  } finally { await model.close(); }

  // A wllama build without native-state import refuses the states; a release whose states bind another base runs degraded.
  await assert.rejects(api.loadBrowserLocalModel({ kind: 'url', url: 'https://example.test/base.gguf' },
    { engine: { ...engine, engine: { ...fake, capabilities: { loraBlobs: true }, isModelLoaded: () => false } }, gpuLayers: 0, nativeStates: release.nativeStates }),
  /weights-unavailable: this wllama build cannot load native states/);
  manifest = sealed('d'.repeat(64));
  const other = await api.loadBrowserRelease('https://example.test/release', { fetch, baseGgufSha256: baseGguf, executor: base, storage: null });
  assert.equal(other.nativeStates.length, 0);
  assert.equal(other.weights[0].resolve('', binding.component, null).status, 'degraded');
});
