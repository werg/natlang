import assert from 'node:assert/strict';
import { test } from 'node:test';

async function api() {
  const process = globalThis.process;
  try { globalThis.process = undefined; return await import('../dist/browser/natlang.js'); }
  finally { globalThis.process = process; }
}

test('browser client retries automatic GPU loading on CPU and owns model lifecycle', async () => {
  const { BrowserNatlangClient, BrowserLocalModel } = await api();
  const loads = [], models = [];
  let attempts = 0;
  const factory = () => {
    const index = ++attempts;
    let loaded = false, turns = 0;
    const engine = { isSupportWebGPU: () => true, isModelLoaded: () => loaded,
      async loadModelFromHF() {}, async loadModel() {},
      async loadModelFromUrl(_url, options) {
        loads.push(options); if (index === 1) throw new Error('GPU allocation failed'); loaded = true;
      },
      async createChatCompletion(request) {
        turns++;
        if (turns > 1) return { choices: [{ finish_reason: 'stop',
          message: { content: 'finished' } }], usage: { completion_tokens: 2 } };
        const write = request.tools.find(tool => tool.function.name.startsWith('write_alt_') &&
          (tool.function.parameters.properties.path.const === 'return' ||
            tool.function.parameters.properties.path.enum?.includes('return')) &&
          tool.function.parameters.properties.type?.const === 'Num');
        return { choices: [{ finish_reason: 'tool_calls', message: { content: null,
          tool_calls: [{ id: 'client_1', type: 'function', function: { name: write.function.name,
            arguments: '{"path":"return","type":"Num","value":7}' } }] } }],
          usage: { completion_tokens: 9 } };
      },
      async exit() {},
      getLoadedContextInfo: () => ({ n_ctx: 2048, n_layer: 17 }),
    };
    const model = new BrowserLocalModel({ engine, gpuProbe: async () => ({ usable: true,
      apiAvailable: true, adapterAvailable: true, shaderF16: true,
      browser: 'chromium', reason: 'ready' }) });
    models.push(model); return model;
  };
  const client = new BrowserNatlangClient({ modelFactory: factory });
  const loaded = await client.loadModel({ kind: 'url', url: '/model.gguf', id: 'local-q4' },
    { contextTokens: 2048 });
  assert.equal(loads[0].n_gpu_layers, 99999);
  assert.equal(loads[1].n_gpu_layers, 0);
  assert.match(loaded.gpuFallbackReason, /GPU allocation failed/);
  assert.equal(client.model.loaded, true);
  const result = await client.run({ source: { kind: 'program', program: { $lambda: {
    type: 'Lambda<{}, Num>', instructions: 'Write seven to return.',
  } } }, options: { seed: { mode: 'compatibility' } } });
  assert.equal(result.value, 7);
  assert.equal(result.model.id, 'local-q4');
  assert.equal(result.model.turns.length, 2);
  assert.equal(result.model.turns[0].completionTokens, 9);
  await client.close();
  assert.equal(models.every(model => !model.loaded), true);
  await assert.rejects(client.run({ source: { kind: 'program', program: {} } }), /closed/);
});

test('browser client shares an application host with crisp eval', async () => {
  const { BrowserNatlangClient } = await api();
  const application = { count: 4 };
  const client = new BrowserNatlangClient({ host: application, mode: 'retained' });
  try {
    const result = await client.run({ source: { kind: 'program', program: { $lambda: {
      type: 'Lambda<{}, Num>', engine: 'typescript-host',
      code: 'host.count += 3; return host.count;',
    } } } });
    assert.equal(result.value, 7);
    assert.equal(application.count, 7);
    assert.equal(result.model, null);
  } finally { await client.close(); }
});
