import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promoteBrowserModel } from '../../scripts/publish_browser_model.mjs';

async function fixture() {
  const root = await mkdtemp(join(tmpdir(), 'natlang-browser-publish-'));
  await mkdir(join(root, 'runs/demo/merged'), { recursive: true });
  await mkdir(join(root, 'models/templates'), { recursive: true });
  await mkdir(join(root, 'ts-host/src'), { recursive: true });
  await writeFile(join(root, 'runs/demo/merged/natlang_training.json'), JSON.stringify({
    step: 300, corpus: { data_sha256: 'a'.repeat(64) },
  }));
  await writeFile(join(root, 'runs/demo/merged/tokenizer_config.json'), '{}');
  await writeFile(join(root, 'runs/demo/merged/config.json'), '{"model_type":"lfm2"}');
  await writeFile(join(root, 'models/templates/LFM2.5-350M.jinja'), 'official tool template');
  const header = Buffer.alloc(1_000_000); header.write('GGUF'); header.writeUInt32LE(3, 4);
  await writeFile(join(root, 'models/candidate.gguf'), header);
  return root;
}

test('publication verifies GGUF and provenance, then atomically changes one shared default', async () => {
  const root = await fixture();
  const first = await promoteBrowserModel({ root, run: 'runs/demo', artifact: 'models/candidate.gguf',
    name: 'demo', downloadUrl: 'https://models.example/demo.gguf' });
  assert.equal(first.entry.quant, 'Q4_K_M');
  assert.equal(first.catalog.defaultId, first.entry.id);
  assert.match(await readFile(join(root, 'ts-host/src/model-default.ts'), 'utf8'),
    /https:\/\/models\.example\/demo\.gguf/);
  assert.equal(await readFile(join(root, 'models/templates/LFM2.5-350M.jinja'), 'utf8'),
    'official tool template');
  assert.equal((await stat(join(root, 'models/candidate.gguf'))).size, 1_000_000);
  assert.equal((await readFile(join(root, 'models/templates', first.entry.file.replace('-Q4_K_M.gguf', '.jinja')), 'utf8')),
    'official tool template');
  const second = await promoteBrowserModel({ root, run: 'runs/demo', artifact: 'models/candidate.gguf',
    name: 'demo-eight', quant: 'Q8_0' });
  assert.equal(second.catalog.defaultId, second.entry.id);
  assert.equal(second.catalog.models.length, 2);
  const published = JSON.parse(await readFile(join(root, 'models/browser-catalog.json'), 'utf8'));
  assert.equal(published.models[1].id, first.entry.id);

  const { loadBrowserModelCatalog } = await import('../dist/browser/models.js');
  const catalog = await loadBrowserModelCatalog('/models/browser-catalog.json', async () => ({
    ok: true, json: async () => published,
  }));
  assert.equal(catalog.defaultId, second.entry.id);
  assert.equal(catalog.source, 'published');
});

test('browser catalog falls back only when no publication exists', async () => {
  const { loadBrowserModelCatalog } = await import('../dist/browser/models.js');
  const fallback = await loadBrowserModelCatalog('/models/browser-catalog.json',
    async () => ({ status: 404 }));
  assert.equal(fallback.source, 'builtin');
  assert.equal(fallback.defaultId, fallback.models[0].id);
  const empty = await loadBrowserModelCatalog('/models/browser-catalog.json', async () => ({
    ok: true, json: async () => ({ schema: 'natlang.browser-model-catalog/1',
      defaultId: '', models: [] }),
  }));
  assert.deepEqual(empty.models, []);
  await assert.rejects(loadBrowserModelCatalog('/models/browser-catalog.json',
    async () => ({ ok: true, json: async () => ({
      schema: 'natlang.browser-model-catalog/1', defaultId: 'bad', models: [null],
    }) })), /invalid published browser model catalog/);
});

test('invalid artifact cannot replace the published pointer', async () => {
  const root = await fixture();
  await writeFile(join(root, 'models/candidate.gguf'), Buffer.alloc(1_000_000));
  await assert.rejects(promoteBrowserModel({ root, run: 'runs/demo',
    artifact: 'models/candidate.gguf', name: 'invalid' }), /GGUF header/);
  await assert.rejects(readFile(join(root, 'models/browser-catalog.json')), /ENOENT/);
});
