import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { mkdtemp, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { promoteBrowserModel, publishNativeDefault } from '../../scripts/publish_browser_model.mjs';

async function fixture(t) {
  const root = await mkdtemp(join(tmpdir(), 'natlang-browser-publish-'));
  t.after(() => rm(root, { recursive: true, force: true }));
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

test('publication verifies GGUF and provenance, then atomically changes one shared default', async (t) => {
  const root = await fixture(t);
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

test('invalid artifact cannot replace the published pointer', async (t) => {
  const root = await fixture(t);
  await writeFile(join(root, 'models/candidate.gguf'), Buffer.alloc(1_000_000));
  await assert.rejects(promoteBrowserModel({ root, run: 'runs/demo',
    artifact: 'models/candidate.gguf', name: 'invalid' }), /GGUF header/);
  await assert.rejects(readFile(join(root, 'models/browser-catalog.json')), /ENOENT/);
});

test('a native GGUF export becomes the CLI default with its serving settings and download URL', async () => {
  const root = await mkdtemp(join(tmpdir(), 'native-default-')), directory = join(root, 'export');
  await mkdir(join(root, 'ts-host/src'), { recursive: true }); await mkdir(directory);
  const gguf = Buffer.alloc(1_000_001); gguf.write('GGUF', 0, 'ascii'); gguf.writeUInt32LE(3, 4);
  await writeFile(join(directory, 'model-q8_0.gguf'), gguf);
  await writeFile(join(directory, 'chat-template.jinja'), '{{ messages }}');
  const sha = text => createHash('sha256').update(text).digest('hex');
  const manifest = { schema: 'natlang.native-gguf/1', step: 40658, architecture: 'lfm2', checkpoint_sha256: 'ab'.repeat(32),
    serving: { chat_template: 'chat-template.jinja', chat_template_sha256: sha('{{ messages }}'),
      chat_template_kwargs: { preserve_thinking: true }, logit_bias: { 64402: -100 } },
    files: [{ outtype: 'q8_0', file: 'model-q8_0.gguf', bytes: gguf.length, sha256: sha(gguf) }] };
  await writeFile(join(directory, 'manifest.json'), JSON.stringify(manifest));
  const uploads = [];
  const result = await publishNativeDefault({ root, directory, tag: 'natlang-default-test',
    upload: async item => { uploads.push(item); return `https://example.test/${item.tag}/${item.name}`; } });
  assert.equal(uploads.length, 1);
  assert.equal(uploads[0].name, `natlang-lfm25-350m-step40658-${'ab'.repeat(6)}-Q8_0.gguf`);
  assert.equal(result.downloadUrl, `https://example.test/natlang-default-test/${uploads[0].name}`);
  assert.deepEqual(result.request, { temperature: 0, chat_template_kwargs: { preserve_thinking: true }, logit_bias: { 64402: -100 } });
  assert.equal(await readFile(join(root, 'models/templates', result.template), 'utf8'), '{{ messages }}');
  const source = await readFile(join(root, 'ts-host/src/model-default.ts'), 'utf8');
  assert.match(source, /export const DEFAULT_MODEL_RELEASE = Object\.freeze\(/);
  assert.ok(source.includes(result.sha256) && source.includes('"logit_bias"'));
  manifest.files[0].sha256 = 'cd'.repeat(32);
  await writeFile(join(directory, 'manifest.json'), JSON.stringify(manifest));
  await assert.rejects(publishNativeDefault({ root, directory, tag: 'x', upload: async () => 'https://x' }), /differs from its manifest/);
  await rm(root, { recursive: true, force: true });
});
