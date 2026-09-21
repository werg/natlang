#!/usr/bin/env node
/** Convert a completed merged checkpoint, then atomically publish the browser default. */
import { createHash, randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { createReadStream, existsSync } from 'node:fs';
import { copyFile, mkdir, open, readFile, rename, stat, unlink, writeFile } from 'node:fs/promises';
import { basename, join, relative, resolve, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = resolve(fileURLToPath(new URL('..', import.meta.url)));
const schema = 'natlang.browser-model-catalog/1';
const slug = value => /^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$/.test(value);
const inside = (root, path, folder) => {
  const absolute = resolve(root, path);
  if (!absolute.startsWith(join(root, folder) + sep)) throw new Error(`${path} must be under ${folder}/`);
  return absolute;
};
const digest = async path => {
  const hash = createHash('sha256');
  for await (const chunk of createReadStream(path)) hash.update(chunk);
  return hash.digest('hex');
};

export async function verifyGguf(path) {
  const info = await stat(path);
  if (info.size < 1_000_000) throw new Error('GGUF is too small to be a trained model');
  const handle = await open(path, 'r');
  try {
    const header = Buffer.alloc(8);
    await handle.read(header, 0, 8, 0);
    if (header.toString('ascii', 0, 4) !== 'GGUF' || ![2, 3].includes(header.readUInt32LE(4)))
      throw new Error('GGUF header is invalid');
  } finally { await handle.close(); }
  return info.size;
}

async function templateFor(merged, explicit, root) {
  if (explicit) return readFile(explicit, 'utf8');
  const tokenizer = JSON.parse(await readFile(join(merged, 'tokenizer_config.json'), 'utf8'));
  if (typeof tokenizer.chat_template === 'string' && tokenizer.chat_template.trim())
    return tokenizer.chat_template;
  const config = JSON.parse(await readFile(join(merged, 'config.json'), 'utf8'));
  if (config.model_type === 'lfm2')
    return readFile(join(root, 'models/templates/LFM2.5-350M.jinja'), 'utf8');
  throw new Error('checkpoint has no chat template; pass --template for this architecture');
}

async function withCatalogLock(root, action) {
  const path = join(root, 'models/browser-catalog.lock');
  let handle;
  for (let attempt = 0; attempt < 40; attempt++) {
    try { handle = await open(path, 'wx'); break; }
    catch (error) { if (error.code !== 'EEXIST') throw error;
      await new Promise(resolve => setTimeout(resolve, 250)); }
  }
  if (!handle) throw new Error('browser catalog is locked by another publication');
  try { return await action(); }
  finally { await handle.close(); await unlink(path); }
}

/** Promote only after the artifact, template, and training provenance are verified. */
export async function promoteBrowserModel({ root = repository, run, artifact, name, quant = 'Q4_K_M',
  template, contextTokens = 8192, moveArtifact = false, downloadUrl = '' }) {
  root = resolve(root);
  if (!slug(name) || !['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0'].includes(quant))
    throw new Error('invalid browser model name or quantization');
  if (downloadUrl && (() => { try { return new URL(downloadUrl).protocol === 'https:'; } catch { return false; } })() === false)
    throw new Error('downloadUrl must be an absolute HTTPS URL');
  if (!Number.isSafeInteger(contextTokens) || contextTokens < 512)
    throw new Error('contextTokens must be at least 512');
  const merged = inside(root, join(run, 'merged'), 'runs');
  const artifactPath = inside(root, artifact, 'models');
  const training = JSON.parse(await readFile(join(merged, 'natlang_training.json'), 'utf8'));
  if (!Number.isSafeInteger(training.step) || training.step < 1 || !training.corpus?.data_sha256)
    throw new Error('merged checkpoint lacks completed training provenance');
  const bytes = await verifyGguf(artifactPath);
  const sha256 = await digest(artifactPath);
  const templateText = await templateFor(merged, template, root);
  if (!templateText.trim()) throw new Error('chat template is empty');
  const stem = `${name}-step${training.step}-${sha256.slice(0, 12)}`;
  const modelFile = `${stem}-${quant}.gguf`;
  const templateFile = `${stem}.jinja`;
  await mkdir(join(root, 'models/templates'), { recursive: true });
  const finalPath = join(root, 'models', modelFile);
  if (artifactPath !== finalPath) {
    const existing = await stat(finalPath).catch(() => null);
    if (existing && await digest(finalPath) !== sha256) throw new Error('immutable model filename collision');
    if (!existing) {
      if (moveArtifact) await rename(artifactPath, finalPath);
      else await copyFile(artifactPath, finalPath);
    }
  }
  const templatePath = join(root, 'models/templates', templateFile);
  await writeFile(templatePath, templateText, { flag: 'wx' }).catch(async error => {
    if (error.code !== 'EEXIST' || await readFile(templatePath, 'utf8') !== templateText) throw error;
  });
  const entry = { id: stem.toLowerCase() + '-' + quant.toLowerCase(),
    label: `${name} · step ${training.step} · ${quant}`, trainingRun: relative(root, resolve(root, run)),
    file: modelFile, templateUrl: `/models/templates/${templateFile}`, quant, bytes, sha256,
    contextTokens, url: `/models/${modelFile}`, ...(downloadUrl ? { downloadUrl } : {}) };
  const catalog = await withCatalogLock(root, async () => {
    const path = join(root, 'models/browser-catalog.json');
    const previous = await readFile(path, 'utf8').then(JSON.parse).catch(error => {
      if (error.code === 'ENOENT') return { models: [] };
      throw error;
    });
    if (!Array.isArray(previous.models)) throw new Error('existing browser catalog is invalid');
    const next = { schema, defaultId: entry.id,
      models: [entry, ...previous.models.filter(model => model.id !== entry.id)],
      publishedAt: new Date().toISOString() };
    const pending = `${path}.${randomUUID()}.tmp`;
    await writeFile(pending, JSON.stringify(next, null, 2) + '\n');
    await rename(pending, path);
    return next;
  });
  const runtimeSource = join(root, 'ts-host/src/model-default.ts');
  if (existsSync(join(root, 'ts-host/src'))) {
    // This tracked canonical asset makes the CLI release self-contained even
    // though per-checkpoint browser artifacts under models/ are ignored.
    const runtimeTemplate = 'LFM2.5-350M.jinja';
    await writeFile(join(root, 'models/templates', runtimeTemplate), templateText);
    const runtimeDefault = { id: entry.id, label: entry.label, trainingRun: entry.trainingRun,
      file: entry.file, quant: entry.quant, bytes: entry.bytes, sha256: entry.sha256,
      contextTokens: entry.contextTokens, downloadUrl, template: runtimeTemplate };
    const pending = `${runtimeSource}.${randomUUID()}.tmp`;
    await writeFile(pending, `/** Generated by scripts/publish_browser_model.mjs; do not edit by hand. */\nexport const DEFAULT_MODEL_RELEASE = Object.freeze(${JSON.stringify(runtimeDefault, null, 2)});\n`);
    await rename(pending, runtimeSource);
  }
  return { entry, catalog };
}

async function convert(run, name, quant, root) {
  const partial = `models/${name}-${quant}-partial-${randomUUID()}.gguf`;
  const runPath = relative(root, inside(root, run, 'runs'));
  const args = ['scripts/to_gguf.sh', `${runPath}/merged`, partial, quant];
  const code = await new Promise((resolveCode, reject) => {
    const child = spawn('bash', args, { cwd: root, stdio: 'inherit' });
    child.once('error', reject); child.once('close', resolveCode);
  });
  if (code !== 0) throw new Error(`GGUF conversion failed (${code})`);
  return partial;
}

async function main() {
  const args = process.argv.slice(2);
  const option = name => args[args.indexOf(name) + 1];
  const run = option('--run'), name = option('--name'), quant = args.includes('--quant') ? option('--quant') : 'Q4_K_M';
  const template = args.includes('--template') ? option('--template') : undefined;
  const artifact = args.includes('--artifact') ? option('--artifact') : undefined;
  const downloadUrl = args.includes('--download-url') ? option('--download-url') : '';
  if (!run || !name || !slug(name) || !['Q4_K_M', 'Q5_K_M', 'Q6_K', 'Q8_0'].includes(quant))
    throw new Error('usage: publish_browser_model.mjs --run runs/NAME --name SLUG [--quant Q4_K_M] [--template PATH] [--artifact models/FILE.gguf] [--download-url HTTPS_URL]');
  inside(repository, run, 'runs');
  const candidate = artifact ?? await convert(run, name, quant, repository);
  try {
    const result = await promoteBrowserModel({ run, name, quant, artifact: candidate, template,
      moveArtifact: !artifact, downloadUrl });
    console.log(JSON.stringify({ defaultId: result.catalog.defaultId, model: result.entry.url,
      sha256: result.entry.sha256, bytes: result.entry.bytes }));
  } catch (error) { if (!artifact) await unlink(resolve(repository, candidate)).catch(() => {}); throw error; }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url))
  main().catch(error => { console.error(error); process.exitCode = 1; });
