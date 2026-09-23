#!/usr/bin/env node
// Acquire pinned external sources for the curriculum into an untracked cache and record a manifest.
// node scripts/inline-curriculum/acquire.mjs [--source folio] [--cache ../vendor/datasets] [--manifest FILE]
// Every file is fetched from a pinned revision, checked against a recorded checksum when one is known,
// and listed in the manifest with its URL, revision, checksum, and original split. Raw files never
// enter the repository; adapters read them from the cache.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

export const SOURCES = {
  folio: {
    name: 'FOLIO', homepage: 'https://github.com/Yale-LILY/FOLIO', license: 'MIT',
    // The corrected v2 release (huggingface.co/datasets/yale-nlp/FOLIO) is gated behind accepting its terms
    // with a Hugging Face account; until that is done, the openly published v0.0 release is used.
    release: 'v0.0 (GitHub); v2 on Hugging Face is preferred once its terms are accepted',
    revision: '5d7bb84c7edab3fb358e057d2807f19cf5cf5e2d',
    files: [
      { path: 'data/v0.0/folio-train.jsonl', split: 'train', sha256: '008d34b750d31fa7f014e953228adf4db81ec34bbda9e7f67c96c60438d1e6b2' },
      { path: 'data/v0.0/folio-validation.jsonl', split: 'validation', sha256: '6922c988ef10987bd6545568ee8e63e897af80994591fa20539767da58f8e3d1' },
    ],
    url: (revision, path) => `https://raw.githubusercontent.com/Yale-LILY/FOLIO/${revision}/${path}`,
  },
};

export const cachePath = (cache, source, revision, path) => join(cache, source, revision, path);

async function main() {
  const { values } = parseArgs({ options: { source: { type: 'string', default: 'folio' },
    cache: { type: 'string', default: '../vendor/datasets' },
    manifest: { type: 'string', default: '../data/teacher/inline-curriculum/sources.manifest.json' } } });
  const cache = resolve(values.cache), manifestPath = resolve(values.manifest);
  let manifest = { version: 'natlang.curriculum_sources/1', sources: {} };
  try { manifest = JSON.parse(await readFile(manifestPath, 'utf8')); } catch {}
  for (const key of values.source.split(',')) {
    const source = SOURCES[key];
    if (!source) throw new Error(`unknown source ${key}; known: ${Object.keys(SOURCES).join(', ')}`);
    const files = [];
    for (const file of source.files) {
      const url = source.url(source.revision, file.path);
      const target = cachePath(cache, key, source.revision, file.path);
      let bytes;
      try { bytes = await readFile(target); }
      catch {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
        bytes = Buffer.from(await response.arrayBuffer());
        await mkdir(dirname(target), { recursive: true });
        await writeFile(target, bytes);
      }
      const sha256 = createHash('sha256').update(bytes).digest('hex');
      if (file.sha256 && file.sha256 !== sha256) throw new Error(`${url}: checksum ${sha256} does not match the pinned ${file.sha256}`);
      const recorded = manifest.sources[key]?.files?.find(item => item.path === file.path);
      if (recorded && recorded.revision === source.revision && recorded.sha256 !== sha256)
        throw new Error(`${url}: checksum changed since the manifest recorded it`);
      files.push({ path: file.path, url, revision: source.revision, split: file.split, sha256, bytes: bytes.length });
      console.log(`${key} ${file.path} ${sha256.slice(0, 12)} ${bytes.length} bytes`);
    }
    manifest.sources[key] = { name: source.name, homepage: source.homepage, license: source.license, release: source.release,
      revision: source.revision, acquired: new Date().toISOString().slice(0, 10), files };
  }
  await mkdir(dirname(manifestPath), { recursive: true });
  await writeFile(manifestPath, JSON.stringify(manifest, null, 2) + '\n');
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
