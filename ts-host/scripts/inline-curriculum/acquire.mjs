#!/usr/bin/env node
// Acquire pinned external sources for the curriculum into an untracked cache and record a manifest.
// node scripts/inline-curriculum/acquire.mjs [--source folio] [--cache ../vendor/datasets] [--manifest FILE]
// Every file is fetched from a pinned revision, checked against a recorded checksum when one is known,
// and listed in the manifest with its URL, revision, checksum, and original split. Raw files never
// enter the repository; adapters read them from the cache.
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
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
  prontoqa: {
    name: 'PrOntoQA-OOD', homepage: 'https://github.com/asaparov/prontoqa', license: 'Apache-2.0',
    release: 'generated_ood_data.zip as regenerated on 2024-10-17',
    revision: '0a6412b6fddf46324a1cb96e066dd7b3d89b87d6',
    // Each generated file holds in-context examples (used for training) and one test example per entry (held out).
    files: [{ path: 'generated_ood_data.zip', split: 'mixed', sha256: '0becba04e1e1eb80593709c6a3db2a46badd5587a29587ffaeb68ed5068c9de9', extract: true }],
    url: (revision, path) => `https://raw.githubusercontent.com/asaparov/prontoqa/${revision}/${path}`,
  },
  kqapro: {
    name: 'KQA Pro', homepage: 'https://github.com/shijx12/KQAPro_Baselines',
    // The authors release KQA Pro under CC BY-SA 4.0 (the mirror's card says MIT; the original terms govern).
    // Share-alike may extend to derived training rows: review before any are used for training.
    license: 'CC-BY-SA-4.0',
    release: 'Hugging Face mirror drt/kqa_pro (the maintainers\' Tsinghua cloud link no longer serves the archive)',
    revision: '0b26da66cec9a4d1e42bde3560aeae9f89f6433b',
    files: [
      { path: 'kb.json', split: 'kb', sha256: '04da7408320c5cb7023c44372cce32846d56d369d8865d2e61a18c3956661a7c' },
      { path: 'train.json', split: 'train', sha256: 'e9fbe4c1cdf207aac83ae0d5e4a1a53a9965a2b13b403de699ca6d5dae6e4510' },
      { path: 'val.json', split: 'validation', sha256: 'b4aed6ab3d7ad071722064fe3bb02bc028cfbeb15da5f7115d57a1e2d198f3bb' },
    ],
    url: (revision, path) => `https://huggingface.co/datasets/drt/kqa_pro/resolve/${revision}/${path}`,
  },
  // Checksums below were recorded on first acquisition (2026-09-24); the maintainers publish none.
  proofwriter: {
    name: 'ProofWriter', homepage: 'https://allenai.org/data/proofwriter', license: 'CC-BY-4.0',
    release: 'V2020.12.3 (AI2 public data bucket)', revision: 'V2020.12.3',
    files: [{ path: 'proofwriter-dataset-V2020.12.3.zip', split: 'mixed', sha256: 'bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26', extract: true }],
    url: (revision, path) => `https://aristo-data-public.s3.amazonaws.com/proofwriter/${path}`,
  },
  entailmentbank: {
    name: 'EntailmentBank', homepage: 'https://allenai.org/data/entailmentbank', license: 'CC-BY-4.0',
    release: 'v3 via the Hugging Face mirror ariesutiono/entailment-bank-v3 (the official copy is a Google Drive folder)',
    revision: '2d1b8010d08c2e6ce17c4879447b9a3ce7531d5e',
    files: [
      { path: 'task2_train.jsonl', split: 'train', sha256: '36cdb362c24755b9640ed54e671fc9c72427b6c918f79429551a0800e9055a1b' },
      { path: 'task2_dev.jsonl', split: 'validation', sha256: '3271adc67c65149780adbd3729f6b19404ff288e1849905fc16c1c22814a28f7' },
      { path: 'task2_test.jsonl', split: 'test', sha256: '8bba350ef207f92f9d153e3b1651d90535586a157e444fe037756a8c1cb84f0f' },
    ],
    url: (revision, path) => `https://huggingface.co/datasets/ariesutiono/entailment-bank-v3/resolve/${revision}/${path}`,
  },
  anli: {
    name: 'Abductive NLI (αNLI)', homepage: 'https://github.com/allenai/abductive-commonsense-reasoning', license: 'Apache-2.0',
    release: 'anli.zip (data only), ICLR 2020 release', revision: 'iclr2020',
    files: [{ path: 'anli.zip', split: 'mixed', sha256: '4e00551fd9ee04c92e823a8fe078e017c78b35739b36e8f9f122b4bf8a84b16b', extract: true }],
    url: (revision, path) => `https://storage.googleapis.com/ai2-mosaic/public/abductive-commonsense-reasoning-iclr2020/${path}`,
  },
  commaqa: {
    name: 'CommaQA', homepage: 'https://github.com/allenai/CommaQA', license: 'Apache-2.0',
    release: 'v1 (AI2 public datasets bucket)', revision: 'v1',
    files: [{ path: 'commaqa_explicit.zip', split: 'mixed', sha256: '5305ad2cdf471a358fbb3ef57e5b19024761e910e2a94def10bd19847f3314a0', extract: true }, { path: 'commaqa_numeric.zip', split: 'mixed', sha256: 'b296251c9cd9d63469a88cb44e1048f9050d8bc2866f9f3d87af4180a4ddfdbe', extract: true }],
    url: (revision, path) => `https://ai2-public-datasets.s3.amazonaws.com/commaqa/${revision}/${path}`,
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
      // Archives are unpacked next to themselves, into a directory named after the archive.
      if (file.extract) execFileSync('unzip', ['-o', '-q', target, '-d', target.replace(/\.zip$/, '')]);
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
