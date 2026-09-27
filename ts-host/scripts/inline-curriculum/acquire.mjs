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
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const PYTHON = process.env.NATLANG_PYTHON ?? fileURLToPath(new URL('../../../.venv/bin/python', import.meta.url));
const TABLE_TO_JSONL = `
import csv, json, sys
kind, source, target = sys.argv[1:]
with open(target, 'w') as out:
    if kind == 'parquet':
        import pyarrow.parquet as pq
        parquet = pq.ParquetFile(source)
        # Class labels are stored as indexes; the Hugging Face features in metadata name them.
        features = json.loads((parquet.schema_arrow.metadata or {}).get(b'huggingface', b'{}')).get('info', {}).get('features', {})
        for batch in parquet.iter_batches(batch_size=512):
            for row in batch.to_pylist():
                for column, feature in features.items():
                    if isinstance(feature, dict) and feature.get('names') and row.get(column) is not None:
                        row[column] = feature['names'][row[column]]
                out.write(json.dumps(row) + '\\n')
    else:
        with open(source, newline='') as f:
            for row in csv.DictReader(f): out.write(json.dumps(row) + '\\n')
`;

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
    files: [{ path: 'proofwriter-dataset-V2020.12.3.zip', split: 'mixed', sha256: 'bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26', extract: ['*/OWA/depth-5/meta-*.jsonl'] }],
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
  textworld: {
    name: 'TextWorld', homepage: 'https://github.com/microsoft/TextWorld', license: 'MIT',
    release: 'games generated with textworld 1.7.0 (tw-make custom, seeds 1-300) and exported by textworld_export.py',
    revision: 'textworld-1.7.0-seeds-1-300',
    // Generated, not downloaded: TEXTWORLD_PYTHON is a Python with textworld==1.7.0 installed.
    generate: target => execFileSync(process.env.TEXTWORLD_PYTHON ?? '../vendor/textworld-venv/bin/python',
      [new URL('./textworld_export.py', import.meta.url).pathname, join(target, 'games'), '--seeds', '1-300'], { stdio: 'inherit' }),
    files: [],
  },
  scienceworld: {
    name: 'ScienceWorld', homepage: 'https://github.com/allenai/ScienceWorld', license: 'Apache-2.0',
    release: 'scienceworld 1.2.2 (pinned below 1.3.0, whose action-ordering change alters trajectories); tasks, descriptions, and gold paths for variations 0-2 of every task',
    revision: 'scienceworld-1.2.2-v0-2',
    // Generated from the pinned package: SCIENCEWORLD_PYTHON is a Python with scienceworld==1.2.2 installed (Java 11+).
    generate: target => execFileSync(process.env.SCIENCEWORLD_PYTHON ?? '../vendor/scienceworld-venv/bin/python',
      [new URL('./scienceworld_bridge.py', import.meta.url).pathname, 'export', join(target, 'tasks.json'), '--variations', '3'], { stdio: 'inherit' }),
    files: [],
  },
  alfworld: {
    name: 'ALFWorld', homepage: 'https://github.com/alfworld/alfworld', license: 'MIT',
    release: 'alfworld 0.4.2 text games (json_2.1.1, via alfworld-download into vendor/datasets/alfworld/data); 100 games per split with handcoded-expert command lists',
    revision: 'alfworld-0.4.2-100-per-split',
    // Generated from the pinned package: ALFWORLD_PYTHON has alfworld==0.4.2; ALFWORLD_DATA holds its downloaded data.
    generate: target => execFileSync(process.env.ALFWORLD_PYTHON ?? '../vendor/alfworld-venv/bin/python',
      [new URL('./alfworld_bridge.py', import.meta.url).pathname, 'export', join(target, 'games.json'), '--per-split', '100'],
      { stdio: 'inherit', env: { ...process.env, ALFWORLD_DATA: process.env.ALFWORLD_DATA ?? resolve('../vendor/datasets/alfworld/data') } }),
    files: [],
  },
  commaqa: {
    name: 'CommaQA', homepage: 'https://github.com/allenai/CommaQA', license: 'Apache-2.0',
    release: 'v1 (AI2 public datasets bucket)', revision: 'v1',
    files: [{ path: 'commaqa_explicit.zip', split: 'mixed', sha256: '5305ad2cdf471a358fbb3ef57e5b19024761e910e2a94def10bd19847f3314a0', extract: true }, { path: 'commaqa_numeric.zip', split: 'mixed', sha256: 'b296251c9cd9d63469a88cb44e1048f9050d8bc2866f9f3d87af4180a4ddfdbe', extract: true }],
    url: (revision, path) => `https://ai2-public-datasets.s3.amazonaws.com/commaqa/${revision}/${path}`,
  },
  // Labeled text for per-item nl judgments (labeled.mjs). Tables are converted to JSONL beside the download
  // (convert: parquet or csv), one object per row with the table's columns, class labels by name.
  sms_spam: {
    name: 'SMS Spam Collection', homepage: 'https://archive.ics.uci.edu/dataset/228/sms+spam+collection', license: 'CC-BY-4.0',
    release: 'Hugging Face ucirvine/sms_spam', revision: 'cae486f927c250fe1d4a5b55f11357964ed1646c',
    files: [{ path: 'plain_text/train-00000-of-00001.parquet', split: 'train', convert: 'parquet' }],
    url: (revision, path) => `https://huggingface.co/datasets/ucirvine/sms_spam/resolve/${revision}/${path}`,
  },
  sst2: {
    name: 'SST-2', homepage: 'https://nlp.stanford.edu/sentiment/', license: 'unknown (research use)',
    release: 'Hugging Face stanfordnlp/sst2', revision: '8d51e7e4887a4caaa95b3fbebbf53c0490b58bbb',
    files: [{ path: 'data/train-00000-of-00001.parquet', split: 'train', convert: 'parquet' }],
    url: (revision, path) => `https://huggingface.co/datasets/stanfordnlp/sst2/resolve/${revision}/${path}`,
  },
  ag_news: {
    name: 'AG News', homepage: 'http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html', license: 'unknown (non-commercial research use)',
    release: 'Hugging Face fancyzhx/ag_news', revision: 'eb185aade064a813bc0b7f42de02595523103ca4',
    files: [{ path: 'data/train-00000-of-00001.parquet', split: 'train', convert: 'parquet' }],
    url: (revision, path) => `https://huggingface.co/datasets/fancyzhx/ag_news/resolve/${revision}/${path}`,
  },
  emotion: {
    name: 'Emotion (dair-ai)', homepage: 'https://github.com/dair-ai/emotion_dataset', license: 'other (educational and research use)',
    release: 'Hugging Face dair-ai/emotion, split configuration', revision: 'cab853a1dbdf4c42c2b3ef2173804746df8825fe',
    files: [{ path: 'split/train-00000-of-00001.parquet', split: 'train', convert: 'parquet' }],
    url: (revision, path) => `https://huggingface.co/datasets/dair-ai/emotion/resolve/${revision}/${path}`,
  },
  banking77: {
    name: 'BANKING77', homepage: 'https://github.com/PolyAI-LDN/task-specific-datasets', license: 'CC-BY-4.0',
    release: 'banking_data CSVs (GitHub)', revision: '57ec275d8078af65b7731c2a98be812d844a6d6b',
    files: [{ path: 'banking_data/train.csv', split: 'train', convert: 'csv' }],
    url: (revision, path) => `https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/${revision}/${path}`,
  },
  clinc_oos: {
    name: 'CLINC150', homepage: 'https://github.com/clinc/oos-eval', license: 'CC-BY-3.0',
    release: 'Hugging Face clinc/clinc_oos, plus configuration', revision: '155b9c710419136e17307b80d0a13e68cd46b4ec',
    files: [{ path: 'plus/train-00000-of-00001.parquet', split: 'train', convert: 'parquet' }],
    url: (revision, path) => `https://huggingface.co/datasets/clinc/clinc_oos/resolve/${revision}/${path}`,
  },
  coedit: {
    name: 'CoEdIT', homepage: 'https://huggingface.co/datasets/grammarly/coedit', license: 'Apache-2.0',
    release: 'grammarly/coedit train and validation JSONL', revision: 'e9a255c33ef910bc33a9d2b522653fa87521583e',
    files: [{ path: 'train.jsonl', split: 'train', sha256: '2913249158d6a178dc638e870212ff8a432d128eb6b4bdbe969ee805e6063ce3' },
      { path: 'validation.jsonl', split: 'validation', sha256: '9827b75183d6d06d1e0a48cfb4d5c849a8ae2eec4e05a6f887c40824cf2c1dc9' }],
    url: (revision, path) => `https://huggingface.co/datasets/grammarly/coedit/resolve/${revision}/${path}`,
  },
  hotpotqa: {
    name: 'HotpotQA distractor', homepage: 'https://huggingface.co/datasets/hotpotqa/hotpot_qa', license: 'CC-BY-SA-4.0',
    release: 'hotpotqa/hotpot_qa distractor train', revision: '1908d6afbbead072334abe2965f91bd2709910ab',
    files: [{ path: 'distractor/train-00000-of-00002.parquet', split: 'train', convert: 'parquet',
      sha256: '76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68' },
    { path: 'distractor/train-00001-of-00002.parquet', split: 'train', convert: 'parquet',
      sha256: '713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f' }],
    url: (revision, path) => `https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/${revision}/${path}`,
  },
  cuad: {
    name: 'CUAD v1', homepage: 'https://huggingface.co/datasets/theatticusproject/cuad', license: 'CC-BY-4.0',
    release: 'CUAD v1 SQuAD-style JSON and master clauses CSV', revision: 'a3c393f5d103fd0c516374e4fdff676c8176dcb1',
    files: [{ path: 'CUAD_v1/CUAD_v1.json', split: 'mixed',
      sha256: 'ed0b77d85bdf4014d7495800e8e4a70565b48ee6f8a2e5dca9cf8655dbf10eae' },
      { path: 'CUAD_v1/master_clauses.csv', split: 'mixed', convert: 'csv',
        sha256: '4da237bec677bf5b02212d523857cd57a801adde60e8021de063c8cc06823720' }],
    url: (revision, path) => `https://huggingface.co/datasets/theatticusproject/cuad/resolve/${revision}/${path}`,
  },
  enronqa: {
    name: 'EnronQA', homepage: 'https://huggingface.co/datasets/MichaelR207/enron_qa_0922', license: 'dataset-card terms',
    release: 'MichaelR207/enron_qa_0922 train/dev/test', revision: 'c0b3a9190fd970e83cfbe7d399a08860e43e221e',
    files: ['data/train-00000-of-00002.parquet', 'data/train-00001-of-00002.parquet',
      'data/dev-00000-of-00001.parquet', 'data/test-00000-of-00001.parquet']
      .map(path => ({ path, split: path.split('/')[1].split('-')[0], convert: 'parquet' })),
    url: (revision, path) => `https://huggingface.co/datasets/MichaelR207/enron_qa_0922/resolve/${revision}/${path}`,
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
    if (source.generate) {
      const target = cachePath(cache, key, source.revision, '');
      await mkdir(target, { recursive: true });
      source.generate(target);
    }
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
      // extract: true unpacks everything; a list of patterns unpacks only the members an adapter reads.
      if (file.extract) execFileSync('unzip', ['-o', '-q', target, ...(Array.isArray(file.extract) ? file.extract : []), '-d', target.replace(/\.zip$/, '')]);
      // A table becomes JSONL beside itself (TABLE.jsonl), read by the adapters without a table library.
      if (file.convert) execFileSync(PYTHON, ['-c', TABLE_TO_JSONL, file.convert, target, `${target}.jsonl`]);
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
