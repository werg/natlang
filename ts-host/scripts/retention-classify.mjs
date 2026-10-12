#!/usr/bin/env node
/**
 * The natural-language side of storage retention (applications/ops-retention/DECOMPOSITION.md). scripts/storage_retention.py
 * writes the facts and crisp verdicts of every reported file and runs this script when its classifier mode is `nl` or
 * `shadow`; it reads the served verdicts back.
 *
 *   node scripts/retention-classify.mjs --input ITEMS.json --out VERDICTS.json --mode nl|shadow \
 *       [--shadow-log LOG.jsonl] [--concurrency N] --server URL --model NAME
 *
 * ITEMS.json: { thresholds, items: [{ id, path, logical, bytes, facts, crisp: { tier, reason } }] }.
 * VERDICTS.json: { mode, classifier, summary, classified: [{ id, tier, reason, served }] }.
 * Each disagreement, refused verdict or failed call is appended to the shadow log, one JSON object per line.
 * `--dry-run` prints the item count and calls no model. Needs a served model; nothing here loads one.
 */
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { explainerIdentity } from './advisory-file.mjs';

const root = fileURLToPath(new URL('../../applications/ops-retention/', import.meta.url));
const dist = new URL('../../applications/dist/ops-retention/index.js', import.meta.url).href;

export async function main(argv, { createRuntime } = {}) {
  const { values } = parseArgs({ args: argv, options: { input: { type: 'string' }, out: { type: 'string' }, mode: { type: 'string' },
    'shadow-log': { type: 'string' }, concurrency: { type: 'string' }, server: { type: 'string' }, model: { type: 'string' },
    'dry-run': { type: 'boolean' } } });
  if (!values.input || !values.out || !values.mode) { console.error('usage: retention-classify.mjs --input ITEMS.json --out VERDICTS.json --mode nl|shadow ... (see the file header)'); return 2; }
  const app = await import(dist);
  const { thresholds, items } = JSON.parse(readFileSync(values.input, 'utf8'));
  const classifier = explainerIdentity({ classifyRetention: readFileSync(join(root, 'classifyRetention.nl'), 'utf8') }, values.model ?? 'unspecified');
  if (values['dry-run']) { console.log(JSON.stringify({ items: items.length, mode: values.mode, classifier: classifier.explainer })); return 0; }
  let rt;
  if (createRuntime) rt = await createRuntime();
  else {
    if (!values.server || !values.model) throw new Error('--server and --model are required (or --dry-run)');
    const { createNatlangRuntime, openAICompatibleModelTurn } = await import(new URL('../dist/index.js', import.meta.url).href);
    rt = createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: values.server, model: values.model }), seed: { mode: 'backend' } });
  }
  try {
    const { classified, records, summary } = await app.classifyItems(items, thresholds, values.mode, app.nlClassify(rt),
      { concurrency: values.concurrency ? Number(values.concurrency) : undefined });
    if (values['shadow-log'] && records.length) {
      mkdirSync(dirname(values['shadow-log']), { recursive: true });
      appendFileSync(values['shadow-log'], records.map(record => JSON.stringify({ ...record, classifier: classifier.explainer })).join('\n') + '\n');
    }
    mkdirSync(dirname(values.out), { recursive: true });
    writeFileSync(values.out, JSON.stringify({ mode: summary.mode, classifier: classifier.explainer, summary, classified }));
    console.log(JSON.stringify({ ...summary, classifier: classifier.explainer, out: values.out }));
  } finally { rt.close?.(); }
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(await main(process.argv.slice(2)));
