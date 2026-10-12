#!/usr/bin/env node
/**
 * The natural-language side of job placement (applications/ops-placement/DECOMPOSITION.md). scripts/coreweave/placement.py
 * writes the needs, fleet facts and crisp decision of each job and runs this script when its mode is `nl` or `shadow`;
 * it reads the served decisions back.
 *
 *   node scripts/placement-decide.mjs --input ITEMS.json --out DECISIONS.json --mode nl|shadow \
 *       [--shadow-log LOG.jsonl] --server URL --model NAME
 *
 * ITEMS.json: { items: [{ id, job, fleet, crisp: { where, reason } }] }.
 * DECISIONS.json: { mode, placer, summary, placed: [{ id, where, reason, served }] }.
 * `--dry-run` prints the item count and calls no model.
 */
import { appendFileSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { parseArgs } from 'node:util';
import { fileURLToPath } from 'node:url';
import { explainerIdentity } from './advisory-file.mjs';

const root = fileURLToPath(new URL('../../applications/ops-placement/', import.meta.url));
const dist = new URL('../../applications/dist/ops-placement/index.js', import.meta.url).href;

export async function main(argv, { createRuntime } = {}) {
  const { values } = parseArgs({ args: argv, options: { input: { type: 'string' }, out: { type: 'string' }, mode: { type: 'string' },
    'shadow-log': { type: 'string' }, server: { type: 'string' }, model: { type: 'string' }, 'dry-run': { type: 'boolean' } } });
  if (!values.input || !values.out || !values.mode) { console.error('usage: placement-decide.mjs --input ITEMS.json --out DECISIONS.json --mode nl|shadow ... (see the file header)'); return 2; }
  const app = await import(dist);
  const { items } = JSON.parse(readFileSync(values.input, 'utf8'));
  const placer = explainerIdentity({ placeJob: readFileSync(join(root, 'placeJob.nl'), 'utf8') }, values.model ?? 'unspecified');
  if (values['dry-run']) { console.log(JSON.stringify({ items: items.length, mode: values.mode, placer: placer.explainer })); return 0; }
  let rt;
  if (createRuntime) rt = await createRuntime();
  else {
    if (!values.server || !values.model) throw new Error('--server and --model are required (or --dry-run)');
    const { createNatlangRuntime, openAICompatibleModelTurn } = await import(new URL('../dist/index.js', import.meta.url).href);
    rt = createNatlangRuntime({ model: openAICompatibleModelTurn({ endpoint: values.server, model: values.model }), seed: { mode: 'backend' } });
  }
  try {
    const { placed, records, summary } = await app.placeItems(items, values.mode, app.nlPlace(rt));
    if (values['shadow-log'] && records.length) {
      mkdirSync(dirname(values['shadow-log']), { recursive: true });
      appendFileSync(values['shadow-log'], records.map(record => JSON.stringify({ ...record, placer: placer.explainer })).join('\n') + '\n');
    }
    mkdirSync(dirname(values.out), { recursive: true });
    writeFileSync(values.out, JSON.stringify({ mode: summary.mode, placer: placer.explainer, summary, placed }));
    console.log(JSON.stringify({ ...summary, placer: placer.explainer, out: values.out }));
  } finally { rt.close?.(); }
  return 0;
}

if (import.meta.url === `file://${process.argv[1]}`) process.exit(await main(process.argv.slice(2)));
