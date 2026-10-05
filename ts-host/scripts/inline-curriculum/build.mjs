#!/usr/bin/env node
// Build and verify a shard of the inline-natlang curriculum.
// node scripts/inline-curriculum/build.mjs --seed 1 --shapes 4 [--start 0] [--families a,b] [--split test] --out ../data/teacher/inline-curriculum/s1.ir.jsonl
// Every case is validated, its decisive observations are checked to be absent from its opening, its
// reference solution is replayed through the collector's execution path and admitted, and every
// counterfactual group is checked to share one opening with differing results. A shard is written
// only when every case verifies, with a report beside it.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { verifyCases } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { datasetQualityReport } from './folder-data.mjs';
import { FAMILIES, buildRecords } from './families.mjs';

const { values } = parseArgs({ allowNegative: true, options: { seed: { type: 'string', default: '1' }, shapes: { type: 'string', default: '2' },
  start: { type: 'string', default: '0' },
  families: { type: 'string' }, track: { type: 'string' }, out: { type: 'string' }, 'allow-failures': { type: 'boolean', default: false },
  // Held-out generated problems: --split test marks this build's synthetic cases as test (use a seed no training build uses).
  split: { type: 'string', default: 'train' }, hints: { type: 'boolean', default: true } } });
if (!values.out) throw new Error('--out FILE is required');
const seed = Number(values.seed), shapes = Number(values.shapes), start = Number(values.start);
if ([102,900].includes(seed) && values.split !== 'test')
  throw new Error(`seed ${seed} is reserved for native evaluation; use --split test or a fresh training seed`);
const selected = values.families ? values.families.split(',') : Object.keys(FAMILIES).filter(name => !FAMILIES[name].demonstration);
for (const name of selected) if (!FAMILIES[name]) throw new Error(`unknown family ${name}; known: ${Object.keys(FAMILIES).join(', ')}`);
if (values.track && selected.some(name => (FAMILIES[name].track ?? 'interpreter') !== values.track))
  throw new Error(`--families contains a family outside track ${values.track}`);
if (!selected.length) throw new Error('no families selected');

const records = buildRecords({ seed, shapes, start, families: selected, split: values.split, hints: values.hints });

const verification = await verifyCases(records, TOOLS_PROMPT);
const failures = verification.filter(item => !item.ok);
for (const item of failures) console.error(`FAIL ${item.id}\n  ${item.problems.join('\n  ')}`);
const out = resolve(values.out);
await mkdir(dirname(out), { recursive: true });
const byFamily = {};
for (const record of records) byFamily[record.curriculum.family] = (byFamily[record.curriculum.family] ?? 0) + 1;
const report = { seed, shapes, start, families: selected, cases: records.length, verified: records.length - failures.length,
  by_family: byFamily, source_quarantine: datasetQualityReport(), failures: failures.map(item => ({ id: item.id, problems: item.problems })) };
await writeFile(`${out}.report.json`, JSON.stringify(report, null, 2) + '\n');
if (failures.length && !values['allow-failures']) {
  console.error(`${failures.length} of ${records.length} cases failed verification; no shard written (report: ${out}.report.json)`);
  process.exit(1);
}
const kept = records.filter(record => !failures.some(item => item.id === record.id));
await writeFile(out, kept.map(record => JSON.stringify(record)).join('\n') + '\n');
console.log(`${kept.length} verified cases across ${Object.keys(byFamily).length} families -> ${out}`);
