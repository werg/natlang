#!/usr/bin/env node
// Build and verify a shard of the inline-natlang curriculum.
// node scripts/inline-curriculum/build.mjs --seed 1 --shapes 4 [--start 0] [--families a,b] --out ../data/teacher/inline-curriculum/s1.ir.jsonl
// Every case is validated, its decisive observations are checked to be absent from its opening, its
// reference solution is replayed through the collector's execution path and admitted, and every
// counterfactual group is checked to share one opening with differing results. A shard is written
// only when every case verifies, with a report beside it.
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { verifyCases } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { FAMILIES } from './families.mjs';

const { values } = parseArgs({ options: { seed: { type: 'string', default: '1' }, shapes: { type: 'string', default: '2' },
  start: { type: 'string', default: '0' },
  families: { type: 'string' }, out: { type: 'string' }, 'allow-failures': { type: 'boolean', default: false } } });
if (!values.out) throw new Error('--out FILE is required');
const seed = Number(values.seed), shapes = Number(values.shapes), start = Number(values.start);
const selected = values.families ? values.families.split(',') : Object.keys(FAMILIES);
for (const name of selected) if (!FAMILIES[name]) throw new Error(`unknown family ${name}; known: ${Object.keys(FAMILIES).join(', ')}`);

const records = [];
for (const name of selected) {
  const family = FAMILIES[name];
  const count = Math.max(1, Math.round(shapes * (family.weight ?? 1)));
  for (let index = start; index < start + count; index++) for (const record of family.build(seed, index)) {
    // Generated problems are distinct per seed; a source's problems are its own (story, world) groups across shards.
    const tag = family.source ? `${family.source}` : `s${seed}`;
    record.id = record.id.replace('inline-curriculum:', `inline-curriculum:${tag}:`);
    record.source_ids = [record.id];
    if (!family.source) {
      record.curriculum.split_group = `s${seed}:${record.curriculum.split_group}`;
      record.source_groups = [record.curriculum.split_group];
      if (record.curriculum.pair_group) record.curriculum.pair_group = `s${seed}:${record.curriculum.pair_group}`;
    }
    records.push(record);
  }
}
const ids = new Set();
for (const record of records) { if (ids.has(record.id)) throw new Error(`duplicate case id ${record.id}`); ids.add(record.id); }

const verification = await verifyCases(records, TOOLS_PROMPT);
const failures = verification.filter(item => !item.ok);
for (const item of failures) console.error(`FAIL ${item.id}\n  ${item.problems.join('\n  ')}`);
const out = resolve(values.out);
await mkdir(dirname(out), { recursive: true });
const byFamily = {};
for (const record of records) byFamily[record.curriculum.family] = (byFamily[record.curriculum.family] ?? 0) + 1;
const report = { seed, shapes, start, families: selected, cases: records.length, verified: records.length - failures.length,
  by_family: byFamily, failures: failures.map(item => ({ id: item.id, problems: item.problems })) };
await writeFile(`${out}.report.json`, JSON.stringify(report, null, 2) + '\n');
if (failures.length && !values['allow-failures']) {
  console.error(`${failures.length} of ${records.length} cases failed verification; no shard written (report: ${out}.report.json)`);
  process.exit(1);
}
const kept = records.filter(record => !failures.some(item => item.id === record.id));
await writeFile(out, kept.map(record => JSON.stringify(record)).join('\n') + '\n');
console.log(`${kept.length} verified cases across ${Object.keys(byFamily).length} families -> ${out}`);
