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
import { FAMILIES } from './families.mjs';

export const ITERATE_HINT = 'Hint: step this with iterateOn. Write a step function (it may be an nl function) from the current state to the next, and run it with .iterateOn(initial).until(done), rather than a loop with a guessed bound.';
function hinted(record) {
  const twin = structuredClone(record);
  twin.id = `${record.id}:hinted`;
  twin.source_ids = [twin.id];
  const root = twin.semantics.root;
  twin.semantics.files[root] = twin.semantics.files[root].replace(/\n$/, '') + `\n\n${ITERATE_HINT}\n`;
  twin.curriculum.hint = ITERATE_HINT;
  twin.curriculum.hinted_of = record.id;
  if (twin.curriculum.pair_group) twin.curriculum.pair_group = `${twin.curriculum.pair_group}:hinted`;
  return twin;
}

const { values } = parseArgs({ options: { seed: { type: 'string', default: '1' }, shapes: { type: 'string', default: '2' },
  start: { type: 'string', default: '0' },
  families: { type: 'string' }, out: { type: 'string' }, 'allow-failures': { type: 'boolean', default: false },
  // Held-out generated problems: --split test marks this build's synthetic cases as test (use a seed no training build uses).
  split: { type: 'string', default: 'train' }, hints: { type: 'boolean', default: true } } });
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
      record.split = values.split;
      record.curriculum.split_group = `s${seed}:${record.curriculum.split_group}`;
      record.source_groups = [record.curriculum.split_group];
      if (record.curriculum.pair_group) record.curriculum.pair_group = `s${seed}:${record.curriculum.pair_group}`;
    }
    // A source adapter samples a large dataset, so two indexes can land on the same problem: keep the first.
    // Generated families must never repeat an id.
    if (records.some(other => other.id === record.id)) {
      if (family.source) continue;
      throw new Error(`duplicate case id ${record.id}`);
    }
    records.push(record);
    // An iterateOn case also gets a twin whose instructions end with an explicit hint. Admission strips the
    // hint from the twin's trajectory, so it trains iterateOn without being asked; with the unhinted run it
    // makes a same-request preference pair (pairs.mjs).
    if (values.hints && record.curriculum.iterate === 'required' && record.curriculum.track !== 'authoring') records.push(hinted(record));
  }
}

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
