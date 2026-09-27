#!/usr/bin/env node
// Score a collected held-out shard, including missing jobs and source-record leakage.
// node score.mjs --ir HELDOUT.ir.jsonl --results RESULTS.jsonl --train TRAIN.ir.jsonl --out SCORE.json
import { readFile, writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { recordDigest } from '../../dist/teacher/collector.js';

const { values } = parseArgs({ options: {
  ir: { type: 'string' }, results: { type: 'string' }, train: { type: 'string', multiple: true },
  out: { type: 'string' },
} });
if (!values.ir || !values.results || !values.out)
  throw new Error('usage: score.mjs --ir HELDOUT.ir.jsonl --results RESULTS.jsonl [--train TRAIN.ir.jsonl ...] --out SCORE.json');

const readRows = async path => (await readFile(path, 'utf8')).split('\n').filter(Boolean).map((line, index) => {
  try { return JSON.parse(line); }
  catch (error) { throw new Error(`${path}:${index + 1}: ${error.message}`); }
});
const sourceGroups = row => row.source_groups ?? row.task?.program_ir?.source_groups ?? [];
const family = row => row.curriculum?.family ?? row.task?.program_ir?.curriculum?.family ?? 'unknown';
const oracle = row => {
  const spec = row.semantics?.oracle ?? row.task?.program_ir?.semantics?.oracle ?? 'exact';
  return typeof spec === 'string' ? spec : spec.level;
};
const bump = (groups, key, accepted, completed) => {
  const counter = groups[key] ??= { total: 0, completed: 0, accepted: 0 };
  counter.total++;
  if (completed) counter.completed++;
  if (accepted) counter.accepted++;
};

const cases = await readRows(values.ir);
const results = await readRows(values.results);
const train = (await Promise.all((values.train ?? []).map(readRows))).flat();
const trainSources = new Set(train.flatMap(sourceGroups));
const leakage = [...new Set(cases.flatMap(sourceGroups).filter(id => trainSources.has(id)))].sort();
const planned = new Map();
for (const record of cases) {
  if (!record.id || planned.has(record.id)) throw new Error(`duplicate or missing held-out case id: ${record.id}`);
  planned.set(record.id, record);
}
const observed = new Map();
for (const row of results) {
  const id = row.task?.program_ir?.id;
  if (!planned.has(id)) throw new Error(`result for unplanned case: ${id}`);
  if (observed.has(id)) throw new Error(`duplicate result for case: ${id}`);
  if (recordDigest(row.task.program_ir) !== recordDigest(planned.get(id)))
    throw new Error(`result does not match the held-out program IR: ${id}`);
  if (row.provenance?.program_ir_sha256 && row.provenance.program_ir_sha256 !== recordDigest(planned.get(id)))
    throw new Error(`result provenance does not match the held-out program IR: ${id}`);
  observed.set(id, row);
}
const byFamily = {}, byOracle = {}, byStatus = {};
for (const [id, record] of planned) {
  const result = observed.get(id);
  const completed = !!result && result.outcome?.status === 'done';
  const accepted = completed && result.outcome?.accepted === true;
  bump(byFamily, family(record), accepted, completed);
  bump(byOracle, oracle(record), accepted, completed);
  const status = result?.outcome?.status ?? 'missing';
  byStatus[status] = (byStatus[status] ?? 0) + 1;
}
const summary = { version: 'natlang.heldout_score/1', ir: values.ir, results: values.results,
  train: values.train ?? [], total: cases.length, collected: observed.size,
  accepted: Object.values(byFamily).reduce((sum, item) => sum + item.accepted, 0),
  missing_ids: [...planned.keys()].filter(id => !observed.has(id)),
  source_leakage: leakage, by_family: byFamily, by_oracle: byOracle, by_status: byStatus };
await writeFile(values.out, JSON.stringify(summary, null, 2) + '\n');
console.log(`${summary.accepted}/${summary.total} accepted (${summary.collected} collected); ${leakage.length} overlapping source groups -> ${values.out}`);
if (leakage.length) process.exitCode = 2;
