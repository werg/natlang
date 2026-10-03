#!/usr/bin/env node
/**
 * Apply the Neuralese training-data rewrites (spec/NEURALESE_DATA.md: eager typing, explicit captures) to
 * materialized trajectory records (JSONL rows with `messages` and an optional `target`).
 *
 *   node scripts/neuralese-rewrite-trajectories.mjs --out rewritten.jsonl input.jsonl [...]
 *
 * Every rewrite is checked by recompiling (same diagnostics; same capture sets); a rewrite that fails its check is left
 * out and counted. Records keep their ID, lineage, licence and split group, and gain `data_rewrite` provenance (the
 * migration and compiler revision). Replaying the recorded execution is a separate step.
 */
import { createReadStream, createWriteStream, readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { DATA_REWRITE_MIGRATION, DATA_REWRITE_VERSION, rewriteTrajectory } from '../dist/compiler/data-rewrites.js';

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  out: { type: 'string' }, summary: { type: 'string' }, scope: { type: 'string' },
} });
if (!values.out || !positionals.length) {
  console.error('usage: neuralese-rewrite-trajectories.mjs --out <file.jsonl> [--summary <file.json>] [--scope <scope.json>] <input.jsonl>...');
  process.exit(2);
}
const pkg = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8'));
const scope = values.scope ? JSON.parse(readFileSync(values.scope, 'utf8')) : undefined;
const out = createWriteStream(values.out);
const totals = { records: 0, rewritten: 0, evals: 0, annotated: 0, sites: 0, skipped: {}, failed: 0, unreadable: 0 };
for (const input of positionals) {
  for await (const line of createInterface({ input: createReadStream(input), crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    totals.records++;
    let row;
    try { row = JSON.parse(line); } catch { totals.unreadable++; continue; }
    if (!Array.isArray(row.messages)) { out.write(line + '\n'); continue; }
    const { record, stats } = rewriteTrajectory(row, { scope });
    totals.evals += stats.evals; totals.annotated += stats.annotated; totals.sites += stats.sites; totals.failed += stats.failed.length;
    for (const [reason, count] of Object.entries(stats.skipped)) totals.skipped[reason] = (totals.skipped[reason] ?? 0) + count;
    const changed = JSON.stringify(record.messages) !== JSON.stringify(row.messages) || JSON.stringify(record.target) !== JSON.stringify(row.target);
    if (changed) totals.rewritten++;
    out.write(JSON.stringify({ ...record, data_rewrite: { migration: DATA_REWRITE_MIGRATION, version: DATA_REWRITE_VERSION,
      compiler: `${pkg.name}@${pkg.version}`, changes: ['eager-typing', 'explicit-captures'], changed, stats } }) + '\n');
  }
}
await new Promise(resolve => out.end(resolve));
const summary = JSON.stringify(totals, null, 2);
if (values.summary) createWriteStream(values.summary).end(summary + '\n');
console.log(summary);
