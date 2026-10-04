#!/usr/bin/env node
/**
 * Convert natlang trajectory records to Neuralese form (src/compiler/neuralese-conversion.ts; DECISIONS.md 40, 41):
 * runtime prompt pieces and program guidance become soft parameters, compaction handover notes become model writes
 * read by the pinned note, and every other site is counted with the reason it stays exact.
 *
 *   node scripts/neuralese-convert-trajectories.mjs --out converted.jsonl --pieces pieces.jsonl [--summary s.json]
 *     [--convert instructions] [--limit N] input.jsonl [...]
 *
 * `pieces.jsonl` holds each soft parameter's name, kind and initial text once (records name them only).
 */
import { createReadStream, createWriteStream, writeFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { convertTrajectory, NEURALESE_CONVERSION_VERSION } from '../dist/compiler/neuralese-conversion.js';

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  out: { type: 'string' }, pieces: { type: 'string' }, summary: { type: 'string' }, convert: { type: 'string' }, limit: { type: 'string' },
} });
if (!values.out || !values.pieces || !positionals.length) {
  console.error('usage: neuralese-convert-trajectories.mjs --out FILE --pieces FILE [--summary FILE] [--convert instructions] [--limit N] input.jsonl...');
  process.exit(2);
}
const convert = values.convert ? values.convert.split(',') : [];
const limit = values.limit ? Number(values.limit) : Infinity;
const out = createWriteStream(values.out, { flags: 'wx' });
const pieces = new Map();
const totals = { version: NEURALESE_CONVERSION_VERSION, convert, records: 0, unreadable: 0, passed_through: 0, sites: {}, pieces: {} };
outer: for (const input of positionals) {
  for await (const line of createInterface({ input: createReadStream(input), crlfDelay: Infinity })) {
    if (!line.trim()) continue;
    if (totals.records >= limit) break outer;
    totals.records++;
    let row;
    try { row = JSON.parse(line); } catch { totals.unreadable++; continue; }
    if (!Array.isArray(row.messages)) { totals.passed_through++; out.write(line + '\n'); continue; }
    const { record, pieces: used } = convertTrajectory(row, { convert });
    for (const piece of used) if (!pieces.has(piece.name)) pieces.set(piece.name, piece);
    for (const [kind, site] of Object.entries(record.neuralese_conversion.sites)) {
      const total = totals.sites[kind] ??= { converted: 0, exact: {} };
      total.converted += site.converted;
      for (const [reason, n] of Object.entries(site.exact)) total.exact[reason] = (total.exact[reason] ?? 0) + n;
    }
    if (!out.write(JSON.stringify(record) + '\n')) await new Promise(resolve => out.once('drain', resolve));
  }
}
await new Promise(resolve => out.end(resolve));
writeFileSync(values.pieces, [...pieces.values()].map(piece => JSON.stringify(piece)).join('\n') + '\n', { flag: 'wx' });
for (const piece of pieces.values()) totals.pieces[piece.kind] = (totals.pieces[piece.kind] ?? 0) + 1;
const summary = JSON.stringify(totals, null, 2);
if (values.summary) writeFileSync(values.summary, summary + '\n');
console.log(summary);
