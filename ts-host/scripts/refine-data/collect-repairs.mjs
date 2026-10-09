#!/usr/bin/env node
/** Collect `refine-repair` records (src/teacher/refine-repair.ts) from teacher collector rows.
 *
 *   collect-repairs.mjs --in rows.jsonl --out repairs.jsonl [--trace events.jsonl]
 *
 * `rows.jsonl` holds `natlang.teacher_trajectory.native/1` rows (the collector's output, for example
 * runs/teacher.native.jsonl). Rows the collector did not accept are skipped. `--trace` is an optional JSONL of trace
 * objects (`{events:[...]}`) or bare events; when given, each repair must be confirmed by a failing then passing
 * `refinement_check`. Reads files only: no model, no network. An existing output is not overwritten.
 */
import { createReadStream, writeFileSync, existsSync, readFileSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { parseArgs } from 'node:util';
import { collectRefineRepairs } from '../../dist/teacher/refine-repair.js';

const { values } = parseArgs({ options: { in: { type: 'string' }, out: { type: 'string' }, trace: { type: 'string' } } });
if (!values.in || !values.out) throw new Error('usage: collect-repairs.mjs --in rows.jsonl --out repairs.jsonl [--trace events.jsonl]');
if (existsSync(values.out)) throw new Error(`${values.out} exists; outputs are immutable`);
const trace = values.trace ? readFileSync(values.trace, 'utf8').split('\n').filter(Boolean).flatMap(line => {
  const item = JSON.parse(line);
  return Array.isArray(item.events) ? item.events : [item];
}) : undefined;
const out = [];
let rows = 0, accepted = 0;
for await (const line of createInterface({ input: createReadStream(values.in) })) {
  if (!line.trim()) continue;
  rows++;
  const row = JSON.parse(line);
  if (row.outcome?.accepted === true) accepted++;
  out.push(...collectRefineRepairs(row, trace ? { trace } : {}).map(record => JSON.stringify(record)));
}
writeFileSync(values.out, out.length ? out.join('\n') + '\n' : '', { flag: 'wx' });
console.error(JSON.stringify({ rows, accepted, repairs: out.length }));
