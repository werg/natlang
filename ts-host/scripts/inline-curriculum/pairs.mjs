#!/usr/bin/env node
// Preference pairs from hinted and unhinted runs of the same case.
// node scripts/inline-curriculum/pairs.mjs RESULTS.jsonl [...] --out PAIRS.jsonl
// A hinted run that was admitted (with its hint stripped) and an unhinted run of the same case that was rejected
// for not using iterateOn are replayed side by side: at the first root decision where they differ, with an
// identical model request, the hinted decision is chosen and the unhinted one rejected. Only that first
// divergent decision is a valid pair; later requests differ.
import { readFile, writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { admitRow, callName } from '../../dist/teacher/curriculum.js';
import { stripHint } from './admit.mjs';

const { values, positionals } = parseArgs({ allowPositionals: true, options: { out: { type: 'string' } } });
if (!positionals.length || !values.out) throw new Error('usage: pairs.mjs RESULTS.jsonl... --out PAIRS.jsonl');
const rows = [];
for (const path of positionals) for (const line of (await readFile(path, 'utf8')).split('\n')) if (line.trim()) rows.push(JSON.parse(line));
const byId = new Map(rows.filter(row => row.task?.program_ir?.curriculum).map(row => [row.task.program_ir.id, row]));
const pairs = [], skipped = {};
const skip = reason => { skipped[reason] = (skipped[reason] ?? 0) + 1; };
for (const row of byId.values()) {
  const c = row.task.program_ir.curriculum;
  if (!c.hinted_of) continue;
  const plain = byId.get(c.hinted_of);
  if (!plain) { skip('no_unhinted_run'); continue; }
  if (!admitRow(row).admitted) { skip('hinted_not_admitted'); continue; }
  if (!admitRow(plain).reasons.includes('iterate_missing')) { skip('unhinted_used_iterateOn_or_failed_otherwise'); continue; }
  const chosen = stripHint(row);
  const root = callName(chosen.trajectory[0].context);
  const mine = chosen.trajectory.filter(turn => callName(turn.context) === root);
  const theirs = plain.trajectory.filter(turn => callName(turn.context) === root);
  let found = false;
  for (let i = 0; i < Math.min(mine.length, theirs.length); i++) {
    if (JSON.stringify(mine[i].context) !== JSON.stringify(theirs[i].context)) break;
    if (JSON.stringify(mine[i].assistant) === JSON.stringify(theirs[i].assistant)) continue;
    pairs.push({ version: 'natlang.curriculum_preference_pair/1', id: `${c.hinted_of}:pair:${i}`, program_ir_id: c.hinted_of,
      family: c.family, reason: 'iterateOn', decision: i, context: mine[i].context, chosen: mine[i].assistant, rejected: theirs[i].assistant });
    found = true;
    break;
  }
  if (!found) skip('no_same_request_divergence');
}
await writeFile(values.out, pairs.map(pair => JSON.stringify(pair)).join('\n') + (pairs.length ? '\n' : ''));
console.log(JSON.stringify({ pairs: pairs.length, skipped }));
