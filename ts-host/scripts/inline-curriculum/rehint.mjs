#!/usr/bin/env node
/**
 * Bring a built shard's hint twins up to date without rebuilding it: every case that requires a technique gets a
 * twin with the current hint (hintFor: the technique's requirement and the family's sketch), replacing an older
 * twin or added right after its case. Changed and new twins are verified like built cases.
 *
 *   node scripts/inline-curriculum/rehint.mjs IN.ir.jsonl OUT.ir.jsonl
 */
import { readFile, writeFile } from 'node:fs/promises';
import { verifyCases } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { hintFor, hinted } from './lib.mjs';

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('usage: rehint.mjs IN.ir.jsonl OUT.ir.jsonl');
const records = (await readFile(input, 'utf8')).split('\n').filter(Boolean).map(line => JSON.parse(line));
const twins = new Map(records.filter(record => record.curriculum?.hinted_of).map(record => [record.curriculum.hinted_of, record]));
const result = [], changed = [];
let added = 0, updated = 0, kept = 0;
for (const record of records) {
  if (record.curriculum?.hinted_of) continue;
  result.push(record);
  const hint = record.curriculum?.track === 'authoring' ? null : hintFor(record.curriculum ?? {});
  const old = twins.get(record.id);
  if (!hint) { if (old) result.push(old); continue; }
  if (old?.curriculum.hint === hint) { result.push(old); kept++; continue; }
  const twin = hinted(record, hint);
  result.push(twin); changed.push(twin);
  if (old) updated++; else added++;
}
const failures = (await verifyCases(changed, TOOLS_PROMPT)).filter(item => !item.ok);
for (const item of failures) console.error(`FAIL ${item.id}\n  ${item.problems.join('\n  ')}`);
const failed = new Set(failures.map(item => item.id));
const written = result.filter(record => !failed.has(record.id));
await writeFile(output, written.map(record => JSON.stringify(record)).join('\n') + '\n');
console.log(`${written.length} cases (${added} twins added, ${updated} updated, ${kept} unchanged, ${failures.length} failed verification) -> ${output}`);
