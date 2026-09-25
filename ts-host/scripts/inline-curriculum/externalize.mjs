#!/usr/bin/env node
/**
 * Move the external modules of a built shard's cases into services (lib.mjs externalize), and verify every changed
 * case like a built one. Cases that fail verification are left out and listed.
 *
 *   node scripts/inline-curriculum/externalize.mjs IN.ir.jsonl OUT.ir.jsonl
 */
import { readFile, writeFile } from 'node:fs/promises';
import { verifyCases } from '../../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { externalize } from './lib.mjs';

const [input, output] = process.argv.slice(2);
if (!input || !output) throw new Error('usage: externalize.mjs IN.ir.jsonl OUT.ir.jsonl');
const records = (await readFile(input, 'utf8')).split('\n').filter(Boolean).map(line => JSON.parse(line));
const changed = [];
for (const record of records) {
  const before = JSON.stringify(record.semantics);
  externalize(record.semantics, record.curriculum?.family);
  if (JSON.stringify(record.semantics) !== before) changed.push(record);
}
const failures = (await verifyCases(changed, TOOLS_PROMPT)).filter(item => !item.ok);
for (const item of failures) console.error(`FAIL ${item.id}\n  ${item.problems.join('\n  ')}`);
const failed = new Set(failures.map(item => item.id));
const written = records.filter(record => !failed.has(record.id));
await writeFile(output, written.map(record => JSON.stringify(record)).join('\n') + '\n');
console.log(`${written.length} cases (${changed.length} with external services, ${failures.length} failed verification) -> ${output}`);
