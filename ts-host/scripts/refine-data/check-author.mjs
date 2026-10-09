#!/usr/bin/env node
/** Check the refined types of refine-author examples with the runtime's own type parser (no model).
 *
 *   check-author.mjs examples.jsonl      prints {checked, failed:[{id, type, error}]} and exits 1 on any failure.
 */
import { readFileSync } from 'node:fs';
import { parseType, formatType } from '../../dist/native/types.js';

const rows = readFileSync(process.argv[2], 'utf8').split('\n').filter(Boolean).map(line => JSON.parse(line));
const failed = [];
let checked = 0;
for (const row of rows) for (const type of row.refined_types ?? []) {
  checked++;
  try {
    const parsed = parseType(type);
    if (parsed.kind !== 'refined') throw new Error(`parsed as ${parsed.kind}, not refined`);
    formatType(parsed);
  } catch (error) { failed.push({ id: row.id, type, error: String(error?.message ?? error) }); }
}
console.log(JSON.stringify({ checked, failed }));
process.exit(failed.length ? 1 : 0);
