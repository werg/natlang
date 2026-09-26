#!/usr/bin/env node
/**
 * Extend a redo shard with the programs another teacher's runs were refused for, so a second teacher can try them.
 *
 *   node scripts/inline-curriculum/redo.mjs SHARD.ir.jsonl --ledger LEDGER.jsonl [--ledger ...] --out REDO.ir.jsonl
 *
 * A program is taken when a ledger (admit.mjs) rejected a run of it and admitted none, for a reason other than its
 * family being retired. The shard only grows: programs already in it keep their place, and new ones are appended, so
 * a collector over it keeps its job indices while the first teacher is still running.
 */
import { existsSync, readFileSync, appendFileSync } from 'node:fs';
import { parseArgs } from 'node:util';

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  ledger: { type: 'string', multiple: true }, out: { type: 'string' } } });
if (positionals.length !== 1 || !values.ledger?.length || !values.out)
  throw new Error('usage: redo.mjs SHARD.ir.jsonl --ledger LEDGER.jsonl [--ledger ...] --out REDO.ir.jsonl');

const lines = path => readFileSync(path, 'utf8').split('\n').filter(Boolean);
const admitted = new Set(), rejected = new Set();
for (const path of values.ledger) for (const line of lines(path)) {
  const entry = JSON.parse(line);
  if (entry.admitted) admitted.add(entry.program_id);
  else if (!entry.reasons.includes('retired_family')) rejected.add(entry.program_id);
}
const present = new Set(existsSync(values.out) ? lines(values.out).map(line => JSON.parse(line).id) : []);
const added = lines(positionals[0]).filter(line => {
  const id = JSON.parse(line).id;
  return rejected.has(id) && !admitted.has(id) && !present.has(id);
});
if (added.length) appendFileSync(values.out, added.join('\n') + '\n');
process.stdout.write(`${added.length} programs added (${present.size + added.length} in ${values.out})\n`);
