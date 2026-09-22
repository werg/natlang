#!/usr/bin/env node
import { randomUUID } from 'node:crypto';
import { mkdir, readFile, readdir, rename, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { materializeStudioRow } from '../dist/teacher/studio-materializer.js';

const args = process.argv.slice(2);
const take = flag => { const index = args.indexOf(flag); return index < 0 ? null : args[index + 1]; };
const positional = args.filter((value, index) => !value.startsWith('--') && !args[index - 1]?.startsWith('--'));
if (positional.length !== 2) {
  console.error('usage: node scripts/materialize-studio-teacher.mjs JOBS OUT [--cases CASES.jsonl] [--include-eval]');
  process.exit(2);
}
const jobs = resolve(positional[0]), output = resolve(positional[1]);
const casesPath = take('--cases');
const current = casesPath ? new Set((await readFile(resolve(casesPath), 'utf8')).split(/\r?\n/)
  .filter(Boolean).map(line => { const item = JSON.parse(line); return `${item.id}\0${item.source_revision}`; })) : null;
const turns = [];
let acceptedRows = 0;
for (const name of (await readdir(jobs)).filter(name => name.endsWith('.result.json')).sort()) {
  const row = JSON.parse(await readFile(resolve(jobs, name), 'utf8'));
  if (current && !current.has(`${row.case?.id}\0${row.case?.source_revision}`)) continue;
  if (!args.includes('--include-eval') && row.case?.split !== 'train') continue;
  const projected = materializeStudioRow(row);
  if (projected.length) acceptedRows++;
  turns.push(...projected);
}
await mkdir(dirname(output), { recursive: true });
const staged = `${output}.building-${process.pid}-${randomUUID()}`;
await writeFile(staged, turns.map(turn => JSON.stringify(turn)).join('\n') + (turns.length ? '\n' : ''), { flag: 'wx' });
await rename(staged, output);
console.log(JSON.stringify({ output, accepted_rows: acceptedRows, training_decisions: turns.length }));
