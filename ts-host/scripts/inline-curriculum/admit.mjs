#!/usr/bin/env node
// Admit collected curriculum rows: the collector's contract verdict plus the causal checks.
// node scripts/inline-curriculum/admit.mjs RESULTS.jsonl [MORE.jsonl ...] --ledger LEDGER.jsonl [--admitted ADMITTED.jsonl] [--show]
// The ledger holds one admission per row with its concrete rejection reasons; coverage by family,
// slice, domain, mode, inline use, and turn count is printed and written beside the ledger.
import { readFile, writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { admitRow, callName, coverage, openingLength } from '../../dist/teacher/curriculum.js';

const { values, positionals } = parseArgs({ allowPositionals: true, options: {
  ledger: { type: 'string' }, admitted: { type: 'string' }, show: { type: 'boolean', default: false } } });
if (!positionals.length || !values.ledger) throw new Error('usage: admit.mjs RESULTS.jsonl... --ledger LEDGER.jsonl [--admitted OUT.jsonl] [--show]');

const rows = [];
for (const path of positionals) for (const line of (await readFile(path, 'utf8')).split('\n')) if (line.trim()) rows.push(JSON.parse(line));
const curriculumRows = rows.filter(row => row.task?.program_ir?.curriculum);
const admissions = curriculumRows.map(admitRow);
await writeFile(values.ledger, admissions.map(item => JSON.stringify(item)).join('\n') + (admissions.length ? '\n' : ''));
if (values.admitted) await writeFile(values.admitted, curriculumRows.filter((_, i) => admissions[i].admitted)
  .map(row => JSON.stringify(row)).join('\n') + '\n');
const summary = coverage(admissions);
await writeFile(values.ledger.replace(/\.jsonl$/, '') + '.coverage.json', JSON.stringify(summary, null, 2) + '\n');

for (const [i, item] of admissions.entries()) {
  const f = item.facts;
  console.log(`${item.admitted ? 'ADMIT ' : 'REJECT'} ${item.program_id}  root turns ${f.rootTurns}, evals ${f.evals}, inline ${f.inlineCalls}, ` +
    `named ${f.namedChildCalls}, edits ${f.edits}${item.reasons.length ? `  — ${item.reasons.join('; ')}` : ''}`);
  if (values.show) {
    const trajectory = curriculumRows[i].trajectory ?? [];
    const root = callName(trajectory[0]?.context ?? []);
    for (const turn of trajectory) {
      if (callName(turn.context) !== root) continue;
      const last = turn.context.at(-1);
      if (turn.context.length > openingLength(turn.context)) console.log(`    ← ${String(last.content ?? '').slice(0, 400).replace(/\n/g, '\n      ')}`);
      for (const call of turn.assistant?.calls ?? []) console.log(`    → ${call.tool} ${JSON.stringify(call.arguments).slice(0, 600)}`);
      if (!turn.assistant?.calls?.length && turn.assistant?.content) console.log(`    → reply ${turn.assistant.content.slice(0, 300)}`);
    }
    console.log(`    = ${JSON.stringify(curriculumRows[i].outcome?.value)} (${curriculumRows[i].outcome?.status}); expected ${JSON.stringify(curriculumRows[i].task.program_ir.semantics.expected)}`);
  }
}
console.log(`\n${summary.admitted}/${summary.total} admitted; rejections ${JSON.stringify(summary.rejections)}`);
console.log(`slice shares ${JSON.stringify(summary.shares.slice)}; domain shares ${JSON.stringify(summary.shares.domain)}`);
