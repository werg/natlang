#!/usr/bin/env node
import { readFile } from 'node:fs/promises';

const path = process.argv[2];
if (!path) throw new Error('usage: analyze-read-before-code-probe.mjs RESULTS.jsonl');
const rows = (await readFile(path, 'utf8')).split(/\r?\n/).filter(Boolean).map(line => JSON.parse(line));
const summaries = new Map();
const detail = [];
for (const row of rows) {
  const program = row.task.program_ir, family = program.family;
  const actions = row.outcome?.action_ledger ?? [];
  const firstEval = actions.findIndex(action => action.name === 'eval');
  const before = firstEval < 0 ? actions : actions.slice(0, firstEval);
  const observations = before.filter(action => action.outcome === 'ok' &&
    ['read_value', 'read_file'].includes(action.name)).map(action =>
    action.name === 'read_value' ? String(action.arguments?.expression ?? '') :
      String(action.arguments?.path ?? ''));
  const wanted = family.includes('input_numeric') ? ['brief.memo', 'brief.values'] :
    family.includes('input_text') ? ['brief.memo', 'brief.entries'] :
    family.includes('file_table') ? ['policy.md', 'accounts.json'] : ['criteria.md', 'events.json'];
  const readBeforeCode = wanted.every(item => observations.includes(item));
  const accepted = row.outcome?.accepted === true;
  const current = summaries.get(family) ?? { cases: 0, accepted: 0, read_before_code: 0, both: 0 };
  current.cases++; if (accepted) current.accepted++;
  if (readBeforeCode) current.read_before_code++;
  if (accepted && readBeforeCode) current.both++;
  summaries.set(family, current);
  detail.push({ id: program.id, accepted, read_before_code: readBeforeCode,
    observations, evals: actions.filter(action => action.name === 'eval').length,
    status: row.outcome?.status, value: row.outcome?.value });
}
console.log(JSON.stringify({ cases: rows.length,
  families: Object.fromEntries([...summaries].sort(([a], [b]) => a.localeCompare(b))), detail }, null, 2));
