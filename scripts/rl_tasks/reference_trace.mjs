#!/usr/bin/env node
// The reference run of each case, call by call (scripts/rl_tasks/sources/invented.py: episode budgets and stage tasks).
// usage: node scripts/rl_tasks/reference_trace.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json
// Each row is replay_trace.mjs traceRow: the run's status, value, model calls, child calls and the entry that
// answered each.
import fs from 'node:fs';
import { traceRow } from './replay_trace.mjs';
const [runtime, input, out] = process.argv.slice(2);
const { replayReference } = await import(runtime + '/dist/teacher/curriculum.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const results = [];
for (const record of rows) results.push(await traceRow(record, replayReference, TOOLS_PROMPT));
fs.writeFileSync(out, JSON.stringify({ cases: rows.length, results }, null, 1) + '\n');
console.log(JSON.stringify({ cases: rows.length, calls: results.map(r => r.calls ?? null) }));
