#!/usr/bin/env node
// Which scripted answers the reference's result depends on (scripts/rl_tasks/sources/invented.py, C2).
// usage: node scripts/rl_tasks/entry_dependence.mjs RUNTIME_TS_HOST PROBES.json OUT.json
// PROBES.json: [{case: <a natlang.program/2 case>, probes: [{entry, value}]}]. Each probe replays the case's reference
// with children entry `entry` answering `value` (another answer of the same judgment) instead of its own; the entry
// matters when the replay then returns another value or does not run to an answer.
import fs from 'node:fs';
import { canonical } from './replay_trace.mjs';
const [runtime, input, out] = process.argv.slice(2);
const { replayReference } = await import(runtime + '/dist/teacher/curriculum.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const results = [];
for (const { case: record, probes } of JSON.parse(fs.readFileSync(input, 'utf8'))) {
  const base = JSON.stringify(canonical((await replayReference(record, TOOLS_PROMPT)).run.outcome.value ?? null));
  const row = { id: record.id, entries: {} };
  for (const { entry, value } of probes) {
    const variant = structuredClone(record);
    variant.curriculum.reference.children[entry] = { ...variant.curriculum.reference.children[entry], value };
    try {
      const { run } = await replayReference(variant, TOOLS_PROMPT);
      row.entries[entry] = run.outcome.status !== 'done' || JSON.stringify(canonical(run.outcome.value ?? null)) !== base;
    } catch { row.entries[entry] = true; }
  }
  results.push(row);
}
fs.writeFileSync(out, JSON.stringify({ results }, null, 1) + '\n');
