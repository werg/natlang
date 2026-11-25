#!/usr/bin/env node
// Judgment ablations of invented programs (scripts/rl_tasks/sources/invented.py, requirement S2).
// usage: node scripts/rl_tasks/ablate.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json
// Per case and per ablation in `curriculum.external.checks.ablations` ({judgment, children: [{match, value}]}): the
// ablation's children are placed before the reference's (the replay answers a child call with the first entry whose
// match fragments all appear in the call's opening), so that judgment's calls get the trap reading while every other
// call keeps its reference answer. The whole reference then replays through the program's own code. The ablation
// passes when the replay runs to an answer and the oracle rejects it: the task's answer depends on that judgment.
// A replay that does not run to an answer, or whose answer is accepted, fails.
import fs from 'node:fs';
const [runtime, input, out] = process.argv.slice(2);
const { replayReference } = await import(runtime + '/dist/teacher/curriculum.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);
const results = [];
for (const record of rows) {
  for (const ablation of record.curriculum?.external?.checks?.ablations ?? []) {
    const variant = structuredClone(record);
    variant.curriculum.reference.children = [...ablation.children, ...(variant.curriculum.reference.children ?? [])];
    const row = { id: record.id, judgment: ablation.judgment };
    try {
      const { run } = await replayReference(variant, TOOLS_PROMPT);
      row.status = run.outcome.status;
      row.accepted = run.outcome.accepted === true;
      row.value = run.outcome.value;
      row.rejected = run.outcome.status === 'done' && run.outcome.accepted === false;
      if (run.outcome.status !== 'done') row.error = JSON.stringify(run.outcome.detail ?? run.outcome.status).slice(0, 700);
    } catch (e) { row.rejected = false; row.error = String(e).slice(0, 300); }
    results.push(row);
  }
}
fs.writeFileSync(out, JSON.stringify({ cases: rows.length, ablations: results.length,
  rejected: results.filter(r => r.rejected).length, results }, null, 1) + '\n');
console.log(JSON.stringify({ cases: rows.length, ablations: results.length, rejected: results.filter(r => r.rejected).length }));
