#!/usr/bin/env node
// A builder's dry run of an invented program (`invented.py dry-run`): the program's TypeScript runs with the scripted
// judgment answers, with no model, in seconds. Per case: the reference replay call by call (replay_trace.mjs), and with
// CONTROLS=1 each wrong answer (mutant), judgment ablation and keyword baseline replayed the way qualification replays
// them (verify.mjs, ablate.mjs): what each returns and whether the checker accepts it.
// usage: node scripts/rl_tasks/dry_run.mjs RUNTIME_TS_HOST CASES.jsonl OUT.json [CONTROLS]
import fs from 'node:fs';
import { canonical, traceRow } from './replay_trace.mjs';
const [runtime, input, out, controls] = process.argv.slice(2);
const { replayReference } = await import(runtime + '/dist/teacher/curriculum.js');
const { TOOLS_PROMPT } = await import(runtime + '/dist/native/prompt.js');
const rows = fs.readFileSync(input, 'utf8').split('\n').filter(Boolean).map(JSON.parse);

async function replayed(variant) {
  try {
    const { run } = await replayReference(variant, TOOLS_PROMPT);
    const ran = run.outcome.status === 'done';
    return { ran, accepted: ran ? run.outcome.accepted === true : null, value: run.outcome.value ?? null,
      ...(ran ? {} : { error: JSON.stringify(run.outcome.detail ?? run.outcome.status).slice(0, 700) }) };
  } catch (e) { return { ran: false, accepted: null, value: null, error: String(e).slice(0, 400) }; }
}

const results = [];
for (const record of rows) {
  const row = await traceRow(record, replayReference, TOOLS_PROMPT);
  if (controls === '1') {
    const checks = record.curriculum?.external?.checks ?? {};
    const reference = record.curriculum.reference;
    row.mutants = [];
    for (const mutant of checks.wrong_mutants ?? []) {
      const variant = structuredClone(record);
      if (mutant.children) variant.curriculum.reference.children = [...mutant.children, ...(reference.children ?? [])];
      if (mutant.code) variant.curriculum.reference.root = [['eval', { code: mutant.code, finish: true }]];
      const result = { mistake: String(mutant.mistake ?? '').slice(0, 160), ...await replayed(variant) };
      if ('value' in mutant) result.value_matches = JSON.stringify(canonical(mutant.value)) === JSON.stringify(canonical(result.value));
      row.mutants.push(result);
    }
    row.ablations = [];
    for (const ablation of checks.ablations ?? []) {
      const variant = structuredClone(record);
      variant.curriculum.reference.children = [...ablation.children, ...(reference.children ?? [])];
      row.ablations.push({ judgment: ablation.judgment, ...await replayed(variant) });
    }
    row.baselines = [];
    for (const code of checks.keyword_baseline_code ?? []) {
      const variant = structuredClone(record);
      variant.curriculum.reference.root = [['eval', { code, finish: true }]];
      row.baselines.push(await replayed(variant));
    }
  }
  results.push(row);
}
fs.writeFileSync(out, JSON.stringify({ cases: rows.length, results }, null, 1) + '\n');
