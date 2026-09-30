#!/usr/bin/env node
/**
 * Handoff tasks from runs that failed (teacher/handoff.ts): for each site of a run that was not accepted, a program
 * record that replays the run up to the site and hands the rest to whichever teacher collects it. A site is used only
 * if the failed turn, replayed under the current runtime, sees what it saw and fails again.
 *
 *   node scripts/build-handoffs.mjs OUT.ir.jsonl RESULTS.jsonl [MORE.jsonl ...] [--workers 6]
 *
 * Run the output with the collector like any IR file; build-preference-pairs.mjs turns the accepted runs into pairs.
 */
import { writeFile } from 'node:fs/promises';
import { jsonlRows } from './jsonl-stream.mjs';
import { parseArgs } from 'node:util';
import { retiredFamily } from '../dist/teacher/curriculum-policy.js';
import { quarantineReason, runtimeFailureReason, trainingQualityReason } from '../dist/teacher/curriculum-policy.js';
import { sourceConversionProblems } from '../dist/teacher/source-conversion.js';
import { programRunId } from '../dist/teacher/collector.js';
import { failsInPlace, handoffAt, handoffRecord, handoffSites } from '../dist/teacher/handoff.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { stripHint } from './inline-curriculum/admit.mjs';
import { dpoHoldReasons } from './admission-dispositions.mjs';

export const replayOptions = provenance => ({ systemPrompt: TOOLS_PROMPT, contextTokens: provenance.context_tokens ?? 16384,
  maxTurns: provenance.max_turns, rootSeed: provenance.seed_policy?.root ?? 909 });

export async function* rowsOf(paths) {
  const seen = new Set();
  for (const path of paths) for await (const row of jsonlRows(path)) {
    if (row.version !== 'natlang.teacher_trajectory.native/1' || seen.has(row.id)) continue;
    seen.add(row.id);
    // A hinted case is handed over and trained without its hint, as admission trains hinted runs.
    yield row.task?.program_ir?.curriculum ? stripHint(row) : row;
  }
}

/** Run `work` over items with `workers` at a time. */
export async function pool(items, workers, work) {
  let next = 0;
  await Promise.all(Array.from({ length: workers }, async () => {
    for (let item = next++; item < items.length; item = next++) await work(items[item]);
  }));
}

async function main() {
  const { values, positionals } = parseArgs({ allowPositionals: true, options: { workers: { type: 'string', default: '6' } } });
  const [output, ...inputs] = positionals;
  if (!inputs.length) throw new Error('usage: build-handoffs.mjs OUT.ir.jsonl RESULTS.jsonl [MORE.jsonl ...] [--workers N]');
  const jobs = [], skipped = {};
  for await (const row of rowsOf(inputs)) {
    if (row.outcome?.accepted !== false) continue;
    const record = row.task?.program_ir;
    if (!record) { skipped.missing_program_ir = (skipped.missing_program_ir ?? 0) + 1; continue; }
    const reasons = [...sourceConversionProblems(row), quarantineReason(record), trainingQualityReason(row),
      runtimeFailureReason(row), retiredFamily(record)].filter(Boolean);
    const holds = dpoHoldReasons(reasons);
    if (holds.length) {
      for (const hold of holds) skipped[`hold:${hold.reason}`] = (skipped[`hold:${hold.reason}`] ?? 0) + 1;
      continue;
    }
    try { for (const site of handoffSites(row)) jobs.push({ row, site }); }
    catch { jobs.push({ row, site: null }); }
  }
  const records = [], byModel = {};
  const skip = reason => { skipped[reason] = (skipped[reason] ?? 0) + 1; };
  await pool(jobs, Number(values.workers), async ({ row, site }) => {
    if (!site) return skip('the run cannot be materialized');
    const handoff = handoffAt(row, site);
    try {
      const failure = await failsInPlace(row, site.index, handoff.rejected, site.kind, replayOptions(row.provenance),
        programRunId(0, row.provenance));
      if (failure) return skip(failure);
    } catch (error) { return skip(`replay failed: ${String(error?.message ?? error).split('\n')[0].slice(0, 80)}`); }
    records.push(handoffRecord(row, handoff));
    const key = `${handoff.source.model} ${site.kind}`;
    byModel[key] = (byModel[key] ?? 0) + 1;
  });
  records.sort((a, b) => a.id.localeCompare(b.id));
  await writeFile(output, records.map(record => JSON.stringify(record)).join('\n') + (records.length ? '\n' : ''));
  console.log(JSON.stringify({ output, handoffs: records.length, sites: jobs.length, by_model: byModel, skipped }, null, 1));
}

if (import.meta.url === `file://${process.argv[1]}`) await main();
