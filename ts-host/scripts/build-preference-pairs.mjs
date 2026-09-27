#!/usr/bin/env node
/**
 * Preference pairs (teacher/handoff.ts preferencePair): a decision preferred to a failed one made in its place.
 *
 *   node scripts/build-preference-pairs.mjs OUT.jsonl [--handoffs RESULTS.jsonl,...]
 *     [--variants CORRECTED.jsonl --parents ADMITTED.jsonl] [--workers 6]
 *
 * --handoffs: runs of handoff tasks (build-handoffs.mjs); an accepted, admitted run gives its teacher's decision at the
 * handoff over the failed one. --variants: corrected variants (corrections.mjs) with the rows they came from; the fix
 * made first is preferred to the parent's first failed attempt, with the same reasoning.
 */
import { writeFile } from 'node:fs/promises';
import { parseArgs } from 'node:util';
import { admitRow } from '../dist/teacher/curriculum.js';
import { preferencePair } from '../dist/teacher/handoff.js';
import { recorded } from '../dist/teacher/replay.js';
import { pool, replayOptions, rowsOf } from './build-handoffs.mjs';

const { values, positionals } = parseArgs({ allowPositionals: true, options: { handoffs: { type: 'string' },
  variants: { type: 'string' }, parents: { type: 'string' }, workers: { type: 'string', default: '6' } } });
const [output] = positionals;
if (!output || (!values.handoffs && !values.variants) || (!values.variants !== !values.parents))
  throw new Error('usage: build-preference-pairs.mjs OUT.jsonl [--handoffs RESULTS,...] [--variants CORRECTED --parents ADMITTED]');

const jobs = [];
for await (const row of rowsOf((values.handoffs ?? '').split(',').filter(Boolean))) {
  const handoff = row.task.program_ir.handoff;
  if (!handoff || !row.outcome?.accepted || row.handoff?.at == null) continue;
  jobs.push({ row, index: row.handoff.at, rejected: handoff.rejected, kind: handoff.kind, runId: row.handoff.run_id,
    evidence: { kind: 'handoff', site: handoff.kind, source: handoff.source, teacher: row.provenance.model } });
}
if (values.variants) {
  const parents = new Map();
  for await (const row of rowsOf([values.parents])) parents.set(row.id, row);
  for await (const row of rowsOf([values.variants])) {
    const variant = row.provenance.variant, parent = parents.get(variant.parent);
    if (!parent) continue;
    jobs.push({ row, index: variant.decision, rejected: recorded(parent.trajectory[variant.left_out[0]]), kind: 'failed_action',
      runId: variant.run_id, evidence: { kind: 'corrected', parent: parent.id, model: row.provenance.model } });
  }
}
const pairs = [], skipped = {};
const skip = reason => { skipped[reason] = (skipped[reason] ?? 0) + 1; };
await pool(jobs, Number(values.workers), async ({ row, index, rejected, kind, runId, evidence }) => {
  if (row.task.program_ir.curriculum) {
    const admission = admitRow(row);
    if (!admission.admitted || admission.notes.includes('judged_directly')) return skip('the chosen run is not admitted');
  }
  try {
    const pair = await preferencePair(row, index, rejected, kind, evidence, replayOptions(row.provenance), runId);
    if ('rejected' in pair && typeof pair.rejected === 'string') return skip(pair.rejected);
    pairs.push(pair);
  } catch (error) { skip(`replay failed: ${String(error?.message ?? error).split('\n')[0].slice(0, 80)}`); }
});
pairs.sort((a, b) => a.id.localeCompare(b.id));
await writeFile(output, pairs.map(pair => JSON.stringify(pair)).join('\n') + (pairs.length ? '\n' : ''));
const kinds = {};
for (const pair of pairs) kinds[`${pair.evidence.kind} ${pair.kind}`] = (kinds[`${pair.evidence.kind} ${pair.kind}`] ?? 0) + 1;
console.log(JSON.stringify({ output, pairs: pairs.length, candidates: jobs.length, kinds, skipped }, null, 1));
