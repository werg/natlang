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
import { writeFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { parseArgs } from 'node:util';
import { admitRow } from '../dist/teacher/curriculum.js';
import { quarantineReason, runtimeFailureReason, trainingQualityReason } from '../dist/teacher/curriculum-policy.js';
import { sourceConversionProblems } from '../dist/teacher/source-conversion.js';
import { preferencePair } from '../dist/teacher/handoff.js';
import { handoffTurns, recorded } from '../dist/teacher/replay.js';
import { pool, replayOptions, rowsOf } from './build-handoffs.mjs';
import { classifyAdmissionReason, dpoHoldReasons } from './admission-dispositions.mjs';
import { fileDigest } from './jsonl-stream.mjs';

const { values, positionals } = parseArgs({ allowPositionals: true, options: { handoffs: { type: 'string' },
  variants: { type: 'string' }, parents: { type: 'string' }, workers: { type: 'string', default: '6' } } });
const [output] = positionals;
if (!output || (!values.handoffs && !values.variants) || (!values.variants !== !values.parents))
  throw new Error('usage: build-preference-pairs.mjs OUT.jsonl [--handoffs RESULTS,...] [--variants CORRECTED --parents ADMITTED]');
const workers=Number(values.workers);
if(!Number.isInteger(workers)||workers<1)throw new Error('workers must be a positive integer');
const runtimeIdentity=async()=>Object.fromEntries(await Promise.all(['native/runtime.js','teacher/handoff.js',
  'teacher/curriculum.js','teacher/curriculum-policy.js','teacher/source-conversion.js'].map(async name=>
    [name,createHash('sha256').update(await readFile(new URL('../dist/'+name,import.meta.url))).digest('hex')])));
const replayRuntime=await runtimeIdentity();

const jobs = [];
for await (const row of rowsOf((values.handoffs ?? '').split(',').filter(Boolean))) {
  const handoff = row.task.program_ir.handoff;
  if (!handoff || !row.outcome?.accepted || !row.handoff) continue;
  const { at } = handoffTurns(row.trajectory, handoff);
  if (at < 0) continue;
  jobs.push({ row, index: at, rejected: handoff.rejected, kind: handoff.kind, runId: row.handoff.run_id,
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
const pairs = [], skipped = {}, audit = [];
const skip = reason => { skipped[reason] = (skipped[reason] ?? 0) + 1; };
await pool(jobs, workers, async ({ row, index, rejected, kind, runId, evidence }) => {
  const candidateId = `${row.id}:preference:${index}`;
  const sourceParent = evidence.kind === 'handoff' ? evidence.source?.trajectory_id : evidence.parent;
  const addAudit = (status, reasons = []) => audit.push({ id: candidateId, trajectory_id: row.id,
    source_parent: sourceParent ?? null, decision_index: index, result: status,
    reasons: reasons.map(raw => ({ ...classifyAdmissionReason(raw) })) });
  const record = row.task?.program_ir;
  if (!record) { skip('missing_program_ir'); return addAudit('skipped', ['missing_program_ir']); }
  const holdReasons = [
    ...(record.split === 'train' ? [] : ['not_training_split']),
    ...sourceConversionProblems(row),
    quarantineReason(record),
    trainingQualityReason(row),
    runtimeFailureReason(row),
  ].filter(Boolean);
  const holds = dpoHoldReasons(holdReasons);
  if (holds.length) {
    holds.forEach(item => skip(`hold:${item.reason}`));
    return addAudit('skipped_hold', holds.map(item => item.reason));
  }
  if (record.curriculum) {
    const admission = admitRow(row);
    if (!admission.admitted) {
      admission.reasons.forEach(reason => skip(`chosen_not_admitted:${reason}`));
      return addAudit('skipped_chosen_not_admitted', admission.reasons);
    }
  }
  try {
    const pair = await preferencePair(row, index, rejected, kind, evidence, replayOptions(row.provenance), runId);
    if ('rejected' in pair && typeof pair.rejected === 'string') {
      skip(pair.rejected);
      return addAudit('skipped_causal_check', [pair.rejected]);
    }
    pairs.push(pair);
    addAudit('pair_causally_verified');
  } catch (error) {
    const reason = `replay failed: ${String(error?.message ?? error).split('\n')[0].slice(0, 80)}`;
    skip(reason);
    addAudit('skipped_replay_error', [reason]);
  }
});
pairs.sort((a, b) => a.id.localeCompare(b.id));
if(JSON.stringify(replayRuntime)!==JSON.stringify(await runtimeIdentity()))throw new Error('Runtime changed during DPO replay; use a frozen runtime.');
const pairBytes = pairs.map(pair => JSON.stringify(pair)).join('\n') + (pairs.length ? '\n' : '');
audit.sort((a, b) => a.id.localeCompare(b.id));
const auditBytes = audit.map(row => JSON.stringify(row)).join('\n') + (audit.length ? '\n' : '');
await writeFile(output, pairBytes);
await writeFile(`${output}.audit.jsonl`, auditBytes);
const kinds = {}, dispositions = {};
for (const pair of pairs) kinds[`${pair.evidence.kind} ${pair.kind}`] = (kinds[`${pair.evidence.kind} ${pair.kind}`] ?? 0) + 1;
for (const reason of Object.keys(skipped)) {
  const rawReason = reason.replace(/^(?:hold|chosen_not_admitted):/, '');
  const disposition = classifyAdmissionReason(rawReason);
  dispositions[disposition.category] = (dispositions[disposition.category] ?? 0) + skipped[reason];
}
const manifest = { version: 'natlang.preference_pair_build/1', output, audit: `${output}.audit.jsonl`,
  runtime_identity:replayRuntime,
  inputs:await Promise.all([...new Set([...(values.handoffs??'').split(',').filter(Boolean),values.variants,values.parents].filter(Boolean))]
    .map(async path=>({path,sha256:await fileDigest(path)}))),
  pair_sha256: createHash('sha256').update(pairBytes).digest('hex'), audit_sha256: createHash('sha256').update(auditBytes).digest('hex'),
  candidates: jobs.length, audited_candidates: audit.length,
  causally_verified_pairs: pairs.length, pair_labels_causally_verified:true,
  final_training_audited:false, labels_finalized: false, kinds, skipped, skipped_dispositions: dispositions,
  skipped_disposition_measure:'Reason occurrences; one candidate can have multiple reasons.' };
await writeFile(`${output}.manifest.json`, JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify(manifest, null, 1));
