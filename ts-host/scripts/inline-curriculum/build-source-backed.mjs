#!/usr/bin/env node
/** Build task IR and native static results without calling a model. */
import { mkdir, readFile, writeFile, open, access } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { resolve, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { loadSourceCache, buildTaskSources, digest } from './directory-sources.mjs';
import { buildTrajectorySources, pinnedRepositoryLicense } from './trajectory-sources.mjs';
import { referenceRow } from './references.mjs';
import { admitRow, validateCurriculum, renderOpening } from '../../dist/teacher/curriculum.js';
import { defaultToolSurfaceHash, recordDigest } from '../../dist/teacher/collector.js';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { SOURCE_REVIEW_OPTIONS, configureSourceReview, writeIntakeReviews } from './source-review-intake.mjs';
import { loadWorkflowSources } from './workflow-sources.mjs';
import { sourceConversionDigest, retiredWorkflowEvaluationReleased, workflowReferenceVisibility } from '../../dist/teacher/source-conversion.js';

export async function buildSourceBundle({ cache, out, limit = 12, trajectoryLimit = 8, excludeManifests = [], licenseResolver = pinnedRepositoryLicense, workflowCache = null, progressEvery = 0 }) {
  if (workflowCache) {
    // Published source snapshots are immutable; use a fresh output for later revisions.
    const exists = await access(join(out, 'static.manifest.json')).then(() => true, error => { if (error.code === 'ENOENT') return false; throw error; });
    if (exists) throw new Error('published_workflow_bundle_exists_use_new_output');
  }
  const workflow = workflowCache ? await loadWorkflowSources(workflowCache, limit) : null;
  const { manifest: acquisition, sources } = workflow ?? await loadSourceCache(cache);
  const tasks = workflow?.tasks ?? buildTaskSources(sources, limit);
  const trajectories = workflow?.trajectories ?? await buildTrajectorySources(sources, trajectoryLimit, licenseResolver);
  const candidates = [...tasks.records, ...trajectories.records];
  await mkdir(out, { recursive: true });
  // Completed replays contain repeated contexts; retaining the whole corpus before serialization
  // grows memory with output size. Write verified cases incrementally; publish the manifest last.
  const streams = {};
  for (const name of ['train.ir.jsonl', 'static.results.jsonl', 'static.turns.jsonl']) {
    streams[name] = { handle: await open(join(out, name), 'w'), hash: createHash('sha256'), rows: 0 };
  }
  const append = async (name, value) => {
    const raw = JSON.stringify(value) + '\n';
    await streams[name].handle.writeFile(raw);
    streams[name].hash.update(raw); streams[name].rows++;
  };
  let cases = 0, trainingDecisions = 0, heldDecisions = 0, convertedCases = 0;
  const bySource = {};
  const rejected = [...tasks.rejected, ...trajectories.rejected];
  const seen = new Set();
  const excluded = new Set();
  for (const path of excludeManifests) {
    const manifest = JSON.parse(await readFile(path, 'utf8'));
    const raw = await readFile(resolve(path, '..', manifest.ir.path), 'utf8');
    if (digest(raw) !== manifest.ir.sha256) throw new Error('excluded_bundle_checksum_mismatch');
    for (const line of raw.trim().split('\n').filter(Boolean)) {
      const record = JSON.parse(line);
      excluded.add(digest([record.source, record.source_ids]));
    }
  }
  const options = { modelId: 'source-static-reference', rootSeed: 928, systemPrompt: TOOLS_PROMPT,
    // Model-free replay retains full evidence; student-specific token gates still
    // decide whether a rendered sample fits its actual training window.
    contextTokens: workflow ? 32768 : 16384, maxTurns: 60, followCutoffPages: !!workflow, followEvalCutoffPages: !!workflow,
    toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference' };
  try { for (const candidate of candidates) {
    // Bind hashes to the durable JSON representation, including omission of optional undefined fields.
    const record = JSON.parse(JSON.stringify(candidate));
    let writing = false;
    try {
      if (excluded.has(digest([record.source, record.source_ids]))) throw new Error('already_integrated_source_case');
      validateCurriculum(record);
      const opening = await renderOpening(record, TOOLS_PROMPT);
      if (record.curriculum.decisive.some(item => opening.includes(item.marker))) throw new Error('source_evidence_in_opening');
      if ((record.split !== 'train' || ['test','validation','dev'].includes(record.external_source.original_split)) && !retiredWorkflowEvaluationReleased(record))
        throw new Error('held_out_source');
      const identity = digest([record.source, record.source_ids, record.semantics.folder_files]);
      if (seen.has(identity)) throw new Error('duplicate_source_case');
      seen.add(identity);
      const row = JSON.parse(JSON.stringify(await referenceRow(record, cases, options)));
      const visibility = workflow ? workflowReferenceVisibility(row) : null;
      if (workflow && !visibility) throw new Error('reference_source_evidence_not_visible');
      const admission = admitRow(row);
      if (!admission.admitted) throw new Error(`native_admission:${admission.reasons.join(',')}`);
      const failures = row.outcome.action_ledger?.filter(event => ['error', 'refused', 'rejected'].includes(event.outcome)) ?? [];
      if (failures.length) throw new Error('native_action_failed');
      row.provenance.source_conversion = {
        version: 'natlang.source_static_conversion/1', adapter: record.generation.generator,
        source: record.source, source_ids: record.source_ids, source_revisions: record.source_revisions,
        license: record.license, source_snapshot_sha256: record.external_source.snapshot_sha256,
        program_ir_sha256: recordDigest(record), native_replay_accepted: true,
        native_outcome_sha256: sourceConversionDigest(row.outcome), conversion_scope: record.external_source.trajectory?.conversion_scope ?? 'source_task_reference',
        native_trajectory_sha256: sourceConversionDigest(row.trajectory),
        source_success: record.external_source.trajectory?.source_success ?? null,
        whole_issue_replayed: false, original_trajectory_id: record.external_source.trajectory?.id ?? null,
        original_row_sha256: record.external_source.trajectory?.original_row_sha256 ?? null,
        ...(visibility ? {visible_source_inputs:visibility} : {}),
      };
      if (record.external_source.trajectory) row.provenance.collection_role = 'external_replay';
      const sourceAdmission = admitRow(row);
      if (!sourceAdmission.admitted) throw new Error(`source_admission:${sourceAdmission.reasons.join(',')}`);
      const materialized = materializeNativeRows([row]);
      if (materialized.unlinked.length || !materialized.turns.some(turn => turn.training_admission.approved))
        throw new Error('no_linked_training_decisions');
      // I/O failures are fatal, not source rejections: never silently publish a partial case.
      writing = true;
      await append('train.ir.jsonl', record);
      await append('static.results.jsonl', row);
      for (const turn of materialized.turns) await append('static.turns.jsonl', turn);
      cases++;
      if (progressEvery && cases % progressEvery === 0)
        console.error(JSON.stringify({event:'static_replay_progress',cases,candidates:candidates.length}));
      bySource[record.source] = (bySource[record.source] ?? 0) + 1;
      if (record.external_source.trajectory) convertedCases++;
      trainingDecisions += materialized.turns.filter(t => t.training_admission.approved).length;
      heldDecisions += materialized.turns.filter(t => !t.training_admission.approved).length;
    } catch (error) {
      if (writing) throw error;
      rejected.push({ source: record.source, id: record.id, reason: error.message });
    }
  } } finally { for (const stream of Object.values(streams)) await stream.handle.close(); }
  const writeJsonl = async (name, values) => {
    const raw = values.map(v => JSON.stringify(v)).join('\n') + (values.length ? '\n' : '');
    await writeFile(join(out, name), raw);
    return { path: name, sha256: digest(raw), rows: values.length };
  };
  const fileInfo = name => ({ path: name, sha256: streams[name].hash.digest('hex'), rows: streams[name].rows });
  const ir = fileInfo('train.ir.jsonl');
  const results = fileInfo('static.results.jsonl');
  await writeJsonl('rejections.jsonl', rejected);
  await writeJsonl('trajectory-audit.jsonl', trajectories.audits);
  const rejectionCounts = {};
  for (const row of rejected) rejectionCounts[row.reason] = (rejectionCounts[row.reason] ?? 0) + 1;
  const report = { version: 'natlang.source_static_bundle/1', model_calls: 0, acquisition_sha256: digest(acquisition),
    excluded_manifests: excludeManifests, source_answer_policy: workflow ? 'verified_typed_labels_with_masked_synthetic_reasoning' : 'default',
    ...(workflow ? {source_input_visibility:'natlang.visible_source_inputs/1',adapter_revision:'visible-inputs-explicit-field-batches-v5'} : {}),
    cases, by_source: bySource, results, ir, turns: fileInfo('static.turns.jsonl'),
    training_decisions: trainingDecisions,
    held_decisions: heldDecisions,
    trajectory_rows_acquired: ['swesmith', 'nebius', 'nvidia'].reduce((n, key) => n + (sources[key]?.rows.length ?? 0), 0),
    converted_trajectory_cases: convertedCases,
    full_swe_trajectories_converted: 0,
    trajectory_rows_audited: trajectories.audits.length, rejection_counts: rejectionCounts,
    acquisition_failures: acquisition.failures ?? {} };
  await writeFile(join(out, 'static.manifest.json'), JSON.stringify(report, null, 2) + '\n');
  return report;
}

async function main() {
  const { values } = parseArgs({ options: { cache: { type: 'string', default: '../vendor/directory-sources' },
    out: { type: 'string', default: '../data/teacher/source-backed' }, limit: { type: 'string', default: '12' },
    'workflow-cache': { type: 'string' }, 'trajectory-limit': { type: 'string', default: '8' }, 'exclude-manifest': { type: 'string', multiple: true },
    'progress-every': { type: 'string', default:'0' }, ...SOURCE_REVIEW_OPTIONS } });
  configureSourceReview(values);
  const limit = Number(values.limit), trajectoryLimit = Number(values['trajectory-limit']);
  if (!Number.isInteger(limit) || limit < 1 || !Number.isInteger(trajectoryLimit) || trajectoryLimit < 0) throw new Error('invalid_limits');
  const progressEvery = Number(values['progress-every']);
  if (!Number.isInteger(progressEvery) || progressEvery < 0) throw new Error('invalid_progress_interval');
  const report = await buildSourceBundle({ cache: resolve(values.cache), out: resolve(values.out), limit, trajectoryLimit, progressEvery, workflowCache: values['workflow-cache'] ? resolve(values['workflow-cache']) : null,
    excludeManifests: (values['exclude-manifest'] ?? []).map(path => resolve(path)) });
  const reviewed = await writeIntakeReviews(resolve(values.out, 'source-review.jsonl'));
  console.log(JSON.stringify(reviewed ? { ...report, source_review: reviewed } : report, null, 2));
  if (!report.cases) throw new Error('no_source_cases_admitted');
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
