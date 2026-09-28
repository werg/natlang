#!/usr/bin/env node
/** Build task IR and native static results without calling a model. */
import { mkdir, readFile, writeFile } from 'node:fs/promises';
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
import { sourceConversionDigest } from '../../dist/teacher/source-conversion.js';

export async function buildSourceBundle({ cache, out, limit = 12, trajectoryLimit = 8, licenseResolver = pinnedRepositoryLicense }) {
  const { manifest: acquisition, sources } = await loadSourceCache(cache);
  const tasks = buildTaskSources(sources, limit);
  const trajectories = await buildTrajectorySources(sources, trajectoryLimit, licenseResolver);
  const candidates = [...tasks.records, ...trajectories.records], rows = [], records = [];
  const rejected = [...tasks.rejected, ...trajectories.rejected];
  const seen = new Set();
  const options = { modelId: 'source-static-reference', rootSeed: 928, systemPrompt: TOOLS_PROMPT,
    contextTokens: 16384, maxTurns: 60, toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference' };
  for (const candidate of candidates) {
    // Bind hashes to the durable JSON representation, including omission of optional undefined fields.
    const record = JSON.parse(JSON.stringify(candidate));
    try {
      validateCurriculum(record);
      const opening = await renderOpening(record, TOOLS_PROMPT);
      if (record.curriculum.decisive.some(item => opening.includes(item.marker))) throw new Error('source_evidence_in_opening');
      if (record.split !== 'train' || record.external_source.original_split === 'test' || record.external_source.original_split === 'validation')
        throw new Error('held_out_source');
      const identity = digest([record.source, record.source_ids, record.semantics.folder_files]);
      if (seen.has(identity)) throw new Error('duplicate_source_case');
      seen.add(identity);
      const row = JSON.parse(JSON.stringify(await referenceRow(record, rows.length, options)));
      const admission = admitRow(row);
      if (!admission.admitted) throw new Error(`native_admission:${admission.reasons.join(',')}`);
      const failures = row.outcome.action_ledger?.filter(event => ['error', 'refused', 'rejected'].includes(event.outcome)) ?? [];
      if (failures.length) throw new Error('native_action_failed');
      row.provenance.source_conversion = {
        version: 'natlang.source_static_conversion/1', adapter: 'natlang.directory_source_adapter/1',
        source: record.source, source_ids: record.source_ids, source_revisions: record.source_revisions,
        license: record.license, source_snapshot_sha256: record.external_source.snapshot_sha256,
        program_ir_sha256: recordDigest(record), native_replay_accepted: true,
        native_outcome_sha256: sourceConversionDigest(row.outcome), conversion_scope: record.external_source.trajectory?.conversion_scope ?? 'source_task_reference',
        native_trajectory_sha256: sourceConversionDigest(row.trajectory),
        source_success: record.external_source.trajectory?.source_success ?? null,
        whole_issue_replayed: false, original_trajectory_id: record.external_source.trajectory?.id ?? null,
        original_row_sha256: record.external_source.trajectory?.original_row_sha256 ?? null,
      };
      if (record.external_source.trajectory) row.provenance.collection_role = 'external_replay';
      const sourceAdmission = admitRow(row);
      if (!sourceAdmission.admitted) throw new Error(`source_admission:${sourceAdmission.reasons.join(',')}`);
      const materialized = materializeNativeRows([row]);
      if (materialized.unlinked.length || !materialized.turns.some(turn => turn.training_admission.approved))
        throw new Error('no_linked_training_decisions');
      rows.push(row); records.push(record);
    } catch (error) { rejected.push({ source: record.source, id: record.id, reason: error.message }); }
  }
  await mkdir(out, { recursive: true });
  const writeJsonl = async (name, values) => {
    const raw = values.map(v => JSON.stringify(v)).join('\n') + (values.length ? '\n' : '');
    await writeFile(join(out, name), raw);
    return { path: name, sha256: digest(raw), rows: values.length };
  };
  const ir = await writeJsonl('train.ir.jsonl', records);
  const results = await writeJsonl('static.results.jsonl', rows);
  const turns = materializeNativeRows(rows);
  await writeJsonl('static.turns.jsonl', turns.turns);
  await writeJsonl('rejections.jsonl', rejected);
  await writeJsonl('trajectory-audit.jsonl', trajectories.audits);
  const bySource = {};
  for (const record of records) bySource[record.source] = (bySource[record.source] ?? 0) + 1;
  const rejectionCounts = {};
  for (const row of rejected) rejectionCounts[row.reason] = (rejectionCounts[row.reason] ?? 0) + 1;
  const report = { version: 'natlang.source_static_bundle/1', model_calls: 0, acquisition_sha256: digest(acquisition),
    cases: records.length, by_source: bySource, results, ir,
    training_decisions: turns.turns.filter(t => t.training_admission.approved).length,
    held_decisions: turns.turns.filter(t => !t.training_admission.approved).length,
    trajectory_rows_acquired: ['swesmith', 'nebius', 'nvidia'].reduce((n, key) => n + (sources[key]?.rows.length ?? 0), 0),
    converted_trajectory_cases: records.filter(r => r.external_source.trajectory).length,
    full_swe_trajectories_converted: 0,
    trajectory_rows_audited: trajectories.audits.length, rejection_counts: rejectionCounts,
    acquisition_failures: acquisition.failures ?? {} };
  await writeFile(join(out, 'static.manifest.json'), JSON.stringify(report, null, 2) + '\n');
  return report;
}

async function main() {
  const { values } = parseArgs({ options: { cache: { type: 'string', default: '../vendor/directory-sources' },
    out: { type: 'string', default: '../data/teacher/source-backed' }, limit: { type: 'string', default: '12' },
    'trajectory-limit': { type: 'string', default: '8' } } });
  const limit = Number(values.limit), trajectoryLimit = Number(values['trajectory-limit']);
  if (!Number.isInteger(limit) || limit < 1 || !Number.isInteger(trajectoryLimit) || trajectoryLimit < 0) throw new Error('invalid_limits');
  const report = await buildSourceBundle({ cache: resolve(values.cache), out: resolve(values.out), limit, trajectoryLimit });
  console.log(JSON.stringify(report, null, 2));
  if (!report.cases) throw new Error('no_source_cases_admitted');
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
