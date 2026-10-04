#!/usr/bin/env node
/** Build a support-only native static-SFT pilot from pinned NLLB-Seed references. */
import { mkdir, open, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { jsonlRows, fileDigest } from '../jsonl-stream.mjs';
import { curriculumCase, evalCall, returnCall } from './lib.mjs';
import { digest } from './directory-sources.mjs';
import { referenceRow } from './references.mjs';
import { admitRow } from '../../dist/teacher/curriculum.js';
import { recordDigest, defaultToolSurfaceHash } from '../../dist/teacher/collector.js';
import { sourceConversionDigest } from '../../dist/teacher/source-conversion.js';
import { TOOLS_PROMPT } from '../../dist/native/prompt.js';
import { materializeNativeRows } from '../../dist/teacher/native-materializer.js';
import { NLLB_CANDIDATE, NLLB_REFERENCE_POLICY, nllbProgramBinding, nllbReferenceVisible } from './nllb-reference-policy.mjs';

const sha = value => createHash('sha256').update(value).digest('hex');
const DEFAULT_CANDIDATE = resolve('runs/self-improvement-expansion-20261004/nllb-seed-static-full-v2');

async function readManifest(candidateDir) {
  const path = join(candidateDir, 'manifest.json'), raw = await readFile(path);
  if (sha(raw) !== NLLB_CANDIDATE.manifest) throw new Error('nllb_candidate_manifest_hash_mismatch');
  const manifest = JSON.parse(raw.toString('utf8'));
  for (const [name, expected] of [['translation-source-ir.jsonl', NLLB_CANDIDATE.ir],
    ['translation-references.host-only.jsonl', NLLB_CANDIDATE.references]]) {
    if (manifest.artifacts?.[name]?.sha256 !== expected || await fileDigest(join(candidateDir, name)) !== expected)
      throw new Error(`nllb_candidate_artifact_pin_mismatch:${name}`);
  }
  if (manifest.source?.archive_sha256 !== NLLB_CANDIDATE.archive || manifest.source?.license !== NLLB_CANDIDATE.license)
    throw new Error('nllb_candidate_source_pin_mismatch');
  return { path, manifest };
}

async function chooseSupportRows(candidateDir, perDirectionLimit) {
  const groupCounts = new Map();
  for await (const row of jsonlRows(join(candidateDir, 'translation-source-ir.jsonl'))) {
    if (row.split !== 'train' || row.generation?.derived_role !== 'train_support') continue;
    if (!Array.isArray(row.source_groups) || row.source_groups.length !== 1) throw new Error(`nllb_invalid_source_groups:${row.id}`);
    const group = row.source_groups[0], direction = `${row.external_source.source_language_code}->${row.external_source.target_language_code}`;
    const info = groupCounts.get(group) ?? new Map(); info.set(direction, (info.get(direction) ?? 0) + 1); groupCounts.set(group, info);
  }
  const selected = new Set(), directionCounts = new Map();
  const sorted = [...groupCounts.keys()].sort((a,b) => sha(Buffer.from(a)).localeCompare(sha(Buffer.from(b))));
  for (const group of sorted) {
    const counts = groupCounts.get(group);
    if ([...counts].some(([direction, count]) => (directionCounts.get(direction) ?? 0) + count > perDirectionLimit)) continue;
    selected.add(group);
    for (const [direction, count] of counts) directionCounts.set(direction, (directionCounts.get(direction) ?? 0) + count);
  }
  return { selected, directionCounts, groups: groupCounts.size };
}

function caseFor(source, reference, candidatePins) {
  const input = source.semantics?.inputs ?? {};
  if (source.split !== 'train' || source.generation?.derived_role !== 'train_support' ||
      reference.role !== 'train_support' || reference.split !== 'train' || reference.visibility !== 'host-only')
    throw new Error(`nllb_not_support_only:${source.id}`);
  const record = curriculumCase({ family: 'nllb_human_reference_translation', shape: sha(Buffer.from(source.id)).slice(0, 20),
    variant: 'reference-v1', splitGroup: source.source_groups[0], slice: 'observation_followup', domain: 'other',
    mode: 'single_call', inline: 'avoid', root: { name: 'translate', args: { source_language: 'string', source_text: 'string', target_language: 'string' },
      returns: 'string', instructions: source.semantics.files[source.semantics.root].split('\n---\n').at(-1).trim() },
    inputs: input, expected: reference.target_text, reference: { root: [
      evalCall('console.log("source_language="+source_language+"\\nsource_text="+source_text+"\\ntarget_language="+target_language);'),
      returnCall(reference.target_text)] }, split: 'train' });
  record.id = `nllb-reference:${source.id}`;
  record.family = 'curriculum_nllb_human_reference_translation';
  record.source = 'NLLB-Seed'; record.source_ids = [...source.source_ids]; record.source_groups = [...source.source_groups];
  record.source_revisions = [...source.source_revisions]; record.license = NLLB_CANDIDATE.license;
  record.gold_sources = ['NLLB-Seed human reference translation; exact source/ref join verified'];
  record.task_modality = 'reference-translation';
  record.generation = { ...source.generation, generator: 'natlang.nllb_seed_translation_static_adapter/1',
    bundle_policy: NLLB_REFERENCE_POLICY, derived_role: 'train_support' };
  record.semantics = { ...source.semantics, expected: reference.target_text };
  record.external_source = { ...source.external_source, original_split: 'train', license: NLLB_CANDIDATE.license,
    snapshot_sha256: candidatePins.manifest,
    quality: { version: 'natlang.source_quality/1', status: 'eligible',
      checks: ['pinned_candidate_ir_reference_rows_joined', 'source_language_and_text_visible_before_answer',
        'train_support_component_only', 'license_and_attribution_preserved'], hold_reasons: [] },
    nllb_reference: { policy: NLLB_REFERENCE_POLICY, candidate_manifest_sha256: candidatePins.manifest,
      candidate_ir_sha256: candidatePins.ir, host_references_sha256: candidatePins.references,
      archive_sha256: candidatePins.archive, task_id: reference.task_id, source_group: reference.source_group,
      source_language: reference.source_language, target_language: reference.target_language } };
  return record;
}

/** Build replay rows for complete train-support components; validation and query rows never enter the bundle. */
export async function buildNllbStaticReference({ candidateDir = DEFAULT_CANDIDATE, out, perDirectionLimit = 16 }) {
  if (!out || !Number.isInteger(perDirectionLimit) || perDirectionLimit < 1) throw new Error('invalid_output_or_direction_limit');
  candidateDir = resolve(candidateDir); out = resolve(out);
  const { manifest: candidateManifest } = await readManifest(candidateDir);
  const selection = await chooseSupportRows(candidateDir, perDirectionLimit);
  const parent = dirname(candidateDir);
  if (dirname(out) !== parent) throw new Error('output_must_be_sibling_of_pinned_candidate');
  await mkdir(out, { recursive: false });
  const names = ['train.ir.jsonl', 'static.results.jsonl', 'static.turns.jsonl', 'source-bindings.jsonl'];
  const streams = new Map();
  for (const name of names) streams.set(name, { file: await open(join(out, name), 'wx'), hash: createHash('sha256'), rows: 0 });
  const append = async (name, row) => { const raw = Buffer.from(JSON.stringify(row) + '\n');
    await streams.get(name).file.write(raw); streams.get(name).hash.update(raw); streams.get(name).rows++; };
  let caseIndex = 0, directAnswers = 0, groups = new Set(), countsByDirection = {};
  try {
    const refs = jsonlRows(join(candidateDir, 'translation-references.host-only.jsonl'));
    for await (const source of jsonlRows(join(candidateDir, 'translation-source-ir.jsonl'))) {
      const next = await refs.next(); if (next.done) throw new Error('nllb_candidate_ref_count_mismatch');
      if (!selection.selected.has(source.source_groups?.[0])) continue;
      const reference = next.value;
      if (reference.ir_id !== source.id || reference.task_id !== source.source_ids?.[0] ||
          reference.source_group !== source.source_groups?.[0] || reference.target_text == null ||
          sha(Buffer.from(source.semantics.inputs.source_text, 'utf8')) !== source.external_source.source_line_content_sha256 ||
          sha(Buffer.from(reference.target_text, 'utf8')) !== reference.target_line_content_sha256)
        throw new Error(`nllb_candidate_reference_join_mismatch:${source.id}`);
      const record = caseFor(source, reference, NLLB_CANDIDATE);
      const row = await referenceRow(record, caseIndex++, { modelId: 'nllb-human-reference-static', rootSeed: 991,
        systemPrompt: TOOLS_PROMPT, contextTokens: 32768, maxTurns: 60,
        followCutoffPages: true, followEvalCutoffPages: true,
        toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference' });
      delete row.provenance.synthetic_reasoning;
      row.provenance.synthetic_reasoning = 'human-reference/1';
      const nllbRef = { schema: NLLB_REFERENCE_POLICY, task_id: reference.task_id, source_group: reference.source_group,
        role: reference.role, split: reference.split, source_pair: source.external_source.source_pair_folder,
        source_language: reference.source_language, target_language: reference.target_language,
        target_text: reference.target_text, source_line_content_sha256: source.external_source.source_line_content_sha256,
        target_line_content_sha256: reference.target_line_content_sha256, visibility: reference.visibility,
        license: reference.license };
      if (!nllbProgramBinding(record, nllbRef) || !nllbReferenceVisible(record, row)) throw new Error(`nllb_reference_visibility_or_binding_failed:${source.id}`);
      const sourceConversion = { version: 'natlang.source_static_conversion/1', adapter: 'natlang.nllb_seed_translation_static_adapter/1',
        source: 'NLLB-Seed', source_ids: source.source_ids, source_revisions: source.source_revisions, license: source.license,
        source_snapshot_sha256: NLLB_CANDIDATE.manifest, program_ir_sha256: recordDigest(record),
        native_replay_accepted: row.outcome.accepted === true, native_outcome_sha256: sourceConversionDigest(row.outcome),
        conversion_scope: 'human_reference_translation', native_trajectory_sha256: sourceConversionDigest(row.trajectory),
        source_success: null, whole_issue_replayed: false, original_trajectory_id: null,
        nllb_reference: { policy: NLLB_REFERENCE_POLICY, task_id: reference.task_id, source_group: reference.source_group,
          candidate_manifest_sha256: NLLB_CANDIDATE.manifest, candidate_ir_sha256: NLLB_CANDIDATE.ir,
          host_references_sha256: NLLB_CANDIDATE.references, archive_sha256: NLLB_CANDIDATE.archive,
          source_line_content_sha256: source.external_source.source_line_content_sha256,
          target_line_content_sha256: reference.target_line_content_sha256 } };
      row.provenance.source_conversion = sourceConversion;
      row.provenance.collection_role = 'reference';
      row.provenance.variant = { decision: row.trajectory.length - 1, kind: 'nllb_reference_answer_only' };
      const admission = admitRow(row);
      if (!row.outcome.accepted || !admission.admitted) throw new Error(`nllb_native_reference_rejected:${source.id}:${admission.reasons.join(',')}`);
      const materialized = materializeNativeRows([row], { directAnswers: true });
      const approved = materialized.turns.filter(turn => turn.training_admission.approved);
      if (materialized.rejectedRows || materialized.unlinked.length || approved.length !== 1)
        throw new Error(`nllb_native_materialization_failed:${source.id}`);
      await append('train.ir.jsonl', record);
      await append('static.results.jsonl', row);
      for (const turn of materialized.turns) await append('static.turns.jsonl', turn);
      await append('source-bindings.jsonl', { schema: 'natlang.nllb_seed_reference_binding/1', source_ir: source, reference });
      directAnswers += approved.length; groups.add(source.source_groups[0]);
      const direction = `${source.external_source.source_language_code}->${source.external_source.target_language_code}`;
      countsByDirection[direction] = (countsByDirection[direction] ?? 0) + 1;
    }
    if (!(await refs.next()).done) throw new Error('nllb_candidate_reference_rows_unconsumed');
  } finally { for (const stream of streams.values()) await stream.file.close(); }
  const artifacts = {};
  for (const name of names) { const { stat } = await import('node:fs/promises');
    const info = await stat(join(out, name)); artifacts[name] = { path: name, rows: streams.get(name).rows, bytes: info.size, sha256: streams.get(name).hash.digest('hex') }; }
  if (!directAnswers) throw new Error('no_nllb_support_cases_selected');
  const report = { version: 'natlang.source_static_bundle/1', model_calls: 0, cases: directAnswers,
    by_source: { 'NLLB-Seed': directAnswers }, ir: artifacts['train.ir.jsonl'], results: artifacts['static.results.jsonl'],
    turns: artifacts['static.turns.jsonl'], source_proof: artifacts['source-bindings.jsonl'],
    training_decisions: directAnswers, held_decisions: 0, trajectory_rows_acquired: 0, converted_trajectory_cases: 0,
    full_swe_trajectories_converted: 0, trajectory_rows_audited: 0, rejection_counts: {},
    source_answer_policy: NLLB_REFERENCE_POLICY,
    nllb_reference_policy: { version: NLLB_REFERENCE_POLICY, candidate_dir: '../nllb-seed-static-full-v2',
      candidate_manifest_sha256: NLLB_CANDIDATE.manifest, candidate_ir_sha256: NLLB_CANDIDATE.ir,
      host_references_sha256: NLLB_CANDIDATE.references, archive_sha256: NLLB_CANDIDATE.archive,
      license: NLLB_CANDIDATE.license, support_only: true, reference_derived: true,
      groups: groups.size, selected_groups_per_direction_limit: perDirectionLimit, by_direction: countsByDirection,
      bindings: artifacts['source-bindings.jsonl'],
      self_improvement_reward: 'held_semantic_evaluator_pending' } };
  await writeFile(join(out, 'static.manifest.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
  return report;
}

async function main() {
  const { values } = parseArgs({ options: { 'candidate-dir': { type: 'string', default: DEFAULT_CANDIDATE },
    out: { type: 'string' }, 'support-groups-per-direction': { type: 'string', default: '16' } } });
  const report = await buildNllbStaticReference({ candidateDir: values['candidate-dir'], out: values.out,
    perDirectionLimit: Number(values['support-groups-per-direction']) });
  console.log(JSON.stringify(report, null, 2));
}
if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) await main();
